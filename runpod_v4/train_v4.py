"""v4 training on RunPod: Qwen3.5-27B, 16-bit LoRA, ONE epoch, with smoke test, checkpoints, validation and early stop.

    bash run_v4.sh                          # starts this script with nohup (see RUNPOD_V4_STEPS.txt)
    python train_v4.py --price 1.09         # directly
    python train_v4.py --resume             # continue from the last checkpoint (e.g. after a pod restart)

Run check_v4.py on a laptop first: it writes keep_ids.json (rows within the max sequence length).

What happens:
 1. Box convention check: the untouched model is asked for a few boxes; whichever convention (0-1000 relative or
    absolute pixels) matches the reference boxes better is used for training (box_mode.txt).
 2. LoRA r = 32 / alpha = 32 on vision + language layers, bf16; loss only on the assistant's turns.
 3. Smoke test after --smoke-steps steps: loss falling, no out-of-memory, answers in the right format, GPU use,
    measured speed -> time and cost of the whole run (smoke_report.txt). A failed check stops the run.
 4. One epoch, hard step cap; validation and a checkpoint every ~10 % of the epoch; the best checkpoint is kept;
    early stop after 3 validations without improvement.
 5. Adapter saved (adapter/, adapter.zip), then the test sets are scored (eval/).
 6. The benchmark (328 questions through the assistant loop and its tools, app/ in the bundle) -> benchmark/.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def log(msg, out=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if out:
        with open(Path(out) / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


class GpuSampler(threading.Thread):
    """GPU utilisation and memory every 20 s (nvidia-smi), for the smoke test and the log."""

    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.mem = [], []

    def run(self):
        while True:
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                u, used, tot = (float(x) for x in q.strip().splitlines()[0].split(","))
                self.util.append(u)
                self.mem.append(used / tot)
            except Exception:                          # noqa: BLE001
                pass
            time.sleep(20)


def box_convention(model, processor, ds, n=6):
    """Ask the untouched model for boxes on a few validation crops; pick the convention that matches better."""
    import data_v4 as DV
    import eval_v4 as E
    import qwen35_format as Q
    rows = [r for r in DV.load_rows(ds, "val") if r["task"] == "bridge_ground"][:n]
    score = {"rel1000": [], "abs": []}
    for row in rows:
        Q.BOX_MODE = "abs"
        msgs, imgs, _, _ = DV.to_chat(row, ds)
        ref_abs = E.boxes(msgs[-1]["content"])
        pred = E.boxes(E.split_think(E.generate(model, processor, msgs[:-1], imgs, False, None, 300))[1])
        if not ref_abs or not pred:
            continue
        W, H = imgs[0].size
        as_rel = [[p[0] * W / 1000, p[1] * H / 1000, p[2] * W / 1000, p[3] * H / 1000] for p in pred]
        score["abs"].append(max(E.iou(ref_abs[0], p) for p in pred))
        score["rel1000"].append(max(E.iou(ref_abs[0], p) for p in as_rel))
    mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in score.items()}
    choice = "rel1000" if mean["rel1000"] >= mean["abs"] else "abs"
    return choice, mean, len(score["abs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("V4_MODEL", "unsloth/Qwen3.5-27B"))
    ap.add_argument("--out", default="/workspace/v4_run")
    ap.add_argument("--max-len", type=int, default=0, help="default: from keep_ids.json (check_v4.py)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--eval-every", type=float, default=0.1, help="share of the epoch between validations")
    ap.add_argument("--val-per-task", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--smoke-steps", type=int, default=60)
    ap.add_argument("--price", type=float, default=1.09, help="pod price per hour (for the cost estimate)")
    ap.add_argument("--budget", type=float, default=0, help="stop after the smoke test if the estimate is above this ($)")
    ap.add_argument("--test-per-task", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-box-check", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true", help="skip the assistant benchmark after the test sets")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import data_v4 as DV
    import eval_v4 as E
    import qwen35_format as Q
    ds, tools = DV.find_dataset(), DV.find_tools()
    keep = json.loads((HERE / "keep_ids.json").read_text()) if (HERE / "keep_ids.json").exists() else None
    max_len = args.max_len or (keep["max_len"] if keep else 4096)
    train_rows = DV.load_rows(ds, "train")
    if keep:
        ids = set(keep["ids"])
        train_rows = [r for r in train_rows if r["id"] in ids]
    val_rows = DV.load_rows(ds, "val", limit_per_task=args.val_per_task, seed=3)
    if keep:
        val_rows = [r for r in val_rows if r["id"] in ids]
    eff = args.batch * args.grad_accum
    steps = max(1, math.ceil(len(train_rows) * args.epochs / eff))
    eval_steps = max(10, round(steps * args.eval_every))
    log(f"model {args.model}; max_len {max_len}; train rows {len(train_rows)}, val rows {len(val_rows)}; "
        f"effective batch {eff}; {steps} steps (cap), validation every {eval_steps}", out)

    from unsloth import FastVisionModel, is_bf16_supported
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback, TrainerCallback

    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=False, use_gradient_checkpointing="unsloth")
    log("model loaded", out)

    if (out / "adapter" / "box_mode.txt").exists():
        Q.BOX_MODE = (out / "adapter" / "box_mode.txt").read_text().strip()
    elif (out / "box_mode.txt").exists():
        Q.BOX_MODE = (out / "box_mode.txt").read_text().strip()
    elif not args.no_box_check:
        FastVisionModel.for_inference(model)
        choice, mean, n = box_convention(model, processor, ds)
        Q.BOX_MODE = choice
        log(f"box convention check on {n} crops: mean IoU rel1000 {mean['rel1000']:.2f}, abs {mean['abs']:.2f} -> using {choice}"
            + ("  (WARNING: both low; check smoke_report)" if max(mean.values()) < 0.3 else ""), out)
        (out / "box_mode.txt").write_text(choice)

    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=3407)
    FastVisionModel.for_training(model)

    sampler = GpuSampler()
    sampler.start()

    class Smoke(TrainerCallback):
        """After --smoke-steps: loss, memory, GPU use, speed -> cost, answer format. Stops the run if a check fails."""

        def __init__(self):
            self.t0 = None
            self.done = False

        def on_train_begin(self, a, state, control, **kw):
            self.t0, self.s0 = time.time(), state.global_step

        def on_step_end(self, a, state, control, **kw):
            if self.done or state.global_step - self.s0 < args.smoke_steps:
                return
            self.done = True
            import torch
            secs = (time.time() - self.t0) / max(1, state.global_step - self.s0)
            left_h = secs * (steps - state.global_step) / 3600
            evals_h = (steps / eval_steps) * len(val_rows) * 0.6 / 3600
            test_h = 1.0
            total_h = left_h + evals_h + test_h
            losses = [h["loss"] for h in state.log_history if "loss" in h]
            first, last = (sum(losses[:3]) / max(1, len(losses[:3])), sum(losses[-3:]) / max(1, len(losses[-3:]))) if losses else (0, 0)
            util = sum(sampler.util[-15:]) / max(1, len(sampler.util[-15:]))
            peak = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            FastVisionModel.for_inference(model)
            fmt = []
            for row in [r for r in val_rows if r["task"] in ("bridge_json", "band_bridge", "reason_level_check", "agent_min_fl", "agent_lookup")][:6]:
                try:
                    res, pred = E.score_row(model, processor, row, ds, tools)
                    fmt.append((row["task"], res))
                except Exception as e:                    # noqa: BLE001
                    fmt.append((row["task"], {"error": str(e)}))
            FastVisionModel.for_training(model)
            fails = []
            if losses and not last < first:
                fails.append(f"loss not falling ({first:.3f} -> {last:.3f})")
            if any("error" in r for _, r in fmt):
                fails.append("generation errors")
            cost = total_h * args.price
            if args.budget and cost > args.budget:
                fails.append(f"estimated cost ${cost:.0f} is above the budget ${args.budget:.0f}")
            lines = [f"SMOKE TEST after {state.global_step} steps",
                     f"  loss {first:.3f} -> {last:.3f}",
                     f"  speed {secs:.1f} s/step; {steps} steps in the epoch",
                     f"  GPU utilisation (last 5 min) {util:.0f} %" + ("  (below 80 %: data loading may be the bottleneck)" if util < 80 else ""),
                     f"  peak GPU memory {peak * 100:.0f} %",
                     f"  remaining training {left_h:.1f} h + validations {evals_h:.1f} h + final test {test_h:.1f} h = {total_h:.1f} h",
                     f"  estimated cost of the rest of the run at ${args.price}/h: ${cost:.0f}",
                     "  format check on validation rows (early in training, values may still be wrong):"]
            lines += [f"    {t}: {json.dumps(r)}" for t, r in fmt]
            lines.append("RESULT: " + ("PASS - training continues" if not fails else "FAIL - " + "; ".join(fails) + " - training stopped"))
            (out / "smoke_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            for l in lines:
                log(l, out)
            if fails:
                control.should_training_stop = True
                (out / "SMOKE_FAILED").write_text("; ".join(fails))

    class Progress(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and ("loss" in logs or "eval_loss" in logs):
                util = sum(sampler.util[-3:]) / max(1, len(sampler.util[-3:]))
                log(f"step {state.global_step}/{steps} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items() if isinstance(v, float))
                    + f" gpu {util:.0f}%", out)

    trainer = SFTTrainer(
        model=model, tokenizer=processor,
        data_collator=DV.Collator(processor, ds, tools, max_len),
        train_dataset=DV.LazyRows(train_rows), eval_dataset=DV.LazyRows(val_rows),
        callbacks=[Smoke(), Progress(), EarlyStoppingCallback(early_stopping_patience=args.patience)],
        args=SFTConfig(
            output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=steps,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01, optim="adamw_8bit",
            bf16=is_bf16_supported(), fp16=not is_bf16_supported(), logging_steps=10,
            eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps, save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            dataloader_num_workers=6, dataloader_pin_memory=True, seed=3407, report_to="none",
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_len),
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    if (out / "SMOKE_FAILED").exists():
        log("stopped by the smoke test - see smoke_report.txt; nothing else to do", out)
        return

    adapter = out / "adapter"
    model.save_pretrained(str(adapter))
    processor.save_pretrained(str(adapter))
    (adapter / "box_mode.txt").write_text(Q.BOX_MODE)
    json.dump(trainer.state.log_history, open(out / "log_history.json", "w"), indent=1)
    shutil.make_archive(str(out / "adapter"), "zip", str(adapter))
    log(f"adapter saved: {adapter} and {out / 'adapter.zip'} (best eval_loss {trainer.state.best_metric})", out)

    FastVisionModel.for_inference(model)
    E.evaluate(model, processor, ds, tools, out / "eval", per_task=args.test_per_task, log=lambda m: log(m, out))
    if not args.no_benchmark:
        benchmark(model, processor, out)
    log("ALL DONE - download adapter.zip, eval/, benchmark/ and progress.log, then stop and delete the pod", out)


def benchmark(model, processor, out):
    """The 328-question benchmark through the assistant loop and tools (app/ in the bundle), with the model in memory.
    A failure here is logged and does not lose anything: the adapter and eval/ are already saved."""
    app = HERE / "app" / "assistant_v4"
    if not app.exists():
        log("benchmark skipped: app/ not in the bundle", out)
        return
    try:
        sys.path.insert(0, str(app))
        import agent as AG
        import run_benchmark as RB
        log("benchmark: 328 questions through the assistant and its tools ...", out)
        RB.run(AG.TransformersModel(model=model, processor=processor), "v4", log=lambda m: log(m, out), out=out / "benchmark")
    except Exception as e:                       # noqa: BLE001
        log(f"benchmark failed ({type(e).__name__}: {e}); run it again with: python app/assistant_v4/run_benchmark.py "
            f"--adapter {out / 'adapter'} --out {out / 'benchmark'}", out)


if __name__ == "__main__":
    main()
