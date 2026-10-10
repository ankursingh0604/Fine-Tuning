"""L-section v5 fine-tuning: Qwen3.5-4B, 16-bit LoRA, with smoke test, validation, early stop and a time limit.
On RunPod (A100 / H100 80 GB): bash run_runpod.sh (batch 4, stops by --max-hours). (No RTX 3060 run script: 157 sheets is too much for 12 GB in reasonable time.)

    bash run_runpod.sh                   # RunPod: starts this with nohup (see LSEC_STEPS.txt)
    python train_lsec.py                  # directly
    python train_lsec.py --resume         # continue from the last checkpoint (after a stop or restart)

Run check_lsec.py first (it writes keep_ids.json: rows within the max sequence length).
Why these settings: Qwen3.5-4B in 16-bit with LoRA needs about 10 GB (Unsloth's figure), so it fits a 12 GB card;
the 9B needs about 22 GB, and 4-bit QLoRA is not recommended for Qwen3.5. Batch 1 x gradient accumulation 8.
What happens:
 1. LoRA r = 16 / alpha = 16 on vision + language layers, bf16; loss only on the answers.
 2. Smoke test after --smoke-steps steps: loss falling, no out-of-memory, peak memory, speed -> hours left, answer
    format on a few validation rows (smoke_report.txt). A failed check stops the run.
 3. --epochs (default 2); validation and a checkpoint every ~10 %; the best checkpoint is kept; early stop after
    --patience validations without improvement.
 4. Adapter saved (adapter/, adapter.zip), then the validation rows's questions are scored (eval/).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LSEC_MODEL", "unsloth/Qwen3.5-4B"))
    ap.add_argument("--out", default=str(HERE / "lsec_run"))
    ap.add_argument("--max-len", type=int, default=0, help="default: from keep_ids.json (check_lsec.py)")
    ap.add_argument("--batch", type=int, default=1, help="rows per GPU step (RunPod 80 GB: 4)")
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch x grad-accum")
    ap.add_argument("--max-hours", type=float, default=0, help="stop training after this many hours, keeping the best "
                    "checkpoint (0 = no limit); the final test scoring comes after it")
    ap.add_argument("--grad-ckpt", default="unsloth", choices=["unsloth", "gpu"],
                    help="unsloth: activations parked in CPU RAM (12 GB card); gpu: recomputed on the GPU, no copies (80 GB card)")
    ap.add_argument("--workers", type=int, default=2, help="data-loading processes (RunPod: 8, so the GPU never waits)")
    ap.add_argument("--min-util", type=float, default=85, help="the smoke test flags GPU utilisation below this (%%)")
    ap.add_argument("--test-per-task", type=int, default=20, help="held-out rows scored per task at the end (0 = all)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--eval-every", type=float, default=0.1, help="share of the whole run between validations")
    ap.add_argument("--val-per-task", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--smoke-steps", type=int, default=50, help="loss is logged every 10 steps: 50 gives 4 values to compare")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="skip scoring the validation rows at the end")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import data_lsec as DG
    import eval_lsec as E
    ds = DG.find_dataset()
    keep = json.loads((HERE / "keep_ids.json").read_text()) if (HERE / "keep_ids.json").exists() else None
    max_len = args.max_len or (keep["max_len"] if keep else 2048)
    ids = set(keep["ids"]) if keep else None
    train_rows = [r for r in DG.load_rows(ds, "train") if ids is None or r["id"] in ids]
    val_rows = [r for r in DG.load_rows(ds, "val", limit_per_task=args.val_per_task, seed=3) if ids is None or r["id"] in ids]
    steps = max(1, math.ceil(len(train_rows) * args.epochs / (args.batch * args.grad_accum)))
    eval_steps = max(10, round(steps * args.eval_every))
    log(f"model {args.model}; max_len {max_len}; train rows {len(train_rows)}, val rows {len(val_rows)}; "
        f"effective batch {args.batch} x {args.grad_accum}; {args.epochs} epochs = {steps} steps; validation every {eval_steps}"
        + (f"; time limit {args.max_hours} h" if args.max_hours else ""), out)

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True          # A100 / H100: faster matrix maths, same bf16 training
    torch.backends.cudnn.allow_tf32 = True
    from unsloth import FastVisionModel, is_bf16_supported
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback, TrainerCallback

    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=False,
                                                         use_gradient_checkpointing="unsloth" if args.grad_ckpt == "unsloth" else True)
    log("model loaded", out)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=3407)
    FastVisionModel.for_training(model)
    sampler = GpuSampler()
    sampler.start()

    class Smoke(TrainerCallback):
        def __init__(self):
            self.done = False

        def on_train_begin(self, a, state, control, **kw):
            self.t0, self.s0 = time.time(), state.global_step

        def on_step_end(self, a, state, control, **kw):
            if state.global_step - self.s0 == 10:               # speed from step 10 on: start-up and compiling left out
                self.t0, self.s10 = time.time(), state.global_step
            if self.done or state.global_step - self.s0 < args.smoke_steps:
                return
            self.done = True
            import torch
            secs = (time.time() - self.t0) / max(1, state.global_step - getattr(self, "s10", self.s0))
            left_h = secs * (steps - state.global_step) / 3600
            if args.max_hours and left_h > args.max_hours:
                log(f"  the full {args.epochs} epochs would take {left_h:.1f} h: training stops at the {args.max_hours} h limit "
                    "and keeps the best checkpoint so far", out)
            evals_h = (steps / eval_steps) * len(val_rows) * 0.5 / 3600
            # this step's loss is logged after this callback: compare the first half of the logged losses with the second
            losses = [h["loss"] for h in state.log_history if "loss" in h]
            half = len(losses) // 2
            first = sum(losses[:half]) / half if half else 0
            last = sum(losses[half:]) / (len(losses) - half) if half else 0
            peak = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            FastVisionModel.for_inference(model)
            fmt = []
            for row in [r for r in val_rows if r["task"] in ("qa_level", "qa_table", "img_tile", "qa_absent")][:4]:
                try:
                    msgs, imgs = DG.to_chat(row, ds)
                    pred = E.generate(model, processor, msgs[:-1], imgs, 200)
                    ok, why, _ = E.score(row, pred)
                    fmt.append((row["task"], ok, why, pred[:120]))
                except Exception as e:                    # noqa: BLE001
                    fmt.append((row["task"], False, f"error {e}", ""))
            FastVisionModel.for_training(model)
            fails = []
            if half and not last < first:                      # needs at least 2 logged losses to compare
                fails.append(f"loss not falling ({first:.3f} -> {last:.3f})")
            if any(w.startswith("error") for _, _, w, _ in fmt):
                fails.append("generation errors")
            util = sum(sampler.util[-6:]) / max(1, len(sampler.util[-6:]))
            low_util = sampler.util and util < args.min_util
            lines = [f"SMOKE TEST after {state.global_step} steps",
                     f"  loss {first:.3f} -> {last:.3f}",
                     f"  speed {secs:.1f} s/step; {steps} steps in the run",
                     f"  GPU utilisation {util:.0f} %, peak GPU memory {peak * 100:.0f} %"
                     + ("  (above 95 %: lower --max-len if it runs out of memory)" if peak > 0.95 else "")
                     + (f"\n  WARNING: GPU utilisation below {args.min_util:.0f} %: the GPU is waiting. Restart with a bigger batch "
                        f"(--batch {args.batch * 2} --grad-accum {max(1, args.grad_accum // 2)}"
                        + (" - memory allows it" if peak < 0.45 else "") + f") and/or more --workers (now {args.workers})"
                        if low_util else ""),
                     f"  remaining: training {left_h:.1f} h + validations {evals_h:.1f} h",
                     "  answers on a few validation rows (early in training, values may still be wrong):"]
            lines += [f"    {t}: {'ok' if ok else 'not yet'} {w} | {p!r}" for t, ok, w, p in fmt]
            lines.append("RESULT: " + ("PASS - training continues" if not fails else "FAIL - " + "; ".join(fails) + " - training stopped"))
            (out / "smoke_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            for l in lines:
                log(l, out)
            if fails:
                control.should_training_stop = True
                (out / "SMOKE_FAILED").write_text("; ".join(fails))

    class TimeLimit(TrainerCallback):
        """Stop at --max-hours; the trainer then loads the best checkpoint as usual."""
        def on_train_begin(self, a, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, a, state, control, **kw):
            if args.max_hours and time.time() - self.t0 > args.max_hours * 3600:
                log(f"time limit {args.max_hours} h reached at step {state.global_step}/{steps}: stopping", out)
                control.should_training_stop = True
                control.should_evaluate = True
                control.should_save = True

    class Progress(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and ("loss" in logs or "eval_loss" in logs):
                log(f"step {state.global_step}/{steps} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items() if isinstance(v, float)), out)

    trainer = SFTTrainer(
        model=model, tokenizer=processor,
        data_collator=DG.Collator(processor, ds),
        train_dataset=DG.LazyRows(train_rows), eval_dataset=DG.LazyRows(val_rows),
        callbacks=[Smoke(), TimeLimit(), Progress(), EarlyStoppingCallback(early_stopping_patience=args.patience)],
        args=SFTConfig(
            output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
            dataloader_drop_last=True,   # (a short last batch broke Qwen3.5's position ids in validation at batch 8)
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=steps,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01, optim="adamw_8bit",
            bf16=is_bf16_supported(), fp16=not is_bf16_supported(), logging_steps=10,
            eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            dataloader_num_workers=args.workers, dataloader_pin_memory=True, dataloader_persistent_workers=args.workers > 0,
            seed=3407, report_to="none",
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_len),
    )
    if not args.resume:                    # a validation pass first: v1's first run crashed in validation after 20 min
        m = trainer.evaluate(eval_dataset=DG.LazyRows(val_rows[:12]))
        log(f"pre-flight validation on 12 rows works (eval_loss {m.get('eval_loss', float('nan')):.3f})", out)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    if (out / "SMOKE_FAILED").exists():
        log("stopped by the smoke test - see smoke_report.txt", out)
        return

    adapter = out / "adapter"
    model.save_pretrained(str(adapter))
    processor.save_pretrained(str(adapter))
    json.dump(trainer.state.log_history, open(out / "log_history.json", "w"), indent=1)
    shutil.make_archive(str(out / "adapter"), "zip", str(adapter))
    log(f"adapter saved: {adapter} and {out / 'adapter.zip'} (best eval_loss {trainer.state.best_metric})", out)
    if not args.no_test:
        FastVisionModel.for_inference(model)
        E.evaluate(model, processor, ds, out / "eval", "val", per_task=args.test_per_task, log=lambda m: log(m, out))
    log("ALL DONE - try it: python ask_lsec.py --adapter " + str(adapter) + " --pdf <L-section.pdf> -i", out)


if __name__ == "__main__":
    main()
