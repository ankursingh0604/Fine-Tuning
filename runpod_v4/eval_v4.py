"""Score a v4 model on the test sets (called at the end of train_v4.py; can also be run on a saved adapter).

    python eval_v4.py --adapter /workspace/v4_run/adapter --per-task 8

Per task (and per DPI for the low-DPI test set):
    value_recall   share of the numbers in the reference answer that the prediction gives
    json_ok        prediction parses as JSON when the reference is JSON; field_acc = share of fields equal
    box_iou        mean best IoU for box answers
    think_ok       the model reasoned exactly when the reference does (thinking switch)
    tool_ok        agent rows: the first tool call names the right tool
Agent conversations are scored teacher-forced: the real tool results up to the last turn are given, and the final
answer is generated (plus, separately, the first tool call).
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_v4 as DV          # noqa: E402
import qwen35_format as Q     # noqa: E402

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
BOX_RE = Q.BOX_RE


def numbers(text):
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in NUM_RE.findall(text or "")}


def split_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    final = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    final = re.sub(r"<\|[^|]+\|>", "", final).strip()
    return (m.group(1).strip() if m else ""), final


def parse_json(text):
    m = re.search(r"```json\s*(.*?)```", text, re.S) or re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / u if u > 0 else 0.0


def boxes(text):
    return [[float(v) for v in m.groups()] for m in BOX_RE.finditer(text)]


def first_tool(text):
    m = re.search(r"<function=([\w_]+)>", text) or re.search(r'"name"\s*:\s*"([\w_]+)"', text)
    return m.group(1) if m else None


def generate(model, processor, msgs, images, thinking, tools, max_new_tokens=1024):
    import torch
    text = DV.render(processor, msgs, thinking, tools, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def score_row(model, processor, row, ds, tools):
    msgs, images, thinking, has_tools = DV.to_chat(row, ds)
    res = {}
    if has_tools:
        k_last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        ref = msgs[k_last]["content"] or ""
        pred = generate(model, processor, msgs[:k_last], [], thinking, tools)
        k_first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        if msgs[k_first].get("tool_calls"):
            pred_first = generate(model, processor, msgs[:k_first], [], thinking, tools, max_new_tokens=400)
            res["tool_ok"] = float(first_tool(pred_first) == msgs[k_first]["tool_calls"][0]["function"]["name"])
    else:
        ref = msgs[-1]["content"]
        pred = generate(model, processor, msgs[:-1], images, thinking, None)
    think, final = split_think(pred)
    ref_nums = numbers(ref)
    if ref_nums:
        res["value_recall"] = len(ref_nums & numbers(final)) / len(ref_nums)
    if not has_tools:
        res["think_ok"] = float(bool(think) == bool(msgs[-1].get("reasoning_content")))
    rj = parse_json(ref)
    if rj is not None:
        pj = parse_json(final)
        res["json_ok"] = float(pj is not None)
        if isinstance(rj, dict) and isinstance(pj, dict):
            flat = lambda d: {k: json.dumps(v, sort_keys=True) for k, v in d.items()}   # noqa: E731
            fr, fp = flat(rj), flat(pj)
            res["field_acc"] = sum(fp.get(k) == v for k, v in fr.items()) / max(1, len(fr))
    rb = boxes(ref)
    if rb:
        pb = boxes(final)
        res["box_iou"] = sum(max((iou(r, p) for p in pb), default=0.0) for r in rb) / len(rb)
    return res, pred


def evaluate(model, processor, ds, tools, out, per_task=8, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = DV.load_rows(ds, "test", limit_per_task=per_task, seed=7)
    agg = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    with open(out / "test_predictions.jsonl", "w", encoding="utf-8") as fp:
        for i, row in enumerate(rows):
            group = row["task"] + (f" @{row['aug'].replace('low_dpi_', '')}dpi" if row.get("aug", "").startswith("low_dpi") else "")
            try:
                res, pred = score_row(model, processor, row, ds, tools)
            except Exception as e:                       # noqa: BLE001
                res, pred = {"error": 1.0}, f"ERROR {type(e).__name__}: {e}"
            for k, v in res.items():
                agg[group][k].append(v)
            fp.write(json.dumps({"id": row["id"], "task": row["task"], "group": group, "scores": res, "prediction": pred},
                                ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                log(f"  eval {i + 1}/{len(rows)} ({(time.time() - t0) / 60:.1f} min)")
    metrics = ["value_recall", "field_acc", "json_ok", "box_iou", "think_ok", "tool_ok", "error"]
    with open(out / "test_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "rows"] + metrics)
        for g in sorted(agg):
            n = max(len(v) for v in agg[g].values())
            w.writerow([g, n] + [f"{sum(agg[g][m]) / len(agg[g][m]):.3f}" if agg[g][m] else "" for m in metrics])
    overall = {m: sum(v for g in agg for v in agg[g][m]) / max(1, sum(len(agg[g][m]) for g in agg)) for m in metrics if any(agg[g][m] for g in agg)}
    lines = [f"Test rows scored: {len(rows)} ({per_task} per task and file), {(time.time() - t0) / 60:.1f} min"]
    lines += [f"  overall {m}: {v:.3f}" for m, v in overall.items()]
    (out / "test_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    return overall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--per-task", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--box-mode", default=None, help="override qwen35_format.BOX_MODE (rel1000 or abs)")
    args = ap.parse_args()
    from unsloth import FastVisionModel
    if args.box_mode:
        Q.BOX_MODE = args.box_mode
    elif (Path(args.adapter) / "box_mode.txt").exists():
        Q.BOX_MODE = (Path(args.adapter) / "box_mode.txt").read_text().strip()
    model, processor = FastVisionModel.from_pretrained(args.adapter, load_in_4bit=False)
    FastVisionModel.for_inference(model)
    evaluate(model, processor, DV.find_dataset(), DV.find_tools(), args.out or Path(args.adapter).parent / "eval")
