"""Free CPU checks before renting a GPU (needs only the Qwen3.5 processor files, ~20 MB, not the model).

    python check_v4.py                       # model: unsloth/Qwen3.5-27B (same tokenizer / template for every size)
    python check_v4.py --max-len 4096

1. Renders samples of every task with the real Qwen3.5 chat template and checks: reasoning kept for thinking rows,
   tool calls and tool results rendered for agent rows, labels only on the assistant's turns (the question and tool
   results are never trained on), image placeholders matching the images.
2. Measures the token length of every training and validation row (text tokens + image tokens: one per 32 x 32 px)
   and recommends the maximum sequence length.
3. Writes keep_ids.json (rows within the limit; train_v4.py trains only on these), check_report.txt and
   rendered_samples.txt.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import data_v4 as DV          # noqa: E402
import qwen35_format as Q     # noqa: E402


def row_length(row, processor, ds, tools):
    """Tokens of a row as trained: rendered text (image placeholders replaced by their real token count)."""
    msgs, imgs, thinking, has_tools = DV.to_chat(row, ds) if DV.kind(row) == "agent" else (None, None, None, None)
    if DV.kind(row) != "agent":
        # sizes from the row (no image loading needed): padded to multiples of 32
        sizes = Q.row_sizes(row)
        n_img = sum((w // 32) * (h // 32) for w, h in sizes)
        conv = Q.convert_row(row)
        user, asst = conv["messages"][0], conv["messages"][1]
        ucontent = [{"type": "image"} if c.get("type") == "image" else {"type": "text", "text": c["text"]} for c in user["content"]]
        ans = DV.text_of(asst["content"])
        m = DV.THINK_RE.match(ans)
        amsg = {"role": "assistant", "content": m.group(2).strip() if m else ans}
        if m:
            amsg["reasoning_content"] = m.group(1).strip()
        text = DV.render(processor, [{"role": "user", "content": ucontent}, amsg], bool(m))
        n_text = len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])
        return n_text - len(sizes) + n_img
    text = DV.render(processor, msgs, thinking, tools)
    return len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-27B")
    ap.add_argument("--max-len", type=int, default=0, help="fix the limit instead of choosing it from the lengths")
    ap.add_argument("--samples-per-task", type=int, default=3)
    args = ap.parse_args()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    ds, tools = DV.find_dataset(), DV.find_tools()
    report, samples = [], []
    problems = Counter()

    # 1. rendering and label checks on samples of every task
    for split in ("train", "val"):
        for row in DV.load_rows(ds, split, limit_per_task=args.samples_per_task, seed=1):
            msgs, imgs, thinking, has_tools = DV.to_chat(row, ds)
            text = DV.render(processor, msgs, thinking, tools if has_tools else None)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            mask = DV.assistant_mask(ids, tok)
            trained = tok.decode([t for t, m in zip(ids, mask) if m])
            final = msgs[-1].get("content") or ""
            if final and final.strip()[:60] not in trained:
                problems["final answer not in the trained tokens"] += 1
            for m in DV.drop_old_thinking(msgs):
                if m["role"] == "user":
                    u = DV.text_of(m["content"]).strip()[:50]
                    if u and u in trained:
                        problems["question text inside the trained tokens"] += 1
                if m["role"] == "tool" and m["content"][:40] in trained:
                    problems["tool result inside the trained tokens"] += 1
                if m.get("reasoning_content") and m["reasoning_content"][:40] not in text:
                    problems["reasoning missing from the rendered text"] += 1
            if has_tools and "tool_call" not in text and any(m.get("tool_calls") for m in msgs):
                problems["tool call missing from the rendered text"] += 1
            n_ph = text.count("<|image_pad|>") or text.count("<|vision_start|>")
            if n_ph != len(imgs):
                problems["image placeholders != images"] += 1
            if len(samples) < 40 and (len(samples) < 10 or DV.kind(row) != "image"):
                samples.append(f"===== {row['id']} [{row['task']}] thinking={thinking}\n{text}\n----- trained part -----\n{trained}\n")

    # 2. lengths of every train / val row
    lengths = {}
    by_kind = defaultdict(list)
    for split in ("train", "val"):
        for row in DV.load_rows(ds, split):
            n = row_length(row, processor, ds, tools)
            lengths[row["id"]] = n
            by_kind[(split, DV.kind(row))].append(n)
    allv = sorted(lengths.values())
    p995 = allv[min(len(allv) - 1, int(0.995 * len(allv)))]
    max_len = args.max_len or int(math.ceil(p995 / 256) * 256)
    keep = [i for i, n in lengths.items() if n <= max_len]
    (HERE / "keep_ids.json").write_text(json.dumps({"max_len": max_len, "ids": keep}), encoding="utf-8")

    report.append(f"Processor: {args.model}")
    report.append(f"Dataset: {ds}")
    report.append("Rendering / label checks on samples of every task: " +
                  ("all passed" if not problems else "PROBLEMS: " + "; ".join(f"{k} x{v}" for k, v in problems.items())))
    report.append(f"Rows measured: {len(lengths)}; 99.5th percentile {p995} tokens; longest {allv[-1]}")
    for (split, k), v in sorted(by_kind.items()):
        v = sorted(v)
        report.append(f"  {split:<5} {k:<9} rows {len(v):>6}  median {v[len(v) // 2]:>5}  p95 {v[int(0.95 * len(v))]:>5}  max {v[-1]:>5}")
    report.append(f"Recommended max sequence length: {max_len}; rows kept {len(keep)} of {len(lengths)} "
                  f"({len(lengths) - len(keep)} longer rows are left out rather than cut off)")
    report.append(f"Tokens in one epoch of the kept training rows: about "
                  f"{sum(n for (s, k), v in by_kind.items() if s == 'train' for n in v if n <= max_len) / 1e6:.1f} million")
    (HERE / "check_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (HERE / "rendered_samples.txt").write_text("\n".join(samples), encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {HERE / 'keep_ids.json'}, check_report.txt, rendered_samples.txt")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
