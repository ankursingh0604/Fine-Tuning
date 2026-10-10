"""Free CPU checks before training (needs only the Qwen3.5 processor files, ~20 MB, not the model).

    python check_lsec.py                      # processor of unsloth/Qwen3.5-4B
    python check_lsec.py --max-len 2048

1. Renders samples of every task with the real Qwen3.5 chat template and checks: thinking off, labels only on the
   answer (the question and the L-section sheet facts are never trained on), one image placeholder per image.
2. Measures every train / val / test row (text tokens + image tokens: one per 32 x 32 px) and chooses the maximum
   sequence length (99.5th percentile, at most --cap for a 12 GB card).
3. Writes keep_ids.json (rows within the limit; train_lsec.py trains only on these), check_report.txt and
   rendered_samples.txt.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_lsec as DG          # noqa: E402


def row_length(row, processor, ds):
    msgs, _ = DG.to_chat({**row, "image": None}, ds)
    if row.get("image"):
        w, h = Image.open(ds / row["image"]).size
        n_img = ((w + 31) // 32) * ((h + 31) // 32)
    else:
        n_img = 0
    text = DG.render(processor, msgs)
    return len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]) + n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--cap", type=int, default=2560, help="largest max length chosen automatically (12 GB card)")
    ap.add_argument("--samples-per-task", type=int, default=2)
    args = ap.parse_args()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    ds = DG.find_dataset()
    problems, samples = Counter(), []
    for split in ("train", "val"):
        for row in DG.load_rows(ds, split, limit_per_task=args.samples_per_task, seed=1):
            msgs, imgs = DG.to_chat(row, ds)
            text = DG.render(processor, msgs)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            mask = DG.assistant_mask(ids, tok)
            trained = tok.decode([t for t, m in zip(ids, mask) if m])
            if msgs[-1]["content"].strip()[:60] not in trained:
                problems["answer not in the trained tokens"] += 1
            q = DG.text_of(msgs[0]["content"]).strip()
            if q[:50] in trained or "L-section sheet facts" in trained:
                problems["question / facts inside the trained tokens"] += 1
            if "<think>" in trained and "</think>" in trained and trained.split("</think>")[0].strip("<think>\n ") != "":
                problems["thinking text in an answer"] += 1
            n_ph = text.count("<|vision_start|>")
            if n_ph != len(imgs):
                problems["image placeholders != images"] += 1
            if len(samples) < 30:
                samples.append(f"===== {row['id']} [{row['task']}]\n{text}\n----- trained part -----\n{trained}\n")
    lengths, by = {}, defaultdict(list)
    for split in ("train", "val", "test"):
        for row in DG.load_rows(ds, split):
            n = row_length(row, processor, ds)
            lengths[row["id"]] = n
            by[(split, row["task"])].append(n)
    allv = sorted(v for k, v in lengths.items())
    p995 = allv[min(len(allv) - 1, int(0.995 * len(allv)))]
    max_len = args.max_len or min(args.cap, int(math.ceil(p995 / 256) * 256))
    keep = [i for i, n in lengths.items() if n <= max_len]
    (HERE / "keep_ids.json").write_text(json.dumps({"max_len": max_len, "ids": keep}), encoding="utf-8")
    rep = [f"Processor: {args.model}", f"Dataset: {ds}",
           "Rendering / label checks: " + ("all passed" if not problems else "PROBLEMS: " + "; ".join(f"{k} x{v}" for k, v in problems.items())),
           f"Rows measured: {len(lengths)}; 99.5th percentile {p995} tokens; longest {allv[-1]}"]
    for (split, task), v in sorted(by.items()):
        v = sorted(v)
        rep.append(f"  {split:<5} {task:<15} rows {len(v):>5}  median {v[len(v) // 2]:>5}  max {v[-1]:>5}")
    tr = [n for (s, t), v in by.items() if s == "train" for n in v if n <= max_len]
    rep.append(f"Max sequence length: {max_len}; rows kept {len(keep)} of {len(lengths)} ({len(lengths) - len(keep)} longer rows left out)")
    rep.append(f"Training rows kept: {len(tr)}; tokens per epoch about {sum(tr) / 1e6:.1f} million")
    (HERE / "check_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    (HERE / "rendered_samples.txt").write_text("\n".join(samples), encoding="utf-8")
    print("\n".join(rep))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
