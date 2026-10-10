"""L-section v5 training data: load data/v5/dataset rows, turn them into Qwen3.5 chat messages, render, mask labels.

Shared by check_lsec.py (CPU), train_lsec.py and eval_lsec.py (GPU). No GPU or Unsloth import here.
Every row is one question and one answer (thinking off); some rows carry one image.
"""
import json
import random
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent


def find_dataset(start=HERE):
    """dataset/ next to the kit (bundle layout) or data/v5/dataset in the project (repo layout)."""
    for cand in (start / "dataset", start.parent / "data" / "v5" / "dataset"):
        if (cand / "train.jsonl").exists():
            return cand
    raise FileNotFoundError("L-section v5 dataset not found next to the kit (dataset/train.jsonl)")


def load_rows(ds, split, limit_per_task=None, seed=0):
    p = ds / f"{split}.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []
    if limit_per_task:
        rng = random.Random(seed)
        by = {}
        for r in rows:
            by.setdefault(r["task"], []).append(r)
        rows = [x for k in sorted(by) for x in rng.sample(by[k], min(limit_per_task, len(by[k])))]
    return rows


def pad32(im):
    """Qwen3.5 uses 16 px patches merged 2 x 2: sides padded to multiples of 32 (white)."""
    W, H = (im.width + 31) // 32 * 32, (im.height + 31) // 32 * 32
    if (W, H) == im.size:
        return im
    c = Image.new("RGB", (W, H), "white")
    c.paste(im, (0, 0))
    return c


def text_of(content):
    return content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content if c.get("type") == "text")


def to_chat(row, ds):
    """(messages, images) in the form the Qwen3.5 chat template takes."""
    user, asst = row["messages"][0], row["messages"][1]
    images = [pad32(Image.open(ds / row["image"]).convert("RGB"))] if row.get("image") else []
    ucontent = [{"type": "image"} if c.get("type") == "image" else {"type": "text", "text": c["text"]} for c in user["content"]] \
        if isinstance(user["content"], list) else [{"type": "text", "text": user["content"]}]
    return [{"role": "user", "content": ucontent}, {"role": "assistant", "content": text_of(asst["content"])}], images


def render(processor, msgs, add_generation_prompt=False):
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)


def assistant_mask(input_ids, tokenizer):
    """1 for tokens inside assistant turns (after '<|im_start|>assistant\\n' up to and including '<|im_end|>')."""
    start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = list(input_ids)
    mask = [0] * len(ids)
    i, n = 0, len(start)
    while i < len(ids):
        if ids[i:i + n] == start:
            j = i + n
            while j < len(ids) and ids[j] != end_id:
                mask[j] = 1
                j += 1
            if j < len(ids):
                mask[j] = 1
            i = j + 1
        else:
            i += 1
    return mask


def image_tokens(img):
    return (img.width // 32) * (img.height // 32)


class Collator:
    """Batches rows: chat template per row, images, labels on the assistant's answer only."""

    def __init__(self, processor, ds):
        self.processor, self.ds = processor, ds
        self.tok = getattr(processor, "tokenizer", processor)

    def __call__(self, rows):
        import torch
        texts, images = [], []
        for row in rows:
            msgs, imgs = to_chat(row, self.ds)
            texts.append(render(self.processor, msgs))
            images += imgs
        batch = self.processor(text=texts, images=images or None, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for b in range(labels.shape[0]):
            mask = torch.tensor(assistant_mask(batch["input_ids"][b].tolist(), self.tok))
            labels[b][mask == 0] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class LazyRows:
    """Rows whose images are only opened when the batch is built (keeps RAM small)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]
