"""v4 training data: load every part of data/v4/dataset, convert to Qwen3.5 conventions, render with the chat template.

Shared by check_v4.py (CPU, before renting), train_v4.py and eval_v4.py (GPU). No GPU or Unsloth import here.

Parts mixed for training (train split), validation (val split) and testing (test split + test_lowdpi):
    normal      {split}.jsonl                 direct answers, thinking off
    reasoning   reasoning_{split}.jsonl       chain of thought, thinking on
    generic     generic_{split}.jsonl         generic table reading, thinking off
    aug         aug_train.jsonl, aug_reasoning_train.jsonl   low-DPI copies and 200 dpi crops (train only)
    agent       agent_{split}.jsonl           tool use (docs/tools_v4.json), thinking per conversation
"""
import json
import random
import re
from pathlib import Path

import sys

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))       # repo layout; in the bundle qwen35_format.py sits next to this file
import qwen35_format as Q     # noqa: E402
PARTS = {
    "train": ["train.jsonl", "reasoning_train.jsonl", "generic_train.jsonl", "aug_train.jsonl",
              "aug_reasoning_train.jsonl", "agent_train.jsonl"],
    "val": ["val.jsonl", "reasoning_val.jsonl", "generic_val.jsonl", "agent_val.jsonl"],
    "test": ["test.jsonl", "reasoning_test.jsonl", "generic_test.jsonl", "agent_test.jsonl", "test_lowdpi.jsonl"],
}
THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", re.S)


def find_dataset(start=HERE):
    """data/v4/dataset next to the kit (bundle layout) or in the project (repo layout)."""
    for base in (start, start.parent):
        for cand in (base / "dataset", base / "data" / "v4" / "dataset"):
            if (cand / "train.jsonl").exists():
                return cand
    raise FileNotFoundError("data/v4/dataset not found next to the kit")


def find_tools(start=HERE):
    for cand in (start / "tools_v4.json", start.parent / "docs" / "tools_v4.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))["tools"]
    raise FileNotFoundError("tools_v4.json not found")


def load_rows(ds, split, limit_per_task=None, seed=0):
    rows = []
    for f in PARTS[split]:
        p = ds / f
        if not p.exists():
            continue
        part = [json.loads(l) for l in open(p, encoding="utf-8")]
        for r in part:
            r["_file"] = f
        rows += part
    if limit_per_task:
        rng = random.Random(seed)
        by = {}
        for r in rows:
            by.setdefault((r["_file"], r["task"]), []).append(r)
        rows = [x for k in sorted(by) for x in rng.sample(by[k], min(limit_per_task, len(by[k])))]
    return rows


def kind(row):
    if row.get("tools") or row["task"].startswith("agent_"):
        return "agent"
    if row["task"].startswith("reason_"):
        return "reasoning"
    return "image"


def text_of(content):
    return content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content if c.get("type") == "text")


# ---------------------------------------------------------------- one row -> chat messages (+ images, thinking flag)

def to_chat(row, ds):
    """(messages, images, thinking, has_tools) in the form the Qwen3.5 chat template takes. Images are loaded and
    padded to multiples of 32; boxes are converted to the model's convention (qwen35_format.BOX_MODE)."""
    k = kind(row)
    if k == "agent":
        msgs = []
        for m in row["messages"]:
            m = dict(m)
            if m["role"] == "assistant":
                if m.get("reasoning"):
                    m["reasoning_content"] = m.pop("reasoning")
                if m.get("tool_calls"):
                    m["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"],
                                                                         "arguments": tc["function"]["arguments"]}}
                                       for tc in m["tool_calls"]]
            msgs.append(m)
        return msgs, [], bool(row.get("thinking")), True
    row = Q.convert_row(row)
    images = [Q.pad_image(Image.open(ds / p).convert("RGB")) for p in (row.get("images") or [row["image"]])]
    user, asst = row["messages"][0], row["messages"][1]
    ucontent = []
    for c in user["content"]:
        ucontent.append({"type": "image"} if c.get("type") == "image" else {"type": "text", "text": c["text"]})
    answer = text_of(asst["content"])
    m = THINK_RE.match(answer)
    amsg = {"role": "assistant", "content": m.group(2).strip()} if m else {"role": "assistant", "content": answer}
    if m:
        amsg["reasoning_content"] = m.group(1).strip()
    return [{"role": "user", "content": ucontent}, amsg], images, bool(m), False


def render(processor, msgs, thinking, tools=None, add_generation_prompt=False):
    """The chat-template text. Falls back to inline <think> blocks if this template ignores reasoning_content."""
    kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=thinking)
    if tools:
        kw["tools"] = tools
    text = processor.apply_chat_template(msgs, **kw)
    lost = [m["reasoning_content"] for m in msgs if m.get("reasoning_content") and m["reasoning_content"][:40] not in text]
    if lost:                                   # put the thinking inline so it is trained on
        msgs2 = []
        for m in msgs:
            m = dict(m)
            if m.get("reasoning_content"):
                m["content"] = f"<think>\n{m.pop('reasoning_content')}\n</think>\n\n{m.get('content') or ''}"
            msgs2.append(m)
        text = processor.apply_chat_template(msgs2, **kw)
    return text


# ---------------------------------------------------------------- labels: train only on the assistant's turns

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
    """Tokens one image costs: 16 px patches merged 2 x 2 -> one token per 32 x 32 px."""
    return (img.width // 32) * (img.height // 32)


class Collator:
    """Batches rows for training: chat template per row (tools, thinking switch), images, labels on assistant turns only."""

    def __init__(self, processor, ds, tools, max_len):
        self.processor, self.ds, self.tools, self.max_len = processor, ds, tools, max_len
        self.tok = getattr(processor, "tokenizer", processor)

    def __call__(self, rows):
        import torch
        texts, images = [], []
        for row in rows:
            msgs, imgs, thinking, has_tools = to_chat(row, self.ds)
            texts.append(render(self.processor, msgs, thinking, self.tools if has_tools else None))
            images.append(imgs)
        flat = [im for imgs in images for im in imgs]
        batch = self.processor(text=texts, images=flat or None, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for b in range(labels.shape[0]):
            mask = torch.tensor(assistant_mask(batch["input_ids"][b].tolist(), self.tok))
            labels[b][mask == 0] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class LazyRows:
    """A dataset of rows whose images are only opened when the batch is built (keeps RAM small)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]
