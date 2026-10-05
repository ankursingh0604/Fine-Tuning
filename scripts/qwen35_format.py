"""Convert v4 dataset rows to Qwen3.5's image and box conventions (used by the training script).

The dataset keeps boxes in absolute pixels of the image as saved (Qwen2.5-VL's convention, sides padded to 28).
Qwen3.5 differs:
  - vision patches of 16 px merged 2 x 2, so image sides should be multiples of 32 (preprocessor_config.json:
    patch_size 16, merge_size 2) - images are padded with white on the right / bottom, which leaves every
    absolute box unchanged;
  - it uses the Qwen3-VL processor, and Qwen3-VL writes grounding boxes as relative coordinates on a 0-1000
    scale ({"bbox_2d": [x1, y1, x2, y2]} with each value = pixel / side * 1000).
BOX_MODE switches the box convention in one place ("rel1000" or "abs"); the training smoke test checks a few boxes
with the base model before the long run.

    .venv\\Scripts\\python scripts\\qwen35_format.py         # validates every row of data/v4/dataset (no files written)
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image

MULTIPLE = 32
BOX_MODE = "rel1000"
BOX_RE = re.compile(r'"bbox_2d":\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def padded_size(w, h, m=MULTIPLE):
    return -(-w // m) * m, -(-h // m) * m


def pad_image(img, m=MULTIPLE):
    W, H = padded_size(*img.size, m)
    if (W, H) == img.size:
        return img
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(img.convert("RGB"), (0, 0))
    return canvas


def convert_boxes(text, size):
    """Every bbox_2d in text, from absolute pixels to the model's convention, for an image of `size` (after padding)."""
    if BOX_MODE == "abs":
        return text
    W, H = size

    def rel(m):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        r = [round(x1 / W * 1000), round(y1 / H * 1000), round(x2 / W * 1000), round(y2 / H * 1000)]
        return '"bbox_2d": [' + ", ".join(str(v) for v in r) + "]"
    return BOX_RE.sub(rel, text)


def row_sizes(row):
    """Padded sizes of a row's images (normal rows: width/height; reasoning rows: sizes)."""
    if row.get("sizes"):
        return [padded_size(*s) for s in row["sizes"]]
    return [padded_size(row["width"], row["height"])]


def convert_row(row):
    """A copy of the row with boxes in the model's convention. Boxes appear only in single-image rows."""
    sizes = row_sizes(row)
    out = json.loads(json.dumps(row))
    for msg in out["messages"]:
        for c in msg["content"] if isinstance(msg["content"], list) else []:
            if c.get("type") == "text" and "bbox_2d" in c["text"]:
                c["text"] = convert_boxes(c["text"], sizes[0])
    out["padded_sizes"] = sizes
    return out


def validate(ds):
    """Every box inside its image before and after conversion; round trip within 1 px."""
    n_rows = n_boxes = 0
    worst = 0.0
    problems = []
    for f in sorted(ds.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            row = json.loads(line)
            if "messages" not in row or not row.get("width") and not row.get("sizes"):
                continue
            n_rows += 1
            W, H = row_sizes(row)[0]
            w, h = (row["width"], row["height"]) if row.get("width") else row["sizes"][0]
            for msg in row["messages"]:
                for c in msg["content"] if isinstance(msg["content"], list) else []:
                    if c.get("type") != "text":
                        continue
                    for m in BOX_RE.finditer(c["text"]):
                        x1, y1, x2, y2 = (float(v) for v in m.groups())
                        n_boxes += 1
                        if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h):
                            problems.append(f"{row['id']}: box {m.group(0)} outside {w}x{h}")
                        rel = [round(x1 / W * 1000), round(y1 / H * 1000), round(x2 / W * 1000), round(y2 / H * 1000)]
                        back = [rel[0] * W / 1000, rel[1] * H / 1000, rel[2] * W / 1000, rel[3] * H / 1000]
                        worst = max(worst, max(abs(a - b) for a, b in zip(back, (x1, y1, x2, y2))))
    return n_rows, n_boxes, worst, problems


if __name__ == "__main__":
    ds = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "v4" / "dataset"
    n_rows, n_boxes, worst, problems = validate(ds)
    print(f"{n_rows} rows, {n_boxes} boxes checked; worst round-trip error {worst:.2f} px; {len(problems)} boxes outside their image")
    for p in problems[:10]:
        print("  ", p)
