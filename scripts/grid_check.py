"""Render a sheet with a PDF-point grid and its region boxes, for checking
annotation coordinates by eye.

    python scripts/grid_check.py 94

Writes data/review/sheet_094_grid.png at 1 px per point, so a pixel position in
the image equals the PDF coordinate. Light lines every 100 pt, dark every 500 pt,
labelled along all four edges; each region box is labelled with its [x0, y0, x1, y1].
"""
import json
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main(sheet):
    ann = json.loads((ROOT / "data" / "annotations" / f"sheet_{sheet:0>3}.json").read_text(encoding="utf-8"))
    page = pymupdf.open(ROOT / ann["source_pdf"])[ann["page_index"]]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)       # 1 px = 1 pt
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = Image.blend(img, Image.new("RGB", img.size, "white"), 0.35)      # fade the drawing so the grid reads
    d = ImageDraw.Draw(img)
    small, big = font(16), font(22)
    W, H = img.size

    for x in range(0, W, 100):
        d.line([(x, 0), (x, H)], fill=(90, 90, 90) if x % 500 == 0 else (200, 200, 200), width=2 if x % 500 == 0 else 1)
        for y in (2, H - 20):
            d.text((x + 3, y), str(x), fill=(0, 0, 0), font=small)
    for y in range(0, H, 100):
        d.line([(0, y), (W, y)], fill=(90, 90, 90) if y % 500 == 0 else (200, 200, 200), width=2 if y % 500 == 0 else 1)
        for x in (2, W - 50):
            d.text((x, y + 3), str(y), fill=(0, 0, 0), font=small)

    colors = [(0, 90, 255), (230, 0, 120), (0, 150, 0), (200, 100, 0), (120, 0, 200)]
    for i, (name, b) in enumerate(ann["regions"].items()):
        c = colors[i % len(colors)]
        d.rectangle(b, outline=c, width=4)
        label = f"{name} [{b[0]:g}, {b[1]:g}, {b[2]:g}, {b[3]:g}]"
        tx, ty = b[0] + 8, b[1] + 8
        box = d.textbbox((tx, ty), label, font=big)
        d.rectangle([box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3], fill="white")
        d.text((tx, ty), label, fill=c, font=big)

    out = ROOT / "data" / "review" / f"sheet_{sheet:0>3}_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print("wrote", out, img.size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "94")
