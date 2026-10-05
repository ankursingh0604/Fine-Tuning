"""Draw what the annotator found on the GAD itself, for checking by eye.

    .venv\\Scripts\\python gad_tools\\overlay_gad.py GAD_ID [GAD_ID ...]      -> data/gad/overlays/<gad_id>.png

Blue boxes: views (with kind); purple: panel sections read as text (notes, tables, title block);
green: levels; orange: labelled dimensions; grey: other dimension figures.
"""
import json
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANN = ROOT / "data" / "gad" / "annotations"
OUT = ROOT / "data" / "gad" / "overlays"
Z = 0.6


def overlay(gid):
    a = json.loads((ANN / f"{gid}.json").read_text(encoding="utf-8"))
    page = pymupdf.open(ROOT / "GAD" / a["source_pdf"])[0]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(Z, Z))
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    d = ImageDraw.Draw(im)

    def box(b, color, w=2):
        d.rectangle([v * Z for v in b], outline=color, width=w)
    for p in a.get("panel", []):
        box(p["box"], (150, 0, 200), 3)
        d.text((p["box"][0] * Z + 3, p["box"][1] * Z - 10), p["kind"], fill=(150, 0, 200))
    for v in a["views"]:
        box(v["bbox"], (0, 70, 255), 3)
        d.text((v["bbox"][0] * Z + 3, v["bbox"][1] * Z + 2), v["kind"], fill=(0, 70, 255))
    for l in a["levels"]:
        box(l["bbox"], (0, 160, 0))
    for x in a["labelled_dims"]:
        box(x["bbox"], (255, 140, 0))
    for x in a["dims"]:
        box(x["bbox"], (150, 150, 150), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / f"{gid}.png")
    return OUT / f"{gid}.png"


if __name__ == "__main__":
    for g in sys.argv[1:]:
        print(overlay(g))
