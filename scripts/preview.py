"""Draw every annotation over the rendered sheets so they can be checked by eye.

    python scripts/preview.py            # every sheet
    python scripts/preview.py 100 104    # chosen sheets

Writes data/review/sheet_XXX_overlay.png at 1 px per PDF point (so a pixel position equals the
coordinate stored in the JSON), with a colour key in the top-left corner.
"""
import json
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotations"
OUT = ROOT / "data" / "review"

# colour per annotation type (RGB)
STYLE = {
    "region": ((40, 90, 230), "sheet region"),
    "plan_callout": ((0, 150, 0), "bridge callout, plan"),
    "lsection_callout": ((200, 0, 160), "bridge callout, L-section"),
    "lsection_levels": ((240, 120, 0), "bridge level block"),
    "curve": ((130, 0, 220), "curve box (proposed)"),
    "existing_curve": ((90, 90, 90), "curve table (existing)"),
    "transition": ((0, 160, 200), "transition point ST/TC/CT/TS"),
    "grade_point": ((200, 160, 0), "grade point (plan)"),
    "gp_lsec": ((160, 110, 0), "grade point / VPI (L-section)"),
    "gradient_segment": ((230, 60, 60), "gradient segment (schematic)"),
    "band_column": ((0, 120, 120), "data-band column"),
    "bearing": ((120, 60, 0), "bearing"),
    "km": ((0, 0, 0), "KM post / KM mark"),
    "tbm": ((0, 170, 170), "TBM marker / table row"),
    "station": ((60, 120, 60), "station block"),
    "officer": ((100, 100, 200), "officer / reference drawing"),
}


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def boxes(a):
    for name, b in a["regions"].items():
        yield "region", name, b
    for br in a["bridges"]:
        for k in ("plan_callout", "lsection_callout", "lsection_levels"):
            if k in br:
                yield k, br["bridge_id"], br[k]["bbox"]
    for c in a["curves"]:
        yield "curve", c["curve_no"], c["bbox"]
    for c in a.get("existing_curves", []):
        yield "existing_curve", c.get("curve_no", ""), c["bbox"]
    for t in a.get("transition_points", []):
        yield "transition", t["type"], t["bbox"]
    for g in a.get("plan_grade_points", []):
        yield "grade_point", g["chainage"], g["bbox"]
    for g in a.get("grade_points", []):
        yield "gp_lsec", "", g["bbox"]
    for v in a.get("vpis", []):
        yield "gp_lsec", "", v["bbox"]
    for s in a.get("gradient_segments", []):
        yield "gradient_segment", s["gradient_label"], s["bbox"]
    for c in (a.get("bands") or {}).get("columns", []):
        yield "band_column", "", c["bbox"]
    for b in a.get("bearings", []):
        yield "bearing", "", b["bbox"]
    for k in a.get("km_posts", []) + a.get("km_marks", []):
        yield "km", "", k["bbox"]
    for t in a["tbm_markers"] + a["tbm_benchmarks"]:
        yield "tbm", t["tbm_id"], t["bbox"]
    for s in a.get("stations", []):
        yield "station", s["name"], s["bbox"]
    for o in a.get("officers", []) + a.get("reference_drawings", []):
        yield "officer", "", o["bbox"]


def render_sheet(pdf, a):
    pix = pdf[a["page_index"]].get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = Image.blend(img, Image.new("RGB", img.size, "white"), 0.25)
    d = ImageDraw.Draw(img)
    small = font(12)
    for kind, label, b in boxes(a):
        color = STYLE[kind][0]
        width = 3 if kind == "region" else 2
        d.rectangle(b, outline=color, width=width)
        if label and kind not in ("band_column", "gp_lsec"):
            d.text((b[0] + 2, b[1] - 13 if kind != "region" else b[1] + 3), str(label), fill=color, font=small)
    # colour key
    key_font = font(15)
    x0, y0, row_h = 6, 6, 20
    d.rectangle([x0, y0, x0 + 330, y0 + 10 + row_h * len(STYLE)], fill="white", outline="black")
    for i, (kind, (color, text)) in enumerate(STYLE.items()):
        y = y0 + 6 + i * row_h
        d.rectangle([x0 + 8, y + 3, x0 + 34, y + 15], outline=color, width=3)
        d.text((x0 + 42, y), text, fill="black", font=key_font)
    return img


def main(sheets):
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = {}
    for f in sorted(ANN.glob("sheet_*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        if sheets and a["sheet_info"]["sheet_no"] not in sheets:
            continue
        pdf = pdfs.setdefault(a["source_pdf"], pymupdf.open(ROOT / a["source_pdf"]))
        render_sheet(pdf, a).save(OUT / f"{f.stem}_overlay.png")
        print("wrote", OUT / f"{f.stem}_overlay.png")


if __name__ == "__main__":
    main(sys.argv[1:])
