"""Read a whole Plan & L-Section sheet image with the fine-tuned model and save what it found.

    python read_sheet.py --adapter adapter.zip --image sheet_100.png
    python read_sheet.py --adapter adapter.zip --image sheet.jpg --4bit --out my_results

Writes, next to --out (default: a folder named after the image):
    <name>_bridges.csv   one row per bridge: callout, level block, band values at the nearest column, checks
    <name>_read.json     everything read, including the title block and TBM table
    <name>_overlay.png   the sheet with every bridge the model found (red = flagged or failed a check)
The image can be 50-200 dpi; 150-200 dpi gives the most reliable numbers.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from sheet_reader import Reader

Image.MAX_IMAGE_PIXELS = None          # whole A0 sheets at 200 dpi are ~70 megapixels

LEVELS = ["existing_formation_level", "min_formation_level_required", "proposed_formation_level", "bed_level",
          "high_flood_level", "free_board"]
BAND = ["chainage", "ground_level", "prop_fl", "prop_rl", "cut_fill", "exg_up_fl", "fl_difference", "track_distance"]


def save(result, out, name):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in result["bridges"]:
        d = b["data"] or {}
        lv = d.get("levels") or {}
        col = (b["band"] or {}).get("nearest_column") or {}
        rows.append({"bridge_id": b["bridge_id"], "read_from": b["read_from"],
                     **{k: d.get(k) for k in ("existing_type", "existing_span", "crossing", "proposal", "category", "chainage_m")},
                     **{k: lv.get(k) for k in LEVELS},
                     **{f"band_{k}": col.get(k) for k in BAND},
                     "band_distance_m": (b["band"] or {}).get("distance_m"),
                     "checks": " | ".join(c["message"] for c in b["checks"])})
    with open(out / f"{name}_bridges.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["bridge_id"])
        w.writeheader()
        w.writerows(rows)
    result["overlay"].save(out / f"{name}_overlay.png")
    keep = {k: v for k, v in result.items() if k != "overlay"}
    (out / f"{name}_read.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--out", help="output folder")
    args = ap.parse_args()

    from app import Model
    model = Model(args.adapter, args.four_bit)
    calls = {"n": 0}
    ask = model.ask

    def counted(imgs, q, *a, **k):
        calls["n"] += 1
        if calls["n"] % 10 == 0:
            print(f"  ... {calls['n']} questions asked")
        return ask(imgs, q, *a, **k)
    model.ask = counted

    t = time.time()
    result = Reader(model).read(Image.open(args.image))
    name = Path(args.image).stem
    out = save(result, Path(args.out or f"{name}_read"), name)

    print(f"\nRead in {(time.time() - t) / 60:.1f} min, {calls['n']} questions, image about {result['dpi']} dpi")
    for w in result["warnings"]:
        print("WARNING:", w)
    t_ = result["title"] or {}
    print(f"Sheet {t_.get('sheet_no', '?')}: {t_.get('title', '')} {t_.get('chainage_from', '')} - {t_.get('chainage_to', '')}")
    print(f"Bridges found: {len(result['bridges'])}, read: {sum(1 for b in result['bridges'] if b['data'])}, "
          f"with band values: {sum(1 for b in result['bridges'] if b['band'])}; TBMs: {len(result['tbm'] or [])}")
    for f in result["findings"]:
        print(f"[{f['severity']}] {f['message']}")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
