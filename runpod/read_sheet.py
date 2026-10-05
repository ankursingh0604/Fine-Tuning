"""Read a whole Plan & L-Section sheet image with the fine-tuned model, save what it found, and answer questions.

    python read_sheet.py --adapter adapter.zip --image sheet_100.png
    python read_sheet.py --adapter adapter.zip --image sheet.jpg --4bit --out my_results

Ask questions (the sheet is read once - several minutes - and the reading is reused afterwards):
    python read_sheet.py --adapter adapter.zip --image sheet_100.png -q "ground level at bridge 560"
    python read_sheet.py --adapter adapter.zip --image sheet_100.png -q "bridge 560" -q "which bridges are flagged"
    python read_sheet.py --adapter adapter.zip --image sheet_100.png -i          (keep asking at a prompt)
Questions about bridges, flags, the title block and TBMs are answered instantly from the saved reading.
A chainage away from any bridge ("ground level at CH 1241000") makes the model read that band column,
so keep --adapter on the command line for those.

Writes, next to --out (default: a folder named after the image):
    <name>_bridges.csv   one row per bridge: callout, level block, band values interpolated at the bridge chainage, checks
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

from sheet_qa import SheetQA
from sheet_reader import Reader, Sheet, band_values

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
        col = band_values(b["band"])
        band = b["band"] or {}
        rows.append({"bridge_id": b["bridge_id"], "read_from": b["read_from"],
                     **{k: d.get(k) for k in ("existing_type", "existing_span", "crossing", "proposal", "category", "chainage_m")},
                     **{k: lv.get(k) for k in LEVELS},
                     **{f"band_{k}": col.get(k) for k in BAND},
                     "band_x1": band.get("x1"), "band_x2": band.get("x2"),
                     "band_note": band.get("error") or ("interpolated" if "y" in band and band.get("x1") != band.get("x2")
                                                        else "on a printed column" if "y" in band else ""),
                     "checks": " | ".join(c["message"] for c in b["checks"])})
    with open(out / f"{name}_bridges.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["bridge_id"])
        w.writeheader()
        w.writerows(rows)
    result["overlay"].save(out / f"{name}_overlay.png")
    keep = {k: v for k, v in result.items() if k != "overlay"}
    (out / f"{name}_read.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


class Session:
    """Loads the model and the sheet only when something needs reading, then keeps them."""

    def __init__(self, args, model=None):
        self.args, self.model, self.sheet, self.calls = args, model, None, 0

    def reader(self):
        if self.model is None:
            if not self.args.adapter:
                raise SystemExit("Reading needs the model: add --adapter adapter.zip")
            from app import Model
            self.model = Model(self.args.adapter, self.args.four_bit)
            ask = self.model.ask

            def counted(imgs, q, *a, **k):
                self.calls += 1
                if self.calls % 10 == 0:
                    print(f"  ... {self.calls} questions asked")
                return ask(imgs, q, *a, **k)
            self.model.ask = counted
        return Reader(self.model)

    def read(self):
        t = time.time()
        result = self.reader().read(Image.open(self.args.image))
        print(f"\nRead in {(time.time() - t) / 60:.1f} min, {self.calls} questions to the model, image about {result['dpi']} dpi")
        return result

    def band_reader(self, result):
        """For chainages away from the bridges: the model reads that band column from the image."""
        if not self.args.adapter and self.model is None:
            return None

        def read_band(ch, name):
            rd = self.reader()
            if self.sheet is None:
                print("  (reading that band column from the image ...)")
                self.sheet = Sheet(Image.open(self.args.image))
            if not result.get("band_layout"):
                result["band_layout"] = rd.band_layout(self.sheet)
            return rd.band_at(self.sheet, result["band_layout"], ch, name) if result["band_layout"] else None
        return read_band


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="adapter.zip (needed to read the sheet; not needed to ask about a sheet already read)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--out", help="output folder (default: <image name>_read)")
    ap.add_argument("--question", "-q", action="append", default=[],
                    help='a question about the sheet, e.g. "ground level at bridge 560"; can be given several times')
    ap.add_argument("--interactive", "-i", action="store_true", help="keep asking questions at a prompt")
    ap.add_argument("--reread", action="store_true", help="read the sheet again even if it was read before")
    args = ap.parse_args()

    name = Path(args.image).stem
    out = Path(args.out or f"{name}_read")
    saved = out / f"{name}_read.json"
    session = Session(args)
    if saved.exists() and not args.reread:
        result = json.loads(saved.read_text(encoding="utf-8"))
        print(f"Using the earlier reading of this sheet: {saved}  (--reread to read it again)")
    else:
        result = session.read()
        save(result, out, name)
        print(f"Saved to {out.resolve()}")
        result.pop("overlay", None)

    qa = SheetQA(result, session.band_reader(result))
    if not args.question and not args.interactive:
        print(qa.summary())
        for f in result["findings"]:
            print(f"[{f['severity']}] {f['message']}")
        print('\nAsk about it:  python read_sheet.py --image ' + args.image + ' -q "ground level at bridge 560"   (or -i)')
    for q in args.question:
        print(f"\nQ: {q}")
        print(qa.answer(q))
    if args.interactive:
        print("\nAsk about this sheet (help for examples, Enter on an empty line to stop).")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            print(qa.answer(q))


if __name__ == "__main__":
    main()
