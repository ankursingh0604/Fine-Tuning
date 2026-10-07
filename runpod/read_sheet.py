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

A PDF works too (--image sheet.pdf, --page N for multi-page files): it is rendered at the resolution where its text is
as tall as on the trained sheets, whatever the page size.

Anything else printed on the sheet (find_crop.py): a question the list above does not cover, and every gradient /
grade-point question ("gradient at CH 10046", "slope at 9390"), is answered by finding that text on the sheet, cutting
a crop there and asking the model. Grade points: the gradient on each side of the stem and the FL, each read from its
own crop with all other text blanked (so a neighbour's FL is never taken); "not printed" when there is none. For PDFs
every reading is compared with the PDF's own text. The crops are saved in <out>/crops/.
    python read_sheet.py --adapter adapter.zip --image sheet.pdf --find-only -q "gradient at CH 10046"
--find-only skips the whole-sheet reading (minutes) when only such questions are wanted.

Data-band values on a PDF of any drawing set (band_table.py): "cut/fill at C-8 TPCC2", "ground level at BR NO. 15",
"FL at 11540" - the band's rows are found from their printed headings, the model reads the column(s) at that chainage
and the value is interpolated between the printed columns.
"""
import argparse
import csv
import json
import re
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

    def ask(self, imgs, q):
        """The model's answer for crops (used by find_crop.py); the model is loaded on first use."""
        from app import prepare
        self.reader()
        return self.model.ask([prepare(im) for im in imgs], q)

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
    ap.add_argument("--page", type=int, default=1, help="page of a PDF (default 1)")
    ap.add_argument("--find-only", action="store_true",
                    help="skip the whole-sheet reading; answer by finding the text and asking about a crop (fast)")
    args = ap.parse_args()

    name = Path(args.image).stem
    out = Path(args.out or f"{name}_read")
    sheet = None
    is_pdf = Path(args.image).suffix.lower() == ".pdf"
    if is_pdf:
        import find_crop
        sheet = find_crop.open_sheet(args.image, out, args.page - 1)
        print(sheet.note)
        args.image = str(sheet.path)               # the rest reads the rendered page, as for any image
    saved = out / f"{name}_read.json"
    session = Session(args)
    if args.find_only:
        result = {"bridges": [], "findings": [], "title": {}}
    elif saved.exists() and not args.reread:
        result = json.loads(saved.read_text(encoding="utf-8"))
        print(f"Using the earlier reading of this sheet: {saved}  (--reread to read it again)")
    else:
        result = session.read()
        save(result, out, name)
        print(f"Saved to {out.resolve()}")
        result.pop("overlay", None)

    qa = SheetQA(result, session.band_reader(result))
    finder = None

    def get_finder():
        nonlocal finder, sheet
        import find_crop
        if finder is None:                           # built only when needed (for an image: one OCR pass)
            sheet = sheet or find_crop.open_sheet(args.image, out)
            finder = find_crop.Finder(sheet, session.ask, out)
        return finder

    def answer(q):
        """The fixed question list first; gradients, grade points and anything it does not know: find, crop, ask.
        For a PDF, data-band values too: its band is found from the PDF's own layout, whatever the drawing set."""
        import find_crop
        import band_table
        if args.find_only or find_crop.Finder.handles(q):
            return get_finder().answer(q)
        if is_pdf and (re.search(r"\bcurves?\b|\b(list|all|how many|which)\b.*\bbridges?\b", q, re.I)
                       or find_crop.CURVE_ID.search(q) and not band_table.asks_row(q)):
            a = get_finder().object_answer(q)                 # bridges by type / status, curves: read from the callouts
            if a:
                return a
        if is_pdf and band_table.asks_row(q) and get_finder().bands():
            a = get_finder().band_answer(q, on_sheet_only=True)     # "... at 560" may mean bridge 560: left to the list
            if a:
                return a
        a = qa.answer(q)
        if "I did not understand that question" in a or "I could not find that bridge" in a:
            return get_finder().answer(q)
        return a

    if not args.question and not args.interactive and not args.find_only:
        print(qa.summary())
        for f in result["findings"]:
            print(f"[{f['severity']}] {f['message']}")
        print('\nAsk about it:  python read_sheet.py --image ' + args.image + ' -q "ground level at bridge 560"   (or -i)')
    for q in args.question:
        print(f"\nQ: {q}")
        print(answer(q))
    if args.interactive:
        print("\nAsk about this sheet (help for examples, Enter on an empty line to stop).")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            print(answer(q))


if __name__ == "__main__":
    main()
