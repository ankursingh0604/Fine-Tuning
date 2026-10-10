"""Ask the L-section v5 model about a Plan & L-section PDF (any number of sheets).

    python ask_lsec.py --adapter lsec_run/adapter --pdf "../MKN_PnP_1061-1098_4th Line.pdf" -i
    python ask_lsec.py --adapter lsec_run/adapter --pdf 6.pdf -q "degree of curve 207"
    options:  --show-facts   also print the facts the model was given
              --4bit         about 4 GB of GPU memory instead of 10 (when another program shares the GPU)
              --offline      no web search: a term nobody knows stays "meaning unknown"

How it answers: every sheet is read once from the PDF's text layer (scripts/annotate_v5.py: bridges, levels, the
DETAILS OF BRIDGES table, curves, bands, gradients, stations, TBMs, notes, title, and every other printed text; saved
next to the PDF as <name>.lsec.json). For a question the sheet it is about is picked (the bridge / curve / chainage it
names, else the sheet whose facts fit best), its relevant facts go to the model - with the band values at a chainage
the question names (printed, or between two printed columns) and the meaning of a term the sheet does not explain
(the glossary, else a web search for the term alone) - and, when the question is about one spot (a bridge, a curve, a
text), a crop of that spot.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "scripts"), str(ROOT / "runpod"), str(ROOT / "gad_tools")]
import lsec_facts as LF        # noqa: E402

CONTEXT = "railway L-section drawing"


def read_pdf(pdf):
    """Every sheet of the PDF, annotated as for training (cached next to it)."""
    cache = Path(str(pdf) + ".lsec.json")
    if cache.exists() and cache.stat().st_mtime >= Path(pdf).stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    import pymupdf
    import annotate as A
    import annotate_v4 as V4
    import annotate_v5 as V5
    print(f"reading {Path(pdf).name} ...", flush=True)
    doc, out = pymupdf.open(pdf), []
    for i, ann in enumerate(A.annotate_pdf(pdf)):
        page = doc[i]
        if page.rotation:
            page.remove_rotation()
        ann["sheet_id"] = f"page {i + 1}"
        ann["findings"] = V4.findings(ann)
        try:
            ann["v5"] = V5.v5_reading(page)
        except Exception as e:                       # noqa: BLE001
            ann["v5"] = {"error": str(e)}
        names = V5.fill_panel(ann, A.page_lines(page))
        hide = V5.private_texts(ann)
        ann["all_text"] = [t for t in ann["all_text"] if not hide(t) and (t["text"], tuple(round(v, 1) for v in t["bbox"])) not in names]
        ann.pop("officers", None)
        out.append(ann)
    try:
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return out


def chainages(q):
    """Chainages a question names, in metres ('985+506', 'CH 985506.5')."""
    out = [int(a) * 1000 + float(b) for a, b in re.findall(r"(\d{2,4})\s*\+\s*(\d{1,3}(?:\.\d+)?)", q)]
    out += [float(x) for x in re.findall(r"(?<![\d+.])(\d{5,7}(?:\.\d+)?)(?![\d.])", q)]
    return out


def pick(anns, q):
    """(the sheet a question is about, must refs, spot box in PDF points or None)."""
    nums = [re.sub(r"\s", "", n).upper() for n in re.findall(r"\b(?:br(?:idge)?\.?\s*(?:no\.?)?\s*)(\d+[a-z]{0,3}(?:\s*(?:up|dn))?)", q, re.I)]
    for a in anns:                                          # a bridge it names
        for b in (a.get("v5") or {}).get("bridges", []):
            if any(b["num"] == n or re.sub(r"(UP|DN)$", "", b["num"]) == re.sub(r"(UP|DN)$", "", n) for n in nums):
                box = b["callouts"][0]["bbox"] if b["callouts"] else (b.get("level_block") or {}).get("bbox")
                return a, [("bridge", b["num"]), ("levels", b["num"]), ("table", re.sub(r"(UP|DN)$", "", b["num"])), ("table", b["num"])], box
    cv = re.search(r"\b(?:curve|c\.?\s*no\.?)\s*[-.:]?\s*(\d+)", q, re.I)
    if cv:
        for a in anns:
            for c in (a.get("v5") or {}).get("curves", []):
                if str(c["num"]) == cv.group(1):
                    return a, [("curve", c["num"], c.get("track"))], c.get("bbox")
    for ch in chainages(q):                                  # a chainage: the sheet whose band holds it
        for a in anns:
            cols = (a.get("bands") or {}).get("columns") if isinstance(a.get("bands"), dict) else None
            if cols and min(c["chainage"] for c in cols) <= ch <= max(c["chainage"] for c in cols):
                return a, [], None
    best = max(anns, key=lambda a: max(LF.score_facts(LF.all_facts(a), q) or [0]))
    # a printed text the question quotes: its spot
    qt = set(LF.tokens(q))
    hit = max(((len(qt & set(LF.tokens(t["text"]))), t) for t in best.get("all_text") or []), key=lambda x: x[0], default=(0, None))
    box = hit[1]["bbox"] if hit[1] is not None and hit[0] >= 2 else None
    return best, [], box


def extras(ann, q, web):
    """Fact lines added for this question: band values at its chainages, the meaning of terms the sheet does not explain."""
    import glossary as G
    out = []
    for ch in chainages(q):
        out += [f"Band value at chainage {LF.km(ch)}: {n} = {v:g} ({how})" for _, n, v, how in LF.band_at(ann, ch)]
    known = {k.upper() for k in (ann.get("abbreviations") or {})}
    for term in dict.fromkeys(re.findall(r"\b[A-Z][A-Z.]{1,7}\b", q)):
        k = G.key(term)
        if len(k) < 2 or k in known or k in ("BR", "CH", "NO", "FL", "RL", "UP", "DN", "LC", "KM"):
            continue
        m, src = G.meaning(term, web=web, context=CONTEXT)
        if src == "web" and m:
            out.append(f"Web search for '{term}' (not confirmed by the drawing): {m[:300]}")
        elif src == "unknown":
            out.append(f"'{term}': no meaning found (not on the sheet, not in the glossary, no web result)")
    return out


def spot(pdf, ann, box):
    """A 768 x 768 crop around the spot, as the training's spot crops (the sheet rendered as Railsight renders it)."""
    import find_crop as F
    s = F.open_sheet(Path(pdf), HERE / "_ask_sheets" / ann["sheet_id"].replace(" ", "_"), ann["page_index"])
    z = s.dpi / 72
    W, H = s.image.size
    cx, cy = (box[0] + box[2]) / 2 * z, (box[1] + box[3]) / 2 * z
    half = 384 * s.dpi / 150
    return s.image.crop((int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half)))).resize((768, 768))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("-q", "--question", action="append")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--show-facts", action="store_true")
    ap.add_argument("--export", metavar="DIR", help="write the Bridge List, Gradient, Prop. Curve List and Station CSVs (Sample "
                                                   "template layouts) to DIR and stop - no model needed")
    args = ap.parse_args()
    anns = read_pdf(args.pdf)
    print(f"{Path(args.pdf).name}: {len(anns)} sheet(s), {sum(len((a.get('v5') or {}).get('bridges', [])) for a in anns)} bridges")
    if args.export:
        import lsec_exports
        files, findings = lsec_exports.write_all(anns, args.export, Path(args.pdf).stem)
        for k, (p, n) in files.items():
            print(f"  {k}: {n} rows -> {p}")
        for f in findings:
            print("  CHECK: " + f)
        return
    from model_v5 import Model
    model = Model(args.adapter, args.four_bit)

    def ask(q):
        ann, must, box = pick(anns, q)
        prompt = LF.prompt(ann, q, must, extras(ann, q, not args.offline))
        imgs = [spot(args.pdf, ann, box)] if box else []
        if args.show_facts:
            print("\n" + prompt + "\n")
        print(f"[{ann['sheet_id']}] " + model.ask(imgs, prompt) + "\n", flush=True)

    for q in args.question or []:
        ask(q)
    if args.interactive or not args.question:
        print("Ask about the drawing (empty line to stop).")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            ask(q)


if __name__ == "__main__":
    main()
