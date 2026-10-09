"""Ask questions about a GAD with the fine-tuned model: a vector PDF, a scanned PDF, or a PNG / JPG photo of the sheet.

    python ask_gad.py --adapter gad_run/adapter --pdf "GAD.pdf" -q "What is the proposed soffit level?"
    python ask_gad.py --adapter gad_run/adapter --pdf "GAD.pdf" -i          # ask several questions
    python ask_gad.py --adapter gad_run/adapter --pdf "scan.png" -i         # a scan / photo (text read by OCR)
    options:  --show-facts   also print the facts the model was given
              --base         use the untouched base model instead of the adapter (for comparison)
              --4bit         about 4 GB of GPU memory instead of 10 (when another program shares the GPU)
              --offline      no web search: a label nobody knows stays "meaning unknown"
    A label not in the glossary (gad_tools/glossary.json + the built-in terms) is searched on the web - the term only,
    nothing from the drawing - and the model says the meaning is from the web, not confirmed by the drawing.

How it answers: the drawing is read from the PDF's text layer - or, for a scan / photo, with the local OCR (needs
rapidocr_onnxruntime; a minute or two) - (views, levels, tables, notes, title block, about 10 s, saved next to the
file as <name>.gad.json); for each question the relevant facts go to the model, which
explains them (value, label as printed, where it is, what it means). A question about one spot - a figure ("what is
800?"), a level, any text written on the views, a dismantle note - also gets a full-detail crop of that spot, so the
model looks at the drawing there (arrows, circles, colours, what is next to it); a question about one view ("what does
section B-B show?") gets that view's picture.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gad_tools")]
import annotate_gad as A      # noqa: E402
import data_gad as DG         # noqa: E402
import eval_gad as E          # noqa: E402
import facts as FX            # noqa: E402

VIEW_Q = re.compile(r"\b(show|shows|represent|represents|explain|diagram|view|drawn|depict|look like)\b", re.I)


def read_gad(pdf):
    cache = Path(str(pdf) + ".gad.json")
    if cache.exists() and cache.stat().st_mtime >= Path(pdf).stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    print(f"reading {Path(pdf).name} ...", flush=True)
    ann = A.annotate(pdf)
    try:
        cache.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return ann


def view_in(question, ann):
    q = set(FX.tokens(question))
    best = None
    for v in ann["views"]:
        t = set(FX.tokens(v["title"])) - {"section", "details", "detail", "typical", "plan", "half", "of"}
        if t and t <= q and (best is None or len(t) > best[0]):
            best = (len(t), v)
    return best[1] if best else None


IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def view_image(pdf, view, ann=None):
    if Path(pdf).suffix.lower() in IMAGE_TYPES:            # a photo / scan of the sheet: cut the view out of the image
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(pdf).convert("RGB")
        k = img.width / ann["page_size"][0]                # points -> pixels
        x0, y0, x1, y1 = (max(0, v * k) for v in view["bbox"])
        crop = img.crop((int(x0), int(y0), int(min(img.width, x1)), int(min(img.height, y1))))
        z = min(1024, max(512, int(max(crop.size) * 0.6 / k))) / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * z)), max(1, int(crop.height * z))))
        return DG.pad32(crop)
    import pymupdf
    page = pymupdf.open(pdf)[0]
    if page.rotation:
        page.remove_rotation()
    r = pymupdf.Rect(view["bbox"]) & page.rect
    z = min(1024, max(512, int(max(r.width, r.height) * 0.6))) / max(r.width, r.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=r)
    from PIL import Image
    return DG.pad32(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))


SPOT_DPI, SPOT_PX = 150, 768          # the crop the model was trained with (build_dataset_gad.img_spot_rows)


def spot_of(question, ann):
    """What on the sheet the question is about: (bbox, refs) of a figure / level / text written on a view / dismantle
    note it names - in the view it names, when it names one - or None."""
    view = view_in(question, ann)
    in_view = lambda x: view is None or x.get("view") == view["title"]
    # 1. a text written on the views that the question quotes (most of its words): "300 THK. STONE PITCHING ..."
    q = set(FX.tokens(question)) - {"view", "plan", "section", "half", "bottom", "top", "elevation", "sectional", "what", "why"}
    best = None
    for v in ann.get("views", []):
        for c in v.get("callouts", []):
            if not isinstance(c, dict) or not c.get("bbox"):
                continue
            t = set(FX.tokens(c["text"]))
            hit = len(q & t)
            if hit >= 2 and hit >= 0.6 * len(t):
                key = (hit, view is not None and v is view)
                if best is None or key > best[0]:
                    best = (key, c)
    if best:
        return best[1]["bbox"], [("call", best[1]["text"])]
    if re.search(r"dismantl", question, re.I) and ann.get("dismantle"):
        d = next((d for d in ann["dismantle"] if in_view(d)), ann["dismantle"][0])
        return d["bbox"], [("call", d["text"])]
    # 2. a figure or a level the question names by its value
    for n in re.findall(r"\d+(?:\.\d+)?", question):
        x = float(n)
        for d in sorted(ann.get("labelled_dims", []), key=lambda d: not in_view(d)):
            if abs(d["value"] - x) < 1e-6 and in_view(d):
                return d["bbox"], [("dim", d["label"], d["value"])]
        for d in sorted(ann.get("dims", []), key=lambda d: not in_view(d)):
            if abs(d["value"] - x) < 1e-6:
                return d["bbox"], [("pdim", d["view"], d["value"])]
        if "." in n:
            for l in sorted(ann.get("levels", []), key=lambda l: not in_view(l)):
                if abs(l["value"] - x) < 0.0005 and l.get("bbox"):
                    return l["bbox"], [("level", l["label"], l["value"])]
    return None


def spot_image(pdf, ann, bbox):
    """A 768 x 768 crop at 150 dpi centred on bbox (points), from the PDF or from the scan / photo."""
    from PIL import Image
    half = SPOT_PX * 72 / SPOT_DPI / 2
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    clip = (cx - half, cy - half, cx + half, cy + half)
    if Path(pdf).suffix.lower() in IMAGE_TYPES:
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(pdf).convert("RGB")
        k = img.width / ann["page_size"][0]
        crop = img.crop(tuple(int(max(0, c * k)) for c in clip)).resize((SPOT_PX, SPOT_PX))
        return DG.pad32(crop)
    import pymupdf
    page = pymupdf.open(pdf)[0]
    if page.rotation:
        page.remove_rotation()
    r = pymupdf.Rect(clip) & page.rect
    z = SPOT_DPI / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=r)
    return DG.pad32(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))


def build(ann, pdf, question):
    # a question about one spot (a figure, a level, a label, a dismantle note): that spot's crop + the facts
    sp = spot_of(question, ann)
    if sp:
        bbox, refs = sp
        text = FX.prompt(ann, question, must=refs)
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}], [spot_image(pdf, ann, bbox)], text
    v = view_in(question, ann) if VIEW_Q.search(question) else None
    if v:
        refs = [("view", v["title"])] + [("level", l["label"], l["value"]) for l in ann["levels"] if l["view"] == v["title"]][:12] \
            + [("dim", d["label"], d["value"]) for d in ann["labelled_dims"] if d["view"] == v["title"]][:6]
        text = FX.prompt(ann, question, must=refs)
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}], [view_image(pdf, v, ann)], text
    text = FX.prompt(ann, question)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}], [], text


def look_up_terms(ann, web=True, log=print):
    """Labels nobody has a meaning for: searched on the web once (the term only - nothing from the drawing), cached in
    gad_tools/glossary.json for an engineer to confirm. The results go to the model as 'found on the web, not confirmed'."""
    import glossary
    ann["_meanings"] = ann.get("_meanings") or {}
    asked = []
    for o in ann.get("other_values", []):
        k = glossary.key(o["label"])
        if k in ann["_meanings"]:
            continue
        m, src = glossary.meaning(o["label"])
        if src == "unknown" and web:
            m, src = glossary.meaning(o["label"], web=True, context="railway bridge drawing")
            asked.append(f"{o['label']} ({'found' if m else 'nothing found'})")
        ann["_meanings"][k] = (m, src)
    if asked:
        log("terms not in the glossary, looked up on the web (term only): " + ", ".join(asked)
            + " - see gad_tools/glossary.json to confirm or correct them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--pdf", required=True, help="the GAD: a vector PDF, a scanned PDF, or a PNG / JPG of the sheet")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="load the model in 4-bit (about 4 GB instead of 10): when the GPU is shared with another program")
    ap.add_argument("-q", "--question")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--show-facts", action="store_true")
    ap.add_argument("--offline", action="store_true", help="no web search for unknown terms")
    args = ap.parse_args()
    if not (args.adapter or args.base):
        sys.exit("give --adapter (the trained weights) or --base")
    ann = read_gad(args.pdf)
    b = ann.get("bridge", {})
    print(f"GAD {ann['gad_id']}: {b.get('category', '')} {b.get('bridge_no', '')} {b.get('description', '')}; "
          f"{len(ann['views'])} views, {len(ann['levels'])} levels, {len(ann['notes'].get('notes', []))} notes"
          + (f"; {len(ann['findings'])} disagreement(s) on the drawing" if ann["findings"] else ""))
    look_up_terms(ann, web=not args.offline)
    if ann.get("text_source") == "ocr":
        print("NOTE: this sheet has no text layer - its text was read with OCR, so a value can be misread. Check any value "
              "you use against the drawing (--show-facts shows what was read).")
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(args.model if args.base else args.adapter, load_in_4bit=args.four_bit)
    FastVisionModel.for_inference(model)

    def ask(q):
        msgs, imgs, text = build(ann, args.pdf, q)
        if args.show_facts:
            print("\n" + text.split("\n\nQuestion:")[0] + "\n")
        print(E.generate(model, processor, msgs, imgs) + "\n", flush=True)

    if args.question:
        ask(args.question)
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
