"""Ask questions about a GAD (vector PDF) with the fine-tuned model.

    python ask_gad.py --adapter gad_run/adapter --pdf "GAD.pdf" -q "What is the proposed soffit level?"
    python ask_gad.py --adapter gad_run/adapter --pdf "GAD.pdf" -i          # ask several questions
    options:  --show-facts   also print the facts the model was given
              --base         use the untouched base model instead of the adapter (for comparison)

How it answers: the drawing is read exactly from the PDF's text layer (views, levels, tables, notes, title block,
about 10 s, saved next to the PDF as <name>.gad.json); for each question the relevant facts go to the model, which
explains them (value, label as printed, where it is, what it means). A question about one view ("what does section
B-B show?") also gets that view's picture.
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


def view_image(pdf, view):
    import pymupdf
    page = pymupdf.open(pdf)[0]
    if page.rotation:
        page.remove_rotation()
    r = pymupdf.Rect(view["bbox"]) & page.rect
    z = min(1024, max(512, int(max(r.width, r.height) * 0.6))) / max(r.width, r.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=r)
    from PIL import Image
    return DG.pad32(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))


def build(ann, pdf, question):
    v = view_in(question, ann) if VIEW_Q.search(question) else None
    if v:
        refs = [("view", v["title"])] + [("level", l["label"], l["value"]) for l in ann["levels"] if l["view"] == v["title"]][:12] \
            + [("dim", d["label"], d["value"]) for d in ann["labelled_dims"] if d["view"] == v["title"]][:6]
        text = FX.prompt(ann, question, must=refs)
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}], [view_image(pdf, v)], text
    text = FX.prompt(ann, question)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}], [], text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("-q", "--question")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--show-facts", action="store_true")
    args = ap.parse_args()
    if not (args.adapter or args.base):
        sys.exit("give --adapter (the trained weights) or --base")
    ann = read_gad(args.pdf)
    b = ann.get("bridge", {})
    print(f"GAD {ann['gad_id']}: {b.get('category', '')} {b.get('bridge_no', '')} {b.get('description', '')}; "
          f"{len(ann['views'])} views, {len(ann['levels'])} levels, {len(ann['notes'].get('notes', []))} notes"
          + (f"; {len(ann['findings'])} disagreement(s) on the drawing" if ann["findings"] else ""))
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(args.model if args.base else args.adapter, load_in_4bit=False)
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
