"""Ask the fine-tuned railway-drawing model questions in a browser.

    python app.py --adapter /workspace/adapter.zip                 # RunPod, full precision
    python app.py --adapter results/adapter.zip --4bit --port 7860   # small GPU (e.g. 4 GB laptop)

Tab "Ask about a PDF": upload a drawing PDF (one page or many) and ask about anything printed on it - bridges, curves,
grade points, band values at any chainage. Each answer names its page and shows the crops the model read (find_crop.py,
pdf_book.py - the same code as read_sheet.py). A UI of its own can call open_pdf() and ask_pdf() below directly.

Tab "Read a whole sheet": upload an image of a whole sheet (50-200 dpi) and the model reads every bridge,
its levels and band values, the title block and TBMs (sheet_reader.py), with a CSV to download.

Tab "Ask about a crop": upload a crop or screenshot of a sheet (or pick an example), type a question, and get the fine-tuned
model's answer. Tick "Also ask the untouched model" to see the base model's answer next to it. Boxes in
an answer ({"bbox_2d": [...]}) are drawn on the image.

Crops work best at the scale the model was trained on: sheets rendered at 150 dpi, where callout text is
about 20 px tall, and no more than about 1400 x 900 px.
"""
import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

MAX_PIXELS = 1400 * 896            # the largest images the model was trained on
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- image and answer helpers

def prepare(img):
    """Scale down to MAX_PIXELS if needed and pad each side to a multiple of 28 (the model's patch
    grid), so the model sees exactly the image that boxes in its answer refer to."""
    img = img.convert("RGB")
    up28 = lambda v: -(-v // 28) * 28
    w, h = img.size
    s = 1.0
    while up28(int(w * s)) * up28(int(h * s)) > MAX_PIXELS:   # the padded size must also fit
        s *= 0.98
    if s < 1:
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        w, h = img.size
    W, H = up28(w), up28(h)
    if (W, H) != (w, h):
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(img, (0, 0))
        img = canvas
    return img


def boxes(text):
    """[(box, label)] from a ```json answer with bbox_2d entries."""
    m = re.search(r"```json\s*(.*?)```", text, re.S) or re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    items = obj if isinstance(obj, list) else [obj]
    return [(i["bbox_2d"], str(i.get("label", ""))) for i in items
            if isinstance(i, dict) and isinstance(i.get("bbox_2d"), list) and len(i["bbox_2d"]) == 4]


def draw(img, found, color):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for (x0, y0, x1, y1), label in found:
        d.rectangle([x0, y0, x1, y1], outline=color, width=4)
        if label:
            d.text((x0 + 4, max(y0 - 14, 0)), label, fill=color)
    return out


# ---------------------------------------------------------------- model

class Model:
    def __init__(self, adapter, four_bit):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        adapter = Path(adapter)
        if adapter.suffix == ".zip":                       # adapter.zip as downloaded from the training pod
            target = Path(tempfile.gettempdir()) / f"railway_adapter_{adapter.stat().st_size}"
            if not (target / "adapter_config.json").exists():
                zipfile.ZipFile(adapter).extractall(target)
            adapter = target
        base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_quant_type="nf4") if four_bit else None
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        print(f"loading {base} {'(4-bit)' if four_bit else ''} + adapter {adapter}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=dtype, quantization_config=quant, device_map="auto")
        self.model = PeftModel.from_pretrained(model, str(adapter)).eval()
        # An older transformers names the layers differently, and the adapter would then attach to nothing,
        # silently leaving the untouched model. Refuse to run in that case.
        n_lora = sum(1 for name, _ in self.model.named_modules() if name.endswith("lora_A"))
        if n_lora == 0:
            raise RuntimeError("the adapter did not attach to any layer: use transformers >= 4.52 (see requirements.txt)")
        print(f"adapter attached to {n_lora} layers")
        self.processor = AutoProcessor.from_pretrained(str(adapter), min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)

    def ask(self, imgs, question, use_adapter=True, max_new_tokens=800):
        imgs = imgs if isinstance(imgs, list) else [imgs]
        messages = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            if use_adapter:
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            else:
                with self.model.disable_adapter():
                    out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- asking about a PDF (no gradio needed: a UI calls these)

def open_pdf(path, ask, log=print):
    """A drawing PDF - one page or many - ready for questions (pdf_book.Book: every page indexed from its text; pages
    rendered only when an answer needs them). ask(images, question) -> the model's answer, e.g. model_ask(Model(...))."""
    import pdf_book
    out = Path(tempfile.mkdtemp(prefix="railway_pdf_"))
    return pdf_book.Book(path, out, ask, log=log)


def model_ask(model):
    """The ask() a Book needs, from a loaded Model: each crop sized for the model as in training."""
    return lambda imgs, q: model.ask([prepare(im) for im in imgs], q)


def ask_pdf(book, question):
    """(answer text, [(crop image, caption)]): the answer, and the crops the model read for it, in order."""
    import time
    import pdf_book
    q = (question or "").strip()
    if book is None:
        return "Upload a PDF first.", []
    if not q:
        return "Type a question (help for examples).", []
    if q.lower() in ("help", "?"):
        return pdf_book.HELP, []
    marks, t = book.crop_marks(), time.time()
    a = book.answer(q)
    crops = book.crops_since(marks)
    shown = [(Image.open(p).copy(), f"{p.parent.parent.name} {p.stem}") for p in crops]
    return a + f"\n\n({len(crops)} crop(s) read by the model, {time.time() - t:.0f} s)", shown


# ---------------------------------------------------------------- app

def examples():
    """Example crops shipped next to this file (examples/examples.json), if present."""
    f = HERE / "examples" / "examples.json"
    if not f.exists():
        return []
    return [[str(HERE / "examples" / e["image"]), str(HERE / "examples" / e["image2"]) if e.get("image2") else None, e["question"]]
            for e in json.loads(f.read_text(encoding="utf-8"))]


def build_ui(model):
    import gradio as gr

    def run(image, image2, question, compare):
        if image is None or not question.strip():
            return "Upload an image and type a question.", "", None
        img = prepare(image)
        imgs = [img] + ([prepare(image2)] if image2 is not None else [])   # e.g. bridge callout + data-band crop
        ft = model.ask(imgs, question)
        base = model.ask(imgs, question, use_adapter=False) if compare else ""
        shown = draw(img, boxes(ft), (210, 30, 30))
        if compare:
            shown = draw(shown, boxes(base), (120, 120, 120))
        return ft, base, shown

    def read_whole(path, progress=gr.Progress()):
        """A whole sheet image -> every bridge with its levels and band values, title block, TBMs, checks."""
        if not path:
            return "Upload a sheet image first.", [], None, None
        from read_sheet import save
        from sheet_reader import Reader
        Image.MAX_IMAGE_PIXELS = None            # whole A0 sheets at 200 dpi are ~70 megapixels
        steps = []

        def log(msg):
            steps.append(msg)
            progress(0, desc=msg.strip())
        name = Path(path).stem
        result = Reader(model, log).read(Image.open(path))
        out = save(result, Path(tempfile.mkdtemp()) / f"{name}_read", name)
        t = result["title"] or {}
        bridges = result["bridges"]
        lines = [f"**Sheet {t.get('sheet_no', '?')}** {t.get('title', '')} {t.get('chainage_from', '')} - {t.get('chainage_to', '')} "
                 f"(image about {result['dpi']} dpi). Bridges found: {len(bridges)}, read: {sum(1 for b in bridges if b['data'])}, "
                 f"with band values: {sum(1 for b in bridges if b['band'])}. TBMs: {len(result['tbm'] or [])}."]
        lay = result.get("layout")
        if lay:
            lines.append(f"Layout: **{lay['status']}** (confidence {lay['confidence']})")
        lines += [f"- **WARNING**: {w}" for w in result["warnings"]]
        lines += [f"- **{f['severity']}**: {f['message']}" for f in result["findings"]]
        rows = []
        for b in bridges:
            d = b["data"] or {}
            lv = d.get("levels") or {}
            from sheet_reader import band_values
            col = band_values(b["band"])
            rows.append([b["bridge_id"], d.get("chainage_m"), " ".join(str(v) for v in (d.get("existing_type"), d.get("existing_span")) if v),
                         d.get("crossing"), d.get("proposal"), lv.get("proposed_formation_level"),
                         lv.get("min_formation_level_required"), lv.get("high_flood_level"),
                         (f"{(b['band'] or {}).get('x1')} - {(b['band'] or {}).get('x2')}" if col else None),
                         col.get("ground_level"), col.get("cut_fill"), ", ".join(c["severity"] for c in b["checks"]) or "ok"])
        return "\n".join(lines), rows, result["overlay"], sorted(str(f) for f in out.iterdir())

    def load_pdf(path, progress=gr.Progress()):
        if not path:
            return None, "Upload a drawing PDF."
        book = open_pdf(path, model_ask(model), log=lambda m: progress(0, desc=m))
        return book, ("```\n" + book.note + "\n```\nAsk anything printed on it below (type **help** for examples). "
                      "Every answer names its page and shows the crops the model read.")

    with gr.Blocks(title="Railway drawing reader") as ui:
        gr.Markdown("## Railway drawing reader")
        with gr.Tab("Ask about a PDF"):
            gr.Markdown("Upload a Plan & L-Section **PDF** (one page or many, any drawing set) and ask about anything printed on "
                        "it: bridges (list, by type, chainage, span, levels), curves (and which bridge is on them), grade "
                        "points, data-band values at any chainage (interpolated), anything else printed. The right page and "
                        "label are found from the PDF's text, cut out, and read by the model; every reading is compared "
                        "with the PDF text (\"matches the PDF text\" or CHECK). Nothing is guessed.")
            pdf_in = gr.File(label="Drawing PDF", file_types=[".pdf"], type="filepath")
            pdf_info = gr.Markdown()
            book_state = gr.State(None)
            with gr.Row():
                pdf_q = gr.Textbox(label="Question", scale=5,
                                   placeholder="e.g.  chainage of existing Br. No. 16   |   list all RCC bridges   |   FL at 11275")
                pdf_go = gr.Button("Ask", variant="primary", scale=1)
            pdf_answer = gr.Textbox(label="Answer", lines=18)
            pdf_crops = gr.Gallery(label="Crops the model read for this answer", columns=4, height="auto")
            pdf_in.upload(load_pdf, [pdf_in], [book_state, pdf_info])
            pdf_go.click(ask_pdf, [book_state, pdf_q], [pdf_answer, pdf_crops])
            pdf_q.submit(ask_pdf, [book_state, pdf_q], [pdf_answer, pdf_crops])
        with gr.Tab("Read a whole sheet"):
            gr.Markdown("Upload an image of a whole Plan & L-Section sheet (50-200 dpi; 150-200 dpi reads most reliably). "
                        "The model reads it in zoomed-in pieces: every bridge callout, its level block, the data-band values at "
                        "the bridge chainage (interpolated between the two printed columns either side), the title block and the TBM table. "
                        "Expect several minutes per sheet.")
            sheet_in = gr.File(label="Sheet image (PNG / JPG)", file_types=["image"], type="filepath")
            read_btn = gr.Button("Read the sheet", variant="primary")
            sheet_summary = gr.Markdown()
            sheet_table = gr.Dataframe(headers=["Bridge", "Chainage", "Existing", "Crossing", "Proposal", "FL", "MIN FL REQ.", "HFL",
                                                "Band columns (x1 - x2)", "Ground level (y)", "Cut(-)/Fill(+) (y)", "Checks"],
                                       label="Bridges read from the sheet", wrap=True, interactive=False)
            sheet_overlay = gr.Image(label="Bridges found (red = flagged or failed a check)", type="pil")
            sheet_files = gr.File(label="Results (CSV, JSON, overlay)", file_count="multiple")
            read_btn.click(read_whole, [sheet_in], [sheet_summary, sheet_table, sheet_overlay, sheet_files])
        with gr.Tab("Ask about a crop"):
            gr.Markdown("Upload a crop of a sheet, then ask a question about it.")
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Drawing crop")
                    image2 = gr.Image(type="pil", label="Second crop (optional, e.g. the data bands for a bridge question)")
                    question = gr.Textbox(label="Question", lines=2,
                                          placeholder="e.g. Read the callout for bridge 560 and return its details as JSON.")
                    compare = gr.Checkbox(label="Also ask the untouched model (slower)", value=False)
                    go = gr.Button("Ask", variant="primary")
                    if examples():
                        gr.Examples(examples(), inputs=[image, image2, question], label="Examples from the drawings")
                with gr.Column():
                    ft_out = gr.Textbox(label="Fine-tuned model (reasoning steps, then the answer)", lines=16)
                    base_out = gr.Textbox(label="Untouched model", lines=8)
                    boxed = gr.Image(label="Boxes from the answers (red: fine-tuned, grey: untouched)")
            go.click(run, [image, image2, question, compare], [ft_out, base_out, boxed])
            question.submit(run, [image, image2, question, compare], [ft_out, base_out, boxed])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/workspace/adapter.zip", help="adapter.zip or an unzipped adapter folder")
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="load the base model in 4-bit (small GPUs)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="also create a temporary public gradio.live link (anyone with the link can use it)")
    args = ap.parse_args()
    build_ui(Model(args.adapter, args.four_bit)).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
