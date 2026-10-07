"""A chat page for the drawings, like ChatGPT / Claude / Gemini: one chat box, "+" to upload a PDF or a sheet image,
the uploaded document shown beside the chat so every answer can be compared with the drawing.

    python chat_ui.py --adapter ../results/v3/adapter.zip            # then open http://localhost:7860
    python chat_ui.py --adapter ../results/v3/adapter.zip --4bit     # small GPU
    python chat_ui.py ... --host 0.0.0.0 --port 7860                 # reachable from other machines on the network

The answers come from the same code as read_sheet.py: every page is indexed from its text (pdf_book.py), each question
goes to the page(s) that print what it asks about, the label is cut out and read by the model, and every reading is
compared with the PDF text. The page (chat_ui/index.html) talks to this server:

    POST /api/upload            a PDF or image (multipart "file")  -> {doc, name, pages, note, suggestions}
    GET  /api/doc/{doc}/page/N  page N as a PNG for the viewer
    POST /api/ask               {"doc": ..., "question": ...}      -> {answer, pages, crops: [{url, caption}], seconds}
    GET  /api/doc/{doc}/file    a crop the model read (?path=...)
"""
import argparse
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEW_WIDTH = 6000              # px: a whole sheet in the viewer, sharp enough to read its labels when zoomed in
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


KINDS = [("brlv", "level block"), ("br", "callout"), ("cvd", "curve details"), ("cv", "curve point"),
         ("band_gp_fl", "grade point FL"), ("band", "band value"), ("gp_chfl", "chainage and FL"), ("gp_left", "gradient left"),
         ("gp_right", "gradient right"), ("gp", "grade point")]


def caption(p):
    """What a crop shows, for people: "page 8 · callout EX Br No 320UP" (from the name find_crop.py saved it under)."""
    stem = re.sub(r"^\d+_", "", p.stem)
    for key, words in KINDS:
        if stem == key or stem.startswith(key + "_"):
            stem = f"{words} {stem[len(key):]}"
            break
    page = p.parent.parent.name[4:] if p.parent.parent.name.startswith("page") else ""
    return (f"page {page} · " if page else "") + re.sub(r"[_\s]+", " ", stem).strip()


class Docs:
    """The uploaded documents (in memory) and the one model they are asked with."""

    def __init__(self, ask, log=print):
        self.ask, self.log = ask, log
        self.docs = {}
        self.root = Path(tempfile.mkdtemp(prefix="railway_chat_"))
        self.model_lock = threading.Lock()      # one model: one question at a time

    def add(self, filename, data):
        import pdf_book
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED:
            raise ValueError(f"{filename} is not a drawing I can read · upload a PDF or a sheet image")
        doc = uuid.uuid4().hex[:12]
        folder = self.root / doc
        folder.mkdir(parents=True)
        path = folder / ("upload" + suffix)
        path.write_bytes(data)
        book = pdf_book.Book(path, folder / "work", self.ask, log=self.log)
        book.display_name = filename
        self.docs[doc] = book
        return doc, book

    def get(self, doc):
        if doc not in self.docs:
            raise KeyError("That drawing is no longer loaded · upload it again")
        return self.docs[doc]

    def page_png(self, doc, n):
        """Page n (1-based) as a PNG for the viewer, made once."""
        book = self.get(doc)
        if not 1 <= n <= len(book.sheets):
            raise KeyError(f"there is no page {n}")
        out = book.out / f"view_p{n}.png"
        if not out.exists():
            if book.is_pdf:
                import pymupdf
                page = pymupdf.open(book.path)[n - 1]
                if page.rotation:
                    page.remove_rotation()
                zoom = min(VIEW_WIDTH / page.rect.width, book.sheets[n - 1].dpi / 72)
                page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(out)
            else:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                img = Image.open(book.path).convert("RGB")
                if img.width > VIEW_WIDTH:
                    img = img.resize((VIEW_WIDTH, round(img.height * VIEW_WIDTH / img.width)))
                img.save(out)
        return out

    def answer(self, doc, question):
        import pdf_book
        book = self.get(doc)
        q = question.strip()
        if q.lower() in ("help", "?"):
            return {"answer": pdf_book.HELP, "pages": [], "crops": [], "seconds": 0}
        with self.model_lock:
            t = time.time()
            marks = book.crop_marks()
            text = book.answer(q)
            crops = book.crops_since(marks)
            seconds = time.time() - t
        pages = sorted({int(n) for m in re.findall(r"\[pages? ([\d, ]+)\]", text) for n in re.findall(r"\d+", m)} |
                       {int(p.parent.parent.name[4:]) for p in crops if p.parent.parent.name.startswith("page")})
        return {"answer": text, "pages": pages, "seconds": round(seconds, 1),
                "crops": [{"url": f"/api/doc/{doc}/file?path={p.relative_to(book.out).as_posix()}", "caption": caption(p)}
                          for p in crops]}

    def file(self, doc, rel):
        book = self.get(doc)
        p = (book.out / rel).resolve()
        if not p.is_file() or book.out.resolve() not in p.parents:
            raise KeyError("no such crop")
        return p

    @staticmethod
    def trimmed(p):
        """A crop as shown to people: the white the model's canvas adds around the label trimmed away (PNG bytes)."""
        import io
        from PIL import Image, ImageOps
        img = Image.open(p).convert("RGB")
        box = ImageOps.invert(img.convert("L")).point(lambda v: 255 if v > 12 else 0).getbbox()
        if box:
            m = 12
            img = img.crop((max(0, box[0] - m), max(0, box[1] - m), min(img.width, box[2] + m), min(img.height, box[3] + m)))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()


def create_app(docs):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="Railway drawing assistant")

    class Ask(BaseModel):
        doc: str
        question: str

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "chat_ui" / "index.html").read_text(encoding="utf-8")

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        try:
            doc, book = await run_in_threadpool(docs.add, file.filename or "upload", data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"doc": doc, "name": book.display_name, "pages": len(book.sheets), "note": book.note,
                "suggestions": book.suggestions()}

    @app.get("/api/doc/{doc}/page/{n}")
    async def page(doc: str, n: int):
        try:
            return FileResponse(await run_in_threadpool(docs.page_png, doc, n), media_type="image/png")
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/ask")
    async def ask(req: Ask):
        try:
            return JSONResponse(await run_in_threadpool(docs.answer, req.doc, req.question))
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/doc/{doc}/file")
    async def file(doc: str, path: str, trim: int = 0):
        from fastapi.responses import Response
        try:
            p = docs.file(doc, path)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if trim:
            return Response(await run_in_threadpool(docs.trimmed, p), media_type="image/png")
        return FileResponse(p)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="adapter.zip or an unzipped adapter folder")
    ap.add_argument("--4bit", dest="four_bit", action="store_true", help="load the base model in 4-bit (small GPUs)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other machines on the network")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    import uvicorn
    from app import Model, model_ask
    docs = Docs(model_ask(Model(args.adapter, args.four_bit)))
    print(f"open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    uvicorn.run(create_app(docs), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
