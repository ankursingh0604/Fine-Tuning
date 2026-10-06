"""Find, crop, ask: answer questions about anything printed on a sheet by cutting the right crop and asking the model.

Used by read_sheet.py for what its fixed question list does not cover (and for every gradient / grade-point question).

    sheet = open_sheet("1..pdf", out_dir)          # a PDF (rendered at the training text size) or an image
    finder = Finder(sheet, ask, out_dir)            # ask(images, question) -> model answer
    print(finder.answer("gradient at CH 10046"))

1. PDF input: the page is rendered at the resolution where its text is as tall as on the training sheets (a full sheet
   at 150 dpi), whatever the page size, and the text positions come from the PDF's text layer (exact). Images keep
   their resolution; the text positions come from the local OCR (RapidOCR), read once and cached.
2. Find: the question's chainage / numbers and words are matched against the printed text; nothing found -> said so.
3. Crop and ask: a crop around what was found goes to the model with the question.
4. Grade points ("gradient / slope / rise / fall at CH ..."): the symbol is found (its chainage label, its FL, the
   gradient label on each side of the stem). Every other text in the crop is blanked, so a neighbouring symbol's FL can
   never be taken for this one's, and each label is read by the model in its own small crop: the gradient on the left,
   the gradient on the right, the chainage / FL block. No FL next to the chainage -> "not printed for this grade point".
   For PDFs every reading is compared with the PDF's own text and a difference is shown as CHECK.
Crops used for an answer are saved in <out>/crops/ so they can be looked at.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
TRAIN_TEXT_PT = 9.9          # median text height (pt) on the trained sheets, which were read at 150 dpi
TRAIN_DPI = 150
GRADIENT = re.compile(r"^(?:(?:rise|fall)\s*1\s*in\s*[\d.,]+(?:\s*\([^)]*\))?|level|horizontal|(?:1\s*:\s*)?[\d.]+\s*[FR])$", re.I)
FL_RE = re.compile(r"\bF\.?L\.?\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.I)
GRADE_WORDS = re.compile(r"\b(gradient|grade|slope|rise|fall|falling|rising|grade\s*point|gp|vpi)\b", re.I)
STOP = set("what is the of at on in a an and or to for give me tell show please value values written this that which "
           "sheet drawing crop side sides both left right before after its it does say says printed label point ch "
           "chainage km m metre metres".split())


# ======================================================================== the sheet and its text
class Word:
    """One printed line of text: its box (px), reading direction and text height (px)."""
    __slots__ = ("text", "box", "dir", "h")

    def __init__(self, text, box, d, h):
        self.text, self.box, self.dir, self.h = text.strip(), tuple(box), d, max(3.0, h)

    @property
    def c(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def length(self):
        """Length along the reading direction, from the box and the text height."""
        cx, sy = abs(self.dir[0]), abs(self.dir[1])
        w, hh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        L = (w - self.h * sy) / cx if cx >= 0.5 else (hh - self.h * cx) / sy
        return max(L, self.h)

    @property
    def start(self):
        return (self.c[0] - self.dir[0] * self.length / 2, self.c[1] - self.dir[1] * self.length / 2)

    @property
    def end(self):
        return (self.c[0] + self.dir[0] * self.length / 2, self.c[1] + self.dir[1] * self.length / 2)


def numbers(text):
    """Numbers in a text, '1252+885.000' -> 1252885.0, '10,046' -> 10046."""
    out = []
    for m in re.finditer(r"\d+(?:\s*\+\s*\d+)?(?:[.,]\d+)?", text):
        t = re.sub(r"\s", "", m.group())
        if "+" in t:
            a, b = t.split("+")
            out.append(int(a) * 1000 + float(b.replace(",", ".")))
        else:
            t = t.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", t) else t.replace(",", ".")
            try:
                out.append(float(t))
            except ValueError:
                pass
    return out


def norm(t):
    return re.sub(r"\s+", " ", t).strip().upper()


def meaning(label):
    """What a gradient label says, in words: 'Rise 1 in 1000' / '1150.000 F' / '1:955 F' / 'Level'. The CAD writes a
    level stretch as an enormous '1 in' number (e.g. 103785702.065 R)."""
    t = label.strip()
    if re.fullmatch(r"(?i)level|horizontal", t):
        return "level"
    m = re.search(r"(?i)\b(rise|fall)\s*1\s*in\s*([\d.,]+)", t) or re.search(r"(?:1\s*:\s*)?([\d.]+)\s*([FR])\b", t)
    if not m:
        return None
    if m.group(1).lower() in ("rise", "fall"):
        word, n = ("rising" if m.group(1).lower() == "rise" else "falling"), m.group(2)
    else:
        n, word = m.group(1), "rising" if m.group(2).upper() == "R" else "falling"
    try:
        v = float(n.replace(",", ""))
    except ValueError:
        return None
    return "level" if v > 1e5 else f"1 in {v:g} {word}"


class Sheet:
    def __init__(self, image, words, source, path, note=""):
        self.image, self.words, self.source, self.path, self.note = image, words, source, path, note


def _pdf_words(page, zoom):
    words = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            t = " ".join(s["text"] for s in l["spans"]).strip()
            if t and spans:
                x0, y0, x1, y1 = l["bbox"]
                size = max(s["size"] for s in spans)
                words.append(Word(t, (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom), (l["dir"][0], l["dir"][1]), size * zoom))
    return words


def _quad_word(text, p):
    """A Word from an OCR quadrilateral (corners clockwise from top-left). The text runs along the box's long side and
    its height is the short side; tall boxes are vertical text, which on these drawings reads bottom to top."""
    e01 = (p[1][0] - p[0][0], p[1][1] - p[0][1])
    e03 = (p[3][0] - p[0][0], p[3][1] - p[0][1])
    l01, l03 = math.hypot(*e01) or 1, math.hypot(*e03) or 1
    if l01 >= l03:
        d, h = (e01[0] / l01, e01[1] / l01), l03
    else:
        d, h = (-e03[0] / l03, -e03[1] / l03), l01
    box = (min(q[0] for q in p), min(q[1] for q in p), max(q[0] for q in p), max(q[1] for q in p))
    return Word(text, box, d, h)


def _ocr_words(img, cache):
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved and "q" in saved[0]:
            return [_quad_word(w["t"], w["q"]) for w in saved]
    from layout import ocr
    eng = ocr()
    if eng is None:
        return []
    W, H = img.size
    T, S = 1600, 1400
    found = []
    print("  (reading the sheet's text with the local OCR, once ...)")
    for y in range(0, max(1, H - T + S), S):
        for x in range(0, max(1, W - T + S), S):
            tile = np.asarray(img.crop((x, y, min(W, x + T), min(H, y + T))).convert("RGB"))
            res, _ = eng(tile)
            for quad, text, score in res or []:
                p = [(float(px) + x, float(py) + y) for px, py in quad]
                w = _quad_word(text, p)
                if not any(o.text == w.text and math.dist(o.c, w.c) < 12 for o, _ in found):    # tiles overlap
                    found.append((w, p))
    cache.write_text(json.dumps([{"t": w.text, "q": p} for w, p in found], ensure_ascii=False), encoding="utf-8")
    return [w for w, _ in found]


def open_sheet(path, out_dir, page_no=0):
    """A PDF page rendered at the training text size (with its text layer), or an image (with OCR text)."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        import pymupdf
        doc = pymupdf.open(path)
        page = doc[page_no]
        if page.rotation:
            page.remove_rotation()                     # text positions and the picture then share one frame
        sizes = sorted(s["size"] for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
                       for s in l["spans"] if s["text"].strip())
        med = sizes[len(sizes) // 2] if sizes else TRAIN_TEXT_PT
        dpi = int(min(900, max(100, round(TRAIN_DPI * TRAIN_TEXT_PT / med / 25) * 25)))
        png = out_dir / f"{path.stem}_p{page_no + 1}_{dpi}dpi.png"
        if not png.exists():
            page.get_pixmap(dpi=dpi).save(png)
        note = (f"{path.name}: page {page_no + 1} of {len(doc)}, rendered at {dpi} dpi so its text is as tall as on the trained "
                f"sheets (median text {med:.1f} pt); text positions from the PDF (exact).")
        return Sheet(Image.open(png).convert("RGB"), _pdf_words(page, dpi / 72), "pdf", png, note)
    img = Image.open(path).convert("RGB")
    words = _ocr_words(img, out_dir / f"{path.stem}_ocr.json")
    note = f"{path.name}: image; text positions from the local OCR ({len(words)} pieces of text)." if words else \
        f"{path.name}: image; the local OCR is not installed, so text cannot be located (pip install rapidocr_onnxruntime)."
    return Sheet(img, words, "ocr", path, note)


# ======================================================================== crops
def pad_canvas(img, min_side=448):
    """Keep the text at its size: place small crops on white instead of letting the model's processor enlarge them."""
    W, H = max(min_side, img.width), max(min_side, img.height)
    W, H = -(-W // 28) * 28, -(-H // 28) * 28
    c = Image.new("RGB", (W, H), "white")
    c.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return c


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def grow(box, m, W, H):
    return (max(0, box[0] - m), max(0, box[1] - m), min(W, box[2] + m), min(H, box[3] + m))


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ======================================================================== finder
class Finder:
    def __init__(self, sheet, ask, out_dir):
        self.s, self.ask, self.out = sheet, ask, Path(out_dir) / "crops"
        self.out.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def save(self, img, what):
        self.n += 1
        p = self.out / f"{self.n:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', what)[:40]}.png"
        img.save(p)
        return p

    # ------------------------------------------------------------ entry points
    @staticmethod
    def handles(q):
        """Gradient / grade-point questions always come here (the fixed question list has no answer for them)."""
        return bool(GRADE_WORDS.search(q)) and bool(numbers(q))

    def answer(self, q):
        if not self.s.words:
            return "I cannot locate text on this sheet: " + self.s.note
        if GRADE_WORDS.search(q) and numbers(q):
            gp = self.grade_points(q)
            if gp:
                return gp
        return self.generic(q)

    # ------------------------------------------------------------ finding text
    @staticmethod
    def codes(q):
        """Identifiers in the question such as 'C.NO.-1', 'BR NO. 12', 'TBM3', 'LC-247' (letters joined to a number)."""
        return {re.sub(r"[^A-Z0-9]", "", m.upper()) for m in re.findall(r"[A-Za-z][A-Za-z.]*\s?(?:NO\.?\s?)?[-.]?\s?\d+[A-Za-z]?", q)}

    def matches(self, q):
        """Printed texts that match the question: the same number or identifier first, then shared words."""
        qnums = [n for n in numbers(q) if n >= 10]
        qcodes = self.codes(q)
        qwords = {w for w in re.findall(r"[A-Za-z]{3,}", q.lower()) if w not in STOP}
        scored = []
        for w in self.s.words:
            nums = numbers(w.text)
            flat = re.sub(r"[^A-Z0-9]", "", w.text.upper())
            s = 10 * sum(1 for a in qnums if any(abs(a - b) < 0.6 for b in nums))
            s += 10 * sum(1 for c in qcodes if c and re.search(re.escape(c) + r"(?!\d)", flat))
            s += sum(1 for x in qwords if x in w.text.lower())
            if s:
                scored.append((s, w))
        scored.sort(key=lambda t: -t[0])
        return scored

    def generic(self, q):
        m = self.matches(q)
        qnums = [n for n in numbers(q) if n >= 10]
        if qnums and (not m or m[0][0] < 10):
            # a number asked about is printed nowhere: say so, never answer about something else
            terms = ", ".join(str(int(n)) if n == int(n) else str(n) for n in qnums)
            return f"{terms} is not printed anywhere on this sheet, so I cannot answer that from it (nothing is guessed)."
        if not m:
            return "I could not find what the question is about printed on this sheet, so I cannot answer from it (nothing is guessed)."
        w = m[0][1]
        W, H = self.s.image.size
        cx, cy = w.c
        box = (max(0, cx - 450), max(0, cy - 320), min(W, cx + 450), min(H, cy + 320))
        crop = self.s.image.crop(tuple(int(v) for v in box))
        p = self.save(crop, w.text)
        ans = self.ask([crop], q)
        return f"(read from a crop around \"{w.text}\" - {p})\n{ans}"

    # ------------------------------------------------------------ grade points
    def symbols(self, q):
        """Grade-point symbols whose chainage label carries a number from the question."""
        qnums = [n for n in numbers(q) if n >= 10]
        out = []
        for w in self.s.words:
            if not re.search(r"\bCH\b|CH\s*[:.]", w.text, re.I) or re.match(r"\s*(UP\s*TO|UPTO|FROM)\b", w.text, re.I):
                continue                                   # "UPTO CH: ..." are band-table extents, not symbols
            nums = numbers(w.text)
            if not any(abs(a - b) < 0.6 for a in qnums for b in nums):
                continue
            sym = self.symbol(w)
            if sym and (sym["left"] or sym["right"]):
                key = tuple(x.text if x else None for x in (sym["ch"], sym["fl"], sym["left"], sym["right"]))
                if key not in {tuple(x.text if x else None for x in (o["ch"], o["fl"], o["left"], o["right"])) for o in out}:
                    out.append(sym)                        # the drawing sometimes prints a symbol twice
        return out

    def symbol(self, ch):
        h = ch.h
        dx, dy = ch.dir
        near = lambda o, r: math.dist(o.c, ch.c) < r
        same_dir = lambda o: o.dir[0] * dx + o.dir[1] * dy > 0.95
        # FL: printed with the chainage (same direction, the next line or the same line)
        fls = [o for o in self.s.words if o is not ch and FL_RE.search(o.text) and same_dir(o) and near(o, 3.0 * h)]
        if FL_RE.search(ch.text):
            fl = ch
        else:
            fl = min(fls, key=lambda o: math.dist(o.c, ch.c)) if fls else None
        # The bar bends at the grade point (the vertex): the gradient before it is written along the left arm and READS
        # INTO the vertex (its end is there); the one after it is written along the right arm and STARTS there. The arms
        # can have different angles. The vertex is the label end / start nearest to this symbol's chainage label;
        # a neighbouring symbol has its own vertex, nearer to its own chainage label.
        grads = [o for o in self.s.words if GRADIENT.match(o.text) and near(o, 16 * h)]
        if not grads:
            return None
        # candidate vertices: a left label's end meeting a right label's start (both arms), or a lone end / start
        cands = []
        for a in grads:
            for b in grads:
                # two arms of one symbol: different labels that do not overlap (drawings repeat a shared stretch's
                # label almost on top of itself at neighbouring symbols - those copies are never a pair)
                if a is not b and norm(a.text) != norm(b.text) and not overlaps(a.box, b.box) \
                        and math.dist(a.end, b.start) < 4 * h:
                    mid = ((a.end[0] + b.start[0]) / 2, (a.end[1] + b.start[1]) / 2)
                    cands.append((mid, a, b))
        cands += [(o.end, o, None) for o in grads] + [(o.start, None, o) for o in grads]
        labels = [o for o in self.s.words if re.search(r"\bCH\b|CH\s*[:.]", o.text, re.I)]
        owner = lambda p: min(labels, key=lambda o: math.dist(o.c, p))
        mine = [c for c in cands if owner(c[0]) is ch]                   # a vertex belongs to its nearest chainage label
        if not mine:
            return None
        # nearest vertex to this label; a two-armed one wins over a lone label at (almost) the same place
        best = min(mine, key=lambda c: math.dist(c[0], ch.c) - (2 * h if c[1] and c[2] else 0))
        vx, left, right = best
        return {"ch": ch, "fl": fl, "left": left, "right": right, "vertex": vx}

    def clean_crop(self, keep, region):
        """The region with every text that is not in `keep` painted white (no neighbour's label can be read)."""
        region = tuple(int(v) for v in region)
        orig = self.s.image.crop(region)
        img = orig.copy()
        d = ImageDraw.Draw(img)
        kept = [k.box for k in keep if k]
        for o in self.s.words:
            if overlaps(o.box, region) and not any(o.box == k for k in kept):
                b = o.box
                d.rectangle([b[0] - region[0] - 2, b[1] - region[1] - 2, b[2] - region[0] + 2, b[3] - region[1] + 2], fill="white")
        for k in kept:                     # a kept label overlapped by another one keeps its own pixels
            box = tuple(int(round(v)) for v in (k[0] - region[0], k[1] - region[1], k[2] - region[0], k[3] - region[1]))
            box = (max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3]))
            if box[2] > box[0] and box[3] > box[1]:
                img.paste(orig.crop(box), box[:2])
        return img

    def read_label(self, word, region_keep, what):
        """One label in its own small crop, read by the model."""
        W, H = self.s.image.size
        box = grow(word.box, 0.6 * word.h, W, H)
        crop = pad_canvas(self.clean_crop(region_keep, box))
        p = self.save(crop, what)
        ans = self.ask([crop], "What text is written in this crop? Reply with the exact text only.")
        return ans.strip().strip('"').strip(), p

    def check(self, read, word):
        if self.s.source != "pdf":
            return ""
        r, t = norm(read).replace(" ", ""), norm(word.text).replace(" ", "")
        if r == t:
            return " (matches the PDF text)"
        if t in r:                         # the label overlaps a printed copy of itself, so the crop shows a bit of both
            return f" (contains the PDF text \"{word.text}\"; the drawing prints an overlapping copy here)"
        return f" - CHECK: the PDF text says \"{word.text}\""

    def grade_points(self, q):
        syms = self.symbols(q)
        if not syms:
            return None
        W, H = self.s.image.size
        out = []
        for sym in syms[:3]:
            ch, fl, left, right = sym["ch"], sym["fl"], sym["left"], sym["right"]
            keep = [ch, fl, left, right]
            region = grow(union([k.box for k in keep if k]), 2.5 * ch.h, W, H)
            whole = self.clean_crop(keep, region)
            p_whole = self.save(whole, f"gp_{ch.text}")
            lines = [f"Grade point {ch.text}:"]
            for side, w in (("left of the stem", left), ("right of the stem", right)):
                if w is None:
                    lines.append(f"  gradient {side}: none printed")
                    continue
                read, _ = self.read_label(w, [w], f"gp_{side.split()[0]}_{ch.text}")
                mean = meaning(read)
                lines.append(f"  gradient {side}: {read}" + (f" (= {mean})" if mean and mean != read.lower() else "") + self.check(read, w))
            # the chainage / FL block, read together
            block = [ch] + ([fl] if fl and fl is not ch else [])
            b = grow(union([k.box for k in block]), 0.6 * ch.h, W, H)
            crop = pad_canvas(self.clean_crop(block, b))
            self.save(crop, f"gp_chfl_{ch.text}")
            read = self.ask([crop], "What text is written in this crop? Reply with the exact text only.")
            m = FL_RE.search(read)
            if fl is None:
                lines.append("  FL: not printed for this grade point" + (f" - CHECK: the model read \"{read.strip()}\"" if m else ""))
            else:
                fl_read = m.group(1) if m else None
                fl_true = FL_RE.search(fl.text).group(1)
                if fl_read is None:
                    lines.append(f"  FL: the model could not read it" + (f" (the PDF text says {fl_true})" if self.s.source == "pdf" else ""))
                else:
                    ok = self.s.source != "pdf" or abs(float(fl_read) - float(fl_true)) < 1e-9
                    lines.append(f"  FL: {fl_read}" + ((" (matches the PDF text)" if self.s.source == "pdf" else "") if ok
                                                       else f" - CHECK: the PDF text says {fl_true}"))
            lines.append(f"  (crops in {self.out}; the whole symbol, other text blanked: {p_whole.name})")
            out.append("\n".join(lines))
        if len(syms) > 3:
            out.append(f"... and {len(syms) - 3} more symbols with that number.")
        return "\n\n".join(out)
