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
GRADIENT = re.compile(r"^(?:(?:(?:rise|fall)\s*1\s*in\s*[\d.,]+|level|horizontal)(?:\s*\([^)]*\))?|(?:1\s*:\s*)?[\d.]+\s*[FR])$", re.I)
# the gradient band also abbreviates: "F 1 in 308 (-0.325%)", "R 1000" (= rise 1 in 1000); only read inside a band
BAND_GRADIENT = re.compile(r"^[RF]\s+(?:1\s*in\s*)?[\d.,]+(?:\s*\([^)]*\))?$", re.I)
FL_RE = re.compile(r"\bF\.?L\.?\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.I)
# a bridge named in a question: "Br. No. 15", "BR NO. 575A", "EXG. BR. NO. 320UP", "bridge 560"
# "Br. No. 239AUP", "bridge 214 UP", and a number with UP / DN on its own: "FL value for 224UP"
BRIDGE_REF = re.compile(r"(?:\bbr(?:idge)?\.?\s*(?:no\.?)?\s*[:.\-]?\s*|\b(?=\d{1,4}A?(?:UP|DN)\b))(\d+(?:[a-z]{1,3}|\s(?:up|dn))?)\b", re.I)
# a curve named in a question: "curve 8", "curve no. 8", "C-8", "C. NO. - 8", "C.NO.17U" (not "CH 11540", "TPCC2")
# a list of bridges after one "bridge": "bridge 224, 225, 227 & 230", "BR. NOS. 12 and 14", "bridges 224UP / 225 DN"
BRIDGE_LIST = re.compile(r"\bbr(?:idge)?s?\.?\s*(?:nos?\.?)?\s*[:.\-]?\s*(\d+[a-z]{0,3}(?:\s+(?:up|dn)\b)?"
                         r"(?:\s*(?:,|&|/|\band\b)\s*\d+[a-z]{0,3}(?:\s+(?:up|dn)\b)?)*)", re.I)


def bridge_nums(q):
    """Every bridge number a question names, in its order: "224", "224UP" ("bridge 224, 225, 227, 230" -> all four)."""
    found = [n for m in BRIDGE_LIST.finditer(q) for n in re.findall(r"\d+[a-z]{0,3}(?:\s+(?:up|dn)\b)?", m.group(1), re.I)]
    found += BRIDGE_REF.findall(q)
    return list(dict.fromkeys(re.sub(r"\s+", "", n).upper() for n in found))


# "details about bridge 224, 225": the template-layout CSV (with "csv" / "excel" ... too)
DETAILS_Q = re.compile(r"\bdetails?\b|\babout\b|\binfo(?:rmation)?\b", re.I)
CURVE_ID = re.compile(r"\b(?:curve\s*(?:no\.?)?|c\.?\s*no\.?|c)\s*[-.:]?\s*(\d+[a-z]*)\b", re.I)
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


def fmt(v):
    """A chainage or level to the mm, without trailing zeros: 11701.654, 11384, 3.83."""
    return f"{v:.3f}".rstrip("0").rstrip(".")


def meaning(label):
    """What a gradient label says, in words: 'Rise 1 in 1000' / 'R 1000' / 'F 1 in 308' / '1150.000 F' / '1:955 F' /
    'Level'. The CAD writes a level stretch as an enormous '1 in' number (e.g. 103785702.065 R)."""
    t = label.strip()
    if re.fullmatch(r"(?i)(level|horizontal)(\s*\([^)]*\))?", t):
        return "level"
    m = re.search(r"(?i)\b(rise|fall)\s*1\s*in\s*([\d.,]+)", t) or re.match(r"(?i)([RF])\s+(?:1\s*in\s*)?([\d.,]+)", t)
    if m:
        word, n = ("rising" if m.group(1).lower() in ("rise", "r") else "falling"), m.group(2)
    else:
        m = re.search(r"(?:1\s*:\s*)?([\d.]+)\s*([FR])\b", t)
        if not m:
            return None
        n, word = m.group(1), "rising" if m.group(2).upper() == "R" else "falling"
    try:
        v = float(n.replace(",", ""))
    except ValueError:
        return None
    return "level" if v > 1e5 else f"1 in {v:g} {word}"


class Sheet:
    """A page: its text (positions in the picture's pixels) and its picture. The picture can be made on first use
    (render), so a PDF's pages can all be indexed from their text and only the pages an answer needs are rendered."""

    def __init__(self, image, words, source, path, note="", render=None):
        self._image, self.words, self.source, self.path, self.note, self._render = image, words, source, path, note, render

    @property
    def image(self):
        if self._image is None:
            self._image = self._render()
        return self._image


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


def open_sheet(path, out_dir, page_no=0, lazy=False):
    """A PDF page rendered at the training text size (with its text layer), or an image (with OCR text). With lazy, a
    PDF page is rendered only when its picture is first needed."""
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

        def render():
            if not png.exists():
                d = pymupdf.open(path)
                pg = d[page_no]
                if pg.rotation:
                    pg.remove_rotation()
                pg.get_pixmap(dpi=dpi).save(png)
            return Image.open(png).convert("RGB")
        note = (f"{path.name}: page {page_no + 1} of {len(doc)}, rendered at {dpi} dpi so its text is as tall as on the trained "
                f"sheets (median text {med:.1f} pt); text positions from the PDF (exact).")
        sheet = Sheet(None if lazy else render(), _pdf_words(page, dpi / 72), "pdf", png, note, render)
        sheet.pdf, sheet.page_no, sheet.dpi = path, page_no, dpi
        return sheet
    img = Image.open(path).convert("RGB")
    words = _ocr_words(img, out_dir / f"{path.stem}_ocr.json")
    note = f"{path.name}: image; text positions from the local OCR ({len(words)} pieces of text)." if words else \
        f"{path.name}: image; the local OCR is not installed, so text cannot be located (pip install rapidocr_onnxruntime)."
    return Sheet(img, words, "ocr", path, note)


def export_bridges(rows, path, q, owner):
    """Write [(row read by the model, row from the PDF text, where)] as the JSON list of bridges; the answer says how
    many, where the file is, and every value the model read differently from the PDF. owner.last_export keeps the file
    and rows for a UI."""
    import bridge_list
    got = [g for g, _, _ in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = bridge_list.dump(got)
    path.write_text(text, encoding="utf-8")
    csv_path = path.with_suffix(".csv")                    # the same list for Excel
    bridge_list.write_csv(got, csv_path)
    issues = [line for g, w, where in rows for line in bridge_list.problems(g, w, where)]
    owner.last_export = {"path": path, "csv": csv_path, "rows": got, "issues": issues}
    if not got:
        return f"No bridges matching that are printed here, so the list is empty ({path.name}, {csv_path.name})"
    head = (f"{len(got)} item{'s' if len(got) != 1 else ''} (bridges, level crossings, ROBs and RUBs), in chainage order · "
            f"saved as {path} and {csv_path.name} (for Excel)")
    tail = ("Every value was read by the model and matches the PDF text" if not issues else
            "CHECK · these readings differ from the PDF text:\n" + "\n".join("  " + i for i in issues))
    return f"{head}\n{text}\n{tail}"


CARD_Q = re.compile(r"\b(csv|excel|xlsx|spreadsheet|template)\b", re.I)


def wants_card(q, nums):
    """The question asks for bridges' details as the template-layout CSV: "... in csv / excel", or "details / about /
    information" of bridges it names (not a curve, not one value such as "FL of bridge 264")."""
    if re.search(r"\bcurves?\b", q, re.I):
        return False
    return bool(CARD_Q.search(q)) or bool(nums) and bool(DETAILS_Q.search(q))


def not_found(nums, rows):
    """The bridge numbers asked for that no row is ("224" is found by "224 DN" / "224DN")."""
    have = [re.sub(r"\s", "", (r.get("br_no") or "")).upper() for r in rows]
    return [n for n in nums if not any(h == n or re.fullmatch(re.escape(n) + r"(UP|DN)", h) for h in have)]


def sheet_objects_text(parts):
    import sheet_objects
    return sheet_objects.joined(parts).text


def export_cards(items, folder, name, owner):
    """Write [(card, row read, row from the PDF text, where)] in the arch_rcc layout: one bridge -> one CSV; several ->
    one CSV each (the same layout, existing / proposed columns) packed in a zip. The answer gives the file, the filled
    rows (one bridge) or the bridges in it, and every value read differently from the PDF. owner.last_export keeps the
    file for a UI."""
    import zipfile
    import bridge_card
    import bridge_list
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tag = lambda g: re.sub(r"[^A-Za-z0-9]+", "", g["br_no"] or "bridge")
    if len(items) == 1:
        path = bridge_card.write([items[0][0]], folder / f"{name}_{tag(items[0][1])}_details.csv")
        preview = path.read_text(encoding="utf-8-sig")
    else:
        path = folder / f"{name}_bridges_details.zip"
        used = set()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for c, g, _, _ in items:
                fn = f"{name}_{tag(g)}_details.csv"
                k = 2
                while fn in used:                          # (the same number twice on a sheet: both kept)
                    fn, k = f"{name}_{tag(g)}_{k}_details.csv", k + 1
                used.add(fn)
                z.writestr(fn, BOM + "".join(",".join(_csv_cell(v) for v in r) + "\r\n" for r in bridge_card.table([c])))
        preview = "\n".join(sorted(used))
    issues = [line for _, g, w, where in items for line in bridge_list.problems(g, w, where)]
    issues += [where + g["_degree_check"] for _, g, _, where in items if g.get("_degree_check")]
    owner.last_export = {"path": path, "csv": path, "rows": [g for _, g, _, _ in items], "issues": issues, "kind": "card",
                         "preview": preview}
    head = (f"{len(items)} bridge{'s' if len(items) != 1 else ''} in the arch_rcc layout"
            + (" (one CSV each, in a zip)" if len(items) > 1 else "")
            + f" - RDSO NO. {bridge_card.RDSO}, skew angle 0, end distance 0; rows not printed on an L-section left empty · "
              f"saved as {path}")
    tail = ("Every value was read by the model and matches the PDF text" if not issues else
            "CHECK · these readings differ from the PDF text:\n" + "\n".join("  " + i for i in issues))
    if len(items) > 1:
        body = ("In it: " + ", ".join(f"{g['br_no']} (CH {bridge_card.chainage_m(g['chainage']) or '?'})" for _, g, _, _ in items)
                + ". For one bridge ask 'bridge <number> details in csv'.")
    else:
        body = bridge_card.text([items[0][0]])
    return f"{head}\n{body}\n{tail}"


BOM = chr(0xFEFF)                     # (so Excel opens the CSV as UTF-8)


def _csv_cell(v):
    """A CSV cell as Python's csv module writes it (quoted when it holds a comma, a quote or a line break)."""
    v = "" if v is None else str(v)
    return '"' + v.replace('"', '""') + '"' if re.search(r'[,"\r\n]', v) else v


# ======================================================================== crops
def upright(img, d):
    """A crop of text written along direction d (image x right, y down) turned so the text reads left to right, as on
    the model's training crops: vertical band values given as they are come back with their characters reversed
    ("-0.606" read as "9090-"). Text within 30 degrees of horizontal is left as it is (the tilted gradient labels on a
    profile, which the model reads well); steeper text such as the plan's diagonal callouts is turned."""
    ang = math.degrees(math.atan2(-d[1], d[0]))           # counter-clockwise from left-to-right
    return img if abs(ang) <= 30 else img.rotate(-ang, expand=True, fillcolor="white")


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
        self.saved = []                  # every crop written, in order (a UI shows the ones behind an answer)

    def save(self, img, what):
        self.n += 1
        p = self.out / f"{self.n:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', what)[:40]}.png"
        img.save(p)
        self.saved.append(p)
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
        obj = self.crossing_answer(q) or self.object_answer(q)
        if obj:
            return obj
        band = self.band_answer(q)
        if band:
            return band
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

    # ------------------------------------------------------------ bridges and curves
    def objects(self):
        """The sheet's bridges and curves (sheet_objects.py), found once."""
        if getattr(self, "_objects", None) is None:
            import sheet_objects
            self._objects = sheet_objects.SheetObjects(self.s.words)
        return self._objects

    def strip_crop(self, parts):
        """Only the label's own lines: each line's strip (along its text, a text height thick) is kept and everything
        else is white. A diagonal label's upright box also holds the labels printed parallel to it; this does not."""
        import sheet_objects
        W, H = self.s.image.size
        lab = sheet_objects.joined(parts)
        region = tuple(int(v) for v in grow(lab.box, 0.6 * lab.h, W, H))
        mask = Image.new("L", (region[2] - region[0], region[3] - region[1]), 0)
        d = ImageDraw.Draw(mask)
        for p in parts:
            (dx, dy), (nx, ny) = p.dir, (-p.dir[1], p.dir[0])
            a, b = p.length / 2 + 0.3 * p.h, 0.65 * p.h
            pts = [(p.c[0] + sa * a * dx + sb * b * nx - region[0], p.c[1] + sa * a * dy + sb * b * ny - region[1])
                   for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            d.polygon(pts, fill=255)
        img = Image.composite(self.s.image.crop(region), Image.new("RGB", mask.size, "white"), mask)
        img = upright(img, lab.dir)
        box = img.convert("L").point(lambda v: 255 if v < 250 else 0).getbbox()
        if box:                                             # turned text: trim the empty corners the turn leaves
            m = int(0.6 * lab.h)
            img = img.crop((max(0, box[0] - m), max(0, box[1] - m), min(img.width, box[2] + m), min(img.height, box[3] + m)))
        return img, lab

    def read_parts(self, parts, what):
        """A label - all its printed lines - read by the model from a crop of its own strip, turned to read left to
        right. Read once per label (several answers can need the same one)."""
        reads = self.__dict__.setdefault("_reads", {})
        key = tuple(id(p) for p in parts)
        if key not in reads:
            img, lab = self.strip_crop(parts)
            crop = pad_canvas(img)
            self.save(crop, what)
            read = self.ask([crop], "What text is written in this crop? Reply with the exact text only.")
            reads[key] = (" ".join(read.split()).strip('"').strip(), lab)
        return reads[key]

    def object_answer(self, q):
        """Questions about the bridges and curves printed on the sheet; None when the question is about neither."""
        ql = q.lower()
        objs = self.objects()
        import band_table
        named = CURVE_ID.search(q) and not re.search(r"\bcurves?\b", ql) and not band_table.asks_row(q)
        cid = CURVE_ID.search(q)
        has_block = bool(cid and objs.curve_blocks(cid.group(1).upper()))      # "CURVE No. 207" printed here
        if (re.search(r"\bcurves?\b", ql) or named) and (objs.curves or has_block or not objs.bridges):
            return self.curve_answer(q)                     # "curve no. 8", "C-8", "C. NO. - 17U", "C-8 TPCC2"
        bridge_q = re.search(r"\bbr(?:idge)?s?\b|\bbr\.|\b\d{1,4}a?(?:up|dn)\b", ql)      # (also a bare "224UP")
        if not bridge_q or not objs.bridges:
            return None
        nums = bridge_nums(q)
        if wants_card(q, nums):                             # "details about bridge 224, 225" / "... in csv": the template
            return self.bridge_csv(nums)
        if not nums:
            if re.search(r"\b(list|all|how many|which|what|show|every|count)\b", ql):
                return self.list_bridges(q)
            return None
        # a column of the bridge table that no callout / level block carries ("discharge of 229UP", "free board of
        # bridge 220", "remarks for bridge 231"): the table's cells for it
        if self.table_asks(ql, levels=False):
            got = [self.table_answer(n, ql, levels=False) for n in dict.fromkeys(nums)]
            if any(got):
                return "\n\n".join(g or f"Bridge {n}: no row for it in a bridge table on this sheet." for n, g in zip(dict.fromkeys(nums), got))
        # a band value at the bridge (ground level, cut / fill, track distance, a difference): the band reader. A level
        # printed in the bridge's own level block (FL / RL / HFL / BL / HC / VC, "RL 4TH LINE") is answered from it
        if re.search(r"ground|\bogl\b|\bgl\b|\bcut\b|\bfill\b|track\s*(?:dist|cent)|differ", ql):
            return None
        levels = re.search(self.LEVEL_Q, ql)
        if band_table.asks_row(q) and not levels:
            return None
        if re.search(r"\brail\b|\br\.?l\b", ql) and not any(re.search(r"\bR\.?\s*L\b", sheet_objects_text(b.levels[0]), re.I)
                                                        for n in nums for b in objs.find_bridges(num=n) if b.levels):
            return None                                     # no rail level in the bridge's block: the band's rail level
        return self.bridge_answer(q, nums)

    @staticmethod
    def asked_status(ql):
        ex = re.search(r"\b(ex|exg|exist\w*|old)\b", ql)
        pr = re.search(r"\b(prop\w*|new)\b", ql)
        return "existing" if ex and not pr else "proposed" if pr and not ex else None

    TYPE_WORDS = [(r"\br\.?\s*c\.?\s*c\b|\brcb\b", ("RCC", "RCB")), (r"\bbox\b|\brcb\b", ("BOX", "RCB")),
                  (r"hume|\bpipes?\b", ("HUME", "PIPE")), (r"girder", ("GIRDER",)), (r"\bplate\b", ("PLATE",)),
                  (r"\bslabs?\b", ("SLAB",)), (r"\barch\w*", ("ARCH",)), (r"\bpsc\b", ("PSC",)),
                  (r"flat\s*top|\bf\.?\s*t\b", ("FT",))]

    def check_fields(self, b, read, keys=("chainage", "span", "type", "proposal")):
        """The callout as the model read it, and whether the fields shown in the answer agree with the PDF text."""
        import sheet_objects
        f, p = b.fields(read), b.fields()
        if self.s.source != "pdf":
            return f, ""
        nt = lambda v: sheet_objects.norm_type(v or "")

        def same(k):
            if k == "chainage":
                return f[k] is not None and p[k] is not None and abs(f[k] - p[k]) < 1e-6
            return nt(f[k]) == nt(p[k])
        return f, (" (matches the PDF text)" if all(same(k) for k in keys) else
                   f" - CHECK: the PDF text says \"{sheet_objects.joined(b.best()).text}\"")

    def list_bridges(self, q):
        """The bridge list as a JSON file in the agreed format (bridge_list.py), every value read by the model."""
        import bridge_list
        rows = [(g, w, "") for g, w in bridge_list.rows(self, q)]
        name = Path(getattr(self.s, "pdf", None) or self.s.path).stem
        return export_bridges(rows, self.out.parent / f"{name}_bridges.json", q, self)

    def bridge_cards(self, nums):
        """[(card, got, want)] for the bridges asked for (every bridge on the sheet when none is named), in chainage
        order - each read by the model from its own callout and level block (bridge_list.bridge_row)."""
        import bridge_card
        import bridge_list
        groups = bridge_list.groups_of(self.objects())
        if nums:
            groups = {n: bs for n, bs in groups.items()
                      if any(n == x or re.fullmatch(re.escape(x) + r"(UP|DN)", n) for x in nums)}
        out = []
        for num, bs in groups.items():
            got, want, lv_read, lv_pdf = bridge_list.bridge_row(self, num, bs)
            deg, note = self.curve_degree(bridge_list.chainage_value(want["chainage"]), num)
            notes = [note] if note else []
            if want.get("_prop_chainage"):                 # the proposed bridge at its own chainage: its own curve
                deg_pr, note_pr = self.curve_degree(bridge_list.chainage_value(want["_prop_chainage"]), num)
                deg = (deg[0], deg_pr[1])
                notes += [note_pr] if note_pr and note_pr != note else []
            # the card's own level cells, checked like the rest: HFL (the highest flood level) and the rail levels
            if (got.get("_prop_chainage") or None) != (want.get("_prop_chainage") or None):
                notes.append(f"{num} · proposed chainage: read {got.get('_prop_chainage')!r}, the PDF says {want.get('_prop_chainage')!r}")
            for what, f in (("HFL", bridge_card.hfl_max), ("rail level", bridge_card.rail_levels)):
                r, p_ = f(lv_read), f(lv_pdf)
                if r != p_:
                    notes.append(f"{num} · {what}: read {r!r}, the PDF says {p_!r}")
            c = bridge_card.card(got, lv_read, deg)
            # the sheet's DETAILS OF BRIDGES table: what the callout and level block leave empty (bed level, the
            # proposed type / cells / span / box height, FL, HFL), and where both give a value and they differ, a note
            row_t = self.details_table().get(num) or self.details_table().get(re.sub(r"(UP|DN)$", "", num))
            if row_t:
                notes += self.fill_from_table(c, row_t, num)
            # TC_DISTANCE: the band's track distance at the (proposed) bridge's chainage
            ch_tc = bridge_list.chainage_value(want.get("_prop_chainage") or want["chainage"])
            tc, note_tc = self.track_distance_at(ch_tc, num)
            if tc:
                c["tc"] = (tc, tc)
            notes += [note_tc] if note_tc else []
            if notes:
                got["_degree_check"] = "; ".join(notes)
            out.append((c, got, want))
        out.sort(key=lambda t: bridge_list.chainage_value(t[2]["chainage"]))
        return out

    def details_table(self):
        """The sheet's DETAILS OF BRIDGES table (bridge_table.py), read once: {bridge no: {column: [Word]}}."""
        if getattr(self, "_details", None) is None:
            import bridge_table
            page = None
            if self.s.source == "pdf" and getattr(self.s, "pdf", None) is not None:
                import pymupdf
                page = pymupdf.open(self.s.pdf)[self.s.page_no]
                if page.rotation:
                    page.remove_rotation()
            self._details = bridge_table.read(self.s.words, page, getattr(self.s, "dpi", 72) / 72)
        return self._details

    def fill_from_table(self, c, row, num):
        """Fill the card's empty cells from the bridge's row of the DETAILS OF BRIDGES table, every cell read by the model
        from its own crop and checked against the PDF; returns the notes (misreadings, callout vs table differences)."""
        import bridge_card
        import bridge_table
        notes = []
        norm = lambda v: re.sub(r"[^A-Z0-9.+]", "", (v or "").upper())

        def cell(key):
            ws = row.get(key)
            if not ws:
                return ""
            read = " ".join(self.read_parts(ws, f"tbl_{num}_{key}")[0].split())
            pdf = bridge_table.text(ws)
            if self.s.source == "pdf" and norm(read) != norm(pdf):
                notes.append(f"{num} · bridge table {key}: read {read!r}, the PDF says {pdf!r}")
            return read

        def fill(k, i, v):
            cur = list(c.get(k, ("", "")))
            if v and not cur[i]:
                cur[i] = v
            elif v and cur[i] and norm(v) != norm(cur[i]) and not (re.fullmatch(r"-?\d+(?:\.\d+)?", v) and
                                                                  re.fullmatch(r"-?\d+(?:\.\d+)?", cur[i]) and float(v) == float(cur[i])):
                notes.append(f"{num} · {k} ({'existing' if i == 0 else 'proposed'}): callout / level block {cur[i]!r}, bridge table {v!r}")
            c[k] = tuple(cur)
        bed = cell("bed")
        fill("bed_level", 0, bed)
        fill("bed_level", 1, bed)
        for i, key in ((0, "type_ex"), (1, "type_pr")):
            conf, cat, st = bridge_table.bridge_type(cell(key))
            if st:
                fill("type", i, st)
            if conf:
                n, sp, h = bridge_card.config(conf)
                fill("cells", i, n)
                fill("span", i, sp)
                fill("height", i, h)
        fill("fl", 0, cell("fl_ex"))
        fill("fl", 1, cell("fl_pr"))
        hfls = [v for v in (cell("chfl"), cell("ohfl")) if re.fullmatch(r"-?\d+(?:\.\d+)?", v or "")]
        if hfls:
            top = max(hfls, key=float)
            fill("hfl", 0, top)
            fill("hfl", 1, top)
        return notes

    def track_distance_at(self, ch, num):
        """(track distance at chainage ch as text, note): the band's TRACK DISTANCE row there - the printed column, or
        the two columns either side interpolated - every value read by the model and checked; ("", "") without one."""
        import band_table
        if ch == float("inf") or not self.bands():
            return "", ""
        rows, _ = band_table.match_rows(self.bands(), "track distance")
        rows = [(b, r) for b, r in rows if re.search(r"TRACK|DIST|C\s*/\s*C|CENTRE", r.heading, re.I)]
        for b, r in rows:
            vals = b.values(r)
            if not vals or not (vals[0][0] - 1e-6 <= ch <= vals[-1][0] + 1e-6):
                continue
            # a track-distance row that repeats another row of the band column for column (8.pdf prints the PROP. FL
            # values in it on every page) is a drafting error on the sheet - said so, not used
            mine = [w.text for w in r.words]
            twin = next((o for o in b.rows if o is not r and o is not b.chain and len(o.words) == len(r.words)
                         and sum(a == w.text for a, w in zip(mine, o.words)) >= 0.9 * len(mine)), None)
            if twin is not None:
                return "", (f"{num} · TC distance: this sheet's row '{r.heading}' repeats the row '{twin.heading}' value for "
                            f"value (e.g. {mine[len(mine) // 2]}) - a drafting error on the drawing; left empty")
            exact = [w for c_, w in vals if abs(c_ - ch) < 1e-6]
            pdf = self.s.source == "pdf"
            if exact:
                y, _ = self.read_value(exact[0], r)
                p = band_table.value(exact[0].text)
                note = "" if not pdf or (y is not None and abs(y - p) < 1e-9) else f"{num} · TC distance: read {y!r}, the PDF says {p!r}"
                return (f"{y:.3f}" if y is not None else ""), note
            i = max(k for k, (c_, _) in enumerate(vals) if c_ < ch)
            (c1, w1), (c2, w2) = vals[i], vals[i + 1]
            (y1, _), (y2, _) = self.read_value(w1, r), self.read_value(w2, r)
            p1, p2 = band_table.value(w1.text), band_table.value(w2.text)
            f = lambda a, b_: a + (b_ - a) * (ch - c1) / (c2 - c1)
            note = "" if not pdf or (y1, y2) == (p1, p2) else f"{num} · TC distance columns: read {y1!r}, {y2!r}, the PDF says {p1!r}, {p2!r}"
            return (f"{f(y1, y2):.3f}" if y1 is not None and y2 is not None else ""), note
        return "", ""

    def curve_degree(self, ch, num):
        """((existing, proposed) degree of curve, check note) at chainage ch: the degree of the curve whose start and
        end chainages hold ch, read by the model from that curve's block - per status when both are printed, else the
        one curve for both columns; ("", "") off every curve."""
        import sheet_objects
        if ch == float("inf"):
            return ("", ""), ""
        objs = self.objects()
        # every curve whose start and end chainages hold ch; two lines' curves at one place ("4TH LINE" and "3RD LINE"):
        # the sheet's own line first
        main = objs.main_track
        found = {}
        for c in objs.curves:
            if c.points and c.span[0] <= ch <= c.span[1]:
                found.setdefault(c.status, []).append(c)
        for cs in found.values():
            cs.sort(key=lambda c: (c.track is not None and c.track != main))
        got, notes = {}, []
        for st, cs in found.items():
            for c in cs:
                printed, parts = objs.degree(c)
                if printed is None:
                    continue
                read, _ = self.read_parts(parts, f"cvd_{c.block_num or c.num}")
                m = sheet_objects.DEGREE.search(read)
                val = m.group(1) if m else None
                if val is None or float(val) != float(printed):
                    notes.append(f"{num} · degree of curve ({c.name}): read {val!r}, the PDF says {printed!r}")
                got[st] = val or ""
                break
        ex, pr = got.get("existing"), got.get("proposed")
        one = ex or pr or ""
        return (ex if ex is not None else one, pr if pr is not None else one), "; ".join(notes)

    def bridge_csv(self, nums, where=""):
        """The bridge(s) in the RAILWAY_RCC_BOX_TEMPLATE layout as a CSV (bridge_card.py); the answer shows the filled
        rows and every value the model read differently from the PDF text."""
        cards = self.bridge_cards(nums)
        if not cards:
            there = ", ".join(sorted(b.name for b in self.objects().bridges)) or "none"
            return (f"No bridge {', '.join(nums)} is printed on this sheet (bridges here: {there}; nothing is guessed)."
                    + "".join(self.on_other_pages(n) for n in nums)) if nums else "No bridge callouts are printed on this sheet."
        name = Path(getattr(self.s, "pdf", None) or self.s.path).stem
        text = export_cards([(c, g, w, where) for c, g, w in cards], self.out.parent / "exports", name, self)
        missing = not_found(nums, [w for _, _, w in cards])
        return text + (f"\nNot printed on this sheet (so not in the file; nothing is guessed): {', '.join(missing)}" if missing else "")

    def bridges_asked(self, q):
        """([Bridge], description): the bridges a list question asks for (by status and type)."""
        ql = q.lower()
        st = self.asked_status(ql)
        kinds = [alts for pat, alts in self.TYPE_WORDS if re.search(pat, ql)]
        said = sorted((m.start(), m.group()) for m in (re.search(p, ql) for p, _ in self.TYPE_WORDS) if m)   # as asked
        what = " ".join(x for x in (st, " ".join(dict.fromkeys(w for _, w in said)).upper()) if x)
        return self.objects().find_bridges(status=st, kinds=kinds), what

    def bridge_line(self, b):
        """One bridge for a list: its callout as the model read it, checked against the PDF."""
        read, _ = self.read_parts(b.best(), f"br_{b.name}")
        f, ok = self.check_fields(b, read)
        ch = fmt(f["chainage"]) if f["chainage"] is not None else "?"
        return (f"{b.name}: span {f['span'] or '?'}, {f['type'] or '?'}" +
                (f", proposal: {f['proposal']}" if f["proposal"] else "") + f", at CH {ch}{ok}")

    def bridge_answer(self, q, nums):
        import sheet_objects
        ql = q.lower()
        st = self.asked_status(ql)
        want = {k for k, pat in (("chainage", r"chainage|\bch\b|where|locat|position"), ("span", r"\bspan|opening|size"),
                                 ("type", r"\btype|kind|structure|proposal"),
                                 ("levels", self.LEVEL_Q)) if re.search(pat, ql)}
        out = []
        for num in dict.fromkeys(nums):
            bs = self.objects().find_bridges(num=num, status=st)
            if not bs and want == {"levels"}:
                # one level block serves both: "proposed FL of 229UP" when only the existing callout is printed (the
                # block's 3RD LINE FL is the proposed one); the status then only picks which FL is shown
                bs = [b for b in self.objects().find_bridges(num=num) if b.levels]
            if not bs:
                there = ", ".join(b.name for b in self.objects().find_bridges(num=num)) or "none"
                out.append(f"No {(st + ' ') if st else ''}bridge {num} is printed on this sheet "
                           f"(bridge {num} callouts here: {there}; nothing is guessed)." + self.on_other_pages(num))
                continue
            if want - {"levels"} or not want:
                full = not want - {"levels"}
                shown = [k for k in ("chainage", "span", "type") if k in want or full] + (["proposal"] if "type" in want or full else [])
                for b in bs:
                    read, _ = self.read_parts(b.best(), f"br_{b.name}")
                    f, ok = self.check_fields(b, read, shown)
                    parts = []
                    if "chainage" in shown:
                        parts.append(f"at CH {fmt(f['chainage']) if f['chainage'] is not None else '? (not read)'}")
                    if "span" in shown:
                        parts.append(f"span {f['span'] or '?'}")
                    if "type" in shown:
                        import bridge_list
                        chs = list(bridge_list.AT_CH.finditer(read))           # a proposal printed with its own chainage
                        own = (f" at CH {bridge_list.chainage_str(chs[-1].group(1))}" if f["proposal"] and len(chs) > 1 else "")
                        parts.append(f"{f['type'] or '?'}" + (f", proposal: {f['proposal']}{own}" if f["proposal"] else ""))
                    out.append(f"{b.name}: {', '.join(parts)}{ok}\n  (the model read: \"{read}\")")
            if "levels" in want or not want:
                blocks = bs[0].levels
                clr = self.table_answer(num, ql, blocks)
                if not blocks:
                    out.append(clr or f"Br. No. {num}: no level block (FL / HFL / BL) is printed for it on this sheet.")
                    continue
                out.append(self.level_answer(blocks[0], f"brlv_{num}", f"Br. No. {num}", ql, st) + (f"\n{clr}" if clr else ""))
        return "\n\n".join(out)

    # the bridge table's columns a question can name: (column key or prefix of the _ex / _pr pair, its name, words);
    # level columns (bed, CHFL, OHFL, HC, VC) are answered from the table when the bridge's level block lacks them
    TABLE_COLS = [("discharge", "discharge", r"discharge|\bdisch\b|cumec"), ("fb", "free board", r"free\s*board|\bf\.?b\b"),
                  ("cushion", "earth cushion", r"cushion"), ("slab", "top slab thickness", r"slab\s*thick|top\s*slab"),
                  ("remarks", "remarks", r"remark|\bnote"), ("crossing", "type of crossing", r"crossing|descri|over\s*what|what\s+.*\bcross"),
                  ("hc", "HC", r"\bh\.?c\b|horizontal\s*clearance"), ("vc", "VC", r"\bv\.?c\b|vertical\s*clearance"),
                  ("clearance", "clearance", r"(?<!horizontal )(?<!vertical )clearance(?!\s*\(?[hv])"),
                  ("chfl", "CHFL", r"\bc\.?h\.?f\.?l\b"), ("ohfl", "OHFL", r"\bo\.?h\.?f\.?l\b"), ("bed", "bed level / road level", r"\bbed\b|\bb\.?l\b|road\s*level")]
    LEVEL_COLS = ("hc", "vc", "chfl", "ohfl", "bed")

    def table_asks(self, ql, levels=True):
        """The bridge-table columns the question names (level columns too unless levels=False)."""
        return [(k, n) for k, n, pat in self.TABLE_COLS if re.search(pat, ql) and (levels or k not in self.LEVEL_COLS)]

    def table_answer(self, num, ql, blocks=(), levels=True):
        """Whatever the question asks of the bridge's row in the bridge table (discharge, free board, earth cushion,
        top slab, clearance, HC / VC, remarks, the crossing ...): those cells, read by the model and checked against
        the PDF. Only when asked - the CSV card keeps its own fixed rows. "" when the table has no row for it or the
        question names no column; a level column already printed in the bridge's level block is left to the block."""
        import bridge_table
        want = self.table_asks(ql, levels)
        if not want:
            return ""
        import sheet_objects
        printed = {re.sub(r"[^A-Z]", "", k.upper()) for b in blocks[:1] for k, _ in sheet_objects.BLOCK_KV.findall(" ".join(p.text for p in b))}
        lab = {"hc": ("HC",), "vc": ("VC",), "chfl": ("CHFL",), "ohfl": ("OHFL",), "bed": ("BEDLEVEL", "BL", "ROADLEVEL")}
        want = [(k, n) for k, n in want if not (k in lab and printed & set(lab[k]))]
        key = re.sub(r"\s", "", num.upper())
        row = self.details_table().get(key) or self.details_table().get(re.sub(r"(UP|DN)$", "", key))
        if not want or not row:
            return ""
        st = self.asked_status(ql)
        lines = []
        for k, name in want:
            cols = [(c, name + (" (existing line)" if c.endswith("_ex") else " (proposed line)" if c.endswith("_pr") else ""))
                    for c in ([k] if k in row else [k + "_ex", k + "_pr"])
                    if not (st == "existing" and c.endswith("_pr") or st == "proposed" and c.endswith("_ex"))]
            got = [(c, label) for c, label in cols if row.get(c)]
            if not got:
                lines.append(f"  {name}: not printed in the bridge table for {num}")
            for c, label in got:
                read = " ".join(self.read_parts(row[c], f"tbl_{key}_{c}")[0].split())
                pdf = bridge_table.text(row[c])
                lines.append(f"  {label}: {read}" + ("" if self.s.source != "pdf" else " (matches the PDF text)"
                                                     if read.replace(" ", "") == pdf.replace(" ", "") else f" - CHECK: the PDF text says {pdf}"))
        return f"{num} in the bridge table (read by the model):\n" + "\n".join(lines)

    def read_block(self, parts, what):
        """A level block as the model read it. A line whose "label = value" the reading left out (a tall block of ten
        lines: one can be skipped - "FL DN LINE: ? (not read)") is read again from a crop of that line alone, and added."""
        import sheet_objects
        read, lab = self.read_parts(parts, what)
        key = lambda k: re.sub(r"[^A-Z0-9]", "", re.sub(r"^(?:UP|DN)\s+(?=\S)", "", k.strip(), flags=re.I).upper())
        have = {key(k) for k, _ in sheet_objects.BLOCK_KV.findall(read)}
        extra = []
        for n, p in enumerate(parts):
            if any(key(k) not in have for k, _ in sheet_objects.BLOCK_KV.findall(p.text)):
                r, _ = self.read_parts([p], f"{what}_l{n}")
                extra.append(r)
        return (" ".join([read] + extra), lab) if extra else (read, lab)

    def level_answer(self, block, what, name, ql, st):
        """The values of a level block the question asks for, as the model read them, each checked against the PDF text.
        Labels as printed: FL / EXG FL / PROP. FL / FL UP LINE / FL 3RD LINE, RL ..., BED LEVEL, OHFL, ROAD LEVEL, HC, VC."""
        import sheet_objects
        read, lab = self.read_block(block, what)
        # a bridge number's own UP / DN ("BR. NO. 575 UP" / "EXG FL = ...") is not part of the first label
        strip = lambda pairs: {re.sub(r"^(?:UP|DN)\s+(?=\S)", "", k.strip(), flags=re.I): v for k, v in pairs}
        got = strip(sheet_objects.BLOCK_KV.findall(read))
        true = strip(sheet_objects.BLOCK_KV.findall(lab.text))
        key = lambda k: re.sub(r"[^A-Z0-9]", "", k.upper())
        got = {key(k): v for k, v in got.items()}

        def cat(k2):
            """FL covers EX. FL / EXG FL, PROP. FL, FL, MIN FL REQ., FL UP LINE, FL 3RD LINE; RL the rail levels."""
            if k2.endswith("HFL"):
                return "HFL"
            if k2 in ("BL", "BEDLEVEL", "BEDLVL"):
                return "BL"
            if "ROAD" in k2:
                return "ROAD"
            if k2 in ("HC", "VC"):
                return k2
            if re.match(r"(EXG?|PROP\w*|MIN)?FL", k2):
                return "FL"
            if k2.startswith("RL"):
                return "RL"
            return k2
        # the levels asked for; none named -> all of them
        which = {c for c, pat in (("HFL", r"\b[oc]?\.?h\.?f\.?l\b|flood"), ("BL", r"\bb\.?l\b|\bbed\b"), ("FL", r"\bf\.?l\b|formation"),
                                  ("RL", r"\br\.?l\b|rail\s*level"), ("ROAD", r"road\s*level"),
                                  ("HC", r"\bh\.?c\b|horizontal\s*clearance"), ("VC", r"\bv\.?c\b|vertical\s*clearance"))
                 if re.search(pat, ql)}
        lines = []
        for k, v in true.items():
            k2 = key(k)
            if which and cat(k2) not in which:
                continue
            # existing: not the proposed line ("PROP. FL", "FL 3RD LINE"); proposed: not the existing ("EXG FL", "FL UP LINE")
            newline = k2.startswith("PROP") or re.search(r"\d(ST|ND|RD|TH)LINE", k2)
            oldline = k2.startswith("EX") or re.search(r"(UP|DN|DOWN)LINE", k2)
            if cat(k2) in ("FL", "RL") and (st == "existing" and newline or st == "proposed" and oldline):
                continue
            if st == "existing" and k2 == "FL" and any(key(x).startswith("EX") for x in true):
                continue                                   # a plain "FL" beside an "EXG FL" is the proposed one
            r = got.get(k2)
            lines.append(f"  {k.strip()}: {r if r is not None else '? (not read)'}" +
                         ("" if self.s.source != "pdf" else " (matches the PDF text)" if r is not None and float(r) == float(v)
                          else f" - CHECK: the PDF text says {v}"))
        if not lines:
            lines = [f"  (none of the values asked for is printed in it; it gives: {', '.join(k.strip() for k in true) or 'nothing'})"]
        return f"{name} level block (read by the model):\n" + "\n".join(lines)

    LEVEL_Q = r"\b([oc]?\.?h\.?f\.?l|b\.?l|bed|f\.?l|formation|levels?|r\.?l|rail\s*level|h\.?c|v\.?c|clearance|road\s*level)\b"

    def crossing_answer(self, q):
        """Levels and clearances of a level crossing / ROB / RUB ("HC and VC of LC 123", "FL at LC-117"); None otherwise."""
        import sheet_objects
        ql = q.lower()
        m = re.search(r"\b(lc|rob|rub|fob)\s*(?:no\.?)?\s*[-.]?\s*(\d+[a-z]*)\b", ql)
        if not m or not (re.search(self.LEVEL_Q, ql) or self.table_asks(ql)):
            return None
        kind, num = m.group(1).upper(), m.group(2).upper()
        if not re.search(self.LEVEL_Q, ql):                 # "remarks of LC 123", "what does LC 123 cross": the table
            return self.table_answer(f"{kind}{num}", ql) or None
        cs = [c for c in self.objects().crossings if c.kind == kind and (c.num or "") == num]
        if not cs:
            return self.table_answer(f"{kind}{num}", ql) or None
        c = next((c for c in cs if c.levels), None)
        clr = self.table_answer(f"{kind}{num}", ql, c.levels if c else ())
        if not c:
            return clr or f"{kind} {num}: no level block (FL / road level / HC / VC) is printed for it on this sheet."
        return self.level_answer(c.levels[0], f"xinglv_{c.label}", f"{kind} {num}", ql, self.asked_status(ql)) + (f"\n{clr}" if clr else "")

    def curve_answer(self, q):
        """'Which curve is near Br. No. 15' / 'curve at CH 11384' / 'details of curve 8'."""
        ql = q.lower()
        objs = self.objects()
        cnum = CURVE_ID.search(q)
        if not objs.curves:
            if cnum and objs.curve_blocks(cnum.group(1).upper()):
                return self.curve_details(cnum.group(1).upper())          # only its "CURVE No." block is on this sheet
            return "No curve points (TPTC / TPCC ...) are printed on this sheet, so I cannot answer that from it."
        places = []
        nums = bridge_nums(q)
        if nums:
            st = self.asked_status(ql)
            for b in (b for n in nums for b in objs.find_bridges(num=n, status=st)):
                read, _ = self.read_parts(b.best(), f"br_{b.name}")
                f, ok = self.check_fields(b, read, ("chainage",))
                if f["chainage"] is not None:
                    places.append((f["chainage"], f"{b.name} at CH {fmt(f['chainage'])}{ok}"))
            if not places:
                return f"No bridge {', '.join(nums)} is printed on this sheet (nothing is guessed)."
        elif cnum and re.search(r"\b(bridges?|br)\b", ql):
            return self.bridges_near_curve(cnum.group(1).upper())          # "which bridge is near curve 8"
        elif re.search(r"\b(bridges?|br)\b", ql):                         # "bridge near the curve at CH 1245700"
            cs = [c for ch, _ in self.places(q) for found in objs.curves_near(ch).values() for d, c in found if d == 0]
            if cs:
                return self.bridges_near_curve(None, list(dict.fromkeys(cs)))
        elif cnum and not re.search(r"\bnear|nearest|close|around|at\s+ch|\bat\s+\d", ql):
            pt = re.search(r"\b(TPTC|TPCC|TC|CT|ST|TS|SC|CS)\s*-?\s*(\d)?\b", q, re.I)
            return self.curve_details(cnum.group(1).upper(), (pt.group(1).upper() + (pt.group(2) or "")) if pt else None)
        else:
            places = [(ch, f"CH {fmt(ch)}" + (f" (\"{w.text}\")" if w is not None else "")) for ch, w in self.places(q)]
        if not places:
            return None
        out = []
        for ch, where in places:
            lines = [f"{where}:"]
            for st, found in objs.curves_near(ch).items():
                for d, c in found:
                    lines.append(f"  {st}: {self.curve_position(c, ch, d)}")
            out.append("\n".join(lines))
        return "\n\n".join(out)

    def read_point(self, c, pt, value=False):
        """A curve point's label read by the model: "TPCC1 11367.15" (with CHECK if it differs from the PDF), or with
        value=True (chainage read or None, that text)."""
        ch, w = c.points[pt]
        read, _ = self.read_parts([w], f"cv_{c.num}_{pt}")
        got = re.search(r"CH\.?\s*[:.]?\s*(\d[\d+.]*)", read, re.I)          # "AT Ch. 11367.15m" / "ATCH:11295m"
        n = numbers(got.group(1)) if got else []
        val = n[0] if n else None
        ok = "" if self.s.source != "pdf" else "" if val is not None and abs(val - ch) < 1e-6 else \
            f" - CHECK: the PDF text says \"{w.text}\""
        text = f"{pt} {fmt(val) if val is not None else '? (not read)'}{ok}"
        return (val, text) if value else text

    def curve_position(self, c, ch, d):
        pts = sorted(c.points.items(), key=lambda t: t[1][0])
        name = f"{c.name}" + (f" ({c.hand})" if c.hand else "")
        if d > 0:
            first, last = pts[0], pts[-1]
            pt = first[0] if ch < first[1][0] else last[0]
            side = "before" if ch < first[1][0] else "after"
            pos = f"not on a curve; the nearest is {name}, {fmt(d)} m {side} its {self.read_point(c, pt)}"
        else:                                               # on the curve: its transition or its circular part
            pos = f"on {name} (only {', '.join(p for p, _ in pts)} of it are on this sheet)"
            for (a, (ca, _)), (b, (cb, _)) in zip(pts, pts[1:]):
                if ca <= ch <= cb:
                    part = "circular part" if a in ("TPCC1", "TC", "SC") and b in ("TPCC2", "CT", "CS") else "transition"
                    pos = f"on {name}, in its {part} between {self.read_point(c, a)} and {self.read_point(c, b)}"
                    break
        return pos + (f"\n      {self.curve_summary(c)}" if c.details else "")

    def curve_summary(self, c):
        read, lab = self.read_parts(c.details[0], f"cvd_{c.num}")
        return f"details (read by the model): {read}" + self.same_numbers(read, lab.text)

    def same_numbers(self, read, text):
        """For a block of values: the model's reading agrees with the PDF when it has the same numbers in the same
        order (symbols and spacing - "△：" for "Δ :" - are not what is asked)."""
        if self.s.source != "pdf":
            return ""
        nums = lambda t: re.findall(r"\d+(?:\.\d+)?", t)
        return " (matches the PDF text)" if nums(read) == nums(text) else f" - CHECK: the PDF text says \"{text}\""

    def on_other_pages(self, num):
        """For a multi-page PDF: the other pages that print bridge `num` (so the user can open that page)."""
        pdf = getattr(self.s, "pdf", None)
        if pdf is None:
            return ""
        import pymupdf
        import sheet_objects
        doc = pymupdf.open(pdf)
        pages = []
        for i, page in enumerate(doc):
            if i == self.s.page_no:
                continue
            for m in sheet_objects.BR_RE.finditer(page.get_text()):
                n = sheet_objects.bridge_num(m)
                if n == num or re.fullmatch(re.escape(num) + r"(UP|DN)", n):
                    pages.append(i + 1)
                    break
        if not pages:
            return ""
        p = ", ".join(map(str, pages))
        return f" It is printed on page {p} of this PDF - open that page with --page {pages[0]}."

    def bridges_near_curve(self, num, curves=None):
        """The bridges on a curve (which part of it each is on), else the nearest bridge before and after it. Every
        chainage in the answer is read by the model (the curve's points and each bridge's callout)."""
        objs = self.objects()
        cs = curves or objs.find_curves(num)
        if not cs:
            have = ", ".join(dict.fromkeys(c.block_num or c.num for c in objs.curves))
            return f"No curve {num} is printed on this sheet (curves here: {have}; nothing is guessed)."
        if not objs.bridges:
            return "No bridge callouts are printed on this sheet, so I cannot answer that from it."
        out = []
        for c in cs:
            pts = sorted(c.points, key=lambda p: c.points[p][0])
            reads = {p: self.read_point(c, p, value=True) for p in pts}
            lo, hi = c.span
            lines = [f"{c.name}" + (f" ({c.hand})" if c.hand else "") +
                     f": {'; '.join(reads[p][1] for p in pts)}"]
            on, before, after = [], [], []
            for b in objs.bridges:
                ch = b.fields()["chainage"]
                if ch is None:
                    continue
                (on if lo <= ch <= hi else before if ch < lo else after).append((ch, b))
            for ch, b in on:
                read, _ = self.read_parts(b.best(), f"br_{b.name}")
                f, ok = self.check_fields(b, read, ("chainage",))
                part = "on the curve"
                for a, z in zip(pts, pts[1:]):
                    if c.points[a][0] <= ch <= c.points[z][0]:
                        part = ("on its circular part" if a in ("TPCC1", "TC", "SC") and z in ("TPCC2", "CT", "CS")
                                else "on its transition") + f" ({a} - {z})"
                        break
                lines.append(f"  {b.name} at CH {fmt(f['chainage']) if f['chainage'] is not None else '?'}{ok}: {part}")
            if not on:
                lines.append("  no bridge lies on this curve on this sheet")
            for label, group, key in (("before", before, lambda t: -t[0]), ("after", after, lambda t: t[0])):
                if not group:
                    continue
                near = sorted(group, key=key)
                first = near[0][0]
                for ch, b in (t for t in near if abs(t[0] - first) < 20):   # EX and PROP of the same bridge
                    read, _ = self.read_parts(b.best(), f"br_{b.name}")
                    f, ok = self.check_fields(b, read, ("chainage",))
                    d = (lo - ch) if label == "before" else (ch - hi)
                    lines.append(f"  nearest {label} it: {b.name} at CH {fmt(f['chainage']) if f['chainage'] is not None else '?'}{ok}"
                                 f", {fmt(d)} m {label} the curve")
            out.append("\n".join(lines))
        return "\n\n".join(out)

    def curve_details(self, num, point=None):
        """A curve: where it starts and ends, its circular part, every point as the model read it, its details. With
        point ("TPCC2"), that point first."""
        import sheet_objects
        objs = self.objects()
        cs = objs.find_curves(num)
        if not cs:
            blocks = objs.curve_blocks(num)
            if blocks:                    # its "CURVE No." block only (its points are on another sheet, or not matched)
                read, lab = self.read_parts(blocks[0], f"cvd_{num}")
                m = sheet_objects.DEGREE.search(read)
                return (f"curve {num}: only its details block is on this sheet (its start / end chainages are not)\n"
                        + (f"  degree of curve: {m.group(1)}\n" if m else "")
                        + f"  details (read by the model): {read}" + self.same_numbers(read, lab.text))
            have = ", ".join(dict.fromkeys(c.block_num or c.num for c in objs.curves))
            return f"No curve {num} is printed on this sheet (curves here: {have or 'none'}; nothing is guessed)."
        out = []
        for c in cs:
            pts = sorted(c.points, key=lambda p: c.points[p][0])
            reads = {p: self.read_point(c, p, value=True) for p in pts}
            lines = [f"{c.name}" + (f" ({c.hand})" if c.hand else "") + ":"]
            if point:
                hit = [p for p in pts if p == point or p.rstrip("12") == point]
                lines.append("  " + ("; ".join(reads[p][1] for p in hit) if hit else
                                     f"no {point} of this curve is printed on this sheet"))
            v = {p: reads[p][0] for p in pts}
            start = next((p for p in pts if p in ("TPTC1", "ST1", "TS1")), None)
            end = next((p for p in pts if p in ("TPTC2", "TS2", "ST2")), None)
            c1 = next((p for p in pts if p in ("TPCC1", "TC", "SC")), None)
            c2 = next((p for p in pts if p in ("TPCC2", "CT", "CS")), None)
            if start and end and v[start] is not None and v[end] is not None:
                lines.append(f"  the curve runs from CH {fmt(v[start])} ({start}) to CH {fmt(v[end])} ({end})")
            else:
                lines.append(f"  only part of the curve is on this sheet (it continues on the "
                             f"{'previous' if not start else 'next'} sheet)")
            if c1 and c2 and v[c1] is not None and v[c2] is not None:
                lines.append(f"  its circular part runs from CH {fmt(v[c1])} ({c1}) to CH {fmt(v[c2])} ({c2})")
            lines.append("  points (each read by the model): " + "; ".join(reads[p][1] for p in pts))
            if c.details:
                read, _ = self.read_parts(c.details[0], f"cvd_{c.num}")
                m = sheet_objects.DEGREE.search(read)
                if m:
                    lines.insert(1, f"  degree of curve: {m.group(1)}")
                lines.append("  " + self.curve_summary(c))
            out.append("\n".join(lines))
        return "\n\n".join(out)

    # ------------------------------------------------------------ data bands
    def bands(self):
        """The sheet's data bands (band_table.py), found once."""
        if getattr(self, "_bands", None) is None:
            import band_table
            self._bands = band_table.find_bands(self.s.words)
        return self._bands

    def band_rows(self, q):
        """(rows the question asks about, rows of that kind) - see band_table.match_rows."""
        import band_table
        return band_table.match_rows(self.bands(), q) if self.bands() else ([], [])

    def places(self, q):
        """[(chainage, label)] the question is about: its own chainage numbers (metres or km+m; label None), else the
        chainage printed in the label it names ("cut/fill at C-8 TPCC2" -> "C-8. TPCC2 AT Ch. 11701.654m."). Labels
        tying for the best match each count ("BR NO. 15" -> the EX. and the PROP. bridge, at their own chainages)."""
        # a bridge named in the question: the chainage printed in its callout (all its lines joined)
        nums = bridge_nums(q)
        if nums and self.objects().bridges:
            import sheet_objects
            out = []
            for b in (b for n in dict.fromkeys(nums) for b in self.objects().find_bridges(num=n, status=self.asked_status(q.lower()))):
                ch = b.fields()["chainage"]
                if ch is not None and all(abs(ch - o) > 0.01 for o, _ in out):
                    out.append((ch, sheet_objects.joined(b.best())))
            if out:
                return out[:4]
        # numbers that name something (BR NO. 15, C-8, LC-226, UP Main 2) are not chainages
        bare = re.sub(r"(?i)\b(?:br(?:idge)?|no|c|lc|rob|rub|tbm|bm|curve|sl|sheet|line|main|span)\b\.?\s*(?:no\b\.?)?\s*[:\-.]?\s*\d+",
                      " ", q)
        qn = [n for n in numbers(bare) if n >= 10]
        if qn:
            return [(n, None) for n in qn[:3]]
        m = [(s, w) for s, w in self.matches(q) if s >= 10]       # an identifier or number from the question
        out = []
        for s, w in m:
            if s < m[0][0]:
                break
            c = re.search(r"\bCH(?:AINAGE)?\b\.?\s*[:.]?\s*(\d[\d\s+.,]*)", w.text, re.I)
            ch = numbers(c.group(1)) if c else []
            if ch and all(abs(ch[0] - o) > 0.01 for o, _ in out):
                out.append((ch[0], w))
        return out[:3]

    def band_answer(self, q, on_sheet_only=False):
        """A data-band value at a chainage: the printed column there, or the two columns either side interpolated.
        Every value is read by the model from its own crop (neighbouring columns blanked) and, for a PDF, compared
        with the PDF text. None when the question is not about a band row at a place on this sheet. With
        on_sheet_only, a number outside the band's chainages is left to the caller (it may be a bridge number)."""
        import band_table
        if not self.bands() or not band_table.asks_row(q):
            return None
        rows, about = self.band_rows(q)
        places = self.places(q)
        if on_sheet_only:
            places = [(ch, w) for ch, w in places if any(b.covers(ch) for b in self.bands())]
        if not places:
            return None
        if not about:
            names = "; ".join(r.heading for b in self.bands() for r in b.rows if r.heading)
            return f"This sheet's data band has no row for that. Its rows are: {names} (nothing is guessed)."
        if not rows:
            kinds = "; ".join(dict.fromkeys(r.heading for _, r in about))
            return (f"This sheet's data band has no row for exactly that. Its rows of that kind are: {kinds}. "
                    "Ask about one of those (nothing is guessed).")
        return "\n\n".join(self.band_value(b, r, ch, label) for ch, label in places for b, r in rows[:4])

    def read_value(self, w, row):
        """(value or None, text for the answer): one band value read by the model, checked against the PDF. The crop
        keeps only the value itself: the band's column lines run right beside it and could pass for a minus sign."""
        import band_table
        W, H = self.s.image.size
        region = tuple(int(v) for v in grow(w.box, 0.6 * w.h, W, H))
        img = self.clean_crop([w], region)
        # table rules cross the whole area around the value (the row's own lines run right along its ends, and cut
        # short they look like dashes); a printed number never spans it
        a = np.asarray(img).copy()
        dark = a.min(axis=2) < 200
        a[dark.mean(axis=1) > 0.85, :] = 255
        a[:, dark.mean(axis=0) > 0.85] = 255
        img = Image.fromarray(a)
        # the column's ordinate line runs straight through its value, broken only by the digits, so it touches both
        # ends of the text: keep no margin along the reading direction, a little across it
        ax, ay = abs(w.dir[0]) > 0.5, abs(w.dir[1]) > 0.5
        mx, my = (0 if ax else 0.15 * w.h), (0 if ay else 0.15 * w.h)
        keep = [int(round(v)) for v in (max(0, w.box[0] - mx), max(0, w.box[1] - my), min(W, w.box[2] + mx), min(H, w.box[3] + my))]
        white = Image.new("RGB", img.size, "white")
        white.paste(img.crop((keep[0] - region[0], keep[1] - region[1], keep[2] - region[0], keep[3] - region[1])),
                    (keep[0] - region[0], keep[1] - region[1]))
        crop = pad_canvas(upright(white, w.dir))
        self.save(crop, f"band_{row.heading[:24]}_{w.text}")
        read = self.ask([crop], "What text is written in this crop? Reply with the exact text only.").strip().strip('"').strip()
        m = re.search(r"-?\d+(?:\.\d+)?", read.replace(" ", ""))
        y = float(m.group()) if m else None
        if self.s.source != "pdf":
            return y, read
        true = band_table.value(w.text)
        if y is not None and abs(y - true) < 1e-9:
            return y, f"{m.group()} (matches the PDF text)"
        return y, f"{read or '(nothing read)'} - CHECK: the PDF text says \"{w.text}\""

    def band_value(self, band, row, ch, label):
        import band_table
        vals = band.values(row)
        g = lambda v: f"{v:.3f}".rstrip("0").rstrip(".")                 # chainages to the mm, without trailing zeros
        where = f"CH {g(ch)}" + (f" (\"{label.text}\")" if label else "")
        head = f"{row.heading} at {where}:"
        lo, hi = vals[0][0], vals[-1][0]
        if not lo - 1e-6 <= ch <= hi + 1e-6:
            return f"{head}\n  outside this sheet's band - the row runs from CH {g(lo)} to CH {g(hi)} (nothing is guessed)."
        exact = [(c, w) for c, w in vals if abs(c - ch) < 1e-6]
        if exact:
            c, w = exact[0]
            return f"{head}\n  printed at CH {g(c)}: {self.read_value(w, row)[1]}"
        i = max(k for k, (c, _) in enumerate(vals) if c < ch)
        (c1, w1), (c2, w2) = vals[i], vals[i + 1]
        pdf = self.s.source == "pdf"
        pv = lambda w: band_table.value(w.text) if pdf else None
        dec = max(len(t.split(".")[1]) if "." in t else 0 for t in (w1.text, w2.text))
        brk = self.grade_break(vals, i)
        if brk is None:
            (y1, t1), (y2, t2) = self.read_value(w1, row), self.read_value(w2, row)
            pts = [(c1, y1, t1, pv(w1), ""), (c2, y2, t2, pv(w2), "")]
            lines = [head, f"  not printed at CH {g(ch)} itself; the printed columns either side:"]
        else:
            # the gradient changes at a grade point between the two columns: a straight line from column to column
            # would cut the corner, so the value follows the stretch that holds the chainage, from the grade point
            gch, gw, flw = brk
            if flw is not None:                             # an FL row: the FL printed at the grade point
                read, _ = self.read_label(flw, [flw], f"band_gp_fl_{gw.text}")
                m = FL_RE.search(read) or re.search(r"(-?\d+(?:\.\d+)?)", read)
                yg = float(m.group(1)) if m else None
                true = float(FL_RE.search(flw.text).group(1))
                tg = (f"{m.group(1)}" if m else f"{read or '(nothing read)'}") + \
                     ("" if not pdf else " (FL printed at the grade point, matches the PDF text)" if yg == true
                      else f" - CHECK: the PDF text says \"{flw.text}\"")
                pg = true if pdf else None
            else:                                           # where the stretch on the other side reaches the grade point
                (ca, wa), (cb, wb) = (vals[i - 1], vals[i]) if ch > gch else (vals[i + 1], vals[i + 2])
                (ya, _), (yb, _) = self.read_value(wa, row), self.read_value(wb, row)
                ext = lambda a, b: a + (b - a) / (cb - ca) * (gch - ca)
                yg = ext(ya, yb) if ya is not None and yb is not None else None
                pg = ext(pv(wa), pv(wb)) if pdf else None
                tg = (f"{yg:.{dec}f}" if yg is not None else "? (not read)") + \
                     f" (the stretch through CH {g(ca)} and {g(cb)} carried to the grade point)"
            if ch > gch:
                y2, t2 = self.read_value(w2, row)
                pts = [(gch, yg, tg, pg, " (grade point)"), (c2, y2, t2, pv(w2), "")]
            else:
                y1, t1 = self.read_value(w1, row)
                pts = [(c1, y1, t1, pv(w1), ""), (gch, yg, tg, pg, " (grade point)")]
            lines = [head, f"  not printed at CH {g(ch)} itself, and the gradient changes between the printed columns "
                           f"{g(c1)} and {g(c2)} - at the grade point \"{gw.text}\" - so the value is taken along the "
                           f"stretch that holds CH {g(ch)}:"]
        (xa, ya, ta, pa, na), (xb, yb, tb, pb, nb) = pts
        lines += [f"  x1 = {g(xa)}{na}   y1 = {ta}", f"  x2 = {g(xb)}{nb}   y2 = {tb}"]
        interp = lambda a, b: a + (b - a) / (xb - xa) * (ch - xa)
        misread = pdf and (ya, yb) != (pa, pb) and not (ya is not None and yb is not None and pa is not None
                                                      and abs(ya - pa) < 1e-9 and abs(yb - pb) < 1e-9)
        if ya is None or yb is None:
            lines.append("  cannot interpolate from the model's readings: a value could not be read (see the crops)")
        else:
            lines.append(("  from the model's readings (CHECK - they differ from the PDF): " if misread else "  ") +
                         f"y = y1 + (y2 - y1) / (x2 - x1) * (x - x1) = {ya:g} + ({yb:g} - {ya:g}) / ({g(xb)} - {g(xa)}) * "
                         f"({g(ch)} - {g(xa)}) = {interp(ya, yb):.{dec}f}")
        if misread:
            lines.append(f"  with the PDF's own values: y = {interp(pa, pb):.{dec}f}")
        return "\n".join(lines)

    def grade_break(self, vals, i):
        """(chainage, chainage label, FL label or None) of a grade point strictly between the printed columns i and
        i+1 at which this row bends - the stretches either side, carried to it, meet there while a straight line from
        column to column would not - else None. The FL label is given when the FL printed with the grade point is the
        row's own value there (an FL row)."""
        import band_table
        if i < 1 or i + 2 >= len(vals):
            return None
        (c0, w0), (c1, w1), (c2, w2), (c3, w3) = vals[i - 1:i + 3]
        try:
            v0, v1, v2, v3 = (band_table.value(w.text) for w in (w0, w1, w2, w3))
        except ValueError:
            return None
        tol = 0.0015                                       # the band prints to the mm
        for w in self.s.words:
            if not re.search(r"\bCH\b|CH\s*[:.]", w.text, re.I):
                continue
            ns = [n for n in numbers(w.text) if c1 < n < c2]
            if not ns:
                continue
            gch = ns[0]
            yl = v1 + (v1 - v0) / (c1 - c0) * (gch - c1)
            yr = v2 + (v3 - v2) / (c3 - c2) * (gch - c2)
            straight = v1 + (v2 - v1) / (c2 - c1) * (gch - c1)
            if abs(yl - yr) > tol or abs(straight - (yl + yr) / 2) <= tol:
                continue
            sym = self.symbol(w)
            if not sym:
                continue
            fl = sym["fl"]
            if fl is not None and fl is not w and abs(float(FL_RE.search(fl.text).group(1)) - (yl + yr) / 2) <= tol:
                return gch, w, fl
            return gch, w, None
        return None

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
        band = self.band_sides(ch) if fl else None
        if band:
            return {"ch": ch, "fl": fl, "left": band[0], "right": band[1], "vertex": ch.c}
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

    def band_sides(self, ch):
        """The L-section's gradient band: the grade point is a tick with its chainage / FL written ACROSS the row, and
        each gradient is written once, centred on its own stretch between two ticks ("Rise 1 in 800 (0.125%)" over
        "L= 270"). Left = the stretch ending at this tick, right = the one starting there; the neighbouring ticks bound
        each side, so a stretch further on is never taken. None when the gradients here are not laid out that way: the
        band's ticks stand exactly in line, its gradients are written exactly square to them, and each gradient has the
        stretch's length printed with it ("L= 270"; on a short stretch the gradient is lifted a little above it). A
        profile's own symbols on a flat stretch can line up too, but their gradient labels carry no length."""
        dx, dy = ch.dir
        across = lambda o: (o.c[0] - ch.c[0]) * dx + (o.c[1] - ch.c[1]) * dy
        in_row = lambda o: abs(across(o)) < ch.length / 2 + ch.h
        grads = [o for o in self.band_labels() if abs(o.dir[0] * dx + o.dir[1] * dy) < 0.02 and in_row(o)]
        lengths = [o for o in self.s.words if re.match(r"L\s*=\s*\d", o.text)]
        if not any(math.dist(l.c, g.c) < 3 * g.h for g in grads for l in lengths):
            return None
        u = min(grads, key=lambda o: math.dist(o.c, ch.c)).dir           # along the row, as the gradients read
        t = lambda o: (o.c[0] - ch.c[0]) * u[0] + (o.c[1] - ch.c[1]) * u[1]
        ticks = [t(o) for o in self.s.words if o is not ch and re.search(r"\bCH\b|CH\s*[:.]", o.text, re.I)
                 and o.dir[0] * dx + o.dir[1] * dy > 0.999 and abs(t(o)) > ch.h
                 and min(abs((p[0] - q[0]) * dx + (p[1] - q[1]) * dy)        # in line: start, middle or end
                         for p, q in ((o.start, ch.start), (o.c, ch.c), (o.end, ch.end))) < 0.5 * ch.h]
        if not ticks:
            return None
        lo = max([x for x in ticks if x < 0], default=-200 * ch.h)
        hi = min([x for x in ticks if x > 0], default=200 * ch.h)
        left = max((o for o in grads if lo < t(o) < 0), key=t, default=None)
        right = min((o for o in grads if 0 < t(o) < hi), key=t, default=None)
        return (left, right) if left or right else None

    def band_labels(self):
        """Every text that can be a gradient in the band, as printed: the usual forms, the band's abbreviations, and a
        label stacked over lines ("Fall 1 in" / "530" / "(-0.189%)") joined into one."""
        if getattr(self, "_band_labels", None) is None:
            words = self.s.words
            out = [o for o in words if GRADIENT.match(o.text) or BAND_GRADIENT.match(o.text)]
            for head in words:
                if not re.fullmatch(r"(?i)(rise|fall|[RF])\s*1\s*in", head.text):
                    continue
                lines = [head]
                for pattern in (r"[\d.,]+", r"\(\s*-?[\d.]+\s*%\s*\)"):      # the number, then the percentage
                    last = lines[-1]
                    below = (-last.dir[1], last.dir[0])
                    nxt = [o for o in words if re.fullmatch(pattern, o.text) and o.dir[0] * last.dir[0] + o.dir[1] * last.dir[1] > 0.999
                           and 0.5 * last.h < (o.c[0] - last.c[0]) * below[0] + (o.c[1] - last.c[1]) * below[1] < 1.8 * last.h
                           and abs((o.c[0] - last.c[0]) * last.dir[0] + (o.c[1] - last.c[1]) * last.dir[1]) < 2 * last.h]
                    if not nxt:
                        break
                    lines.append(min(nxt, key=lambda o: math.dist(o.c, last.c)))
                if len(lines) > 1:
                    out.append(Word(" ".join(o.text for o in lines), union([o.box for o in lines]), head.dir, head.h))
            self._band_labels = out
        return self._band_labels

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
        crop = pad_canvas(upright(self.clean_crop(region_keep, box), word.dir))
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
            crop = pad_canvas(upright(self.clean_crop(block, b), ch.dir))
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
