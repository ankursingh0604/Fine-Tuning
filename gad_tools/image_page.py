"""A scanned GAD (a PDF page with no text layer, or a PNG / JPG photo of the sheet) as a page annotate_gad.py can read.

annotate_gad reads a vector PDF through its page: the text lines with their boxes, the words, the line work. ImagePage
gives the same from a picture:
  text   the local OCR (RapidOCR: CPU, offline, pip install rapidocr_onnxruntime), run on overlapping 1600 px tiles - a
         whole A1 sheet shrunk to the OCR's input size would lose its small text; tall boxes are upright text
  lines  the drawing's ink: runs of dark pixels, as short line segments (enough to find where each view's line work is)
Coordinates are in PDF points (1/72 inch), as on a vector page, so everything downstream is unchanged. The OCR reading
is cached next to the image (<name>.ocr.json): a sheet is read once (a minute or two on a CPU).
"""
import json
import math
from pathlib import Path

import numpy as np

DPI = 250              # OCR resolution: 2.5 mm text on an A1 sheet is ~25 px tall
BIG_PT = 11                             # text this size or larger (titles) is taken from the half-size pass
TILE, STEP = 1600, 1150                 # 450 px overlap: a big title cut by one tile's edge is whole in the next


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.width, self.height = x1 - x0, y1 - y0


def _ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise SystemExit("this GAD is an image (no text layer): the local OCR is needed - pip install rapidocr_onnxruntime")
    return RapidOCR()


def _quad(text, p):
    """(text, bbox, dir, height) from an OCR quadrilateral: the text runs along the long side; a tall box is upright text,
    which on these drawings reads bottom to top."""
    e01 = (p[1][0] - p[0][0], p[1][1] - p[0][1])
    e03 = (p[3][0] - p[0][0], p[3][1] - p[0][1])
    l01, l03 = math.hypot(*e01) or 1, math.hypot(*e03) or 1
    if l01 >= l03:
        d, h = (1, 0) if abs(e01[0]) >= abs(e01[1]) else (0, -1), l03
    else:
        d, h = (0, -1), l01
    box = (min(q[0] for q in p), min(q[1] for q in p), max(q[0] for q in p), max(q[1] for q in p))
    return text, box, d, h


# ---------------------------------------------------------------- OCR clean-up
VOCAB_FILE = Path(__file__).with_name("gad_vocab.txt")
_vocab = None


def vocab():
    """The words printed on the GADs (gad_vocab.txt, made from the vector annotations by build_vocab)."""
    global _vocab
    if _vocab is None:
        _vocab = set(VOCAB_FILE.read_text(encoding="utf-8").split()) if VOCAB_FILE.exists() else set()
    return _vocab


def build_vocab(ann_dir):
    """gad_vocab.txt: every capital word (2+ letters) printed on the annotated GADs - names never reach the annotations."""
    import re
    words = set()
    for f in Path(ann_dir).glob("*.json"):
        words |= set(re.findall(r"\b[A-Z][A-Z]+\b", f.read_text(encoding="utf-8")))
    words |= {"OF", "TO", "AT", "AS", "BY", "ON", "IN", "OR", "NO", "UP", "DN"}
    # the reader's own headings and view words (headings are not stored in the annotations)
    import gad_kinds, annotate_gad
    for pat in list(annotate_gad.HEADINGS.values()) + [v[0] for v in gad_kinds.VIEWS]:
        words |= set(re.findall(r"[A-Z]{2,}", pat))
    words |= {"DESCRIPTION", "EXISTING", "PROPOSED", "BRIDGE", "SR", "NO", "SCALE"}
    VOCAB_FILE.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    return len(words)


def split_words(text):
    """Capital words the OCR ran together ("TYPEOFSUPERSTRUCTURE", "DETAILSOFDROPWALL") split into GAD words - only when
    the whole run divides into known words (fewest pieces); anything else is left as read."""
    import re
    V = vocab()
    if not V:
        return text

    def seg(w):
        n = len(w)
        best = [None] * (n + 1)
        best[0] = []
        for i in range(n):
            if best[i] is None:
                continue
            for j in range(i + 1, min(n, i + 20) + 1):
                piece = w[i:j]
                ok = piece in V if len(piece) > 1 else piece in "ABCXY"      # a section letter: "...AT A-A"
                if ok and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                    best[j] = best[i] + [w[i:j]]
        return best[n]

    def fix(m):
        w = m.group(0)
        if w in V or len(w) < 6:
            return w
        parts = seg(w)
        return " ".join(parts) if parts and len(parts) > 1 else w
    return re.sub(r"[A-Z]{6,}", fix, text)


def _consecutive(s):
    """'23' -> [2, 3], '1011' -> [10, 11]: the digits as two or more consecutive numbers, or None."""
    for first in range(1, min(3, len(s)) + 1):
        a, out, rest = int(s[:first]), [int(s[:first])], s[first:]
        while rest:
            nxt = str(out[-1] + 1)
            if not rest.startswith(nxt):
                break
            out.append(int(nxt))
            rest = rest[len(nxt):]
        if not rest and len(out) > 1:
            return out
    return None


def _split_stacked(text, box, d, h):
    """Row numbers of a table, which the OCR reads as upright text: a column of them ("11 10 9 8 7 6 5 4"), two rows run
    together ("-23" = 2 above 3), or one number in a box taller than wide ("1"). Returned as level numbers, one per row,
    top to bottom - the table reader needs where each row's number is, so a misread "-6" for 16 still marks its row."""
    import re
    # row numbers only: whole numbers of 1-2 digits (an upright level "391.412" or chainage "56373" is a value, left alone)
    # ("13." - a note number - keeps its point; "391.412" has 3-digit groups and is left alone below)
    if d != (0, -1) or not re.fullmatch(r"[\d\s\-.]+", text) or not re.search(r"\d", text) or re.search(r"\d\.\d", text) \
            or any(len(g) > 4 for g in re.findall(r"\d+", text)) or (len(re.findall(r"\d+", text)) == 1 and len(re.sub(r"\D", "", text)) > 2
                                                                    and not _consecutive(re.sub(r"\D", "", text))):
        return None
    x0, y0, x1, y1 = box
    w = max(x1 - x0, 1e-6)
    toks = re.findall(r"\d+", text)
    if len(toks) > 1 and any(len(t) > 2 for t in toks):   # "393 306": a value read with a space for its point
        return None
    nums = sorted(int(t) for t in toks) if len(toks) > 1 else None
    if nums is None and (y1 - y0) / w > 1.8:
        nums = _consecutive(re.sub(r"\D", "", text))
    if nums is None:                                      # one number: plain text
        return [(re.sub(r"\D", "", text), box, (1, 0), min(y1 - y0, w * 1.4))]
    step = (y1 - y0) / len(nums)
    return [(str(n), (x0, y0 + k * step, x0 + w, y0 + (k + 1) * step), (1, 0), min(step, w * 1.4)) for k, n in enumerate(nums)]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _drop_partial(lines):
    """A piece of text read where a tile's edge cut it ("ECTION") lies inside the whole reading from the next tile:
    keep the longer one."""
    keep = []
    for l in sorted(lines, key=lambda l: -len(l["text"])):
        b = l["bbox"]
        if any(_area([max(b[0], o["bbox"][0]), max(b[1], o["bbox"][1]), min(b[2], o["bbox"][2]), min(b[3], o["bbox"][3])])
               > 0.6 * max(_area(b), 1e-6) for o in keep):
            continue
        keep.append(l)
    return keep


def _whole_lines(lines):
    """One printed line the OCR read as pieces ("BORE LOG AT CH:6" + "5288.80"): pieces on the same line, a word gap
    apart, joined. Table cells, further apart, stay separate texts as on a vector page."""
    def along(l):
        return (l["bbox"][0], l["bbox"][2]) if l["dir"] == [1, 0] else (-l["bbox"][3], -l["bbox"][1])

    def across(l):
        return (l["bbox"][1] + l["bbox"][3]) / 2 if l["dir"] == [1, 0] else (l["bbox"][0] + l["bbox"][2]) / 2
    out = []
    for l in sorted(lines, key=lambda l: (l["dir"] != [1, 0], round(across(l) / 3), along(l)[0])):
        prev = next((o for o in reversed(out[-6:]) if o["dir"] == l["dir"] and abs(across(o) - across(l)) < 0.35 * max(o["size"], l["size"])
                     and 0 <= along(l)[0] - along(o)[1] < 0.9 * max(o["size"], l["size"])), None)
        if prev:
            prev["text"] = prev["text"] + " " + l["text"]
            b, c = prev["bbox"], l["bbox"]
            prev["bbox"] = [min(b[0], c[0]), min(b[1], c[1]), max(b[2], c[2]), max(b[3], c[3])]
            prev["size"] = max(prev["size"], l["size"])
        else:
            out.append(dict(l))
    return out


class ImagePage:
    def __init__(self, img, cache=None, log=print):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        self.img = img.convert("RGB")
        self.z = 72.0 / DPI                                   # pixels -> points
        W, H = self.img.size
        self.rect = _Rect(0, 0, W * self.z, H * self.z)
        self.rotation = 0
        self.lines = self._read(cache, log)
        self._drawings = None

    def remove_rotation(self):
        pass

    # ---------------------------------------------------------------- text
    def _read(self, cache, log):
        """The raw OCR passes are cached (they take minutes); the clean-up below runs on every load."""
        raw = json.loads(Path(cache).read_text(encoding="utf-8")) if cache and Path(cache).exists() else None
        if not isinstance(raw, dict) or "full" not in raw:
            eng = _ocr_engine()
            log("  reading the sheet's text with the local OCR (once, a few minutes) ...")
            half = self.img.resize((self.img.width // 2, self.img.height // 2))
            raw = {"full": self._pass(eng, self.img, 1.0), "half": self._pass(eng, half, 2.0)}
            if cache:
                Path(cache).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        import unicodedata
        # the OCR box is tighter than the font: sizes calibrated on 810-1 (~590 texts matched to the vector PDF), so
        # titles (~15 pt) and labels (~7-9 pt) come out at their printed sizes
        cal = {("full", True): 1.25, ("full", False): 1.11, ("half", True): 0.97, ("half", False): 0.89}

        def fix(l, which):
            l = dict(l)
            l["size"] = round(l["size"] * cal[(which, l["dir"] == [1, 0])], 1)
            l["text"] = unicodedata.normalize("NFKC", l["text"])          # the OCR's full-width "（SCALE 1:100）"
            return l
        raw = {"full": [fix(l, "full") for l in raw["full"]], "half": [fix(l, "half") for l in raw["half"]]}
        found = []
        for l in raw["full"]:                                  # row numbers read as upright text (also on cached readings)
            parts = _split_stacked(l["text"], l["bbox"], tuple(l["dir"]), l["size"])
            if parts:
                found += [{"text": t, "bbox": list(b), "dir": list(d), "size": round(h * 0.8, 1)} for t, b, d, h in parts]
            else:
                found.append(dict(l))
        # big titles (view names, 14-17 pt) are wider than a tile's overlap and come cut in two: the half-size pass reads
        # them whole and replaces the cut pieces - big text only (level labels, ~7-10 pt, are read better at full size)
        big = [dict(l) for l in raw["half"] if l["size"] >= BIG_PT]
        found = [l for l in found if not any(_area([max(l["bbox"][0], b["bbox"][0]), max(l["bbox"][1], b["bbox"][1]),
                                                    min(l["bbox"][2], b["bbox"][2]), min(l["bbox"][3], b["bbox"][3])])
                                             > 0.5 * max(_area(l["bbox"]), 1e-6) for b in big)] + big
        found = _whole_lines(_drop_partial(found))
        for l in found:
            l["text"] = split_words(l["text"])
        return found

    def _pass(self, eng, img, k):
        """OCR of img in overlapping full-size tiles (the last one flush with the edge: a thin edge strip, scaled up by the
        OCR, ran out of memory); k = original pixels per pixel of img."""
        W, H = img.size
        starts = lambda n: sorted(set(list(range(0, max(1, n - TILE), STEP)) + [max(0, n - TILE)]))
        found = []
        for y in starts(H):
            for x in starts(W):
                tile = np.asarray(img.crop((x, y, min(W, x + TILE), min(H, y + TILE))))
                if (tile.min(axis=2) < 140).mean() < 0.002:      # blank paper: nothing to read (most of an A1 sheet)
                    continue
                res, _ = eng(tile)
                for quad, text, score in res or []:
                    if not text.strip() or float(score) < 0.5:
                        continue
                    p = [((float(px) + x) * k, (float(py) + y) * k) for px, py in quad]
                    t, box, d, h = _quad(text.strip(), p)
                    for t, box, d, h in [(t, box, d, h)]:    # (kept raw: row numbers are split at clean-up time)
                        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        if any(o["text"] == t and math.dist(o["_c"], c) < 12 * k for o in found):      # the tiles overlap
                            continue
                        found.append({"text": t, "bbox": [v * self.z for v in box], "dir": list(d), "size": round(h * self.z * 0.8, 1),
                                      "_c": c})
        for f in found:
            f.pop("_c")
        return found

    def get_text(self, kind="text"):
        if kind == "dict":
            return {"blocks": [{"lines": [{"spans": [{"text": l["text"], "size": l["size"], "color": 0}], "bbox": l["bbox"],
                                           "dir": tuple(l["dir"])}]} for l in self.lines]}
        if kind == "words":                                  # each line's words, spaced along it in proportion
            out = []
            for i, l in enumerate(self.lines):
                x0, y0, x1, y1 = l["bbox"]
                ws, t = l["text"].split(), l["text"]
                pos = 0
                for j, w in enumerate(ws):
                    a = t.index(w, pos) / max(1, len(t))
                    b = (t.index(w, pos) + len(w)) / max(1, len(t))
                    pos = t.index(w, pos) + len(w)
                    if tuple(l["dir"]) == (1, 0):
                        out.append((x0 + a * (x1 - x0), y0, x0 + b * (x1 - x0), y1, w, i, 0, j))
                    else:                                    # bottom to top
                        out.append((x0, y1 - b * (y1 - y0), x1, y1 - a * (y1 - y0), w, i, 0, j))
            return out
        return "\n".join(l["text"] for l in self.lines)

    # ---------------------------------------------------------------- ink
    def get_drawings(self):
        """The ink as horizontal and vertical runs of dark pixels at 1 pt resolution (a pixel is dark where any pixel of its
        block is: thin lines survive the reduction)."""
        if self._drawings is not None:
            return self._drawings
        g = np.asarray(self.img.convert("L"), dtype=np.uint8)
        f = max(1, round(1 / self.z))                         # pixels per point
        H, W = g.shape[0] // f, g.shape[1] // f
        dark = g[:H * f, :W * f].reshape(H, f, W, f).min(axis=(1, 3)) < 140
        items = []
        for axis in (1, 0):
            arr = dark if axis == 1 else dark.T
            pad = np.zeros((arr.shape[0], 1), dtype=bool)
            d = np.diff(np.hstack([pad, arr, pad]).astype(np.int8), axis=1)
            for r, s in zip(*np.nonzero(d == 1)):
                e = np.argmax(d[r, s:] == -1) + s
                if e - s < 2:
                    continue
                if axis == 1:
                    items.append(("l", _Pt(float(s), float(r)), _Pt(float(e), float(r))))
                else:
                    items.append(("l", _Pt(float(r), float(s)), _Pt(float(r), float(e))))
        self._drawings = [{"items": items, "rect": _Rect(0, 0, 1, 1), "dashes": None}]
        return self._drawings


def open_image_page(path, log=print):
    """A PNG / JPG of a sheet, or a PDF page with no text layer, as an ImagePage."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pymupdf
        page = pymupdf.open(path)[0]
        if page.rotation:
            page.remove_rotation()
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(path)
        dpi = (img.info.get("dpi") or (DPI, DPI))[0] or DPI
        if abs(dpi - DPI) > 5:                                # bring the scan to the OCR resolution
            img = img.resize((round(img.width * DPI / dpi), round(img.height * DPI / dpi)))
    return ImagePage(img, cache=str(path) + ".ocr.json", log=log)
