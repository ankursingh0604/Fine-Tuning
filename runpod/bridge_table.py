"""The "DETAILS OF BRIDGES" table some L-section sheets print beside the profile, read from where its text is:

    S.No | Bridge No | Type of Crossing | Chainage | Type of Bridge (DN Line, 4th Line) | Top Slab Thickness |
    Formation Level (DN Line, 4th Line) | CHFL | OHFL | Bed Level/Road Level | Free Board (..) | Clearance (..) |
    Earth Cushion (..) | Discharge | Remarks

    rows = bridge_table.read(sheet.words)        # {bridge no ("259DN"): {column key: [Word, ...]}}

Columns are found from the headings (a heading over two sub-columns - "Type of Bridge" over "DN Line" and "4th Line" -
gives two columns: the existing line's, then the proposed line's); a value belongs to the column whose heading it is
printed under. Rows are the S.No numbers; a value printed over two lines ("1 x 2 x 2.75-Minor-RCC" / "Box") is one.
"""
import re

TITLE = re.compile(r"DETAILS\s+OF\s+BRIDGES", re.I)
# (key, heading words) - single columns; the groups below have an existing and a proposed sub-column
# ("S.No", or "S.N" printed over "o")
SINGLE = [("sno", r"^S\.?\s*N(?:O|\.)?\.?$|^S\.?\s*NO\b"), ("br", r"^BRIDGE$|^BRIDGE\s+NO\b"),
          ("crossing", r"^TYPE\s+OF$|CROSSING|^DESCRIPTION$"), ("chainage", r"^CHAINAGE$"),
          ("slab", r"^TOP\s+SLAB"), ("chfl", r"^CHFL$"), ("ohfl", r"^OHFL$"), ("bed", r"^BED$|^BED\s+LEVEL|LEVEL\s*/\s*ROAD"),
          ("hc", r"^HC$"), ("vc", r"^VC$"), ("discharge", r"^DISCH(?:ARGE)?$"), ("remarks", r"^REMARKS?$")]
UNIT = re.compile(r"\s*\((?:m|mm|cumecs?|cume)\)\s*$", re.I)        # "Chainage (m)", "HC(M)", "Top Slab Thickness (mm)"
# a table without a title is found by its "Type of Bridge" heading (5.pdf, abc.pdf: the same table, other headings)
ANCHOR = re.compile(r"^TYPE\s+OF\s+BRIDGE$", re.I)
GROUPS = [("type", r"^TYPE\s+OF\s+BRIDGE$"), ("fl", r"^FORMATION\s+LEVEL$"), ("fb", r"^FREE\s+BOARD$"), ("clearance", r"^CLEARANCE$"),
          ("cushion", r"^EARTH\s+CUSHION$")]
SUB = re.compile(r"^(?:DN|UP|DOWN|\d\s*(?:ST|ND|RD|TH))(?:\s+LINE)?$", re.I)      # ("4th Line", or "4th" over "Line")


class Piece:
    """Part of a printed text item that runs over several columns ("227.221   227.294": the CHFL and the OHFL)."""

    def __init__(self, w, text, a, b):
        cw = (w.box[2] - w.box[0]) / max(1, len(w.text))
        self.text, self.h, self.dir, self.word = text, w.h, w.dir, w
        self.box = (w.box[0] + a * cw, w.box[1], w.box[0] + b * cw, w.box[3])
        self.c = ((self.box[0] + self.box[2]) / 2, w.c[1])
        self.length = self.box[2] - self.box[0]          # (read like a Word: find_crop.strip_crop)


def pieces(w, centres=()):
    """The item split into the columns it runs over: at its wide gaps (two spaces or more), and - when it is wider than
    one column (its box holds two column centres or more: "0.41 257.9", the slab and the FL) - word by word, words
    under the same column kept together. An item within one column is itself."""
    across = sum(1 for x in centres if w.box[0] < x < w.box[2])
    if across < 2 and not re.search(r"\S\s{2,}\S", w.text):
        return [w]
    if across < 2:
        return [Piece(w, w.text[a:b], a, b) for a, b in ((m.start(), m.end()) for m in re.finditer(r"\S+(?:\s\S+)*", w.text))]
    cw = (w.box[2] - w.box[0]) / max(1, len(w.text))
    out, cur = [], None
    for m in re.finditer(r"\S+", w.text):
        x = w.box[0] + (m.start() + m.end()) / 2 * cw
        col = min(range(len(centres)), key=lambda k: abs(centres[k] - x))
        if cur and cur[0] == col:
            cur[2] = m.end()
        else:
            cur = [col, m.start(), m.end()]
            out.append(cur)
    return [Piece(w, w.text[a:b], a, b) for _, a, b in out]


def grid(page, zoom, x0, x1, y0, y1):
    """The x of the table's vertical rules (in the words' units: PDF points x zoom) that run through the rows' area."""
    cover = {}
    for d in page.get_drawings():
        for it in d["items"]:
            segs = []
            if it[0] == "l" and abs(it[1].x - it[2].x) < 0.5:
                segs.append((it[1].x, it[1].y, it[2].y))
            elif it[0] == "re":
                r = it[1]
                segs += [(r.x0, r.y0, r.y1), (r.x1, r.y0, r.y1)]
            for x, ya, yb in segs:
                x, ya, yb = x * zoom, min(ya, yb) * zoom, max(ya, yb) * zoom
                if x0 - 2 <= x <= x1 + 2 and yb > y0 and ya < y1:
                    k = round(x / 2)
                    cover[k] = cover.get(k, 0) + min(yb, y1) - max(ya, y0)
    xs = sorted(k * 2 for k, v in cover.items() if v >= 0.5 * (y1 - y0))
    out = []
    for x in xs:
        if not out or x - out[-1] > 4:
            out.append(x)
    return out


def split_at(w, rules):
    """A text item that runs over a column rule ("0.41 257.9": the slab and the FL), split word by word at it."""
    inner = [x for x in rules if w.box[0] + 1 < x < w.box[2] - 1]
    if not inner:
        return [w]
    cw = (w.box[2] - w.box[0]) / max(1, len(w.text))
    out, cur = [], None
    for m in re.finditer(r"\S+", w.text):
        x = w.box[0] + (m.start() + m.end()) / 2 * cw
        col = sum(1 for r in inner if r < x)
        if cur and cur[0] == col:
            cur[2] = m.end()
        else:
            cur = [col, m.start(), m.end()]
            out.append(cur)
    return [Piece(w, w.text[a:b], a, b) for _, a, b in out]


def read(words, page=None, zoom=1.0):
    """{bridge no without spaces, upper case: {key: [Word]}} for the sheet's DETAILS OF BRIDGES table ({} if none).
    page / zoom: the PDF page and the words' scale (sheet dpi / 72), for the table's drawn column rules.
    Keys: sno, br, crossing, chainage, type_ex, type_pr, slab, fl_ex, fl_pr, chfl, ohfl, bed, fb_ex, fb_pr,
    clearance_ex, clearance_pr, cushion_ex, cushion_pr, discharge, remarks."""
    title = next((w for w in words if TITLE.search(w.text)), None)
    if title is not None:
        h = title.h
        top, bottom = title.box[1] - h, title.box[3] + 6 * h
    else:
        title = next((w for w in words if ANCHOR.match(w.text.strip()) and abs(w.dir[0]) > 0.9), None)
        if title is None:
            return {}
        h = title.h
        top, bottom = title.box[1] - 2 * h, title.box[3] + 5 * h
    # (two sub-headings can be one text item, "DN Line   4th Line": split at the wide gap)
    near = [p for w in words if top <= w.c[1] <= bottom and abs(w.dir[0]) > 0.9 and abs(w.c[0] - title.c[0]) < 160 * h for p in pieces(w)]
    head = lambda w: UNIT.sub("", w.text.strip())
    cols = {}
    for key, pat in SINGLE:
        hits = [w for w in near if re.search(pat, head(w), re.I)]
        if hits:
            cols.setdefault(key, min(hits, key=lambda w: w.c[1]).c[0])
    groups = [(key, w) for key, pat in GROUPS for w in near if re.search(pat, head(w), re.I)]
    subs = [w for w in near if SUB.match(w.text.strip())]
    for key, g in groups:
        mine = sorted((s for s in subs if s.c[1] > g.c[1] and min(groups, key=lambda t: abs(t[1].c[0] - s.c[0]))[1] is g),
                      key=lambda s: s.c[0])
        for s in mine:
            existing = bool(re.match(r"^(?:DN|UP|DOWN)\b", s.text.strip(), re.I))
            cols[f"{key}_{'ex' if existing else 'pr'}"] = s.c[0]
    if "sno" not in cols or "br" not in cols:
        return {}
    head_bottom = max(w.box[3] for w in near if any(abs(w.c[0] - x) < 1 for x in cols.values())) if cols else title.box[3]
    x0 = min(cols.values()) - 4 * h
    x1 = max(cols.values()) + 4 * h
    # rows: the serial numbers under S.No, each row from half way to the one before to half way to the next
    sno = sorted((w for w in words if w.c[1] > head_bottom and abs(w.c[0] - cols["sno"]) < 1.5 * h and re.fullmatch(r"\d{1,3}", w.text.strip())
                  and abs(w.dir[0]) > 0.9), key=lambda w: w.c[1])
    # stop where the serials stop following on (1, 2, 3 ...): what is further down is not this table
    run = []
    for w in sno:
        if run and (int(w.text) != int(run[-1].text) + 1 or w.c[1] - run[-1].c[1] > 8 * h):
            break
        run.append(w)
    if not run:
        return {}
    ys = [w.c[1] for w in run]
    step = (ys[-1] - ys[0]) / (len(ys) - 1) if len(ys) > 1 else 3 * h
    bands = [((ys[i - 1] + y) / 2 if i else y - step / 2, (y + ys[i + 1]) / 2 if i + 1 < len(ys) else y + step / 2) for i, y in enumerate(ys)]
    # the columns: between the table's drawn vertical rules when the page is given (a long or wrapped value then stays
    # in its cell, and text that runs over a rule is split at it), else around the headings
    rules = grid(page, zoom, x0, x1, bands[0][0], bands[-1][1]) if page is not None else []
    cells = []                                         # (left, right, key)
    for a, b in zip(rules, rules[1:]):
        keys = [k for k, x in cols.items() if a < x < b]
        if keys:
            cells.append((a, b, min(keys, key=lambda k: abs(cols[k] - (a + b) / 2))))
    if len(cells) < len(cols) - 2:                     # (no usable rules: the headings' centres)
        cells = []
    centres = sorted(cols.values())
    inside = [p for w in words if x0 <= w.c[0] <= x1 and bands[0][0] <= w.c[1] <= bands[-1][1] and abs(w.dir[0]) > 0.9
              for p in (split_at(w, [a for a, _, _ in cells[1:]]) if cells else pieces(w, centres))]

    def key_of(w):
        if cells:
            hit = next((k for a, b, k in cells if a <= w.c[0] < b), None)
            return hit
        return min(cols, key=lambda k: abs(cols[k] - w.c[0]))
    out = {}
    for (a, b) in bands:
        row = {}
        for w in sorted((w for w in inside if a <= w.c[1] < b), key=lambda w: (w.c[1], w.c[0])):
            key = key_of(w)
            if key:
                row.setdefault(key, []).append(w)
        br = " ".join(w.text for w in row.get("br", []))
        if br.strip():
            out[re.sub(r"\s", "", br).upper()] = row
    return out


def text(cell):
    """A cell's printed text (its lines joined)."""
    return re.sub(r"\s+", " ", " ".join(w.text for w in cell or [])).strip()


ONE = r"\d+\s*[xX×]\s*[\d.]+(?:\s*[xX×]\s*[\d.]*)?"       # "1 x 3.66", "1 x 4 x 4.7", "1 x 3.66 x" (height not printed)
TYPE = re.compile(rf"^\s*(?P<conf>{ONE}(?:\s*\+\s*{ONE})*)\s*-\s*(?P<cat>MINOR|MAJOR|IMPORTANT|RUB|ROB|LHS|RUB/LHS)?\s*-?\s*(?P<st>.+)$", re.I)


def bridge_type(t):
    """'1 x 3.66 x -Minor-PSC slab' -> ('1 x 3.66 x', 'MINOR', 'PSC SLAB'); 'Existing structure is suficient' ->
    (None, None, None) - not a bridge type."""
    m = TYPE.match(t or "")
    if not m:
        return None, None, None
    return m.group("conf").strip(), (m.group("cat") or "").upper() or None, re.sub(r"\s+", " ", m.group("st")).strip(" -").upper()
