"""Annotate RCC box GADs (General Arrangement Drawings) from the PDF text layer.

    .venv\\Scripts\\python gad_tools\\annotate_gad.py                 # every PDF in GAD/ -> data/gad/annotations/<gad_id>.json
    .venv\\Scripts\\python gad_tools\\annotate_gad.py --pdf FILE.pdf   # one file, printed (used by the CLI for an uploaded GAD)

Every GAD here is a single-page vector PDF, so values are read exactly from the text layer:

    views        title, kind, scale, region (found from the drawing's own line work), what the view represents
    levels       every labelled level (RAIL LVL, SOFFIT LVL, HFL, FOUNDING LVL ...) with its meaning, proposed/existing,
                 the element it belongs to, and the view it is in
    dims         labelled dimensions (T/C track centres, BARREL LENGTH, "150 THK WEARING COURSE", weep hole diameter,
                 gaps, slopes) with their meaning; other dimension figures with their view and direction only
    centre_lines the track centre lines on the views (existing UP / DN, proposed 3rd line)
    tables       comparative table (existing vs proposed), track details, depth of track structure, bore logs
    notes        notes, special note, fill note, design criteria, specifications, reference drawings, abbreviations,
                 legends, seismic zone, standard of loading
    title_block  division, section, chainage, drawing numbers, revision, date, title (bridge, box size, location)
    findings     values that disagree between places on the same drawing (e.g. box size in the title vs the table)

The signature block (names of officials) is not read.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gad_kinds as K       # noqa: E402

GAD_DIR = ROOT / "GAD"
OUT = ROOT / "data" / "gad" / "annotations"
CELL = 6.0                                     # occupancy grid cell (pt) for finding view regions

SCALE_RE = re.compile(r"^\(?\s*SCALE\s*[:\-]?\s*(?:1\s*:\s*(\d+)|(N\.?\s*T\.?\s*S\.?|NOT\s+TO\s+SCALE))\s*\)?\.?$", re.I)
TITLE_WITH_SCALE = re.compile(r"^(?P<t>.+?)\s*\(\s*SCALE\s*1\s*:\s*(?P<s>\d+)\s*\)$", re.I)
HEADINGS = {
    "notes": r"^NOTES?\s*:?$",
    "special_note": r"^SPECIAL\s+NOTES?\s*:?$",
    "add_note": r"^ADD(ITIONAL)?\s+NOTES?\s*:?$",
    "design_criteria": r"^DESIGN\s+CRITERIA\s*:?$",
    "specifications": r"^SPECIFICATIONS?\s*:?$",
    "fill_note": r"^FILL\s+NOTES?\s*:?$",
    "reference_drawings": r"^REFERENCE\s+DRAWINGS?\s*:?$",
    "abbreviations": r"^ABBREVIATIONS?\s*:?$",
    "legends": r"^LEGENDS?\s*:?$",
    "comparative_table": r"^((COMPARATIVE|HYDRAULIC)\s+TABLE|HYDRAULIC\s+DATA|COMPARATIVE\s+STATEMENT)\s*:?$",
    "track_details": r"^TRACK\s+DETAILS?\s*:?$",
    "depth_of_track_structure": r"^DEPTH\s+OF\s+TRACK\s+STRUCTURE\s*:?$",
}
# (the railway's name alone starts the title block; "CENTRAL RAILWAY LETTER NO. ..." inside a note does not)
STOPS = r"^(SEISMIC\s+ZONE\s*:|STANDARD\s+OF\s+LOADING\s*:|(WEST\s+)?CENTRAL\s+RAILWAY\s*$|SIGNATURE\s+BLOCK\b|DIVISION\b|RAILWAY\s+OFFICIALS\b)"
LEVEL_WORD = re.compile(r"\b(LVL|LEVEL|HFL|H\.F\.L|OHFL|CHFL|SOFFIT|RL|R\.L|FDN|FND|FOUND|FOUNDING|INVERT|FORMATION|BED)\b", re.I)
LEVEL_NUM = re.compile(r"(?<![\d.])(\d{2,4}\.\d{2,3})(?![\d.])")
DIM_NUM = re.compile(r"^\d{2,5}(?:\.\d{1,3})?$")
SIGNATURE = re.compile(r"Digitally signed|^Date\s*:\s*20\d\d|Signature block", re.I)


# ======================================================================== text
def load_lines(page):
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = re.sub(r"\s+", " ", " ".join(s["text"].strip() for s in spans)).strip()
            d = (int(round(l["dir"][0])), int(round(l["dir"][1])))
            lines.append({"text": text, "bbox": [round(v, 1) for v in l["bbox"]], "size": round(max(s["size"] for s in spans), 1),
                          "dir": d})
    for i, l in enumerate(lines):
        l["i"] = i
    return lines


def rbox(bbox, d):
    """bbox in the line's reading frame: x along the text, y downwards relative to the letters."""
    x0, y0, x1, y1 = bbox
    if d == (0, -1):                       # reads bottom -> top: letters' tops point to -x
        pts = [(-y0, x0), (-y1, x1)]
    elif d == (0, 1):                      # reads top -> bottom: letters' tops point to +x
        pts = [(y0, -x0), (y1, -x1)]
    else:
        pts = [(x0, y0), (x1, y1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def inside(pt, box, pad=0):
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def neighbour(lines, line, up=True, max_gap=14, need=None):
    """The nearest line directly above (or below) in the reading frame, same direction, overlapping along the text."""
    rb = rbox(line["bbox"], line["dir"])
    best = None
    for o in lines:
        if o is line or o["dir"] != line["dir"] or (need and not need(o)):
            continue
        ob = rbox(o["bbox"], o["dir"])
        if min(rb[2], ob[2]) - max(rb[0], ob[0]) < -4:
            continue
        g = rb[1] - ob[3] if up else ob[1] - rb[3]
        if -3 <= g <= max_gap and (best is None or g < best[0]):
            best = (g, o)
    return best[1] if best else None


def rows_of(lines, tol=3.5):
    """Group horizontal lines into rows (by y), each row sorted by x."""
    rows = []
    for l in sorted([l for l in lines if l["dir"] == (1, 0)], key=lambda l: centre(l["bbox"])[1]):
        cy = centre(l["bbox"])[1]
        if rows and abs(rows[-1]["y"] - cy) <= tol:
            rows[-1]["items"].append(l)
        else:
            rows.append({"y": cy, "items": [l]})
    for r in rows:
        r["items"].sort(key=lambda l: l["bbox"][0])
        r["text"] = " ".join(l["text"] for l in r["items"])
    return rows


# ======================================================================== panel: tables, notes, title block
def find_headings(lines):
    hs = []
    for l in lines:
        if l["dir"] != (1, 0):
            continue
        t = l["text"].strip().upper()
        for kind, pat in HEADINGS.items():
            if re.match(pat, t):
                hs.append({"kind": kind, "line": l})
                break
    return hs


def heading_regions(lines, heads, page_w, page_h=1e9):
    stops = [l for l in lines if l["dir"] == (1, 0) and re.match(STOPS, l["text"].strip().upper())]
    regions = []
    order = {"comparative_table": 0, "track_details": 1}
    for h in sorted(heads, key=lambda h: order.get(h["kind"], 2)):
        hb = h["line"]["bbox"]
        # the section ends at the next heading (or stop line) below it in the same column ...
        # (an ADD NOTE box set into the column, indented and in smaller type, does not end it: 799-1 lost notes 6-27)
        below = [g["line"]["bbox"][1] for g in heads if g is not h and g["line"]["bbox"][1] > hb[3] - 1
                 and hb[0] - 70 <= g["line"]["bbox"][0] <= hb[0] + 120
                 and not (g["kind"] == "add_note" and h["kind"] != "add_note" and g["line"]["bbox"][0] > hb[0] + 25)]
        # (a table ends at its last numbered row anyway: only a stop line in its own column ends it - the signature
        # block's "RAILWAY OFFICIALS" beside 752-3's table cut it at row 7)
        reach = 120 if h["kind"] in ("comparative_table", "track_details") else 400
        below += [s["bbox"][1] for s in stops if s["bbox"][1] > hb[3] and hb[0] - 70 <= s["bbox"][0] <= hb[0] + reach]
        bottom = min(below + [page_h - 5])
        # ... and its right edge is the next heading to the right beside it (not one further down the same column)
        right = min([g["line"]["bbox"][0] for g in heads if g["line"]["bbox"][0] > hb[0] + 60 and hb[1] - 5 <= g["line"]["bbox"][1] < bottom]
                    + [page_w])
        if h["kind"] == "depth_of_track_structure":            # a small box; the signature boxes sit right of it
            right = min(right, hb[0] + 230)
            # ... and only a few "RAIL HEIGHT = 172MM ... TOTAL = 762MM" lines tall: it ends at its last such line (with
            # nothing below it, the box ran to the bottom of the sheet and took in the key plan drawn there)
            cand = [l for l in lines if l["dir"] == (1, 0) and l["text"].strip() and hb[0] - 18 <= l["bbox"][0] < right and hb[3] - 2 < l["bbox"][1] < bottom]
            last = hb[3]
            for r in rows_of(cand):                                 # a row's item and value can be separate texts
                top, low = min(l["bbox"][1] for l in r["items"]), max(l["bbox"][3] for l in r["items"])
                if top - last > 30:
                    break
                if re.search(r"[=:]|\bTOTAL\b|\d\s*MM\b", r["text"], re.I):    # other text beside it (designations) is passed over
                    last = low
            bottom = min(bottom, last + 8)
        if h["kind"] in ("comparative_table", "track_details"):
            # the table is as wide as its header row (details of the drawing may sit right next to it)
            hdr = [l for l in lines if l["dir"] == (1, 0) and hb[3] - 2 <= l["bbox"][1] <= hb[3] + 40 and hb[0] - 15 <= l["bbox"][0] < min(right, hb[0] + 420)
                   and re.search(r"PROPOSED|EXISTING", l["text"], re.I) and not re.search(r"-\s*(PROPOSED|EXISTING)|PROP\.?\s*-|EXG\.?\s*-", l["text"], re.I)]
            # (an abbreviation list beside the table, "PROP. - PROPOSED", is not its header: 827-3)
            if hdr:
                right = min(right, max(l["bbox"][2] for l in hdr) + 25)
                h["width"] = right - hb[0]
            else:
                # (no header of its own: as wide as the comparative table, else a table's usual width - 780-1's track
                # details ran across the sheet and hid the curtain wall detail's title)
                w = next((g.get("width") for g in heads if g.get("width")), None) or                     next((r["box"][2] - r["box"][0] for r in regions if r["kind"] == "comparative_table"), None) or 340
                right = min(right, hb[0] + w + 5)
        if h["kind"] in ("comparative_table", "track_details"):
            # the table ends at its last numbered row
            serial = sorted([l for l in lines if re.match(r"\d{1,2}\.?(\s|$)", l["text"].strip()) and hb[0] - 18 <= l["bbox"][0] < hb[0] + 45
                             and hb[3] - 2 < l["bbox"][1] < bottom], key=lambda l: l["bbox"][1])
            last = hb[3]
            for k, l in enumerate(serial):
                if l["bbox"][1] - last > (50 if k == 0 else 30):    # (a column header row sits above row 1)
                    break
                last = l["bbox"][3]
            bottom = min(bottom, last + 8)
        box = [hb[0] - 18, hb[1] - 1, right - 3, bottom - 0.5]
        body = [l for l in lines if l is not h["line"] and inside(centre(l["bbox"]), box) and l["bbox"][1] >= hb[3] - 2]
        regions.append({"kind": h["kind"], "heading": h["line"]["text"], "heading_box": hb, "box": box, "lines": body})
    # an ADD NOTE box set inside another region (the notes column): its own lines are the ones left-aligned with its
    # heading, one under the other; they are taken out of the region around it, and only they are the add note
    for o in regions:
        if o["kind"] != "add_note":
            continue
        hx, last, own = o["heading_box"][0], o["heading_box"][3], []
        for l in sorted([l for l in lines if l is not o.get("_h") and l["bbox"][1] > o["heading_box"][1] + 1
                         and abs(l["bbox"][0] - hx) < 8], key=lambda l: l["bbox"][1]):
            if l["bbox"][1] - last > 25:
                break
            own.append(l)
            last = l["bbox"][3]
        host = [r for r in regions if r is not o and inside(centre(o["heading_box"]), r["box"])]
        if host:
            o["lines"] = own
            o["box"] = union([o["heading_box"]] + [l["bbox"] for l in own])
            ids = {id(l) for l in own} | {id(l) for l in lines if l["bbox"] == o["heading_box"]}   # and its heading
            for r in host:
                r["lines"] = [l for l in r["lines"] if id(l) not in ids]
    return regions


def numbered(lines):
    """Numbered list items ("1. ...", "11. | ...") with their continuation lines."""
    items = []
    for r in rows_of(lines):
        t = r["text"].replace("|", " ")
        t = re.sub(r"\s+", " ", t).strip()
        m = re.match(r"^(\d{1,2})\s*[.)]?\s*(?=\S)(.*)$", t)
        if m and (m.group(2) and not re.match(r"^\d", m.group(2)) or not items):
            items.append({"no": m.group(1), "text": m.group(2).strip()})
        elif items:
            items[-1]["text"] = (items[-1]["text"] + " " + t).strip()
        else:
            items.append({"no": None, "text": t})
    for it in items:
        it["text"] = re.sub(r"\s+", " ", it["text"]).strip()
    return [it for it in items if it["text"]]


def parse_specifications(lines):
    out = []
    for it in numbered(lines):
        m = re.match(r"^(.*?)\s*:\s*(.+)$", it["text"])
        if m:
            out.append({"no": it["no"], "item": m.group(1).strip(), "value": m.group(2).strip()})
        else:
            out.append({"no": it["no"], "item": it["text"], "value": None})
    return out


def parse_abbreviations(lines):
    out = {}
    for r in rows_of(lines):
        t = re.sub(r"\s+", " ", r["text"].replace("|", " "))
        for m in re.finditer(r"([A-Z][A-Z0-9./()&]*\.?)\s*[-–=:]\s*([A-Za-z][A-Za-z .,/()&']*?)(?=\s+[A-Z][A-Z0-9./()&]*\.?\s*[-–=:]\s|$)", t):
            out[m.group(1).rstrip(".").strip()] = m.group(2).strip()
    return out


def parse_key_values(lines):
    out = []
    for r in rows_of(lines):
        t = re.sub(r"\s+", " ", r["text"].replace("|", " "))
        for m in re.finditer(r"([A-Z][A-Z .&/()]*?)\s*[=:]\s*([^=:]+?)(?=\s+[A-Z][A-Z .&/()]*?\s*[=:]|$)", t):
            out.append({"item": m.group(1).strip(), "value": m.group(2).strip()})
    return out


def parse_table(region, words, cols_from=None):
    """Comparative table / track details from the words in the table box: numbered rows of
    (description, existing, proposed). cols_from: column positions of the comparative table (track details
    usually has no header of its own)."""
    box = region["box"]
    hb = region["heading_box"]
    ws = [w for w in words if inside(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2), box) and w[1] >= hb[3] - 1]
    cols = {}
    head_y = None
    abbr = [(w[1] + w[3]) / 2 for w in ws if w[4] in ("-", "–")]          # "PROP. - PROPOSED" in an abbreviation list
    prop = [w for w in ws if re.match(r"PROPOSED", w[4], re.I) and w[1] < hb[3] + 40 and not any(abs((w[1] + w[3]) / 2 - y) < 3 for y in abbr)]
    if prop:
        p = min(prop, key=lambda w: w[1])
        head_y = (p[1] + p[3]) / 2
        cols["proposed"] = (p[0] + p[2]) / 2
        ex = [w for w in ws if re.match(r"EXISTING", w[4], re.I) and abs((w[1] + w[3]) / 2 - head_y) < 5 and w[0] < p[0]]
        if ex:
            e = max(ex, key=lambda w: w[0])
            cols["existing"] = (e[0] + e[2]) / 2
    elif cols_from:
        cols = dict(cols_from)
    split = (min(cols.values()) - 0.5 * abs(cols.get("proposed", 0) - cols.get("existing", 0)) - 5) if len(cols) == 2 else \
        (min(cols.values()) - 40 if cols else None)
    # rows: one per serial number at the left edge; words between two serial numbers belong to the upper row
    serials = sorted([w for w in ws if re.fullmatch(r"\d{1,2}\.?", w[4]) and w[0] < hb[0] + 45 and w[0] >= hb[0] - 18
                      and (head_y is None or w[1] > head_y + 3)], key=lambda w: w[1])
    out = []
    for i, s in enumerate(serials):
        # a row runs from halfway to the serial above to halfway to the one below (a serial centred on a two-line
        # description, 815-2's "VERTICAL CLEARANCE AS / PER OHFL", keeps both lines)
        cy = (s[1] + s[3]) / 2
        py = (serials[i - 1][1] + serials[i - 1][3]) / 2 if i else None
        ny = (serials[i + 1][1] + serials[i + 1][3]) / 2 if i + 1 < len(serials) else None
        y0 = (py + cy) / 2 if py is not None and cy - py < 30 else cy - 6
        y1 = (cy + ny) / 2 if ny is not None and ny - cy < 30 else cy + 9
        if out and cy - 5 - out[-1]["_y"] > 30:              # a gap: the table has ended
            break
        rw = sorted([w for w in ws if y0 <= (w[1] + w[3]) / 2 < y1 and w is not s], key=lambda w: (round(w[1] / 4), w[0]))
        desc = " ".join(w[4] for w in rw if split is None or (w[0] + w[2]) / 2 < split)
        if not desc:
            continue
        row = {"description": re.sub(r"\s+", " ", desc).strip(), "_y": cy - 5}
        gapc = abs(cols.get("proposed", 0) - cols.get("existing", 0)) if len(cols) == 2 else 0
        for w in rw:
            cx = (w[0] + w[2]) / 2
            # a word goes where its text line stands ("CURVED (2.30 DEG.)" is one value: 829-2's "DEG.)" sat nearer
            # the proposed column), when that line is no wider than a column
            host = next((l for l in region["lines"] if inside((cx, (w[1] + w[3]) / 2), l["bbox"])), None)
            if host and gapc and host["bbox"][2] - host["bbox"][0] < gapc:
                cx = centre(host["bbox"])[0]
            if split is not None and cx < split:
                continue
            col = min(cols, key=lambda c: abs(cols[c] - cx)) if cols else "value"
            row[col] = (row[col] + " " + w[4]) if col in row else w[4]
        key, meaning = K.table_row_kind(row["description"])
        row["kind"] = key
        if meaning:
            row["meaning"] = meaning
        out.append(row)
    for r in out:
        r.pop("_y")
    return {"columns": sorted(cols), "rows": out, "_cols": cols}


TITLE_KEYS = [("division", r"DIVISION"), ("section", r"SECTION"), ("pink_book_item", r"PINK\s+BOOK\s+ITEM\s+NO\.?"),
              ("km_chainage", r"KM\s*/\s*CHAINAGE"), ("project", r"PROJECT"), ("name_of_work", r"NAME\s+OF\s+WORK"),
              ("title", r"TITLE"), ("consultants_dwg_no", r"CONSULTANTS?\s+DWG\.?\s+NO\.?"), ("size", r"SIZE"),
              ("hq_dwg_no", r"HQ'?S\s+DWG\.?\s+NO\.?"), ("dwg_no", r"DWG\.?\s+NO\.?"), ("rev_no", r"REV\.?\s*NO\.?"),
              ("date", r"DATE"), ("scale", r"SCALE")]


def parse_title_block(lines, page):
    div = next((l for l in lines if re.match(r"^DIVISION\b", l["text"].upper())), None)
    if not div:
        return {}
    x0 = div["bbox"][0] - 15                           # (the title's lines may start a little left of DIVISION: 789-1)
    top = next((l["bbox"][1] for l in lines if l["text"].upper().strip() in ("CENTRAL RAILWAY", "WEST CENTRAL RAILWAY")
                and l["bbox"][0] >= x0 - 60 and l["bbox"][1] < div["bbox"][1]), div["bbox"][1] - 30)
    region = [l for l in lines if l["bbox"][0] >= x0 and l["bbox"][1] >= top - 2 and l["dir"] == (1, 0)
              and not SIGNATURE.search(l["text"])]
    out, key = {}, None
    keypat = re.compile(r"(?<![A-Z])(" + "|".join(f"(?P<{k}>{p})" for k, p in TITLE_KEYS) + r")\s*:", re.I)
    for r in rows_of(region):
        t = re.sub(r"\s+", " ", r["text"].replace("|", " ")).strip()
        if re.fullmatch(r"(WEST )?CENTRAL RAILWAY", t.upper()):
            out["railway"] = t
            continue
        pos = [(m.start(), m.end(), next(k for k, v in m.groupdict().items() if v)) for m in keypat.finditer(t)]
        if not pos:
            if key in ("name_of_work", "title", "project"):
                out[key] = (out.get(key, "") + " " + t).strip()
            continue
        if pos[0][0] > 0 and key in ("name_of_work", "title"):
            out[key] = (out.get(key, "") + " " + t[:pos[0][0]]).strip()
        for i, (s, e, k) in enumerate(pos):
            v = t[e:pos[i + 1][0] if i + 1 < len(pos) else len(t)].strip()
            out[k] = (out.get(k, "") + " " + v).strip() if k in out else v
            key = k
    for k in list(out):
        out[k] = re.sub(r"\s+", " ", out[k]).strip(" :")
    return out


def parse_revisions(lines):
    """The revision history of the title block: rows of DATE | REV. NO | DESCRIPTION above that header row
    ("01/07/2025  R0  FIRST SUBMISSION", "18/08/2025  R1  REVISED AS PER RAILWAY'S OBSERVATIONS")."""
    heads = [l for l in lines if l["dir"] == (1, 0) and re.fullmatch(r"\s*REV\.?\s*NO\.?\s*", l["text"], re.I)]
    for rev in heads:
        cy = centre(rev["bbox"])[1]
        same = [l for l in lines if l["dir"] == (1, 0) and abs(centre(l["bbox"])[1] - cy) < 5]
        date = next((l for l in same if re.fullmatch(r"\s*DATE\s*", l["text"], re.I) and l["bbox"][0] < rev["bbox"][0]), None)
        desc = next((l for l in same if re.fullmatch(r"\s*DESCRIPTION\s*", l["text"], re.I) and l["bbox"][0] > rev["bbox"][0]), None)
        if not (date and desc):
            continue
        x0, x1 = date["bbox"][0] - 25, desc["bbox"][2] + 260
        above = [l for l in lines if l["dir"] == (1, 0) and x0 <= l["bbox"][0] <= x1 and cy - 140 < centre(l["bbox"])[1] < cy - 3]
        out = []
        for r in sorted(rows_of(above), key=lambda r: -r["y"]):           # upwards from the header
            d = next((m.group() for l in r["items"] for m in [re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", l["text"])] if m), None)
            n = next((m.group() for l in r["items"] for m in [re.fullmatch(r"\s*(R\s*\d+)\s*", l["text"], re.I)] if m), None)
            if not (d or n):
                if out:
                    break
                continue
            words = [l["text"].strip() for l in r["items"] if l["bbox"][0] >= desc["bbox"][0] - 140
                     and not re.fullmatch(r"\s*(R\s*\d+|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*", l["text"], re.I)
                     and not re.fullmatch(r"\s*(DRAWN|CHECKED|APPROVED)\s*", l["text"], re.I)]   # the signature boxes' heads
            out.append({"rev_no": re.sub(r"\s", "", n).upper() if n else None, "date": d,
                        "description": re.sub(r"\s+", " ", " ".join(words)).strip() or None})
        if out:
            return sorted(out, key=lambda x: int(re.sub(r"\D", "", x["rev_no"] or "0") or 0))
    return []


def bridge_from_title(title):
    """Bridge no., box size, chainage and the relation to the existing bridge, as written in the title."""
    t = re.sub(r"\s+", " ", title or "")
    t = re.sub(r"(?<=\d)\.\.(?=\d)", ".", t)                  # "1x1..830m" (827-1)
    info = {}
    m = re.search(r"(MINOR|MAJOR|IMPORTANT)\s+BRIDGE|\bRUB\b|ROAD UNDER BRIDGE|SUBWAY", t, re.I)
    if m:
        info["category"] = m.group(0).upper()
    # "BRIDGE NO.810/1 (1x3.660x6.530m RCC BOX AT CH ...)", "RUB 822/2 (1x4.013x3.666m. RCC BOX CH. 77875 )"
    m = re.search(r"(?:BRIDGE\s*NO\.?|\bRUB\s*(?:NO\.?)?)\s*([A-Z0-9/\-]+)\s*\(\s*([^)]*?)\s*(?:CH[.:]?\s*([\d+.]+)\s*M?)?\s*\)", t, re.I)
    if m:
        info["bridge_no"] = m.group(1)
        info["description"] = re.sub(r"\s+", " ", m.group(2)).strip(" ,")
        if m.group(3):
            info["chainage"] = m.group(3)
    else:
        m = re.search(r"BRIDGE\s*NO\.?\s*([A-Z0-9/\-]+)", t, re.I)
        if m:
            info["bridge_no"] = m.group(1)
    head = info.get("description") or re.split(r"\bEXISTING\b|\bEX\.", t, flags=re.I)[0]
    # cells x clear width x clear height, units written or not: "1x 4.560mx3.648m", "2x4.890x4.245m"
    m = re.search(r"(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*M?\s*[xX×]\s*(\d+(?:\.\d+)?)\s*\)?\s*M?", head, re.I)
    if m:
        info["box"] = {"cells": int(m.group(1)), "clear_width_m": float(m.group(2)), "clear_height_m": float(m.group(3)),
                       "as_printed": m.group(0).strip()}
    else:
        # a slab bridge: spans x clear span ("1x1.22m RCC SLAB", "PROPOSED SPAN (1X4.57) PSC SLAB")
        m = re.search(r"(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*M?\s*\)?\s*((?:RCC|PSC)\s*(?:\(?L\.?L\.?\)?\s*)?SLAB)", t, re.I)
        if m:
            info["slab"] = {"spans": int(m.group(1)), "clear_span_m": float(m.group(2)), "type": re.sub(r"\s+", " ", m.group(3)).upper(),
                            "as_printed": m.group(0).strip()}
    if not info.get("chainage"):
        m = re.search(r"CH(?:AINAGE)?\.?\s*[:\-]?\s*([\d+.]+)\s*M?\b", t, re.I)
        if m:
            info["chainage"] = m.group(1)
    m = re.search(r"\b(ON\s+[UD]/S|ON\s+(?:UP|DN|DOWN)\s+SIDE|IN\s+LIEU|REPLAC\w*|EXTENSION|ALONG\w*)\b.*?(?:EXISTING\s+)?BRIDGE\s*NO\.?\s*([A-Z0-9/\-]+)\s*(\([^)]*\))?", t, re.I)
    if m:
        info["relation_to_existing"] = re.sub(r"\s+", " ", m.group(0)).strip()
    return info


# ======================================================================== views
def find_view_titles(lines, panel_boxes):
    titles = []
    used = set()
    for s in lines:
        m = SCALE_RE.match(s["text"].strip())
        mt = TITLE_WITH_SCALE.match(s["text"].strip()) if not m and s["size"] >= 9 else None
        if not (m or mt) or any(inside(centre(s["bbox"]), b) for b in panel_boxes):
            continue
        if mt:
            titles.append({"title": mt.group("t"), "scale": f"1:{mt.group('s')}", "lines": [s], "dir": s["dir"]})
            continue
        stack, ref = [], s
        for _ in range(3):
            first = stack[-1]["size"] if stack else None
            up = neighbour(lines, ref, up=True, max_gap=24,
                           need=lambda o: o["size"] >= (0.85 * first if first else 9.5) and o["i"] not in used
                           and not SCALE_RE.match(o["text"]) and re.search(r"[A-Z]{3}", o["text"])
                           and re.sub(r"(?<=\d)m\b", "", o["text"]).upper() == re.sub(r"(?<=\d)m\b", "", o["text"]) and not LEVEL_NUM.search(o["text"]))
            if not up:
                break
            stack.insert(0, up)
            used.add(up["i"])
            ref = up
        if stack:
            title = re.sub(r"\s+", " ", " ".join(l["text"] for l in stack)).replace("`", "'")
            titles.append({"title": title, "scale": f"1:{m.group(1)}" if m.group(1) else "not to scale", "lines": stack + [s], "dir": s["dir"]})
    # a key plan is often titled without a scale under it ("KEY PLAN" alone): a sketch for location, not to scale
    for s in lines:
        if s["i"] in used or any(s in t["lines"] for t in titles) or any(inside(centre(s["bbox"]), b) for b in panel_boxes):
            continue
        if re.fullmatch(r"\s*KEY\s*PLAN\s*(?:\(\s*N\.?\s*T\.?\s*S\.?\s*\)|\(\s*NOT\s+TO\s+SCALE\s*\))?\s*", s["text"], re.I) and s["size"] >= 9:
            titles.append({"title": "KEY PLAN", "scale": None, "lines": [s], "dir": s["dir"]})
    return titles


def occupancy(page, lines, skip_boxes):
    W, H = page.rect.width, page.rect.height
    gw, gh = int(W / CELL) + 2, int(H / CELL) + 2
    g = np.zeros((gh, gw), dtype=bool)
    segs = []
    for p in page.get_drawings():
        dashes = (p.get("dashes") or "").strip()
        if dashes and not dashes.startswith("[]"):                # centre lines run through aligned views: leave them out
            continue
        for it in p["items"]:
            op = it[0]
            if op == "l":
                segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
            elif op == "c":
                segs.append((it[1].x, it[1].y, it[4].x, it[4].y))
            elif op == "re":
                r = it[1]
                segs += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
            elif op == "qu":
                q = it[1]
                pts = [q.ul, q.ur, q.lr, q.ll, q.ul]
                segs += [(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y) for i in range(4)]
    if segs:
        s = np.array(segs, dtype=float)
        length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
        s = s[length < 0.35 * max(W, H)]                        # the page frame and border lines are not view content
        length = length[length < 0.35 * max(W, H)]
        n = np.maximum(2, (length / (CELL / 2)).astype(int) + 1)
        idx = np.repeat(np.arange(len(s)), n)
        t = np.concatenate([np.linspace(0, 1, k) for k in n])
        xs = s[idx, 0] + (s[idx, 2] - s[idx, 0]) * t
        ys = s[idx, 1] + (s[idx, 3] - s[idx, 1]) * t
        gx = np.clip((xs / CELL).astype(int), 0, gw - 1)
        gy = np.clip((ys / CELL).astype(int), 0, gh - 1)
        g[gy, gx] = True
    for l in lines:
        b = l["bbox"]
        g[int(b[1] / CELL):int(b[3] / CELL) + 1, int(b[0] / CELL):int(b[2] / CELL) + 1] = True
    for b in skip_boxes:
        g[max(0, int(b[1] / CELL)):int(b[3] / CELL) + 1, max(0, int(b[0] / CELL)):int(b[2] / CELL) + 1] = False
    return g


def dilate(g, r):
    out = g.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx or dy:
                sh = np.zeros_like(g)
                ys = slice(max(0, dy), g.shape[0] + min(0, dy))
                yd = slice(max(0, -dy), g.shape[0] + min(0, -dy))
                xs = slice(max(0, dx), g.shape[1] + min(0, dx))
                xd = slice(max(0, -dx), g.shape[1] + min(0, -dx))
                sh[ys, xs] = g[yd, xd]
                out |= sh
    return out


def components(g):
    """Connected components (4-neighbour) of the occupancy grid: (label grid, [{id, box, cells}])."""
    lab = np.zeros(g.shape, dtype=np.int32)
    n = 0
    H, W = g.shape
    comps = []
    for y0, x0 in zip(*np.nonzero(g)):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        ymin = ymax = y0
        xmin = xmax = x0
        cnt = 0
        while stack:
            y, x = stack.pop()
            cnt += 1
            ymin, ymax, xmin, xmax = min(ymin, y), max(ymax, y), min(xmin, x), max(xmax, x)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < H and 0 <= xx < W and g[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        comps.append({"id": n, "box": [xmin * CELL, ymin * CELL, (xmax + 1) * CELL, (ymax + 1) * CELL], "cells": cnt})
    return lab, comps


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def view_regions(page, lines, titles, skip_boxes):
    """Each view = the connected line work its title sits under (or inside), plus nearby fragments.

    No dilation: views on a GAD are separated by white space but aligned views share long centre lines, so
    growing regions would merge them. Components lying mostly in the panel (notes, tables, title block) are ignored."""
    g = occupancy(page, [l for l in lines if not any(inside(centre(l["bbox"]), b) for b in skip_boxes)], skip_boxes)
    lab, comps = components(g)
    comps = [c for c in comps if c["cells"] >= 4]
    comps = [c for c in comps if not any(overlap(c["box"], b) > 0.3 * max(area(c["box"]), 1) for b in skip_boxes)]
    tboxes = [union([l["bbox"] for l in t["lines"]]) for t in titles]
    # several titles inside one piece of line work (views in frames that touch, e.g. key plan | road L-section |
    # ground profile | bore log in a row): share its cells out, each to the nearest title, a view being drawn above its title
    keep, next_id = [], int(lab.max()) + 1
    for c in comps:
        inn = [ti for ti, tb in enumerate(tboxes) if inside(centre(tb), c["box"])]
        if len(inn) < 2:
            keep.append(c)
            continue
        ys, xs = np.nonzero(lab == c["id"])
        px, py = (xs + 0.5) * CELL, (ys + 0.5) * CELL
        cost = []
        for ti in inn:
            tb = tboxes[ti]
            dx = np.maximum(0, np.maximum(tb[0] - px, px - tb[2]))
            if titles[ti]["dir"] == (1, 0):
                dy = np.where(py < tb[1], tb[1] - py, np.maximum(0, py - tb[3]) * 3)
            else:
                dy = np.maximum(0, np.maximum(tb[1] - py, py - tb[3]))
            cost.append(2 * dx + dy)
        who = np.argmin(np.array(cost), axis=0)
        for k in range(len(inn)):
            sel = who == k
            if sel.sum() < 4:
                continue
            lab[ys[sel], xs[sel]] = next_id
            keep.append({"id": next_id, "box": [int(xs[sel].min()) * CELL, int(ys[sel].min()) * CELL, (int(xs[sel].max()) + 1) * CELL,
                                                (int(ys[sel].max()) + 1) * CELL], "cells": int(sel.sum())})
            next_id += 1
    comps = keep
    owner = {}
    for ci, c in enumerate(comps):
        best = None
        for ti, tb in enumerate(tboxes):
            d = titles[ti]["dir"]
            cb, rb = rbox(c["box"], d), rbox(tb, d)
            over = min(cb[2], rb[2]) - max(cb[0], rb[0])
            below = rb[1] - cb[3]                                   # title below the view, in the title's frame
            beside = cb[1] - rb[3] if d != (1, 0) else 1e9          # rotated titles may also sit on the other side
            if inside(centre(tb), c["box"]):
                score = 0
            elif over > 0.3 * (rb[2] - rb[0]) and (-15 <= below <= 90 or -15 <= beside <= 90):
                score = min(v for v in (below, beside) if -15 <= v <= 90)
            else:
                continue
            if best is None or score < best[0]:
                best = (score, ti)
        if best:
            owner.setdefault(best[1], []).append(ci)
    regions, comp_view = [], {}
    for ti in range(len(titles)):
        mains = sorted(owner.get(ti, []), key=lambda ci: -comps[ci]["cells"])
        box = tboxes[ti]
        if mains:
            box = union([box, comps[mains[0]]["box"]])
            comp_view[comps[mains[0]]["id"]] = ti
            for ci in mains[1:]:                                    # smaller parts right above the title as well
                if comps[ci]["cells"] >= 0.15 * comps[mains[0]]["cells"]:
                    box = union([box, comps[ci]["box"]])
                    comp_view[comps[ci]["id"]] = ti
        regions.append(box)
    # a view whose drawing touches a bigger view's line work gets no component of its own: grow it from the labels
    # around its title that are written in the same direction as the title (e.g. a rotated small section beside the plan)
    overrides = []
    title_ids = {l["i"] for t in titles for l in t["lines"]}
    for ti, t in enumerate(titles):
        own = [comps[ci]["box"] for ci in owner.get(ti, [])]
        tb = tboxes[ti]
        if own and not all(inside((b[0], b[1]), tb, pad=20) and inside((b[2], b[3]), tb, pad=20) for b in own):
            continue                                              # it has line work beyond its own title
        d, box = t["dir"], tboxes[ti]
        cand = [l for l in lines if l["dir"] == d and l["i"] not in title_ids and not any(inside(centre(l["bbox"]), b) for b in skip_boxes)]
        for _ in range(12):
            near = [l for l in cand if gap(l["bbox"], box) <= 70 and not inside(centre(l["bbox"]), box)]
            if not near:
                break
            nb = union([box] + [l["bbox"] for l in near])
            if max(nb[2] - nb[0], nb[3] - nb[1]) > 700:
                break
            box = nb
        if box != tboxes[ti]:
            regions[ti] = box
            overrides.append((box, ti))
    GRID["overrides"] = overrides
    # fragments (labels and leaders separated by white space) join the view they touch or sit in
    for _ in range(2):
        for c in comps:
            if c["id"] in comp_view:
                continue
            within = [ri for ri, r in enumerate(regions) if inside(centre(c["box"]), r)]
            if within:
                ri = min(within, key=lambda ri: area(regions[ri]))
                comp_view[c["id"]] = ri
                continue
            cands = [(gap(c["box"], r), ri) for ri, r in enumerate(regions)]
            if not cands:
                continue
            dmin, ri = min(cands)
            if dmin <= 18 and area(c["box"]) < 0.5 * area(regions[ri]) and \
                    sum(overlap(union([regions[ri], c["box"]]), r) > 0.25 * area(r) for j, r in enumerate(regions) if j != ri) == 0:
                regions[ri] = union([regions[ri], c["box"]])
                comp_view[c["id"]] = ri
    return regions, (lab, comp_view)


GRID = {}                                       # (label grid, component -> view index) of the page being annotated


def assign_view(pt, views):
    """The view a point belongs to: the view owning the line work at that point (exact even where view boxes
    overlap), else the smallest view box containing it."""
    for box, ti in GRID.get("overrides", []):
        if inside(pt, box, pad=2):
            return views[ti]
    if GRID.get("lab") is not None:
        lab, comp_view = GRID["lab"], GRID["comp_view"]
        cy, cx = int(pt[1] / CELL), int(pt[0] / CELL)
        for r in range(3):
            ids = lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].ravel()
            vs = Counter(comp_view[i] for i in ids if i in comp_view)
            if vs:
                return views[vs.most_common(1)[0][0]]
    hits = [v for v in views if inside(pt, v["bbox"], pad=2)]
    if not hits:
        return None
    return min(hits, key=lambda v: (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]))


# ======================================================================== values in the views
def read_levels(lines, views, exclude):
    out, used = [], set()
    for l in lines:
        if l["i"] in exclude or l["i"] in used:
            continue
        t = l["text"]
        nums = LEVEL_NUM.findall(t)
        label = None
        if nums and LEVEL_WORD.search(t):
            label = LEVEL_NUM.sub("", t)
            parts = [l]
        elif nums and re.fullmatch(r"\s*[-+]?\d{2,4}\.\d{2,3}\s*(m|M)?\s*", t):
            up = neighbour(lines, l, up=True, max_gap=12, need=lambda o: LEVEL_WORD.search(o["text"]) and not LEVEL_NUM.search(o["text"])
                           and o["i"] not in exclude)
            if not up:
                continue
            label, parts = up["text"], [up, l]
        else:
            continue
        # a bare "LVL." label: the element is written on the line above ("TOP OF RCC BOX" / "LVL. 343.219")
        ref = parts[0]
        for _ in range(2):
            if not re.fullmatch(r"\s*(LVL|LEVEL|LVL\.|LEVEL\.)\s*[.:=-]*\s*(M|m)?\s*", re.sub(r"[\d.]+", "", label)):
                if K.level_kind(label)[0] != "reduced_level" or re.search(r"\bR\.?L\b", label):
                    break
            up = neighbour(lines, ref, up=True, max_gap=12, need=lambda o: not LEVEL_NUM.search(o["text"]) and o["i"] not in exclude
                           and re.search(r"[A-Z]{2}", o["text"]) and not SCALE_RE.match(o["text"]))
            if not up:
                break
            label = up["text"] + " " + label
            parts.insert(0, up)
            ref = up
        # a qualifier on the line above ("EXISTING DN TRACK" / "RAIL LVL 404.547")
        up = neighbour(lines, ref, up=True, max_gap=10, need=lambda o: not LEVEL_NUM.search(o["text"]) and o["i"] not in exclude
                       and o["i"] not in used and re.match(r"^(EXISTING|EXIST\.?|EXG\.?|EX\.?|PROP\.?|PROPOSED)\b(?!.*\b(LVL|LEVEL)\b)", o["text"].strip(), re.I)
                       and len(o["text"]) <= 30)
        if up and not re.match(r"^(EXISTING|EXIST|EXG|EX|PROP|PROPOSED)\b", label.strip(), re.I):
            label = up["text"] + " " + label
            parts.insert(0, up)
        label = re.sub(r"\s+", " ", re.sub(r"[=:]\s*$", "", label.replace("|", " "))).strip(" .:=-")
        if re.match(r"^R\.?L\.?\b", label):              # bore-log marks: the soil text beside them is not part of the label
            label = "RL"
        for p in parts:
            used.add(p["i"])
        kind, meaning = K.level_kind(label)
        for v in nums:
            box = union([p["bbox"] for p in parts])
            view = assign_view(centre(box), views)
            el = K.OF_ELEMENT.search(label)
            out.append({"label": label, "value": float(v), "unit": "m", "kind": kind, "meaning": meaning,
                        "status": K.status_of(label) or (view and "existing" if view and view["kind"] == "existing_bridge_section" else None),
                        "element": el.group(1).upper() if el else None,
                        "view": view["title"] if view else None, "bbox": box, "lines": [p["i"] for p in parts]})
    return out, used


def centre_lines(lines, views):
    out = []
    for l in lines:
        m = re.match(r"^(?:C|℄|CL|C/L|C\.L\.?)\s+OF\s+(.+)$", l["text"].strip(), re.I)
        if not m:
            continue
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if not re.search(r"TRACK|LINE|BRIDGE|BOX|ROAD|DRAIN|RUB", name, re.I):
            continue
        rb = l["bbox"]
        x = rb[0] + 3 if l["dir"] == (1, 0) else None
        y = rb[3] - 3 if l["dir"] != (1, 0) else None
        view = assign_view(centre(rb), views)
        out.append({"name": name, "x": x, "y": y, "dir": l["dir"], "view": view["title"] if view else None, "line": l["i"]})
    return out


def describe_between(cls, tline, view):
    same = [c for c in cls if c["view"] == view and c["x"] is not None]
    if tline["dir"] != (1, 0) or len(same) < 2:
        return None
    cx = centre(tline["bbox"])[0]
    left = [c for c in same if c["x"] <= cx]
    right = [c for c in same if c["x"] > cx]
    if not left or not right:
        return None
    a, b = max(left, key=lambda c: c["x"]), min(right, key=lambda c: c["x"])
    return [a["name"], b["name"]]


def read_dims(lines, views, cls, exclude):
    labelled, plain, slopes = [], [], []
    for l in lines:
        if l["i"] in exclude:
            continue
        t = re.sub(r"\s+", " ", l["text"].replace("|", " ")).strip()
        view = assign_view(centre(l["bbox"]), views)
        vt = view["title"] if view else None
        hit = False
        for pat, kind, meaning in K.LABELLED_DIMS:
            m = re.search(pat, t, re.I)
            if not m:
                continue
            what = (m.groupdict().get("what") or "").strip(" .,-")
            # "BY (SOLING OF 350MM THK.)" -> "SOLING": no brackets, no leading BY/WITH, no trailing OF
            what = re.sub(r"[()]", " ", what)
            what = re.sub(r"^\s*(?:BY|WITH|OF|AND)\s+", "", what, flags=re.I)
            what = re.sub(r"\s+(?:OF|BY|WITH|AND)\s*$", "", what, flags=re.I)
            what = re.sub(r"\s+", " ", what).strip(" .,-") or None
            between = describe_between(cls, l, vt) if kind == "track_centres" else None
            mean = meaning.format(what=(what or "item").lower(),
                                  between=(" and the ".join(n.lower() for n in between) if between else "the two tracks shown"))
            labelled.append({"label": t, "value": float(m.group("v")), "unit": "mm", "kind": kind, "what": what,
                             "between": between, "meaning": mean, "view": vt, "bbox": l["bbox"], "line": l["i"]})
            hit = True
            break
        if hit:
            continue
        m = K.SLOPE_RE.match(t)
        if m and view:
            slopes.append({"label": t, "kind": "side_slope", "meaning": f"a slope of {m.group('a')} horizontal to {m.group('b')} vertical",
                           "view": vt, "bbox": l["bbox"], "line": l["i"]})
            continue
        m = K.GRADIENT_RE.search(t)
        if m and view and re.search(r"SLOPE|GRAD|IN\s+\d", t, re.I):
            up = neighbour(lines, l, up=True, max_gap=12)
            what = up["text"] if up and re.search(r"SLOPE|GRAD", up["text"], re.I) else t
            slopes.append({"label": (what + " " + t).strip() if what != t else t, "kind": "gradient",
                           "meaning": f"a gradient of 1 in {m.group('n')} ({what.lower()})", "view": vt, "bbox": l["bbox"], "line": l["i"]})
            continue
        if DIM_NUM.match(t) and view and view["kind"] not in ("key_plan", "bore_log"):
            plain.append({"value": float(t), "unit": "mm", "direction": "horizontal" if l["dir"] == (1, 0) else "vertical" if l["dir"][0] == 0 else "inclined",
                          "view": vt, "bbox": l["bbox"], "line": l["i"]})
    return labelled, plain, slopes


def parse_key_plan(view, lines):
    """What the key plan shows, as things the model can answer from: the stations on either side, the bridge
    callouts (existing and proposed), the tracks, curves, gradients, the chainage / KM / FL markers (labels printed
    stacked together are one marker), the railway boundary and land to be acquired, flow, bore holes."""
    inv = [l for l in lines if inside(centre(l["bbox"]), view["bbox"]) and l["i"] not in view.get("title_lines", [])
           and not SCALE_RE.match(l["text"].strip())]

    def ends(l):                                         # start and "across" direction of a line in its own frame
        d = l["dir"]
        start = (l["bbox"][0], l["bbox"][1]) if d == (1, 0) else (l["bbox"][0], l["bbox"][3]) if d == (0, -1) else (l["bbox"][2], l["bbox"][1])
        return start, (-d[1], d[0])

    groups, used = [], set()
    for l in sorted(inv, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        if l["i"] in used:
            continue
        g, cur = [l], l
        while True:
            (sx, sy), (nx, ny) = ends(cur)
            nxt = None
            for o in inv:
                if o["i"] in used or o in g or o["dir"] != cur["dir"]:
                    continue
                (ox, oy), _ = ends(o)
                across = (ox - sx) * nx + (oy - sy) * ny
                along = (ox - sx) * cur["dir"][0] + (oy - sy) * cur["dir"][1]
                if 0.5 * cur["size"] < across < 2.2 * cur["size"] and abs(along) < 3 * cur["size"]:
                    nxt = o if nxt is None or across < ((ends(nxt)[0][0] - sx) * nx + (ends(nxt)[0][1] - sy) * ny) else nxt
            if nxt is None:
                break
            g.append(nxt)
            cur = nxt
        used.update(x["i"] for x in g)
        groups.append(g)

    kp = {"stations": [], "bridges": [], "tracks": [], "curves": [], "gradients": [], "markers": [], "boundary": [],
          "flow": False, "bore_holes": 0, "other": []}
    cx_view = (view["bbox"][0] + view["bbox"][2]) / 2
    for g in groups:
        t = re.sub(r"\s+", " ", " ".join(x["text"].replace("|", " ") for x in g)).strip()
        T = t.upper()
        x = centre(union([o["bbox"] for o in g]))[0]
        if re.search(r"\bBR(?:IDGE)?\b|\bBR-|\bRCC\s+BOX\b|\bSLAB\b", T) and len(T) > 12:
            kp["bridges"].append(t)
        elif re.search(r"\b(STATION|STN\.?)\b", T) or re.fullmatch(r"TO\s+(?!BE\b|THE\b|ALL\b)[A-Z]{3,}(?:\s+[A-Z]{3,})?", T) or \
                (g[0]["size"] >= 10.5 and re.fullmatch(r"[A-Z]{4,}", T) and not re.search(r"KEY|PLAN|FLOW|LEVEL|BOUNDARY|NORTH", T)):
            # a station: "... STATION" / "... STN.", "TO NAGPUR", or a town's name alone in large type ("ITARSI")
            kp["stations"].append({"name": re.sub(r"\b(TO|STATION|STN\.?)\b", "", T).strip(" .") or T,
                                   "text": t, "side": "left" if x < cx_view else "right"})
        elif re.search(r"\bCH\s*[:.]|\bKM\s*[:.]|\bFL\s*[:.]", T):
            m = {}
            for key, pat in (("ch", r"\bCH\s*[:.]?\s*([\d+.]+)"), ("km", r"\bKM\s*[:.]?\s*([\d.]+)"), ("fl", r"\bFL\s*[:.]?\s*([\d.]+)")):
                v = re.search(pat, T)
                if v:
                    m[key] = v.group(1).rstrip(".")
            m["text"] = t
            kp["markers"].append(m)
        elif re.search(r"\bM/L\b|\b\d(?:ST|ND|RD|TH)\s*/?\s*L\b|\bLINE\b|\bLOOP\b|\bGOODS\b", T):
            if T not in [s.upper() for s in kp["tracks"]]:
                kp["tracks"].append(t)
        elif re.fullmatch(r"R\s*\d+(?:\.\d+)?\s*M?", T):
            kp["curves"].append(t)
        elif re.search(r"\b(FALL|RISE)\s*1\s*IN\s*\d|^R\s*1\s*IN\b|^F\s*1\s*IN\b|^LEVEL$", T):
            kp["gradients"].append(t)
        elif re.search(r"BOUNDARY|LAND\s+TO\s+BE\s+ACQUIRED|ROW\b", T):
            if T not in [s.upper() for s in kp["boundary"]]:
                kp["boundary"].append(t)
        elif T == "FLOW":
            kp["flow"] = True
        elif re.fullmatch(r"B\.?H\.?(\s*-?\s*\d+)?", T):
            kp["bore_holes"] += 1
        else:
            kp["other"].append(t)
    kp["stations"].sort(key=lambda s: 0 if s["side"] == "left" else 1)
    return kp


def read_bore_log(view, lines, skip=()):
    box = view["bbox"]
    if box[3] - box[1] < 60:
        # the view came out as its title alone (its line work joined a bigger view's): the log is the column of text
        # standing right above the title
        box = [box[0] - 60, box[1] - 330, box[2] + 90, box[3]]
    inside_lines = [l for l in lines if inside(centre(l["bbox"]), box) and not any(inside(centre(l["bbox"]), b) for b in skip)]
    sbc, layers, rls = [], [], []
    for l in sorted(inside_lines, key=lambda l: centre(l["bbox"])[1]):
        t = re.sub(r"\s+", " ", l["text"].replace("|", " ")).strip()
        m = re.search(r"SBC\s*[=:]?\s*(\d+(?:\.\d+)?)\s*T\s*/\s*M\s*[²2]?\s*(\d+(?:\.\d+)?)?", t, re.I)   # "T/M 2" is m², not a depth
        if m:
            depth = float(m.group(2)) if m.group(2) else None
            if depth is None:                                    # the depth is often its own text on the same row
                cy = centre(l["bbox"])[1]
                d = [o for o in inside_lines if abs(centre(o["bbox"])[1] - cy) < 4 and 0 <= o["bbox"][0] - l["bbox"][2] < 120
                     and re.fullmatch(r"\d{1,2}\.\d{1,3}", o["text"].strip())]
                if d:
                    depth = float(min(d, key=lambda o: o["bbox"][0])["text"])
            sbc.append({"sbc_t_per_m2": float(m.group(1)), "depth_m": depth, "text": t})
            continue
        m = re.match(r"^R\.?L\.?\s*[:=]?\s*(\d{2,4}\.\d{1,3})", t, re.I)
        if m:
            rls.append(float(m.group(1)))
            continue
        if re.search(r"SOIL|SAND|CLAY|ROCK|GRAVEL|MURUM|MOORUM|SILT|BOULDER|STRATA|BASALT|SHALE", t, re.I) and not SCALE_RE.match(t) \
                and "BORE" not in t.upper() and l not in view.get("title_lines", []):
            if layers and centre(l["bbox"])[1] - layers[-1]["y"] <= 13:
                layers[-1]["soil"] += " " + t
                layers[-1]["y"] = centre(l["bbox"])[1]
            else:
                layers.append({"soil": t, "y": centre(l["bbox"])[1]})
    for x in layers:
        x.pop("y")
    m = re.search(r"CH\.?\s*[:\-]?\s*([\d+.]+)\s*M?|AT\s*([\d+.]+)\s*M", view["title"], re.I)
    return {"view": view["title"], "chainage": (m.group(1) or m.group(2)) if m else None, "sbc": sbc, "layers": layers, "rl_marks": rls}


BAND_KINDS = ("road_lsection", "drain_lsection", "ground_profile")
BAND_LABEL = re.compile(r"LEVEL|LVL|CHAINAGE|OFFSET|DISTANCE|\bR\.?L\b|DEPTH", re.I)
BAND_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_bands(lines, views, exclude):
    """The value tables under road / drain L-sections and ground profiles: rows labelled at the left ("ROAD LEVEL (M)",
    "INVERT LVL OF DRAIN", "GROUND LEVEL", "CHAINAGE", "OFFSET"), one value per column, often written upright.
    Read over the whole sheet (a band's labels and values often fall in a neighbouring view's box), then each band goes to
    the L-section / profile view whose title stands under it. Returns {view index: band}, ids of the value texts."""
    pool = [l for l in lines if l["i"] not in exclude]
    labels = [l for l in pool if l["dir"] == (1, 0) and BAND_LABEL.search(l["text"]) and not BAND_NUM.fullmatch(l["text"].strip())
              and len(l["text"]) < 40 and not re.search(r"\d{3}\.\d|:", l["text"])]
    nums = []
    for l in pool:
        t = l["text"].strip()
        if BAND_NUM.fullmatch(t):
            nums.append(l)
        elif re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)+", t):
            # two values of one column run together into one text ("81360 453.296" written upright: chainage below,
            # ground level above): each part where it stands along the text
            x0, y0, x1, y1 = l["bbox"]
            for m in re.finditer(r"\S+", t):
                f0, f1 = m.start() / len(t), m.end() / len(t)
                if l["dir"] == (0, -1):                       # upright, read bottom to top
                    bb = (x0, y1 - f1 * (y1 - y0), x1, y1 - f0 * (y1 - y0))
                elif l["dir"] == (0, 1):
                    bb = (x0, y0 + f0 * (y1 - y0), x1, y0 + f1 * (y1 - y0))
                else:
                    bb = (x0 + f0 * (x1 - x0), y0, x0 + f1 * (x1 - x0), y1)
                nums.append({"i": l["i"], "text": m.group(0), "bbox": bb, "dir": l["dir"]})
    near = defaultdict(list)
    for n in nums:
        cy = centre(n["bbox"])[1]
        cand = [(abs(centre(b["bbox"])[1] - cy), k) for k, b in enumerate(labels) if n["bbox"][0] > b["bbox"][2] - 2]
        if cand and min(cand)[0] < 20:
            near[min(cand)[1]].append(n)
    rows = []
    for k, ns in near.items():
        chain, last = [], labels[k]["bbox"][2]
        for n in sorted(ns, key=lambda n: n["bbox"][0]):
            if n["bbox"][0] - last > (220 if not chain else 150):
                break
            chain.append(n)
            last = n["bbox"][2]
        if len(chain) >= 3:
            rows.append({"label": re.sub(r"\s+", " ", labels[k]["text"]).strip(), "lab": labels[k], "vals": chain})
    # rows stacked under one another with their labels in one column make one band
    bands = []
    for r in sorted(rows, key=lambda r: r["lab"]["bbox"][1]):
        b = next((b for b in bands if abs(b[-1]["lab"]["bbox"][0] - r["lab"]["bbox"][0]) < 60
                  and 0 < r["lab"]["bbox"][1] - b[-1]["lab"]["bbox"][1] < 70), None)
        if b:
            b.append(r)
        else:
            bands.append([r])
    out, ids = {}, set()
    for b in bands:
        box = union([r["lab"]["bbox"] for r in b] + [n["bbox"] for r in b for n in r["vals"]])
        cand = []
        for vi, v in enumerate(views):
            if v["kind"] not in BAND_KINDS:
                continue
            tb = union([l["bbox"] for l in lines if l["i"] in v["title_lines"]]) if v.get("title_lines") else v["bbox"]
            if box[0] - 30 <= centre(tb)[0] <= box[2] + 30 and -10 <= tb[1] - box[3] <= 120:
                cand.append((tb[1] - box[3], vi))
        if not cand:
            continue
        vi = min(cand)[1]
        cols = []
        for r in b:
            for n in r["vals"]:
                x = centre(n["bbox"])[0]
                c = next((c for c in cols if abs(c["_x"] - x) < 8 and r["label"] not in c), None)
                if c is None:
                    c = {"_x": x}
                    cols.append(c)
                c[r["label"]] = n["text"].strip()
        cols.sort(key=lambda c: c["_x"])
        for c in cols:
            c.pop("_x")
        band = out.setdefault(vi, {"rows": [], "columns": []})
        band["rows"] += [{"label": r["label"], "values": [n["text"].strip() for n in r["vals"]]} for r in b]
        band["columns"] += cols
        ids |= {n["i"] for r in b for n in r["vals"]}
    return out, ids

# people are never read from a GAD: the names come from its own digital signatures ("<NAME> Digitally signed by <NAME>")
SIGNED_BY = re.compile(r"signed\s+by\s*:?\s*([A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*){0,3})|^\s*([A-Z][A-Z.]+(?:\s+[A-Z][A-Z.]+){0,3})\s+Digitally\b")
DESIGNATION = re.compile(r"^\(?\s*(SSE|JE|AEN|XEN|AXEN|DEN|ADEN|SR\.?\s*DEN|SRDEN|DYCE|DY\.?\s*CE|CE|CPM|ADRM|DRM|EE|AEE|PCE|CAO)\b[-A-Z0-9/ .()]*\)?\s*$")
NOT_NAMES = {"DIGITALLY", "SIGNED", "DATE", "BY", "THE", "AND"}


def private_lines(lines):
    """Ids of the lines that are about people: digital signatures, the names in them wherever else they are printed,
    and the officials' designations."""
    words = set()
    for l in lines:
        for m in SIGNED_BY.finditer(l["text"]):
            words |= {w.strip(".") for w in (m.group(1) or m.group(2)).upper().split() if len(w.strip(".")) >= 4} - NOT_NAMES
    name_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, words))) + r")\b", re.I) if words else None
    return {l["i"] for l in lines if (SIGNATURE.search(l["text"]) and not re.fullmatch(r"\s*signature\s+block\s*", l["text"], re.I))
            or DESIGNATION.match(l["text"].strip()) or re.search(r"Date\s*:\s*20\d\d\.\d\d\.\d\d\s+\d\d:\d\d", l["text"])
            or (name_re and name_re.search(l["text"]))}


# ======================================================================== one GAD
def gad_id_of(path):
    m = re.search(r"GAD[- ]+(.+?)_V(\d+)", Path(path).stem)
    return (re.sub(r"\s+", "", m.group(1)), int(m.group(2))) if m else (Path(path).stem, None)


def annotate(pdf):
    doc = pymupdf.open(pdf)
    page = doc[0]
    if page.rotation:                                  # some sheets are stored rotated: read them as they are seen
        page.remove_rotation()
    lines = load_lines(page)
    private = private_lines(lines)
    sigs = [l for l in lines if SIGNATURE.search(l["text"])]
    pboxes = [l["bbox"] for l in lines if l["i"] in private]
    lines = [l for l in lines if l["i"] not in private]
    words = [w for w in page.get_text("words") if not any(inside(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2), b) for b in pboxes)]
    heads = find_headings(lines)
    regions = heading_regions(lines, heads, page.rect.width, page.rect.height)
    panel_boxes = [r["box"] for r in regions]
    tb = parse_title_block(lines, page)
    div = next((l for l in lines if re.match(r"^DIVISION\b", l["text"].upper())), None)
    skip = list(panel_boxes)
    if div:
        skip.append([div["bbox"][0] - 8, div["bbox"][1] - 60, page.rect.width, page.rect.height])
    if sigs:
        # one box per group of signatures that sit together: one box round all of them reached over the drawing
        # between them (812-1: it hid the bore log)
        groups = []
        for l in sorted(sigs, key=lambda l: (l["bbox"][1], l["bbox"][0])):
            g = next((g for g in groups if gap(union(g), l["bbox"]) < 40), None)
            if g:
                g.append(l["bbox"])
            else:
                groups.append([l["bbox"]])
        skip += [[u[0] - 60, u[1] - 10, u[2] + 60, u[3] + 10] for u in (union(g) for g in groups)]   # (the names beside them too)
    # the strip of boxes along the bottom edge (contractor, design consultant, client, authority engineer: logos, names,
    # addresses, signatures) starts at a long rule above its labels: none of it is drawing
    strip_labels = [l for l in lines if re.fullmatch(r"\(?(CONTRACTOR|DESIGN\s+CONSULTANTS?|CLIENT|AUTHORITY\s+ENGINEER)\)?", l["text"].strip(), re.I)]
    if strip_labels:
        rules = [(min(it[1].x, it[2].x), it[1].y, max(it[1].x, it[2].x)) for d in page.get_drawings() for it in d["items"]
                 if it[0] == "l" and abs(it[1].y - it[2].y) < 0.6 and abs(it[2].x - it[1].x) > 400]
        for l in strip_labels:
            cx = centre(l["bbox"])[0]
            over = [r for r in rules if r[0] - 2 <= cx <= r[2] + 2 and l["bbox"][1] - 160 <= r[1] <= l["bbox"][1] - 40]   # (not a band table's rule higher up)
            if over:
                x0, y, x1 = max(over, key=lambda r: r[1])          # the one right above the cells
                skip.append([x0 - 2, y - 2, x1 + 2, page.rect.height])
            # and the cell above each label (the name, firm or logo it labels), wherever the strip's rules are
            skip.append([l["bbox"][0] - 40, l["bbox"][1] - 130, l["bbox"][2] + 40, min(page.rect.height, l["bbox"][3] + 40)])
    # a small box drawn round a panel item (the depth of track structure): its frame is not view line work
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 600 and r.height < 600 and any(r.x0 - 1 <= h["heading_box"][0] and h["heading_box"][2] <= r.x1 + 1
                                                    and r.y0 - 1 <= h["heading_box"][1] and h["heading_box"][3] <= r.y1 + 1 for h in regions):
            skip.append([r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2])
    titles = find_view_titles(lines, panel_boxes)
    boxes, (lab, comp_view) = view_regions(page, lines, titles, skip)
    GRID.update(lab=lab, comp_view=comp_view)
    views = []
    for t, box in zip(titles, boxes):
        kind, what = K.view_kind(t["title"])
        views.append({"title": t["title"], "kind": kind, "scale": t["scale"], "represents": what, "bbox": [round(v, 1) for v in box],
                      "title_lines": [l["i"] for l in t["lines"]]})
    panel_line_ids = {l["i"] for r in regions for l in r["lines"]} | {l["i"] for l in lines if any(inside(centre(l["bbox"]), b) for b in skip)}
    title_line_ids = {i for v in views for i in v["title_lines"]}
    exclude = panel_line_ids | title_line_ids
    # a view titled only "SECTIONAL ELEVATION" next to "SECTIONAL ELEVATION AT A-A", labelled "EXISTING ...", is the
    # elevation of the existing bridge
    if any(re.search(r"\bA\s*-\s*A\b", v["title"]) for v in views if v["kind"] == "sectional_elevation"):
        for v in views:
            if v["kind"] == "sectional_elevation" and not re.search(r"\b[A-Z]\s*-\s*[A-Z]\b", v["title"]):
                txt = " ".join(l["text"] for l in lines if l["i"] not in exclude and assign_view(centre(l["bbox"]), views) is v)
                if re.search(r"\bEXIST(ING)?\b|\bEX\.", txt, re.I):
                    v["kind"] = "existing_bridge_section"
                    v["represents"] = K.view_kind("SECTIONAL ELEVATION OF EXISTING BRIDGE")[1] + \
                        " (Its title says only 'SECTIONAL ELEVATION'; its labels show it is the existing bridge.)"

    bands, band_ids = parse_bands(lines, views, exclude)
    for vi, band in bands.items():
        views[vi]["band"] = band
    levels, used = read_levels(lines, views, exclude | band_ids)
    cls = centre_lines(lines, views)
    cl_ids = {c["line"] for c in cls} | {l["i"] for l in lines if l["text"].strip() == "L"}
    labelled, plain, slopes = read_dims(lines, views, cls, exclude | used | cl_ids | band_ids)
    for v in views:
        v_lines = [l for l in lines if l["i"] not in exclude and assign_view(centre(l["bbox"]), views) is v]
        v["texts"] = [re.sub(r"\s+", " ", l["text"].replace("|", " ")).strip() for l in sorted(v_lines, key=lambda l: (round(centre(l["bbox"])[1] / 6), l["bbox"][0]))]
    bore_logs = [read_bore_log(v, lines, skip) for v in views if v["kind"] == "bore_log"]
    for v in views:
        if v["kind"] == "key_plan":
            v["key_plan"] = parse_key_plan(v, lines)

    notes = {}
    tables = {}
    order = {"comparative_table": 0, "track_details": 1}
    for r in sorted(regions, key=lambda r: order.get(r["kind"], 2)):
        k = r["kind"]
        if k in ("notes", "special_note", "add_note", "design_criteria", "fill_note", "reference_drawings"):
            notes.setdefault(k, []).extend(numbered(r["lines"]))
        elif k == "specifications":
            notes[k] = parse_specifications(r["lines"])
        elif k == "abbreviations":
            notes[k] = parse_abbreviations(r["lines"])
        elif k == "legends":
            notes[k] = [re.sub(r"\s+", " ", x["text"]) for x in rows_of(r["lines"])]
        elif k in ("comparative_table", "track_details"):
            t = parse_table(r, words, cols_from=(tables.get("comparative_table") or {}).get("_cols"))
            if len(t["rows"]) >= len((tables.get(k) or {}).get("rows", [])):    # the same heading met again elsewhere
                tables[k] = t
        elif k == "depth_of_track_structure":
            tables[k] = parse_key_values(r["lines"])
    for key, pat in (("seismic_zone", r"^SEISMIC\s+ZONE\s*:\s*(.*)$"), ("standard_of_loading", r"^STANDARD\s+OF\s+LOADING\s*:\s*(.*)$")):
        lab = next((l for l in lines if l["dir"] == (1, 0) and re.match(pat, l["text"].upper().replace("|", " ").strip())), None)
        if lab:
            cy = centre(lab["bbox"])[1]
            row = rows_of([o for o in lines if abs(centre(o["bbox"])[1] - cy) < 4 and lab["bbox"][0] <= o["bbox"][0] < lab["bbox"][0] + 400])
            text = re.sub(r"\s+", " ", row[0]["text"].replace("|", " ")).strip() if row else lab["text"]
            m = re.match(pat, text, re.I)
            if m and m.group(1).strip(" :"):
                notes[key] = m.group(1).strip(" :")
    for t in tables.values():
        if isinstance(t, dict):
            t.pop("_cols", None)

    gid, ver = gad_id_of(pdf)
    bridge = bridge_from_title(tb.get("title", ""))
    tb["revisions"] = parse_revisions(lines)
    ann = {"gad_id": gid, "version": ver, "source_pdf": Path(pdf).name, "page_size": [page.rect.width, page.rect.height],
           "title_block": tb, "bridge": bridge, "views": views, "levels": levels, "centre_lines": cls,
           "labelled_dims": labelled, "slopes": slopes, "dims": plain, "tables": tables, "bore_logs": bore_logs, "notes": notes,
           "panel": [{"kind": r["kind"], "box": [round(v, 1) for v in union([r["heading_box"]] + [l["bbox"] for l in r["lines"]])]}
                     for r in regions] + ([{"kind": "title_block", "box": [round(v, 1) for v in skip[len(panel_boxes)]]}] if div else [])}
    ann["findings"] = findings(ann)
    ann["counts"] = {"views": len(views), "levels": len(levels), "labelled_dims": len(labelled), "dims": len(plain),
                     "notes": len(notes.get("notes", [])), "table_rows": sum(len(t["rows"]) for t in tables.values() if isinstance(t, dict))}
    return ann


def findings(ann):
    """Values that disagree between places on the same drawing (each place is kept as printed)."""
    out = []
    box = (ann["bridge"].get("box") or {})
    comp = {r["kind"]: r for r in (ann["tables"].get("comparative_table") or {}).get("rows", [])}
    span = (comp.get("span") or {}).get("proposed")
    if box and span:
        m = re.search(r"(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)", span)
        if m and (int(m.group(1)), float(m.group(2)), float(m.group(3))) != (box["cells"], box["clear_width_m"], box["clear_height_m"]):
            out.append(f"box size: title says {box['as_printed']}, comparative table (proposed) says {span}")
    track = {r["kind"]: r for r in (ann["tables"].get("track_details") or {}).get("rows", [])}
    for kind, lvl_kind in (("rail_level", "rail_level"), ("formation_level", "formation_level")):
        tv = (track.get(kind) or {}).get("proposed")
        vals = {l["value"] for l in ann["levels"] if l["kind"] == lvl_kind and l["status"] == "proposed"}
        if tv and vals:
            try:
                f = float(re.sub(r"[^\d.]", "", tv))
                if all(abs(f - v) > 0.0015 for v in vals):
                    out.append(f"proposed {kind.replace('_', ' ')}: track details say {tv}, the views show {', '.join(f'{v:.3f}' for v in sorted(vals))}")
            except ValueError:
                pass
    for kind, row_kinds in (("bed_level", ("bed_level",)), ("hfl", ("hfl", "observed_hfl", "calculated_hfl"))):
        tvals = []
        for rk in row_kinds:
            tv = (comp.get(rk) or {}).get("proposed")
            try:
                tvals.append((rk, tv, float(re.sub(r"[^\d.]", "", tv.split()[0]))))
            except (AttributeError, ValueError, IndexError):
                pass
        vals = {l["value"] for l in ann["levels"] if l["kind"] == kind and l["status"] != "existing"}
        if tvals and vals:
            # views agree if every value they show is one of the table's values (OHFL or CHFL for the HFL)
            off = [v for v in vals if all(abs(f - v) > 0.0015 for _, _, f in tvals)]
            if off:
                tab = ", ".join(f"{rk.replace('_', ' ').replace('observed hfl', 'OHFL').replace('calculated hfl', 'CHFL')} {tv}" for rk, tv, _ in tvals)
                out.append(f"{kind.replace('_', ' ').replace('hfl', 'HFL')}: comparative table says {tab}, the views show "
                           f"{', '.join(f'{v:.3f}' for v in sorted(vals))}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf")
    args = ap.parse_args()
    if args.pdf:
        print(json.dumps(annotate(args.pdf), indent=1, ensure_ascii=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for pdf in sorted(GAD_DIR.glob("*.pdf")):
        ann = annotate(pdf)
        (OUT / f"{ann['gad_id']}.json").write_text(json.dumps(ann, indent=1, ensure_ascii=False), encoding="utf-8")
        stats.update(ann["counts"])
        print(f"{ann['gad_id']:<12} views {ann['counts']['views']:>2}  levels {ann['counts']['levels']:>3}  "
              f"labelled dims {ann['counts']['labelled_dims']:>3}  dims {ann['counts']['dims']:>3}  notes {ann['counts']['notes']:>2}  "
              f"table rows {ann['counts']['table_rows']:>2}  findings {len(ann['findings'])}")
    print("total", dict(stats))


if __name__ == "__main__":
    main()
