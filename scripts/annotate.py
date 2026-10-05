"""Auto-annotate the Plan & L-Section sheets from the PDF's vector text layer.

Every callout, curve box, table and title-block field on these sheets is real
text with coordinates, so the annotations are read from the PDF rather than
drawn by hand. Writes one JSON per sheet to data/annotations/. All boxes are in
PDF points (1/72 inch), origin at the top-left of the page, as [x0, y0, x1, y1].

    python scripts/annotate.py [path/to/drawing.pdf]
"""
import json
import math
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "MKN_PNP_1210-1251_3rd Line.pdf"
OUT_DIR = ROOT / "data" / "annotations"

# Page template shared by every sheet of this drawing set (A0, 3770 x 2384 pt).
DRAWING_RIGHT = 3165      # main drawing area ends here; the right-hand panel starts
PANEL = (3175, 3740)      # x-range of the notes / tables / title block panel
PLAN_BOTTOM = 1075        # alignment plan above, L-section below
BANDS_TOP = 1600          # profile graph above, data bands (levels, chainages) below
BANDS_BOTTOM = 2300


# ---------------------------------------------------------------- text lines

def page_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()
            text = re.sub(r"\s+\.$", "", text)          # "CURVE No. 345        ." -> "CURVE No. 345"
            out.append({
                "text": text,
                "bbox": [round(v, 1) for v in line["bbox"]],
                "dir": line["dir"],
                "origin": spans[0]["origin"],
                "color": spans[0]["color"],
            })
    return out


def horizontal(line):
    return abs(line["dir"][1]) < 0.02 and line["dir"][0] > 0


def union(boxes):
    boxes = list(boxes)
    return [round(min(b[0] for b in boxes), 1), round(min(b[1] for b in boxes), 1),
            round(max(b[2] for b in boxes), 1), round(max(b[3] for b in boxes), 1)]


def inside(box, region):
    return box[0] >= region[0] and box[1] >= region[1] and box[2] <= region[2] and box[3] <= region[3]


def frame(line):
    """(along, offset) of a line's origin: along its reading direction, and down
    its text stack (successive lines of one callout have increasing offset)."""
    dx, dy = line["dir"]
    ox, oy = line["origin"]
    return ox * dx + oy * dy, ox * -dy + oy * dx


def stack(anchor, lines, max_gap, max_total, stop=None):
    """Lines that continue a multi-line label starting at `anchor`, in reading
    order: same direction, below it in text space, overlapping it along the
    reading axis, with no gap wider than `max_gap` between consecutive lines."""
    a_along, a_off = frame(anchor)
    a_len = len(anchor["text"]) * 6.0
    cands = []
    for ln in lines:
        if ln is anchor:
            continue
        if ln["dir"][0] * anchor["dir"][0] + ln["dir"][1] * anchor["dir"][1] < 0.98:
            continue
        along, off = frame(ln)
        d = off - a_off
        if not (0.5 < d <= max_total):
            continue
        if along + len(ln["text"]) * 6.0 < a_along - 20 or along > a_along + a_len + 20:
            continue
        cands.append((d, ln))
    cands.sort(key=lambda c: c[0])
    group, last = [anchor], 0.0
    for d, ln in cands:
        if d - last > max_gap or (stop and re.match(stop, ln["text"])):
            break
        group.append(ln)
        last = d
    return group


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- bridges

CALLOUT_STOP = r"C/L OF|TBM|Near Railway|(ST|TS|SC|CS|TC|CT|TP1|TP2|J1|J2) AT CH|KM"

def norm_id(bid):
    """'LC 261' -> 'LC-261', so callout and level block agree however the sheet spaces it."""
    return re.sub(r"^LC\s+", "LC-", bid) if bid else bid


def parse_callout(text):
    """'C/L OF EXG. BR. NO. 516 RCC SLAB - 1 X 3.66 - STREAM PRO TO BE EXTENDED
    AS 1 X 4 X 4.58 - RCC BOX (MINOR) AT CH: 1211678.925' -> fields."""
    m = re.match(r"C/L OF EXG\.?\s*(?:BR\.?\s*NO\.?\s*(?P<no>LC[ -]?\d+\w*|[\w-]+(?:\([\w -]+\))?)"
                 r"|(?P<rob>ROB(?:-\d+)?)\b|(?P<lc>LC[ -]?\d+\w*))?\s*(?P<rest>.*)", text)
    if not m:
        return None
    bridge_id = norm_id(m.group("no") or m.group("rob") or m.group("lc"))
    rest = m.group("rest")
    cm = re.search(r"AT CH:\s*([\d.]+)", text)
    if bridge_id == "ROB":                      # an ROB printed without a number: name it by its chainage
        bridge_id = f"ROB at CH {num(cm.group(1))}" if cm else None
    on_line = None
    if rest.startswith(("UP ", "DN ")):          # existing structure on the UP / DN line ("DN RCC SLAB - ...")
        on_line, rest = f"existing {rest[:2]} line", rest[3:]
    # Callouts that need no extension
    no_ext = None
    if "EXISTING STRUCTURE IS SUFFICIENT" in rest:
        no_ext = "none - existing structure is sufficient"
        rest = rest.replace("EXISTING STRUCTURE IS SUFFICIENT", "")
    elif "SPACE AVAILABLE" in rest and not m.group("rob"):
        note = re.search(r"(ROB PROPOSED BY [\w.]+?)\.?\s*SPACE", rest)
        no_ext = "none - space available for proposed line" + (f" ({note.group(1).lower()})" if note else "")
        rest = re.sub(r"(ROB PROPOSED BY [\w.]+\s*)?SPACE AVAILABLE FOR PROP\. LINE", "", rest)
    existing, _, after = rest.partition("PRO TO BE EXTENDED AS")
    existing = re.sub(r"\s*AT CH:.*$", "", existing).strip()
    ex_type = ex_span = crossing = proposal = category = None
    span_re = r"\d+\s*X\s*[\d.]+(?:\s*\+\s*\d+\s*X\s*[\d.]+)*"
    rail = re.match(rf"^-?\s*RAIL PRO\s+(?P<span>{span_re})\s*-\s*(?P<type>\w+)\s*\((?P<cat>[^)]*)\)", existing)
    if m.group("rob"):
        # "C/L OF EXG. ROB-224 MDR SPACE AVAILABLE FOR PROP. LINE AT CH: ..." - nothing to extend.
        road, _, tail = existing.partition("SPACE AVAILABLE")
        ex_type, crossing = "ROB", road.strip() or None
        if tail:
            proposal = "none - space available for proposed line"
    elif rail:
        # "572B - RAIL PRO 1 X 45.7 - OWG (ROR)": a new girder carrying rail over rail.
        crossing = "RAIL"
        proposal, category = f"{rail.group('span')} - {rail.group('type')}", rail.group("cat")
    elif existing.startswith("-"):
        pass                                    # placeholder callout ("LC-231 - - BT ROAD"): no data
    elif m.group("lc") and not re.search(span_re, existing):
        ex_type, crossing = "LEVEL CROSSING", existing.strip(" -") or None
    else:
        sm = re.search(span_re, existing)
        if sm:
            ex_type = existing[:sm.start()].strip(" -") or None
            ex_span = sm.group()
            crossing = existing[sm.end():].strip(" -") or None
        else:                                   # "LEVEL CROSSING - - MDR"
            parts = [p for p in re.split(r"\s*-\s*", existing) if p]
            ex_type = parts[0] if parts else None
            crossing = " ".join(parts[1:]) or None
    if after:
        pm = re.match(r"\s*(?P<prop>.*?)\s*(?:\((?P<cat>[^)]*)\))?\s*(?:AT CH:\s*(?P<ch>[\d.]+))?\s*$", after)
        if pm and re.search(r"\d\s*X", pm.group("prop")):
            proposal = pm.group("prop").strip(" -")
            category = pm.group("cat") or None
    chainage = num(cm.group(1)) if cm else None
    if no_ext:
        proposal = no_ext
    return {
        "bridge_id": bridge_id,
        "existing_on": on_line,
        "existing_type": ex_type,
        "existing_span": ex_span,
        "crossing": crossing,
        "proposal": proposal,
        "category": category,
        "chainage_m": chainage,
    }


LEVEL_KEYS = {"EXG FL": "existing_formation_level", "MIN FL REQ.": "min_formation_level_required",
              "MIN FL REQ": "min_formation_level_required", "FL": "proposed_formation_level",
              "B.L": "bed_level", "HFL": "high_flood_level", "FB": "free_board"}


def bridges(lines):
    lsec_region = [0, PLAN_BOTTOM - 150, DRAWING_RIGHT, BANDS_TOP]
    by_id = {}

    def entry(bid):
        return by_id.setdefault(bid, {"bridge_id": bid})

    # Callouts: one on the plan (near-vertical text), one on the L-section (45 deg).
    for ln in lines:
        if not ln["text"].startswith("C/L OF") or horizontal(ln):
            continue
        group = stack(ln, lines, max_gap=24, max_total=70, stop=CALLOUT_STOP)
        text = " ".join(g["text"] for g in group)
        fields = parse_callout(text)
        if not fields or not fields["bridge_id"]:
            continue
        box = union(g["bbox"] for g in group)
        where = "lsection_callout" if box[1] > PLAN_BOTTOM - 150 and box[3] < BANDS_TOP else "plan_callout"
        b = entry(fields["bridge_id"])
        b[where] = {"bbox": box, "text": text}
        # The L-section callout is the complete one; the plan copy can be cut at the sheet edge.
        for k, v in fields.items():
            if v is not None and (where == "lsection_callout" or b.get(k) is None):
                b[k] = v

    # Level blocks on the L-section: "EXG. BR. NO. 516" followed by EXG FL, MIN FL REQ, FL, B.L, HFL, FB.
    for ln in lines:
        m = re.match(r"^EXG\.?\s*(?:BR\.?\s*NO\.?\s*(?P<no>LC[ -]?\d+\w*|\S+)?(?:\s+(?:UP|DN))?|(?P<rob>ROB-\d+)|(?P<lc>LC[ -]?\d+))$",
                     ln["text"])
        if not (m and horizontal(ln) and inside(ln["bbox"], lsec_region)):
            continue
        group = [ln]
        bid = m.group("no") or m.group("rob") or m.group("lc")
        if not bid:                             # "EXG. BR. NO." with the id wrapped onto the next line
            nxt = next((o for o in lines if horizontal(o) and abs(o["bbox"][0] - ln["bbox"][0]) < 4
                        and 0 < o["bbox"][1] - ln["bbox"][1] < 14), None)
            if not nxt:
                continue
            bid = re.sub(r"\s+UP$", "", nxt["text"])
            group.append(nxt)
            ln = nxt                            # the level lines start below the wrapped id
        bid = norm_id(bid)
        levels = {}
        for other in sorted(lines, key=lambda o: o["bbox"][1]):
            if (other is not ln and horizontal(other) and abs(other["bbox"][0] - ln["bbox"][0]) < 4
                    and 0 < other["bbox"][1] - ln["bbox"][1] < 100):
                km = re.match(r"^(EXG FL|MIN FL REQ\.?|FL|B\.L|HFL|FB)\s*=\s*(-?[\d.]+)", other["text"])
                if not km:
                    break
                levels[LEVEL_KEYS[km.group(1)]] = num(km.group(2))
                group.append(other)
        b = entry(bid)
        b["lsection_levels"] = {"bbox": union(g["bbox"] for g in group), **levels}
    # A record is "complete" when its callout gives the structure, the proposal and the chainage;
    # partial ones (cut at the sheet edge, placeholders) are kept but not used for detail questions.
    for b in by_id.values():
        b["complete"] = bool(b.get("chainage_m") and b.get("proposal") and (b.get("existing_type") or b.get("crossing")))
    return sorted(by_id.values(), key=lambda b: (b.get("chainage_m") or 0, b["bridge_id"]))


# ---------------------------------------------------------------- curves

CURVE_KEYS = {"Degree": "degree", "Δ": "deflection_angle", "R": "radius", "Radius": "radius",
              "TTL": "total_tangent_length", "TL": "tangent_length", "CCL": "circular_curve_length",
              "TRL": "transition_length", "TCL": "total_curve_length", "Ca": "cant", "Cant": "cant",
              "Shift": "shift", "Vmax": "max_speed", "Speed": "speed"}


def curves(lines):
    out = []
    for ln in lines:
        m = re.match(r"^(?:3RD LINE )?CURVE No\.\s*(\w+)", ln["text"], re.I)
        if not m:
            continue
        existing = ln["text"].startswith("Curve")       # black "Curve No. 84C" = existing line
        group = stack(ln, lines, max_gap=20, max_total=150, stop=r"CURVE No|Curve No")
        params, hand = {}, None
        for g in group[1:]:
            km = re.match(r"^(\S+)\s*=\s*(.+)$", g["text"])
            if km and km.group(1) in CURVE_KEYS:
                params[CURVE_KEYS[km.group(1)]] = km.group(2).strip()
        # "(LH)"/"(RH)" is a separate line sitting on the header.
        for o in lines:
            if o["text"] in ("(LH)", "(RH)") and abs(o["bbox"][1] - ln["bbox"][1]) < 4 \
                    and ln["bbox"][0] < o["bbox"][0] < ln["bbox"][2] + 10:
                hand = o["text"].strip("()")
        if "curve no" not in ln["text"].lower() or not params:
            continue
        box = union(g["bbox"] for g in group)
        if box[1] >= BANDS_TOP:
            where = "lsection_band"
        elif box[3] <= PLAN_BOTTOM:
            where = "plan"
        else:
            continue
        out.append({"curve_no": m.group(1), "line": "existing" if existing else "proposed 3rd line",
                    "hand": {"LH": "left hand", "RH": "right hand"}.get(hand), "location": where,
                    "bbox": box, "params": params})
    return out


# ---------------------------------------------------------------- L-section data bands

# The eight numeric rows at the bottom of the L-section, top to bottom, one column every 20 m.
BAND_ROWS = ["cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl",
             "ground_level", "chainage"]


def bands(lines):
    """Every column of the data bands: {chainage, x, values{row: number}, checks_ok}.

    The numbers are vertical text. Rows are found by clustering their heights (the row order is
    fixed on these sheets) and columns by lining each number up under its chainage."""
    cut = next((l for l in lines if l["text"].startswith("CUT(-)") and horizontal(l)), None)
    if not cut:
        return None
    km = [l["bbox"][1] for l in lines if l["text"].startswith("KM") and horizontal(l) and l["bbox"][1] > cut["bbox"][1]]
    y0, y1 = cut["bbox"][1] - 8, (min(km) - 2 if km else BANDS_BOTTOM)
    x_min = cut["bbox"][2] + 5
    cells = []
    for l in lines:
        b = l["bbox"]
        if abs(l["dir"][0]) < 0.05 and y0 <= b[1] and b[3] <= y1 and b[0] > x_min and re.fullmatch(r"-?\d+(\.\d+)?", l["text"]):
            cells.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, l["text"], b))
    rows = []
    for c in sorted(cells, key=lambda c: c[1]):
        if rows and c[1] - rows[-1][-1][1] < 12:
            rows[-1].append(c)
        else:
            rows.append([c])
    rows = [r for r in rows if len(r) > 5]                 # drop stray numbers
    if len(rows) != len(BAND_ROWS):
        return {"error": f"found {len(rows)} numeric rows, expected {len(BAND_ROWS)}"}
    row_y = [[round(min(c[3][1] for c in r) - 2, 1), round(max(c[3][3] for c in r) + 2, 1)] for r in rows]
    columns = []
    for cx, cy, text, b in sorted(rows[-1], key=lambda c: c[0]):
        vals, boxes = {}, [b]
        for name, r in zip(BAND_ROWS[:-1], rows[:-1]):
            near = min(r, key=lambda c: abs(c[0] - cx))
            if abs(near[0] - cx) < 4:
                vals[name] = float(near[2])
                boxes.append(near[3])
        vals["chainage"] = float(text)
        # checks_ok: the printed numbers agree with each other (FL - GL = cut/fill, FL - existing FL = difference),
        # so the column was read correctly and can be used. RL - FL against the sheet's rail-level note (762 mm)
        # is a check on the drawing itself, kept separately: some 4th-line sheets print 0.764-0.765.
        ok = all(k in vals for k in BAND_ROWS) and (
            abs(vals["prop_fl"] - vals["ground_level"] - vals["cut_fill"]) < 0.0025
            and abs(vals["prop_fl"] - vals["exg_up_fl"] - vals["fl_difference"]) < 0.0025)
        col = {"chainage": vals["chainage"], "x": round(cx, 1), "values": vals, "checks_ok": ok, "bbox": union(boxes)}
        if "prop_rl" in vals and "prop_fl" in vals:
            col["rl_minus_fl"] = round(vals["prop_rl"] - vals["prop_fl"], 3)
            col["rail_level_ok"] = abs(col["rl_minus_fl"] - 0.762) < 0.0025
        columns.append(col)
    # Row labels are centred in the label column, so take every label that starts left of the numbers
    # (the widest ones, such as DIFFERENCE BETWEEN, reach further right than the CUT label).
    labels = [l for l in lines if horizontal(l) and l["bbox"][0] < x_min and l["bbox"][2] < x_min + 40
              and y0 <= l["bbox"][1] <= y1]
    label_text = " ".join(l["text"] for l in labels).upper()
    prop = re.search(r"PROP\.?\s*(\w+)\s*LINE", label_text)
    exg = re.search(r"EXG\.?\s*(\w+)\s*LINE", label_text)
    return {"rows": BAND_ROWS, "row_y": row_y,
            "label_strip": union([l["bbox"] for l in labels] + [[x_min - 5, y0, x_min - 5, y1]]),
            "lines": {"proposed": f"{prop.group(1)} LINE" if prop else None,
                      "existing": f"{exg.group(1)} LINE" if exg else None},
            "columns": columns}


# ---------------------------------------------------------------- everything else on the sheet

# On these sheets the letters read S = straight, T = transition, C = circular curve: a curve runs
# ST -> TC -> CT -> TS, and the gaps equal TRL, CCL, TRL of its curve box (29 of 30 complete curves
# match exactly). Confirmed by Ankur. The sheets' abbreviations table names the same four points
# TTP1, CTP1, CTP2, TTP2, which other drawing sets may print instead.
TRANSITION = {"ST": "straight to transition (curve starts)", "TC": "transition to circular curve",
              "CT": "circular curve to transition", "TS": "transition to straight (curve ends)"}
TRANSITION_ALSO = {"ST": "TTP1 (Tangent to Transition Point-1)", "TC": "CTP1 (Curve Tangent Point-1)",
                   "CT": "CTP2 (Curve Tangent Point-2)", "TS": "TTP2 (Tangent to Transition Point-2)"}


def km_to_m(s):
    """'1240+535.820' or '1240535.820' -> 1240535.82"""
    m = re.match(r"^(\d+)\+([\d.]+)$", s)
    return int(m.group(1)) * 1000 + float(m.group(2)) if m else num(s)


def x_to_chainage(band):
    """Map an x position on the L-section to chainage, using the band columns as a ruler."""
    cols = sorted(((c["x"], c["chainage"]) for c in (band or {}).get("columns", [])), key=lambda c: c[0])
    if len(cols) < 2:
        return lambda x: None

    def f(x):
        if not cols[0][0] <= x <= cols[-1][0]:
            return None
        for (xa, ca), (xb, cb) in zip(cols, cols[1:]):
            if xa <= x <= xb:
                return round(ca + (cb - ca) * (x - xa) / (xb - xa), 1)
    return f


def gradient_words(label):
    """'1:955 F' / '955.000 F' -> '1 in 955 falling'; huge numbers and 'Horizontal' -> 'level'."""
    m = re.match(r"^(?:1:)?([\d.]+)\s*([FR])$", label.strip())
    # The CAD writes level stretches as "1 in" a huge number (1 in 816901, 1 in 34159584552264): treat as level.
    if label.strip().startswith("Horizontal") or (m and float(m.group(1)) > 1e5):
        return "level"
    if not m:
        return None
    n = float(m.group(1))
    return f"1 in {n:g} {'falling' if m.group(2) == 'F' else 'rising'}"


RED = 16711680          # proposed work is drawn in red, existing work in black (notes 2 and 3)


def plan_grade_points(lines, plan_bottom):
    """Grade change symbols on the plan: a stem labelled 'CH: 1211+136.000' / 'FL:196.805' under a bar
    with the gradient before the point on its left and the gradient after it on its right.
    Red symbols are the proposed 3rd line (proposed chainage); black ones the existing UP line, whose
    chainage is in the existing km system (about 40.13 km lower)."""
    def words(label):
        return "level" if label == "LEVEL" else gradient_words(label) if label else None
    out = []
    for l in lines:
        m = re.match(r"^CH:\s*(\d+\+[\d.]+)$", l["text"])
        if not m or horizontal(l) or l["bbox"][3] > plan_bottom:
            continue
        b = l["bbox"]
        sx = (b[0] + b[2]) / 2
        fl = next((o for o in lines if o["text"].startswith("FL:") and not horizontal(o)
                   and 0 < o["bbox"][0] - b[0] < 16 and abs(o["bbox"][3] - b[3]) < 40), None)
        bar = [o for o in lines if o is not l and re.fullmatch(r"(1:)?[\d.]+ [FR]|LEVEL", o["text"])
               and 0 < b[1] - o["bbox"][3] < 40 and abs((o["bbox"][0] + o["bbox"][2]) / 2 - sx) < 110]
        cxs = lambda o: (o["bbox"][0] + o["bbox"][2]) / 2
        left = max((o for o in bar if cxs(o) < sx), key=cxs, default=None)
        right = min((o for o in bar if cxs(o) >= sx), key=cxs, default=None)
        out.append({"line": "proposed 3rd line" if l["color"] == RED else "existing UP line",
                    "chainage": m.group(1), "chainage_m": km_to_m(m.group(1)),
                    "chainage_system": "proposed" if l["color"] == RED else "existing",
                    "fl": num(fl["text"].split(":")[1]) if fl else None,
                    "gradient_before": words(left["text"]) if left else None, "label_before": left["text"] if left else None,
                    "gradient_after": words(right["text"]) if right else None, "label_after": right["text"] if right else None,
                    "bbox": union([b] + [o["bbox"] for o in (fl, left, right) if o])})
    # Each line's points in chainage order: check that the levels agree with the gradient between them.
    for line in ("proposed 3rd line", "existing UP line"):
        pts = sorted({p["chainage_m"]: p for p in out if p["line"] == line}.values(), key=lambda p: p["chainage_m"])
        for a, c in zip(pts, pts[1:]):
            if a["fl"] is None or c["fl"] is None:
                continue
            dist, rise = c["chainage_m"] - a["chainage_m"], c["fl"] - a["fl"]
            a["next_point_chainage_m"], a["level_change_to_next_m"] = c["chainage_m"], round(rise, 3)
            a["implied_gradient_to_next"] = "level" if abs(rise) < 0.0005 else f"1 in {dist / abs(rise):.0f} {'rising' if rise > 0 else 'falling'}"
    return out


def extras(lines, band, regions):
    """The rest of the sheet's meaningful items, each with its box and, on the L-section, its chainage."""
    ch_at = x_to_chainage(band)
    plan_bottom = regions["alignment_plan"][3]
    out = {"transition_points": [], "km_posts": [], "grade_points": [], "gradient_segments": [],
           "vpis": [], "bearings": [], "km_marks": [], "level_scale": {}, "stations": [], "officers": [],
           "reference_drawings": []}
    for l in lines:
        t, b = l["text"], l["bbox"]
        cx = (b[0] + b[2]) / 2
        m = re.match(r"^(TS|SC|CS|ST|TC|CT|TP1|TP2|J1|J2) AT CH:\s*([\d+.]+)$", t)
        if m:
            # Some drawing sets print TP1 / J1 / J2 / TP2 for the same four points (their abbreviation tables:
            # TP1 = Straight to Transition Point, J1 = Transition to Curve, J2 = Curve to Transition, TP2 = Transition to Straight)
            typ = {"TP1": "ST", "J1": "TC", "J2": "CT", "TP2": "TS"}.get(m.group(1), m.group(1))
            out["transition_points"].append({"type": typ, "printed_as": m.group(1), "meaning": TRANSITION[typ],
                                             "also_called": TRANSITION_ALSO[typ],
                                             "chainage_m": km_to_m(m.group(2)), "view": "plan" if b[3] <= plan_bottom else "L-section",
                                             "text": t, "bbox": b})
            continue
        m = re.match(r"^KM:\s*(\d+\+\d+)$", t)
        if m and b[3] <= plan_bottom:
            out["km_posts"].append({"km": m.group(1), "chainage_m": km_to_m(m.group(1)), "text": t, "bbox": b})
            continue
        m = re.match(r"^(?:GP CH|CH):\s*([\d+.]+)$", t)
        if m and not horizontal(l):
            if b[3] <= plan_bottom:
                continue                    # plan grade points are read as whole symbols (plan_grade_points below)
            # the FL label sits just right of the chainage label, same direction
            fl = min((o for o in lines if o is not l and o["text"].replace(" ", "").startswith("FL:") and not horizontal(o)
                      and 0 < o["bbox"][0] - b[0] < 26 and abs(o["bbox"][3] - b[3]) < 60), default=None,
                     key=lambda o: o["bbox"][0] - b[0])
            if fl and b[1] < regions["lsection_data_bands"][1]:
                out["grade_points"].append({"chainage_m": km_to_m(m.group(1)), "fl": num(fl["text"].split(":")[1]),
                                            "view": "plan" if b[3] <= plan_bottom else "L-section",
                                            "text": f"{t} / {fl['text']}", "bbox": union([b, fl["bbox"]])})
            continue
        if t.startswith("Mg. Bg."):
            out["bearings"].append({"bearing": t.replace("Mg. Bg.", "").strip(), "chainage_m": ch_at(cx), "text": t, "bbox": b})
            continue
        m = re.match(r"^KM : ([\d.]+)$", t)          # proposed km; the existing line's km below it is "KM:1199.869"
        if m:
            exg = next((o for o in lines if re.match(r"^KM:\s*[\d.]+$", o["text"]) and abs(o["bbox"][2] - b[2]) < 12
                        and 0 < o["bbox"][1] - b[1] < 20), None)
            out["km_marks"].append({"proposed_km": m.group(1), "existing_km": exg["text"].split(":")[1] if exg else None,
                                    "chainage_m": round(float(m.group(1)) * 1000, 1), "text": t + (f" / {exg['text']}" if exg else ""),
                                    "bbox": union([b] + ([exg["bbox"]] if exg else []))})
    out["plan_grade_points"] = plan_grade_points(lines, plan_bottom)
    # Existing line's curve tables on the plan (black): a label column, each value printed to its right on the
    # same (slightly tilted) row, so pair them by position along the text's own axes.
    LABEL = r"^(Curve No\.|Degree =|Δ =|Radius =|TL =|TRL =|CCL =|TCL =|Speed =|Cant = .*)$"
    out["existing_curves"] = []
    for anchor in (l for l in lines if l["text"] == "Curve No." and l["bbox"][3] <= plan_bottom):
        a_along, a_off = frame(anchor)
        same_dir = [o for o in lines if o["dir"][0] * anchor["dir"][0] + o["dir"][1] * anchor["dir"][1] > 0.99]
        labels = [o for o in same_dir if re.match(LABEL, o["text"]) and 0 <= frame(o)[1] - a_off < 140 and abs(frame(o)[0] - a_along) < 20]
        item, boxes = {}, []
        for lab in labels:
            l_along, l_off = frame(lab)
            key = lab["text"].split("=")[0].strip().rstrip(".").lower().replace(" ", "_").replace("δ", "deflection_angle")
            if lab["text"].startswith("Cant ="):
                item["cant"] = lab["text"].split("=", 1)[1].strip()
                boxes.append(lab["bbox"])
                continue
            vals = [o for o in same_dir if o not in labels and abs(frame(o)[1] - l_off) < 5 and 0 < frame(o)[0] - l_along < 95]
            if vals:
                v = min(vals, key=lambda o: frame(o)[0])
                item[{"curve_no": "curve_no", "degree": "degree", "deflection_angle": "deflection_angle", "radius": "radius",
                      "tl": "tangent_length", "trl": "transition_length", "ccl": "circular_curve_length",
                      "tcl": "total_curve_length", "speed": "speed"}.get(key, key)] = v["text"]
                boxes += [lab["bbox"], v["bbox"]]
        if item.get("curve_no"):
            out["existing_curves"].append({"line": "existing", **item, "bbox": union(boxes)})
    # Grade points marked under the chainage row: "G.P." with its chainage beside it.
    for l in lines:
        if l["text"] == "G.P." and not horizontal(l):
            ch = next((o for o in lines if re.match(r"^CH: \d+", o["text"]) and not horizontal(o)
                       and 0 < o["bbox"][0] - l["bbox"][0] < 20 and abs(o["bbox"][1] - l["bbox"][1]) < 40), None)
            if ch:
                out["grade_points"].append({"chainage_m": num(ch["text"].split(":")[1]), "fl": None, "view": "chainage band",
                                            "text": f"G.P. {ch['text']}", "bbox": union([l["bbox"], ch["bbox"]])})
    # Vertical schematic rows: a gradient segment is three stacked labels; a VPI is a vertical "FL: 182.67".
    # The row labels sit in the L-section's label column, which starts further right on short sheets.
    label_right = (band or {}).get("label_strip", [0, 0, 400, 0])[2] + 5
    rows = []
    for l in lines:
        if horizontal(l) and l["text"] == "VERTICAL" and l["bbox"][2] <= label_right:
            above = next((o for o in lines if horizontal(o) and abs(o["bbox"][0] - l["bbox"][0]) < 20
                          and 0 < l["bbox"][1] - o["bbox"][1] < 16), None)
            rows.append(("proposed 3rd line" if above and "PROP" in above["text"] else "existing UP line", l["bbox"][1] - 14, l["bbox"][1] + 40))
    for line_name, y0, y1 in rows:
        tops = [l for l in lines if horizontal(l) and y0 <= l["bbox"][1] <= y1 and l["bbox"][0] > label_right
                and (re.fullmatch(r"1:[\d.]+ [FR]", l["text"]) or l["text"].startswith("Horizontal"))]
        for top in tops:
            tx = (top["bbox"][0] + top["bbox"][2]) / 2
            below = sorted((o for o in lines if horizontal(o) and o is not top and abs((o["bbox"][0] + o["bbox"][2]) / 2 - tx) < 12
                            and 0 < o["bbox"][1] - top["bbox"][1] < 26), key=lambda o: o["bbox"][1])
            pct = next((o["text"] for o in below if o["text"].startswith("(")), None)
            length = next((o["text"] for o in below if re.fullmatch(r"[\d.]+ M", o["text"])), None)
            out["gradient_segments"].append({"line": line_name, "gradient_label": top["text"], "gradient": gradient_words(top["text"]),
                                             "percent": pct.strip("()") if pct else None,
                                             "length_m": num(length.split()[0]) if length else None,
                                             "mid_chainage_m": ch_at(tx), "bbox": union([top["bbox"]] + [o["bbox"] for o in below])})
        for l in lines:
            if not horizontal(l) and re.fullmatch(r"FL: [\d.]+", l["text"]) and y0 <= l["bbox"][1] <= y1:
                out["vpis"].append({"line": line_name, "fl": num(l["text"].split(":")[1]),
                                    "chainage_m": ch_at((l["bbox"][0] + l["bbox"][2]) / 2), "text": l["text"], "bbox": l["bbox"]})
    # L-section level scale
    prof = regions["lsection_profile"]
    ticks = [l for l in lines if horizontal(l) and re.fullmatch(r"\d{3}\.\d{2}", l["text"]) and inside(l["bbox"], prof)
             and l["bbox"][0] < prof[0] + 200]
    datum = next((l for l in lines if l["text"].startswith("DATUM")), None)
    scale = next((l for l in lines if l["text"].startswith("V:H")), None)
    if ticks:
        vals = sorted(num(l["text"]) for l in ticks)
        out["level_scale"] = {"datum": datum["text"].split("=")[1] if datum else None,
                              "vertical_to_horizontal": scale["text"].split("=")[1] if scale else None,
                              "lowest_tick": vals[0], "highest_tick": vals[-1],
                              "tick_step": round(vals[1] - vals[0], 2) if len(vals) > 1 else None,
                              "bbox": union([l["bbox"] for l in ticks] + [x["bbox"] for x in (datum, scale) if x])}
    # Stations named at the sheet ends (direction arrows) and along the line
    for l in lines:
        if horizontal(l) and re.fullmatch(r"[A-Z][A-Z .]+ STN\.?", l["text"]):
            near = sorted((o for o in lines if horizontal(o) and o is not l and 0 < o["bbox"][1] - l["bbox"][1] < 40
                           and abs((o["bbox"][0] + o["bbox"][2]) / 2 - (b := l["bbox"])[0] - (b[2] - b[0]) / 2) < 80),
                          key=lambda o: (o["bbox"][1], o["bbox"][0]))
            out["stations"].append({"name": l["text"].rstrip("."), "direction": "left" if l["bbox"][0] < 1500 else "right",
                                    "details": [o["text"] for o in near], "bbox": union([l["bbox"]] + [o["bbox"] for o in near])})
    # Officers (name in brackets, designation below) and reference drawings
    panel = regions["reference_drawings_and_signatures"]
    for l in lines:
        if horizontal(l) and inside(l["bbox"], panel) and re.fullmatch(r"\(.+\)", l["text"]):
            des = next((o for o in lines if horizontal(o) and 0 < o["bbox"][1] - l["bbox"][1] < 18 and abs(o["bbox"][0] - l["bbox"][0]) < 15), None)
            out["officers"].append({"name": l["text"].strip("()"), "designation": des["text"] if des else None,
                                    "bbox": union([l["bbox"]] + ([des["bbox"]] if des else []))})
        if horizontal(l) and inside(l["bbox"], panel) and "DRG" in l["text"] and l["bbox"][1] < panel[1] + 110:
            out["reference_drawings"].append({"drawing": l["text"], "bbox": l["bbox"]})
    return out


# ---------------------------------------------------------------- right-hand panel

def text_in(lines, region, horiz=True):
    return [l for l in lines if inside(l["bbox"], region) and (not horiz or horizontal(l))]


def by_rows(lines, tol=4):
    rows = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        if rows and abs(rows[-1][0]["bbox"][1] - l["bbox"][1]) < tol:
            rows[-1].append(l)
        else:
            rows.append([l])
    return [sorted(r, key=lambda l: l["bbox"][0]) for r in rows]


def right_of(lines, label, dy=5, max_dx=400):
    lab = next((l for l in lines if l["text"].startswith(label) and horizontal(l)), None)
    if not lab:
        return None, None
    cands = [l for l in lines if horizontal(l) and l is not lab and abs(l["bbox"][1] - lab["bbox"][1]) < dy
             and 0 <= l["bbox"][0] - lab["bbox"][2] < max_dx]
    rest = lab["text"][len(label):].strip()
    first = min(cands, key=lambda l: l["bbox"][0]) if cands else None
    return lab, (rest + (first["text"] if first else "")).strip() or None


PANEL_HEADINGS = ("BRIDGE DETAILS", "LEGENDS:", "LEGEND:", "ABBREVIATIONS", "REFERENCE DRAWINGS")


def tbm_table(lines, top):
    rows = []
    nxt = [l["bbox"][1] for l in lines if l["text"].strip() in PANEL_HEADINGS and PANEL[0] <= l["bbox"][0] and top < l["bbox"][1] < 875]
    region = [PANEL[0], top, PANEL[1], min(nxt, default=875)]
    cells = text_in(lines, region)
    hdr = {c["text"].strip().upper(): c for c in cells if c["text"].strip().upper() in ("CHAINAGE", "EASTING", "NORTHING", "MSL")}
    if "CHAINAGE" not in hdr and {"EASTING", "NORTHING", "MSL"} <= set(hdr):
        return tbm_table_by_header(cells, hdr)
    for idl in cells:
        if not re.match(r"^BMT?\d+$", idl["text"]):
            continue
        yc = (idl["bbox"][1] + idl["bbox"][3]) / 2
        same = sorted((c for c in cells if c is not idl and abs((c["bbox"][1] + c["bbox"][3]) / 2 - yc) < 5
                       and 3270 < c["bbox"][0] < 3635), key=lambda c: c["bbox"][0])
        desc = sorted((c for c in cells if c["bbox"][0] >= 3635 and abs((c["bbox"][1] + c["bbox"][3]) / 2 - yc) < 17),
                      key=lambda c: c["bbox"][1])
        vals = [c["text"] for c in same]
        if len(vals) < 4:
            continue
        rows.append({"tbm_id": idl["text"], "chainage_m": num(vals[0]), "easting": num(vals[1]),
                     "northing": num(vals[2]), "msl_m": num(vals[3]),
                     "description": " ".join(d["text"] for d in desc),
                     "bbox": union([idl["bbox"], *[c["bbox"] for c in same + desc]])})
    return rows


def tbm_table_by_header(cells, hdr):
    """TBM tables without a chainage column (e.g. "TBM 216 | Easting | Northing | MSL | Description")."""
    cols = {k: (c["bbox"][0] + c["bbox"][2]) / 2 for k, c in hdr.items()}
    desc_x = max(c["bbox"][2] for c in hdr.values())
    rows = []
    for idl in cells:
        if not re.match(r"^T?BMT?\s*-?\s*\d+\w*$", idl["text"].strip(), re.I):
            continue
        yc = (idl["bbox"][1] + idl["bbox"][3]) / 2
        row = [c for c in cells if c is not idl and abs((c["bbox"][1] + c["bbox"][3]) / 2 - yc) < 5]
        vals = {}
        for k, x in cols.items():
            near = [c for c in row if num(c["text"]) is not None and abs((c["bbox"][0] + c["bbox"][2]) / 2 - x) < 45]
            vals[k] = num(near[0]["text"]) if near else None
        if sum(v is not None for v in vals.values()) < 3:
            continue
        desc = sorted((c for c in cells if c["bbox"][0] > desc_x and abs((c["bbox"][1] + c["bbox"][3]) / 2 - yc) < 17),
                      key=lambda c: c["bbox"][1])
        rows.append({"tbm_id": idl["text"].strip(), "chainage_m": None, "easting": vals.get("EASTING"),
                     "northing": vals.get("NORTHING"), "msl_m": vals.get("MSL"),
                     "description": " ".join(d["text"] for d in desc),
                     "bbox": union([idl["bbox"], *[c["bbox"] for c in row + desc]])})
    return rows


BRIDGE_TABLE_COLS = ["bridge_id", "chainage_m", "existing_type", "existing_span", "crossing",
                     "proposed_structure", "category", "proposed_span"]


def bridge_table(lines):
    """The BRIDGE DETAILS table some sheets print in the right panel: one row per bridge with its number,
    chainage, existing structure and span, what it crosses, and the proposed structure, type and span.
    Columns are found from where the data sits (the header wording varies and wraps); their order is
    the printed one: BR. NO. | CHAINAGE | EXG. STRUCTURE | EXG. CONFIGURATION | DESCRIPTION |
    PROP. STRUCTURE | PROP. TYPE | PROP. CONFIGURATION."""
    head = next((l for l in lines if l["text"].strip() == "BRIDGE DETAILS" and l["bbox"][0] >= PANEL[0]), None)
    if not head:
        return []
    top = head["bbox"][3]
    bottom = min([l["bbox"][1] for l in lines if l["text"].strip() in PANEL_HEADINGS and l["bbox"][0] >= PANEL[0]
                  and l["bbox"][1] > top + 5], default=top + 600)
    cells = [c for c in text_in(lines, [PANEL[0], top, PANEL[1], bottom])]
    idre = r"^(\d+[A-Z]?(UP|DN)?|LC[- ]?\d+\w*|LHS|RUB[\w -]*|ROB[- ]?\d*\w*)$"
    first_x = min((c["bbox"][0] for c in cells if c["text"].strip().startswith("BR")), default=PANEL[0] + 30)
    anchors = sorted((c for c in cells if re.match(idre, c["text"].strip()) and c["bbox"][0] < first_x + 40),
                     key=lambda c: c["bbox"][1])
    if not anchors:
        return []
    data_top = anchors[0]["bbox"][1] - 12
    data = [c for c in cells if c["bbox"][1] >= data_top]
    # column centres: cluster the x-centres of the cells on the anchor rows
    xs = sorted((c["bbox"][0] + c["bbox"][2]) / 2 for a in anchors for c in data
                if abs((c["bbox"][1] + c["bbox"][3]) / 2 - (a["bbox"][1] + a["bbox"][3]) / 2) < 4)
    cols = []
    for x in xs:
        if cols and x - cols[-1][-1] < 22:
            cols[-1].append(x)
        else:
            cols.append([x])
    centres = [sum(c) / len(c) for c in cols if len(c) >= (1 if len(anchors) <= 3 else max(2, len(anchors) // 3))]
    if len(centres) != len(BRIDGE_TABLE_COLS):
        return []
    rows = []
    for i, a in enumerate(anchors):
        y0 = a["bbox"][1] - 12
        y1 = anchors[i + 1]["bbox"][1] - 12 if i + 1 < len(anchors) else bottom
        parts = {k: [] for k in BRIDGE_TABLE_COLS}
        for c in sorted((c for c in data if y0 <= (c["bbox"][1] + c["bbox"][3]) / 2 < y1), key=lambda c: c["bbox"][1]):
            cx = (c["bbox"][0] + c["bbox"][2]) / 2
            k = BRIDGE_TABLE_COLS[min(range(len(centres)), key=lambda j: abs(centres[j] - cx))]
            parts[k].append(c)
        row = {k: " ".join(c["text"].strip() for c in v) or None for k, v in parts.items()}
        row["chainage_m"] = num(row["chainage_m"])
        row["bbox"] = union([c["bbox"] for v in parts.values() for c in v])
        rows.append(row)
    return rows


def tbm_markers(lines):
    out = []
    for ln in lines:
        m = re.match(r"^TBM - (BMT?\d+)", ln["text"])
        if not m or ln["bbox"][3] > PLAN_BOTTOM:
            continue
        group = stack(ln, lines, max_gap=24, max_total=40, stop=r"C/L OF|TBM")
        text = " ".join(g["text"] for g in group)
        em = re.search(r"ELEV\.?\s*-\s*([\d.]+)", text)
        out.append({"tbm_id": m.group(1), "text": text, "elevation_m": num(em.group(1)) if em else None,
                    "bbox": union(g["bbox"] for g in group)})
    return out


def notes(lines):
    # The notes run from the top of the panel down to the TBM table's heading.
    tbm = [l["bbox"][1] for l in lines if l["text"].strip() == "TBM DETAILS" and l["bbox"][0] >= PANEL[0]]
    region = [PANEL[0], 55, PANEL[1], min(tbm, default=470) - 2]
    body = text_in(lines, region)
    out, cur = [], None
    for row in by_rows(body):
        first = row[0]["text"]
        if re.match(r"^\d+\.$", first):
            cur = {"no": int(first[:-1]), "text": " ".join(l["text"] for l in row[1:])}
            out.append(cur)
        elif cur and first.strip() not in ("NOTE:", "NOTES:"):
            # a wrapped line of the current note (indented under its text, or flush left as on some sheets)
            cur["text"] += " " + " ".join(l["text"] for l in row)
    for n in out:
        n["text"] = re.sub(r"\s+", " ", n["text"]).strip()
    return out


def abbreviations(lines):
    region = [PANEL[0], 1150, PANEL[1], 1365]
    items = text_in(lines, region)
    out, where = {}, {}                      # where: key -> (x0, y0) of its "= meaning" text, for wrapped lines
    used = set()
    for row in by_rows(items, tol=3):
        pending = None
        for l in row:
            t = l["text"]
            if "=" in t and not t.startswith("="):
                k, _, v = t.partition("=")
                out[k.strip()] = v.strip()
                where[k.strip()] = (l["bbox"][0], l["bbox"][1])
                used.add(id(l))
            elif t.startswith("=") and pending:
                out[pending[0]] = t[1:].strip()
                where[pending[0]] = (l["bbox"][0], l["bbox"][1])
                used.update({id(l), id(pending[1])})
                pending = None
            else:
                pending = (t, l)
    # A meaning that wraps continues on the next line in the same column ("= Point of Vertical" / "Insertion").
    for l in sorted(items, key=lambda l: l["bbox"][1]):
        if id(l) in used or "=" in l["text"] or l["text"].strip().upper() == "ABBREVIATIONS":
            continue
        cands = [(k, xy) for k, xy in where.items() if abs(l["bbox"][0] - xy[0]) < 25 and 6 < l["bbox"][1] - xy[1] < 20]
        if cands:
            k, xy = min(cands, key=lambda c: l["bbox"][1] - c[1][1])
            out[k] = f"{out[k]} {l['text'].strip()}"
            where[k] = (xy[0], l["bbox"][1])
            used.add(id(l))
    return out


def legend(lines):
    region = [PANEL[0], 895, PANEL[1], 1125]
    return [l["text"] for l in sorted(text_in(lines, region), key=lambda l: (l["bbox"][0] > 3500, l["bbox"][1]))]


def sheet_info(lines):
    info = {}
    title = [l for l in lines if horizontal(l) and inside(l["bbox"], [3240, 2070, PANEL[1], 2125])]
    t_rows = [" ".join(l["text"] for l in r) for r in by_rows(title)]
    info["title"] = t_rows[0] if t_rows else None
    rng = re.search(r"(\d+\+[\d.]+)\s*TO\s*(\d+\+[\d.]+)", " ".join(t_rows))
    info["chainage_from"], info["chainage_to"] = (rng.group(1), rng.group(2)) if rng else (None, None)
    _, drg = right_of(lines, "DRG.No:", max_dx=60)
    info["drawing_no"] = re.sub(r"\s+", "", drg) if drg else None
    _, info["sheet_no"] = right_of(lines, "SHEET NO :")
    _, info["scale"] = right_of(lines, "SCALE :")
    _, info["date"] = right_of(lines, "DATE :")
    _, info["previous_sheet"] = right_of(lines, "PREVIOUS SHEET:")
    _, info["next_sheet"] = right_of(lines, "NEXT SHEET:")
    for key, pat in (("gauge", r"GAUGE:\s*(.+)"), ("standard_of_construction", r"STD\. OF CONSTRUCTION\s*(.+)"),
                     ("year_of_survey", r"YEAR OF SURVEY:\s*(.+)")):
        l = next((l for l in lines if re.match(pat, l["text"])), None)
        info[key] = re.match(pat, l["text"]).group(1).strip(" '") if l else None
    client = [l for l in lines if horizontal(l) and inside(l["bbox"], [3300, 1885, PANEL[1], 1940])]
    info["client"] = ", ".join(l["text"] for l in sorted(client, key=lambda l: l["bbox"][1])) or None
    project = [l for l in lines if horizontal(l) and inside(l["bbox"], [3250, 1950, PANEL[1], 2068])]
    info["project"] = " ".join(" ".join(l["text"] for l in r) for r in by_rows(project)) or None
    issue = [l for l in lines if horizontal(l) and inside(l["bbox"], [3180, 1810, PANEL[1], 1840])]
    cols = ["no", "date", "rev", "prepared_by", "checked_by", "approved_by"]
    rows = by_rows(issue)
    info["issue_record"] = [dict(zip(cols, [l["text"] for l in r])) for r in rows if len(r) == 6]
    stations = []
    for l in lines:
        if horizontal(l) and re.match(r"^[A-Z ]+ STN\.?$", l["text"]):
            stations.append({"name": l["text"].rstrip("."), "side": "left" if l["bbox"][0] < 1500 else "right"})
    info["stations"] = stations
    return info


# ---------------------------------------------------------------- regions

def regions(lines, info_lines):
    datum = next((l for l in lines if l["text"].startswith("DATUM")), None)
    lsec_left = (datum["bbox"][0] - 45) if datum else 150
    tbm = next((l for l in lines if l["text"] == "TBM DETAILS"), None)
    tbm_top = (tbm["bbox"][1] - 22) if tbm else 490
    gauge = [l for l in lines if re.match(r"GAUGE|STD\. OF|YEAR OF SURVEY", l["text"])]
    # The last sheet of a set has a short L-section: end the region where its text ends.
    band_text = [l["bbox"][2] for l in lines if inside(l["bbox"], [0, BANDS_TOP, DRAWING_RIGHT, BANDS_BOTTOM])]
    lsec_right = min(DRAWING_RIGHT, max(band_text) + 30) if band_text else DRAWING_RIGHT
    r = {
        "alignment_plan": [45, 55, DRAWING_RIGHT, PLAN_BOTTOM],
        "lsection_profile": [lsec_left, PLAN_BOTTOM, lsec_right, BANDS_TOP],
        "lsection_data_bands": [lsec_left, BANDS_TOP, lsec_right, BANDS_BOTTOM],
        "notes": [PANEL[0], 55, PANEL[1], 462],
        "tbm_table": [PANEL[0], tbm_top, PANEL[1], 872],
        "legend": [PANEL[0], 872, PANEL[1], 1127],
        "abbreviations": [PANEL[0], 1127, PANEL[1], 1365],
        "reference_drawings_and_signatures": [PANEL[0], 1365, PANEL[1], 1745],
        "issue_record": [PANEL[0], 1745, PANEL[1], 1868],
        "title_block": [PANEL[0], 1868, PANEL[1], 2335],
    }
    if gauge:
        b = union(l["bbox"] for l in gauge)
        r["sheet_info_box"] = [b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6]
    return r


def chainage_m(km):
    """'1211+000.000' -> 1211000.0"""
    m = re.match(r"^(\d+)\+([\d.]+)$", km or "")
    return int(m.group(1)) * 1000 + float(m.group(2)) if m else None


def annotate_page(page, pdf_name, i):
    """Everything on one sheet, as a dict (the content of data/annotations/sheet_XXX.json)."""
    lines = page_lines(page)
    info = sheet_info(lines)
    tbm_hdr = next((l for l in lines if l["text"] == "TBM DETAILS"), None)
    bridge_list = bridges(lines)
    # Each sheet repeats a strip of its neighbours past the match lines, so a
    # callout can be printed here but belong to the previous or next sheet.
    lo, hi = (chainage_m(info["chainage_from"]), chainage_m(info["chainage_to"]))
    for b in bridge_list:
        ch = b.get("chainage_m")
        b["belongs_to"] = (None if ch is None or lo is None else
                           "this sheet" if lo <= ch <= hi else
                           "previous sheet" if ch < lo else "next sheet")
    ann = {
        "source_pdf": pdf_name,
        "page_index": i,
        "page_size_pt": [round(page.rect.width, 1), round(page.rect.height, 1)],
        "drawing_type": "Detailed Plan and L-Section (alignment plan + longitudinal section)",
        "sheet_info": info,
        "regions": regions(lines, info),
        "bridges": bridge_list,
        "curves": curves(lines),
        "bands": bands(lines),
        "tbm_benchmarks": tbm_table(lines, tbm_hdr["bbox"][3] if tbm_hdr else 500),
        "tbm_markers": tbm_markers(lines),
        "bridge_table": bridge_table(lines),
        "notes": notes(lines),
        "abbreviations": abbreviations(lines),
        "legend": legend(lines),
    }
    bd = ann["bands"]
    if (not info.get("chainage_from") or not info.get("chainage_to")) and isinstance(bd, dict) and bd.get("columns"):
        chs = sorted(c["chainage"] for c in bd["columns"])
        fmt = lambda m: f"{int(m // 1000)}+{m % 1000:07.3f}"      # noqa: E731
        info["chainage_from"], info["chainage_to"] = fmt(chs[0]), fmt(chs[-1])
        info["chainage_source"] = "data bands (the title block does not give the range" + \
            (", it prints '####')" if any(l["text"].strip() == "####" for l in lines) else ")")
        lo, hi = chs[0], chs[-1]
        for b in bridge_list:
            ch = b.get("chainage_m")
            b["belongs_to"] = (None if ch is None else "this sheet" if lo <= ch <= hi else
                               "previous sheet" if ch < lo else "next sheet")
    ann.update(extras(lines, ann["bands"], ann["regions"]))
    # Every text item on the sheet, as printed: the base for "read anything" questions and the coverage check.
    ann["all_text"] = [{"text": l["text"], "bbox": l["bbox"], "dir": [round(d, 3) for d in l["dir"]]} for l in lines]
    return ann


def annotate_pdf(pdf_path):
    """Annotations for every page of a PDF, without writing anything."""
    doc = pymupdf.open(pdf_path)
    return [annotate_page(page, Path(pdf_path).name, i) for i, page in enumerate(doc)]


def annotate(pdf_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, ann in enumerate(annotate_pdf(pdf_path)):
        info = ann["sheet_info"]
        name = f"sheet_{info['sheet_no'] or i:0>3}.json"
        (OUT_DIR / name).write_text(json.dumps(ann, indent=1, ensure_ascii=False), encoding="utf-8")
        summary.append((name, len(ann["bridges"]), len(ann["curves"]), len(ann["tbm_benchmarks"]),
                        len(ann["tbm_markers"]), len(ann["notes"])))
    print(f"{'file':<16}{'bridges':>8}{'curves':>8}{'tbm':>6}{'marks':>7}{'notes':>7}")
    for s in summary:
        print(f"{s[0]:<16}{s[1]:>8}{s[2]:>8}{s[3]:>6}{s[4]:>7}{s[5]:>7}")


if __name__ == "__main__":
    pdfs = sys.argv[1:] or sorted(str(p) for p in ROOT.glob("MKN_PNP_*.pdf"))
    for pdf in pdfs:
        annotate(pdf)
