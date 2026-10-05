"""Build the instruction-tuning dataset (images + JSONL) from data/annotations/.

    python scripts/build_dataset.py

Every answer is generated from the annotations, which were read from the PDF's
text layer, so the values are exactly what is printed on the drawing.

Output (data/dataset/):
    images/            PNG crops; every side is a multiple of 28 (Qwen2.5-VL patch
                       size), so the processor never rescales them and the box
                       coordinates in the answers stay exact.
    train.jsonl, val.jsonl, test.jsonl
    stats.json

Each JSONL row:
    {"id", "sheet", "split", "task", "image", "width", "height",
     "messages": [{"role": "user", "content": [{"type": "image", "image": <path>},
                                               {"type": "text", "text": <question>}]},
                  {"role": "assistant", "content": [{"type": "text", "text": <answer>}]}]}

Boxes use Qwen2.5-VL's grounding format: absolute pixels in the image as given,
[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}].

The split is by sheet, never by tile: overlapping tiles and repeated callouts at
sheet edges would otherwise leak test answers into training.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotations"
OUT = ROOT / "data" / "dataset"
IMG = OUT / "images"

SPLITS = {"96": "val", "100": "test", "104": "test", "110": "test"}      # every other sheet is train
ZOOM = 150 / 72                            # render at 150 dpi: 9.9 pt callout text ~ 20 px tall
TILE = 1008                                # 36 x 28 px
STRIDE = 756                               # 25% overlap
EMPTY_TILE_KEEP = 0.3                      # share of tiles without structures kept as negatives
SHEET_VIEW = (1400, 896)                   # whole-sheet thumbnail for layout questions
SEED = 7

# Per-sheet facts the wording depends on. The defaults are the 3rd-line drawing set (v3); the v4 builder sets
# them for each sheet (4th-line sheets: proposed 4th line, existing DN line, rail-level note 4, ...).
SHEET = {"prop": "3rd line", "exg": "UP line", "rail_note": 5, "ruling": (150, 6), "v4": False}

INTERP_KEYS = ["cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl", "ground_level"]


def bracket(cols, ch):
    """The two printed columns either side of chainage ch (the same column twice if ch is on a column), or None.
    cols: band columns sorted by chainage, as printed (adjacent, one every 20 m)."""
    for a, b in zip(cols, cols[1:]):
        if a["chainage"] == ch:
            return a, a
        if a["chainage"] < ch < b["chainage"]:
            return a, b
    if cols and cols[-1]["chainage"] == ch:
        return cols[-1], cols[-1]
    return None


def interpolate(c1, c2, ch):
    """Rule (Ankur, 2026-10-05): y - y1 = (y2 - y1) / (x2 - x1) * (x - x1) for every band row; always return y."""
    x1, x2 = c1["chainage"], c2["chainage"]
    t = 0.0 if x2 == x1 else (ch - x1) / (x2 - x1)
    return {k: round(c1["values"][k] + (c2["values"][k] - c1["values"][k]) * t, 3) for k in INTERP_KEYS
            if k in c1["values"] and k in c2["values"]}, t


def interp_record(vals):
    return {("exg_line_fl" if k == "exg_up_fl" and SHEET["v4"] else k): v for k, v in vals.items()}

rng = random.Random(SEED)
# Grounding questions (v2) draw from their own stream, so adding them leaves every other row unchanged.
grng = random.Random(SEED + 1)
# Data-band questions (v3) have their own stream too.
brng = random.Random(SEED + 2)
BAND_WINDOW = 16                           # band columns per crop (every 20 m, so about 300 m of chainage)


# ---------------------------------------------------------------- rendering

def pad28(img):
    w, h = img.size
    W, H = -(-w // 28) * 28, -(-h // 28) * 28
    if (W, H) == (w, h):
        return img
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(img, (0, 0))
    return canvas


def save(img, name):
    """Palette PNG: the drawings use few colours, so this keeps files ~4x smaller losslessly enough."""
    path = IMG / (re.sub(r"[^\w.-]+", "_", name) + ".png")      # ids like "ROB at CH 1242581.635" or "578AX(LC-247)"
    try:
        img.convert("P", palette=Image.ADAPTIVE, colors=128).save(path, optimize=True)
    except ValueError:                                           # PIL cannot palettise some near-blank crops
        img.save(path, optimize=True)
    return f"images/{path.name}"


def render(page, clip_pt, size=None):
    """Render a clip at ZOOM. With `size`, the result is exactly that many pixels
    (the renderer rounds the clip outward by a pixel, which would otherwise push a
    1008 px tile to 1036 after padding)."""
    x0, y0, x1, y1 = clip_pt
    r = page.rect
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, r.width), min(y1, r.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=pymupdf.Rect(x0, y0, x1, y1), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if size:
        canvas = Image.new("RGB", size, "white")
        canvas.paste(img.crop((0, 0, min(img.width, size[0]), min(img.height, size[1]))), (0, 0))
        return canvas, (x0, y0)
    return pad28(img), (x0, y0)


def to_px(box, origin, size):
    """PDF-point box -> pixel box inside a crop, clipped. Returns (box, visible_fraction)."""
    ox, oy = origin
    x0, y0, x1, y1 = [(v - o) * ZOOM for v, o in zip(box, (ox, oy, ox, oy))]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, size[0]), min(y1, size[1])
    area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
    vis = max(cx1 - cx0, 0) * max(cy1 - cy0, 0) / area
    return [round(cx0), round(cy0), round(cx1), round(cy1)], vis


def square_around(box, side_pt, jitter=0.0):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if jitter:
        cx += rng.uniform(-jitter, jitter)
        cy += rng.uniform(-jitter, jitter)
    return [cx - side_pt / 2, cy - side_pt / 2, cx + side_pt / 2, cy + side_pt / 2]


# ---------------------------------------------------------------- wording

TYPE = {"RCC SLAB": "RCC slab", "STONE SLAB": "stone slab", "SLAB": "slab", "BOX": "box", "PG": "plate girder (PG)",
        "ARCH": "arch", "LEVEL CROSSING": "level crossing", "ROB": "road over bridge (ROB)"}
CROSSING = {"STREAM": "a stream", "DRAIN": "a drain", "BT ROAD": "a BT (bituminous) road", "ROAD": "a road",
            "MDR": "a major district road (MDR)", "RAIL": "a railway line", "ROB": "a road"}
CATEGORY = {"MINOR": "minor bridge", "MAJOR": "major bridge",
            "RUB/LHS": "road under bridge / limited height subway (RUB/LHS)", "ROR": "rail over rail (ROR)"}
LEVEL_NAMES = {"existing_formation_level": "existing formation level (EXG FL)",
               "min_formation_level_required": "minimum formation level required (MIN FL REQ.)",
               "proposed_formation_level": "proposed formation level (FL)", "bed_level": "bed level (B.L)",
               "high_flood_level": "high flood level (HFL)", "free_board": "free board (FB)"}
CURVE_NAMES = {"degree": "degree of curve", "deflection_angle": "deflection angle (Δ)", "radius": "radius (R)",
               "total_tangent_length": "total tangent length (TTL)", "circular_curve_length": "circular curve length (CCL)",
               "transition_length": "transition length (TRL)", "cant": "cant (Ca)", "shift": "shift",
               "max_speed": "maximum speed (Vmax)"}
REGION_NAMES = {"alignment_plan": "alignment plan", "lsection_profile": "L-section (longitudinal profile)",
                "lsection_data_bands": "L-section data bands (levels, gradients, chainages)", "notes": "general notes",
                "tbm_table": "TBM (temporary bench mark) details table", "legend": "legend",
                "abbreviations": "abbreviations", "reference_drawings_and_signatures": "reference drawings and signatures",
                "issue_record": "issue record", "title_block": "title block", "sheet_info_box": "gauge / survey info box"}


def fenced(obj):
    """JSON in a ```json fence (Qwen2.5-VL's own output habit), one list item per
    line: indenting every value would roughly double the answer tokens."""
    if isinstance(obj, list):
        body = "[\n" + ",\n".join("  " + json.dumps(o, ensure_ascii=False) for o in obj) + "\n]"
    else:
        body = json.dumps(obj, ensure_ascii=False)
    return "```json\n" + body + "\n```"


def span_words(s):
    n, _, length = s.partition(" X ")
    return f"{n} span{'s' if n.strip() != '1' else ''} of {length.strip()} m"


def proposal_words(p):
    if p.startswith("none"):
        return "no extension is needed; the existing structure already has space for the proposed line"
    m = re.match(r"^(\d+) X ([\d.]+) X ([\d.]*)\s*-\s*(.+)$", p)
    if m:
        cells, width, height, kind = m.groups()
        size = f"{width} m wide x {height} m high" if height else f"{width} m wide (height not given on the drawing)"
        return f"to be extended as a {kind} of {cells} cell{'s' if cells != '1' else ''}, {size} (written {p})"
    m = re.match(r"^(.+?)\s*-\s*(.+)$", p)
    if m:
        spans = " + ".join(span_words(s.strip()) for s in m.group(1).split("+"))
        kind = {"OWG": "open web girder (OWG)"}.get(m.group(2), m.group(2))
        return f"to be extended as {spans}, {kind} (written {p})"
    return f"to be extended as {p}"


def min_fl_flag(b):
    """Rule (Ankur): when the proposed FL is below MIN FL REQ., flag it in this exact form."""
    lv = b.get("lsection_levels") or {}
    fl, req = lv.get("proposed_formation_level"), lv.get("min_formation_level_required")
    if fl is None or req is None or fl >= req:
        return None
    return f"FLAG: For bridge {b['bridge_id']} min. FL = {req}, and FL = {fl} ({round(req - fl, 3)} m below the minimum required)."


def bridge_fields(b, with_levels=True):
    out = {k: b.get(k) for k in ("bridge_id", "existing_type", "existing_span", "crossing", "proposal", "category", "chainage_m")}
    if with_levels and b.get("lsection_levels"):
        out["levels"] = {k: v for k, v in b["lsection_levels"].items() if k != "bbox"}
        if min_fl_flag(b):
            out["flag"] = min_fl_flag(b)[len("FLAG: "):]
    return out


def explain(b, with_levels=True):
    bid = b["bridge_id"]
    name = bid if bid.startswith(("ROB", "LC")) else f"Bridge No. {bid}"
    parts = []
    what = TYPE.get(b.get("existing_type") or "", (b.get("existing_type") or "structure").lower())
    s = f"{name} is an existing {what}"
    if b.get("existing_span"):
        s += f" of {span_words(b['existing_span'])}"
    if b.get("crossing"):
        s += f" over {CROSSING.get(b['crossing'], b['crossing'].lower())}"
    if b.get("chainage_m"):
        s += f", with its centre line at chainage {b['chainage_m']} m"
    parts.append(s + ".")
    if b.get("category"):
        parts.append(f"It is classified as a {CATEGORY.get(b['category'], b['category'])}.")
    if b.get("proposal"):
        parts.append(f"For the proposed {SHEET['prop']} it is {proposal_words(b['proposal'])}.")
    lv = b.get("lsection_levels") if with_levels else None
    if lv:
        vals = [f"{LEVEL_NAMES[k]} {v}" for k, v in lv.items() if k in LEVEL_NAMES]
        parts.append("On the L-section: " + ", ".join(vals) + ".")
        if min_fl_flag(b):
            parts.append(min_fl_flag(b))
    return " ".join(parts)


def level_check(b):
    lv = b.get("lsection_levels") or {}
    fl, req = lv.get("proposed_formation_level"), lv.get("min_formation_level_required")
    if fl is None or req is None:
        return None
    diff = round(fl - req, 3)
    ok = diff >= 0
    ans = (f"{'Yes' if ok else 'No'}. The proposed formation level is {fl} and the minimum required is {req}, "
           f"so it is {abs(diff)} m {'above' if ok else 'below'} the requirement.")
    if min_fl_flag(b):
        ans = f"{min_fl_flag(b)} {ans}"
    hfl, fb = lv.get("high_flood_level"), lv.get("free_board")
    if hfl is not None and fb is not None:
        ans += f" The free board shown is {fb} m (FL {fl} minus HFL {hfl} = {round(fl - hfl, 3)} m)."
    return ans


# ---------------------------------------------------------------- question templates

Q = {
    "tile_structures": [
        "Detect every bridge or structure callout in this crop of a railway Plan & L-Section drawing. "
        "Output a JSON list of {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": ...}}.",
        "Find all bridge callouts visible in this drawing crop and give their bounding boxes as JSON.",
        "Locate the bridge / culvert / ROB labels in this image. Return JSON with bbox_2d and label for each.",
    ],
    "tile_region": [
        "Which part of the drawing sheet is this crop taken from?",
        "This is a crop of a railway Plan & L-Section sheet. Which section of the sheet does it show?",
    ],
    "bridge_json": [
        "Read the callout for bridge {bid} in this crop and return its details as JSON.",
        "Extract the existing structure, span, crossing, proposal, category and chainage for bridge {bid} as JSON.",
        "Give the data written for bridge no. {bid} on this drawing as structured JSON.",
    ],
    "bridge_explain": [
        "Explain what this drawing says about bridge {bid}.",
        "Describe bridge {bid} as shown in this crop, in plain words.",
        "What is bridge {bid}, what does it cross, and what is proposed for it?",
    ],
    # v1 wording; kept only so the main random stream is drawn exactly as before (see tiles()).
    "bridge_ground": [
        "Where is the callout for bridge {bid} in this image? Answer with a JSON bbox.",
        "Locate bridge {bid}'s label. Output {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": ...}}.",
    ],
    # v2: always name the view. A bridge has a plan callout, an L-section callout and a level
    # block, and a crop can show more than one of them, so "bridge 562's callout" was ambiguous.
    "bridge_ground_view": [
        "Where is the {view} for bridge {bid} in this image? Answer with a JSON bbox.",
        "Locate the {view} of bridge {bid}. Output {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": ...}}.",
        "Find bridge {bid}'s {view} and give its bounding box as JSON.",
    ],
    "bridge_level_check": [
        "Does the proposed formation level at bridge {bid} meet the minimum formation level required?",
        "At bridge {bid}, is the proposed FL above the MIN FL REQ.? Show the numbers.",
    ],
    "curve_json": [
        "Read the curve data box in this crop and return it as JSON.",
        "Extract the horizontal curve parameters shown here as JSON.",
    ],
    "tbm_json": [
        "Read the TBM details table and return every row as JSON.",
        "Extract the temporary bench marks (ID, chainage, easting, northing, MSL, description) from this table as JSON.",
    ],
    "title_json": [
        "Read the title block and return the sheet metadata as JSON.",
        "Extract drawing number, sheet number, title, chainage range, scale, date, client and issue record from this title block as JSON.",
    ],
    "sheet_layout": [
        "What kind of drawing is this, and where are its main parts? Give the parts as JSON boxes.",
        "Identify the regions of this railway drawing sheet (plan, L-section, notes, tables, title block) with bounding boxes.",
    ],
}

NOTE_QUESTIONS = {
    1: "In what units are the dimensions on this drawing?",
    2: "How is existing work shown on this drawing?",
    3: "How is proposed work shown on this drawing?",
    4: "What gradient is maintained in yards, according to the notes?",
    5: "How high above formation level should the rail level be, and for what track structure?",
    6: "What is the ruling gradient of this section?",
    7: "When is a vertical curve provided, according to the notes?",
    8: "How is land to be acquired shown?",
    9: "From where are the chainages and kilometres reckoned?",
    10: "What is the maximum sectional speed the section is designed for?",
    11: "What axle loading are the proposed formation and bridges designed for?",
}


NOTE_TOPICS = [
    (r"DIMENSIONS", "In what units are the dimensions on this drawing?"),
    (r"EXISTING WORK", "How is existing work shown on this drawing?"),
    (r"PROPOSED WORK", "How is proposed work shown on this drawing?"),
    (r"YARD GRADIENT", "What gradient is maintained in yards, according to the notes?"),
    (r"RAIL LEVEL", "How high above formation level should the rail level be, and for what track structure?"),
    (r"^RULING GRADIENT", "What is the ruling gradient of this section?"),
    (r"VERTICAL CURVE", "When is a vertical curve provided, according to the notes?"),
    (r"LAND TO BE ACQUIRED", "How is land to be acquired shown?"),
    (r"RECKONED|CHAINAGE ARE TAKEN", "From where are the chainages reckoned, according to the notes?"),
    (r"SECTIONAL SPEED", "What is the maximum sectional speed the section is designed for?"),
    (r"AXLE LOADING", "What axle loading are the proposed formation and bridges designed for?"),
    (r"BRIDGE NUMBERS", "How are the bridges numbered on this drawing?"),
    (r"TRACK CENTRE", "What minimum track centre does the drawing require?"),
    (r"TERMS OF REFERENCE:", "What do the terms of reference give for the design speed and the ruling gradient?"),
]


def note_question(n):
    if not SHEET["v4"]:
        return NOTE_QUESTIONS.get(n["no"])
    return next((q for pat, q in NOTE_TOPICS if re.search(pat, n["text"])), None)


def pick(task, **kw):
    return rng.choice(Q[task]).format(**kw)


# ---------------------------------------------------------------- builders

def row(sheet, task, image, size, question, answer):
    return {"sheet": sheet, "task": task, "image": image, "width": size[0], "height": size[1],
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}


def region_of(box_pt, regions):
    best, best_area = None, 0
    for name, r in regions.items():
        w = max(0, min(box_pt[2], r[2]) - max(box_pt[0], r[0]))
        h = max(0, min(box_pt[3], r[3]) - max(box_pt[1], r[1]))
        if w * h > best_area:
            best, best_area = name, w * h
    return best


def callouts(ann):
    for b in ann["bridges"]:
        for view in ("plan_callout", "lsection_callout"):
            if view in b:
                yield b, view, b[view]["bbox"]


def label_for(b, view):
    return f"bridge {b['bridge_id']} ({'plan' if view == 'plan_callout' else 'L-section'} callout)"


def tiles(page, ann, sheet, blocked):
    rows = []
    x_start, y_start = 45, 55
    x_end, y_end = 3165, 2300
    side = TILE / ZOOM
    step = STRIDE / ZOOM
    ys = [y_start + i * step for i in range(int((y_end - y_start - side) // step) + 2)]
    xs = [x_start + i * step for i in range(int((x_end - x_start - side) // step) + 2)]
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            clip = [min(x, x_end - side), min(y, y_end - side)]
            clip += [clip[0] + side, clip[1] + side]
            found, ambiguous = [], False
            for b, view, box in callouts(ann):
                px, vis = to_px(box, clip[:2], (TILE, TILE))
                if vis >= 0.9:
                    found.append({"bbox_2d": px, "label": label_for(b, view)})
                elif vis > 0.1:
                    ambiguous = True          # half a callout in view: no clean answer, skip the task
            region = region_of(clip, ann["regions"])
            keep_empty = not found and rng.random() < EMPTY_TILE_KEEP
            if not (found or keep_empty):
                continue
            img, origin = render(page, clip, (TILE, TILE))
            path = save(img, f"s{sheet}_tile_r{r}c{c}")
            if region:                        # None = blank part of a short last sheet
                rows.append(row(sheet, "tile_region", path, img.size, pick("tile_region"),
                                f"This crop is from the {REGION_NAMES[region]} of the sheet."))
            if not ambiguous:
                ans = fenced(found) if found else "There are no bridge or structure callouts in this crop."
                rows.append(row(sheet, "tile_structures", path, img.size, pick("tile_structures"), ans))
            for f in found:
                # ids may contain spaces ("ROB at CH 1242581.635") or brackets ("578AX(LC-247)")
                bid, view = re.match(r"bridge (.+) \(([^()]+)\)$", f["label"]).groups()
                if bid in blocked:
                    continue
                if rng.random() < 0.5:                 # the v1 draws, kept so the rest of the dataset is unchanged
                    rng.choice(Q["bridge_ground"])
                rows.append(ground_row(sheet, path, img.size, bid, view, f))
    return rows


def ground_row(sheet, path, size, bid, view, box):
    q = grng.choice(Q["bridge_ground_view"]).format(view=view, bid=bid)
    return row(sheet, "bridge_ground", path, size, q, fenced([box]))


def bridge_crops(page, ann, sheet, variants, blocked):
    rows = []
    side = TILE / ZOOM
    for b in ann["bridges"]:
        if not b["complete"] or b["bridge_id"] in blocked:
            continue
        bid = b["bridge_id"]
        views = []
        if "lsection_callout" in b:
            core = b["lsection_callout"]["bbox"]
            if "lsection_levels" in b:
                lv = b["lsection_levels"]["bbox"]
                core = [min(core[0], lv[0]), min(core[1], lv[1]), max(core[2], lv[2]), max(core[3], lv[3])]
            views.append(("lsec", core, "lsection_levels" in b))
        if "plan_callout" in b:
            views.append(("plan", b["plan_callout"]["bbox"], False))
        for vname, core, has_levels in views:
            for v in range(variants):
                clip = square_around(core, side, jitter=40 if v else 0)
                img, origin = render(page, clip, (TILE, TILE))
                path = save(img, f"s{sheet}_br{bid}_{vname}{v}")
                # Only claim the level block if it is actually fully inside this crop.
                lv_in = has_levels and to_px(b["lsection_levels"]["bbox"], origin, img.size)[1] >= 0.98
                call = b["lsection_callout" if vname == "lsec" else "plan_callout"]["bbox"]
                if to_px(call, origin, img.size)[1] < 0.98:
                    continue
                rows.append(row(sheet, "bridge_json", path, img.size, pick("bridge_json", bid=bid),
                                fenced(bridge_fields(b, with_levels=lv_in))))
                rows.append(row(sheet, "bridge_explain", path, img.size, pick("bridge_explain", bid=bid),
                                explain(b, with_levels=lv_in)))
                if v == 0:
                    # Grounding inside the bridge crop: the callout, and the level block when it is in view.
                    view = "plan callout" if vname == "plan" else "L-section callout"
                    targets = [(view, call)] + ([("level block", b["lsection_levels"]["bbox"])] if lv_in else [])
                    for name, box in targets:
                        px = to_px(box, origin, img.size)[0]
                        rows.append(ground_row(sheet, path, img.size, bid, name,
                                               {"bbox_2d": px, "label": f"bridge {bid} ({name})"}))
                chk = level_check(b) if lv_in else None
                if chk and v == 0:
                    rows.append(row(sheet, "bridge_level_check", path, img.size, pick("bridge_level_check", bid=bid), chk))
                if lv_in and v == 0:
                    for k, val in b["lsection_levels"].items():
                        if k in LEVEL_NAMES and rng.random() < 0.5:
                            rows.append(row(sheet, "bridge_level_qa", path, img.size,
                                            f"What is the {LEVEL_NAMES[k]} at bridge {bid}?",
                                            f"The {LEVEL_NAMES[k]} at bridge {bid} is {val}."))
    return rows


def curve_crops(page, ann, sheet):
    rows = []
    side = 672 / ZOOM
    for c in ann["curves"]:
        img, origin = render(page, square_around(c["bbox"], side), (672, 672))
        if to_px(c["bbox"], origin, img.size)[1] < 0.98:
            continue
        where = "plan" if c["location"] == "plan" else "L-section"
        path = save(img, f"s{sheet}_curve{c['curve_no']}_{where[:4]}")
        others = [o for o in ann["curves"] if o is not c and o["location"] == c["location"]
                  and to_px(o["bbox"], origin, img.size)[1] > 0.1]
        body = {"curve_no": c["curve_no"], "line": c["line"], "hand": c["hand"], **c["params"]}
        q = pick("curve_json") if not others else f"Read the data box for curve no. {c['curve_no']} and return it as JSON."
        rows.append(row(sheet, "curve_json", path, img.size, q, fenced(body)))
        k = rng.choice(list(c["params"]))
        rows.append(row(sheet, "curve_qa", path, img.size,
                        f"What is the {CURVE_NAMES[k]} of curve no. {c['curve_no']}?",
                        f"The {CURVE_NAMES[k]} of curve no. {c['curve_no']} is {c['params'][k]}."))
    return rows


def panel_crops(page, ann, sheet):
    rows = []
    R = ann["regions"]
    info = ann["sheet_info"]

    img, _ = render(page, R["title_block"])
    path = save(img, f"s{sheet}_title")
    title = {k: info.get(k) for k in ("drawing_no", "sheet_no", "title", "chainage_from", "chainage_to", "scale", "date", "client")}
    title["issue_record"] = info.get("issue_record")
    rows.append(row(sheet, "title_json", path, img.size, pick("title_json"), fenced(title)))
    rows.append(row(sheet, "title_qa", path, img.size, "Which chainage range does this sheet cover?",
                    f"This sheet covers chainage {info['chainage_from']} to {info['chainage_to']}."))

    img, _ = render(page, R["tbm_table"])
    path = save(img, f"s{sheet}_tbm")
    tbm = [{k: t[k] for k in ("tbm_id", "chainage_m", "easting", "northing", "msl_m", "description")} for t in ann["tbm_benchmarks"]]
    if tbm:
        rows.append(row(sheet, "tbm_json", path, img.size, pick("tbm_json"), fenced(tbm)))
        t = rng.choice(tbm)
        rows.append(row(sheet, "tbm_qa", path, img.size, f"What is the MSL value of {t['tbm_id']}?",
                        f"{t['tbm_id']} is at MSL {t['msl_m']} m" + (f", chainage {t['chainage_m']} m." if t["chainage_m"] is not None else ".")))
    if SHEET["v4"] and ann.get("bridge_table"):
        # BRIDGE DETAILS table (Topo sheets): read it whole, and one row at a time
        bt = ann["bridge_table"]
        box = [min(r["bbox"][0] for r in bt) - 8, min(r["bbox"][1] for r in bt) - 60, max(r["bbox"][2] for r in bt) + 8,
               max(r["bbox"][3] for r in bt) + 6]
        img, _ = render(page, box)
        path = save(img, f"s{sheet}_brtable")
        body = [{k: r[k] for k in r if k != "bbox"} for r in bt]
        rows.append(row(sheet, "bridge_table_json", path, img.size,
                        rng.choice(["Read the bridge details table and return every row as JSON.",
                                    "Extract every bridge from this BRIDGE DETAILS table as JSON."]), fenced(body)))
        r_ = rng.choice(body)
        rows.append(row(sheet, "bridge_table_qa", path, img.size,
                        f"According to the bridge details table, what is proposed for bridge {r_['bridge_id']}?",
                        f"Bridge {r_['bridge_id']} (CH {r_['chainage_m']}): existing {r_['existing_type'] or '-'} {r_['existing_span'] or ''} "
                        f"over {r_['crossing'] or '-'}; proposed {r_['proposed_structure'] or '-'} {r_['proposed_span'] or ''} "
                        f"({r_['category'] or '-'}).".replace("  ", " ")))

    # Notes, legend and abbreviations are the same on every sheet: a few questions per sheet is enough.
    img, _ = render(page, R["notes"])
    path = save(img, f"s{sheet}_notes")
    for n in rng.sample(ann["notes"], 3):
        if note_question(n):
            rows.append(row(sheet, "notes_qa", path, img.size, note_question(n),
                            f"Note {n['no']}: {n['text'].capitalize()}"))
    img, _ = render(page, R["abbreviations"])
    path = save(img, f"s{sheet}_abbr")
    for k in rng.sample(sorted(ann["abbreviations"]), 3):
        rows.append(row(sheet, "abbr_qa", path, img.size, f"What does {k} stand for on this drawing?",
                        f"{k} stands for {ann['abbreviations'][k]}."))
    return rows


def band_names():
    p, e = SHEET["prop"], SHEET["exg"]
    return {"cut_fill": "cut (-) / fill (+), FL - GL (m)",
            "fl_difference": f"difference between the proposed {p} FL and the existing {e} FL",
            "prop_rl": f"proposed {p} rail level (RL)", "prop_fl": f"proposed {p} formation level (FL)",
            "track_distance": f"track distance between the proposed {p} and the existing {e} (m)",
            "exg_up_fl": f"existing {e} formation level (FL)",
            "ground_level": f"ground level below the proposed {p} (m)", "chainage": f"proposed {p} chainage"}


BAND_NAMES = band_names()           # the 3rd-line wording, as in v3


def ch_text(ch):
    return f"{ch:.0f}" if float(ch).is_integer() else f"{ch}"


def band_record(col):
    v = dict(col["values"])
    v["chainage"] = int(v["chainage"]) if float(v["chainage"]).is_integer() else v["chainage"]
    if SHEET["v4"] and "exg_up_fl" in v:
        # v4: the existing line is UP on 3rd-line sheets and DN on 4th-line sheets, so the key names no line
        v = {("exg_line_fl" if k == "exg_up_fl" else k): x for k, x in v.items()}
    return {"chainage": v.pop("chainage"), **v}


def band_crops(page, ann, sheet, blocked_ch):
    """Crops of the L-section data bands: the row-label strip joined to a window of BAND_WINDOW
    columns, so every crop shows which row is which. Questions ask for a column by chainage."""
    bd = ann.get("bands")
    if not bd or "error" in bd:
        return []
    cols = [c for c in bd["columns"] if c["checks_ok"]]
    y0, y1 = bd["row_y"][0][0] - 6, bd["row_y"][-1][1] + 6
    ls = bd["label_strip"]
    strip, _ = render(page, [ls[0] - 4, y0, ls[2] + 6, y1])
    rows = []
    for w in range(0, len(cols), BAND_WINDOW):
        win = cols[w:w + BAND_WINDOW]
        if len(win) < 4:
            continue
        # Columns are 11.3 pt apart and 9.9 pt wide: cut in the 1.4 pt gap so no neighbour shows half its digits.
        x0, x1 = win[0]["bbox"][0] - 0.6, win[-1]["bbox"][2] + 0.6
        band, _ = render(page, [x0, y0, x1, y1])
        img = Image.new("RGB", (strip.width + band.width, max(strip.height, band.height)), "white")
        img.paste(strip, (0, 0))
        img.paste(band, (strip.width, 0))
        img = pad28(img)
        path = save(img, f"s{sheet}_bands{w // BAND_WINDOW:02d}")
        lo, hi = ch_text(win[0]["chainage"]), ch_text(win[-1]["chainage"])
        usable = [c for c in win if c["chainage"] not in blocked_ch]
        if not usable:
            continue
        c = brng.choice(usable)
        rows.append(row(sheet, "band_json", path, img.size,
                        brng.choice(["Read all the data-band values at chainage {ch} and return them as JSON.",
                                     "Give every L-section data-band row for chainage {ch} as JSON.",
                                     "What does the data band say at chainage {ch}? Return JSON."]).format(ch=ch_text(c["chainage"])),
                        fenced(band_record(c))))
        for _ in range(2):
            c = brng.choice(usable)
            names = band_names()
            k = brng.choice([k for k in names if k != "chainage"])
            rows.append(row(sheet, "band_qa", path, img.size,
                            f"What is the {names[k]} at chainage {ch_text(c['chainage'])}?",
                            f"At chainage {ch_text(c['chainage'])} the {names[k]} is {c['values'][k]}."))
        if brng.random() < 0.5:
            rows.append(row(sheet, "band_range", path, img.size, "Which chainages do the data bands in this crop cover?",
                            f"This crop covers chainage {lo} to {hi}: {len(win)} columns, one every 20 m."))
        else:
            others = [c for c in cols if not (win[0]["chainage"] <= c["chainage"] <= win[-1]["chainage"])]
            if others:
                c = brng.choice(others)
                rows.append(row(sheet, "band_absent", path, img.size,
                                f"What is the proposed {SHEET['prop']} FL at chainage {ch_text(c['chainage'])}?",
                                f"Chainage {ch_text(c['chainage'])} is not in this crop. It shows chainage {lo} to {hi}."))
        # The use case: a bridge's chainage (read from its callout) -> the data-band values at that chainage.
        # Rule (Ankur): take the NEAREST printed column, never interpolate between columns.
        for b in ann["bridges"]:
            ch = b.get("chainage_m")
            if not ch or not b["complete"] or not (win[0]["chainage"] <= ch <= win[-1]["chainage"]):
                continue
            name = b["bridge_id"] if b["bridge_id"].startswith(("ROB", "LC")) else f"bridge {b['bridge_id']}"
            if SHEET.get("interpolate"):
                # v4 rule: interpolate between the two printed columns either side, and always return y.
                # Both columns must be in this crop, printed consistently (in `cols`, which are checks_ok) and adjacent.
                br = bracket(win, ch)
                if not br or br[0]["chainage"] in blocked_ch or br[1]["chainage"] in blocked_ch:
                    continue
                c1, c2 = br
                if c2 is not c1 and abs(c2["chainage"] - c1["chainage"] - 20) > 0.5:
                    continue                       # a column between them failed its checks: do not span the gap
                y, t = interpolate(c1, c2, ch)
                rows.append(row(sheet, "band_bridge", path, img.size,
                                f"{name[0].upper() + name[1:]} is at chainage {ch}. Give the data-band values at that chainage as JSON, "
                                f"interpolating between the columns either side.",
                                fenced({"chainage": ch, "x1": band_record(c1)["chainage"], "x2": band_record(c2)["chainage"],
                                        "column_x1": band_record(c1), "column_x2": band_record(c2),
                                        "y": interp_record(y)})))
                continue
            near = min(win, key=lambda c: abs(c["chainage"] - ch))
            if near["chainage"] in blocked_ch:
                continue
            rows.append(row(sheet, "band_bridge", path, img.size,
                            f"{name[0].upper() + name[1:]} is at chainage {ch}. Give the data-band values at the nearest column as JSON.",
                            fenced({"bridge_chainage": ch, "nearest_column": band_record(near),
                                    "distance_m": round(abs(near["chainage"] - ch), 3)})))
    return rows


def sheet_view(page, ann, sheet):
    w, h = page.rect.width, page.rect.height
    s = min(SHEET_VIEW[0] / w, SHEET_VIEW[1] / h)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    canvas = Image.new("RGB", SHEET_VIEW, "white")
    canvas.paste(img, (0, 0))
    path = save(canvas, f"s{sheet}_sheet")
    parts = [{"bbox_2d": [round(v * s) for v in box], "label": REGION_NAMES[name]} for name, box in ann["regions"].items()]
    ans = (f"This is a railway Detailed Plan and L-Section sheet: the alignment plan of the proposed {SHEET['prop']} is on top, "
           "the longitudinal section (profile and data bands) below it, and notes, TBM table, legend, abbreviations "
           "and the title block in the right-hand panel.\n" + fenced(parts))
    return [row(sheet, "sheet_layout", path, SHEET_VIEW, pick("sheet_layout"), ans)]


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    for old in IMG.glob("*.png"):
        old.unlink()
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(ANN.glob("sheet_*.json"))]
    split_of = {a["sheet_info"]["sheet_no"]: SPLITS.get(a["sheet_info"]["sheet_no"], "train") for a in anns}
    # A bridge that appears on a val/test sheet (edge callouts repeat on the next sheet) is held out of train.
    held = {b["bridge_id"] for a in anns if split_of[a["sheet_info"]["sheet_no"]] != "train" for b in a["bridges"]}
    # Band columns at a sheet boundary appear on both sheets: keep a val/test sheet's columns out of train.
    held_ch = {c["chainage"] for a in anns if split_of[a["sheet_info"]["sheet_no"]] != "train"
               for c in (a.get("bands") or {}).get("columns", [])}

    out = {"train": [], "val": [], "test": []}
    pdfs = {}
    for a in anns:
        sheet = a["sheet_info"]["sheet_no"]
        split = split_of[sheet]
        pdf = pdfs.setdefault(a["source_pdf"], pymupdf.open(ROOT / a["source_pdf"]))
        page = pdf[a["page_index"]]
        blocked = held if split == "train" else set()
        rows = (tiles(page, a, sheet, blocked) + bridge_crops(page, a, sheet, 2 if split == "train" else 1, blocked)
                + curve_crops(page, a, sheet) + panel_crops(page, a, sheet) + sheet_view(page, a, sheet)
                + band_crops(page, a, sheet, held_ch if split == "train" else set()))
        for i, r in enumerate(rows):
            r["id"] = f"s{sheet}_{i:04d}"
            r["split"] = split
        out[split] += rows
        print(f"sheet {sheet:>3} ({split:<5}) {len(rows):>4} rows")

    stats = {}
    for split, rows in out.items():
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "sheet", "split", "task", "image", "width", "height", "messages")},
                                   ensure_ascii=False) + "\n")
        stats[split] = {"rows": len(rows), "images": len({r["image"] for r in rows}),
                        "tasks": dict(sorted(Counter(r["task"] for r in rows).items()))}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in IMG.glob("*.png")) / 1e6
    print(json.dumps(stats, indent=1))
    print(f"{len(list(IMG.glob('*.png')))} images, {size:.0f} MB")


if __name__ == "__main__":
    main()
