"""Build the GAD (RCC box) fine-tuning dataset from data/gad/annotations.

    .venv\\Scripts\\python gad_tools\\build_dataset_gad.py

Splits (by whole GAD): one GAD is held back for your own test (TEST_GAD: never in train or validation; its questions
go to test.jsonl, only to score the trained model), a few GADs are validation, the rest train.

Tasks
  Question answering with the GAD facts as context (the way the CLI works on an uploaded vector PDF):
    qa_level      levels: value, label as printed, view, meaning; several values for the same level are all given
    qa_value      "what is 342.494?" - every place a value appears and what it denotes there
    qa_label      "what does PROP SOFFIT LVL mean?"
    qa_dim        labelled dimensions (track centres, barrel length, thicknesses, weep holes, gaps, slopes)
    qa_plain_dim  unlabelled dimension figures: view and direction, and that the drawing does not say what they measure
    qa_table      comparative table, track details, depth of track structure
    qa_note       notes, special/fill notes, specifications, design criteria, reference drawings
    qa_abbr       abbreviations (the drawing's own list first)
    qa_title      drawing numbers, revision, date, location, bridge, box size
    qa_bore       bore log: SBC by depth, soil layers
    qa_view       which views there are and what each represents
    qa_finding    values that disagree between places on the drawing
    qa_computed   simple arithmetic from printed levels (cushion over the box, clearance), shown as computed
    qa_summary    a summary of the GAD
    qa_absent     things that are not on the drawing -> said so, nothing invented
  Images (so the model knows what each diagram looks like and can read it):
    img_view      a whole view -> which view it is and what it represents (no values: too small to read at this size)
    img_view_facts  a whole view + its facts -> what it shows, with the values and their meanings
    img_tile      a 768 x 768 tile of a view at 150 dpi -> the labelled levels and dimensions fully visible in it, as JSON
    img_table     comparative table / track details / specifications / title block crop at 150 dpi -> JSON
Output: data/gad/dataset/{train,val,test}.jsonl and images/. Thinking is off for every row (direct answers).
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import facts as FX           # noqa: E402
import gad_kinds as K        # noqa: E402

ANN = ROOT / "data" / "gad" / "annotations"
PDFS = ROOT / "GAD"
OUT = ROOT / "data" / "gad" / "dataset"
IMG = OUT / "images"
TEST_GAD = "810-1"                       # held back completely: you upload this one yourself to test
VAL_GADS = {"752-2", "787-1", "818-2", "RUB-801-1"}
DPI = 150
TILE = 768
rng = random.Random(29)

KIND_NAME = {"rail_level": "rail level", "formation_level": "formation level", "top_of_box_level": "top of box level",
             "soffit_level": "soffit level", "hfl": "HFL (high flood level)", "observed_hfl": "observed HFL",
             "calculated_hfl": "calculated HFL", "bed_level": "bed level", "founding_level": "founding level",
             "wall_top_level": "top of wall level", "invert_level": "invert level", "road_level": "road level",
             "scour_level": "scour level", "ground_level": "ground level", "lwl": "low water level", "reduced_level": "level"}
SHORT = {"hfl": "HFL", "observed_hfl": "observed HFL", "calculated_hfl": "calculated HFL"}


def lv(v):
    return f"{v:.3f}"


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def vary(q):
    """Light variation of a question's wording (values untouched)."""
    r = rng.random()
    if r < 0.15:
        q = q[0].lower() + q[1:]
    if rng.random() < 0.15:
        q = rng.choice(["Quick one: ", "Hey, ", "Can you check: ", "Tell me: "]) + q[0].lower() + q[1:]
    if rng.random() < 0.1:
        q = q.rstrip("?")
    return q


def elem_phrase(el):
    if not el:
        return ""
    e = el.lower().replace("r/wall", "return wall").replace("r/ wall", "return wall").replace("rcc ", "")
    return f" of the {e}"


def views_phrase(views):
    vs = sorted(set(v for v in views if v))
    return ("on " + " and ".join(vs)) if vs else "outside the drawn views"


# ======================================================================== question / answer generators
def qa_levels(a):
    out = []
    groups = defaultdict(list)
    for l in a["levels"]:
        if l["kind"] == "reduced_level" and not re.search(r"\bR\.?L\b", l["label"]):
            continue
        groups[(l["kind"], l["status"], l["element"])].append(l)
    for (kind, status, el), ls in groups.items():
        if kind == "reduced_level":
            continue
        name = f"{status + ' ' if status else ''}{KIND_NAME[kind]}{elem_phrase(el)}"
        byval = defaultdict(list)
        for l in ls:
            byval[l["value"]].append(l)
        refs = [("level", l["label"], l["value"]) for l in ls]
        meaning = ls[0]["meaning"]
        if len(byval) == 1:
            v, items = next(iter(byval.items()))
            labels = sorted({l["label"] for l in items})
            ans = (f"The {name} is {lv(v)} m (written as {', '.join(repr(x) for x in labels)} {views_phrase([l['view'] for l in items])}). "
                   f"The {KIND_NAME[kind].split(' (')[0]} is {meaning}.")
        else:
            parts = [f"{lv(v)} m ({', '.join(sorted({repr(l['label']) for l in items}))} {views_phrase([l['view'] for l in items])})"
                     for v, items in sorted(byval.items(), key=lambda x: -x[0])]
            spread = max(byval) - min(byval)
            ans = (f"The drawing gives {len(byval)} different values for the {name}: " + "; ".join(parts) +
                   f". They differ by up to {spread:.3f} m, so check which one applies. The {KIND_NAME[kind].split(' (')[0]} is {meaning}.")
        short = SHORT.get(kind, KIND_NAME[kind])
        qs = [f"What is the {status + ' ' if status else ''}{short}{elem_phrase(el)}?",
              f"Give me the {status + ' ' if status else ''}{short}{elem_phrase(el)} on this GAD.",
              f"What {status + ' ' if status else ''}{short}{elem_phrase(el)} does the drawing show?"]
        for q in rng.sample(qs, 2):
            out.append(("qa_level", vary(q), ans, refs))
    # "the rail level" without saying existing or proposed: both are given
    by_kind = defaultdict(dict)
    for (kind, status, el), ls in groups.items():
        if status and not el:
            by_kind[kind][status] = ls
    for kind, st in by_kind.items():
        if len(st) == 2 and kind in KIND_NAME and kind != "reduced_level":
            parts, refs = [], []
            for s in ("existing", "proposed"):
                vals = sorted({l["value"] for l in st[s]})
                parts.append(f"the {s} {KIND_NAME[kind]} is {' / '.join(lv(v) + ' m' for v in vals)}")
                refs += [("level", l["label"], l["value"]) for l in st[s]]
            ans = cap("; ".join(parts)) + f" (the existing value is for the existing bridge/track, the proposed one for the new box). " \
                  f"The {KIND_NAME[kind].split(' (')[0]} is {st['proposed'][0]['meaning']}."
            out.append(("qa_level", vary(f"What is the {SHORT.get(kind, KIND_NAME[kind])}?"), ans, refs))
    # labels
    seen = set()
    for l in a["levels"]:
        if l["label"] in seen:
            continue
        seen.add(l["label"])
        same = [x for x in a["levels"] if x["label"] == l["label"]]
        vals = sorted({x["value"] for x in same})
        ans = (f"'{l['label']}' is {('the ' + l['status'] + ' ') if l['status'] else 'the '}{KIND_NAME[l['kind']]}{elem_phrase(l['element'])}: "
               f"{l['meaning']}. On this drawing it is {' / '.join(lv(v) + ' m' for v in vals)} ({views_phrase([x['view'] for x in same])}).")
        if l["kind"] == "reduced_level" and not re.search(r"\bR\.?L\b", l["label"]):
            ans = (f"'{l['label']}' is a level written without saying which part it belongs to: {' / '.join(lv(v) + ' m' for v in vals)} "
                   f"({views_phrase([x['view'] for x in same])}). The drawing does not name the element, so I can't tell which surface it is.")
        q = rng.choice([f"What does '{l['label']}' mean on this drawing?", f"What is {l['label']}?", f"Explain the label {l['label']}."])
        out.append(("qa_label", vary(q), ans, [("level", x["label"], x["value"]) for x in same]))
    return out


def qa_values(a, facts):
    """'What is 342.494?' -> every place the value appears."""
    out = []
    vals = Counter(l["value"] for l in a["levels"])
    for v in rng.sample(sorted(vals), min(8, len(vals))):
        where = [l for l in a["levels"] if l["value"] == v]
        rows = [f for f in facts if f["group"] == "table" and re.search(rf"(?<![\d.]){re.escape(lv(v))}(?![\d])", f["text"])]
        parts = []
        for label in sorted({l["label"] for l in where}):
            ls = [l for l in where if l["label"] == label]
            parts.append(f"'{label}' {views_phrase([l['view'] for l in ls])} - the {('' if not ls[0]['status'] else ls[0]['status'] + ' ')}"
                         f"{KIND_NAME[ls[0]['kind']]}{elem_phrase(ls[0]['element'])} ({ls[0]['meaning']})")
        for r in rows:
            parts.append(r["text"])
        ans = f"{lv(v)} m appears as: " + "; ".join(parts) + "."
        refs = [("level", l["label"], l["value"]) for l in where] + [r["ref"] for r in rows]
        q = rng.choice([f"What is {lv(v)} on this drawing?", f"What does the value {lv(v)} denote?", f"Where does {lv(v)} appear and what is it?"])
        out.append(("qa_value", vary(q), ans, refs))
    return out


def qa_dims(a):
    out = []
    for d in a["labelled_dims"]:
        ref = [("dim", d["label"], d["value"])]
        v = f"{d['value']:g}"
        if d["kind"] == "track_centres":
            btw = (" between the " + " and the ".join(x.lower() for x in d["between"])) if d["between"] else ""
            ans = f"The track-centre distance{btw} is {v} mm ('{d['label']}' on {d['view']}). T/C means track centres: {d['meaning']}."
            qs = [f"What is the track centre distance{btw}?", "What is the T/C on this drawing?", f"What does {v} mean on the drawing?"]
        elif d["kind"] == "barrel_length":
            ans = f"The barrel length is {v} mm ('{d['label']}' on {d['view']}): {d['meaning']}."
            qs = ["What is the barrel length?", "How long is the box barrel?", f"What is {v}?"]
        elif d["kind"] == "thickness":
            ans = f"{cap(d['meaning'])} is {v} mm ('{d['label']}' on {d['view']})."
            what = (d["what"] or "item").lower()
            qs = [f"How thick is the {what}?", f"What is the thickness of the {what}?"]
        elif d["kind"] == "diameter":
            ans = f"{cap(d['meaning'])} is {v} mm ('{d['label']}' on {d['view']})."
            qs = [f"What is the diameter of the {(d['what'] or 'pipe').lower()}?", f"What does '{d['label']}' mean?"]
        else:
            ans = f"'{d['label']}' on {d['view']}: {d['meaning']} ({v} mm)."
            qs = [f"What does '{d['label']}' mean?"]
        out.append(("qa_dim", vary(rng.choice(qs)), ans, ref))
    for s in a["slopes"][:4]:
        out.append(("qa_dim", vary(rng.choice([f"What does '{s['label']}' mean on {s['view']}?", f"What is the slope '{s['label']}'?"])),
                    f"'{s['label']}' on {s['view']} is {s['meaning']}.", [("slope", s["label"])]))
    if not any(d["kind"] == "barrel_length" for d in a["labelled_dims"]):
        out.append(("qa_absent", vary("What is the barrel length?"),
                    "The drawing does not write a barrel length label, so I can't give it from the text of this GAD.", []))
    return out


def qa_plain(a):
    out = []
    by_view = defaultdict(list)
    for d in a["dims"]:
        by_view[d["view"]].append(d)
    for view, ds in by_view.items():
        for d in rng.sample(ds, min(2, len(ds))):
            v = f"{d['value']:g}"
            same = [x for x in ds if x["value"] == d["value"]]
            ans = (f"{v} is a dimension in millimetres written {d['direction']}ly on {view}"
                   + (f" ({len(same)} times)" if len(same) > 1 else "") +
                   ". The drawing does not write what it measures (it is a plain dimension figure), so I can only say which view it is "
                   "on and its direction; the drawn dimension line shows the exact points it is measured between.")
            out.append(("qa_plain_dim", vary(rng.choice([f"What does {v} on the {view.lower()} represent?", f"What is the {v} dimension on {view}?"])),
                        ans, [("plain", view, d["direction"])]))
    return out


GRADE_RE = re.compile(r"^(?:(R|RISE|F|FALL)\s*(?:1\s*IN\s*)?(\d+(?:\.\d+)?)|LEVEL)$", re.I)


def grade_words(g):
    m = GRADE_RE.match((g or "").strip())
    if not m:
        return g
    if not m.group(1):
        return "level (no gradient)"
    return f"a {'rise' if m.group(1).upper().startswith('R') else 'fall'} of 1 in {m.group(2)}"


def qa_tables(a):
    out = []
    for name, t in a["tables"].items():
        if not isinstance(t, dict):
            continue
        tname = name.replace("_", " ")
        for r in t["rows"]:
            ref = [("table", name, r["description"])]
            ex, pr = r.get("existing"), r.get("proposed")
            if r.get("kind") == "grade":
                ex, pr = (f"{ex} ({grade_words(ex)})" if ex else ex), (f"{pr} ({grade_words(pr)})" if pr else pr)
            desc = r["description"].lower()
            mean = f" - {r['meaning']}" if r.get("meaning") and desc.replace("type of ", "") not in r["meaning"] else ""
            if ex and pr and ex == pr:
                ans = f"In the {tname}, the {desc} is {pr} for both the existing and the proposed bridge{mean}."
            elif ex and pr:
                ans = f"In the {tname}, the {desc} is {pr} for the proposed bridge and {ex} for the existing bridge{mean}."
            elif pr or ex or r.get("value"):
                ans = f"In the {tname}, the {desc} is {pr or ex or r.get('value')}{mean}."
            else:
                ans = f"The {tname} lists '{r['description']}' but gives no value for it."
            qs = [f"What is the {desc}?", f"What is the {desc} of the proposed bridge?", f"Compare the existing and proposed {desc}."]
            out.append(("qa_table", vary(rng.choice(qs)), ans, ref))
    dts = a["tables"].get("depth_of_track_structure")
    if isinstance(dts, list) and dts:
        txt = "; ".join(f"{x['item'].lower()} {x['value']}" for x in dts)
        out.append(("qa_table", vary(rng.choice(["What is the depth of track structure?", "What makes up the depth of the track structure?"])),
                    f"The depth of track structure is made up of: {txt}.", [("table", "depth")]))
        for x in dts:
            if re.search(r"BALLAST|TOTAL", x["item"], re.I):
                out.append(("qa_table", vary(f"What is the {x['item'].lower()} in the track structure?"),
                            f"The {x['item'].lower()} is {x['value']} (depth of track structure table).", [("table", "depth")]))
    return out


NOTE_Q = [
    (r"CLEAR COVER.*?(\d+\s*MM)", ["What is the clear cover to reinforcement?", "How much clear cover is specified?"], "the clear cover is {0}"),
    (r"(FE[- ]?\d+D?)", ["What grade of reinforcement steel is used?", "Which steel bars are specified?"], "the reinforcement is {0} bars"),
    (r"(\d+\s*T)\s*AXLE LOAD", ["What axle load is the bridge designed for?"], "it is designed for a {0} axle load"),
    (r"FOUNDING PRESSURE OF (?:THE )?RCC BOX\s*=?\s*([\d.]+\s*T/M\S*)", ["What is the founding pressure of the box?"], "the founding pressure of the RCC box is {0}"),
    (r"ALL DIMENSIONS ARE IN (\w+)", ["What units are the dimensions in?", "Are the dimensions in mm or m?"], "dimensions are in {0} (levels in metres)"),
    (r"SEISMIC ZONE[- ]?(\w+)", ["Which seismic zone is the bridge in?"], "the bridge is in seismic zone {0}"),
    (r"BED SLOPE.*?(1\s*IN\s*\d+)", ["What bed slope is specified?"], "the bed slope is at least {0}"),
    (r"YEAR OF\s+CONSTRUCTION IS (\d{4})", ["When was the existing bridge built?"], "the existing bridge was built in {0}"),
    (r"(BGML|MBG|RBG|HM)\s+LOADING", ["What is the loading standard of the existing bridge?"], "the existing bridge's loading standard is {0} loading"),
    (r"DRAWING SHALL NOT BE SCALED", ["Can I scale dimensions off this drawing?"], "no - the drawing must not be scaled; only written dimensions are to be followed"),
    (r"WEEP HOLES OF ([\d/]+\s*MM DIA)", ["What size are the weep holes?"], "the weep holes are {0} pipes"),
    (r"REINFORCEMENT DETAILS.*SEPARATE DRAWING", ["Where are the reinforcement details?"], "they are in a separate drawing"),
]


def qa_notes(a):
    out = []
    n = a["notes"]
    for key, label in (("notes", "Note"), ("special_note", "Special note"), ("fill_note", "Fill note"), ("add_note", "Additional note")):
        items = n.get(key, [])
        for it in rng.sample(items, min(6 if key == "notes" else 2, len(items))):
            out.append(("qa_note", vary(rng.choice([f"What does {label.lower()} {it['no']} say?", f"Read {label.lower()} {it['no']}."])),
                        f"{label} {it['no']}: {it['text']}", [(key, it["no"], it["text"][:30])]))
        for it in items:
            for pat, qs, tmpl in NOTE_Q:
                m = re.search(pat, it["text"], re.I)
                if m:
                    ans = cap(tmpl.format(*[g.strip() for g in m.groups()])) + f" ({label.lower()} {it['no']}: \"{it['text']}\")."
                    out.append(("qa_note", vary(rng.choice(qs)), ans, [(key, it["no"], it["text"][:30])]))
    for it in n.get("specifications", []):
        if it.get("value"):
            item = it["item"].lower()
            out.append(("qa_note", vary(rng.choice([f"What is the {item}?", f"What does the drawing specify for the {item.replace('grade of ', '')}?"])),
                        f"The {item} is {it['value']} (specification {it['no']}).", [("spec", it["item"])]))
    if n.get("design_criteria"):
        lst = "; ".join(f"{it['no']}. {it['text']}" for it in n["design_criteria"])
        out.append(("qa_note", vary(rng.choice(["Which codes is the design based on?", "What are the design criteria?"])),
                    f"The design criteria listed are: {lst}", [("design_criteria", it["no"], it["text"][:30]) for it in n["design_criteria"]]))
    if n.get("reference_drawings"):
        lst = "; ".join(f"{it['no']}. {it['text']}" for it in n["reference_drawings"])
        out.append(("qa_note", vary("Which reference drawings are given?"), f"The reference drawings are: {lst}",
                    [("reference_drawings", it["no"], it["text"][:30]) for it in n["reference_drawings"]]))
    if n.get("seismic_zone"):
        out.append(("qa_note", vary("What is the seismic zone?"), f"Seismic zone {n['seismic_zone']} (as written beside the title block).", [("seismic",)]))
    if n.get("standard_of_loading"):
        out.append(("qa_note", vary("What is the standard of loading?"), f"The standard of loading is {n['standard_of_loading']}.", [("loading",)]))
    return out


def qa_abbr(a):
    out = []
    ab = a["notes"].get("abbreviations") or {}
    for k in rng.sample(sorted(ab), min(5, len(ab))):
        out.append(("qa_abbr", vary(rng.choice([f"What does {k} stand for?", f"What is meant by {k} on this drawing?"])),
                    f"{k} stands for {ab[k].lower()} (from the drawing's abbreviation list).", [("abbr", k)]))
    for k, v in K.ABBREVIATION_HINTS.items():
        if k not in ab and rng.random() < 0.3:
            out.append(("qa_abbr", vary(f"What does {k} mean?"),
                        f"{k} is not in this drawing's abbreviation list; on GADs it generally means {v}.", []))
    return out


def qa_title(a):
    out = []
    tb, b = a["title_block"], a["bridge"]
    simple = [("dwg_no", ["What is the drawing number?", "What is the DWG no.?"], "The drawing number is {v}."),
              ("hq_dwg_no", ["What is the HQ's drawing number?"], "The HQ's drawing number is {v}."),
              ("consultants_dwg_no", ["What is the consultant's drawing number?"], "The consultant's drawing number is {v}."),
              ("rev_no", ["What is the revision of this drawing?", "Which revision is this?"], "The revision is {v}."),
              ("date", ["What is the date on the drawing?"], "The drawing is dated {v}."),
              ("division", ["Which division is this bridge in?"], "It is in the {v} division."),
              ("section", ["Which section is this on?"], "It is on the {v} section."),
              ("km_chainage", ["What is the KM/chainage of the bridge?"], "The KM/chainage in the title block is {v}."),
              ("project", ["What is the project?"], "The project is {v}."),
              ("size", ["What is the sheet size?"], "The sheet size is {v}."),
              ("name_of_work", ["What is the name of work?"], "The name of work is: {v}.")]
    for key, qs, tmpl in simple:
        if tb.get(key):
            ref = "work" if key == "name_of_work" else "project" if key in ("division", "section", "km_chainage", "project") else "drawing"
            out.append(("qa_title", vary(rng.choice(qs)), tmpl.format(v=tb[key]), [ref]))
    if b.get("bridge_no"):
        out.append(("qa_title", vary(rng.choice(["Which bridge is this GAD for?", "What is the bridge number?"])),
                    f"It is {b.get('category', 'bridge').lower()} no. {b['bridge_no']} ({b.get('description', '')})"
                    + (f" at CH {b['chainage']}" if b.get("chainage") else "") + (f", {b['relation_to_existing'].lower()}" if b.get("relation_to_existing") else "") + ".",
                    ["bridge", "title"]))
    if b.get("box"):
        bx = b["box"]
        out.append(("qa_title", vary(rng.choice(["What is the size of the box?", "What is the box size?", "How big is the opening?"])),
                    f"The title gives the box as {bx['as_printed']} m: {bx['cells']} cell(s) of clear width {bx['clear_width_m']:g} m and clear "
                    f"height {bx['clear_height_m']:g} m." + (" Note: the comparative table gives a different size - see the disagreements." if any("box size" in f for f in a["findings"]) else ""),
                    ["bridge", "title"] + [("table", "comparative_table", "SPAN")]))
    return out


def qa_bore(a):
    out = []
    for bl in a["bore_logs"]:
        ref = [("bore", bl["view"])]
        if bl["sbc"]:
            s = rng.choice(bl["sbc"])
            if s.get("depth_m") is not None:
                out.append(("qa_bore", vary(f"What is the SBC at {s['depth_m']:g} m depth?"),
                            f"The bore log ({bl['view']}) gives an SBC of {s['sbc_t_per_m2']:g} t/m² at {s['depth_m']:g} m depth. "
                            "SBC is the safe bearing capacity of the soil.", ref))
            out.append(("qa_bore", vary("What safe bearing capacities does the bore log give?"),
                        "From the bore log: " + "; ".join(f"{x['sbc_t_per_m2']:g} t/m² at {x['depth_m']:g} m" if x.get("depth_m") is not None else f"{x['sbc_t_per_m2']:g} t/m²" for x in bl["sbc"]) + ".", ref))
        if bl["layers"]:
            out.append(("qa_bore", vary(rng.choice(["What soil layers does the bore log show?", "What is the soil at the bridge site?"])),
                        "The bore log shows (top to bottom): " + "; ".join(x["soil"].lower() for x in bl["layers"]) + ".", ref))
        if bl.get("chainage"):
            out.append(("qa_bore", vary("At what chainage was the bore log taken?"), f"The bore log is at chainage {bl['chainage']} m ({bl['view']}).", ref))
    return out


def view_levels_text(a, title):
    ls = [l for l in a["levels"] if l["view"] == title]
    seen, parts = set(), []
    for l in ls:
        if (l["label"], l["value"]) in seen:
            continue
        seen.add((l["label"], l["value"]))
        parts.append(f"{l['label']} {lv(l['value'])} m")
    ds = [f"{d['label']}" for d in a["labelled_dims"] if d["view"] == title]
    return parts, ds


def qa_views(a):
    out = []
    vs = a["views"]
    out.append(("qa_view", vary(rng.choice(["Which views are on this GAD?", "What diagrams does this drawing contain?", "List the views on the drawing."])),
                "The drawing has these views: " + "; ".join(f"{v['title']} (scale {v['scale']})" for v in vs) + ".", ["views"]))
    for v in vs:
        parts, ds = view_levels_text(a, v["title"])
        ans = f"The {v['title']} (scale {v['scale']}) is {v['represents']}"
        if parts:
            ans += " On this drawing it shows the levels " + ", ".join(parts[:10]) + ("" if len(parts) <= 10 else ", and more") + "."
        if ds:
            ans += " Labelled dimensions on it: " + "; ".join(ds[:6]) + "."
        q = rng.choice([f"What does the {v['title'].lower()} represent?", f"What is shown in {v['title']}?", f"Explain the {v['title'].lower()}."])
        out.append(("qa_view", vary(q), ans, [("view", v["title"])] + [("level", l["label"], l["value"]) for l in a["levels"] if l["view"] == v["title"]][:10]
                    + [("dim", d["label"], d["value"]) for d in a["labelled_dims"] if d["view"] == v["title"]][:6]))
    return out


def qa_findings(a):
    q = vary(rng.choice(["Are there any inconsistencies on this drawing?", "Do any values disagree on this GAD?", "Is anything on the drawing contradictory?"]))
    if a["findings"]:
        ans = "Yes. " + " ".join(f"{cap(f)}." for f in a["findings"]) + " Each place is shown as printed; check which one is correct."
        return [("qa_finding", q, ans, [("finding", f[:30]) for f in a["findings"]])]
    return [("qa_finding", q, "I found no disagreement between the values I compare (box size in the title vs the comparative table, HFL and bed "
             "level in the table vs the views, rail and formation levels in the track details vs the views).", [])]


def first_level(a, kind, status="proposed"):
    ls = [l for l in a["levels"] if l["kind"] == kind and (l["status"] == status or (status == "proposed" and l["status"] is None))]
    vals = Counter(l["value"] for l in ls)
    return (vals.most_common(1)[0][0], [l for l in ls if l["value"] == vals.most_common(1)[0][0]]) if vals else (None, [])


def qa_computed(a):
    out = []
    fl, fls = first_level(a, "formation_level")
    tob, tobs = first_level(a, "top_of_box_level")
    if fl and tob:
        out.append(("qa_computed", vary(rng.choice(["What is the cushion (fill) over the box?", "How much fill is there between the formation and the top of the box?"])),
                    f"Computed from the printed levels: proposed formation level {lv(fl)} m - proposed top of box level {lv(tob)} m = {fl - tob:.3f} m "
                    "of fill (cushion) over the box. This is my calculation; the drawing does not print it.",
                    [("level", l["label"], l["value"]) for l in fls[:1] + tobs[:1]]))
    sof, sofs = first_level(a, "soffit_level")
    hfl, hfls = first_level(a, "hfl", status=None) if first_level(a, "hfl", status=None)[0] else first_level(a, "hfl")
    if sof and hfl:
        comp = next((r for r in (a["tables"].get("comparative_table") or {}).get("rows", []) if r.get("kind") == "vertical_clearance"), None)
        ans = (f"Computed from the printed levels: proposed soffit level {lv(sof)} m - HFL {lv(hfl)} m = {sof - hfl:.3f} m between the HFL and the underside "
               "of the top slab.")
        if comp and comp.get("proposed"):
            ans += f" The comparative table gives the vertical clearance as {comp['proposed']}."
        out.append(("qa_computed", vary("What is the clearance between the HFL and the soffit?"), ans,
                    [("level", l["label"], l["value"]) for l in sofs[:1] + hfls[:1]] + ([("table", "comparative_table", comp["description"])] if comp else [])))
    return out


def qa_summary(a):
    b, tb = a["bridge"], a["title_block"]
    parts = []
    if b.get("bridge_no"):
        parts.append(f"GAD of {b.get('category', 'bridge').lower()} no. {b['bridge_no']} ({b.get('description', '')})"
                     + (f" at CH {b['chainage']}" if b.get("chainage") else "") + (f", {b['relation_to_existing'].lower()}" if b.get("relation_to_existing") else ""))
    if tb.get("project"):
        parts.append(f"project: {tb['project']}" + (f", {tb['division']} division" if tb.get("division") else "") + (f", {tb['section']} section" if tb.get("section") else ""))
    lv_parts = []
    refs = ["bridge", "title", "views"]
    for kind in ("rail_level", "formation_level", "top_of_box_level", "soffit_level", "bed_level", "founding_level"):
        v, ls = first_level(a, kind)
        if v:
            lv_parts.append(f"{KIND_NAME[kind]} {lv(v)} m")
            refs += [("level", l["label"], l["value"]) for l in ls[:1]]
    hv, hls = first_level(a, "hfl", status=None)
    if hv:
        lv_parts.append(f"HFL {lv(hv)} m")
        refs += [("level", l["label"], l["value"]) for l in hls[:1]]
    if lv_parts:
        parts.append("proposed levels: " + ", ".join(lv_parts))
    bl = next((d for d in a["labelled_dims"] if d["kind"] == "barrel_length"), None)
    if bl:
        parts.append(f"barrel length {bl['value']:g} mm")
        refs.append(("dim", bl["label"], bl["value"]))
    spec = next((s for s in a["notes"].get("specifications", []) if re.search(r"RCC BOX", s["item"])), None)
    if spec:
        parts.append(f"{spec['item'].lower()} {spec['value']}")
        refs.append(("spec", spec["item"]))
    parts.append(f"{len(a['views'])} views ({', '.join(v['title'] for v in a['views'])})")
    if tb.get("dwg_no"):
        parts.append(f"drawing no. {tb['dwg_no']}, revision {tb.get('rev_no', '-')}, dated {tb.get('date', '-')}")
    if a["findings"]:
        parts.append("disagreements to check: " + "; ".join(a["findings"]))
        refs += [("finding", f[:30]) for f in a["findings"]]
    return [("qa_summary", vary(rng.choice(["Summarize this GAD.", "Give me a summary of this drawing.", "What is this drawing about?"])),
             cap("; ".join(parts)) + ".", refs)]


ABSENT = [("What is the pier height?", "This GAD is for an RCC box; it has no piers, so there is no pier height."),
          ("What type of bearings are used?", "An RCC box has no bearings, and the drawing does not mention any."),
          ("What is the pile length?", "The foundation here is an open foundation (no piles); the drawing gives no pile length."),
          ("What is the girder depth?", "There is no girder: the superstructure is the RCC box itself. The drawing gives no girder depth."),
          ("What is the diameter of the main reinforcement bars?", "The drawing does not give bar diameters; reinforcement details are in a separate drawing (see the notes and reference drawings)."),
          ("What is the cost of the bridge?", "The drawing does not give any cost."),
          ("Who is the contractor's site engineer?", "I don't give personal names from the drawing; the drawing's signature block is not part of what I read.")]


def qa_absent(a):
    out = []
    # a value that is not on this drawing: say so, never attach a meaning to it
    present = {f"{l['value']:.3f}" for l in a["levels"]} | {f"{d['value']:g}" for d in a["labelled_dims"] + a["dims"]}
    if a["levels"]:
        base = rng.choice(a["levels"])["value"]
        for v in (f"{base + rng.choice([-1, 1]) * rng.uniform(0.05, 3):.3f}", str(rng.randrange(1200, 9900))):
            if v not in present:
                out.append(("qa_absent", vary(rng.choice([f"What is {v}?", f"What does {v} denote on this drawing?"])),
                            f"{v} does not appear among the values I read from this drawing (levels, dimensions, tables and notes), "
                            "so I can't say what it denotes here.", []))
    for q, ans in rng.sample(ABSENT, 3):
        if "open foundation" in ans and not any(r.get("kind") == "foundation" and "OPEN" in (r.get("proposed") or "") for r in (a["tables"].get("comparative_table") or {}).get("rows", [])):
            continue
        out.append(("qa_absent", vary(q), ans, []))
    return out


# ======================================================================== images
def page_of(a):
    page = pymupdf.open(PDFS / a["source_pdf"])[0]
    if page.rotation:
        page.remove_rotation()
    return page


def pad32(im):
    W, H = (im.width + 31) // 32 * 32, (im.height + 31) // 32 * 32
    if (W, H) == im.size:
        return im
    c = Image.new("RGB", (W, H), "white")
    c.paste(im, (0, 0))
    return c


def render(page, clip, dpi=None, long_side=None):
    r = pymupdf.Rect(clip) & page.rect
    if long_side:
        z = long_side / max(r.width, r.height)
    else:
        z = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=r)
    return pad32(Image.frombytes("RGB", (pix.width, pix.height), pix.samples)), z


def save(im, name):
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f"{name}.png"
    im.save(p, optimize=True)
    return f"images/{name}.png"


def img_rows(a):
    rows = []
    page = page_of(a)
    gid = a["gad_id"]
    for vi, v in enumerate(a["views"]):
        bw, bh = v["bbox"][2] - v["bbox"][0], v["bbox"][3] - v["bbox"][1]
        if bw < 40 or bh < 40:
            continue
        im, _ = render(page, v["bbox"], long_side=min(1024, max(512, int(max(bw, bh) * 0.6))))
        path = save(im, f"{gid}_v{vi}")
        q = rng.choice(["Which view of the GAD is this and what does it represent?", "What does this diagram show?",
                        "What is this part of the drawing?"])
        ans = f"This is the {v['title']} (scale {v['scale']}). It is {v['represents']}"
        rows.append(("img_view", path, q, ans, []))
        parts, ds = view_levels_text(a, v["title"])
        if parts or ds:
            q2 = rng.choice(["Explain this view and the values on it.", "What does this diagram show, and what do its values mean?"])
            ans2 = ans + (" Its levels: " + "; ".join(parts[:12]) + "." if parts else "") + (" Labelled dimensions: " + "; ".join(ds[:6]) + "." if ds else "")
            refs = [("view", v["title"])] + [("level", l["label"], l["value"]) for l in a["levels"] if l["view"] == v["title"]][:12] \
                + [("dim", d["label"], d["value"]) for d in a["labelled_dims"] if d["view"] == v["title"]][:6]
            rows.append(("img_view_facts", path, q2, ans2, refs))
        # tiles at 150 dpi with the labelled values fully inside them
        tile_pt = TILE * 72 / DPI
        items = [("level", l) for l in a["levels"] if l["view"] == v["title"]] + [("dim", d) for d in a["labelled_dims"] if d["view"] == v["title"]]
        if not items:
            continue
        x0, y0, x1, y1 = v["bbox"]
        starts = []
        for _ in range(3 if bw * bh > 4 * tile_pt * tile_pt else 1):
            kind, it = rng.choice(items)
            cx, cy = (it["bbox"][0] + it["bbox"][2]) / 2, (it["bbox"][1] + it["bbox"][3]) / 2
            tx = min(max(x0, cx - rng.uniform(0.2, 0.8) * tile_pt), max(x0, x1 - tile_pt))
            ty = min(max(y0, cy - rng.uniform(0.2, 0.8) * tile_pt), max(y0, y1 - tile_pt))
            if all(abs(tx - sx) > tile_pt / 3 or abs(ty - sy) > tile_pt / 3 for sx, sy in starts):
                starts.append((tx, ty))
        for ti, (tx, ty) in enumerate(starts):
            clip = (tx, ty, tx + tile_pt, ty + tile_pt)
            inside = lambda b: b[0] >= clip[0] and b[1] >= clip[1] and b[2] <= clip[2] and b[3] <= clip[3]
            lvls = [l for l in a["levels"] if l["view"] == v["title"] and inside(l["bbox"])]
            dims = [d for d in a["labelled_dims"] if d["view"] == v["title"] and inside(d["bbox"])]
            plain = [d for d in a["dims"] if d["view"] == v["title"] and inside(d["bbox"])]
            if not lvls and not dims:
                continue
            im, _ = render(page, clip, dpi=DPI)
            path = save(im, f"{gid}_v{vi}_t{ti}")
            seen, L = set(), []
            for l in sorted(lvls, key=lambda l: (l["bbox"][1], l["bbox"][0])):
                if (l["label"], l["value"]) not in seen:
                    seen.add((l["label"], l["value"]))
                    L.append({"label": l["label"], "value": float(lv(l["value"])), "unit": "m", "means": KIND_NAME[l["kind"]] + elem_phrase(l["element"])})
            D = [{"label": d["label"], "value_mm": d["value"], "means": d["meaning"]} for d in dims]
            ans = {"view": v["title"], "levels": L, "labelled_dimensions": D, "other_dimension_figures_mm": sorted({p["value"] for p in plain})}
            q = rng.choice(["Read the labelled levels and dimensions that are fully visible in this crop of a GAD, as JSON.",
                            "List every level and labelled dimension fully visible in this part of the drawing (JSON), with what each means."])
            rows.append(("img_tile", path, q, json.dumps(ans, ensure_ascii=False), []))
    # tables and title block
    for p in a.get("panel", []):
        if p["kind"] not in ("comparative_table", "track_details", "specifications", "title_block", "depth_of_track_structure"):
            continue
        im, _ = render(page, p["box"], dpi=DPI)
        if im.width * im.height > 1100 * 1100:
            im, _ = render(page, p["box"], long_side=1100)
        path = save(im, f"{gid}_{p['kind']}")
        if p["kind"] in ("comparative_table", "track_details"):
            t = a["tables"].get(p["kind"]) or {}
            ans = [{k: r[k] for k in ("description", "existing", "proposed", "value") if r.get(k)} for r in t.get("rows", [])]
        elif p["kind"] == "specifications":
            ans = [{"item": s["item"], "value": s["value"]} for s in a["notes"].get("specifications", [])]
        elif p["kind"] == "depth_of_track_structure":
            ans = a["tables"].get("depth_of_track_structure") or []
        else:
            ans = {k: v for k, v in a["title_block"].items()}
        if not ans:
            continue
        q = {"comparative_table": "Read this comparative table (existing vs proposed bridge) as JSON.",
             "track_details": "Read the track details table as JSON.", "specifications": "Read the specifications as JSON.",
             "depth_of_track_structure": "Read the depth of track structure as JSON.", "title_block": "Read the title block of this GAD as JSON."}[p["kind"]]
        rows.append(("img_table", path, q, json.dumps(ans, ensure_ascii=False), []))
    return rows


# ======================================================================== assemble
def make_row(gid, split, task, n, question, answer, a, refs, image=None):
    prompt = FX.prompt(a, question, must=refs) if task.startswith(("qa_", "img_view_facts")) else question
    content = ([{"type": "image"}] if image else []) + [{"type": "text", "text": prompt}]
    row = {"id": f"{gid}_{task}_{n:03d}", "gad": gid, "split": split, "task": task,
           "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}
    if image:
        row["image"] = image
    return row


def main():
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(ANN.glob("*.json"))]
    assert any(a["gad_id"] == TEST_GAD for a in anns), f"{TEST_GAD} not annotated"
    out = {"train": [], "val": [], "test": []}
    stats = Counter()
    for a in anns:
        gid = a["gad_id"]
        # the test GAD is never trained or validated on; its questions are only used to score the trained model
        split = "test" if gid == TEST_GAD else "val" if gid in VAL_GADS else "train"
        facts = FX.all_facts(a)
        qa = (qa_levels(a) + qa_values(a, facts) + qa_dims(a) + qa_plain(a) + qa_tables(a) + qa_notes(a) + qa_abbr(a) + qa_title(a)
              + qa_bore(a) + qa_views(a) + qa_findings(a) + qa_computed(a) + qa_summary(a) + qa_absent(a))
        for n, (task, q, ans, refs) in enumerate(qa):
            out[split].append(make_row(gid, split, task, n, q, ans, a, refs))
            stats[task] += 1
        for n, (task, path, q, ans, refs) in enumerate(img_rows(a)):
            out[split].append(make_row(gid, split, task, 500 + n, q, ans, a, refs, image=path))
            stats[task] += 1
        print(f"{gid:<12} {split:<5} rows so far {len(out['train'])} train / {len(out['val'])} val / {len(out['test'])} test", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        rng.shuffle(rows)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "test_gad.txt").write_text(f"{TEST_GAD}\n{next(a['source_pdf'] for a in anns if a['gad_id'] == TEST_GAD)}\n", encoding="utf-8")
    print({k: len(v) for k, v in out.items()}, dict(stats))


if __name__ == "__main__":
    main()
