"""Build the GAD (RCC box) fine-tuning dataset from data/gad/annotations.

    .venv\\Scripts\\python gad_tools\\build_dataset_gad.py

Splits (by whole GAD): five GADs are held back for the test (TEST_GADS: never in train or validation; their questions
go to test.jsonl, only to score the trained model, and their PDFs go in the bundle for your own test), a few GADs are
validation, the rest train.

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
    qa_revision   the revision table: each revision's number, date and description
    qa_band       road / drain L-section and ground profile value tables: what they hold, a value at a chainage/offset
    qa_keyplan    key plan: stations on each side, tracks, curves, gradients, chainage marks, boundary, bore holes, flow
    qa_bore       bore log: SBC by depth, soil layers
    qa_view       which views there are and what each represents
    qa_finding    values that disagree between places on the drawing
    qa_computed   simple arithmetic from printed levels (cushion over the box, clearance), shown as computed
    qa_summary    a summary of the GAD
    qa_absent     things that are not on the drawing -> said so, nothing invented
    qa_callout    anything written on the views (labels over several lines joined): what it is, where, "is X on view Y?"
    qa_cl         a centre line: no value of its own; the distances from it to the other centre lines
    qa_why        "why ...?" from what the drawing shows (legend, title, tables, notes, levels); no reason on it -> said so
  Images (so the model knows what each diagram looks like and can read it):
    img_view      a whole view -> which view it is and what it represents (no values: too small to read at this size)
    img_view_facts  a whole view + its facts -> what it shows, with the values and their meanings
    img_tile      a 768 x 768 tile of a view at 150 dpi -> the labelled levels and dimensions fully visible in it, as JSON
    img_table     comparative table / track details / specifications / title block crop at 150 dpi -> JSON
    img_spot      a crop around what a question is about + the facts + the question (as ask_gad sends it) -> the answer
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
# held back completely, one of each kind: a plain single box, a twin box, a RUB (road L-section and ground profile),
# a DYCE-format drawing (a slab bridge, notes in another layout) and a box on a curve
TEST_GADS = ["810-1", "823-2", "RUB-801-1", "799-1", "752-3"]
VAL_GADS = {"752-2", "787-1", "818-2", "RUB-789-1"}
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


# the way engineers actually type: shorthand for the same thing (the facts finder knows these words: facts.SYN)
SHORTHAND = [(r"\bformation level\b", ["FL", "formation lvl", "formation"]), (r"\bsoffit level\b", ["soffit lvl", "soffit"]),
             (r"\bfounding level\b", ["FDN level", "foundation level", "founding lvl", "FDN lvl"]),
             (r"\bbed level\b", ["BL", "bed lvl"]), (r"\btop of box level\b", ["TOB level", "box top level", "top of box"]),
             (r"\brail level\b", ["rail lvl", "rail top level"]), (r"\bexisting\b", ["exg", "ex.", "old"]),
             (r"\bproposed\b", ["prop.", "new"]), (r"\bdrawing number\b", ["dwg no", "drawing no.", "DWG number"]),
             (r"\bcomparative table\b", ["hydraulic data", "comparison table"]), (r"\btrack centre distance\b", ["T/C", "track centres"]),
             (r"\bvertical clearance\b", ["VC", "clearance"]), (r"\bthis GAD\b", ["this drawing", "the GAD", "this sheet"]),
             (r"\bbridge\b", ["br.", "bridge"])]
OPENERS = [(r"^What is the ", ["What's the ", "Give me the ", "Tell me the ", "Find the ", "", "What will be the ", "Show me the "]),
           (r"^Which ", ["What ", "Which "])]


def vary(q):
    """Variation of a question's wording, numbers untouched: openers, the shorthand engineers use, endings, case,
    the odd typo. (v1 varied only the first letter: real questions are worded every which way.)"""
    for pat, alts in OPENERS:
        if re.search(pat, q) and rng.random() < 0.5:
            q = re.sub(pat, rng.choice(alts), q, count=1)
            q = q[:1].upper() + q[1:]
    for pat, alts in SHORTHAND:
        if re.search(pat, q, re.I) and rng.random() < 0.3:
            q = re.sub(pat, rng.choice(alts), q, count=1, flags=re.I)
    if rng.random() < 0.12:
        q = rng.choice(["Quick one: ", "Hey, ", "Can you check: ", "Tell me: ", "Pls tell ", "I need "]) + q[0].lower() + q[1:]
    if rng.random() < 0.08 and q.endswith("?"):
        q = q[:-1] + rng.choice([" on this GAD?", " for this bridge?", " here?", " as per the drawing?"])
    r = rng.random()
    if r < 0.2:
        q = q[0].lower() + q[1:]
    elif r < 0.25:
        q = q.upper()
    if rng.random() < 0.2:
        q = q.rstrip("?").rstrip()
    if rng.random() < 0.05:                       # a typo in one longer word (never in a number)
        ws = [i for i, w in enumerate(q.split(" ")) if len(w) > 5 and w.isalpha()]
        if ws:
            parts = q.split(" ")
            w = parts[rng.choice(ws)]
            j = rng.randrange(1, len(w) - 2)
            parts[parts.index(w)] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            q = " ".join(parts)
    return q


def bridge_desc(b):
    """ " (1x3.660x6.530m RCC BOX)" - the title's description of the bridge, or its size when the title has no brackets."""
    d = b.get("description") or (b.get("box") or b.get("slab") or {}).get("as_printed")
    return f" ({d})" if d else ""


def elem_phrase(el):
    if not el:
        return ""
    e = el.lower().replace("r/wall", "return wall").replace("r/ wall", "return wall").replace("rcc ", "")
    return f" of the {e}"


def views_phrase(views):
    vs = sorted(set(v for v in views if v))
    return ("on " + " and ".join(vs)) if vs else "outside the drawn views"


# ======================================================================== question / answer generators
# how engineers type a level, whatever the drawing prints: the kind's shorthand and the status's
LEVEL_SHORT = {"formation_level": ["FL", "F.L", "FRL", "formation lvl", "formation"], "rail_level": ["rail lvl", "rail level", "TRL", "rail top"],
               "soffit_level": ["soffit lvl", "soffit", "SOFFIT LVL"], "top_of_box_level": ["TOB", "TOB lvl", "top of box", "box top lvl"],
               "bed_level": ["BL", "B.L", "bed lvl", "BED LVL"], "founding_level": ["FDN lvl", "FND LVL", "founding lvl", "FDN"],
               "hfl": ["HFL", "H.F.L"], "observed_hfl": ["OHFL", "O.H.F.L"], "calculated_hfl": ["CHFL", "C.H.F.L"]}
STATUS_SHORT = {"proposed": ["PROP", "PROP.", "prop", "proposed", "new"], "existing": ["EXG", "EX.", "exg", "existing", "exist", "old"], None: [""]}


def terse(kind, status):
    """'PROP FL', 'exg BL?', 'FL (proposed)', 'TOB lvl': a question as short as engineers type it."""
    k = rng.choice(LEVEL_SHORT.get(kind, [KIND_NAME.get(kind, kind)]))
    st = rng.choice(STATUS_SHORT.get(status, [""]))
    form = rng.choice(["{s} {k}", "{s} {k}?", "{k} ({s})", "{s} {k} value", "what is {s} {k}", "{s} {k} of the box"])
    q = re.sub(r"\s+", " ", form.format(s=st, k=k)).replace("( )", "").replace("()", "").strip()
    return rng.choice([q, q.upper(), q.lower()])


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
        if not el:                                     # and as engineers type it: "PROP FL", "exg BL?", "TOB lvl"
            out.append(("qa_level", terse(kind, status), ans, refs))
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
            out.append(("qa_level", terse(kind, None), ans, refs))
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
            ans = f"The track-centre distance{btw} is {v} mm ('{d['label']}' on {FX.view_name(d['view'])}). T/C means track centres: {d['meaning']}."
            qs = [f"What is the track centre distance{btw}?", "What is the T/C on this drawing?", f"What does {v} mean on the drawing?"]
        elif d["kind"] == "barrel_length":
            ans = f"The barrel length is {v} mm ('{d['label']}' on {FX.view_name(d['view'])}): {d['meaning']}."
            qs = ["What is the barrel length?", "How long is the box barrel?", f"What is {v}?"]
        elif d["kind"] == "thickness":
            ans = f"{cap(d['meaning'])} is {v} mm ('{d['label']}' on {FX.view_name(d['view'])})."
            what = (d["what"] or "item").lower()
            qs = [f"How thick is the {what}?", f"What is the thickness of the {what}?"]
        elif d["kind"] == "diameter":
            ans = f"{cap(d['meaning'])} is {v} mm ('{d['label']}' on {FX.view_name(d['view'])})."
            qs = [f"What is the diameter of the {(d['what'] or 'pipe').lower()}?", f"What does '{d['label']}' mean?"]
        else:
            ans = f"'{d['label']}' on {FX.view_name(d['view'])}: {d['meaning']} ({v} mm)."
            qs = [f"What does '{d['label']}' mean?"]
        out.append(("qa_dim", vary(rng.choice(qs)), ans, ref))
    for s in a["slopes"][:4]:
        out.append(("qa_dim", vary(rng.choice([f"What does '{s['label']}' mean on {FX.view_name(s['view'])}?", f"What is the slope '{s['label']}'?"])),
                    f"'{s['label']}' on {FX.view_name(s['view'])} is {s['meaning']}.", [("slope", s["label"])]))
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
        ds = [d for d in ds if not FX.dim_span(d)]
        for d in rng.sample(ds, min(2, len(ds))):
            v = f"{d['value']:g}"
            same = [x for x in ds if x["value"] == d["value"]]
            ans = (f"{v} is a dimension in millimetres written {d['direction']}ly on {view}"
                   + (f" ({len(same)} times)" if len(same) > 1 else "") +
                   ". The drawing does not write what it measures (it is a plain dimension figure), so I can only say which view it is "
                   "on and its direction; the drawn dimension line shows the exact points it is measured between.")
            out.append(("qa_plain_dim", vary(rng.choice([f"What does {v} on the {view.lower()} represent?", f"What is the {v} dimension on {view}?"])),
                        ans, [("plain", view, d["direction"])]))
    # what an unlabelled dimension measures, read from where its dimension line's ends are (never a guessed name)
    with_ends = [d for d in a["dims"] if FX.dim_span(d)]
    circ = [d for d in with_ends if d.get("circles")]
    for d in list(dict.fromkeys(map(id, circ[:2] + rng.sample(with_ends, min(5, len(with_ends)))))):
        d = next(x for x in with_ends if id(x) == d)
        out.append(plain_answer(a, d))
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
                    f"It is {b.get('category', 'bridge').lower()} no. {b['bridge_no']}{bridge_desc(b)}"
                    + (f" at CH {b['chainage']}" if b.get("chainage") else "") + (f", {b['relation_to_existing'].lower()}" if b.get("relation_to_existing") else "") + ".",
                    ["bridge", "title"]))
    if b.get("box"):
        bx = b["box"]
        out.append(("qa_title", vary(rng.choice(["What is the size of the box?", "What is the box size?", "How big is the opening?"])),
                    f"The title gives the box as {bx['as_printed'].rstrip('mM ')} m: {bx['cells']} cell(s) of clear width {bx['clear_width_m']:g} m and clear "
                    f"height {bx['clear_height_m']:g} m." + (" Note: the comparative table gives a different size - see the disagreements." if any("box size" in f for f in a["findings"]) else ""),
                    ["bridge", "title"] + [("table", "comparative_table", "SPAN")]))
    elif b.get("slab"):
        sl = b["slab"]
        out.append(("qa_title", vary(rng.choice(["What is the size of the box?", "What is the span of the bridge?", "Is this an RCC box?"])),
                    f"This GAD is not for a box: the title gives a slab bridge, {sl['as_printed']} - {sl['spans']} span(s) of clear span "
                    f"{sl['clear_span_m']:g} m ({sl['type']}).", ["bridge", "title"]))
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
            out.append(("qa_bore", vary(rng.choice(["What safe bearing capacities does the bore log give?", "What are the SBC values shown in the sheet?",
                                                    "List the SBC with depth."])),
                        "From the bore log: " + "; ".join(f"{x['sbc_t_per_m2']:g} t/m² at {x['depth_m']:g} m" if x.get("depth_m") is not None else f"{x['sbc_t_per_m2']:g} t/m²" for x in bl["sbc"]) + ".", ref))
            # a depth the log does not print: said so, with the printed values either side - never a made-up value
            dep = sorted((x["depth_m"], x["sbc_t_per_m2"]) for x in bl["sbc"] if x.get("depth_m") is not None)
            if len(dep) >= 2:
                k = rng.randrange(len(dep) - 1)
                (d1, s1), (d2, s2) = dep[k], dep[k + 1]
                d = round(rng.uniform(d1, d2) * 2) / 2
                if d1 < d < d2 and all(abs(d - x) > 1e-6 for x, _ in dep):
                    out.append(("qa_bore", vary(rng.choice([f"What is the SBC at {d:g} m depth?", f"SBC at {d:g}m?"])),
                                f"The bore log does not print an SBC at {d:g} m. The nearest printed values are {s1:g} t/m² at {d1:g} m "
                                f"and {s2:g} t/m² at {d2:g} m ({bl['view']}).", ref))
                d = dep[-1][0] + rng.choice([1, 2, 3])
                out.append(("qa_bore", vary(f"What is the SBC at {d:g} m depth?"),
                            f"The bore log does not go that deep: its deepest SBC is {dep[-1][1]:g} t/m² at {dep[-1][0]:g} m ({bl['view']}).", ref))
        if bl.get("deepest_m") is not None or (bl.get("rl_top") is not None and bl.get("rl_bottom") is not None and bl["rl_top"] > bl["rl_bottom"]):
            parts = []
            if bl.get("deepest_m") is not None:
                parts.append(f"its deepest SBC is printed at {bl['deepest_m']:g} m depth")
            if bl.get("rl_top") is not None and bl.get("rl_bottom") is not None and bl["rl_top"] > bl["rl_bottom"]:
                parts.append(f"it runs from RL {bl['rl_top']:.3f} m down to RL {bl['rl_bottom']:.3f} m, "
                             f"{round(bl['rl_top'] - bl['rl_bottom'], 3):g} m")
            out.append(("qa_bore", vary(rng.choice(["How deep does the bore log go?", "What is the depth of the bore log?", "Up to what depth was the soil tested?"])),
                        f"The bore log ({bl['view']}) goes down to about {bl['deepest_m'] if bl.get('deepest_m') is not None else round(bl['rl_top'] - bl['rl_bottom'], 3):g} m: "
                        + "; ".join(parts) + ".", ref))
        if bl["layers"]:
            out.append(("qa_bore", vary(rng.choice(["What soil layers does the bore log show?", "What is the soil at the bridge site?"])),
                        "The bore log shows (top to bottom): " + "; ".join(x["soil"].lower() for x in bl["layers"]) + ".", ref))
        if bl.get("chainage"):
            out.append(("qa_bore", vary("At what chainage was the bore log taken?"), f"The bore log is at chainage {bl['chainage']} m ({bl['view']}).", ref))
    return out


def qa_review(a):
    """Reviewer's markup on the drawing (blue comments): listed when there is any, said so when there is none."""
    rc = a["notes"].get("review_comments") or []
    q = vary(rng.choice(["Are there any review comments on this drawing?", "What has the reviewer marked on this GAD?",
                         "Are there any remarks or corrections marked on the drawing?"]))
    if not rc:
        return [("qa_review", q, "No reviewer's comments or markup are marked on this drawing.", [])]
    ans = (f"Yes, {len(rc)} reviewer's comment(s) are marked on the drawing (blue markup, not part of the design): "
           + "; ".join(f"{(it['no'] + '. ') if it['no'] else ''}{it['text']}" for it in rc) + ".")
    return [("qa_review", q, ans, [("review_comments", it["no"], it["text"][:30]) for it in rc])]


def qa_revisions(a):
    revs = a["title_block"].get("revisions") or []
    if not revs:
        return []
    out = [("qa_revision", vary(rng.choice(["What is the revision history of this drawing?", "List the revisions of this GAD.",
                                             "How many times has this drawing been revised?"])),
            f"The revision table lists {len(revs)} entr{'y' if len(revs) == 1 else 'ies'}: "
            + "; ".join(f"{r['rev_no']} dated {r['date'] or '(no date written)'}, {(r['description'] or '(no description)').lower()}" for r in revs)
            + f". The latest is {revs[-1]['rev_no']}.", ["revisions"])]
    r = rng.choice(revs)
    out.append(("qa_revision", vary(rng.choice([f"What was revision {r['rev_no']} about?", f"When was {r['rev_no']} issued?"])),
                f"{r['rev_no']} is dated {r['date'] or '(no date written)'}: {(r['description'] or 'no description is written').lower()}.", ["revisions"]))
    return out


def qa_keyplan(a):
    out = []
    for v in a["views"]:
        kp = v.get("key_plan")
        if not kp:
            continue
        side = {s: [x["text"] for x in kp.get("stations", []) if x["side"] == s] for s in ("left", "right")}
        if side["left"] or side["right"]:
            ans = "From the key plan: " + "; ".join(f"towards the {s} end of the line: {', '.join(t)}" for s, t in side.items() if t) + "."
            out.append(("qa_keyplan", vary(rng.choice(["Which stations are on either side of the bridge?", "Between which stations is this bridge?",
                                                        "Where does the line go on each side of the bridge?"])),
                        ans, [("kp", "stations", s) for s, t in side.items() if t]))
        if kp.get("tracks"):
            out.append(("qa_keyplan", vary(rng.choice(["Which tracks are shown on the key plan?", "Which lines does the key plan show?"])),
                        "The key plan labels these tracks: " + "; ".join(dict.fromkeys(kp["tracks"])) + ".", [("kp", "tracks")]))
        if kp.get("curves"):
            out.append(("qa_keyplan", vary(rng.choice(["Is there any curve near the bridge?", "What curves are shown on the key plan?"])),
                        "The key plan writes these curve marks (radius labels): " + ", ".join(dict.fromkeys(kp["curves"]))
                        + ". It is a location sketch, so check the alignment drawing for the full curve data.", [("kp", "curves")]))
        else:
            out.append(("qa_keyplan", vary("Is there any curve near the bridge?"),
                        "The key plan does not mark any curve radius near the bridge.", [("kp", "tracks")]))
        if kp.get("gradients"):
            out.append(("qa_keyplan", vary(rng.choice(["What gradient is the track on at the bridge?", "Which gradients does the key plan show?"])),
                        "The key plan shows the gradients " + "; ".join(dict.fromkeys(kp["gradients"])) + ".", [("kp", "gradients")]))
        if kp.get("markers"):
            out.append(("qa_keyplan", vary(rng.choice(["Which chainages or KMs are marked on the key plan?", "What reference marks are on the key plan?"])),
                        "The key plan marks: " + "; ".join(m["text"] for m in kp["markers"]) + ".", [("kp", "markers")]))
        if kp.get("boundary"):
            out.append(("qa_keyplan", vary(rng.choice(["Does the key plan show the railway boundary?", "Is any land to be acquired?"])),
                        "The key plan shows: " + "; ".join(dict.fromkeys(kp["boundary"])) + "."
                        + (" So some land is marked to be acquired." if any("ACQUI" in b for b in kp["boundary"]) else ""), [("kp", "boundary")]))
        if kp.get("bridges"):
            out.append(("qa_keyplan", vary(rng.choice(["What does the key plan say about the bridge?", "How is the bridge described on the key plan?"])),
                        "The key plan's callout reads: " + " / ".join(kp["bridges"]) + ".", [("kp", "bridges")]))
        if kp.get("bore_holes") or kp.get("flow"):
            ans = []
            if kp.get("bore_holes"):
                ans.append(f"{kp['bore_holes']} bore hole location(s) (BH)")
            if kp.get("flow"):
                ans.append("the direction of flow (FLOW arrow)")
            out.append(("qa_keyplan", vary("Does the key plan show the bore holes or the flow direction?"),
                        "Yes, the key plan marks " + " and ".join(ans) + ".", [("kp", "extra")]))
    return out


def qa_bands(a):
    """The value tables under road / drain L-sections and ground profiles: what they hold, and values at a given column."""
    out = []
    for v in a["views"]:
        bd = v.get("band")
        if not bd or not bd["columns"]:
            continue
        labels = [r["label"] for r in bd["rows"]]
        refs = [("band", v["title"], k) for k in range((len(bd["columns"]) + 11) // 12)]
        name = v["title"].lower().replace("l- section", "L-section").replace("l-section", "L-section")
        desc = "; ".join(f"{r['label']}: {len(r['values'])} values, from {r['values'][0]} (left) to {r['values'][-1]} (right)" for r in bd["rows"])
        out.append(("qa_band", vary(rng.choice([f"What does the table under the {name} show?", f"Explain the value table of the {name}.",
                                                 f"What values are given on the {name}?"])),
                    f"Under the {v['title']} there is a value table with {len(labels)} row(s) and {len(bd['columns'])} columns: {desc}.",
                    refs))
        key = next((l for l in labels if re.search(r"CHAINAGE|OFFSET|DISTANCE", l, re.I)), None)
        others = [l for l in labels if l != key]
        if not key or not others:
            continue
        cols = [(k, c) for k, c in enumerate(bd["columns"]) if key in c and any(o in c for o in others)]
        for k, c in rng.sample(cols, min(3, len(cols))):
            o = rng.choice([o for o in others if o in c])
            out.append(("qa_band", vary(f"On the {name}, what is the {o.lower()} at {key.lower()} {c[key]}?"),
                        f"In the value table under the {v['title']}, the column with {key} {c[key]} gives {o} {c[o]}.",
                        [("band", v["title"], k // 12)]))
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
                "The drawing has these views: " + "; ".join(f"{v['title']} (scale {v['scale'] or 'not written'})" for v in vs) + ".", ["views"]))
    for v in vs:
        parts, ds = view_levels_text(a, v["title"])
        ans = f"The {v['title']} (scale {v['scale'] or 'not written'}) is {v['represents']}"
        if parts:
            ans += " On this drawing it shows the levels " + ", ".join(parts[:10]) + ("" if len(parts) <= 10 else ", and more") + "."
        if ds:
            ans += " Labelled dimensions on it: " + "; ".join(ds[:6]) + "."
        q = rng.choice([f"What does the {v['title'].lower()} represent?", f"What is shown in {v['title']}?", f"Explain the {v['title'].lower()}."])
        out.append(("qa_view", vary(q), ans, [("view", v["title"])] + list(dict.fromkeys(("level", l["label"], l["value"]) for l in a["levels"] if l["view"] == v["title"]))[:10]
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
        parts.append(f"GAD of {b.get('category', 'bridge').lower()} no. {b['bridge_no']}{bridge_desc(b)}"
                     + (f" at CH {b['chainage']}" if b.get("chainage") else "") + (f", {b['relation_to_existing'].lower()}" if b.get("relation_to_existing") else ""))
    if tb.get("project"):
        parts.append(f"project: {tb['project']}" + (f", {tb['division']} division" if tb.get("division") else "") + (f", {tb['section']} section" if tb.get("section") else ""))
    lv_parts = []
    refs = ["bridge", "title", "views", "drawing", "project"]      # (v1 guessed the drawing no. / revision: not in the prompt)
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


# ======================================================================== reasoning: worked checks with the formulas
def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else None


def trow(a, table, kind):
    return next((r for r in (a["tables"].get(table) or {}).get("rows", []) if r.get("kind") == kind), None)


def tref(table, r):
    return ("table", table, r["description"])


def qa_reason(a):
    """Values the drawing gives, checked with the formula behind them (vertical clearance, free board, height of water,
    scour level, depth of track structure, the concrete grade rule and the founding pressure in the notes)."""
    out = []
    C, TD = "comparative_table", "track_details"
    hfls = []
    for kind, name in (("observed_hfl", "OHFL"), ("calculated_hfl", "CHFL"), ("hfl", "HFL")):
        r = trow(a, C, kind)
        if r and num(r.get("proposed")):
            hfls.append((name, num(r["proposed"]), tref(C, r)))

    def check(q, what, base_name, base, base_ref, row, why):
        tv = num(row.get("proposed")) if row else None
        if base is None or base_ref is None or tv is None or not hfls:
            return
        h = next((h for h in hfls if abs(base - h[1] - tv) < 0.006), None)
        if h:
            ans = (f"{cap(what)} = {base_name} - design HFL = {base:.3f} - {h[1]:.3f} ({h[0]}) = {base - h[1]:.3f} m. The comparative "
                   f"table gives {row['proposed']}, so it agrees, and it shows the design HFL used is the {h[0]}. {why}")
            refs = [base_ref, h[2], tref(C, row)]
        else:
            n0, v0, _ = hfls[0]
            ans = (f"{cap(what)} = {base_name} - design HFL. With the {n0} {v0:.3f} m: {base:.3f} - {v0:.3f} = {base - v0:.3f} m. "
                   f"The comparative table gives {row['proposed']}, which none of the HFLs printed on the drawing gives "
                   f"(it would need an HFL of {base - tv:.3f} m) - worth checking which HFL the table used. {why}")
            refs = [base_ref, tref(C, row)] + [x[2] for x in hfls]
        out.append(("qa_reason", vary(q), ans, refs))

    sof, sofs = first_level(a, "soffit_level")
    td_fl = trow(a, TD, "formation_level")
    fl = num(td_fl.get("proposed")) if td_fl else None
    check(rng.choice(["How is the vertical clearance worked out? Check it.", "Is the vertical clearance in the table right?",
                      "Explain the vertical clearance of the box."]),
          "the vertical clearance", "soffit level", sof, ("level", sofs[0]["label"], sofs[0]["value"]) if sofs else None,
          trow(a, C, "vertical_clearance"),
          "It is the free height between the flood level and the underside of the top slab, so that the flood and floating debris pass under the box.")
    check(rng.choice(["How is the free board worked out? Check it.", "Explain the free board on this GAD.", "Is the free board correct?"]),
          "the free board", "formation level", fl, tref(TD, td_fl) if td_fl else None, trow(a, C, "free_board"),
          "It is the height of the formation (top of the embankment) above the flood level, which keeps the track bed dry in a flood.")
    bed_r = trow(a, C, "bed_level")
    bed = num(bed_r.get("proposed")) if bed_r else None
    how_r = trow(a, C, "height_of_water")
    if bed is not None and how_r and num(how_r.get("proposed")) is not None and hfls:
        tv = num(how_r["proposed"])
        h = next((h for h in hfls if abs(h[1] - bed - tv) < 0.006), None)
        if h:
            ans = (f"Height (depth) of water = design HFL - bed level = {h[1]:.3f} ({h[0]}) - {bed:.3f} = {h[1] - bed:.3f} m, which matches "
                   f"the table's {how_r['proposed']}. It is the depth of the flood water over the bed at the box.")
        else:
            ans = (f"Height of water = design HFL - bed level. With the {hfls[0][0]} {hfls[0][1]:.3f} and the bed level {bed:.3f}: "
                   f"{hfls[0][1] - bed:.3f} m, but the table gives {how_r['proposed']} - it does not follow from the printed HFLs and "
                   "bed level; worth checking.")
        out.append(("qa_reason", vary(rng.choice(["How is the height of water worked out?", "Check the height of water in the table."])),
                    ans, [tref(C, bed_r), tref(C, how_r)] + [x[2] for x in hfls]))
    sd_r, sl_r = trow(a, C, "max_scour_depth"), trow(a, C, "max_scour_level")
    if sd_r and sl_r and num(sd_r.get("proposed")) is not None and num(sl_r.get("proposed")) is not None and hfls:
        sd, sl = num(sd_r["proposed"]), num(sl_r["proposed"])
        h = next((h for h in hfls if abs(h[1] - sd - sl) < 0.006), None)
        if h:
            ans = (f"Maximum scour level = design HFL - maximum scour depth = {h[1]:.3f} ({h[0]}) - {sd:.3f} = {h[1] - sd:.3f} m, as the "
                   f"table gives ({sl_r['proposed']}). The scour depth is measured down from the flood level, and the foundation and the "
                   "drop/curtain walls must go below this level so the flood cannot undermine them.")
        else:
            ans = (f"Maximum scour level should be the design HFL minus the maximum scour depth ({sd:.3f} m), but none of the printed "
                   f"HFLs gives the table's {sl_r['proposed']} that way - worth checking.")
        out.append(("qa_reason", vary(rng.choice(["How is the maximum scour level worked out?", "Check the scour level."])),
                    ans, [tref(C, sd_r), tref(C, sl_r)] + [x[2] for x in hfls]))
    # depth of track structure = rail level - formation level = the items of the depth table
    td_rl = trow(a, TD, "rail_level")
    dts = a["tables"].get("depth_of_track_structure") or []
    items = [(x["item"], num(x["value"])) for x in dts if x["item"].upper() != "TOTAL" and num(x["value"]) is not None]
    tot = next((num(x["value"]) for x in dts if x["item"].upper() == "TOTAL"), None)
    if td_rl and td_fl and num(td_rl.get("proposed")) is not None and fl is not None and items:
        rl = num(td_rl["proposed"])
        s_items = sum(v for _, v in items)
        d = round((rl - fl) * 1000)
        ans = (f"Rail level - formation level = {rl:.3f} - {fl:.3f} = {rl - fl:.3f} m = {d} mm. The depth of track structure adds up to "
               + " + ".join(f"{v:g} ({k.lower()})" for k, v in items) + f" = {s_items:g} mm"
               + (f" (printed total {tot:g} mm)" if tot else "") + ". ")
        ans += ("They agree: the formation is one track-structure depth below the rail." if abs(d - s_items) <= 2 else
                f"They do NOT agree ({d} mm vs {s_items:g} mm): the rail or formation level in the track details should be checked.")
        out.append(("qa_reason", vary(rng.choice(["Do the rail level and formation level agree with the depth of track structure?",
                                                   "Check the rail and formation levels against the track structure."])),
                    ans, [tref(TD, td_rl), tref(TD, td_fl), ("table", "depth")]))
    # notes: concrete grade by box size
    notes = a["notes"].get("notes", []) + a["notes"].get("special_note", [])
    gnote = next((it for it in notes if re.search(r"M\s*-?\s*35.*SPAN\s+OR\s+HEIGHT", it["text"], re.I)), None)
    spec = next((s for s in a["notes"].get("specifications", []) if re.search(r"GRADE\s+OF\s+RCC\s+BOX", s["item"], re.I) and s.get("value")), None)
    bx = a["bridge"].get("box")
    if gnote and spec and bx:
        mx = max(bx["clear_width_m"], bx["clear_height_m"])
        need = "M35" if mx > 4 else "M30"
        got = re.sub(r"[\s-]", "", spec["value"].upper())
        ok = got.startswith(need)
        ans = (f"Note {gnote['no']} says M35 concrete is used when the span or height of the box is more than 4 m, M30 otherwise. The box is "
               f"{bx['clear_width_m']:g} m wide and {bx['clear_height_m']:g} m high, so the larger is {mx:g} m, "
               f"{'more' if mx > 4 else 'not more'} than 4 m: {need} is needed. The specifications give {spec['value']} for the RCC box - "
               + ("consistent with the note." if ok else "NOT what the note asks for; worth checking."))
        out.append(("qa_reason", vary(rng.choice(["Is the concrete grade of the box correct as per the notes?",
                                                   "Which concrete grade should the box have and why?"])),
                    ans, ["bridge", ("notes", gnote["no"], gnote["text"][:30]), ("spec", spec["item"])]))
    # notes: founding pressure against the SBC at the founding level
    pnote = next((it for it in notes if re.search(r"FOUNDING\s+PRESSURE\s+OF\s+(?:THE\s+)?RCC\s+BOX\s*=\s*\d", it["text"], re.I)), None)
    fdn, fdns = first_level(a, "founding_level")
    bl = next((b for b in a["bore_logs"] if any(s.get("depth_m") is not None for s in b["sbc"])), None)
    if pnote and fdn is not None and bed is not None and bl:
        p = float(re.search(r"=\s*(\d+(?:\.\d+)?)", pnote["text"]).group(1))
        depth = bed - fdn
        rows = sorted([s for s in bl["sbc"] if s.get("depth_m") is not None], key=lambda s: s["depth_m"])
        at = next((s for s in rows if s["depth_m"] >= depth - 0.05), rows[-1])
        ok = at["sbc_t_per_m2"] >= p
        ans = (f"Note {pnote['no']} gives the founding pressure of the box as {p:g} t/m². The founding level is {fdn:.3f} m, about "
               f"{depth:.2f} m below the bed level {bed:.3f} m. Taking the bore log depths as measured from the bed (an approximation: "
               f"the bore log is measured from the ground at the bore hole), the SBC at about {at['depth_m']:g} m is "
               f"{at['sbc_t_per_m2']:g} t/m², which is "
               + (f"more than {p:g} t/m²: the soil can carry the box, as the notes require." if ok else
                  f"less than {p:g} t/m²: the notes require the SBC at the founding level to exceed the founding pressure, so this "
                  "needs checking (or the ground improvement the notes describe)."))
        out.append(("qa_reason", vary(rng.choice(["Is the soil strong enough for the box?", "Check the founding pressure against the SBC.",
                                                   "Does the SBC satisfy the founding pressure in the notes?"])),
                    ans, [("notes", pnote["no"], pnote["text"][:30]), ("level", fdns[0]["label"], fdns[0]["value"]), tref(C, bed_r),
                          ("bore", bl["view"])]))
    return out


FLAG = re.compile(r"\bNOT\b|worth checking|needs checking|less than \d")


def _set_value(text, new):
    """'0.811m' -> '1.211m': the number replaced, its unit and spacing kept."""
    return re.sub(r"-?\d+(?:\.\d+)?", new, text, count=1)


def qa_reason_flags(a):
    """Checks that must NOT agree: a copy of the GAD's facts with one value changed (a table value, the formation level,
    the box's concrete grade, the box size at the 4 m limit, the founding pressure), and the worked check on it.
    v1 learnt mostly agreeing checks and once invented an HFL to make one agree; these teach it to flag instead.
    Returns [(task, question, answer, refs, changed annotation)] - train / validation only, never the test GADs."""
    import copy
    out = []

    def run(mod, keyword):
        for task, q, ans, refs in qa_reason(mod):
            if keyword in ans.lower() and FLAG.search(ans):
                out.append((task, q, ans, refs, mod))
                return

    C = (a["tables"].get("comparative_table") or {}).get("rows", [])
    for kind, keyword in (("vertical_clearance", "vertical clearance"), ("free_board", "free board"),
                          ("height_of_water", "height (depth) of water"), ("max_scour_level", "scour level")):
        r = next((r for r in C if r.get("kind") == kind and num(r.get("proposed")) is not None), None)
        if not r:
            continue
        mod = copy.deepcopy(a)
        rr = next(x for x in mod["tables"]["comparative_table"]["rows"] if x.get("kind") == kind)
        d = rng.choice([-1, 1]) * rng.uniform(0.15, 0.8)
        rr["proposed"] = _set_value(rr["proposed"], f"{num(rr['proposed']) + d:.3f}")
        run(mod, "height of water" if kind == "height_of_water" else keyword)
    td = (a["tables"].get("track_details") or {}).get("rows", [])
    fl = next((r for r in td if r.get("kind") == "formation_level" and num(r.get("proposed")) is not None), None)
    if fl:
        mod = copy.deepcopy(a)
        rr = next(x for x in mod["tables"]["track_details"]["rows"] if x.get("kind") == "formation_level")
        rr["proposed"] = _set_value(rr["proposed"], f"{num(rr['proposed']) - rng.uniform(0.05, 0.25):.3f}")
        run(mod, "track structure")
    spec_i = next((i for i, x in enumerate(a["notes"].get("specifications", [])) if re.search(r"GRADE\s+OF\s+RCC\s+BOX", x["item"], re.I)
                   and x.get("value") and re.search(r"M\s*-?\s*3[05]", x["value"])), None)
    if spec_i is not None:
        mod = copy.deepcopy(a)
        sp = mod["notes"]["specifications"][spec_i]
        sp["value"] = re.sub(r"3[05]", lambda m: "30" if m.group() == "35" else "35", sp["value"], count=1)
        run(mod, "concrete")
    bx = a["bridge"].get("box")
    if bx and spec_i is not None:
        # the 4 m limit: exactly 4 is not more than 4
        w = rng.choice([4.0, 3.98, 4.05, 4.0])
        mod = copy.deepcopy(a)
        b2 = mod["bridge"]["box"]
        old = b2["as_printed"].rstrip("mM ")
        b2["clear_width_m"], b2["clear_height_m"] = w, min(b2["clear_height_m"], 3.9)
        new = f"{b2['cells']}x{w:.3f}x{b2['clear_height_m']:.3f}"
        b2["as_printed"] = new + "m"
        for holder, key in ((mod["bridge"], "description"), (mod["title_block"], "title")):
            if holder.get(key):
                holder[key] = holder[key].replace(old, new)
        need = "M35" if w > 4 else "M30"
        sp = mod["notes"]["specifications"][spec_i]
        sp["value"] = re.sub(r"3[05]", "30" if need == "M35" else "35", sp["value"], count=1)    # the wrong one
        run(mod, "concrete")
    notes = a["notes"].get("notes", []) + a["notes"].get("special_note", [])
    pn = next((it for it in notes if re.search(r"FOUNDING\s+PRESSURE\s+OF\s+(?:THE\s+)?RCC\s+BOX\s*=\s*\d", it["text"], re.I)), None)
    bl = next((b for b in a["bore_logs"] if any(x.get("depth_m") is not None for x in b["sbc"])), None)
    if pn and bl:
        top = max(x["sbc_t_per_m2"] for x in bl["sbc"])
        mod = copy.deepcopy(a)
        for key in ("notes", "special_note"):
            for it in mod["notes"].get(key, []):
                if it["text"] == pn["text"]:
                    it["text"] = re.sub(r"(=\s*)(\d+(?:\.\d+)?)", lambda m: m.group(1) + f"{top + rng.uniform(3, 15):.2f}", it["text"], count=1)
        run(mod, "founding pressure")
    return out


def qa_multi(a):
    """Questions that need several facts at once: all the levels, the existing vs the proposed bridge, the hydraulic data,
    the box's levels top to bottom with the heights between them."""
    out = []
    parts, refs = [], []
    for kind in ("rail_level", "formation_level", "top_of_box_level", "soffit_level", "bed_level", "founding_level"):
        for st in ("existing", "proposed"):
            v, ls = first_level(a, kind, st)
            if v is not None and ls and (ls[0]["status"] == st or st == "proposed"):
                parts.append(f"{st} {KIND_NAME[kind]} {lv(v)} m")
                refs.append(("level", ls[0]["label"], ls[0]["value"]))
    hv, hls = first_level(a, "hfl", status=None)
    if hv is not None:
        parts.append(f"HFL {lv(hv)} m")
        refs.append(("level", hls[0]["label"], hls[0]["value"]))
    if len(parts) >= 3:
        out.append(("qa_multi", vary(rng.choice(["List all the levels on this GAD.", "Give me all the existing and proposed levels.",
                                                  "What are the important levels of the bridge?"])),
                    "The levels on the drawing: " + "; ".join(parts) + ".", refs))
    rows = (a["tables"].get("comparative_table") or {}).get("rows", [])
    cmp_rows = [r for r in rows if r.get("kind") in ("span", "superstructure", "substructure", "foundation") and (r.get("existing") or r.get("proposed"))]
    if cmp_rows:
        ans = "From the comparative table, existing vs proposed: " + "; ".join(
            f"{r['description'].lower()}: {r.get('existing') or '(blank)'} -> {r.get('proposed') or '(blank)'}" for r in cmp_rows) + "."
        out.append(("qa_multi", vary(rng.choice(["Compare the existing and the proposed bridge.", "What changes from the existing bridge to the new one?",
                                                  "How is the proposed bridge different from the existing one?"])),
                    ans, [("table", "comparative_table", r["description"]) for r in cmp_rows]))
    hyd = [r for r in rows if r.get("kind") in ("catchment_area", "design_discharge", "velocity", "observed_hfl", "calculated_hfl", "hfl",
                                                "vertical_clearance", "free_board", "max_scour_depth", "max_scour_level", "height_of_water",
                                                "waterway_required") and r.get("proposed")]
    if len(hyd) >= 3:
        out.append(("qa_multi", vary(rng.choice(["Give me the hydraulic data of the bridge.", "Summarize the hydraulic particulars.",
                                                  "What are the discharge, velocity and flood levels?"])),
                    "The hydraulic data (proposed bridge): " + "; ".join(f"{r['description'].lower()} {r['proposed']}" for r in hyd) + ".",
                    [("table", "comparative_table", r["description"]) for r in hyd]))
    tob, tobs = first_level(a, "top_of_box_level")
    sof, sofs = first_level(a, "soffit_level")
    fdn, fdns = first_level(a, "founding_level")
    fl, fls = first_level(a, "formation_level")
    if tob is not None and sof is not None and fdn is not None:
        ans = (f"Top to bottom (proposed): " + (f"formation {lv(fl)} m, then {fl - tob:.3f} m of cushion to " if fl is not None else "")
               + f"the top of the box {lv(tob)} m; the top slab down to the soffit {lv(sof)} m ({tob - sof:.3f} m); the founding level "
               f"{lv(fdn)} m, {tob - fdn:.3f} m below the top of the box. These differences are my calculation from the printed levels.")
        out.append(("qa_multi", vary(rng.choice(["Explain the levels of the box from top to bottom.", "How deep is the box below the formation?",
                                                  "Walk me through the box's levels."])),
                    ans, [("level", x[0]["label"], x[0]["value"]) for x in (tobs, sofs, fdns, fls) if x]))
    return out


DISTRACTORS = ["a public transit company", "a stock-exchange ticker symbol", "a football club", "a chemical compound",
               "a software product", "a television channel", "an airline code"]


def qa_other(a):
    """Values printed with a label the reader has no rule for ("LTC = 120 M", "Vmax = 130 KMPH", "HC = 7.8"): never
    dropped. What the label means comes from the glossary, a web search for the term (at question time), or nowhere -
    and the answer says which. The web results are simulated here (the dataset needs no internet): one that fits, one
    about something else (abbreviations are ambiguous: "LTC" finds the London Transit Commission), or none.
    Returns [(task, question, answer, refs, annotation with the simulated meaning)]."""
    import copy
    import glossary
    out, seen = [], set()
    vals = [o for o in a.get("other_values", []) if glossary.key(o["label"]) not in seen and not seen.add(glossary.key(o["label"]))]
    for o in rng.sample(vals, min(4, len(vals))):
        k = glossary.key(o["label"])
        val = f"{o['value']:g}" + (f" {o['unit'].lower()}" if o.get("unit") else "")
        where = f"'{o['label']} = {val}' is printed on the {o['view']}"
        ref = [("other", o["label"], o["value"])]

        def ask():
            return vary(rng.choice([f"What is {o['label']} on this drawing?", f"What does {o['label']} = {val} mean?",
                                    f"What is the {o['label']} value and what does it denote?", f"Explain {o['label']}."]))
        known, src = glossary.meaning(o["label"])
        # each value twice: as the glossary knows it, and once as if it were new to the reader (a simulated lookup)
        if known:
            out.append(("qa_other", ask(), f"{where}. {o['label']} is the {known}.", ref, a))
        mod = copy.deepcopy(a)
        mod["_meanings"] = dict(a.get("_meanings") or {})
        r = rng.random()
        if known and r < 0.4:
            mod["_meanings"][k] = (f"{o['label']}: {known} (en.wikipedia.org)", "web")
            ans = (f"{where}. {o['label']} is not in my glossary; a web search for the term suggests it means the {known}. That fits this "
                   f"drawing, but it is from the web and not confirmed by the drawing - please check it (and add it to the glossary if right).")
        elif r < 0.75:
            d = rng.choice(DISTRACTORS)
            mod["_meanings"][k] = (f"{o['label']} may refer to {d} (en.wikipedia.org)", "web")
            ans = (f"{where}. Its meaning is not in my glossary, and a web search for '{o['label']}' only found {d}, which does not fit a "
                   f"railway drawing - so I can't say what it denotes here. Please ask the drawing office (and add it to the glossary).")
        else:
            mod["_meanings"][k] = (None, "unknown")
            ans = (f"{where}, but its meaning is not in my glossary and a web search found nothing, so I can't say what it denotes. "
                   f"Please ask the drawing office (and add it to the glossary).")
        out.append(("qa_other", ask(), ans, ref, mod))
    return out

def qa_components(a):
    """What each part of the box shown on the drawing is, why it is there, and where the drawing shows it."""
    out = []
    comps = FX.components(a)
    if not comps:
        return out
    out.append(("qa_component", vary(rng.choice(["What are the components of this box and what does each do?",
                                                  "Explain all the parts of the structure on this GAD.", "List the components shown on the drawing."])),
                "The drawing shows these parts: " + " ".join(f"{cap(n)}: {w}" for n, w, _, _ in comps), [("comp", n) for n, _, _, _ in comps]))
    for name, what, views, in_notes in rng.sample(comps, min(6, len(comps))):
        pat = next(p for p, n, _ in K.COMPONENTS if n == name)
        lvls = [l for l in a["levels"] if re.search(pat, l["label"], re.I)][:3]
        dims = [d for d in a["labelled_dims"] if re.search(pat, d["label"], re.I)][:3]
        ans = f"The {name} is {what} "
        ans += (f"On this drawing it is shown on {', '.join(views[:4])}." if views else "On this drawing it is mentioned in the notes.")
        if lvls or dims:
            ans += " Its values: " + "; ".join([f"{l['label']} {lv(l['value'])} m" for l in lvls] + [f"{d['label']}" for d in dims]) + "."
        out.append(("qa_component", vary(rng.choice([f"What is the {name} and why is it provided?", f"Explain the {name} on this GAD.",
                                                      f"What does the {name} do?"])),
                    ans, [("comp", name)] + [("level", l["label"], l["value"]) for l in lvls] + [("dim", d["label"], d["value"]) for d in dims]))
    return out


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


# ======================================================================== anything written on the sheet, centre lines, "why"
def loose_view(title):
    """How people name a view when they ask: 'half bottom plan', 'section A-A', 'sectional elevation' ..."""
    t = title.lower()
    alts = [t, t.split(" / ")[0], re.sub(r"\bsectional elevation at\b", "section", t), re.sub(r"\s*/\s*", " / ", t),
            re.sub(r"\bhalf top plan\b", "half section plan", t)]
    return rng.choice([x for x in alts if x])


def colour_phrase(a, colour):
    cw = FX.colour_words(a, colour)
    return f" (drawn in {cw})" if cw else ""


def qa_callouts(a):
    """Any text written on the views - what it is, where it is, and 'is X on view Y?' answered from where it really is."""
    out = []
    idx = FX.callout_index(a)
    if not idx:
        return out
    kinds = {v["title"]: v["kind"] for v in a["views"]}
    texts = [t for t in idx if not all(kinds.get(v) == "bore_log" for v, _ in idx[t]) and not re.match(r"^SBC\s*=", t, re.I)]
    if not texts:
        return out
    long_first = sorted(texts, key=lambda t: -len(t.split()))
    pick = list(dict.fromkeys(long_first[:4] + rng.sample(texts, min(5, len(texts)))))
    all_views = [v["title"] for v in a["views"]]
    for t in pick:
        where = idx[t]
        views = list(dict.fromkeys(v for v, _ in where))
        ans = f"'{t}' is written on the {', '.join(views[:4])}{colour_phrase(a, where[0][1])}."
        refs = [("call", t)]
        comp = FX.component_of(t)
        if comp:
            ans += f" It refers to the {comp[0]}: {comp[1]}"
        dim = next((d for d in a["labelled_dims"] if d["label"] == t), None)
        if dim:
            ans += f" It also gives a dimension: {fnum_mm(dim['value'])} - {dim['meaning']}."
            refs.append(("dim", dim["label"], dim["value"]))
        dis = next((d for d in a.get("dismantle", []) if d["text"] == t), None)
        if dis:
            ans += (" It marks part of the existing structure that is to be removed (its arrows point at the parts drawn as dismantling "
                    "works); ask 'why' for the reason read from the drawing.")
        v0 = rng.choice(views)
        words = t.split()
        part = " ".join(words[: max(2, min(4, len(words) - 1))]) if len(words) >= 4 and rng.random() < 0.4 else t
        q = rng.choice([f"What is {part} on {loose_view(v0)}?", f"What does '{part}' mean?", f"Where is {part.lower()} shown?",
                        f"{part} on {loose_view(v0)}", f"Explain {part.lower()}.", f"What is {part.lower()}?"])
        out.append(("qa_callout", vary(q), ans, refs))
        # asked on a view where it is not written: say where it is (v1 said "not on the drawing")
        others = [v for v in all_views if v not in views]
        if others and rng.random() < 0.5:
            vo = rng.choice(others)
            out.append(("qa_callout", vary(rng.choice([f"Is {t} shown on {loose_view(vo)}?", f"What is {t} on {loose_view(vo)}?"])),
                        f"'{t}' is not written on the {vo}; it is on the {', '.join(views[:4])}{colour_phrase(a, where[0][1])}."
                        + (f" It refers to the {comp[0]}: {comp[1]}" if comp else ""), refs))
    return out


def fnum_mm(v):
    return f"{v:g} mm"


def qa_centre_lines(a):
    """'What is the value of CL OF EXISTING UP TRACK?' - a centre line has no value; the distances from it do."""
    out = []
    links = FX.centre_line_links(a)
    cls = list(dict.fromkeys((c["name"], c["view"]) for c in a.get("centre_lines", []) if c.get("view")))
    for name, view in rng.sample(cls, min(4, len(cls))):
        lk = links.get((view, name), [])
        ans = f"The centre line of {name} (on the {view}) is a reference line - it has no value of its own."
        if lk:
            ans += " The distances from it on this view: " + "; ".join(
                f"{v:g} mm to the centre line of {o}" + (" (an unlabelled dimension whose two ends are on these centre lines)" if how == "unlabelled dimension"
                                                       else f" ({how})") for o, v, how in lk) + "."
        else:
            ans += " No distance from it to another centre line is written on this view."
        q = rng.choice([f"What is the value of CL OF {name}?", f"C/L of {name} on {loose_view(view)}?", f"What is the value of C OF {name} on {loose_view(view)}",
                        f"What is the track centre distance of {name.lower()}?", f"centre line of {name.lower()}"])
        out.append(("qa_cl", vary(q), ans, [("cl", name, view)]))
        for o, v, how in lk[:2]:
            if rng.random() < 0.6:
                src = ("an unlabelled dimension: its dimension line runs between the two centre lines" if how == "unlabelled dimension" else how)
                out.append(("qa_cl", vary(rng.choice([f"What is the distance between {name} and {o}?", f"Distance from C/L of {name.lower()} to {o.lower()}?",
                                                       f"track centres between {name.lower()} and {o.lower()} on {loose_view(view)}"])),
                            f"{v:g} mm between the centre lines of {name} and {o} on the {view} ({src}).", [("cl", name, view), ("cl", o, view)]))
    return out


def relation_words(b):
    rel = b.get("relation_to_existing") or ""
    m = re.search(r"ON\s+(D/S|U/S)\s+OF\s+(EXISTING\s+BRIDGE\s+NO\.?\s*[\w/ .-]+?)\s*(\(.*)?$", rel, re.I)
    if not m:
        return None
    side = "downstream (D/S)" if m.group(1).upper() == "D/S" else "upstream (U/S)"
    return f"on the {side} side of {m.group(2).strip().lower().replace('existing bridge', 'existing bridge')}"


def qa_why(a):
    """'Why ...?' answered from what the drawing shows - the legend, the title, the tables, the notes, the levels - and,
    where the drawing holds no reason, saying so and giving what it does show."""
    out = []
    b = a["bridge"]
    rel = relation_words(b)
    C = "comparative_table"
    # ---- why is this dismantled
    for d in a.get("dismantle", [])[:2]:
        bits = []
        if d.get("short_strokes"):
            lk = next((k for k in a["notes"].get("legend_key", []) if re.search(r"DISMANT|DIAMANT", k["label"], re.I)), None)
            bits.append("parts drawn " + (f"as the legend's '{lk['label']}' ({lk['colour']} {lk['style']} lines)" if lk else "in short dashes / strokes"))
        if d.get("angled"):
            bits.append("angled walls at the end of the existing structure (the existing wing walls)")
        if d.get("parts"):
            bits.append("labelled " + " / ".join(f"'{p}'" for p in d["parts"]))
        ans = f"The note '{d['text']}' on the {d['view']} marks part of the existing structure to be removed"
        ans += (": its arrows point at " + "; ".join(bits) + "." if bits else ".")
        if rel:
            ans += f" The new box is built {rel} (as the title says), right against it"
            ans += ", and the proposed work (red) is drawn right where the arrows point." if d.get("proposed_next_to") else "."
            ans += (" So the existing parts that stand where the new box goes, or where it joins the existing bridge, have to be taken down "
                    "to make room for the new box and to connect the two.")
        elif d.get("proposed_next_to"):
            ans += (" Proposed work (red) is drawn right where the arrows point, so the existing part is removed to make room for it.")
        else:
            ans += " The drawing does not show what replaces it there, so the reason cannot be read from it."
        ans += " The drawing does not write the reason in words; this is read from what is drawn."
        q = rng.choice(["Why does it need to be dismantled?", "Why is this to be dismantled?", f"Why dismantle the part marked on the {loose_view(d['view'])}?",
                        "What is to be dismantled and why?", "why to be dismantled"])
        out.append(("qa_why", vary(q), ans, ["bridge", ("call", d["text"]), ("legend_key",)]))
    # ---- which HFL the design uses, and why
    hfls = {}
    for kind, name in (("observed_hfl", "OHFL"), ("calculated_hfl", "CHFL")):
        r = trow(a, C, kind)
        if r and num(r.get("proposed")) is not None:
            hfls[name] = (num(r["proposed"]), tref(C, r))
    vc, sof = trow(a, C, "vertical_clearance"), first_level(a, "soffit_level")
    if len(hfls) == 2 and vc and num(vc.get("proposed")) is not None and sof[0] is not None:
        used = next((n for n, (v, _) in hfls.items() if abs(sof[0] - v - num(vc["proposed"])) < 0.006), None)
        (o_v, _), (c_v, _) = hfls["OHFL"], hfls["CHFL"]
        refs = [tref(C, vc), hfls["OHFL"][1], hfls["CHFL"][1], ("level", sof[1][0]["label"], sof[1][0]["value"])]
        if abs(o_v - c_v) < 0.0005:
            ans = f"The OHFL and the CHFL are the same here ({o_v:.3f} m), so the clearance and free board come out the same with either."
        elif used:
            other = "CHFL" if used == "OHFL" else "OHFL"
            uv, ov = hfls[used][0], hfls[other][0]
            ans = (f"The table's vertical clearance is worked out from the {used} ({uv:.3f} m), not the {other} ({ov:.3f} m): soffit "
                   f"{sof[0]:.3f} - {uv:.3f} = {sof[0] - uv:.3f} m, as the table gives ({vc['proposed']}). ")
            ans += (f"The {used} is the higher of the two (by {abs(uv - ov):.3f} m): designing for the higher flood level is the safe "
                    "choice, since it gives the smaller clearance and free board. " if uv > ov else
                    f"The {used} is the lower of the two (by {abs(uv - ov):.3f} m) - the higher flood level would be the safe choice, so "
                    "this is worth confirming. ")
            ans += "The drawing does not write the reason; this is read from its values."
        else:
            ans = (f"The table's vertical clearance ({vc['proposed']}) does not follow from either HFL with the soffit level {sof[0]:.3f} m "
                   f"(OHFL {o_v:.3f}, CHFL {c_v:.3f}), so the drawing does not show which one was used - worth checking.")
        out.append(("qa_why", vary(rng.choice(["Why was CHFL taken instead of OHFL?", "Which HFL is used for design and why?",
                                                "Why is the OHFL not used?", "why chfl and not ohfl"])), ans, refs))
    # ---- why the box size differs from the existing opening
    sp = trow(a, C, "span")
    if sp and sp.get("existing") and sp.get("proposed"):
        ex, pr = sp["existing"], sp["proposed"]
        ans = f"The comparative table gives the existing opening as {ex} and the proposed box as {pr}"
        ans += " - the same size." if re.sub(r"\s", "", ex.lower()) == re.sub(r"\s", "", pr.lower()) else "."
        req, how = trow(a, C, "waterway_required"), trow(a, C, "height_of_water")
        bx = b.get("box")
        refs = [tref(C, sp)]
        if req and num(req.get("proposed")) and bx and how and num(how.get("proposed")):
            area = bx["cells"] * bx["clear_width_m"] * num(how["proposed"])
            ans += (f" The waterway required is {req['proposed']}; the proposed box gives about {bx['cells']} x {bx['clear_width_m']:g} m x "
                    f"{num(how['proposed']):g} m (height of water) = {area:.2f} sq.m of opening below the flood level - "
                    + ("enough." if area >= num(req["proposed"]) else "LESS than required; worth checking."))
            refs += [tref(C, req), tref(C, how)]
        ans += (" The drawing does not write why this size was chosen; the reason would be in the hydraulic and design calculations, "
                "which are not on the GAD.")
        out.append(("qa_why", vary(rng.choice(["Why is the proposed box size different from the existing?", "Why this box size?",
                                                "Why did the span change?", "Why was this opening chosen?"])), ans, refs))
    # ---- why a proposed level differs from the existing one
    for kind in ("rail_level", "formation_level", "soffit_level", "bed_level"):
        e, es = first_level(a, kind, "existing")
        p, ps = first_level(a, kind, "proposed")
        if e is None or p is None or not es or es[0]["status"] != "existing" or abs(e - p) < 0.0005:
            continue
        name = KIND_NAME[kind]
        ans = (f"The existing {name} is {e:.3f} m and the proposed {name} is {p:.3f} m - the proposed is {abs(p - e):.3f} m "
               f"{'higher' if p > e else 'lower'}. The GAD does not write why. ")
        ans += ("The proposed rail and formation levels come from the new line's own longitudinal profile (its L-section); the GAD only "
                "shows them at this bridge." if kind in ("rail_level", "formation_level") else
                "The proposed box's levels follow from its own size and foundation, worked out in the design; the GAD only prints them.")
        out.append(("qa_why", vary(rng.choice([f"Why is the proposed {name} different from the existing?", f"Why did the {name} change?",
                                                f"why is new {name} not same as old"])), ans,
                    [("level", es[0]["label"], es[0]["value"]), ("level", ps[0]["label"], ps[0]["value"])]))
        break
    # ---- why this concrete grade
    notes = a["notes"].get("notes", []) + a["notes"].get("special_note", [])
    gnote = next((it for it in notes if re.search(r"M\s*-?\s*35.*SPAN\s+OR\s+HEIGHT", it["text"], re.I)), None)
    spec = next((s for s in a["notes"].get("specifications", []) if re.search(r"GRADE\s+OF\s+RCC\s+BOX", s["item"], re.I) and s.get("value")), None)
    bx = b.get("box")
    if gnote and spec and bx:
        mx = max(bx["clear_width_m"], bx["clear_height_m"])
        need = "M35" if mx > 4 else "M30"
        ok = re.sub(r"[\s-]", "", spec["value"].upper()).startswith(need)
        ans = (f"Because of note {gnote['no']}: M35 is used when the span or height of the box is more than 4 m, M30 otherwise. This box is "
               f"{bx['clear_width_m']:g} m wide and {bx['clear_height_m']:g} m high; the larger, {mx:g} m, is {'more' if mx > 4 else 'not more'} "
               f"than 4 m, so {need}. The specifications give {spec['value']}" + (" - as the note asks." if ok else " - NOT what the note asks; worth checking."))
        out.append(("qa_why", vary(rng.choice([f"Why is {spec['value']} used for the box?", "Why this concrete grade?", f"why {need.lower()}"])),
                    ans, ["bridge", ("notes", gnote["no"], gnote["text"][:30]), ("spec", spec["item"])]))
    # ---- why the foundation is at this level
    fdn, fdns = first_level(a, "founding_level")
    bed_r, sl_r = trow(a, C, "bed_level"), trow(a, C, "max_scour_level")
    if fdn is not None and bed_r and num(bed_r.get("proposed")) is not None:
        bed = num(bed_r["proposed"])
        ans = f"The drawing does not write why. What it shows: the founding level {fdn:.3f} m is {bed - fdn:.3f} m below the bed level {bed:.3f} m"
        refs = [("level", fdns[0]["label"], fdns[0]["value"]), tref(C, bed_r)]
        if sl_r and num(sl_r.get("proposed")) is not None:
            sl = num(sl_r["proposed"])
            ans += (f", and {abs(sl - fdn):.3f} m {'below' if fdn < sl else 'above'} the maximum scour level {sl:.3f} m in the table"
                    + (" - so the flood's scour does not reach under it" if fdn < sl else
                       " - the box floor is protected from scour by the drop / curtain walls, which go deeper") + ".")
            refs.append(tref(C, sl_r))
        else:
            ans += "."
        pn = next((it for it in notes if re.search(r"FOUNDING\s+PRESSURE", it["text"], re.I)), None)
        if pn:
            ans += f" Note {pn['no']} gives the founding pressure the soil there must carry ({pn['text'][:90].strip()})."
            refs.append(("notes", pn["no"], pn["text"][:30]))
        out.append(("qa_why", vary(rng.choice(["Why is the founding level here?", "Why is the foundation at this depth?", "why this fdn level"])), ans, refs))
    # ---- why a part is provided
    comps = FX.components(a)
    for name, what, views, _ in rng.sample(comps, min(2, len(comps))):
        out.append(("qa_why", vary(rng.choice([f"Why is the {name} provided?", f"Why is there a {name}?", f"why {name}"])),
                    f"The {name} is {what} " + (f"On this drawing it is shown on {', '.join(views[:3])}." if views else "It is mentioned in the notes."),
                    [("comp", name)]))
    # ---- why two values for the same level
    groups = defaultdict(list)
    for l in a["levels"]:
        if l["kind"] != "reduced_level":
            groups[(l["kind"], l["status"])].append(l)
    multi = [(k, ls) for k, ls in groups.items() if len({x["value"] for x in ls}) > 1]
    if multi:
        (kind, st), ls = rng.choice(multi)
        vals = defaultdict(list)
        for l in ls:
            vals[l["value"]].append(l)
        parts = [f"{v:.3f} m ('{xs[0]['label']}' on {FX.view_name(xs[0]['view'])}" + (f", {xs[0]['element']}" if xs[0].get("element") else "") + ")" for v, xs in vals.items()]
        els = {xs[0].get("element") for xs in vals.values()}
        name = (f"{st} " if st else "") + KIND_NAME.get(kind, kind)
        ans = f"The drawing gives {len(vals)} values for the {name}: " + "; ".join(parts) + ". "
        ans += ("They belong to different parts, so different values are expected." if len(els) == len(vals) and None not in els else
                "The drawing does not say why they differ; if they are for the same part, one may be a drafting error - worth checking.")
        out.append(("qa_why", vary(rng.choice([f"Why are there two values for the {name}?", f"Why does the {name} differ between views?"])), ans,
                    [("level", xs[0]["label"], v) for v, xs in vals.items()]))
    # ---- whys the drawing holds no answer to
    if bx:
        w2 = bx["clear_width_m"] + rng.choice([0.5, 1.0])
        req = trow(a, C, "waterway_required")
        ans = (f"The drawing does not say why {bx['clear_width_m']:g} m was chosen rather than {w2:g} m; that choice is made in the hydraulic "
               "and design calculations, which are not on the GAD.")
        if req and req.get("proposed"):
            ans += f" What the drawing does show: the waterway required is {req['proposed']} (comparative table)."
        out.append(("qa_why", vary(f"Why was a clear width of {bx['clear_width_m']:g} m chosen instead of {w2:g} m?"), ans,
                    ["bridge"] + ([tref(C, req)] if req else [])))
    if b.get("chainage"):
        out.append(("qa_why", vary(rng.choice([f"Why is the bridge at CH {b['chainage']}?", "Why was this location chosen for the bridge?"])),
                    f"The drawing does not say why. It gives the location - CH {b['chainage']}" + (f", {rel}" if rel else "")
                    + " - but the reason for a bridge at a place (the stream or road it crosses) is not written on the GAD.", ["bridge"]))
    return out


def img_spot_rows(a, page, gid):
    """A crop of the drawing around what a question is about (as ask_gad sends it), with the facts and the question: the
    model learns to look at the spot itself - arrows, circles, colours, what is next to it - not only at the facts."""
    rows = []
    tile_pt = TILE * 72 / DPI
    cands = []
    for v in a["views"]:
        for c in v.get("callouts", []):
            if isinstance(c, dict) and c.get("bbox"):
                cands.append(("call", c["text"], c["bbox"], v["title"]))
    for d in a["dims"]:
        if FX.dim_span(d):
            cands.append(("pdim", d, d["bbox"], d["view"]))
    for d in a.get("dismantle", []):
        cands.append(("dis", d, d["bbox"], d["view"]))
    if not cands:
        return rows
    pri = [c for c in cands if c[0] in ("dis", "pdim") and (c[0] == "dis" or c[1].get("circles") or any(e.get("centre_line") for e in c[1].get("ends", [])))]
    pick = pri[:4] + rng.sample(cands, min(4, len(cands)))
    seen = set()
    for kind, it, bb, view in pick:
        key = (kind, str(bb))
        if key in seen:
            continue
        seen.add(key)
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        clip = (cx - tile_pt * rng.uniform(0.35, 0.65), cy - tile_pt * rng.uniform(0.35, 0.65))
        clip = (clip[0], clip[1], clip[0] + tile_pt, clip[1] + tile_pt)
        if kind == "call":
            qa = [x for x in qa_callouts_for(a, it) if x]
        elif kind == "pdim":
            qa = [plain_answer(a, it)]
        else:
            qa = [x for x in qa_why(a) if x[2].startswith(f"The note '{it['text']}'")][:1]
        if not qa:
            continue
        task, q, ans, refs = qa[0]
        im, _ = render(page, clip, dpi=DPI)
        path = save(im, f"{gid}_spot{len(rows)}")
        q = rng.choice([f"(Crop of the {view}.) ", "This is a crop of the GAD. ", ""]) + q
        rows.append(("img_spot", path, q, ans, refs))
    return rows


def qa_callouts_for(a, text):
    idx = FX.callout_index(a)
    where = idx.get(text)
    if not where:
        return []
    views = list(dict.fromkeys(v for v, _ in where))
    ans = f"'{text}' is written on the {', '.join(views[:4])}{colour_phrase(a, where[0][1])}."
    comp = FX.component_of(text)
    if comp:
        ans += f" It refers to the {comp[0]}: {comp[1]}"
    dim = next((d for d in a["labelled_dims"] if d["label"] == text), None)
    refs = [("call", text)]
    if dim:
        ans += f" It also gives a dimension: {fnum_mm(dim['value'])} - {dim['meaning']}."
        refs.append(("dim", dim["label"], dim["value"]))
    return [("qa_callout", vary(rng.choice([f"What is {text.lower()}?", "What is written here and what does it mean?", f"Explain {text.lower()}."])), ans, refs)]


def plain_answer(a, d):
    """The answer for an unlabelled dimension with its ends found (qa_plain's wording)."""
    v, view, ends = f"{d['value']:g}", d["view"], d.get("ends") or []
    sp = FX.dim_span(d)
    if d.get("circles"):
        c = d["circles"]
        what = f"the circles of the {c['name']}" if c.get("name") else "the circles drawn there"
        ans = (f"{v} mm on the {view} is not labelled, but its dimension line " + sp + ". So " +
               (f"{v} mm is the diameter of {what}." if c["measures"] == "diameter" else f"{v} mm is the centre-to-centre spacing of {what}.")
               + " (Read from the drawing's lines.)")
    elif len(ends) == 2 and all(e.get("centre_line") for e in ends):
        ans = (f"{v} mm on the {view} is not labelled, but its dimension line {sp}, so it is the distance between those two centre lines "
               f"({ends[0]['centre_line'].lower()} to {ends[1]['centre_line'].lower()}).")
    elif sp.startswith("has both ends"):
        ans = (f"{v} mm on the {view} is not labelled. Its dimension line {sp}, so it measures something at that spot - such as a "
               f"thickness or a short width or step there; the drawing does not say exactly what.")
    elif sp.startswith("runs from"):
        ans = (f"{v} mm on the {view} is not labelled. Its dimension line {sp}, so it most likely measures the distance between those "
               f"points - read from the drawing's lines, not written on the drawing.")
    else:
        ans = (f"{v} mm on the {view} is not labelled. Its dimension line {sp}, so I can only say where it starts - the drawing does "
               f"not show what the other end is.")
    q = rng.choice([f"What does {v} on the {loose_view(view)} measure?", f"What is the {v} dimension?", f"{v} on {loose_view(view)} - what is it?",
                    f"What does the dimension {v} denote?", f"what is {v}"])
    return ("qa_plain_dim", vary(q), ans, [("pdim", view, d["value"])])


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
        ans = f"This is the {v['title']} (scale {v['scale'] or 'not written'}). It is {v['represents']}"
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
            if rng.random() < 0.8:                 # the tool knows which view a crop is from (v1 misnamed it without)
                q = f"This crop is from the {v['title']} of the GAD. " + q
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
    rows += img_spot_rows(a, page, gid)
    return rows


# ======================================================================== assemble
def make_row(gid, split, task, n, question, answer, a, refs, image=None):
    prompt = FX.prompt(a, question, must=refs) if task.startswith(("qa_", "img_view_facts", "img_spot")) else question
    content = ([{"type": "image"}] if image else []) + [{"type": "text", "text": prompt}]
    row = {"id": f"{gid}_{task}_{n:03d}", "gad": gid, "split": split, "task": task,
           "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}
    if image:
        row["image"] = image
    return row


def main():
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(ANN.glob("*.json"))]
    missing = set(TEST_GADS) - {a["gad_id"] for a in anns}
    assert not missing, f"{missing} not annotated"
    out = {"train": [], "val": [], "test": []}
    stats = Counter()
    for a in anns:
        gid = a["gad_id"]
        # the test GAD is never trained or validated on; its questions are only used to score the trained model
        split = "test" if gid in TEST_GADS else "val" if gid in VAL_GADS else "train"
        facts = FX.all_facts(a)
        qa = (qa_levels(a) + qa_values(a, facts) + qa_dims(a) + qa_plain(a) + qa_tables(a) + qa_notes(a) + qa_abbr(a) + qa_title(a)
              + qa_review(a) + qa_revisions(a) + qa_keyplan(a) + qa_bands(a) + qa_bore(a) + qa_views(a) + qa_findings(a) + qa_computed(a) + qa_reason(a) + qa_multi(a) + qa_components(a) + qa_summary(a) + qa_absent(a)
              + qa_callouts(a) + qa_centre_lines(a) + qa_why(a))
        seen_qa = set()                                  # the same question with the same answer once per GAD
        qa = [x for x in qa if (x[1].lower().rstrip("?"), x[2]) not in seen_qa and not seen_qa.add((x[1].lower().rstrip("?"), x[2]))]
        for n, (task, q, ans, refs) in enumerate(qa):
            out[split].append(make_row(gid, split, task, n, q, ans, a, refs))
            stats[task] += 1
        for n, (task, q, ans, refs, mod) in enumerate(qa_other(a)):
            out[split].append(make_row(gid, split, task, 450 + n, q, ans, mod, refs))
            stats[task] += 1
        if split != "test":                              # checks on changed copies: never on the held-out GADs
            flags = qa_reason_flags(a)
            flags = rng.sample(flags, min(4, len(flags)))   # (about 250 flagged to 370 agreeing: no bias either way)
            for n, (task, q, ans, refs, mod) in enumerate(flags):
                out[split].append(make_row(gid, split, "qa_reason_flag", 400 + n, q, ans, mod, refs))
                stats["qa_reason_flag"] += 1
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
    # one line per held-out GAD: id <tab> PDF name
    (OUT / "test_gad.txt").write_text("".join(f"{g}\t{next(a['source_pdf'] for a in anns if a['gad_id'] == g)}\n" for g in TEST_GADS),
                                      encoding="utf-8")
    print({k: len(v) for k, v in out.items()}, dict(stats))


if __name__ == "__main__":
    main()
