"""The facts the L-section model (v5) is given about one sheet for one question - as GAD v2's gad_tools/facts.py.

Shared by the dataset builder (scripts/build_dataset_v5.py) and the question tool, so the model is trained on exactly
the context it gets in use.

    facts = all_facts(ann)               # every fact of a sheet annotation (data/v5/annotations/<id>.json), one line each
    text = prompt(ann, question, must)   # the facts relevant to a question + the question
    band_at(ann, ch)                     # every band row's value at a chainage: printed, or interpolated (said so)

Values are as printed; the only arithmetic is a band value between two printed columns (linear), said so.
Names of people (issue record, signatures) are never in a fact.
"""
import math
import re
from collections import Counter

STOP = set("a an the of on in at to for and or is are was what which where how much many does do this that these those "
           "there here it its be by with from as me give tell show please can you i we our sheet drawing value values "
           "shown given written mentioned about printed".split())
SYN = {
    "fl": ["fl", "formation"], "formation": ["formation", "fl"], "rl": ["rl", "rail"], "rail": ["rail", "rl"],
    "bl": ["bed", "bl"], "bed": ["bed", "bl"], "hfl": ["hfl", "flood", "chfl", "ohfl"], "chfl": ["chfl", "hfl"], "ohfl": ["ohfl", "hfl"],
    "gl": ["ground", "gl"], "ogl": ["ground", "ogl"], "ground": ["ground", "gl"], "cut": ["cut", "fill"], "fill": ["fill", "cut"],
    "tc": ["track", "distance", "t/c"], "track": ["track", "distance"], "distance": ["distance", "track"],
    "gradient": ["gradient", "grade", "rising", "falling", "level"], "grade": ["gradient", "grade", "gp"], "slope": ["gradient"],
    "gp": ["gp", "grade"], "vpi": ["vpi", "fl"], "curve": ["curve", "degree", "radius"], "degree": ["degree", "curve"],
    "radius": ["radius", "curve"], "station": ["station", "stn"], "stn": ["station", "stn"], "km": ["km", "chainage"],
    "chainage": ["chainage", "ch"], "ch": ["chainage", "ch"], "tbm": ["tbm", "bm", "benchmark"], "bm": ["tbm", "bm"],
    "benchmark": ["tbm", "benchmark"], "bridge": ["bridge", "br"], "br": ["bridge", "br"], "span": ["span", "configuration"],
    "lc": ["lc", "crossing"], "crossing": ["crossing", "lc", "xing"], "rob": ["rob"], "rub": ["rub"], "note": ["note"],
    "notes": ["note"], "existing": ["existing", "exg", "ex"], "exg": ["existing", "exg"], "ex": ["existing", "exg"],
    "old": ["existing"], "proposed": ["proposed", "prop"], "prop": ["proposed", "prop"], "new": ["proposed"],
    "dn": ["dn"], "up": ["up"], "line": ["line"], "sheet": ["sheet"], "title": ["title"], "scale": ["scale"],
    "bearing": ["bearing"], "transition": ["transition", "st", "ts"], "datum": ["datum"], "clearance": ["clearance", "vc", "hc"],
    "cushion": ["cushion"], "slab": ["slab"], "box": ["box"], "discharge": ["discharge"], "abbreviation": ["abbreviation"],
}
MAX_FACTS = 30
MAX_CHARS = 3600
BAND_NAME = {"cut_fill": "cut (-) / fill (+)", "fl_difference": "difference between the proposed FL and the existing FL",
             "prop_rl": "proposed rail level", "prop_fl": "proposed formation level", "track_distance": "track distance (track centres)",
             "exg_up_fl": "existing line FL", "exg_dn_fl": "existing line FL", "exg_line_fl": "existing line FL",
             "ground_level": "ground level", "chainage": "chainage"}


def km(v):
    """1030302.828 -> '1030+302.828'."""
    if v is None:
        return "?"
    return f"{int(v // 1000)}+{v % 1000:07.3f}"


def g(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def band_rows(ann):
    """{row key: heading as printed} of the sheet's data band (v4 keys; the printed headings from the v5 reading)."""
    b = ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
    keys = b.get("rows") or []
    heads = {}
    for bb in (ann.get("v5") or {}).get("bands", []):
        for r in bb.get("rows", []):
            heads.setdefault(round(r["y"]), r["heading"])
    out = {}
    for k in keys:
        out[k] = BAND_NAME.get(k, k.replace("_", " "))
    # the printed heading of each row, matched by the row's first value
    for bb in (ann.get("v5") or {}).get("bands", []):
        for r in bb.get("rows", []):
            if not r["values"]:
                continue
            ch0, v0 = r["values"][0]
            for c in b.get("columns", []):
                if abs(c["chainage"] - ch0) < 0.01:
                    for k, v in (c.get("values") or {}).items():
                        try:
                            if v is not None and abs(float(v) - float(v0)) < 1e-9 and r["heading"] and k not in ("chainage",):
                                out[k] = r["heading"]
                        except ValueError:
                            pass
    return out


def band_at(ann, ch):
    """[(row key, row name, value, how)] at chainage ch: the printed column ('printed at CH ...') or the two columns either
    side interpolated ('between CH a (x) and CH b (y)'); [] outside the band."""
    b = ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
    cols = sorted(b.get("columns") or [], key=lambda c: c["chainage"])
    if not cols or not cols[0]["chainage"] - 1e-6 <= ch <= cols[-1]["chainage"] + 1e-6:
        return []
    names = band_rows(ann)
    exact = next((c for c in cols if abs(c["chainage"] - ch) < 1e-6), None)
    out = []
    for k in b.get("rows") or []:
        if k == "chainage":
            continue
        if exact is not None:
            v = (exact.get("values") or {}).get(k)
            if v is not None:
                out.append((k, names.get(k, k), round(float(v), 3), f"printed at CH {km(ch)}"))
            continue
        left = [c for c in cols if c["chainage"] < ch and (c.get("values") or {}).get(k) is not None]
        right = [c for c in cols if c["chainage"] > ch and (c.get("values") or {}).get(k) is not None]
        if not left or not right:
            continue
        a, c = left[-1], right[0]
        va, vc = float(a["values"][k]), float(c["values"][k])
        v = va + (vc - va) * (ch - a["chainage"]) / (c["chainage"] - a["chainage"])
        out.append((k, names.get(k, k), round(v, 3), f"not printed at CH {km(ch)}: between CH {km(a['chainage'])} ({g(va)}) and "
                                                      f"CH {km(c['chainage'])} ({g(vc)}), interpolated"))
    return out


def all_facts(ann):
    """Every fact of a sheet: [{"group", "text", "ref"}]."""
    F = []

    def add(group, text, ref=None):
        F.append({"group": group, "text": re.sub(r"\s+", " ", text).strip(), "ref": ref})
    info = ann.get("sheet_info") or {}
    v5 = ann.get("v5") or {}
    band = ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
    lines = band.get("lines") or {}
    add("sheet", f"Sheet: {info.get('title') or '?'}; drawing no. {info.get('drawing_no') or '?'}; sheet {info.get('sheet_no') or '?'}"
        f" (previous {info.get('previous_sheet') or '-'}, next {info.get('next_sheet') or '-'}); chainage {info.get('chainage_from') or '?'} to "
        f"{info.get('chainage_to') or '?'}; scale {info.get('scale') or '?'}; date {info.get('date') or '?'}", "sheet")
    if lines or v5.get("main_track"):
        add("sheet", f"Lines on this sheet: proposed {lines.get('proposed') or v5.get('main_track') or '?'}, existing {lines.get('existing') or '?'}", "lines")
    extra = [f"{n} {info[k]}" for k, n in (("gauge", "gauge"), ("standard_of_construction", "standard of construction"),
                                         ("year_of_survey", "year of survey"), ("client", "client")) if info.get(k)]
    if extra:
        add("sheet", "Sheet details: " + "; ".join(extra), "sheet2")
    if info.get("project"):
        add("sheet", "Project: " + info["project"], "project")
    for r in info.get("issue_record") or []:                      # (revision and date only: no names)
        add("sheet", f"Issue record: no. {r.get('no')}, revision {r.get('rev')}, dated {r.get('date')}", ("issue", r.get("no")))
    for s in info.get("stations") or []:
        add("station", f"Station named in the title: {s.get('name')} ({s.get('side')} end)", ("station", s.get("name")))
    for s in ann.get("stations") or []:
        add("station", f"Station: {s.get('name')} ({s.get('direction')} end of the sheet)" + (f"; printed with it: {', '.join(s['details'])}" if s.get("details") else ""),
            ("station", s.get("name")))
    # bridges (v5 reading: callout, proposal, level block) and the DETAILS OF BRIDGES table row
    table = {re.sub(r"\s", "", r["bridge"]).upper(): r for r in v5.get("bridge_table", [])}
    for b in v5.get("bridges", []):
        f = b["fields"]
        t = f"Bridge {f.get('br_no') or b['num']}: callout '{' / '.join(c['text'] for c in b['callouts'])}'"
        parts = []
        if f.get("chainage"):
            parts.append(f"chainage {f['chainage']}")
        if f.get("exg_structure") or f.get("exg_configuration"):
            parts.append(f"existing {f.get('exg_structure') or '?'} {f.get('exg_configuration') or ''}".strip())
        if f.get("description"):
            parts.append(f"crossing {f['description']}")
        if f.get("prop_structure") or f.get("prop_configuration"):
            parts.append(f"proposed {f.get('prop_configuration') or ''} {f.get('prop_structure') or ''}".strip()
                         + (f" ({f['prop_type']})" if f.get("prop_type") else "") + (f" at chainage {f['_prop_chainage']}" if f.get("_prop_chainage") else ""))
        add("bridge", t + ("; " + "; ".join(parts) if parts else ""), ("bridge", b["num"]))
        if b.get("level_block"):
            add("bridge", f"Bridge {b['num']} level block: " + "; ".join(f"{v['label']} = {v['value']}" for v in b["level_block"]["values"]),
                ("levels", b["num"]))
        row = table.get(b["num"]) or table.get(re.sub(r"(UP|DN)$", "", b["num"]))
        if row:
            add("bridge", f"Bridge {b['num']} in the DETAILS OF BRIDGES table: " + "; ".join(f"{TABLE_NAME.get(k, k)} {v}" for k, v in row["cells"].items()
                                                                                         if k not in ("sno", "br") and v), ("table", b["num"]))
    for br, row in table.items():
        if not any(b["num"] == br or re.sub(r"(UP|DN)$", "", b["num"]) == br for b in v5.get("bridges", [])):
            add("bridge", f"{row['cells'].get('br', br)} in the DETAILS OF BRIDGES table: " + "; ".join(
                f"{TABLE_NAME.get(k, k)} {v}" for k, v in row["cells"].items() if k not in ("sno", "br") and v), ("table", br))
    for c in v5.get("crossings", []):
        add("bridge", f"{c['label']}: '{c['text']}'" + (f"; levels: {c['level_block']['text']}" if c.get("level_block") else ""), ("xing", c["label"]))
    # curves
    for c in v5.get("curves", []):
        pts = "; ".join(f"{p} {km(ch)}" for p, ch in c["points"].items())
        add("curve", f"Curve {c['num']}" + (f" ({c['track']})" if c.get("track") else "") + (f", {c['hand']} hand" if c.get("hand") else "")
            + f": runs from {km(min(c['points'].values()))} to {km(max(c['points'].values()))}; points {pts}"
            + (f"; degree {c['degree']}" if c.get("degree") else "") + (f"; details: {c['details'][0]}" if c.get("details") else ""),
            ("curve", c["num"], c.get("track")))
    for c in ann.get("curves") or []:
        p = c.get("params") or {}
        if p:
            add("curve", f"Curve {c.get('curve_no')} ({c.get('line')}, {c.get('hand')}) parameters: " + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in p.items()),
                ("curve_params", c.get("curve_no"), c.get("line")))
    for c in ann.get("existing_curves") or []:
        add("curve", f"Existing line curve {c.get('curve_no')}: " + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in c.items()
                                                                           if k not in ("bbox", "curve_no", "line") and v), ("ex_curve", c.get("curve_no")))
    for t in ann.get("transition_points") or []:
        add("curve", f"Curve point {t.get('printed_as')} ({t.get('meaning')}) at chainage {km(t.get('chainage_m'))} (printed '{t.get('text')}')",
            ("tp", t.get("printed_as"), t.get("chainage_m")))
    # gradients, grade points, VPIs
    for s in ann.get("gradient_segments") or []:
        add("gradient", f"Gradient of the {s.get('line')}: '{s.get('gradient_label')}' = {s.get('gradient')} ({s.get('percent')}), length {g(s.get('length_m'))} m, "
            f"around chainage {km(s.get('mid_chainage_m'))}", ("grad", s.get("mid_chainage_m")))
    for p in ann.get("plan_grade_points") or []:
        add("gradient", f"Grade point of the {p.get('line')} at chainage {p.get('chainage')}: FL {g(p.get('fl'))}; gradient before {p.get('gradient_before')}, "
            f"after {p.get('gradient_after')}", ("gp", p.get("chainage_m")))
    for p in ann.get("grade_points") or []:
        add("gradient", f"Grade point (L-section) '{p.get('text')}': chainage {km(p.get('chainage_m'))}, FL {g(p.get('fl'))}", ("gp", p.get("chainage_m")))
    # band
    names = band_rows(ann)
    cols = band.get("columns") or []
    if cols:
        add("band", f"Data band: {len(cols)} columns from chainage {km(cols[0]['chainage'])} to {km(cols[-1]['chainage'])}, every "
            f"{g(cols[1]['chainage'] - cols[0]['chainage']) if len(cols) > 1 else '?'} m; rows: " + "; ".join(n for k, n in names.items() if k != "chainage"), "band")
    # marks, bearings, TBMs, scale
    for k in ann.get("km_posts") or []:
        add("marks", f"KM post '{k.get('text')}' at chainage {km(k.get('chainage_m'))}", ("km", k.get("chainage_m")))
    for k in ann.get("km_marks") or []:
        add("marks", f"KM mark '{k.get('text')}': proposed KM {k.get('proposed_km')}, existing KM {k.get('existing_km')}", ("kmm", k.get("chainage_m")))
    for b in ann.get("bearings") or []:
        add("marks", f"Bearing '{b.get('text')}' near chainage {km(b.get('chainage_m'))}", ("bearing", b.get("chainage_m")))
    for t in ann.get("tbm_benchmarks") or []:
        add("tbm", f"TBM {t.get('tbm_id')}: chainage {km(t.get('chainage_m'))}, easting {g(t.get('easting'))}, northing {g(t.get('northing'))}, "
            f"MSL {g(t.get('msl_m'))} m; {t.get('description') or ''}", ("tbm", t.get("tbm_id")))
    ls = ann.get("level_scale") or {}
    if ls:
        add("sheet", f"L-section level scale: datum {ls.get('datum')}, V:H = {ls.get('vertical_to_horizontal')}, ticks {g(ls.get('lowest_tick'))} to "
            f"{g(ls.get('highest_tick'))} every {g(ls.get('tick_step'))}", "scale")
    # notes, abbreviations, legend
    for n in ann.get("notes") or []:
        add("note", f"Note {n.get('no')}: {n.get('text')}", ("note", n.get("no")))
    for k, v in (ann.get("abbreviations") or {}).items():
        add("abbr", f"Abbreviation (the sheet's list): {k} = {v}", ("abbr", k))
    if ann.get("legend"):
        add("note", "Legend: " + "; ".join(ann["legend"]), "legend")
    for f in ann.get("findings") or []:
        add("finding", "Found on the sheet: " + f, ("finding", f[:30]))
    # anything else printed: every text item not already in a fact above (the "read anything" base)
    seen = " ".join(x["text"] for x in F).upper()
    for t in ann.get("all_text") or []:
        s = t["text"].strip()
        if len(re.findall(r"[A-Za-z]", s)) >= 3 and s.upper() not in seen:
            add("text", f"Printed on the sheet: '{s}'", ("text", s))
    return F


TABLE_NAME = {"crossing": "type of crossing", "chainage": "chainage", "type_ex": "type of bridge (existing line)",
              "type_pr": "type of bridge (proposed line)", "slab": "top slab thickness", "fl_ex": "FL existing line",
              "fl_pr": "FL proposed line", "chfl": "CHFL", "ohfl": "OHFL", "bed": "bed level / road level",
              "fb_ex": "free board existing line", "fb_pr": "free board proposed line", "clearance_ex": "clearance existing line",
              "clearance_pr": "clearance proposed line", "cushion_ex": "earth cushion existing line", "cushion_pr": "earth cushion proposed line",
              "hc": "HC (horizontal clearance)", "vc": "VC (vertical clearance)",
              "discharge": "discharge", "remarks": "remarks"}


def tokens(text):
    t = text.lower().replace("'", " ")
    t = re.sub(r"\b((?:[a-z]\.){1,3}[a-z])\b\.?", lambda m: m.group(1).replace(".", ""), t)
    return [w for w in re.findall(r"[a-z]+(?:/[a-z]+)?|\d+(?:\.\d+)?", t) if w not in STOP]


def expand(q):
    out = []
    for w in tokens(q):
        out.append(w)
        out += SYN.get(w, [])
    return out


def numbers(text):
    """Numbers in a text, chainages as metres too ('1030+302.828' -> 1030302.828)."""
    out = set()
    for m in re.finditer(r"(\d+)\s*\+\s*(\d+(?:\.\d+)?)", text):
        out.add(round(int(m.group(1)) * 1000 + float(m.group(2)), 3))
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        out.add(round(float(m.group()), 3))
    return out


def score_facts(facts, question):
    docs = [Counter(tokens(f["text"])) for f in facts]
    df = Counter(w for d in docs for w in d)
    N = len(docs)
    q = expand(question)
    qn = numbers(question)
    scores = []
    for f, d in zip(facts, docs):
        s = sum(math.log(1 + N / df[w]) for w in set(q) if w in d)
        if qn and qn & numbers(f["text"]):
            s += 6
        scores.append(s - (0.5 if f["group"] == "text" else 0))
    return scores


CORE = ("sheet",)


def select(ann, question, must=(), facts=None):
    facts = facts or all_facts(ann)
    sc = score_facts(facts, question)
    core = [i for i, f in enumerate(facts) if f["ref"] in ("sheet", "lines")]
    forced = [i for i, f in enumerate(facts) if f["ref"] in must]
    ranked = [i for i in sorted(range(len(facts)), key=lambda i: -sc[i]) if sc[i] > 0 and i not in core and i not in forced]
    chosen, chars = [], 0
    for i in core + forced + ranked:
        if i in chosen:
            continue
        if (len(chosen) >= MAX_FACTS or chars + len(facts[i]["text"]) > MAX_CHARS) and i not in forced:
            break
        chosen.append(i)
        chars += len(facts[i]["text"])
    order = {g_: k for k, g_ in enumerate(["sheet", "station", "bridge", "curve", "gradient", "band", "marks", "tbm", "note", "abbr", "finding", "text"])}
    chosen.sort(key=lambda i: (order.get(facts[i]["group"], 99), i))
    return [facts[i] for i in chosen]


def prompt(ann, question, must=(), extra=(), facts=None):
    """The prompt: the selected facts (+ extra lines, e.g. band values at the chainage asked) and the question."""
    fs = [f["text"] for f in select(ann, question, must, facts)] + list(extra)
    return "L-section sheet facts (read from the drawing):\n" + "\n".join(f"- {t}" for t in fs) + f"\n\nQuestion: {question}"
