"""The facts the model is given about one GAD for one question.

Shared by the dataset builder (gad_tools/build_dataset_gad.py) and the question CLI (gad_kit/ask_gad.py), so the model
is trained on exactly the context it gets in use.

    facts = all_facts(annotation)          # every fact on the drawing, one line each
    text = context(annotation, question)   # the relevant ones for a question, as the prompt's "GAD facts" block
"""
import math
import re
from collections import Counter, defaultdict

STOP = set("a an the of on in at to for and or is are was what which where how much many does do this that these those "
           "there here it its be by with from as me give tell show please can you i we our gad drawing value values "
           "shown given written mentioned about".split())
SYN = {
    "soffit": ["soffit", "underside", "bottom of top slab"],
    "hfl": ["hfl", "flood", "ohfl", "chfl", "high flood level"],
    "fl": ["formation", "fl"],
    "frl": ["formation", "fl", "frl"],
    "pfl": ["formation", "fl", "proposed", "prop"],
    "fnd": ["founding", "foundation", "fdn", "fnd"],
    "trl": ["rail", "trl"],
    "rl": ["rl", "rail", "reduced"],
    "tob": ["top", "box", "tob"],
    "sof": ["soffit"],
    "ohfl": ["ohfl", "observed", "hfl", "flood"],
    "chfl": ["chfl", "calculated", "hfl", "flood"],
    "exist": ["existing", "exist", "ex", "exg"],
    "ex": ["existing", "exist", "ex", "exg"],
    "lvl": ["lvl", "level"],
    "formation": ["formation", "fl"],
    "founding": ["founding", "foundation", "fdn", "fnd", "found"],
    "foundation": ["founding", "foundation", "fdn"],
    "rail": ["rail", "rl"],
    "bed": ["bed", "stream", "river"],
    "clearance": ["clearance", "vertical"],
    "freeboard": ["free", "board"],
    "discharge": ["discharge", "cumecs"],
    "size": ["span", "box", "opening", "size"],
    "span": ["span", "box", "opening"],
    "opening": ["span", "box", "opening"],
    "barrel": ["barrel", "length"],
    "track": ["track", "t/c", "centre", "center"],
    "centre": ["t/c", "centre", "center", "track"],
    "center": ["t/c", "centre", "center", "track"],
    "concrete": ["grade", "concrete", "m-30", "m-25", "m-20", "m-15", "m-35", "m30", "m25", "m35"],
    "steel": ["steel", "fe", "bars", "reinforcement"],
    "cover": ["cover", "clear"],
    "soil": ["soil", "sbc", "bore", "strata", "clay", "sand", "rock"],
    "sbc": ["sbc", "bearing", "soil", "bore"],
    "bearing": ["sbc", "bearing", "pressure"],
    "earthquake": ["seismic", "zone"],
    "seismic": ["seismic", "zone"],
    "load": ["loading", "load", "axle"],
    "loading": ["loading", "load", "axle"],
    "revision": ["rev", "revision", "history", "submission"],
    "revisions": ["rev", "revision", "history"],
    "station": ["station", "end", "key"],
    "stations": ["station", "end", "key"],
    "key": ["key", "plan"],
    "location": ["key", "plan", "station"],
    "drawing": ["dwg", "drawing"],
    "number": ["no", "dwg"],
    "chainage": ["chainage", "ch", "km"],
    "km": ["chainage", "km"],
    "date": ["date"],
    "gradient": ["grade", "rise", "fall", "slope"],
    "grade": ["grade", "rise", "fall"],
    "curve": ["curve", "alignment", "degree"],
    "slope": ["slope", "gradient"],
    "thickness": ["thk", "thick", "thickness"],
    "thick": ["thk", "thick", "thickness"],
    "weep": ["weep", "holes"],
    "abbreviation": ["abbreviation", "stands", "means"],
    "existing": ["existing", "exist", "ex"],
    # the shorthand engineers type (build_dataset_gad.SHORTHAND)
    "exg": ["existing", "exist", "ex", "exg"],
    "old": ["existing", "exist", "ex"],
    "new": ["proposed", "prop"],
    "prop": ["proposed", "prop"],
    "tob": ["top", "box"],
    "fdn": ["founding", "foundation", "fdn"],
    "dwg": ["dwg", "drawing"],
    "bl": ["bed", "bl"],
    "vc": ["vertical", "clearance"],
    "hydraulic": ["discharge", "velocity", "hfl", "comparative", "table", "hydraulic"],
    "comparison": ["comparative", "table"],
    "t/c": ["t/c", "track", "centre"],
    "proposed": ["proposed", "prop"],
    "scour": ["scour"],
    "velocity": ["velocity", "speed"],
    "catchment": ["catchment"],
    "waterway": ["waterway", "water", "way"],
    "ballast": ["ballast", "cushion"],
    "sleeper": ["sleeper", "psc"],
    "level": ["lvl", "level"],
    "invert": ["invert", "drain"],
    "drain": ["drain", "invert"],
    "road": ["road", "rd"],
    "offset": ["offset", "chainage"],
    "profile": ["ground", "profile"],
    "lvl": ["lvl", "level"],
    "views": ["view"],
    "inconsistency": ["disagreement", "differs"],
    "mismatch": ["disagreement", "differs"],
    "wrong": ["disagreement", "differs"],
    "conflict": ["disagreement", "differs"],
}
CORE_GROUPS = ("bridge", "views")
MAX_FACTS = 26
MAX_CHARS = 3400


def fnum(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) and v != int(v) else (str(int(v)) if isinstance(v, float) else str(v))


def lvl(v):
    return f"{v:.3f}"


def view_name(title):
    return title or "outside the views"


def components(a):
    """[(name, explanation, [views it appears on], in_notes)] for the parts of the box this drawing shows or mentions."""
    import gad_kinds as K
    notes = " ".join(it["text"] for k in ("notes", "special_note", "add_note", "fill_note") for it in (a.get("notes") or {}).get(k, []))
    out = []
    for pat, name, what in K.COMPONENTS:
        views = [v["title"] for v in a.get("views", [])
                 if re.search(pat, " ".join(v.get("texts", [])) + " " + v["title"], re.I)]
        views += sorted({l["view"] for l in a.get("levels", []) if l["view"] and re.search(pat, l["label"], re.I)} - set(views))
        in_notes = bool(re.search(pat, notes, re.I))
        if views or in_notes:
            out.append((name, what, views, in_notes))
    return out


SOURCE_WORDS = {"glossary": "meaning from the drawing office's glossary",
                "built-in": "standard meaning of this term",
                "web": "meaning found by a web search for the term - NOT confirmed by the drawing, may not fit",
                "unknown": "meaning not known: not in the glossary and no web result"}


def term_meaning(a, label):
    """(meaning, source) for a label: an override carried by the annotation (ask_gad's web lookups, the training's
    simulated lookups) first, then the glossary (gad_tools/glossary.py, no web search here)."""
    import glossary
    k = glossary.key(label)
    o = (a.get("_meanings") or {}).get(k)
    if o:
        return o[0], o[1]
    return glossary.meaning(label, web=False)


def end_words(e):
    if e.get("centre_line"):
        return f"the centre line of {e['centre_line']}"
    oc = e.get("on_circle")
    if oc:
        return f"the {oc['at']} of a circle" + (f" of the {oc['name']}" if oc.get("name") else "")
    if e.get("near"):
        return "near " + " / ".join(f"'{t}'" for t in e["near"])
    return None


def dim_span(d):
    """'runs between the centre line of UP TRACK and the centre line of DN TRACK' / 'runs from near 'HFL 395.593' to near ...'
    / None - from the drawn dimension line's ends (annotate_gad.dimension_ends)."""
    ends = d.get("ends") or []
    c = d.get("circles")
    if c:
        what = f"the circles of the {c['name']}" if c.get("name") else "the row of circles drawn there"
        return (f"spans one of {what} edge to edge - the circle's diameter" if c["measures"] == "diameter" else
                f"runs between the same point of two neighbouring circles - the centre-to-centre spacing of {what}")
    w = [end_words(e) for e in ends]
    if len(w) == 2 and all(e.get("centre_line") for e in ends):
        return f"runs between {w[0]} and {w[1]}"
    if len(w) == 2 and w[0] and w[1] and (ends[0].get("near") or [None])[0] == (ends[1].get("near") or [None])[0]:
        return f"has both ends {w[0]}"                  # a short dimension at one place: a thickness, a step there
    if len(w) == 2 and w[0] and w[1]:
        return f"runs from {w[0]} to {w[1]}"
    one = next((x for x in w if x), None)
    return f"has one end {one} (nothing named at its other end)" if one else None


def colour_words(a, colour):
    """'red, like the legend's PROPOSED STRUCTURE' from the sheet's own legend; else just the colour."""
    if colour != "red":                    # (black / blue text is ordinary annotation: its colour says nothing)
        return None
    for k in (a.get("notes") or {}).get("legend_key", []):
        if k["colour"] == colour and k["style"] == "line":
            return f"{colour}, the colour of the legend's '{k['label']}'"
    return None


def centre_line_links(a):
    """{(view, name): [(other name, mm, how)]} - distances between centre lines on a view: unlabelled dimensions whose
    two ends are on them, and labelled track centres (T/C) between them; plus sums along a row of three or more
    (computed, said so)."""
    links = defaultdict(list)
    for d in a.get("dims", []):
        e = d.get("ends") or []
        if len(e) == 2 and e[0].get("centre_line") and e[1].get("centre_line") and e[0]["centre_line"] != e[1]["centre_line"]:
            n0, n1 = e[0]["centre_line"], e[1]["centre_line"]
            links[(d["view"], n0)].append((n1, d["value"], "unlabelled dimension"))
            links[(d["view"], n1)].append((n0, d["value"], "unlabelled dimension"))
    for d in a.get("labelled_dims", []):
        if d.get("kind") == "track_centres" and d.get("between") and len(d["between"]) == 2:
            n0, n1 = d["between"]
            links[(d["view"], n0)].append((n1, d["value"], f"'{d['label']}'"))
            links[(d["view"], n1)].append((n0, d["value"], f"'{d['label']}'"))
    # A-B and B-C with B between them (x order): A-C = sum, computed
    xs = {(c["view"], c["name"]): c["x"] for c in a.get("centre_lines", []) if c.get("x") is not None}
    for (view, b_), lst in list(links.items()):
        for i, (n0, v0, _) in enumerate(lst):
            for n1, v1, _ in lst[i + 1:]:
                x0, xb, x1 = xs.get((view, n0)), xs.get((view, b_)), xs.get((view, n1))
                if None in (x0, xb, x1) or not (min(x0, x1) < xb < max(x0, x1)) or n0 == n1:
                    continue
                if any(o == n1 for o, _, _ in links[(view, n0)]):
                    continue
                how = f"computed: {fnum(v0)} + {fnum(v1)} through the centre line of {b_}"
                links[(view, n0)].append((n1, v0 + v1, how))
                links[(view, n1)].append((n0, v0 + v1, how))
    return links


def callout_index(a):
    """{text: [(view, colour)]} - every text written on the views (labels over several lines joined)."""
    idx = defaultdict(list)
    for v in a.get("views", []):
        for c in v.get("callouts", []):
            if isinstance(c, dict):
                idx[c["text"]].append((v["title"], c.get("colour")))
    return idx


def component_of(text):
    import gad_kinds as K
    for pat, name, what in K.COMPONENTS:
        if re.search(pat, text, re.I):
            return name, what
    return None


def all_facts(a):
    """Every fact of a GAD annotation: [{"group", "text", "ref"}] - one short line each."""
    F = []

    def add(group, text, ref=None):
        F.append({"group": group, "text": re.sub(r"\s+", " ", text).strip(), "ref": ref})
    b, tb = a.get("bridge") or {}, a.get("title_block") or {}
    box = b.get("box")
    parts = [b.get("category", "BRIDGE"), f"NO. {b['bridge_no']}" if b.get("bridge_no") else None,
             b.get("description"), f"at CH {b['chainage']}" if b.get("chainage") else None]
    t = "Bridge (from the title): " + " ".join(p for p in parts if p)
    if box:
        t += f" -> {box['cells']} cell(s), clear width {fnum(box['clear_width_m'])} m, clear height {fnum(box['clear_height_m'])} m"
    elif b.get("slab"):
        sl = b["slab"]
        t += f" -> a slab bridge, not a box: {sl['spans']} span(s) of clear span {fnum(sl['clear_span_m'])} m, {sl['type']}"
    if b.get("relation_to_existing"):
        t += f"; {b['relation_to_existing']}"
    add("bridge", t, "bridge")
    if tb:
        add("title_block", "Title: " + tb.get("title", ""), "title")
        keys = [("dwg_no", "drawing no."), ("hq_dwg_no", "HQ's drawing no."), ("consultants_dwg_no", "consultant's drawing no."),
                ("rev_no", "revision"), ("date", "date"), ("size", "sheet size"), ("scale", "scale")]
        add("title_block", "Drawing: " + "; ".join(f"{n} {tb[k]}" for k, n in keys if tb.get(k)), "drawing")
        keys = [("railway", "railway"), ("division", "division"), ("section", "section"), ("km_chainage", "KM/chainage"),
                ("pink_book_item", "pink book item"), ("project", "project")]
        add("title_block", "Location/project: " + "; ".join(f"{n} {tb[k]}" for k, n in keys if tb.get(k)), "project")
        if tb.get("name_of_work"):
            add("title_block", "Name of work: " + tb["name_of_work"], "work")
    if a.get("views"):
        add("views", f"Views on the drawing ({len(a['views'])}): " + "; ".join(f"{v['title']} ({v['scale'] or 'no scale written'})" for v in a["views"]), "views")
    for v in a.get("views", []):
        add("view", f"View '{v['title']}' (scale {v['scale'] or 'not written'}): {v['represents']}", ("view", v["title"]))
    if tb.get("revisions"):
        add("title_block", "Revision history: " + "; ".join(f"{r['rev_no']} dated {r['date'] or '(no date)'}: {r['description'] or '(no description)'}"
                                                         for r in tb["revisions"]), "revisions")
    # levels, grouped: the same label and value on several views is one fact
    seen = defaultdict(list)
    for l in a.get("levels", []):
        seen[(l["label"], l["value"])].append(view_name(l["view"]))
    for (label, value), views in seen.items():
        l = next(x for x in a["levels"] if x["label"] == label and x["value"] == value)
        st = f"{l['status']} " if l.get("status") else ""
        add("level", f"Level '{label}' = {lvl(value)} m ({st}{l['kind'].replace('_', ' ')}: {l['meaning']}); on: {', '.join(sorted(set(views)))}",
            ("level", label, value))
    for d in a.get("labelled_dims", []):
        add("dim", f"Dimension '{d['label']}' = {fnum(d['value'])} mm ({d['meaning']}); on: {view_name(d['view'])}", ("dim", d["label"], d["value"]))
    for s in a.get("slopes", []):
        add("dim", f"Slope '{s['label']}' ({s['meaning']}); on: {view_name(s['view'])}", ("slope", s["label"]))
    links = centre_line_links(a)
    for c in dict.fromkeys((c["name"], c["view"]) for c in a.get("centre_lines", [])):
        t = f"Centre line of {c[0]} (on {view_name(c[1])}): a reference line - it has no value of its own"
        lk = links.get((c[1], c[0]))
        if lk:
            t += "; distances from it: " + "; ".join(f"{fnum(v)} mm to the centre line of {o} ({how})" for o, v, how in lk)
        else:
            t += "; no distance from it to another centre line is written on this view"
        add("dim", t, ("cl", c[0], c[1]))
    plain = defaultdict(list)
    for d in a.get("dims", []):
        plain[(d["view"], d["direction"])].append(fnum(d["value"]))
    for (view, direction), vals in plain.items():
        add("plain_dims", f"Unlabelled dimension figures (mm, {direction}) on {view_name(view)}: {', '.join(vals)} - the drawing does not "
            "write what each of these measures", ("plain", view, direction))
    for d in a.get("dims", []):
        sp = dim_span(d)
        if sp:
            add("plain_dims", f"Unlabelled dimension {fnum(d['value'])} mm on {view_name(d['view'])}: its dimension line {sp} (found from the "
                "drawing's lines; what it measures is not written on the drawing)", ("pdim", d["view"], d["value"]))
    for name, t in (a.get("tables") or {}).items():
        if not isinstance(t, dict):
            continue
        tname = name.replace("_", " ")
        for r in t["rows"]:
            vals = "; ".join(f"{c} {r[c]}" for c in ("existing", "proposed", "value") if r.get(c))
            add("table", f"{tname.capitalize()}: {r['description']} - {vals or '(blank)'}" + (f" ({r['meaning']})" if r.get("meaning") else ""),
                ("table", name, r["description"]))
    dts = (a.get("tables") or {}).get("depth_of_track_structure")
    if isinstance(dts, list) and dts:
        add("table", "Depth of track structure: " + "; ".join(f"{x['item']} = {x['value']}" for x in dts), ("table", "depth"))
    for bl in a.get("bore_logs", []):
        t = f"Bore log ({bl['view']})"
        if bl.get("sbc"):
            t += ": SBC " + "; ".join(f"{fnum(s['sbc_t_per_m2'])} t/m² at {fnum(s['depth_m'])} m depth" if s.get("depth_m") is not None
                                     else f"{fnum(s['sbc_t_per_m2'])} t/m²" for s in bl["sbc"])
        if bl.get("layers"):
            t += "; soil layers (top to bottom): " + " / ".join(x["soil"] for x in bl["layers"])
        if bl.get("rl_marks"):
            t += "; RL marks: " + ", ".join(lvl(x) for x in bl["rl_marks"])
        if bl.get("deepest_m") is not None:
            t += f"; the log goes down to {fnum(bl['deepest_m'])} m depth (the deepest SBC printed)"
        if bl.get("rl_top") is not None and bl.get("rl_bottom") is not None and bl["rl_top"] > bl["rl_bottom"]:
            t += f"; it runs from RL {lvl(bl['rl_top'])} to RL {lvl(bl['rl_bottom'])}, {fnum(round(bl['rl_top'] - bl['rl_bottom'], 3))} m"
        add("bore_log", t, ("bore", bl["view"]))
    n = a.get("notes") or {}
    for key, label in (("notes", "Note"), ("special_note", "Special note"), ("add_note", "Additional note"), ("fill_note", "Fill note"),
                       ("design_criteria", "Design criteria"), ("reference_drawings", "Reference drawing"),
                       ("review_comments", "Reviewer's comment marked on the drawing (blue markup, not part of the design)")):
        for it in n.get(key, []):
            add("note", f"{label} {it['no'] or ''}: {it['text']}", (key, it["no"], it["text"][:30]))
    for it in n.get("specifications", []):
        add("note", f"Specification {it['no'] or ''}: {it['item']}" + (f" = {it['value']}" if it.get("value") else ""), ("spec", it["item"]))
    for k, v in (n.get("abbreviations") or {}).items():
        add("abbreviation", f"Abbreviation (from the drawing's list): {k} = {v}", ("abbr", k))
    if n.get("legends"):
        add("note", "Legends: " + "; ".join(n["legends"]), ("legends",))
    if n.get("seismic_zone"):
        add("note", f"Seismic zone: {n['seismic_zone']}", ("seismic",))
    if n.get("standard_of_loading"):
        add("note", f"Standard of loading: {n['standard_of_loading']}", ("loading",))
    for o in a.get("other_values", []):
        m, src = term_meaning(a, o["label"])
        val = f"{fnum(o['value'])}" + (f" {o['unit'].lower()}" if o.get("unit") else "")
        add("other", f"Printed on {view_name(o['view'])}: '{o['label']} = {val}' - " + (f"{m} ({SOURCE_WORDS[src]})" if m else SOURCE_WORDS["unknown"]),
            ("other", o["label"], o["value"]))
    for name, what, views, in_notes in components(a):
        where = ("shown on " + ", ".join(views[:4])) if views else "mentioned in the notes"
        add("component", f"Component: {name} ({where}) - {what}", ("comp", name))
    if n.get("legend_key"):
        add("note", "Legend (what each kind of line means on this sheet): " + "; ".join(
            f"{k['label']} = {k['colour']} {'hatching' if k['style'] == 'hatched' else 'dashed lines' if k['style'] == 'dashed' else 'lines'}"
            for k in n["legend_key"]), ("legend_key",))
    dis = {(d["view"], d["text"]): d for d in a.get("dismantle", [])}
    for text, where in callout_index(a).items():
        views = list(dict.fromkeys(v for v, _ in where))
        cw = colour_words(a, where[0][1])
        t = f"Text '{text}' is written on {', '.join(views[:4])}" + (f" (in {cw})" if cw else "")
        comp = component_of(text)
        if comp:
            t += f" - the {comp[0]}: {comp[1]}"
        d = next((dis[(v, text)] for v in views if (v, text) in dis), None)
        if d and d.get("tips"):
            bits = []
            if d.get("short_strokes"):
                lk = next((k for k in (a.get("notes") or {}).get("legend_key", []) if re.search(r"DISMANT|DIAMANT", k["label"], re.I)), None)
                bits.append("drawn in short dashes / strokes" + (f" - the legend's '{lk['label']}' ({lk['colour']} {lk['style']} lines)" if lk else ""))
            if d.get("angled"):
                bits.append("angled walls at the end of the existing structure")
            if d.get("parts"):
                bits.append("labelled " + " / ".join(f"'{p}'" for p in d["parts"]))
            t += (f". Its {d['tips']} arrow(s) point at existing work" + (": " + ", ".join(bits) if bits else "")
                  + ("; proposed (red) work is drawn right next to it" if d.get("proposed_next_to") else ""))
        add("callout", t, ("call", text))
    for f in a.get("findings", []):
        add("finding", f"Disagreement on the drawing: {f}", ("finding", f[:30]))
    for v in a.get("views", []):
        bd = v.get("band")
        if bd:
            labels = [r["label"] for r in bd["rows"]]
            cols = [" / ".join(f"{k} {c[k]}" for k in labels if k in c) for c in bd["columns"]]
            for k in range(0, len(cols), 12):                     # (a long table in parts, so a part fits a prompt)
                part = f" (columns {k + 1}-{min(k + 12, len(cols))} of {len(cols)})" if len(cols) > 12 else ""
                add("band", f"Value table under '{v['title']}'{part} ({len(labels)} rows: {', '.join(labels)}; {len(cols)} columns), column by column, left to right: "
                    + " | ".join(cols[k:k + 12]), ("band", v["title"], k // 12))
    for v in a.get("views", []):
        kp = v.get("key_plan")
        if kp:
            for side in ("left", "right"):
                st = [x["text"] for x in kp.get("stations", []) if x["side"] == side]
                if st:
                    add("key_plan", f"Key plan, {side} end of the line: " + "; ".join(st), ("kp", "stations", side))
            for key, label in (("bridges", "Bridge callouts"), ("tracks", "Tracks"), ("curves", "Curves"), ("gradients", "Gradients"),
                               ("boundary", "Land/boundary")):
                if kp.get(key):
                    add("key_plan", f"Key plan, {label.lower()}: " + " | ".join(dict.fromkeys(kp[key])), ("kp", key))
            if kp.get("markers"):
                add("key_plan", "Key plan, chainage/KM/FL marks: " + " | ".join(m["text"] for m in kp["markers"]), ("kp", "markers"))
            extra = []
            if kp.get("flow"):
                extra.append("a direction-of-flow arrow")
            if kp.get("bore_holes"):
                extra.append(f"{kp['bore_holes']} bore hole location(s)")
            if extra:
                add("key_plan", "Key plan also shows " + " and ".join(extra), ("kp", "extra"))
        elif v["kind"] in ("key_plan",) and v.get("texts"):
            add("view_text", f"Text on the {v['title']}: " + " | ".join(v["texts"])[:900], ("vtext", v["title"]))
    return F


def tokens(text):
    t = text.lower().replace("'", " ")
    # dotted shorthand as one word: "F.L" -> fl, "H.F.L." -> hfl, "B.L" -> bl, "R.L" -> rl (else single letters)
    t = re.sub(r"\b((?:[a-z]\.){1,3}[a-z])\b\.?", lambda m: m.group(1).replace(".", ""), t)
    return [w for w in re.findall(r"[a-z]+(?:/[a-z]+)?|\d+(?:\.\d+)?", t) if w not in STOP]


def expand(q):
    out = []
    for w in tokens(q):
        out.append(w)
        out += SYN.get(w, [])
        if w.endswith("s") and w[:-1] in SYN:
            out += SYN[w[:-1]]
    return out


def score_facts(facts, question):
    docs = [Counter(tokens(f["text"])) for f in facts]
    df = Counter(w for d in docs for w in d)
    N = len(docs)
    q = expand(question)
    qnums = set(re.findall(r"\d+(?:\.\d+)?", question))
    scores = []
    for f, d in zip(facts, docs):
        s = sum(math.log(1 + N / df[w]) for w in set(q) if w in d)
        nums = set(re.findall(r"\d+(?:\.\d+)?", f["text"]))
        for x in qnums:                                   # a number in the question: facts that contain it
            if x in nums or any(y.rstrip("0").rstrip(".") == x.rstrip("0").rstrip(".") for y in nums if "." in y or "." in x):
                s += 6
        if f["group"] == "view" and isinstance(f["ref"], tuple):        # the question names this view
            title = set(tokens(f["ref"][1])) - {"section", "details", "detail", "typical", "plan", "half"}
            if title and title <= set(tokens(question)):
                s += 8
        scores.append(s)
    return scores


def select(a, question, must=(), max_facts=MAX_FACTS):
    facts = all_facts(a)
    sc = score_facts(facts, question)
    core = [i for i, f in enumerate(facts) if f["group"] in CORE_GROUPS]
    forced = [i for i, f in enumerate(facts) if f["ref"] in must]
    ranked = [i for i in sorted(range(len(facts)), key=lambda i: -sc[i]) if sc[i] > 0 and i not in core and i not in forced]
    chosen, chars = [], 0
    for i in core + forced + ranked:
        if i in chosen:
            continue
        if len(chosen) >= max_facts or chars + len(facts[i]["text"]) > MAX_CHARS:
            if i in forced:
                pass
            else:
                break
        chosen.append(i)
        chars += len(facts[i]["text"])
    order = {g: k for k, g in enumerate(["bridge", "views", "title_block", "view", "level", "dim", "plain_dims", "table", "bore_log",
                                         "band", "other", "note", "abbreviation", "component", "callout", "finding", "key_plan", "view_text"])}
    chosen.sort(key=lambda i: (order.get(facts[i]["group"], 99), i))
    return [facts[i] for i in chosen]


def context(a, question, must=()):
    fs = select(a, question, must)
    return "GAD facts (read from the drawing):\n" + "\n".join(f"- {f['text']}" for f in fs)


def prompt(a, question, must=()):
    return f"{context(a, question, must)}\n\nQuestion: {question}"
