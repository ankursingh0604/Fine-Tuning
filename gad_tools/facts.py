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
    "revision": ["rev", "revision"],
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
    "proposed": ["proposed", "prop"],
    "scour": ["scour"],
    "velocity": ["velocity", "speed"],
    "catchment": ["catchment"],
    "waterway": ["waterway", "water", "way"],
    "ballast": ["ballast", "cushion"],
    "sleeper": ["sleeper", "psc"],
    "level": ["lvl", "level"],
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
        add("views", "Views on the drawing: " + "; ".join(f"{v['title']} ({v['scale']})" for v in a["views"]), "views")
    for v in a.get("views", []):
        add("view", f"View '{v['title']}' (scale {v['scale']}): {v['represents']}", ("view", v["title"]))
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
    for c in {(c["name"], c["view"]) for c in a.get("centre_lines", [])}:
        add("dim", f"Centre line shown: {c[0]} (on {view_name(c[1])})", ("cl", c[0]))
    plain = defaultdict(list)
    for d in a.get("dims", []):
        plain[(d["view"], d["direction"])].append(fnum(d["value"]))
    for (view, direction), vals in plain.items():
        add("plain_dims", f"Unlabelled dimension figures (mm, {direction}) on {view_name(view)}: {', '.join(vals)} - the drawing does not "
            "write what each of these measures", ("plain", view, direction))
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
        add("bore_log", t, ("bore", bl["view"]))
    n = a.get("notes") or {}
    for key, label in (("notes", "Note"), ("special_note", "Special note"), ("add_note", "Additional note"), ("fill_note", "Fill note"),
                       ("design_criteria", "Design criteria"), ("reference_drawings", "Reference drawing")):
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
    for f in a.get("findings", []):
        add("finding", f"Disagreement on the drawing: {f}", ("finding", f[:30]))
    for v in a.get("views", []):
        if v["kind"] in ("key_plan",) and v.get("texts"):
            add("view_text", f"Text on the {v['title']}: " + " | ".join(v["texts"])[:900], ("vtext", v["title"]))
    return F


def tokens(text):
    t = text.lower().replace("'", " ")
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
                                         "note", "abbreviation", "finding", "view_text"])}
    chosen.sort(key=lambda i: (order.get(facts[i]["group"], 99), i))
    return [facts[i] for i in chosen]


def context(a, question, must=()):
    fs = select(a, question, must)
    return "GAD facts (read from the drawing):\n" + "\n".join(f"- {f['text']}" for f in fs)


def prompt(a, question, must=()):
    return f"{context(a, question, must)}\n\nQuestion: {question}"
