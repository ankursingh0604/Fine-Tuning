"""Build the L-section v5 dataset (Qwen3.5-4B) from data/v5/annotations/ (run scripts/annotate_v5.py first).

    .venv\\Scripts\\python scripts\\build_dataset_v5.py

Two skills, on all 157 sheets (no sheet held back: the user's choice - validation rows come from every sheet):

  Reading (img_read_*)  a crop made exactly as Railsight makes it (find_crop: the label's own strip, turned to read left
                        to right, on a white canvas) + "What text is written in this crop? Reply with the exact text
                        only." -> the text as printed. Bridge callouts, level blocks, crossings, curve blocks, curve
                        points, DETAILS OF BRIDGES cells, band values (Railsight's band-value crop), and any other text
                        on the sheet - the skill every answer of the kit stands on.
  Answering (qa_*)      the sheet's facts (scripts/lsec_facts.py, as the question tool gives them) + the question ->
                        the answer from them: bridges (callout, proposal, levels, the bridge table), crossings, curves
                        (degree, points, which curve a chainage / bridge is on), band values at any chainage (printed or
                        interpolated, said so), gradients and grade points, stations, KM marks, bearings, TBMs, sheet
                        details, notes, abbreviations, anything else printed ("what is X?"), the meaning of a term
                        (the sheet's list / standard / found on the web / unknown), and what is not on the sheet.
  Spot (img_spot)       a crop of the spot asked about + the facts + the question (as the question tool sends it).

Output: data/v5/dataset/{train,val}.jsonl and images/; stats.json.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runpod"), str(ROOT / "gad_tools")]
import pymupdf                  # noqa: E402
import find_crop as F           # noqa: E402
import sheet_objects as SO      # noqa: E402
import band_table               # noqa: E402
import bridge_table as BT       # noqa: E402
import glossary as G            # noqa: E402
import lsec_facts as LF         # noqa: E402

ANN = ROOT / "data" / "v5" / "annotations"
OUT = ROOT / "data" / "v5" / "dataset"
IMG = OUT / "images"
READ_Q = "What text is written in this crop? Reply with the exact text only."
CONTEXT = "railway L-section drawing"
VAL_SHARE = 0.05
rng = random.Random(41)


# ======================================================================== wording
OPEN = ["What is the ", "Give me the ", "Tell me the ", "", "What's the ", "Find the "]


def vary(q):
    """A question as engineers type it: opener, case, the odd shorthand."""
    if q.startswith("What is the ") and rng.random() < 0.5:
        q = rng.choice(OPEN) + q[len("What is the "):]
    for a, b in ((r"\bformation level\b", ["FL", "formation lvl"]), (r"\brail level\b", ["RL", "rail lvl"]),
                 (r"\bbed level\b", ["BL", "bed lvl"]), (r"\bchainage\b", ["CH", "ch"]), (r"\bexisting\b", ["exg", "ex."]),
                 (r"\bproposed\b", ["prop.", "new"]), (r"\bbridge\b", ["br", "Br. No."]), (r"\bdegree of curve\b", ["degree"])):
        if re.search(a, q, re.I) and rng.random() < 0.25:
            q = re.sub(a, rng.choice(b), q, count=1, flags=re.I)
    r = rng.random()
    q = q.lower() if r < 0.15 else q.upper() if r < 0.2 else q
    return q[:1].upper() + q[1:] if r >= 0.15 else q


# ======================================================================== reading crops (Railsight's own)
class Crops:
    """Railsight's crop functions on one rendered sheet (find_crop.Finder without a model)."""

    def __init__(self, pdf, page_no, sid):
        self.sid = sid
        self.f = F.Finder(F.open_sheet(ROOT / pdf, OUT / "_sheets" / sid, page_no), lambda imgs, q: "", OUT / "_sheets" / sid)
        self.n = 0

    def strip(self, parts, what):
        img, _ = self.f.strip_crop(parts)
        return self.save(F.pad_canvas(img), what)

    def value(self, w, what):
        """A band value's crop, as Finder.read_value makes it."""
        import numpy as np
        from PIL import Image
        W, H = self.f.s.image.size
        region = tuple(int(v) for v in F.grow(w.box, 0.6 * w.h, W, H))
        img = self.f.clean_crop([w], region)
        a = np.asarray(img).copy()
        dark = a.min(axis=2) < 200
        a[dark.mean(axis=1) > 0.85, :] = 255
        a[:, dark.mean(axis=0) > 0.85] = 255
        img = Image.fromarray(a)
        ax, ay = abs(w.dir[0]) > 0.5, abs(w.dir[1]) > 0.5
        mx, my = (0 if ax else 0.15 * w.h), (0 if ay else 0.15 * w.h)
        keep = [int(round(v)) for v in (max(0, w.box[0] - mx), max(0, w.box[1] - my), min(W, w.box[2] + mx), min(H, w.box[3] + my))]
        white = Image.new("RGB", img.size, "white")
        white.paste(img.crop((keep[0] - region[0], keep[1] - region[1], keep[2] - region[0], keep[3] - region[1])),
                    (keep[0] - region[0], keep[1] - region[1]))
        return self.save(F.pad_canvas(F.upright(white, w.dir)), what)

    def spot(self, box_pt, what):
        """A 768 x 768 crop of the sheet around a point (PDF points), as the question tool sends with a question."""
        z = self.f.s.dpi / 72
        W, H = self.f.s.image.size
        cx, cy = (box_pt[0] + box_pt[2]) / 2 * z, (box_pt[1] + box_pt[3]) / 2 * z
        half = 384 * self.f.s.dpi / 150
        img = self.f.s.image.crop((int(max(0, cx - half)), int(max(0, cy - half)), int(min(W, cx + half)), int(min(H, cy + half))))
        return self.save(img.resize((768, 768)), what)

    def save(self, img, what):
        IMG.mkdir(parents=True, exist_ok=True)
        self.n += 1
        name = f"{self.sid}_{self.n:04d}_{re.sub(r'[^A-Za-z0-9]+', '_', what)[:24]}.png"
        img.save(IMG / name)
        return f"images/{name}"


def reading_rows(ann, cr):
    """(task, image, question, answer) - every kind of label on the sheet, read exactly."""
    out = []
    objs = cr.f.objects()
    for b in objs.bridges:
        out.append(("img_read_bridge", cr.strip(b.best(), f"br_{b.num}"), READ_Q, SO.joined(b.best()).text))
        for lv in b.levels[:1]:
            out.append(("img_read_levels", cr.strip(lv, f"lv_{b.num}"), READ_Q, SO.joined(lv).text))
    for c in objs.crossings:
        out.append(("img_read_crossing", cr.strip(c.best(), f"x_{c.label}"), READ_Q, SO.joined(c.best()).text))
    for c in objs.curves:
        for d in c.details[:1]:
            out.append(("img_read_curve", cr.strip(d, f"cv_{c.block_num or c.num}"), READ_Q, SO.joined(d).text))
        for p, (ch, w) in list(c.points.items())[:2]:
            out.append(("img_read_curve_point", cr.strip([w], f"cp_{p}"), READ_Q, w.text))
    # the DETAILS OF BRIDGES table: a few cells of each row
    for br, row in list(cr.f.details_table().items())[:12]:
        for key in rng.sample([k for k in row if k not in ("sno",)], min(3, len(row) - 1)):
            ws = row[key]
            if all(hasattr(w, "length") for w in ws):
                out.append(("img_read_table", cr.strip(ws, f"t_{br}_{key}"), READ_Q, BT.text(ws)))
    # band values (the crop Railsight reads a band value from)
    for b in cr.f.bands():
        for r in b.rows:
            if r is b.chain or len(r.words) < band_table.MIN_VALUES * 2:
                continue
            for w in rng.sample(r.words, min(3, len(r.words))):
                out.append(("img_read_band", cr.value(w, f"bv_{w.text}"), READ_Q, w.text))
    # anything else printed: a sample of the sheet's other texts (plan labels, notes, legend, title, km marks ...)
    used = {id(w) for b in objs.bridges for w in b.best()}
    allowed = {t["text"].strip() for t in ann.get("all_text") or []}          # (names and designations already left out)
    others = [w for w in cr.f.s.words if id(w) not in used and len(re.findall(r"[A-Za-z0-9]", w.text)) >= 2 and w.text.strip() in allowed]
    for w in rng.sample(others, min(30, len(others))):
        out.append(("img_read_text", cr.strip([w], f"tx_{w.text}"), READ_Q, w.text))
    return out


# ======================================================================== answers from the facts
def qa_rows(ann, facts):
    """(task, question, answer, must refs, extra fact lines, spot box in PDF points or None)."""
    out = []
    v5 = ann.get("v5") or {}
    table = {re.sub(r"\s", "", r["bridge"]).upper(): r for r in v5.get("bridge_table", [])}
    # ---- bridges
    for b in v5.get("bridges", []):
        f, num = b["fields"], b["num"]
        name = f.get("br_no") or num
        refs = [("bridge", num)]
        box = b["callouts"][0]["bbox"] if b["callouts"] else None
        if f.get("chainage"):
            ans = f"Bridge {name} is at chainage {f['chainage']}" + (f"; the proposed bridge is at chainage {f['_prop_chainage']}" if f.get("_prop_chainage") else "") + "."
            out.append(("qa_bridge", vary(rng.choice([f"What is the chainage of bridge {num}?", f"Where is bridge {num}?", f"ch of br {num}"])), ans, refs, [], box))
        if f.get("exg_structure") or f.get("exg_configuration"):
            ans = (f"The existing bridge {name} is {f.get('exg_structure') or '(type not printed)'}" + (f", {f['exg_configuration']}" if f.get("exg_configuration") else "")
                   + (f", crossing {f['description']}" if f.get("description") else "") + ".")
            if f.get("prop_structure") or f.get("prop_configuration"):
                ans += (f" It is proposed as {f.get('prop_configuration') or ''} {f.get('prop_structure') or ''}".rstrip() + (f" ({f['prop_type']})" if f.get("prop_type") else "")
                        + (f" at chainage {f['_prop_chainage']}" if f.get("_prop_chainage") else "") + ".")
            else:
                ans += " No proposal is printed with it."
            out.append(("qa_bridge", vary(rng.choice([f"What type is bridge {num}?", f"Existing and proposed type of bridge {num}?",
                                                      f"What is the span of bridge {num}?", f"Bridge {num} details"])), ans, refs, [], box))
        lv = b.get("level_block")
        if lv and lv["values"]:
            v = rng.choice(lv["values"])
            out.append(("qa_bridge_level", vary(rng.choice([f"What is the {v['label']} of bridge {num}?", f"{v['label']} of br {num}",
                                                             f"Give the {v['label']} at bridge {num}"])),
                        f"{v['label']} of bridge {num} = {v['value']} (its level block).", refs + [("levels", num)], [], lv["bbox"]))
            out.append(("qa_bridge_level", vary(f"What are the levels of bridge {num}?"),
                        f"Bridge {num}'s level block gives: " + "; ".join(f"{x['label']} = {x['value']}" for x in lv["values"]) + ".",
                        refs + [("levels", num)], [], lv["bbox"]))
        row = table.get(num) or table.get(re.sub(r"(UP|DN)$", "", num))
        if row:
            cells = {k: v for k, v in row["cells"].items() if v and k not in ("sno", "br")}
            for k in rng.sample(list(cells), min(3, len(cells))):
                q = rng.choice([f"What is the {LF.TABLE_NAME.get(k, k)} of bridge {num}?", f"{LF.TABLE_NAME.get(k, k)} for br {num}"])
                out.append(("qa_bridge_table", vary(q), f"The DETAILS OF BRIDGES table gives the {LF.TABLE_NAME.get(k, k)} of bridge {num} as {cells[k]}.",
                            refs + [("table", re.sub(r"\s", "", row["bridge"]).upper())], [], row["bbox"]))
    # ---- crossings
    for c in v5.get("crossings", []):
        out.append(("qa_crossing", vary(rng.choice([f"What is printed for {c['label']}?", f"Details of {c['label']}"])),
                    f"{c['label']}: '{c['text']}'" + (f"; its levels: {c['level_block']['text']}" if c.get("level_block") else "") + ".",
                    [("xing", c["label"])], [], c["bbox"]))
    # ---- curves
    curves = v5.get("curves", [])
    for c in curves:
        if not c.get("points"):
            continue
        lo, hi = min(c["points"].values()), max(c["points"].values())
        ref = [("curve", c["num"], c.get("track"))]
        if c.get("degree"):
            out.append(("qa_curve", vary(rng.choice([f"What is the degree of curve {c['num']}?", f"degree of C. No. {c['num']}"])),
                        f"Curve {c['num']}" + (f" ({c['track']})" if c.get("track") else "") + f" has a degree of {c['degree']} (its details block).", ref, [], c["bbox"]))
        out.append(("qa_curve", vary(f"Where does curve {c['num']} start and end?"),
                    f"Curve {c['num']} runs from chainage {LF.km(lo)} to {LF.km(hi)}; its points: " + "; ".join(f"{p} {LF.km(v)}" for p, v in c["points"].items()) + ".",
                    ref, [], c["bbox"]))
        ch = round(rng.uniform(lo, hi), 3)
        on = [x for x in curves if x.get("points") and min(x["points"].values()) <= ch <= max(x["points"].values())]
        out.append(("qa_curve_at", vary(f"Is chainage {LF.km(ch)} on a curve?"),
                    f"Yes: chainage {LF.km(ch)} lies on " + " and ".join(f"curve {x['num']}" + (f" ({x['track']})" if x.get("track") else "")
                                                                        + f" ({LF.km(min(x['points'].values()))} to {LF.km(max(x['points'].values()))}"
                                                                        + (f", degree {x['degree']}" if x.get("degree") else "") + ")" for x in on) + ".",
                    [("curve", x["num"], x.get("track")) for x in on], [], None))
    # ---- band values at a chainage (printed or between two printed columns)
    cols = (ann.get("bands") or {}).get("columns") if isinstance(ann.get("bands"), dict) else None
    if cols:
        lo, hi = min(c["chainage"] for c in cols), max(c["chainage"] for c in cols)
        for _ in range(6):
            ch = rng.choice([c["chainage"] for c in cols]) if rng.random() < 0.4 else round(rng.uniform(lo, hi), 3)
            vals = LF.band_at(ann, ch)
            if not vals:
                continue
            k, name, v, how = rng.choice(vals)
            extra = [f"Band value at chainage {LF.km(ch)}: {n} = {g_} ({h})" for _, n, g_, h in vals]
            out.append(("qa_band", vary(rng.choice([f"What is the {name} at chainage {LF.km(ch)}?", f"{name} at CH {LF.g(ch)}", f"{name} at {LF.km(ch)}"])),
                        f"The {name} at chainage {LF.km(ch)} is {v:g} m ({how}).", [], extra, None))
        ch = hi + rng.uniform(200, 3000)
        out.append(("qa_band_absent", vary(f"What is the formation level at chainage {LF.km(ch)}?"),
                    f"Chainage {LF.km(ch)} is not on this sheet: its data band runs from {LF.km(lo)} to {LF.km(hi)}. Nothing is guessed - look at the next sheet.",
                    ["band"], [], None))
    # ---- gradients, grade points
    for p in rng.sample(ann.get("plan_grade_points") or [], min(3, len(ann.get("plan_grade_points") or []))):
        out.append(("qa_gradient", vary(rng.choice([f"What is the gradient after chainage {p['chainage']}?", f"FL at the grade point at {p['chainage']}"])),
                    f"At the grade point at chainage {p['chainage']} ({p.get('line')}) the FL is {LF.g(p.get('fl'))}; the gradient is {p.get('gradient_before')} before it "
                    f"and {p.get('gradient_after')} after it.", [("gp", p.get("chainage_m"))], [], p.get("bbox")))
    for s in rng.sample(ann.get("gradient_segments") or [], min(2, len(ann.get("gradient_segments") or []))):
        out.append(("qa_gradient", vary(f"What is the gradient near chainage {LF.km(s['mid_chainage_m'])}?"),
                    f"Near chainage {LF.km(s['mid_chainage_m'])} the {s.get('line')} is on '{s.get('gradient_label')}': {s.get('gradient')} ({s.get('percent')}), "
                    f"over {LF.g(s.get('length_m'))} m.", [("grad", s.get("mid_chainage_m"))], [], s.get("bbox")))
    # ---- marks, TBMs, stations, sheet, notes, abbreviations
    for t in (ann.get("tbm_benchmarks") or [])[:3]:
        out.append(("qa_tbm", vary(rng.choice([f"What is the level of {t['tbm_id']}?", f"Details of TBM {t['tbm_id']}"])),
                    f"{t['tbm_id']}: MSL {LF.g(t.get('msl_m'))} m at chainage {LF.km(t.get('chainage_m'))} (easting {LF.g(t.get('easting'))}, "
                    f"northing {LF.g(t.get('northing'))}); {t.get('description') or ''}".strip(), [("tbm", t["tbm_id"])], [], t.get("bbox")))
    for k in (ann.get("km_marks") or [])[:2]:
        out.append(("qa_marks", vary(f"What existing KM corresponds to proposed KM {k.get('proposed_km')}?"),
                    f"At proposed KM {k.get('proposed_km')} the existing KM is {k.get('existing_km')} (printed '{k.get('text')}').", [("kmm", k.get("chainage_m"))], [], k.get("bbox")))
    for s in ann.get("stations") or []:
        out.append(("qa_station", vary(rng.choice(["Which stations are on this sheet?", f"Where is {s.get('name')}?"])),
                    f"{s.get('name')} is at the {s.get('direction')} end of the sheet" + (f"; printed with it: {', '.join(s['details'])}" if s.get("details") else "") + ".",
                    [("station", s.get("name"))], [], s.get("bbox")))
    info = ann.get("sheet_info") or {}
    out.append(("qa_sheet", vary(rng.choice(["Which chainages does this sheet cover?", "What is this sheet?", "Sheet number and drawing number?"])),
                f"This is sheet {info.get('sheet_no') or '?'} ({info.get('title') or '?'}), drawing no. {info.get('drawing_no') or '?'}, covering chainage "
                f"{info.get('chainage_from') or '?'} to {info.get('chainage_to') or '?'}; previous sheet {info.get('previous_sheet') or '-'}, next {info.get('next_sheet') or '-'}.",
                ["sheet"], [], None))
    for n in rng.sample(ann.get("notes") or [], min(3, len(ann.get("notes") or []))):
        words = [w for w in re.findall(r"[A-Z]{4,}", n["text"].upper()) if w not in ("SHALL", "THAT", "WITH", "FROM", "WILL", "OTHER", "WISE")]
        if words:
            topic = rng.choice(words).lower()
            out.append(("qa_note", vary(f"What do the notes say about {topic}?"), f"Note {n['no']}: {n['text']}", [("note", n["no"])], [], None))
    for k, v in rng.sample(list((ann.get("abbreviations") or {}).items()), min(2, len(ann.get("abbreviations") or {}))):
        out.append(("qa_abbr", vary(f"What does {k} stand for?"), f"{k} stands for {v} (the sheet's abbreviation list).", [("abbr", k)], [], None))
    # ---- anything else printed: where it is and, for a term, what it means
    texts = [t for t in ann.get("all_text") or [] if len(re.findall(r"[A-Za-z]", t["text"])) >= 3]
    for t in rng.sample(texts, min(6, len(texts))):
        s = t["text"].strip()
        m, src = G.meaning(s, context=CONTEXT) if len(s) <= 12 else (None, "unknown")
        ans = f"'{s}' is printed on this sheet."
        if m:
            ans += f" {s} means {m} ({'from the sheet' if src == 'glossary' else 'standard meaning'})."
        out.append(("qa_text", vary(rng.choice([f"What is '{s}'?", f"Where is {s} written?", f"What does {s} mean?"])), ans, [("text", s)], [], t["bbox"]))
    # ---- a term the sheet does not explain: known / web fits / web off-topic / unknown (as GAD v2)
    terms = sorted({k for k in G.LSEC} | {k for k in (ann.get("abbreviations") or {})})
    if terms:
        k = rng.choice(terms)
        m, _ = G.meaning(k, context=CONTEXT)
        if m:
            out.append(("qa_term", vary(f"What does {k} mean on an L-section?"), f"{k}: {m} (standard meaning on L-section drawings).", [], [], None))
    case = rng.random()
    fake = rng.choice(["PQR", "XTL", "ZMC", "KLD"])
    if case < 0.5:
        extra = [f"Web search for '{fake}' (not confirmed by the drawing): {fake} - an abbreviation used in survey drawings for a control line"]
        out.append(("qa_term_web", vary(f"What is {fake}?"), f"{fake} is not explained on this sheet. A web search for the term says: an abbreviation used in "
                                                             "survey drawings for a control line - found on the web, not confirmed by the drawing.", [], extra, None))
    else:
        out.append(("qa_term_unknown", vary(f"What is {fake}?"), f"{fake} is not printed on this sheet and its meaning is not known to me (no web result). "
                                                                 "Nothing is guessed.", [], [], None))
    # ---- not on this sheet
    have = {b["num"] for b in v5.get("bridges", [])}
    nums = [int(re.match(r"\d+", n).group()) for n in have if re.match(r"\d+", n)]
    if nums:
        miss = max(nums) + rng.randint(30, 400)
        out.append(("qa_absent", vary(f"What is the chainage of bridge {miss}?"),
                    f"Bridge {miss} is not printed on this sheet (bridges here: {', '.join(sorted(have))}). Nothing is guessed - it may be on another sheet.",
                    [("bridge", n) for n in sorted(have)][:4], [], None))
    return out


# ======================================================================== assemble
def one_sheet(path):
    """[record] of one sheet (run in a worker; each sheet has its own seeded random, so the build is repeatable)."""
    global rng
    ann = json.loads(Path(path).read_text(encoding="utf-8"))
    sid = ann["sheet_id"]
    rng = random.Random(f"v5-{sid}")
    facts = LF.all_facts(ann)
    rows = []
    try:
        cr = Crops(ann["source_pdf"], ann["page_index"], sid)
        for task, img, q, a in reading_rows(ann, cr):
            rows.append({"task": task, "image": img, "prompt": q, "answer": a})
    except Exception as e:                                   # noqa: BLE001 - one sheet's crops failing must not stop the rest
        print(f"{sid}: reading crops failed: {type(e).__name__}: {e}", flush=True)
        cr = None
    for task, q, a, must, extra, box in qa_rows(ann, facts):
        row = {"task": task, "prompt": LF.prompt(ann, q, must, extra, facts), "answer": a}
        if box and cr is not None and rng.random() < 0.3:          # some answers with the spot's crop, as the tool sends it
            row["image"] = cr.spot(box, f"spot_{task}")
            row["task"] = "img_spot_" + task[3:]
        rows.append(row)
    recs = []
    for i, r in enumerate(rows):
        split = "val" if rng.random() < VAL_SHARE else "train"
        content = ([{"type": "image"}] if r.get("image") else []) + [{"type": "text", "text": r["prompt"]}]
        rec = {"id": f"{sid}_{r['task']}_{i:04d}", "sheet": sid, "split": split, "task": r["task"],
               "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}]}
        if r.get("image"):
            rec["image"] = r["image"]
        recs.append(rec)
    print(f"{sid:<28} rows {len(recs):>4}", flush=True)
    return recs


def main():
    from concurrent.futures import ProcessPoolExecutor
    paths = [str(p) for p in sorted(ANN.glob("*.json"))]
    out = {"train": [], "val": []}
    stats = Counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for recs in ex.map(one_sheet, paths):
            for r in recs:
                out[r["split"]].append(r)
                stats[r["task"]] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    shuffle = random.Random(7)
    for split, rows in out.items():
        rows.sort(key=lambda r: r["id"])
        shuffle.shuffle(rows)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "stats.json").write_text(json.dumps({"rows": {k: len(v) for k, v in out.items()}, "tasks": dict(sorted(stats.items()))}, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in out.items()}, dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
