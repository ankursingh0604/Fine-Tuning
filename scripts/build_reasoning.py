"""Build the reasoning (chain-of-thought) dataset from data/annotations/.

    python scripts/build_dataset.py      # first: it clears data/dataset/images
    python scripts/build_reasoning.py

Every answer is "<think> numbered steps </think> final answer". The steps only use values that are
printed in the image(s) given with the question, or arithmetic on them, and follow the rules agreed for
these drawings:
  - a chainage between two data-band columns takes the NEAREST column; never interpolate
  - a bridge whose proposed FL is below MIN FL REQ. is flagged "FLAG: For bridge X min. FL = .., and FL = .."
  - curve points ST/TC/CT/TS are also called TTP1/CTP1/CTP2/TTP2 (abbreviations table)
  - RL = FL + 0.762 (note 5); ruling gradient 1 in 150 (note 6)

Writes data/dataset/reasoning_{train,val,test}.jsonl, same splits as build_dataset.py. A row can carry
several images: "images" lists them and the user message has one image entry per image, in order.
"""
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as D           # rendering, saving, splits and wording helpers   # noqa: E402

ROOT, ANN, OUT = D.ROOT, D.ANN, D.OUT
SEED = 11
CROP = 672                          # px; small crops keep multi-image questions inside the token budget
RULING = 150                        # note 6: ruling gradient 1 in 150
rng = random.Random(SEED)


# ---------------------------------------------------------------- formatting

def f3(v):
    return f"{v:.3f}"


def g(v):
    """A number in plain decimal, never scientific: 1242662.9, 1922, 0.125."""
    return f"{v:.3f}".rstrip("0").rstrip(".")


def ch_txt(ch):
    return f"{ch:.0f}" if float(ch).is_integer() else f"{g(ch)}"


def answer(steps, final):
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return f"<think>\n{body}\n</think>\n{final}"


def row(sheet, task, imgs, question):
    paths = [p for p, _ in imgs]
    return {"sheet": sheet, "task": task, "image": paths[0], "images": paths,
            "sizes": [list(s) for _, s in imgs],
            "messages": [{"role": "user", "content": [{"type": "image", "image": p} for p in paths]
                          + [{"type": "text", "text": question}]},
                         {"role": "assistant", "content": [{"type": "text", "text": None}]}]}


def finish(r, text):
    r["messages"][1]["content"][0]["text"] = text
    return r


def name_of(b):
    bid = b["bridge_id"]
    return bid if bid.startswith(("ROB", "LC")) else f"bridge {bid}"


def union(boxes):
    boxes = list(boxes)
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def flag(b):
    return D.min_fl_flag(b)


# ---------------------------------------------------------------- images

def save(img, name):
    return D.save(img, "r_" + name), img.size


def crop_around(page, box, name, side_px=CROP):
    clip = D.square_around(box, side_px / D.ZOOM)
    img, origin = D.render(page, clip, (side_px, side_px))
    return save(img, name), origin, img.size


def fully_in(box, origin, size):
    return D.to_px(box, origin, size)[1] >= 0.98


def band_window(cols, idx, n=16):
    s = max(0, min(idx - n // 2, len(cols) - n))
    return cols[s:s + n]


def band_image(page, ann, win, name):
    """Row-label strip joined to the band columns of `win` (same layout as the direct-answer band crops)."""
    bd = ann["bands"]
    y0, y1 = bd["row_y"][0][0] - 6, bd["row_y"][-1][1] + 6
    ls = bd["label_strip"]
    strip, _ = D.render(page, [ls[0] - 4, y0, ls[2] + 6, y1])
    band, _ = D.render(page, [win[0]["bbox"][0] - 0.6, y0, win[-1]["bbox"][2] + 0.6, y1])
    img = Image.new("RGB", (strip.width + band.width, max(strip.height, band.height)), "white")
    img.paste(strip, (0, 0))
    img.paste(band, (strip.width, 0))
    return save(D.pad28(img), name)


def chainage_to_x(ann):
    cols = sorted((c["chainage"], c["x"]) for c in ann["bands"]["columns"])

    def f(ch):
        for (ca, xa), (cb, xb) in zip(cols, cols[1:]):
            if ca <= ch <= cb:
                return xa + (xb - xa) * (ch - ca) / (cb - ca)
        return None
    return f


# ---------------------------------------------------------------- question families

def r_bridge_bands(page, ann, sheet, blocked, blocked_ch):
    """Bridge L-section crop + data-band crop: ground level, cut/fill and FL check at the bridge."""
    rows = []
    cols = [c for c in ann["bands"]["columns"] if c["checks_ok"]]
    for b in ann["bridges"]:
        ch, lv = b.get("chainage_m"), b.get("lsection_levels") or {}
        if (not b["complete"] or b["bridge_id"] in blocked or "lsection_callout" not in b or not ch
                or "proposed_formation_level" not in lv or not cols or not cols[0]["chainage"] <= ch <= cols[-1]["chainage"]):
            continue
        idx = min(range(len(cols)), key=lambda i: abs(cols[i]["chainage"] - ch))
        near = cols[idx]
        if near["chainage"] in blocked_ch:
            continue
        core = union([b["lsection_callout"]["bbox"], lv["bbox"]])
        (p1, s1), origin, size = crop_around(page, core, f"s{sheet}_rbr{b['bridge_id']}", 1008)
        if not fully_in(lv["bbox"], origin, size):
            continue
        win = band_window(cols, idx)
        p2, s2 = band_image(page, ann, win, f"s{sheet}_rbands_{near['chainage']:.0f}")
        v, dist = near["values"], abs(near["chainage"] - ch)
        other = cols[idx + 1] if near["chainage"] < ch and idx + 1 < len(cols) else cols[idx - 1] if idx else None
        fl, req = lv["proposed_formation_level"], lv.get("min_formation_level_required")
        cut = v["cut_fill"]
        steps = [f"Image 1, the L-section callout: {name_of(b)} has its centre line at CH {g(ch)}. "
                 f"Its level block gives FL {fl}" + (f" and MIN FL REQ. {req}" if req is not None else "") + "."]
        steps.append(f"Image 2, the data bands: the bridge chainage {g(ch)} falls between columns. The nearest is "
                     f"{ch_txt(near['chainage'])}, {dist:.3f} m away"
                     + (f" (the other side, {ch_txt(other['chainage'])}, is {abs(other['chainage'] - ch):.3f} m away)" if other and other is not near else "")
                     + ". Rule: take the nearest column, no interpolation.")
        steps.append(f"Column {ch_txt(near['chainage'])} reads: ground level {f3(v['ground_level'])}, proposed FL {f3(v['prop_fl'])}, "
                     f"cut(-)/fill(+) {f3(cut)}, proposed RL {f3(v['prop_rl'])}.")
        steps.append(f"Check: FL - GL = {f3(v['prop_fl'])} - {f3(v['ground_level'])} = {f3(v['prop_fl'] - v['ground_level'])}, "
                     f"matching the cut/fill row; RL - FL = {f3(v['prop_rl'] - v['prop_fl'])}, the 0.762 m of note 5.")
        steps.append(f"The cut/fill value is {'positive, so the formation is on fill (embankment)' if cut > 0 else 'negative, so the formation is in cutting' if cut < 0 else 'zero, so the formation is at ground level'}.")
        verdict = ""
        if req is not None:
            diff = round(fl - req, 3)
            steps.append(f"FL - MIN FL REQ. = {fl} - {req} = {diff:+.3f} m, so the FL is {'at or above' if diff >= 0 else 'below'} the minimum required.")
            verdict = (f" {flag(b)}" if flag(b) else f" Its FL {fl} is {diff:.3f} m above the required {req}.")
        final = (f"At {name_of(b)} (CH {g(ch)}) the nearest data-band column is {ch_txt(near['chainage'])} ({dist:.3f} m away): "
                 f"ground level {f3(v['ground_level'])} m and {'fill' if cut > 0 else 'cut'} of {abs(cut):.3f} m." + verdict)
        q = rng.choice(["What is the ground level and the cut or fill at {n}, and is its formation level high enough?",
                        "Using the callout and the data bands, give the ground level, cut/fill and FL check for {n}.",
                        "Is {n} on fill or in cutting, by how much, and does its FL meet the minimum required?"]).format(n=name_of(b))
        rows.append(finish(row(sheet, "reason_bridge_bands", [(p1, s1), (p2, s2)], q), answer(steps, final)))
    return rows


def r_level_check(page, ann, sheet, blocked):
    """One bridge crop: FL vs MIN FL REQ. and free board = FL - HFL."""
    rows = []
    for b in ann["bridges"]:
        lv = b.get("lsection_levels") or {}
        fl, req = lv.get("proposed_formation_level"), lv.get("min_formation_level_required")
        if not b["complete"] or b["bridge_id"] in blocked or fl is None or req is None or "lsection_callout" not in b:
            continue
        core = union([b["lsection_callout"]["bbox"], lv["bbox"]])
        (p, s), origin, size = crop_around(page, core, f"s{sheet}_rlv{b['bridge_id']}")
        if not fully_in(lv["bbox"], origin, size):
            continue
        diff = round(fl - req, 3)
        steps = [f"The level block of {name_of(b)} gives FL {fl} and MIN FL REQ. {req}.",
                 f"FL - MIN FL REQ. = {fl} - {req} = {diff:+.3f} m."]
        hfl, fb = lv.get("high_flood_level"), lv.get("free_board")
        if hfl is not None and fb is not None:
            steps.append(f"It also gives HFL {hfl} and FB {fb}. On these sheets FB = FL - HFL: {fl} - {hfl} = {f3(fl - hfl)}, which agrees with the printed FB.")
        steps.append("Below the minimum, so it must be flagged." if diff < 0 else "At or above the minimum, so no flag.")
        final = (flag(b) if diff < 0 else f"Yes. {name_of(b).capitalize()} has FL {fl}, {diff:.3f} m above MIN FL REQ. {req}.")
        if hfl is not None and fb is not None:
            final += f" Free board {fb} m (FL {fl} - HFL {hfl})."
        q = rng.choice(["Check the formation level of {n} against the minimum required, and its free board if shown.",
                        "Is the proposed FL of {n} adequate? Explain from the level block.",
                        "Does {n} meet MIN FL REQ.? Work it out."]).format(n=name_of(b))
        rows.append(finish(row(sheet, "reason_level_check", [(p, s)], q), answer(steps, final)))
    return rows


def r_bands(page, ann, sheet, blocked_ch, n_each=4):
    """Band crop only: cut/fill at an off-column chainage, extremes over a range, consistency, not in crop."""
    rows = []
    cols = [c for c in ann["bands"]["columns"] if c["checks_ok"] and c["chainage"] not in blocked_ch]
    if len(cols) < 20:
        return rows
    names = {"ground_level": "ground level", "cut_fill": "fill (or cut)", "track_distance": "track distance to the existing UP line",
             "prop_fl": "proposed FL"}
    for k in range(n_each):
        # cut/fill at an arbitrary chainage between columns
        idx = rng.randrange(1, len(cols) - 1)
        ch = round(cols[idx]["chainage"] + rng.uniform(-9.9, 9.9), 1)
        near = min(cols, key=lambda c: abs(c["chainage"] - ch))
        win = band_window(cols, cols.index(near))
        p, s = band_image(page, ann, win, f"s{sheet}_rbq{k}_{near['chainage']:.0f}")
        v = near["values"]
        steps = [f"The crop shows data-band columns from {ch_txt(win[0]['chainage'])} to {ch_txt(win[-1]['chainage'])}, one every 20 m.",
                 f"CH {g(ch)} is not a printed column; the nearest is {ch_txt(near['chainage'])}, {abs(near['chainage'] - ch):.1f} m away. Rule: nearest column, no interpolation.",
                 f"Column {ch_txt(near['chainage'])}: cut(-)/fill(+) {f3(v['cut_fill'])}, FL {f3(v['prop_fl'])}, ground level {f3(v['ground_level'])}.",
                 f"Check: FL - GL = {f3(v['prop_fl'] - v['ground_level'])}, matching the cut/fill row (its label says FL - GL; minus is cut, plus is fill)."]
        final = (f"At CH {g(ch)} (nearest column {ch_txt(near['chainage'])}) the formation is "
                 f"{'on fill of' if v['cut_fill'] > 0 else 'in cut of' if v['cut_fill'] < 0 else 'at ground level,'} {abs(v['cut_fill']):.3f} m.")
        rows.append(finish(row(sheet, "reason_band_cutfill", [(p, s)],
                               rng.choice(["Is the formation in cut or fill at CH {c}, and by how much?",
                                           "How much cut or fill is there at chainage {c}?"]).format(c=f"{g(ch)}")),
                           answer(steps, final)))
        # extreme over the crop
        key = rng.choice(list(names))
        want = rng.choice(["highest", "lowest"])
        vals = [(c["chainage"], c["values"][key]) for c in win]
        best = (max if want == "highest" else min)(vals, key=lambda t: t[1])
        listing = ", ".join(f"{ch_txt(c)}: {f3(x)}" for c, x in vals)
        steps = [f"The crop shows columns {ch_txt(win[0]['chainage'])} to {ch_txt(win[-1]['chainage'])}.",
                 f"The {names[key]} row reads, column by column: {listing}.",
                 f"The {want} value is {f3(best[1])}, at {ch_txt(best[0])}."]
        rows.append(finish(row(sheet, "reason_band_extreme", [(p, s)],
                               f"Between CH {ch_txt(win[0]['chainage'])} and {ch_txt(win[-1]['chainage'])}, where is the {names[key]} {want}, and what is it?"),
                           answer(steps, f"The {names[key]} is {want} at CH {ch_txt(best[0])}: {f3(best[1])}.")))
        # consistency of one column
        c = rng.choice(win)
        v = c["values"]
        steps = [f"Column {ch_txt(c['chainage'])} reads: cut/fill {f3(v['cut_fill'])}, FL difference {f3(v['fl_difference'])}, RL {f3(v['prop_rl'])}, "
                 f"FL {f3(v['prop_fl'])}, existing UP line FL {f3(v['exg_up_fl'])}, ground level {f3(v['ground_level'])}.",
                 f"RL - FL = {f3(v['prop_rl'])} - {f3(v['prop_fl'])} = {f3(v['prop_rl'] - v['prop_fl'])}; note 5 puts rail level 762 mm above formation. Agrees.",
                 f"FL - GL = {f3(v['prop_fl'])} - {f3(v['ground_level'])} = {f3(v['prop_fl'] - v['ground_level'])}, against cut/fill {f3(v['cut_fill'])}. Agrees.",
                 f"FL - existing UP line FL = {f3(v['prop_fl'])} - {f3(v['exg_up_fl'])} = {f3(v['prop_fl'] - v['exg_up_fl'])}, against the difference row {f3(v['fl_difference'])}. Agrees."]
        rows.append(finish(row(sheet, "reason_band_check", [(p, s)],
                               f"Are the data-band values at CH {ch_txt(c['chainage'])} consistent with each other and with the notes?"),
                           answer(steps, f"Yes. At CH {ch_txt(c['chainage'])}, RL - FL = 0.762 (note 5), FL - GL equals the cut/fill and FL - existing FL equals the difference row.")))
    # a chainage that is not in the crop
    for k in range(2):
        idx = rng.randrange(len(cols))
        win = band_window(cols, idx)
        outside = [c for c in cols if c["chainage"] < win[0]["chainage"] - 100 or c["chainage"] > win[-1]["chainage"] + 100]
        if not outside:
            continue
        c = rng.choice(outside)
        p, s = band_image(page, ann, win, f"s{sheet}_rbabs{k}_{win[0]['chainage']:.0f}")
        steps = [f"The crop's chainage row runs from {ch_txt(win[0]['chainage'])} to {ch_txt(win[-1]['chainage'])}.",
                 f"CH {ch_txt(c['chainage'])} is outside that range, so its column is not in this image.",
                 "The value cannot be read here, and it should not be guessed."]
        rows.append(finish(row(sheet, "reason_band_absent", [(p, s)], f"What is the ground level at CH {ch_txt(c['chainage'])}?"),
                           answer(steps, f"CH {ch_txt(c['chainage'])} is not in this crop, which covers {ch_txt(win[0]['chainage'])} to {ch_txt(win[-1]['chainage'])}. I can't read it from this image.")))
    return rows


def r_curve_position(page, ann, sheet):
    """L-section crop with the transition-point labels and curve box: is CH X on a curve, transition or straight?"""
    rows = []
    x_of = chainage_to_x(ann)
    pts = sorted({(t["type"], t["chainage_m"]): t for t in ann["transition_points"] if t["view"] == "L-section"}.values(),
                 key=lambda t: t["chainage_m"])
    seqs = [pts[i:i + 4] for i in range(len(pts) - 3) if [t["type"] for t in pts[i:i + 4]] == ["ST", "TC", "CT", "TS"]]
    prof = ann["regions"]["lsection_profile"]
    for q4 in seqs:
        st, tc, ct, ts = (t["chainage_m"] for t in q4)
        for zone, ch in rng.sample([("straight before", st - rng.uniform(15, 60)), ("entry transition", rng.uniform(st, tc)),
                                    ("circular curve", rng.uniform(tc, ct)), ("exit transition", rng.uniform(ct, ts)),
                                    ("straight after", ts + rng.uniform(15, 60))], 2):
            ch = round(ch, 1)
            x = x_of(ch)
            if x is None:
                continue
            clip = [x - 242, prof[3] - 200, x + 242, prof[3] + 284]
            img, origin = D.render(page, clip, (1008, 1008))
            vis = [t for t in pts if fully_in(t["bbox"], origin, img.size)]
            if not all(t in vis for t in q4):
                continue
            p, s = save(img, f"s{sheet}_rcv{ch:.0f}")
            seen = ", ".join(f"{t['type']} at {g(t['chainage_m'])}" for t in vis)
            before = [t for t in vis if t["chainage_m"] <= ch]
            after = [t for t in vis if t["chainage_m"] > ch]
            steps = [f"The transition-point labels visible in the crop are: {seen}.",
                     "On these sheets a curve runs ST -> TC -> CT -> TS: ST (TTP1) straight to transition, TC (CTP1) transition to circular, "
                     "CT (CTP2) circular to transition, TS (TTP2) transition to straight.",
                     f"CH {g(ch)} lies after {before[-1]['type']} at {g(before[-1]['chainage_m'])}" if before else f"CH {g(ch)} lies before the first point, {after[0]['type']} at {g(after[0]['chainage_m'])}"]
            if before and after:
                steps[-1] += f" and before {after[0]['type']} at {g(after[0]['chainage_m'])}."
            else:
                steps[-1] += "."
            curve = next((c for c in ann["curves"] if c["location"] == "lsection_band" and fully_in(c["bbox"], origin, img.size)), None)
            meaning = {"straight before": "on the straight, before the curve starts", "entry transition": "on the entry transition (between ST/TTP1 and TC/CTP1)",
                       "circular curve": "on the circular curve (between TC/CTP1 and CT/CTP2)", "exit transition": "on the exit transition (between CT/CTP2 and TS/TTP2)",
                       "straight after": "on the straight, after the curve ends"}[zone]
            final = f"CH {g(ch)} is {meaning}."
            if curve and zone not in ("straight before", "straight after"):
                cp = curve["params"]
                steps.append(f"The curve box in the crop is curve No. {curve['curve_no']} ({curve['hand'] or 'hand not printed'}), R {cp.get('radius')}, "
                             f"TRL {cp.get('transition_length')}, CCL {cp.get('circular_curve_length')}; TC - ST = {tc - st:.3f} m and CT - TC = {ct - tc:.3f} m.")
                final += f" The curve is No. {curve['curve_no']}, radius {cp.get('radius')}."
            rows.append(finish(row(sheet, "reason_curve_position", [(p, s)],
                                   rng.choice(["Is CH {c} on a curve, a transition or a straight?",
                                               "Where does CH {c} sit relative to the curve in this crop?"]).format(c=f"{g(ch)}")),
                               answer(steps, final)))
    return rows


def r_curve_verify(page, ann, sheet):
    """Curve box: check Shift, TTL and Degree against the curve formulas."""
    rows = []
    for c in ann["curves"]:
        if c["location"] != "lsection_band":
            continue
        p_ = c["params"]
        try:
            R, L = float(p_["radius"][:-1]), float(p_["transition_length"][:-1])
            d = [float(x) for x in re.findall(r"[\d.]+", p_["deflection_angle"])[:3]] + [0, 0]
            delta = d[0] + d[1] / 60 + d[2] / 3600
            S = L * L / (24 * R)
            T = (R + S) * math.tan(math.radians(delta / 2)) + L / 2
            deg = 1750 / R
            ok = (abs(S - float(p_["shift"][:-1])) < 0.0015 and abs(T - float(p_["total_tangent_length"][:-1])) < 0.02
                  and abs(deg - float(p_["degree"])) < 0.0015)
        except (KeyError, ValueError):
            continue
        if not ok:                       # large-angle curves where the simple TTL formula is not exact: skip
            continue
        (p, s), origin, size = crop_around(page, c["bbox"], f"s{sheet}_rcvf{c['curve_no']}")
        if not fully_in(c["bbox"], origin, size):
            continue
        steps = [f"Curve No. {c['curve_no']}: R = {p_['radius']}, TRL = {p_['transition_length']}, Δ = {p_['deflection_angle']}, "
                 f"printed Shift {p_['shift']}, TTL {p_['total_tangent_length']}, Degree {p_['degree']}.",
                 f"Shift = TRL² / 24R = {g(L)}² / (24 × {g(R)}) = {S:.3f} m.",
                 f"Δ = {delta:.4f}°, so TTL = (R + Shift)·tan(Δ/2) + TRL/2 = ({g(R)} + {S:.3f}) × tan({delta / 2:.4f}°) + {g(L / 2)} = {T:.3f} m.",
                 f"Degree = 1750 / R = 1750 / {g(R)} = {deg:.3f}.",
                 "All three match the printed values."]
        rows.append(finish(row(sheet, "reason_curve_verify", [(p, s)],
                               rng.choice(["Verify the shift, total tangent length and degree printed for this curve.",
                                           f"Are the Shift, TTL and Degree of curve No. {c['curve_no']} consistent with its radius, transition length and Δ?"])),
                           answer(steps, f"Yes. For curve No. {c['curve_no']}, Shift {S:.3f} m, TTL {T:.3f} m and Degree {deg:.3f} all agree with the printed values.")))
    return rows


def r_gradient(page, ann, sheet):
    """Two plan grade-point symbols: the gradient between them, checked from the FLs and against the ruling gradient."""
    rows = []
    for line in ("proposed 3rd line", "existing UP line"):
        pts = sorted({g["chainage_m"]: g for g in ann["plan_grade_points"] if g["line"] == line}.values(), key=lambda g: g["chainage_m"])
        for a, b in zip(pts, pts[1:]):
            if not (a["gradient_after"] and a["gradient_after"] == b["gradient_before"] and a["fl"] is not None and b["fl"] is not None):
                continue
            dist, rise = b["chainage_m"] - a["chainage_m"], b["fl"] - a["fl"]
            m = re.match(r"1 in ([\d.]+) (\w+)", a["gradient_after"])
            if m:                                   # use only stretches whose label agrees with the two FLs
                if not rise or abs(dist / abs(rise) - float(m.group(1))) / float(m.group(1)) > 0.03:
                    continue
            elif a["gradient_after"] != "level" or abs(rise) > 0.0005:
                continue
            (p1, s1), o1, z1 = crop_around(page, a["bbox"], f"s{sheet}_rgp{a['chainage_m']:.0f}")
            (p2, s2), o2, z2 = crop_around(page, b["bbox"], f"s{sheet}_rgp{b['chainage_m']:.0f}")
            if not (fully_in(a["bbox"], o1, z1) and fully_in(b["bbox"], o2, z2)):
                continue
            ch = round(rng.uniform(a["chainage_m"] + 1, b["chainage_m"] - 1), 1)
            sysname = "proposed chainage" if line.startswith("proposed") else "existing UP line km"
            steps = [f"Image 1 is a grade change point of the {line} ({'red' if line.startswith('proposed') else 'black'} symbol), "
                     f"at {sysname} {a['chainage']}, FL {f3(a['fl'])}. The label right of its bar ({a['label_after']}) is the gradient after it: {a['gradient_after']}.",
                     f"Image 2 is the next grade point, at {b['chainage']}, FL {f3(b['fl'])}. The label left of its bar ({b['label_before']}) is the gradient before it, "
                     f"{b['gradient_before']}, so it is the same stretch.",
                     f"{g(ch)} lies between {g(a['chainage_m'])} and {g(b['chainage_m'])}, so it is on that stretch."]
            if a["gradient_after"] == "level":
                steps.append(f"Check from the levels: FL {f3(b['fl'])} - {f3(a['fl'])} = {rise:+.3f} m over {g(dist)} m, i.e. level.")
                final = f"At {g(ch)} the {line} is level (between grade points {a['chainage']} and {b['chainage']})."
            else:
                n = dist / abs(rise)
                steps.append(f"Check from the levels: FL {f3(b['fl'])} - {f3(a['fl'])} = {rise:+.3f} m over {g(dist)} m, i.e. 1 in {n:.0f} "
                             f"{'rising' if rise > 0 else 'falling'}, matching the label.")
                N = float(m.group(1))
                steps.append(f"Note 6 gives a ruling gradient of 1 in {RULING}; 1 in {g(N)} is {'flatter' if N >= RULING else 'steeper'} than that.")
                final = (f"At {g(ch)} the {line} is on a gradient of {a['gradient_after']} (between grade points {a['chainage']} and "
                         f"{b['chainage']}), {'within' if N >= RULING else 'steeper than'} the ruling gradient of 1 in {RULING}.")
            rows.append(finish(row(sheet, "reason_gradient", [(p1, s1), (p2, s2)],
                                   rng.choice(["What is the gradient of the {l} at {c}? Check it against the levels.",
                                               "Using these two grade points, give the gradient of the {l} at {c} and compare it with the ruling gradient."]).format(l=line, c=f"{g(ch)}")),
                               answer(steps, final)))
    return rows


def r_free_board(page, ann, sheet, blocked):
    """Two or three bridge crops: which has the smallest free board?"""
    rows = []
    cands = [b for b in ann["bridges"] if b["complete"] and b["bridge_id"] not in blocked and "lsection_callout" in b
             and (b.get("lsection_levels") or {}).get("free_board") is not None]
    for k in range(min(2, len(cands) // 2)):
        group = rng.sample(cands, min(len(cands), rng.choice([2, 3])))
        imgs, ok = [], True
        for b in group:
            lv = b["lsection_levels"]
            (p, s), origin, size = crop_around(page, union([b["lsection_callout"]["bbox"], lv["bbox"]]), f"s{sheet}_rfb{b['bridge_id']}")
            ok &= fully_in(lv["bbox"], origin, size)
            imgs.append((p, s))
        if not ok:
            continue
        steps = []
        for i, b in enumerate(group, 1):
            lv = b["lsection_levels"]
            steps.append(f"Image {i}: {name_of(b)}, FL {lv['proposed_formation_level']}, HFL {lv['high_flood_level']}, FB {lv['free_board']} "
                         f"(FL - HFL = {f3(lv['proposed_formation_level'] - lv['high_flood_level'])}).")
        low = min(group, key=lambda b: b["lsection_levels"]["free_board"])
        steps.append(f"The smallest free board is {low['lsection_levels']['free_board']} m, at {name_of(low)}.")
        rows.append(finish(row(sheet, "reason_free_board", imgs,
                               f"Which of these {len(group)} bridges has the smallest free board, and how much is it?"),
                           answer(steps, f"{name_of(low).capitalize()} has the smallest free board: {low['lsection_levels']['free_board']} m.")))
    return rows


def r_ruling(page, ann, sheet):
    """Vertical schematic crop: steepest gradient shown, against the ruling gradient (note 6)."""
    rows = []
    segs = [g for g in ann["gradient_segments"] if g["line"] == "proposed 3rd line" and g["gradient"] and g["mid_chainage_m"]]
    if not segs:
        return rows
    lab = [l for l in ann["all_text"] if l["text"] == "VERTICAL" and l["bbox"][0] < ann["bands"]["label_strip"][2] + 5]
    if not lab:
        return rows
    y0, y1 = min(l["bbox"][1] for l in lab) - 22, max(l["bbox"][3] for l in lab) + 22
    ls = ann["bands"]["label_strip"]
    for k in range(2):
        seg = rng.choice(segs)
        cx = (seg["bbox"][0] + seg["bbox"][2]) / 2
        x0 = max(ls[2] + 8, cx - 240)
        window = [x0, y0, x0 + 480, y1]
        inwin = [g for g in segs if window[0] <= g["bbox"][0] and g["bbox"][2] <= window[2]]
        if not inwin:
            continue
        strip, _ = D.render(page, [ls[0] - 4, y0, ls[2] + 6, y1])
        band, _ = D.render(page, window)
        img = Image.new("RGB", (strip.width + band.width, max(strip.height, band.height)), "white")
        img.paste(strip, (0, 0))
        img.paste(band, (strip.width, 0))
        p, s = save(D.pad28(img), f"s{sheet}_rrul{k}_{x0:.0f}")
        def n_of(g):
            m = re.match(r"1 in ([\d.]+)", g["gradient"])
            return float(m.group(1)) if m else math.inf
        listing = "; ".join(f"{g['gradient_label']} ({g['percent']}) = {g['gradient']}" for g in sorted(inwin, key=lambda g: g["bbox"][0]))
        steep = min(inwin, key=n_of)
        steps = [f"The proposed 3rd line vertical schematic in the crop shows: {listing}.",
                 "The steepest is the one with the smallest N in 1 in N" + ("" if n_of(steep) < math.inf else "; here every stretch is level") + ".",
                 f"Note 6 gives the ruling gradient of this section as 1 in {RULING}."]
        if n_of(steep) < math.inf:
            steps.append(f"The steepest here is {steep['gradient']}; {g(n_of(steep))} {'>=' if n_of(steep) >= RULING else '<'} {RULING}, so it is "
                         f"{'flatter than or equal to' if n_of(steep) >= RULING else 'steeper than'} the ruling gradient.")
            final = (f"The steepest gradient in this crop is {steep['gradient']} ({steep['percent']}), "
                     f"{'within' if n_of(steep) >= RULING else 'steeper than'} the ruling gradient of 1 in {RULING}.")
        else:
            final = f"Every stretch in this crop is level, well within the ruling gradient of 1 in {RULING}."
        rows.append(finish(row(sheet, "reason_ruling_gradient", [(p, s)],
                               "What is the steepest gradient of the proposed 3rd line in this crop, and is it within the ruling gradient?"),
                           answer(steps, final)))
    return rows


# ---------------------------------------------------------------- main

def main():
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(ANN.glob("sheet_*.json"))]
    split_of = {a["sheet_info"]["sheet_no"]: D.SPLITS.get(a["sheet_info"]["sheet_no"], "train") for a in anns}
    held = {b["bridge_id"] for a in anns if split_of[a["sheet_info"]["sheet_no"]] != "train" for b in a["bridges"]}
    held_ch = {c["chainage"] for a in anns if split_of[a["sheet_info"]["sheet_no"]] != "train" for c in a["bands"]["columns"]}
    out = {"train": [], "val": [], "test": []}
    pdfs = {}
    for a in anns:
        sheet = a["sheet_info"]["sheet_no"]
        split = split_of[sheet]
        page = pdfs.setdefault(a["source_pdf"], pymupdf.open(ROOT / a["source_pdf"]))[a["page_index"]]
        bl, blc = (held, held_ch) if split == "train" else (set(), set())
        rows = (r_bridge_bands(page, a, sheet, bl, blc) + r_level_check(page, a, sheet, bl) + r_bands(page, a, sheet, blc)
                + r_curve_position(page, a, sheet) + r_curve_verify(page, a, sheet) + r_gradient(page, a, sheet)
                + r_free_board(page, a, sheet, bl) + r_ruling(page, a, sheet))
        for i, r in enumerate(rows):
            r["id"], r["split"] = f"r{sheet}_{i:04d}", split
        out[split] += rows
        print(f"sheet {sheet:>3} ({split:<5}) {len(rows):>4} reasoning rows")
    stats = {}
    for split, rows in out.items():
        with open(OUT / f"reasoning_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "sheet", "split", "task", "image", "images", "sizes", "messages")},
                                   ensure_ascii=False) + "\n")
        stats[split] = {"rows": len(rows), "tasks": dict(sorted(Counter(r["task"] for r in rows).items()))}
    (OUT / "reasoning_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
