"""Checks run on the annotations of a Plan & L-Section PDF.

Each finding is a dict:
    sheet, page_index, severity, category, item, chainage, message, bbox (PDF points, for the overlay)
Severities:
    FLAG   a design value outside its requirement (e.g. FL below MIN FL REQ.)
    CHECK  values on the drawing that disagree with each other; one of them is probably wrong
    INFO   something incomplete or worth a look, not necessarily wrong
Values the checks rely on are read from the sheet's own notes where printed (note 5: rail 762 mm above
formation; note 6: ruling gradient), with the usual value as a fallback.
"""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import annotate as A                      # noqa: E402

MIN_TRACK_CENTRES = 4.725                 # m, Indian Railways Schedule of Dimensions (BG), new works
TOL = 0.0025                              # m, rounding tolerance for 3-decimal values


def g(v):
    """A number in plain decimal: 1242662.9, 0.762."""
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, (int, float)) else str(v)


def finding(a, severity, category, item, chainage, message, bbox):
    return {"sheet": a["sheet_info"].get("sheet_no"), "page_index": a["page_index"], "severity": severity,
            "category": category, "item": item, "chainage": chainage, "message": message, "bbox": bbox}


def sheet_constants(a):
    """Rail-to-formation height (note 5) and ruling gradient (note 6), from the notes if printed."""
    text = " ".join(n["text"] for n in a.get("notes", []))
    m = re.search(r"RAIL LEVEL SHOULD BE (\d+)\s*MM ABOVE FORMATION", text)
    rail = int(m.group(1)) / 1000 if m else 0.762
    m = re.search(r"RULING GRADIENT[^.]*?1 IN (\d+)", text)
    ruling = int(m.group(1)) if m else 150
    return rail, ruling


def recognised(a):
    """None if the sheet has the expected Plan & L-Section layout, otherwise the reason it does not."""
    if not a["sheet_info"].get("sheet_no"):
        return "no title block with a sheet number was found"
    if not a.get("bands") or "error" in a["bands"]:
        return "the L-section data bands could not be read" + (f" ({a['bands']['error']})" if a.get("bands") else "")
    return None


# ---------------------------------------------------------------- bridges

def check_bridges(a, complete_elsewhere=frozenset()):
    out = []
    for b in a["bridges"]:
        if b.get("belongs_to") not in (None, "this sheet"):
            continue                       # printed past a match line: checked on its own sheet
        bid, ch = b["bridge_id"], b.get("chainage_m")
        lv = b.get("lsection_levels") or {}
        box = (lv or b.get("lsection_callout") or b.get("plan_callout") or {}).get("bbox")
        fl, req = lv.get("proposed_formation_level"), lv.get("min_formation_level_required")
        if fl is not None and req is not None and fl < req:
            out.append(finding(a, "FLAG", "FL below MIN FL REQ.", f"bridge {bid}", ch,
                               f"FLAG: For bridge {bid} min. FL = {g(req)}, and FL = {g(fl)} ({g(req - fl)} m below the minimum required).",
                               lv["bbox"]))
        hfl, fb = lv.get("high_flood_level"), lv.get("free_board")
        if fl is not None and hfl is not None and fb is not None and abs((fl - hfl) - fb) > TOL:
            out.append(finding(a, "CHECK", "Free board", f"bridge {bid}", ch,
                               f"Bridge {bid}: FB printed {g(fb)} m but FL - HFL = {g(fl)} - {g(hfl)} = {g(fl - hfl)} m.", lv["bbox"]))
        # plan and L-section callouts of the same bridge must agree
        if "plan_callout" in b and "lsection_callout" in b:
            p, l = A.parse_callout(b["plan_callout"]["text"]), A.parse_callout(b["lsection_callout"]["text"])
            diffs = [f"{k.replace('_', ' ')}: plan '{p[k]}' vs L-section '{l[k]}'" for k in
                     ("existing_type", "existing_span", "crossing", "proposal", "category", "chainage_m")
                     if p and l and p.get(k) is not None and l.get(k) is not None and p[k] != l[k]]
            if diffs:
                out.append(finding(a, "CHECK", "Plan vs L-section callout", f"bridge {bid}", ch,
                                   f"Bridge {bid}: the two callouts differ - " + "; ".join(diffs) + ".", b["plan_callout"]["bbox"]))
        prop = b.get("proposal") or ""
        if re.search(r"\d\s*X\s*[\d.]+\s*X\s*-", prop):
            out.append(finding(a, "INFO", "Incomplete callout", f"bridge {bid}", ch,
                               f"Bridge {bid}: proposal '{prop}' has no height.", box))
        elif (not b["complete"] and ("lsection_callout" in b or "plan_callout" in b)
              and b["bridge_id"] not in complete_elsewhere):    # an edge-cut copy of a bridge complete on its own sheet
            out.append(finding(a, "INFO", "Incomplete callout", f"bridge {bid}", ch,
                               f"Bridge {bid}: the callout does not give the structure, proposal and chainage "
                               f"('{(b.get('lsection_callout') or b.get('plan_callout'))['text'][:90]}').", box))
        elif lv and "lsection_callout" not in b and "plan_callout" not in b:
            out.append(finding(a, "INFO", "Level block without callout", f"bridge {bid}", ch,
                               f"Bridge {bid}: a level block is printed but no matching callout was found.", lv.get("bbox")))
    return out


# ---------------------------------------------------------------- data bands

def check_bands(a, rail):
    out = []
    cols = (a.get("bands") or {}).get("columns", [])
    blank = []                                     # consecutive columns with values missing (e.g. line ends)
    for c in cols:
        v, ch = c["values"], c["chainage"]
        need = ("cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl", "ground_level")
        if any(k not in v for k in need):
            blank.append(c)
            continue
        problems = []
        if abs(v["prop_fl"] - v["ground_level"] - v["cut_fill"]) > TOL:
            problems.append(f"FL - GL = {g(v['prop_fl'] - v['ground_level'])} but cut/fill row = {g(v['cut_fill'])}")
        if abs(v["prop_rl"] - v["prop_fl"] - rail) > TOL:
            problems.append(f"RL - FL = {g(v['prop_rl'] - v['prop_fl'])}, note 5 requires {g(rail)}")
        if abs(v["prop_fl"] - v["exg_up_fl"] - v["fl_difference"]) > TOL:
            problems.append(f"FL - existing FL = {g(v['prop_fl'] - v['exg_up_fl'])} but difference row = {g(v['fl_difference'])}")
        if problems:
            out.append(finding(a, "CHECK", "Data band arithmetic", f"CH {g(ch)}", ch, f"CH {g(ch)}: " + "; ".join(problems) + ".", c["bbox"]))
        if v["track_distance"] < MIN_TRACK_CENTRES:
            out.append(finding(a, "FLAG", "Track distance", f"CH {g(ch)}", ch,
                               f"CH {g(ch)}: track distance {g(v['track_distance'])} m is below {MIN_TRACK_CENTRES} m "
                               f"(minimum track centres for new works, IR Schedule of Dimensions).", c["bbox"]))
    if blank:
        runs, cur = [], [blank[0]]
        for c in blank[1:]:
            (cur.append(c) if cols.index(c) == cols.index(cur[-1]) + 1 else (runs.append(cur), cur := [c]))
        runs.append(cur)
        for r in runs:
            out.append(finding(a, "INFO", "Data band values not printed", f"CH {g(r[0]['chainage'])}-{g(r[-1]['chainage'])}",
                               r[0]["chainage"], f"{len(r)} band column(s) from CH {g(r[0]['chainage'])} to {g(r[-1]['chainage'])} "
                               f"have rows left blank (e.g. where the proposed or existing line ends).",
                               A.union([c["bbox"] for c in r])))
    return out


# ---------------------------------------------------------------- gradients

def one_in(label_words):
    m = re.match(r"1 in ([\d.]+)", label_words or "")
    return float(m.group(1)) if m else None


def check_gradients(a, ruling):
    out = []
    for s in a.get("gradient_segments", []):
        n = one_in(s["gradient"])
        if s["line"] == "proposed 3rd line" and n and n < ruling:
            out.append(finding(a, "FLAG", "Gradient steeper than ruling", f"gradient {s['gradient_label']}", s.get("mid_chainage_m"),
                               f"Proposed 3rd line gradient {s['gradient_label']} ({s['gradient']}) is steeper than the ruling "
                               f"gradient of 1 in {ruling} (note 6).", s["bbox"]))
    for line in ("proposed 3rd line", "existing UP line"):
        pts = sorted({p["chainage_m"]: p for p in a.get("plan_grade_points", []) if p["line"] == line}.values(), key=lambda p: p["chainage_m"])
        for p, q in zip(pts, pts[1:]):
            if p["gradient_after"] and q["gradient_before"] and p["gradient_after"] != q["gradient_before"]:
                flat = all((one_in(x) or math.inf) >= 2000 for x in (p["gradient_after"], q["gradient_before"]))
                out.append(finding(a, "INFO" if flat else "CHECK", "Gradient labels disagree", f"{line} {p['chainage']}", p["chainage_m"],
                                   f"{line.capitalize()}: the stretch from {p['chainage']} to {q['chainage']} is labelled "
                                   f"'{p['label_after']}' at one end and '{q['label_before']}' at the other.",
                                   A.union([p["bbox"], q["bbox"]])))
                continue
            n = one_in(p["gradient_after"])
            if p["fl"] is None or q["fl"] is None:
                continue
            dist, rise = q["chainage_m"] - p["chainage_m"], q["fl"] - p["fl"]
            if p["gradient_after"] == "level" and abs(rise) > 0.005:
                implied = f"1 in {dist / abs(rise):.0f} {'rising' if rise > 0 else 'falling'}"
            elif n and (abs(rise) < 1e-6 or abs(dist / abs(rise) - n) / n > 0.03
                        or (rise > 0) != p["gradient_after"].endswith("rising")):
                implied = "level" if abs(rise) < 1e-6 else f"1 in {dist / abs(rise):.0f} {'rising' if rise > 0 else 'falling'}"
            else:
                continue
            out.append(finding(a, "CHECK", "Gradient vs levels", f"{line} {p['chainage']}", p["chainage_m"],
                               f"{line.capitalize()} {p['chainage']} (FL {g(p['fl'])}) to {q['chainage']} (FL {g(q['fl'])}): "
                               f"labelled {p['gradient_after']} ({p['label_after']}) but the levels give {implied}.",
                               A.union([p["bbox"], q["bbox"]])))
            if line == "proposed 3rd line" and n and n < ruling:
                out.append(finding(a, "FLAG", "Gradient steeper than ruling", f"{line} {p['chainage']}", p["chainage_m"],
                                   f"Proposed 3rd line gradient {p['label_after']} from {p['chainage']} is steeper than 1 in {ruling} (note 6).",
                                   p["bbox"]))
    return out


# ---------------------------------------------------------------- curves

def dms(s):
    v = [float(x) for x in re.findall(r"[\d.]+", s or "")[:3]] + [0, 0, 0]
    return v[0] + v[1] / 60 + v[2] / 3600


def mval(s):
    try:
        return float(re.sub(r"[^\d.]", "", s))
    except (TypeError, ValueError):
        return None


def check_curves(a):
    out = []
    boxes = [c for c in a["curves"]]
    for c in boxes:
        p = c["params"]
        R, L, D = mval(p.get("radius")), mval(p.get("transition_length")), dms(p.get("deflection_angle"))
        if not (R and L is not None and D):
            continue
        S = L * L / (24 * R)
        T = (R + S) * math.tan(math.radians(D / 2)) + L / 2
        where = "plan" if c["location"] == "plan" else "L-section"
        if mval(p.get("shift")) is not None and abs(S - mval(p["shift"])) > 0.0015:
            out.append(finding(a, "CHECK", "Curve formula", f"curve {c['curve_no']} ({where})", None,
                               f"Curve {c['curve_no']} ({where} box): Shift printed {p['shift']}, TRL²/24R = {S:.3f} m.", c["bbox"]))
        if mval(p.get("degree")) is not None and abs(1750 / R - mval(p["degree"])) > 0.0015:
            out.append(finding(a, "CHECK", "Curve formula", f"curve {c['curve_no']} ({where})", None,
                               f"Curve {c['curve_no']} ({where} box): Degree printed {p['degree']}, 1750/R = {1750 / R:.3f}.", c["bbox"]))
        if mval(p.get("total_tangent_length")) is not None and abs(T - mval(p["total_tangent_length"])) > 0.02:
            out.append(finding(a, "INFO", "Curve formula", f"curve {c['curve_no']} ({where})", None,
                               f"Curve {c['curve_no']} ({where} box): TTL printed {p['total_tangent_length']}, "
                               f"(R + Shift)·tan(Δ/2) + TRL/2 = {T:.3f} m (difference {abs(T - mval(p['total_tangent_length'])):.3f} m; "
                               f"the simple formula is less exact for large deflection angles).", c["bbox"]))
    # the same curve's plan and L-section boxes must agree
    by_no = {}
    for c in boxes:
        by_no.setdefault(c["curve_no"], {})[c["location"]] = c
    for no, locs in by_no.items():
        if "plan" in locs and "lsection_band" in locs:
            pp, lp = locs["plan"]["params"], locs["lsection_band"]["params"]
            diffs = [f"{k.replace('_', ' ')}: plan {pp[k]} vs L-section {lp[k]}" for k in pp if k in lp and pp[k] != lp[k]]
            if diffs:
                out.append(finding(a, "CHECK", "Plan vs L-section curve box", f"curve {no}", None,
                                   f"Curve {no}: the plan and L-section boxes differ - " + "; ".join(diffs) + ".",
                                   locs["lsection_band"]["bbox"]))
    # transition points ST -> TC -> CT -> TS against the curve box lengths (TRL, CCL, TRL)
    x2c = A.x_to_chainage(a.get("bands"))
    tps = sorted({(t["type"], t["chainage_m"]): t for t in a.get("transition_points", []) if t["view"] == "L-section"}.values(),
                 key=lambda t: t["chainage_m"])
    for i in range(len(tps) - 3):
        q = tps[i:i + 4]
        if [t["type"] for t in q] != ["ST", "TC", "CT", "TS"]:
            continue
        box = next((c for c in boxes if c["location"] == "lsection_band"
                    and (x := x2c((c["bbox"][0] + c["bbox"][2]) / 2)) and q[0]["chainage_m"] - 50 <= x <= q[3]["chainage_m"] + 50), None)
        if not box:
            continue
        trl, ccl = mval(box["params"].get("transition_length")), mval(box["params"].get("circular_curve_length"))
        got = [q[1]["chainage_m"] - q[0]["chainage_m"], q[2]["chainage_m"] - q[1]["chainage_m"], q[3]["chainage_m"] - q[2]["chainage_m"]]
        if trl is None or ccl is None:
            continue
        if abs(got[0] - trl) > 0.05 or abs(got[1] - ccl) > 0.05 or abs(got[2] - trl) > 0.05:
            out.append(finding(a, "CHECK", "Curve points vs curve box", f"curve {box['curve_no']}", q[0]["chainage_m"],
                               f"Curve {box['curve_no']}: ST {g(q[0]['chainage_m'])} -> TC -> CT -> TS {g(q[3]['chainage_m'])} gives "
                               f"{got[0]:.3f} / {got[1]:.3f} / {got[2]:.3f} m, but the L-section box says TRL {box['params']['transition_length']}, "
                               f"CCL {box['params']['circular_curve_length']}.", box["bbox"]))
    return out


# ---------------------------------------------------------------- all

SEVERITY_ORDER = {"FLAG": 0, "CHECK": 1, "INFO": 2}


def check_sheet(a, complete_elsewhere=frozenset()):
    reason = recognised(a)
    if reason:
        return [finding(a, "INFO", "Layout not recognised", f"page {a['page_index'] + 1}", None,
                        f"Page {a['page_index'] + 1} was not checked: {reason}. The checker knows the Plan & L-Section "
                        f"layout of the Mathura-Nagda 3rd line drawings.", None)]
    rail, ruling = sheet_constants(a)
    return check_bridges(a, complete_elsewhere) + check_bands(a, rail) + check_gradients(a, ruling) + check_curves(a)


def check_all(anns):
    complete = frozenset(b["bridge_id"] for a in anns for b in a["bridges"] if b["complete"])
    found = [f for a in anns for f in check_sheet(a, complete)]
    return sorted(found, key=lambda f: (SEVERITY_ORDER[f["severity"]], int(f["sheet"]) if str(f["sheet"]).isdigit() else 0,
                                        f["chainage"] or 0))
