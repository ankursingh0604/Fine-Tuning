"""Answer questions about a checked PDF from its extracted data (no model, so every number is exact).

Understands questions about
  - a bridge          "bridge 560", "ROB-239", "LC-226"   -> details, levels, nearest band column, curve, gradient, flag
  - a chainage        "CH 1242662.9", "1242+662.9", "km 1242.66"
  - a curve           "curve 371"
  - flags and issues  "which bridges are flagged", "list the problems", "summary"
  - TBMs              "TBM BMT38", "bench marks on sheet 100"
Every answer cites the sheet and chainage it comes from.
"""
import re

from checks import g


def km_to_m(text):
    m = re.search(r"(\d{4})\+(\d+(?:\.\d+)?)", text)
    if m:
        return int(m.group(1)) * 1000 + float(m.group(2))
    m = re.search(r"\b(?:CH|chainage)\s*[:.]?\s*(\d{7}(?:\.\d+)?)", text, re.I) or re.search(r"\b(\d{7}(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\bkm\s*[:.]?\s*(\d{4}(?:\.\d+)?)", text, re.I)
    return float(m.group(1)) * 1000 if m else None


def own(anns, ch):
    """The sheet whose chainage range contains ch."""
    for a in anns:
        lo, hi = (km_to_m(a["sheet_info"].get(k) or "") for k in ("chainage_from", "chainage_to"))
        if lo and hi and lo <= ch <= hi:
            return a
    return None


def nearest_column(a, ch):
    cols = [c for c in (a.get("bands") or {}).get("columns", []) if c["checks_ok"]]
    return min(cols, key=lambda c: abs(c["chainage"] - ch)) if cols else None


def curve_position(a, ch):
    tps = sorted({(t["type"], t["chainage_m"]): t for t in a.get("transition_points", []) if t["view"] == "L-section"}.values(),
                 key=lambda t: t["chainage_m"])
    before = [t for t in tps if t["chainage_m"] <= ch]
    after = [t for t in tps if t["chainage_m"] > ch]
    if not before and not after:
        return None
    last = before[-1]["type"] if before else None
    zone = {"ST": "on the entry transition (between ST/TTP1 and TC/CTP1)", "TC": "on the circular curve (between TC/CTP1 and CT/CTP2)",
            "CT": "on the exit transition (between CT/CTP2 and TS/TTP2)", "TS": "on a straight", None: "on a straight"}[last]
    span = (f" (after {before[-1]['type']} at {g(before[-1]['chainage_m'])}" if before else " (") + \
           (f", before {after[0]['type']} at {g(after[0]['chainage_m'])})" if after else ")")
    return zone + span


def gradient_at(a, ch):
    pts = sorted({p["chainage_m"]: p for p in a.get("plan_grade_points", []) if p["line"] == "proposed 3rd line"}.values(),
                 key=lambda p: p["chainage_m"])
    for p, q in zip(pts, pts[1:]):
        if p["chainage_m"] <= ch <= q["chainage_m"] and p["gradient_after"]:
            fl = lambda v: f"{v:.3f}" if v is not None else "not printed"
            return f"{p['gradient_after']} (grade points {p['chainage']} FL {fl(p['fl'])} to {q['chainage']} FL {fl(q['fl'])})"
    return None


def band_text(c, ch):
    v = c["values"]
    side = "fill" if v["cut_fill"] > 0 else "cut" if v["cut_fill"] < 0 else "neither cut nor fill"
    return (f"nearest data-band column CH {g(c['chainage'])} ({abs(c['chainage'] - ch):.3f} m away; nearest column, no interpolation): "
            f"ground level {v['ground_level']:.3f}, proposed FL {v['prop_fl']:.3f}, RL {v['prop_rl']:.3f}, {side} {abs(v['cut_fill']):.3f} m, "
            f"existing UP line FL {v['exg_up_fl']:.3f}, track distance {v['track_distance']:.3f} m")


def find_bridge(anns, bid):
    hits = [(a, b) for a in anns for b in a["bridges"] if b["bridge_id"].upper() == bid.upper()]
    hits.sort(key=lambda h: (h[1].get("belongs_to") != "this sheet", not h[1]["complete"]))
    return hits[0] if hits else (None, None)


def answer_bridge(anns, findings, bid):
    a, b = find_bridge(anns, bid)
    if not b:
        return f"Bridge {bid} is not in this PDF."
    s = a["sheet_info"]["sheet_no"]
    ch = b.get("chainage_m")
    lines = [f"**{b['bridge_id'] if b['bridge_id'].startswith(('ROB', 'LC')) else 'Bridge ' + b['bridge_id']}** (sheet {s}"
             + (f", CH {g(ch)}" if ch and 'at CH' not in b['bridge_id'] else "") + ")"]
    ex = " ".join(x for x in (b.get("existing_type"), b.get("existing_span")) if x)
    if ex or b.get("crossing"):
        lines.append(f"- Existing: {ex or 'structure'}" + (f" over {b['crossing'].lower()}" if b.get("crossing") else "")
                     + (f" ({b['existing_on']})" if b.get("existing_on") else ""))
    if b.get("proposal"):
        lines.append(f"- Proposed: {b['proposal']}" + (f", {b['category']}" if b.get("category") else ""))
    lv = {k: v for k, v in (b.get("lsection_levels") or {}).items() if k != "bbox"}
    if lv:
        names = {"existing_formation_level": "EXG FL", "min_formation_level_required": "MIN FL REQ.", "proposed_formation_level": "FL",
                 "bed_level": "B.L", "high_flood_level": "HFL", "free_board": "FB"}
        lines.append("- Level block: " + ", ".join(f"{names.get(k, k)} {g(v)}" for k, v in lv.items()))
    for f in findings:
        if f["item"] == f"bridge {b['bridge_id']}" and f["severity"] in ("FLAG", "CHECK"):
            lines.append(f"- **{f['severity']}**: {f['message']}")
    if ch:
        c = nearest_column(a, ch) or (nearest_column(own(anns, ch), ch) if own(anns, ch) else None)
        if c:
            lines.append("- Data bands: " + band_text(c, ch))
        pos = curve_position(a, ch)
        if pos:
            lines.append(f"- Alignment: {pos}")
        grad = gradient_at(a, ch)
        if grad:
            lines.append(f"- Gradient of the proposed 3rd line: {grad}")
    return "\n".join(lines)


def answer_chainage(anns, findings, ch):
    a = own(anns, ch)
    if not a:
        rng = [f"{x['sheet_info']['chainage_from']} to {x['sheet_info']['chainage_to']}" for x in anns if x["sheet_info"].get("chainage_from")]
        return f"CH {g(ch)} is not covered by this PDF (it covers {rng[0].split(' to ')[0]} to {rng[-1].split(' to ')[1]})." if rng else "No chainage information found."
    lines = [f"**CH {g(ch)}** (sheet {a['sheet_info']['sheet_no']})"]
    c = nearest_column(a, ch)
    if c:
        lines.append("- Data bands: " + band_text(c, ch))
    pos = curve_position(a, ch)
    if pos:
        lines.append(f"- Alignment: {pos}")
    grad = gradient_at(a, ch)
    if grad:
        lines.append(f"- Gradient of the proposed 3rd line: {grad}")
    near = sorted((b for b in a["bridges"] if b.get("chainage_m") and abs(b["chainage_m"] - ch) <= 200 and b.get("belongs_to") == "this sheet"),
                  key=lambda b: abs(b["chainage_m"] - ch))
    if near:
        lines.append("- Bridges within 200 m: " + ", ".join(
            (b['bridge_id'] if b['bridge_id'].startswith('ROB at') else f"{b['bridge_id']} at CH {g(b['chainage_m'])}") + f" ({b['chainage_m'] - ch:+.1f} m)"
            for b in near))
    return "\n".join(lines)


def answer_curve(anns, no):
    hits = [(a, c) for a in anns for c in a["curves"] if c["curve_no"] == no]
    if not hits:
        return f"Curve {no} is not in this PDF."
    out = []
    for a, c in hits:
        where = "plan" if c["location"] == "plan" else "L-section"
        out.append(f"**Curve {no}** ({where} box, sheet {a['sheet_info']['sheet_no']}, {c.get('hand') or 'hand not printed'}): "
                   + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in c["params"].items()))
    return "\n\n".join(out)


def answer_list(findings, severity=None, category=None):
    sel = [f for f in findings if (not severity or f["severity"] == severity) and (not category or category in f["category"])]
    if not sel:
        return "Nothing found."
    return "\n".join(f"- [{f['severity']}] sheet {f['sheet']}: {f['message']}" for f in sel)


def answer_tbm(anns, text):
    m = re.search(r"\b(BMT?\d+)\b", text, re.I)
    rows = [(a, t) for a in anns for t in a["tbm_benchmarks"] if not m or t["tbm_id"].upper() == m.group(1).upper()]
    if not rows:
        return "No such TBM in this PDF."
    return "\n".join(f"- {t['tbm_id']} (sheet {a['sheet_info']['sheet_no']}): CH {g(t['chainage_m'])}, E {g(t['easting'])}, "
                     f"N {g(t['northing'])}, MSL {g(t['msl_m'])} - {t['description']}" for a, t in rows[:40])


HELP = ("I can answer from the extracted data of this PDF. Try:\n"
        "- **bridge 560** (or ROB-239, LC-226): details, levels, band values at the nearest column, curve, gradient, flags\n"
        "- **CH 1242662.9** or **1242+662.9**: band values, curve position, gradient, nearby bridges\n"
        "- **curve 371**\n- **which bridges are flagged** / **list all problems** / **summary**\n- **TBM BMT38**")


def answer(text, anns, findings):
    t = text.strip()
    if not anns:
        return "Upload and check a PDF first."
    m = re.search(r"\b(ROB-\d+|LC[- ]?\d+\w*|\d{3,4}[A-Z]{0,2}(?:\([\w-]+\))?)\b", t, re.I) if re.search(r"\b(bridge|br\.?|rob|lc)\b", t, re.I) else None
    if m and not re.fullmatch(r"\d{7}", m.group(1)):
        return answer_bridge(anns, findings, re.sub(r"^LC\s+", "LC-", m.group(1).upper()))
    m = re.search(r"\bcurve\s*(?:no\.?)?\s*(\w+)", t, re.I)
    if m:
        return answer_curve(anns, m.group(1))
    if re.search(r"\bTBM|bench\s*mark", t, re.I):
        return answer_tbm(anns, t)
    if re.search(r"flag", t, re.I):
        return answer_list(findings, "FLAG")
    if re.search(r"problem|issue|error|inconsisten|check", t, re.I):
        return answer_list(findings, None) if re.search(r"all|every", t, re.I) else answer_list([f for f in findings if f["severity"] != "INFO"])
    if re.search(r"summary|overview", t, re.I):
        return summary(anns, findings)
    ch = km_to_m(t)
    if ch:
        return answer_chainage(anns, findings, ch)
    return HELP


def summary(anns, findings):
    ok = [a for a in anns if not any(f["category"] == "Layout not recognised" and f["page_index"] == a["page_index"] for f in findings)]
    cnt = {s: sum(f["severity"] == s for f in findings) for s in ("FLAG", "CHECK", "INFO")}
    rng = f"{ok[0]['sheet_info']['chainage_from']} to {ok[-1]['sheet_info']['chainage_to']}" if ok else "-"
    return (f"**{len(anns)} page(s)**, {len(ok)} recognised as Plan & L-Section sheets "
            f"({', '.join(a['sheet_info']['sheet_no'] for a in ok)}), chainage {rng}. "
            f"{sum(len(a['bridges']) for a in ok)} bridge records, {sum(len(a['bands']['columns']) for a in ok)} band columns. "
            f"Findings: **{cnt['FLAG']} FLAG**, **{cnt['CHECK']} CHECK**, {cnt['INFO']} INFO.")
