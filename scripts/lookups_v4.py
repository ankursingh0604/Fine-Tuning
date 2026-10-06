"""Store lookups beyond bridges, bands, curves and TBMs: gradients, grade points, transition points, km posts,
stations, sheet information, notes and a search of all printed text.

Shared by assistant_v4/tools.py (the runtime) and scripts/agent_conversations_v4.py (the training conversations),
so both return exactly the same shapes (tests/replay_agent_v4.py checks it).

    L = build([(sheet_id, annotation, line), ...])     # line: "3rd line" / "4th line" (None for non L-section pages)
    lookup(L, kind, line=..., chainage=..., ...)        # what query(kind=...) returns

Values are as printed on the drawings; nothing is computed except which segment / zone a chainage falls in.
Names of people (issue record, officers) are not returned here.
"""
import re
from collections import defaultdict

KINDS = ("gradient", "grade_points", "transition", "km_post", "stations", "sheet_info", "notes", "text")
ORDER = {"ST": 0, "TC": 1, "CT": 2, "TS": 3}
ZONE_AFTER = {"ST": "entry transition (between ST and TC)", "TC": "circular curve (between TC and CT)",
              "CT": "exit transition (between CT and TS)", "TS": "straight (after TS)"}


def r3(v):
    return round(v, 3) if isinstance(v, float) else v


def build(sheets):
    L = {"gp": defaultdict(dict), "tp": defaultdict(dict), "km": defaultdict(dict), "stations": defaultdict(dict),
         "info": {}, "notes": {}, "text": [], "line": {}}
    for sid, a, line in sorted(sheets, key=lambda t: t[0]):
        info = a.get("sheet_info") or {}
        rec = info.get("issue_record") or []
        L["info"][sid] = {"sheet_id": sid, "line": line, "sheet_no": info.get("sheet_no"), "drawing_no": info.get("drawing_no"),
                          "title": info.get("title"), "chainage_from": info.get("chainage_from"), "chainage_to": info.get("chainage_to"),
                          "scale": info.get("scale"), "date": info.get("date"),
                          "revision": ((rec[-1] or {}).get("rev") if rec else None) or "R0",
                          "revision_date": (rec[-1] or {}).get("date") if rec else None,
                          "previous_sheet": info.get("previous_sheet"), "next_sheet": info.get("next_sheet"),
                          "gauge": info.get("gauge"), "standard_of_construction": info.get("standard_of_construction"),
                          "year_of_survey": info.get("year_of_survey"),
                          "reference_drawings": [d["drawing"] for d in a.get("reference_drawings") or []]}
        L["notes"][sid] = [{"no": n["no"], "text": n["text"]} for n in a.get("notes") or []]
        L["line"][sid] = line
        regions = a.get("regions") or {}
        for t in a.get("all_text") or []:
            x, y = t["bbox"][0], t["bbox"][1]
            area = next((name.replace("_", " ") for name, r in regions.items() if r[0] <= x <= r[2] and r[1] <= y <= r[3]), "drawing")
            L["text"].append((sid, t["text"], area))
        if not line:
            continue
        for g in a.get("plan_grade_points") or []:
            if str(g.get("line", "")).startswith("exist") or g.get("chainage_m") is None:
                continue
            L["gp"][line].setdefault(g["chainage_m"], {
                "chainage_m": r3(g["chainage_m"]), "fl": g.get("fl"), "gradient_before": g.get("gradient_before"),
                "gradient_after": g.get("gradient_after"), "printed_before": g.get("label_before"), "printed_after": g.get("label_after"),
                "next_point_chainage_m": g.get("next_point_chainage_m"), "sheet_id": sid})
        for t in a.get("transition_points") or []:
            if t.get("view") != "L-section" or t.get("chainage_m") is None:
                continue
            L["tp"][line].setdefault((t["type"], t["chainage_m"]), {
                "type": t["type"], "printed_as": t.get("printed_as") or t["type"], "also_called": t.get("also_called"),
                "meaning": t.get("meaning"), "chainage_m": r3(t["chainage_m"]), "sheet_id": sid})
        for k in a.get("km_marks") or []:
            if k.get("chainage_m") is not None:
                L["km"][line].setdefault(k["chainage_m"], {"km": k.get("proposed_km"), "existing_km": k.get("existing_km"),
                                                           "chainage_m": r3(k["chainage_m"]), "sheet_id": sid})
        for k in a.get("km_posts") or []:
            if k.get("chainage_m") is not None and k["chainage_m"] not in L["km"][line]:
                L["km"][line][k["chainage_m"]] = {"km": k.get("km"), "existing_km": None, "chainage_m": r3(k["chainage_m"]), "sheet_id": sid}
        for s in a.get("stations") or []:
            L["stations"][line].setdefault(s["name"], {"name": s["name"], "direction": s.get("direction"),
                                                       "details": s.get("details") or [], "sheet_id": sid})
    return L


def lookup(L, kind, line=None, chainage=None, chainage_from=None, chainage_to=None, sheet_id=None, sheet_no=None, term=None):
    if kind == "gradient" and chainage is None:
        return {"error": "give the chainage"}
    if kind == "gradient":
        gps = sorted(L["gp"].get(line, {}).values(), key=lambda g: g["chainage_m"])
        for g in gps:
            nxt = g.get("next_point_chainage_m")
            if nxt is not None and g["chainage_m"] <= chainage < nxt:
                nx = L["gp"][line].get(nxt)
                return {"line": line, "chainage": chainage, "gradient": g["gradient_after"], "printed_as": g["printed_after"],
                        "from_grade_point": {"chainage_m": g["chainage_m"], "fl": g["fl"]},
                        "to_grade_point": {"chainage_m": r3(nxt), "fl": nx["fl"] if nx else None}, "sheet_id": g["sheet_id"]}
        return {"error": f"no grade points read around CH {chainage} on the {line}"}
    if kind == "grade_points":
        lo, hi = (chainage_from, chainage_to) if chainage_from is not None else (-1e18, 1e18)
        gps = [g for g in L["gp"].get(line, {}).values() if lo <= g["chainage_m"] <= hi and (not sheet_id or g["sheet_id"] == sheet_id)]
        return [{k: g[k] for k in ("chainage_m", "fl", "gradient_before", "gradient_after", "sheet_id")}
                for g in sorted(gps, key=lambda g: g["chainage_m"])] or {"error": f"no grade points read in that range on the {line}"}
    if kind in ("gradient", "transition") and chainage is None:
        return {"error": "give the chainage"}
    if kind == "transition":
        tps = sorted(L["tp"].get(line, {}).values(), key=lambda t: (t["chainage_m"], ORDER.get(t["type"], 9)))
        if not tps:
            return {"error": f"no transition points read on the {line}"}
        before = [t for t in tps if t["chainage_m"] <= chainage]
        after = [t for t in tps if t["chainage_m"] > chainage]
        prev, nxt = (before[-1] if before else None), (after[0] if after else None)
        if prev is None and nxt is None:
            return {"error": f"no transition points around CH {chainage} on the {line}"}
        if prev is None or prev["type"] == "TS":
            zone = "straight" + (" (before the next curve starts at its ST)" if nxt else "")
        else:
            zone = ZONE_AFTER.get(prev["type"], "unknown")
        return {"line": line, "chainage": chainage, "zone": zone, "previous_point": prev, "next_point": nxt}
    if kind == "km_post":
        ks = sorted(L["km"].get(line, {}).values(), key=lambda k: k["chainage_m"])
        if not ks:
            return {"error": f"no km posts read on the {line}"}
        if chainage is None:
            return ks
        before = [k for k in ks if k["chainage_m"] <= chainage]
        after = [k for k in ks if k["chainage_m"] > chainage]
        return {"line": line, "chainage": chainage, "previous_km_post": before[-1] if before else None,
                "next_km_post": after[0] if after else None}
    if kind == "stations":
        st = sorted(L["stations"].get(line, {}).values(), key=lambda s: s["name"])
        return st or {"error": f"no stations read on the {line}"}
    if kind == "sheet_info":
        sid = sheet_id or next((s for s, i in L["info"].items() if i["line"] == line and str(i["sheet_no"]) == str(sheet_no)), None)
        return L["info"].get(sid) or {"error": f"sheet {sheet_id or sheet_no} not found"}
    if kind == "notes":
        ns = L["notes"].get(sheet_id)
        if ns is None:
            return {"error": f"sheet {sheet_id} not found"}
        if term:
            ns = [n for n in ns if term.lower() in n["text"].lower()]
            return ns or {"error": f"no note on sheet {sheet_id} mentions '{term}'"}
        return ns
    if kind == "text":
        t = (term or "").strip().lower()
        if len(t) < 2:
            return {"error": "give a word or phrase to search for"}
        hits = [{"sheet_id": s, "text": x, "area": ar} for s, x, ar in L["text"] if t in x.lower() and (not sheet_id or s == sheet_id)]
        if not hits:
            return {"error": f"'{term}' is not printed on {'sheet ' + sheet_id if sheet_id else 'any sheet read'}"}
        return {"matches": hits[:8], "total": len(hits)}
    return {"error": f"unknown kind {kind}"}
