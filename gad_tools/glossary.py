"""What a label printed on a GAD means - for labels the reader does not know as a level, dimension or table row.

    meaning("LTC")                      -> ("length of the transition curve ...", "built-in")
    meaning("XYZ", web=True)            -> searched on the internet (the term only, never the drawing), kept in glossary.json

Where a meaning comes from, in this order:
  1. glossary.json, approved: an engineer wrote or confirmed it ("approved": true)  -> source "glossary"
  2. BUILT_IN below: the standard terms on these drawings                             -> source "built-in"
  3. glossary.json, not approved: found on the web earlier                           -> source "web"
  4. a web search now (web=True): the term with railway context; the result goes into glossary.json as "web", not
     approved - an engineer can correct it and set "approved": true, and from then on it is the drawing office's meaning
  5. nothing found                                                                    -> (None, "unknown")
Only the term itself is sent (e.g. "LTC railway curve abbreviation"); no drawing, value or name ever leaves the machine.
A web meaning is a hint: abbreviations are ambiguous, so answers say it is from the web and not confirmed by the drawing.
"""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

GLOSSARY = Path(__file__).with_name("glossary.json")

BUILT_IN = {
    # curve data printed on the key plans
    "LTC": "length of the transition curve (the easement between the straight and the circular curve), in metres",
    "TL": "tangent length of the curve: from the intersection point to the start / end of the curve, in metres",
    "CL": "length of the curve (along the circular arc, with its transitions), in metres",
    "CA": "cant: how much the outer rail is raised above the inner rail on the curve, in mm",
    "CD": "cant deficiency: the cant that would be needed at the maximum speed minus the cant provided, in mm",
    "S": "shift of the curve: how far the circular curve is moved inwards to fit the transition curves, in metres",
    "R": "radius of the curve, in metres",
    "D": "degree of the curve (the angle a 30.5 m chord subtends at the centre)",
    "VMAX": "maximum permissible speed on the curve / section, in km/h",
    "V": "speed (design / permissible) on the curve, in km/h",
    "APEX": "the apex of the curve: its middle point, given as a chainage",
    "T1": "the first tangent point of the curve (where the curve begins), given as a chainage",
    "T2": "the second tangent point of the curve (where the curve ends), given as a chainage",
    # levels and clearances
    "HC": "horizontal clearance: the clear width available (e.g. for the road at a level crossing / under a bridge), in metres",
    "VC": "vertical clearance: the clear height available (e.g. under the bridge / for the road), in metres",
    "FB": "free board: the height of the formation (or the structure's underside) above the flood level, in metres",
    "DATUM": "the reference level the section is drawn from (the level of its base line), in metres",
    "RL": "reduced level: the height of a point above the survey datum, in metres",
    "FL": "formation level: the level of the top of the embankment on which the ballast and track rest, in metres",
    "BL": "bed level of the stream / drain, in metres",
    "HFL": "high flood level: the highest water level the bridge is designed for, in metres",
    "OHFL": "observed high flood level: the highest flood level recorded at the site, in metres",
    "CHFL": "calculated high flood level from the hydraulic design, in metres",
    "LWL": "low water level, in metres",
    "MINFLREQ": "the minimum formation level required (from the flood level and the free board), in metres",
    "SBC": "safe bearing capacity of the soil, in t/m²",
    "TC": "track centre distance: between the centre lines of two tracks",
    "GP": "grade point: where the track gradient changes",
    "TRL": "top of rail level, in metres",
    "TTL": "top of track (rail) level, in metres",
    "CCL": "centre line of the curve / culvert (as written on the key plan)",
    "TRACKCC": "track centre-to-centre distance: between the centre lines of two adjacent tracks",
    "PFENDCH": "the chainage of the end of the station platform",
    "CLEARSPAN": "clear span: the clear opening between the inner faces of the supports (walls), in metres",
    "EFFECTIVELENGTH": "effective length: the span length used in the design (between the centres of the supports)",
    "OVERALLLENGTH": "overall length of the structure, end to end",
    "PFL": "proposed formation level: the planned level of the top of the embankment, in metres",
    "EXISTBL": "existing bed level of the stream / drain, in metres",
    "EXGBL": "existing bed level of the stream / drain, in metres",
    "TCL": "track centre line",
}


# Plan & L-section sheets use some of the same letters for other things (TTL, TRL, CCL in a curve's block): their
# meanings there, used first when the context is an L-section
LSEC = {
    "ST": "straight to transition point: where the transition curve starts (the curve begins)",
    "TS": "transition to straight point: where the transition curve ends (the curve is over)",
    "TC": "transition to circular point: where the transition curve meets the circular curve",
    "CT": "circular to transition point: where the circular curve meets the second transition curve",
    "TP1": "tangent point 1: where the curve starts (the straight meets the curve)",
    "TP2": "tangent point 2: where the curve ends (the curve meets the next straight)",
    "J1": "junction point 1: where the transition curve meets the circular curve (start of the circular part)",
    "J2": "junction point 2: where the circular curve meets the second transition curve (end of the circular part)",
    "TTP1": "tangent to transition point 1 (start of the transition curve)", "TTP2": "transition to tangent point 2 (end of the curve)",
    "CTP1": "circular to transition point", "TPTC": "tangent point / transition curve point", "TPCC": "tangent point of the circular curve",
    "GP": "grade point: where the gradient of the formation changes", "VPI": "vertical point of intersection: where two gradients meet",
    "PVI": "point of vertical intersection: where two gradients meet",
    "FL": "formation level: the level of the top of the formation (embankment / cutting) the track sits on, in metres",
    "RL": "rail level: the level of the top of the rail, in metres", "TRL": "transition length: the length of each transition curve, in metres",
    "TTL": "total tangent length of the curve, in metres", "TL": "tangent length of the curve, in metres",
    "CCL": "circular curve length: the length of the circular part of the curve, in metres",
    "TCL": "total curve length, in metres", "CA": "cant (superelevation): how much the outer rail is raised on the curve, in mm",
    "CD": "cant deficiency, in mm", "VMAX": "maximum permitted speed on the curve", "R": "radius of the curve, in metres",
    "DELTA": "deflection angle of the curve", "DEGREE": "degree of curve (sharpness: the angle for a standard chord length)",
    "SHIFT": "shift of the circular curve caused by the transition curves, in metres",
    "MSL": "mean sea level (levels are heights above it)", "TBM": "temporary bench mark: a surveyed point of known level used as reference",
    "BM": "bench mark: a surveyed point of known level", "LC": "level crossing (road crossing the track at rail level)",
    "ROB": "road over bridge (a road bridge over the railway)", "RUB": "road under bridge (the railway on a bridge over a road)",
    "FOB": "foot over bridge", "HC": "horizontal clearance, in metres", "VC": "vertical clearance, in metres",
    "HFL": "high flood level, in metres", "OHFL": "observed high flood level, in metres", "CHFL": "calculated high flood level, in metres",
    "BL": "bed level of the stream / drain under the bridge, in metres", "GL": "ground level, in metres", "OGL": "original ground level, in metres",
    "NGL": "natural ground level, in metres", "MINFLREQ": "minimum formation level required at the bridge (for clearance / free board), in metres",
    "EXG": "existing", "EX": "existing", "PROP": "proposed", "PRO": "proposed", "CH": "chainage: distance along the line, in metres (km+m)",
    "KM": "kilometre (chainage in km)", "MGBG": "magnetic bearing of the straight", "ROW": "right of way: the land width taken for the railway",
    "PSC": "prestressed concrete", "RCC": "reinforced cement concrete", "HP": "hume pipe (a round concrete pipe culvert)",
    "FT": "flat top (a flat slab culvert)", "DATUM": "the reference level the L-section's levels are drawn from",
    "VH": "vertical to horizontal scale ratio of the L-section (the vertical is exaggerated)",
    "LHS": "left hand side", "RHS": "right hand side", "STN": "station", "DN": "down line", "UP": "up line",
    "FB": "free board: height of the formation / soffit above the flood level, in metres",
    "EARTHCUSHION": "earth cushion: depth of fill over the top of a box / pipe, in metres",
}


def key(label):
    return re.sub(r"[^A-Z0-9]", "", (label or "").upper())


def _load():
    try:
        return json.loads(GLOSSARY.read_text(encoding="utf-8")) if GLOSSARY.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(g):
    try:
        GLOSSARY.write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def web_search(term, context="railway bridge drawing", timeout=8):
    """[(snippet, url)] for the term from a web search: Brave Search when BRAVE_API_KEY is set, else DuckDuckGo's HTML page.
    [] when there is no internet or nothing useful."""
    import os
    q = f'"{term}" {context} abbreviation meaning'
    try:
        if os.environ.get("BRAVE_API_KEY"):
            req = urllib.request.Request("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": 5}),
                                         headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_API_KEY"]})
            data = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
            return [(re.sub(r"<[^>]+>", "", r.get("description", "")), r.get("url", "")) for r in data.get("web", {}).get("results", [])][:4]
        # no key: Wikipedia's search API (free; good for spelled-out terms - "cant deficiency" - not for bare
        # abbreviations, so a result is kept only when it actually contains the term)
        req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": f"{term} {context}", "format": "json", "srlimit": 4}),
            headers={"User-Agent": "GAD-reader/1.0 (term lookup)"})
        data = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
        clean = lambda t: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).replace("&#039;", "'").replace("&quot;", '"').strip()
        out = []
        for r in data.get("query", {}).get("search", []):
            title, snip = clean(r["title"]), clean(r["snippet"])
            if term.lower() in (title + " " + snip).lower():
                out.append((f"{title}: {snip}", "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))))
        return out[:3]
    except Exception:                                  # noqa: BLE001 - offline, blocked, changed page: no meaning
        return []


def meaning(label, web=False, context="railway bridge drawing"):
    """(meaning text or None, source: "glossary" | "built-in" | "web" | "unknown")."""
    k = key(label)
    g = _load()
    e = g.get(k)
    if e and e.get("approved") and e.get("meaning"):
        return e["meaning"], "glossary"
    if re.search(r"l-?section|plan\s*&?\s*profile|alignment", context or "", re.I) and k in LSEC:
        return LSEC[k], "built-in"
    if k in BUILT_IN:
        return BUILT_IN[k], "built-in"
    if e and e.get("meaning"):
        return e["meaning"], "web"
    if web:
        hits = [(s, u) for s, u in web_search(label, context) if s]
        if hits:
            text = " | ".join(f"{s} ({u})" for s, u in hits[:3])[:700]
            g[k] = {"label": label, "meaning": text, "source": "web", "approved": False}
            _save(g)
            return text, "web"
    return None, "unknown"
