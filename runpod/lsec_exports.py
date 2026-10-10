"""The four lists of an L-section PDF in the column layouts of the Sample templates (Downloads/Sample/*.xlsx), as CSV
files Excel opens:

    Bridge_List.csv         S.No | Bridge No. | Bridge Type | Approx. Chainage (km) | Proposed Span | Configration |
                            Bridge Type / Description | Bridge Detail | Bed Level | HFL | Clearance
    Graidient_excel.csv     S.No. | Start Station | End Station | Start Elevation | End Elevation | Grade | Gradient | Direction
    Prop_Curve_List.csv     Sr. NO. | Delta | Radius | Degree | TPTC-1 | TPCC-1 | TPCC-2 | TPTC-2 | Transition Length (m) |
                            Circular Curve Length (m) | Total Circular Length | Tangent Length (m) | Cant (Ca) | Speed Potential (KMPH)
    Station_Excel.csv       S.NO. | Name of station | Location (km) | Class of station | Intermediate distance

    files = write_all(anns, out_dir, name)    # anns: the PDF's sheet annotations (ask_lsec.read_pdf / scripts/annotate_v5.py)

Values are the sheets' own (each sheet's reading, the same as the model's facts); a sheet repeats a strip of its
neighbours, so an item printed on two sheets is listed once. Nothing the sheets do not print is filled in (the class
of a station, a bridge's HFL when its table row has none ...).
"""
import csv
import re
from pathlib import Path

BOM = chr(0xFEFF)


def km3(m):
    return f"{m / 1000:.3f}" if m is not None else ""


def station(m):
    """1030302.828 -> '1030+302.83m' (the template's station style)."""
    return f"{int(m // 1000)}+{m % 1000:06.2f}m" if m is not None else ""


def ch_m(s):
    if s is None:
        return None
    s = re.sub(r"\s", "", str(s))
    m = re.fullmatch(r"(\d+)\+(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 1000 + float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def deg(text):
    """"18° 30' 26\"" -> 18.51 (decimal degrees, as the template's Delta)."""
    m = re.search(r"(\d+)\s*°\s*(?:(\d+)\s*['’′])?\s*(?:(\d+(?:\.\d+)?)\s*(?:\"|''|”|″))?", text or "")
    if not m:
        return ""
    return f"{int(m.group(1)) + int(m.group(2) or 0) / 60 + float(m.group(3) or 0) / 3600:.2f}"


def kv(text, *names):
    for n in names:
        m = re.search(rf"\b{n}\s*[:=]\s*(-?\d+(?:\.\d+)?)", text or "", re.I)
        if m:
            return m.group(1)
    return ""


def bridges(anns):
    rows, seen = [], set()
    for a in anns:
        v5 = a.get("v5") or {}
        table = {re.sub(r"\s", "", r["bridge"]).upper(): r["cells"] for r in v5.get("bridge_table", [])}
        for b in v5.get("bridges", []):
            if b.get("callout_elsewhere"):
                continue
            f, num = b["fields"], b["num"]
            key = re.sub(r"(UP|DN)$", "", num)
            if key in seen:
                continue
            seen.add(key)
            t = table.get(num) or table.get(key) or {}
            st = (f.get("prop_structure") or f.get("exg_structure") or "").replace(" ", "")
            span = re.sub(r"\s", "", f.get("prop_configuration") or f.get("exg_configuration") or "").upper().rstrip("X")
            cat = f.get("prop_type") or ""
            ch = ch_m(f.get("_prop_chainage") or f.get("chainage"))
            # bed level and HFL: the DETAILS OF BRIDGES table, else the bridge's level block (BED LEVEL / B.L, the higher
            # of OHFL / CHFL / HFL)
            lv = {re.sub(r"[^A-Z]", "", x["label"].upper()): x["value"] for x in ((b.get("level_block") or {}).get("values") or [])}
            hfls = [x for x in (t.get("chfl"), t.get("ohfl")) if x and re.fullmatch(r"-?\d+(?:\.\d+)?", x)] or \
                   [v for k, v in lv.items() if k.endswith("HFL")]
            hfl = max((float(x) for x in hfls), default=None)
            bed = t.get("bed") or next((v for k, v in lv.items() if k in ("BL", "BEDLEVEL", "BEDLVL")), "")
            rows.append([f"BRNo-{f.get('br_no') or num}".replace(" ", ""), st, km3(ch), span, cat, st,
                         f"CH-{km3(ch)},BRNo-{(f.get('br_no') or num).replace(' ', '')},{span}{cat}", bed,
                         f"{hfl:g}" if hfl is not None else "", t.get("clearance_pr", "")])
        for c in v5.get("crossings", []):
            key = re.sub(r"[\s-]", "", c["label"]).upper()
            if key in seen:
                continue
            seen.add(key)
            m = re.search(r"AT\s*CH\.?\s*:?\s*(\d[\d+.]*)", c["text"]) or re.search(r"\bCH\s*:\s*(\d[\d+.]*)", c["text"])
            ch = ch_m(m.group(1)) if m else None
            what = "ROB" if c["kind"] == "ROB" else "RUB" if c["kind"] == "RUB" else "LC"
            rows.append([c["label"], what, km3(ch), "", what, what, f"CH-{km3(ch)},{c['label']},{what}", "", "", ""])
    rows.sort(key=lambda r: float(r[2]) if r[2] else 1e12)
    return [["S.No", "Bridge No.", "Bridge Type", "Approx. Chainage (km)", "Proposed Span", "Configration", "Bridge Type / Description",
             "Bridge Detail", "Bed Level", "HFL", "Clearance"]] + [[i + 1] + r for i, r in enumerate(rows)]


def gradients(anns):
    """Between consecutive grade points of the proposed line: the printed gradient after a point (else the one the
    two FLs give, said 'computed' in no column - the template has none - so only printed ones are written)."""
    pts = {}
    for a in anns:
        for p in a.get("plan_grade_points") or []:                 # (the proposed line's: the existing line's are its own)
            if p.get("chainage_m") is not None and p.get("fl") is not None and str(p.get("line") or "proposed").lower().startswith("prop"):
                pts.setdefault(round(p["chainage_m"], 2), p)
        for p in a.get("grade_points") or []:
            if p.get("chainage_m") is not None and p.get("fl") is not None:
                pts.setdefault(round(p["chainage_m"], 2), {"chainage_m": p["chainage_m"], "fl": p["fl"]})
    seq = [pts[k] for k in sorted(pts)]
    rows = []
    for p, q in zip(seq, seq[1:]):
        g = p.get("gradient_after") or ""
        m = re.match(r"1 in ([\d.]+) (\w+)", g)
        if m:
            n, d = float(m.group(1)), m.group(2).lower()
            grade, grad, direction = f"{'-' if d.startswith('fall') else ''}{n:.2f}:1", f"1:{n:g}", "Fall" if d.startswith("fall") else "Rise"
        elif g.lower().startswith("level") or abs(q["fl"] - p["fl"]) < 1e-6:
            grade, grad, direction = "Horizontal", "Horizontal", "Level"
        else:
            run = q["chainage_m"] - p["chainage_m"]
            rise = q["fl"] - p["fl"]
            n = abs(run / rise)
            grade, grad, direction = f"{'-' if rise < 0 else ''}{n:.2f}:1", f"1:{n:.0f}", "Fall" if rise < 0 else "Rise"
        rows.append([station(p["chainage_m"]), station(q["chainage_m"]), f"{p['fl']:.3f}m", f"{q['fl']:.3f}m", grade, grad, direction])
    return [["S.No.", "Start Station", "End Station", "Start Elevation", "End Elevation", "Grade ", "Gradient", "Direction"]] + \
           [[i + 1] + r for i, r in enumerate(rows)]


def curves(anns, track=None):
    """The proposed line's curves (the sheet's own line when two lines' curves are printed), once each."""
    seen, rows = set(), []
    for a in anns:
        v5 = a.get("v5") or {}
        main = track or v5.get("main_track")
        for c in v5.get("curves", []):
            if c.get("track") and main and c["track"] != main:
                continue
            p = c.get("points") or {}
            st, tc = p.get("ST1") or p.get("TPTC1") or p.get("TS1"), p.get("TC") or p.get("TPCC1") or p.get("SC")
            ct, ts = p.get("CT") or p.get("TPCC2") or p.get("CS"), p.get("TS2") or p.get("TPTC2") or p.get("ST2")
            key = (round(st or 0, 1), round(ts or 0, 1))
            if key in seen or st is None or ts is None:
                continue
            seen.add(key)
            d = " ".join(c.get("details") or [])
            delta = deg(re.search(r"(?:Δ|∆|DELTA)\s*[:=]\s*([^A-Za-z]+)", d).group(1)) if re.search(r"(?:Δ|∆|DELTA)\s*[:=]", d) else ""
            radius = kv(d, "R")
            rows.append([delta, f"{float(radius):.3f}m" if radius else "", c.get("degree") or kv(d, "Degree"), station(st), station(tc), station(ct), station(ts),
                         kv(d, "TRL", "LTC", "LS"), f"{ct - tc:.2f}" if tc is not None and ct is not None else kv(d, "CCL"),
                         f"{ts - st:.2f}", kv(d, "TTL", "TL"), kv(d, "Ca", "CA"), kv(d, "Vmax", "V MAX", "V", "Proposed Speed")])
    rows.sort(key=lambda r: ch_m(r[3].rstrip("m")) or 0)
    return [["Prop.Curve List"], ["Sr. NO.", "Delta", "Radius", "Degree", "TPTC-1", "TPCC-1", "TPCC-2", "TPTC-2", "Transition Length (m)",
                                  "Circular Curve Length (m)", "Total Circular Length", "Tangent Length (m)", "Cant (Ca)", "Speed Potential (KMPH)"]] + \
           [[i + 1] + r for i, r in enumerate(rows)]


STATION_AT = re.compile(r"^([A-Z][A-Z .()-]{2,32}?)\s*:\s*(\d{6,7}(?:\.\d+)?)$")       # "KESHORAI PATAN: 934351.983"
NOT_A_STATION = re.compile(r"\bCH\b|ELEV|\bFL\b|\bRL\b|\bKM\b|LEVEL|TBM|\bBM\b|DATUM|EASTING|NORTHING|MSL|\bGP\b|PRO\.?$", re.I)


def stations(anns):
    """The stations printed along the line - 'NAME: chainage' at each one - in chainage order; a station named only
    as 'AMLI STATION' (no chainage printed) comes last, its location empty. (NAGDA STN / MATHURA STN at the ends of
    every sheet are the directions the line runs to, not stations on it.)"""
    found = {}
    for a in anns:
        for t in a.get("all_text") or []:
            s = re.sub(r"\s+", " ", t["text"]).strip()
            m = STATION_AT.match(s)
            if m and not NOT_A_STATION.search(m.group(1)):
                found.setdefault(m.group(1).strip(" .-"), float(m.group(2)))
                continue
            m = re.fullmatch(r"([A-Z][A-Z .]{2,30}?)\s+STATION", s)
            if m and not re.search(r"BUILDING|RAILWAY|PLATFORM|EXISTING|NEW", m.group(1)):
                found.setdefault(m.group(1).strip(), None)
    # a name printed both ways ('AMLI' and 'AMLI STATION'): once
    order = sorted(found.items(), key=lambda t: (t[1] is None, t[1] or 0))
    # several stations printed with one and the same chainage (MKN_PnP_1061-1098: every station ":858693") - a
    # drawing error: their locations are kept as printed, the distances between them left empty, and it is said
    from collections import Counter
    same = {ch for ch, n in Counter(ch for _, ch in order if ch).items() if n > 1}
    if same:
        FINDINGS.append("stations " + ", ".join(n for n, ch in order if ch in same) + " are all printed at chainage "
                        + ", ".join(f"{c:g}" for c in sorted(same)) + " - a drawing error; their locations are as printed, distances left empty")
    rows, prev = [], None
    for name, ch in order:
        dist = "" if ch in same or (prev in same) else (f"{(ch - prev) / 1000:.3f}" if ch and prev else ("0" if ch and prev is None else ""))
        rows.append([name, km3(ch) if ch else "", "", dist])
        prev = ch or prev
    return [["S.NO.", "Name of station", "Location (km)", "Class of station", "Intermediate distance"]] + [[i + 1] + r for i, r in enumerate(rows)]


FINDINGS = []          # what the lists found wrong on the drawings (filled while they are made; write_all returns it)


def write_all(anns, out_dir, name):
    """{list name: (csv path, rows)} and the findings, written to out_dir."""
    FINDINGS.clear()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for fname, rows in (("Bridge_List", bridges(anns)), ("Graidient_excel", gradients(anns)), ("Prop_Curve_List", curves(anns)),
                        ("Station_Excel", stations(anns))):
        p = out / f"{name}_{fname}.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        files[fname] = (p, len(rows) - (2 if fname == "Prop_Curve_List" else 1))
    return files, list(FINDINGS)
