"""One bridge's details in the arch_rcc.xlsx layout ("Bridge Data" sheet), as a CSV for Excel.

    "Give me bridge 224 details in csv format"  ->  <sheet>_bridge_224.csv
    "Give me the bridge details in csv"         ->  every bridge on the sheet, two columns (existing / proposed) each

The rows, labels, section headings and blank rows are the template's, in its order; column A the field, then the
existing and the proposed bridge. What an L-section prints fills its rows (GAD NAME and the arch's STONE MASONRY are not on it): the bridge number, its type, cells / span /
box height (from the configuration, "1 X 3 X 2" = 1 cell of 3 m, 2 m high), the chainage, the degree of curve (when
the bridge's chainage lies between a curve's start and end chainages: the "Degree = ..." of that curve's block), bed level, formation level,
rail level and HFL (from its level block; with OHFL and CHFL both printed, the higher of them). Fixed: RDSO NO. = RDSO/B-10152, SKEW ANGLE = 0, END DIST = 0, COLOR
BLACK / RED. The box height only when the configuration gives one; the rest (hydrology, soil, bore log, stations) is
not on an L-section and stays empty.
"""
import csv
import re

RDSO = "RDSO/B-10152"
NOTE = "Note: Fields marked with (*) are mandatory."

# the template, row by row: (label, key) - key None for a heading / blank row; the note sits in column E of row 19
TEMPLATE = [
    ("GAD NAME *", "gad_name"), (None, "head"), ("COLOR", "color"),
    ("BRIDGE NUMBER *", "br_no"), ("RDSO NO. *", "rdso"), ("BRIDGE Type *", "type"), (None, None),
    ("--- METADATA (BOX) ---", None), ("NO OF CELLS / SPAN *", "cells"), ("SPAN SIZE *", "span"),
    ("BOX HEIGHT ( only for rcc box * ) ", "height"), ("STONE MASONRY (m)", None), (None, None), (None, None),
    ("--- L-SECTION ---", None), ("CHAINAGE *", "chainage"), ("SKEW ANGLE", "skew"), ("DEGREE OF CURVE", "degree"),
    ("BED LEVEL *", "bed_level"), ("FORMATION LEVEL *", "fl"), ("RAIL LEVEL ", "rl"), ("TC_DISTANCE", None),
    ("HFL", "hfl"), ("END DIST", "end_dist"), (None, None),
    ("--- HYDROLOGY ---", None), ("SCOUR DEPTH BELOW BED LEVEL ", None), ("MAX SCOUR LEVEL", None), ("DISCHARGE", None),
    ("LINEAR WATERWAY", None), ("FLOW DIRECTION", None), (None, None),
    ("--- SOIL DATA ---", None), ("SBC AT FOUNDATION OF RCC BOX", None),
    ("SBS AT FOUNDATION OF RETURN WALL / WING WALL / RETAINING WALL", None), ("FOUNDATION LEVEL AT RCC BOX", None),
    ("FOUNDATION LEVEL AT RETURN WALL / WING WALL / RETAINING WALL", None), ("BORE LOG INTERVAL (m)", None), (None, None),
    ("--- BORE LOG TABLE ---", None),
] + [x for n in (1, 2, 3) for x in ((f"** LAYER {n} **", None), (f"LAYER {n} SOIL TYPE", None), (f"LAYER {n} FROM (m)", None),
                                     (f"LAYER {n} TO (m)", None), (f"LAYER {n} HATCH PATTERN", None), (None, None))][:-1] + [
    (None, None), ("--- STATION DATA ---", None), ("STATION (A)(LEFT)", None), ("STATION (B)(RIGHT)", None),
]
NOTE_ROW = 19                         # (1-based, as in the template: beside BED LEVEL)

NUM = r"\d+(?:\.\d+)?"
RL_KV = re.compile(r"((?:EX(?:G|IST\w*)?|PROP\w*)?\.?\s*R\.?\s*L\.?(?:\s+(?:UP|DN|DOWN|[1-9](?:ST|ND|RD|TH))\s*(?:LINE|/\s*L))?)\s*[:=\-]\s*(-?\d+(?:\.\d+)?)",
                   re.I)
# every flood level of a level block: "OHFL = ...", "CHFL = ...", "HFL = ...", "H.F.L : ..."
HFL_KV = re.compile(r"\b((?:[OC]\.?\s*)?H\.?\s*F\.?\s*L)\.?\s*[:=\-]\s*(-?\d+(?:\.\d+)?)", re.I)


def hfl_max(text):
    """The HFL for the card: the highest of the flood levels printed (OHFL, CHFL, HFL) - with both an observed and a
    calculated HFL the design takes the higher one. '' when none is printed."""
    vals = [v for _, v in HFL_KV.findall(text or "")]
    return max(vals, key=float) if vals else ""


def config(conf):
    """'1 X 3 X 2' -> ('1', '3', '2'); '1 X 2.59' -> ('1', '2.59', ''); '1 X 1.83 + 1 X 3.66' -> ('2', '1.83 + 3.66', '')."""
    if not conf:
        return "", "", ""
    parts = [re.findall(NUM, p) for p in re.split(r"\+", conf)]
    parts = [p for p in parts if p]
    if not parts:
        return "", "", ""
    if len(parts) == 1:
        p = parts[0]
        return p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else ""
    cells = sum(int(float(p[0])) for p in parts)
    heights = {p[2] for p in parts if len(p) > 2}
    return str(cells), " + ".join(p[1] for p in parts if len(p) > 1), heights.pop() if len(heights) == 1 else ""


def chainage_m(ch):
    """'928+210.273' -> '928210.273' (metres, as the template has it); '11384' stays."""
    if not ch:
        return ""
    s = re.sub(r"\s", "", str(ch))
    if "+" in s:
        km, m = s.split("+", 1)
        try:
            d = len(m.split(".")[1]) if "." in m else 0
            return f"{int(km) * 1000 + float(m):.{d}f}"
        except ValueError:
            return s
    return s


def rail_levels(text):
    """(existing, proposed) rail level from a level block's text: EX / UP / DN line = existing, PROP / the highest
    numbered line = proposed, a plain RL = proposed."""
    ex = pr = None
    ordinal = []
    for k, v in RL_KV.findall(text or ""):
        key = re.sub(r"[^A-Z0-9]", "", k.upper())
        m = re.search(r"(\d)(ST|ND|RD|TH)", key)
        if key.startswith("EX") or re.search(r"(UP|DN|DOWN)", key):
            ex = ex or v
        elif m:
            ordinal.append((int(m.group(1)), v))
        else:
            pr = pr or v
    if ordinal:
        ordinal.sort(key=lambda t: -t[0])
        pr = pr or ordinal[0][1]
        if not ex and len(ordinal) > 1:
            ex = ordinal[1][1]
    return ex, pr


def card(row, level_text="", degree=("", "")):
    """{key: (existing, proposed)} for one bridge from its bridge_list row (as read), its level block's text and the
    degree of the curve it is on (existing, proposed: empty when it is on no curve)."""
    ex_c, ex_s, ex_h = config(row.get("exg_configuration"))
    pr_c, pr_s, pr_h = config(row.get("prop_configuration"))
    rl_ex, rl_pr = rail_levels(level_text)
    hfl = hfl_max(level_text)
    ch = chainage_m(row.get("chainage"))
    bl = row.get("prop_bed_level") or ""
    br = re.sub(r"\s+", " ", row.get("br_no") or "")
    return {
        # (no GAD name on an L-section: what a callout ends with is often a remark, not a name)
        "gad_name": ("", ""), "head": ("EXISTING BRIDGE DETAILS", "PROPOSE BRIDGE DETAIL"),
        "color": ("BLACK", "RED"), "br_no": (br, br), "rdso": (RDSO, RDSO),
        "type": (row.get("exg_structure") or "", row.get("prop_structure") or ""),
        "cells": (ex_c, pr_c), "span": (ex_s, pr_s), "height": (ex_h, pr_h),
        "chainage": (ch, ch), "skew": ("0", "0"), "degree": tuple(degree), "bed_level": (bl, bl),
        "fl": (row.get("exs_fl") or "", row.get("prop_fl") or ""), "rl": (rl_ex or "", rl_pr or ""),
        "hfl": (hfl, hfl), "end_dist": ("0", "0"),
    }


def table(cards):
    """The CSV rows: column A the template's label, then two columns (existing, proposed) per bridge, the note after."""
    out = []
    for i, (label, key) in enumerate(TEMPLATE, start=1):
        r = [label or ""]
        for c in cards:
            r += list(c.get(key, ("", ""))) if key else ["", ""]
        if i == NOTE_ROW:
            r += ["", NOTE]
        out.append(r)
    width = max(len(r) for r in out)
    return [r + [""] * (width - len(r)) for r in out]


def write(cards, path):
    """The table as a CSV Excel opens with the right characters (utf-8 with a byte order mark)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(table(cards))
    return path


def text(cards):
    """The filled rows only, for the chat answer (the CSV has every template row)."""
    lines = []
    for label, key in TEMPLATE:
        if not key or key in ("head",):
            continue
        vals = [v for c in cards for v in c.get(key, ("", ""))]
        if any(vals):
            lines.append(f"  {label.strip():<28} " + " | ".join(v or "-" for v in vals))
    return "\n".join(lines)
