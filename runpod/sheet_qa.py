"""Answer typed questions about a whole sheet from what read_sheet.py read off it.

Understands questions about
  - a bridge               "bridge 560", "full report for 560", "ROB-239", "LC-226"
  - a value at a bridge    "ground level at bridge 560", "FL of 558", "min FL of 562", "HFL at 565",
                           "cut or fill at 560", "rail level at 560", "span of 554", "proposal for 560"
  - a chainage             "CH 1242662.9", "1242+662.9", "ground level at 1241000"
                           (band values at the nearest column - never interpolated)
  - flags and checks       "which bridges are flagged", "problems"
  - the sheet              "list bridges", "how many bridges", "title", "drawing number", "scale", "summary"
  - TBMs                   "TBM BM39", "bench marks"
Values are the model's readings of the image, so a digit can be misread; the checks say when numbers
on the sheet do not add up.
"""
import re

from sheet_reader import g

LEVELS = {"existing_formation_level": "existing FL", "min_formation_level_required": "MIN FL REQ.",
          "proposed_formation_level": "FL", "bed_level": "bed level", "high_flood_level": "HFL", "free_board": "free board"}
BAND = {"ground_level": "ground level", "prop_fl": "proposed FL", "prop_rl": "proposed RL", "cut_fill": "cut(-)/fill(+)",
        "exg_up_fl": "EXG. UP LINE FL", "fl_difference": "FL difference", "track_distance": "track distance"}
DATA = {"existing_type": "existing type", "existing_span": "existing span", "crossing": "crossing", "proposal": "proposal",
        "category": "category", "chainage_m": "chainage"}

# (pattern, [(source, key)]), most specific first; a matched phrase is removed before the next pattern is tried,
# so "min FL" does not also count as "FL". The data-band row labels name other rows inside them
# ("DIFFERENCE BETWEEN PROP. 3RD LINE FL AND EXG. UP LINE FL", "CUT(-)/FILL(+) (FL - GL)",
# "TRACK DISTANCE BETWEEN PROP. 3RD LINE & EXG. UP LINE"), so whole labels are matched before single words.
EXG = r"(?:exg\.?|existing|exist\.?)"
FIELDS = [
    (r"(?:fl\s*)?diff(?:erence)?(?:\s*between)?(?:[^?]*?" + EXG + r"\s*up\s*(?:line\s*)?fl\b)?", [("band", "fl_difference")]),
    (r"track\s*(?:distance|centres?|centers?|spacing)(?:\s*between[^?]*?" + EXG + r"\s*up(?:\s*line)?)?|distance\s*between\s*tracks",
     [("band", "track_distance")]),
    (r"\bcut\w*|\bfill\w*|bank\s*height|embankment|\(?\s*\bfl\s*-\s*gl\b\s*\)?", [("band", "cut_fill")]),
    (r"min(?:imum)?\.?\s*(?:fl|formation(?:\s*level)?)(?:\s*req\w*\.?)?", [("lv", "min_formation_level_required")]),
    (EXG + r"\s*up\s*(?:line\s*)?(?:fl|formation(?:\s*level)?)\b", [("band", "exg_up_fl")]),     # the band row "EXG. UP LINE FL"
    (EXG + r"\s*(?:fl|formation(?:\s*level)?)\b", [("lv", "existing_formation_level"), ("band", "exg_up_fl")]),
    (r"free\s*-?board|\bfb\b", [("lv", "free_board")]),
    (r"\bhfl\b|high\s*flood(?:\s*level)?|flood\s*level", [("lv", "high_flood_level")]),
    (r"bed\s*level|\bbl\b", [("lv", "bed_level")]),
    (r"ground(?:\s*level)?|\bgl\b|\bngl\b", [("band", "ground_level")]),
    (r"rail\s*level|\brl\b", [("band", "prop_rl")]),
    (r"\bchainage\b|\bwhere\b|\blocation\b", [("d", "chainage_m")]),
    (r"\blevels\b", [("lv", k) for k in LEVELS]),
    (r"formation(?:\s*level)?|\bp?fl\b|\blevel\b", [("lv", "proposed_formation_level"), ("band", "prop_fl")]),
    (r"\bspan\b|existing\s*(?:type|bridge|structure)|\btype\b", [("d", "existing_type"), ("d", "existing_span")]),
    (r"proposal|proposed\s*(?:structure|bridge)|new\s*(?:structure|bridge)", [("d", "proposal")]),
    (r"crossing|\bcross(?:es)?\b|\bover\s*what\b", [("d", "crossing")]),
    (r"category|\bmajor\b|\bminor\b", [("d", "category")]),
    (r"\bband", [("band", "*")]),
    (r"\bflag|\bcheck|problem|issue|\bsafe\b", [("checks", "*")]),
]

HELP = """I can answer, from what was read off this sheet:
  bridge 560                     full report: callout, levels, band column, checks
  ground level at bridge 560     also FL, min FL, existing FL, HFL, bed level, free board, rail level,
                                 cut/fill, FL difference, track distance, span, proposal, crossing, category
  CH 1242662.9  /  1242+662.9    nearest bridge, and band values at the nearest column
  which bridges are flagged      FLAG / CHECK findings
  list bridges  |  summary  |  title  |  drawing number  |  scale  |  TBM BM39  |  bench marks"""


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def to_m(text):
    """Chainage in metres from '1242+662.9', 'CH 1242662.9', '1242662.9' or 'km 1242.66'."""
    m = re.search(r"(\d{4})\s*\+\s*(\d+(?:\.\d+)?)", text)
    if m:
        return int(m.group(1)) * 1000 + float(m.group(2)), m.group(0)
    m = re.search(r"(?:\bch(?:ainage)?\s*[:.]?\s*)?\b(\d{7}(?:\.\d+)?)\b", text, re.I)
    if m:
        return float(m.group(1)), m.group(0)
    m = re.search(r"\bkm\s*[:.]?\s*(\d{4}(?:\.\d+)?)", text, re.I)
    return (float(m.group(1)) * 1000, m.group(0)) if m else (None, "")


def name_of(bid):
    return bid if bid.startswith(("ROB", "LC")) else f"Bridge {bid}"


class SheetQA:
    def __init__(self, result, band_reader=None):
        """result: read_sheet.py's _read.json. band_reader(ch, name) -> band dict reads a band column
        with the model (None when the model is not loaded)."""
        self.r = result
        self.band_reader = band_reader
        self.bridges = result.get("bridges") or []
        self.ids = {}
        for b in self.bridges:
            bid = b["bridge_id"]
            self.ids[norm(bid)] = b
            for part in re.findall(r"[A-Z]+-?\d+[A-Z]*|\d+[A-Z]*", bid.split(" at ")[0]):   # 578AX(LC-247) -> 578AX, LC-247
                self.ids.setdefault(norm(part), b)
        t = result.get("title") or {}
        self.sheet = f"sheet {t['sheet_no']}" if t.get("sheet_no") else "this sheet"

    # ------------------------------------------------------------ finding what the question is about

    def find_bridge(self, q):
        for m in re.finditer(r"\b(ROB|RUB|LC)?\s*[-.]?\s*(\d{1,4}[A-Z]{0,2})\b", q.upper()):
            pre, num = m.group(1) or "", m.group(2)
            if len(num) >= 7:
                continue
            for key in ([norm(pre + num)] if pre else []) + [norm(num)]:
                if key in self.ids:
                    return self.ids[key]
        return None

    def nearest_bridge(self, ch):
        near = [(abs(b["data"]["chainage_m"] - ch), b) for b in self.bridges
                if b["data"] and isinstance(b["data"].get("chainage_m"), (int, float))]
        return min(near, key=lambda t: t[0]) if near else (None, None)

    @staticmethod
    def fields(q):
        q = q.lower()
        found = []
        for pat, keys in FIELDS:
            if re.search(pat, q):
                found += [k for k in keys if k not in found]
                q = re.sub(pat, " ", q)
        return found

    # ------------------------------------------------------------ pieces of answers

    def band_line(self, band, ch_asked):
        col = (band or {}).get("nearest_column") or {}
        if not col:
            return None
        vals = ", ".join(f"{lab} {g(col[k])}" for k, lab in BAND.items() if col.get(k) is not None)
        away = abs(col["chainage"] - ch_asked) if isinstance(col.get("chainage"), (int, float)) else None
        return (f"band column CH {g(col.get('chainage'))}" + (f" ({g(away)} m from {g(ch_asked)}, nearest column, not interpolated)"
                                                           if away is not None else "") + f": {vals}")

    def report(self, b):
        d = b["data"]
        if not d:
            return f"{name_of(b['bridge_id'])} was found on {self.sheet} but its callout could not be read."
        lv = d.get("levels") or {}
        out = [f"{name_of(b['bridge_id'])} ({self.sheet}, read from the {b['read_from']}):",
               "  " + " | ".join(f"{lab}: {g(d[k])}" for k, lab in DATA.items() if d.get(k) is not None)]
        if lv:
            out.append("  Levels: " + ", ".join(f"{lab} {g(lv[k])}" for k, lab in LEVELS.items() if lv.get(k) is not None))
        else:
            out.append("  Levels: no level block was read for this bridge.")
        bl = self.band_line(b["band"], d.get("chainage_m"))
        out.append("  " + (bl[0].upper() + bl[1:] if bl else "Band values: not read (bridge off the data bands, or unreadable)."))
        out.append("  Checks: " + ("; ".join(c["message"] for c in b["checks"]) if b["checks"] else "OK (FL >= MIN FL, band arithmetic adds up)."))
        return "\n".join(out)

    def value(self, b, keys):
        d = b["data"] or {}
        lv = d.get("levels") or {}
        col = (b["band"] or {}).get("nearest_column") or {}
        ch = d.get("chainage_m")
        out = []
        for src, k in keys:
            if src == "checks":
                out.append(("; ".join(c["message"] for c in b["checks"])) or f"No flags: FL is not below MIN FL REQ. and the band values add up.")
            elif src == "band" and k == "*":
                out.append(self.band_line(b["band"], ch) or "Band values were not read for this bridge.")
            elif src == "lv":
                if lv.get(k) is not None:
                    out.append(f"{LEVELS[k]} = {g(lv[k])} m (level block)")
                elif not lv:
                    out.append(f"{LEVELS[k]}: no level block was read for this bridge")
                else:
                    out.append(f"{LEVELS[k]}: not given in the level block")
            elif src == "band":
                if col.get(k) is not None:
                    away = f", {g(abs(col['chainage'] - ch))} m away" if isinstance(col.get("chainage"), (int, float)) and isinstance(ch, (int, float)) else ""
                    out.append(f"{BAND[k]} = {g(col[k])} at band column CH {g(col.get('chainage'))} (nearest column to the bridge{away}, not interpolated)")
                elif ("lv", "proposed_formation_level") not in keys or k != "prop_fl":
                    out.append(f"{BAND[k]}: not read (no band column for this bridge)")
            elif d.get(k) is not None:
                out.append(f"{DATA[k]}: {g(d[k])}")
        return f"{name_of(b['bridge_id'])} ({self.sheet}" + (f", CH {g(ch)}" if ch else "") + "): " + "; ".join(out) + "."

    def at_chainage(self, ch, keys):
        dist, b = self.nearest_bridge(ch)
        lines = []
        if b is not None:
            lines.append(f"Nearest bridge: {name_of(b['bridge_id'])} at CH {g(b['data']['chainage_m'])} ({g(dist)} m away).")
        col_band = b["band"] if b is not None and dist is not None and dist <= 10 else None
        if col_band is None:
            if self.band_reader is None:
                lines.append("Band values at this chainage were not read yet: run read_sheet.py with --adapter "
                             "(and the sheet image) so the model can read that column.")
            else:
                col_band = self.band_reader(ch, f"Chainage {g(ch)}")
                if col_band is None:
                    lines.append(f"CH {g(ch)} is not on this sheet's data bands, or the model could not read that column.")
        if col_band:
            col = col_band.get("nearest_column") or {}
            band_keys = [k for s, k in keys if s == "band" and k != "*"]
            if band_keys:
                lines.append("; ".join(f"{BAND[k]} = {g(col.get(k))}" for k in band_keys) +
                             f" at band column CH {g(col.get('chainage'))} (nearest column to {g(ch)}, not interpolated).")
            else:
                lines.append(self.band_line(col_band, ch)[0].upper() + self.band_line(col_band, ch)[1:] + ".")
        return "\n".join(lines)

    # ------------------------------------------------------------ sheet-level answers

    def flagged(self):
        f = self.r.get("findings") or []
        if not f:
            return f"No findings on {self.sheet}: every FL read is at or above its MIN FL REQ. and the band values add up."
        return "\n".join(f"[{x['severity']}] {x['message']}" for x in f)

    def list_bridges(self):
        rows = []
        for b in self.bridges:
            d = b["data"] or {}
            fl = (d.get("levels") or {}).get("proposed_formation_level")
            mark = " FLAG" if any(c["severity"] == "FLAG" for c in b["checks"]) else ""
            rows.append(f"  {b['bridge_id']:<10} CH {g(d.get('chainage_m')) if d.get('chainage_m') else '?':<12} "
                        f"{(d.get('existing_type') or '')} {(d.get('existing_span') or '')} over {d.get('crossing') or '?'}"
                        f"{' -> ' + d['proposal'] if d.get('proposal') else ''}{f'  FL {g(fl)}' if fl is not None else ''}{mark}")
        return f"{len(self.bridges)} bridges on {self.sheet}:\n" + "\n".join(rows)

    def title(self, q):
        t = self.r.get("title") or {}
        if not t:
            return "The title block could not be read."
        for pat, k in ((r"drawing\s*(?:no|number)|dwg", "drawing_no"), (r"scale", "scale"), (r"date", "date"),
                       (r"client|railway|division", "client"), (r"from|to|range|cover", None)):
            if re.search(pat, q, re.I):
                if k is None:
                    return f"{self.sheet.capitalize()} covers chainage {t.get('chainage_from')} to {t.get('chainage_to')}."
                return f"{k.replace('_', ' ').capitalize()}: {t.get(k)}"
        return "\n".join(f"  {k.replace('_', ' ')}: {v}" for k, v in t.items() if v)

    def tbm(self, q):
        rows = self.r.get("tbm") or []
        if not rows:
            return "The TBM table could not be read."
        m = re.search(r"\b(B\s*M\s*T?\s*-?\s*\d+)", q, re.I)
        if m:
            want = norm(m.group(1))
            hit = [t for t in rows if norm(str(t.get("tbm_id", ""))) == want]
            if hit:
                t = hit[0]
                return (f"{t['tbm_id']}: CH {g(t.get('chainage_m'))}, MSL {g(t.get('msl_m'))} m, E {g(t.get('easting'))}, "
                        f"N {g(t.get('northing'))} - {t.get('description')}")
            return f"{m.group(1)} is not in the TBM table of {self.sheet} (it has {', '.join(str(t.get('tbm_id')) for t in rows)})."
        return f"{len(rows)} TBMs on {self.sheet}:\n" + "\n".join(
            f"  {t.get('tbm_id')}: CH {g(t.get('chainage_m'))}, MSL {g(t.get('msl_m'))} m - {t.get('description')}" for t in rows)

    def summary(self):
        t = self.r.get("title") or {}
        n_flag = sum(1 for f in self.r.get("findings") or [] if f["severity"] == "FLAG")
        n_chk = sum(1 for f in self.r.get("findings") or [] if f["severity"] == "CHECK")
        out = [f"{self.sheet.capitalize()}: {t.get('title', '')}, chainage {t.get('chainage_from', '?')} to {t.get('chainage_to', '?')}.",
               f"Image about {self.r.get('dpi')} dpi. {len(self.bridges)} bridges found, {sum(1 for b in self.bridges if b['data'])} read, "
               f"{sum(1 for b in self.bridges if b['band'])} with band values; {len(self.r.get('tbm') or [])} TBMs.",
               f"{n_flag} FLAG, {n_chk} CHECK."]
        lay = self.r.get("layout")
        if lay:
            out.append(f"Layout: {lay['status']} (confidence {lay['confidence']}).")
        out += [f"WARNING: {w}" for w in self.r.get("warnings") or []]
        return "\n".join(out)

    # ------------------------------------------------------------ entry point

    def answer(self, question):
        q = question.strip()
        if not q or re.fullmatch(r"help|\?|what can i ask.*", q.lower()):
            return HELP
        text = self._answer(q)
        lay = self.r.get("layout") or {}
        if lay.get("confidence") == "low":
            return "CAUTION: this sheet's layout differs from the trained sheets - treat these values as unreliable (see summary).\n" + text
        if lay.get("confidence") == "medium":
            return "Note: this sheet's layout differs from the trained sheets - check these values.\n" + text
        return text

    def _answer(self, q):
        ql = q.lower()
        ch, ch_text = to_m(q)
        rest = q.replace(ch_text, " ") if ch_text else q
        b = self.find_bridge(rest)
        if b is None and ch is not None and re.search(r"\b(rob|rub|lc|bridge)\b", ql):
            dist, nb = self.nearest_bridge(ch)          # "ROB at CH 1242581.6"
            if nb is not None and dist <= 1:
                b = nb
        keys = self.fields(rest)
        if b is not None:
            if not keys or re.search(r"full|report|detail|everything|all\b|about", ql):
                return self.report(b)
            return self.value(b, keys)
        if ch is not None:
            return self.at_chainage(ch, keys)
        if re.search(r"\btbm|bench\s*-?\s*marks?|\bbm\s*t?\s*-?\d", ql):
            return self.tbm(q)
        if re.search(r"summary|overview", ql):
            return self.summary()
        if re.search(r"\bflag|problem|issue|finding|wrong|\bcheck", ql):
            return self.flagged()
        if re.search(r"(list|all|how many|which)\s.*(bridges?|structures?|callouts?)|^bridges?$", ql):
            return self.list_bridges()
        if re.search(r"title|sheet\s*(?:no|number)|drawing|scale|\bdate\b|client|chainage range|covers?", ql):
            return self.title(q)
        if re.search(r"\b(?:bridge|rob|lc|rub)\b", ql):
            return f"I could not find that bridge on {self.sheet}. Bridges read: {', '.join(b['bridge_id'] for b in self.bridges)}."
        return "I did not understand that question.\n" + HELP
