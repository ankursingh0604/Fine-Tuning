"""The list of all bridges (and level crossings, ROBs, RUBs) on a drawing as JSON rows in the agreed format:

    {"br_no": "214 UP", "chainage": "928+210.273", "exg_structure": "RCC SLAB", "exg_configuration": "1 X 2.59",
     "description": "DRAIN", "exs_fl": "248.709", "prop_structure": "RCC BOX", "prop_type": "MINOR",
     "prop_configuration": "1 X 3 X 2", "prop_fl": "248.708", "prop_bed_level": null}

Every value comes from the model's reading of the item's own callout and level block (find_crop.Finder.read_parts),
and the same fields parsed from the PDF text are kept beside them, so a reading that differs from the PDF is reported
(never silently replaced). Two ways drawings print a bridge are understood:

  one callout per bridge   "C/L OF EXG. BR. NO. 575 DN RCC SLAB - 1 X 1.83 + 1 X 3.66 + 1 X 1.83 - BT ROAD PRO TO BE
                            EXTENDED AS 1 X 4X4 - RCC BOX (MINOR) AT CH: 1245630.853"  (+ "EXG FL = ...", "FL = ...")
                           read by position: existing structure, its configuration, what it crosses; then the
                           proposal (configuration - structure (type)); also for "C/L OF EXG. LC 137 BT ROAD ...",
                           "ROB-15 (NH 552)", "RUB-257", "BR. NO. LC-226 LEVEL CROSSING - - MDR ..."
  an EX. and a PROP. callout "EX. Br. No. :15, SPAN : 1 X 3.66M. , F.T., AT CH: 11378 m" and
                            "PROP. Br. No. :15, SPAN : 1 X 3.66M. , R.C.C. BOX, AT CH : 11384m." (+ "EX. FL", "PROP. FL",
                            "BL"): merged into one row, the chainage being the proposed one
"""
import json
import re

FIELDS = ["br_no", "chainage", "exg_structure", "exg_configuration", "description", "exs_fl",
          "prop_structure", "prop_type", "prop_configuration", "prop_fl", "prop_bed_level",
          "hc", "vc"]                    # horizontal / vertical clearance (m), printed for level crossings and road bridges

LINE_FL = re.compile(r"\bF\.?\s*L\.?\s+(UP|DN|DOWN|[1-9](?:ST|ND|RD|TH))\s*(?:LINE|/\s*L)\b\.?\s*[:=\-]\s*(-?\d+(?:\.\d+)?)", re.I)
CLEARANCE = re.compile(r"\b(HC|VC)\s*[:=\-]\s*(\d+(?:\.\d+)?)", re.I)     # "HC = 7.819", "VC = 7"
BED_LEVEL = re.compile(r"\bBED\s+LE?VE?L\.?\s*[:=\-]\s*(-?\d+(?:\.\d+)?)", re.I)
# also "LINEAT CH:" (no space), and a level crossing's chainage line "CH: 945+828.373 ROB PROPOSED BY IR."
AT_CH = re.compile(r"(?:AT\s*CH\.?\s*[:.]?|\bCH\s*:)\s*(\d[\d+.,]*(?:\s*\+\s*[\d.]+)?)", re.I)
ONE = r"\d+\s*[xX×]\s*\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?)*"
SPAN = re.compile(rf"{ONE}(?:\s*\+\s*{ONE})*")
ITEM_ID = re.compile(r"^\W*(?:C/L\s*OF\s*)?(?:EXG?\.?|EX\.|EXISTING|PROP\w*\.?)?\s*(?:BR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*)?"
                     r"(?P<id>(?:LC|ROB|RUB|FOB)\b\s*[-.]?\s*(?:\d+[A-Z]*)?(?:\s*\([^)]*\))?"
                     r"|\d+[A-Z]*(?:\s*\([^)]*\))?(?:\s+(?:UP|DN)\b)?)", re.I)
XING = re.compile(r"L-?XING\s*NO\.?\s*\d+[A-Z]*", re.I)
PROP_AS = re.compile(r"PRO\w*\.?\s*TO\s*BE\s*(?:EXTENDED|CONSTRUCTED|PROVIDED|REPLACED|REBUILT)?\s*(?:AS|BY|WITH)?\s*(?P<p>.*)$", re.I)
PROP_NOTE = re.compile(r"(?P<p>(?:ROB|RUB|FOB)\s*PROPOSED.*)$", re.I)
PREFIX = re.compile(r"^\W*(?:C/L\s*OF\s*)?(?:EXG?\.?|EX\.|EXISTING|PROP\w*\.?)?\s*(?:BR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*)?$", re.I)
LEVEL_KV = re.compile(r"((?:EX(?:G|IST\w*)?|PROP\w*|MIN)?\.?\s*F\.?\s*L\.?(?:\s*REQ\.?)?|H\.?\s*F\.?\s*L|B\.?\s*L)\s*[:=\-]\s*(-?\d+(?:\.\d+)?)",
                      re.I)


def clean(s):
    s = re.sub(r"\s+", " ", s or "").strip(" ,;:-")
    s = re.sub(r"^[,.;:\s-]+|[,;:\s-]+$", "", s)
    return s or None


def chainage_str(printed):
    """'1245630.853' -> '1245+630.853', '11384' -> '11+384', '941+904.000' stays (km+m, as the format wants)."""
    s = re.sub(r"\s", "", printed or "").rstrip("mM.,")
    if not s:
        return None
    if "+" in s:
        return s
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return s
    d = len(s.split(".")[1]) if "." in s else 0
    km = int(v // 1000)
    return f"{km}+{v - km * 1000:0{(4 + d) if d else 3}.{d}f}"


def levels(text):
    """exs_fl, prop_fl, prop_bed_level from a level block: "EXG FL = 175.62 MIN FL REQ. = ... FL = 175.535 B.L = ..."
    or "EX. FL : 3.782 PROP. FL : 3.973 HFL : 2.645 BL : 0.301"."""
    out = {"exs_fl": None, "prop_fl": None, "prop_bed_level": None, "hc": None, "vc": None}
    for k, v in CLEARANCE.findall(text or ""):
        out[k.lower()] = out[k.lower()] or v
    # FLs named by line ("FL UP LINE = 237.34", "FL 3RD LINE = 237.343"): an existing line (UP / DN) gives the existing FL,
    # the highest-numbered line the proposed one (a lower numbered line, when there is no UP / DN, the existing one)
    named = [(ln.upper(), v) for ln, v in LINE_FL.findall(text or "")]
    ordinal = sorted(((int(re.match(r"\d", ln).group()), v) for ln, v in named if ln[0].isdigit()), key=lambda t: -t[0])
    exist = [v for ln, v in named if not ln[0].isdigit()]
    if ordinal:
        out["prop_fl"] = ordinal[0][1]
    if exist:
        out["exs_fl"] = exist[0]
    elif len(ordinal) > 1:
        out["exs_fl"] = ordinal[1][1]
    m = BED_LEVEL.search(text or "")
    if m:
        out["prop_bed_level"] = m.group(1)
    for k, v in LEVEL_KV.findall(text or ""):
        key = re.sub(r"[^A-Z]", "", k.upper())
        if key in ("EXFL", "EXGFL", "EXISTFL", "EXISTINGFL"):
            out["exs_fl"] = out["exs_fl"] or v
        elif key in ("PROPFL", "PROPOSEDFL", "FL"):
            out["prop_fl"] = out["prop_fl"] or v
        elif key == "BL":
            out["prop_bed_level"] = out["prop_bed_level"] or v
    return out


def split_existing(part):
    """'RCC SLAB - 1 X 1.83 + 1 X 3.66 - BT ROAD' -> structure, configuration, what it crosses."""
    part = clean(part) or ""
    span = SPAN.search(part)
    if span:
        return (clean(re.sub(r"^\s*(UP|DN)\b", "", part[:span.start()], flags=re.I)), clean(span.group()),
                clean(part[span.end():]))
    segs = [s.strip() for s in re.split(r"(?:^|\s)-(?=\s|$)", part)]     # " - " separates; "NH-552" does not
    segs += [""] * (3 - len(segs))
    return clean(segs[0]), clean(segs[1]), clean(" ".join(segs[2:]))


def parse_callout(text, hint=None):
    """A one-callout item ('C/L OF EXG. BR. NO. ...', 'C/L OF EXG. LC 137 ...', "'SPL' CLASS L-XING NO. 10 ...").
    hint: the item's number as the PDF prints it ("575 DN", "LC 137"); a reading is then split at that number even
    when its spaces are lost ("271UPPSCSLAB" -> 271UP | PSC SLAB) - the number is still taken from the reading, and a
    reading that does not contain it falls back to the usual parse (and so shows up as a difference)."""
    t = re.sub(r"\s+", " ", text or "").strip()
    row = dict.fromkeys(FIELDS)
    ch = AT_CH.search(t)
    if ch:
        row["chainage"] = chainage_str(ch.group(1))
        t = t[:ch.start()]
    if hint:
        h = re.search(r"\s*".join(map(re.escape, re.sub(r"\s", "", hint))), t, re.I)
        if h and len(t[:h.start()]) < 60:
            row["br_no"] = clean(t[h.start():h.end()])
            before = t[:h.start()]
            rest = ("" if PREFIX.match(before) else re.sub(r"(?:C/L\s*OF\s*)?(?:EXG?\.?\s*)?(?:BR\.?\s*NO\.?\s*)?$", "", before, flags=re.I)
                    ) + xing_word(row["br_no"]) + t[h.end():]
            return finish(row, rest, text)
    x = XING.search(t)
    m = ITEM_ID.match(t)
    if x:
        row["br_no"] = clean(x.group())
        rest = t[:x.start()] + xing_word(row["br_no"]) + t[x.end():]
    elif m:
        row["br_no"] = clean(m.group("id"))
        rest = t[m.end():]
    else:
        return row
    return finish(row, rest, text)


def xing_word(br_no):
    """A level crossing's number sits in the middle of what it is ("'SPL' CLASS L-XING NO. 10 (MANNED)"): taking the
    number out must leave "L-XING" in the structure ("'SPL' CLASS L-XING (MANNED)"), not "'SPL' CLASS (MANNED)"."""
    m = re.match(r"\s*(L-?XING)\b", br_no or "", re.I)
    return f" {m.group(1)} " if m else " "


def finish(row, rest, text):
    """The fields after the item's number: existing part, then the proposal."""
    p = PROP_AS.search(rest) or PROP_NOTE.search(rest)
    exg = rest[:p.start()] if p else rest
    row["exg_structure"], row["exg_configuration"], row["description"] = split_existing(exg)
    if p and p.re is PROP_AS:
        prop = p.group("p")
        kind = re.search(r"\(([^)]*)\)\s*$", prop)
        if kind:
            row["prop_type"] = clean(kind.group(1))
            prop = prop[:kind.start()]
        span = SPAN.match(prop.strip())
        if span:
            conf, prop = span.group(), prop.strip()[span.end():]
            dangling = re.match(r"\s*[xX×](?=\s*(?:-|$))", prop)          # printed incomplete: "1 X 6 X - RCC BOX"
            if dangling:
                conf, prop = conf + " X", prop[dangling.end():]
            row["prop_configuration"] = clean(conf)
        row["prop_structure"] = clean(prop)
    elif p:
        row["prop_structure"] = clean(p.group("p"))
    elif re.match(r"^\W*PROP", text or "", re.I) and not row["exg_structure"]:
        row["prop_structure"] = clean(re.sub(r"\(.*", "", row["br_no"] or "")) or None     # "PROPOSED ROB AT CH ..."
    return row


def parse_pair(ex_text, prop_text, num):
    """An EX. and a PROP. callout of one bridge ('EX. Br. No. :15, SPAN : 1 X 3.66M. , F.T., AT CH: 11378 m')."""
    import sheet_objects
    row = dict.fromkeys(FIELDS)
    row["br_no"] = num
    for text, pre in ((ex_text, "exg"), (prop_text, "prop")):
        if not text:
            continue
        f = sheet_objects.Bridge(num, None).fields(text)
        row[f"{pre}_configuration"] = clean(f["span"])
        row[f"{pre}_structure"] = clean(f["type"])
    ch = AT_CH.search(prop_text or "") or AT_CH.search(ex_text or "")
    row["chainage"] = chainage_str(ch.group(1)) if ch else None
    return row


def same(a, b):
    n = lambda v: re.sub(r"[^A-Z0-9+.]", "", str(v).upper()) if v is not None else ""
    return n(a) == n(b)


def rows(finder, q=""):
    """[(row, pdf_row)] for one sheet, in chainage order: row is each item as the model read it, pdf_row the same
    fields from the PDF text. Filters in q (existing / proposed, RCC, box, pipe, girder, slab, arch ...) narrow it."""
    import sheet_objects
    objs = finder.objects()
    out = []

    def read(parts, what):
        return finder.read_parts(parts, what)[0] if parts else ""

    groups = {}
    for b in objs.bridges:
        groups.setdefault(b.num, []).append(b)
    for num, bs in groups.items():
        pdf_text = lambda b: sheet_objects.joined(b.best()).text
        paired = all(sheet_objects.SPAN.search(pdf_text(b)) for b in bs)
        lv = bs[0].levels[0] if bs[0].levels else None
        if paired:                                       # EX. and PROP. callouts
            ex = next((b for b in bs if b.status == "existing"), None)
            pr = next((b for b in bs if b.status == "proposed"), None)
            got = parse_pair(read(ex.best(), f"br_{ex.name}") if ex else None,
                             read(pr.best(), f"br_{pr.name}") if pr else None, num)
            want = parse_pair(pdf_text(ex) if ex else None, pdf_text(pr) if pr else None, num)
        else:                                            # one callout per bridge
            b = bs[0]
            want = parse_callout(pdf_text(b))
            got = parse_callout(read(b.best(), f"br_{b.name}"), hint=want["br_no"])
            got["br_no"] = got["br_no"] or want["br_no"]
        if lv:
            got.update(levels(read(lv, f"brlv_{num}")))
            want.update(levels(sheet_objects.joined(lv).text))
        out.append((got, want))
    for c in objs.crossings:
        want = parse_callout(sheet_objects.joined(c.best()).text)
        got = parse_callout(read(c.best(), f"xing_{c.label}"), hint=want["br_no"])
        if c.levels:
            got.update(levels(read(c.levels[0], f"xinglv_{c.label}")))
            want.update(levels(sheet_objects.joined(c.levels[0]).text))
        out.append((got, want))
    out = [(g, w) for g, w in out if wanted(w, q, finder)]
    out.sort(key=lambda t: chainage_value(t[1]["chainage"]))
    return out


def chainage_value(s):
    try:
        if s and "+" in s:
            a, b = s.split("+")
            return int(a) * 1000 + float(b)
        return float(s)
    except (TypeError, ValueError):
        return float("inf")


def wanted(row, q, finder):
    """The rows a list question asks for: existing / proposed, and structure words (RCC, box, pipe, girder ...)."""
    ql = (q or "").lower()
    st = finder.asked_status(ql)
    if st == "existing" and not row["exg_structure"]:
        return False
    if st == "proposed" and not row["prop_structure"]:
        return False
    kinds = [alts for pat, alts in finder.TYPE_WORDS if re.search(pat, ql)]
    if not kinds:
        return True
    pick = row["exg_structure"] if st == "existing" else row["prop_structure"] if st == "proposed" else \
        f"{row['exg_structure'] or ''} {row['prop_structure'] or ''}"
    t = re.sub(r"[^A-Z0-9]", "", (pick or "").upper())
    return all(any(k in t for k in alts) for alts in kinds)


def problems(got, want, where=""):
    """Lines naming every field where the model's reading differs from the PDF text."""
    return [f"{where}{got['br_no'] or want['br_no']} · {k}: read {json.dumps(got[k])}, the PDF says {json.dumps(want[k])}"
            for k in FIELDS if not same(got[k], want[k])]


def dump(rs):
    return json.dumps([{k: r[k] for k in FIELDS} for r in rs], indent=2, ensure_ascii=False)


def write_csv(rs, path):
    """The same rows as a CSV for Excel: the same columns in the same order, an empty cell where the JSON has null.
    Written with a byte order mark (utf-8-sig) so Excel opens it with the right characters."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rs:
            w.writerow(["" if r[k] is None else r[k] for k in FIELDS])
