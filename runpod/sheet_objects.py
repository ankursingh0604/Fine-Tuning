"""The things printed on an L-section sheet that questions name: bridges (their callouts and level blocks) and curves
(their tangent / curve points and detail blocks). Found from where the text is printed (PDF text layer or OCR), so
no drawing set's layout is assumed. Used by find_crop.py, which has the model read every label it answers from.

    objs = SheetObjects(sheet.words)
    objs.bridges        # [Bridge]   one per bridge number and EX / PROP, every printed copy kept
    objs.curves         # [Curve]    one per curve number, status and line, with its points and details

A label can wrap onto the next line ("PROP. Br. No. :14, SPAN : 1 X 2.44M. ," / "R.C.C. BOX, AT CH : 10646m."): the
lines are joined. A bridge or curve is usually printed more than once (plan and profile): the copies are one object.
"""
import re

from find_crop import Word, numbers, union

# the number with at most three letters after it (575A, 578AX, 224UP, 239AUP) or a spaced "214 UP" - a reading without
# spaces ("575DNRCCSLAB") keeps 575. Use bridge_num() for the number: "214 UP" -> "214UP"
BR_RE = re.compile(r"\bBR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*(\d+(?:[A-Z]{1,3}(?![A-Z])|\s(?:UP|DN)\b)?)", re.I)


def bridge_num(m):
    return re.sub(r"\s+", "", m.group(1)).upper()
BR_HEAD = re.compile(r"^\W*(?:EX(?:G|IST\w*)?\.?\s*|PROP\w*\.?\s*)?BR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*\d+[A-Z]*\s*(?:UP|DN)?\W*$", re.I)
LEVEL_LINE = re.compile(r"^\W*(?:EX(?:G|IST(?:ING)?)?\.?|PROP(?:OSED)?\.?)?\s*(?:FL|F\.L|HFL|H\.F\.L|BL|B\.L|SFL|RL|MIN|BED|DSL|LWL|O\.?HFL|C\.?HFL|"
                        r"ROAD\s*LE?VE?L|HC|VC|EARTH\s*CUSHION)\b", re.I)
# (a level block may go on with the road level and the clearances: "EXG.ROAD LEVEL = 238 / HC = 7.819 / VC = 7")
# "AT CH: 945+828.373", and a callout's chainage line that starts "CH: 945+828.373 ROB PROPOSED BY IR." (level crossings)
AT_CH = re.compile(r"(?:\bAT\s*CH\.?\s*[:.]?|\bCH\s*:)\s*(\d[\d\s+.,]*)", re.I)
ONE_SPAN = r"\d+\s*[xX×]\s*\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?)?\s*M?"
SPAN = re.compile(rf"\bSPAN\s*[:.]?\s*({ONE_SPAN}(?:\s*\+\s*{ONE_SPAN})*)", re.I)
BARE_SPAN = re.compile(rf"({ONE_SPAN}(?:\s*\+\s*{ONE_SPAN})*)", re.I)      # "RCC SLAB - 1 X 1.83 + 1 X 3.66 + 1 X 1.83"
PROPOSAL = re.compile(r"PRO\w*\.?\s*TO\s*BE\s*(\w[^()]*?)\s*(?:\(|AT\s*CH|$)", re.I)     # "PRO TO BE EXTENDED AS ..."
LEVEL_KV = re.compile(r"((?:EX(?:G|IST\w*)?|PROP\w*|MIN)?\.?\s*F\.?\s*L\.?(?:\s*REQ\.?)?|H\.?\s*F\.?\s*L|B\.?\s*L|SFL|RL)\s*[:=\-]\s*(-?\d+(?:\.\d+)?)",
                      re.I)
CURVE_PT = re.compile(r"^\W*(?P<ex>EX(?:G|IST(?:ING)?)?\.?\s*)?C\.?\s*(?:NO\.?)?\s*[-.]?\s*(?P<id>\d+[A-Z]*)\s*[.,]?\s*(?:\([LR]\w*\)\s*)?"
                      r"(?P<pt>TPTC|TPCC|TC|CT|CC|TS|SC|CS|ST)\s*-?\s*(?P<n>\d)?\b.*?\bCH\.?\s*[:.]?\s*(?P<ch>\d[\d+.]*)", re.I)
CURVE_HEAD = re.compile(r"^\W*(?P<ex>EX(?:G)?\.?\s*)?C\.?\s*NO\.?\s*[-.]?\s*(?P<id>\d+[A-Z]*)\s*\((?P<hand>[LR])\w*\)\s*(?:\((?P<line>UP|DN)\))?"
                        r"\s*(?P<line2>UP|DN)?\W*$", re.I)
CURVE_LINE = re.compile(r"^\W*(?:Δ|∆|DELTA|R|TL|CL|TRL|TTL|CCL|SHIFT|CA|CD|MSP|V\s*MAX|VMAX|V|LS|L|DEG(?:REE)?\.?(?:\s*OF\s*CURVE)?)\s*[:=]",
                        re.I)
# a curve's detail block headed by its number in words: "CURVE No. 248" / Degree / Δ / R / TTL / CCL / (RH)
CURVE_BLOCK_HEAD = re.compile(r"^\W*CURVE\s*NO\.?\s*[-.:]?\s*(?P<id>\d+[A-Z]*)\b", re.I)
# curve points named TP1 / J1 / J2 / TP2 (tangent point, junction of transition and circle): "TP1 CH: 984+866.053",
# "J2 AT CH: 985686.053" - the start, the circular part's start and end, the end (as ST / TC / CT / TS)
TPJ = re.compile(r"\b(?P<pt>TP\s*[12]|J\s*[12])\s*(?:AT\s*)?CH\.?\s*[:.]?\s*(?P<ch>\d+(?:\s*\+\s*\d+)?(?:\.\d+)?)", re.I)
TPJ_AS = {"TP1": "ST1", "J1": "TC", "J2": "CT", "TP2": "TS2"}
TRACK = re.compile(r"\b(?:\d\s*(?:ST|ND|RD|TH)|UP|DN|DOWN)\s*LINE\b", re.I)
DEGREE = re.compile(r"\bDEG(?:REE)?\.?(?:\s*OF\s*CURVE)?\s*[:=]\s*(\d+(?:\.\d+)?)", re.I)
CCL_RE = re.compile(r"\bCCL\s*[:=]\s*(\d+(?:\.\d+)?)", re.I)
CROSS_HEAD = re.compile(r"^\W*(?:C/L\s*OF\s*)?(?:EXG?\.?|EX\.|EXISTING|PROP\w*\.?)?\s*(?:BR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*)?"
                        r"(?P<label>(?P<kind>LC|ROB|RUB|FOB)\b\s*[-.]?\s*(?P<num>\d+[A-Z]*)?(?:\s*\([^)]*\))?)", re.I)
# every "label = value" of a level block, whatever the label: "FL = ...", "EXG FL = ...", "FL 3RD LINE = ...",
# "RL UP LINE = ...", "BED LEVEL = ...", "OHFL = ...", "EXG.ROAD LEVEL = ...", "HC = ...", "VC = ..." (a label is words and
# line ordinals; the bridge number before it, "224UP", is not part of it)
BLOCK_KV = re.compile(r"(?<![A-Z0-9.])((?:[A-Z][A-Z.]*|[1-9](?:ST|ND|RD|TH))(?:[ \t]*(?:[A-Z][A-Z./]*|[1-9](?:ST|ND|RD|TH)))*)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                      re.I)
XING_RE = re.compile(r"L-?XING\s*NO\.?\s*(?P<num>\d+[A-Z]*)", re.I)
LONE_PT = re.compile(r"^\W*(?P<pt>ST|TS|TC|CT|SC|CS)\s*(?:AT\s*)?CH\.?\s*[:.]?\s*(?P<ch>\d[\d+.]*)", re.I)


def fmt_ch(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


LINE_RE = re.compile(r"\(\s*(UP|DN)\s*MAIN\s*\d?\s*\)|\b(UP|DN)\s*MAIN\b", re.I)


def _along(w, p):
    return (p[0] - w.c[0]) * w.dir[0] + (p[1] - w.c[1]) * w.dir[1]


def _across(w, p):
    """How far below line w the point p is, in the line's own frame (its next line is about 1.2 text heights down)."""
    return (p[0] - w.c[0]) * -w.dir[1] + (p[1] - w.c[1]) * w.dir[0]


def joined(parts):
    """One Word for a label printed over several lines (its box covers them all)."""
    if len(parts) == 1:
        return parts[0]
    return Word(" ".join(p.text for p in parts), union([p.box for p in parts]), parts[0].dir, parts[0].h)


def status_of(text):
    """Existing or proposed, from what is written before the bridge number ("EX. Br. No.", "C/L OF EXG. BR. NO.")."""
    m = BR_RE.search(text)
    t = (text[:m.start()] if m else text).upper()
    if re.search(r"\bEX(?:G|IST\w*)?\b", t):
        return "existing"
    if re.search(r"\bPROP", t):
        return "proposed"
    return None


def norm_type(t):
    return re.sub(r"[^A-Z0-9]", "", t.upper())


class Bridge:
    def __init__(self, num, status, kind="BR", label=None):
        self.num, self.status, self.kind, self.label = num, status, kind, label or num
        self.labels = []          # [[Word, ...]] every printed copy of the callout, each as its lines
        self.levels = []          # [[Word, ...]] level blocks (BR NO. head + EX. FL / PROP. FL / HFL / BL lines)

    def best(self):
        """The copy to read: a complete one (has the chainage), on fewest lines."""
        return min(self.labels, key=lambda ps: (not AT_CH.search(joined(ps).text), len(ps), -len(joined(ps).text)))

    def fields(self, text=None):
        """span, type, proposal, chainage from a callout's text - "PROP. Br. No. :15, SPAN : 1 X 3.66M. , R.C.C. BOX,
        AT CH : 11384m." or "C/L OF EXG. BR. NO. 575 DN RCC SLAB - 1 X 1.83 + 1 X 3.66 + 1 X 1.83 - BT ROAD PRO TO BE
        EXTENDED AS 1 X 4X4 - RCC BOX (MINOR) AT CH: 1245630.853"."""
        t = text if text is not None else joined(self.best()).text
        clean = lambda s: re.sub(r"^[\s,.;:\-]+|[\s,.;:\-]+$", "", s or "") or None
        ch = AT_CH.search(t)
        n = numbers(ch.group(1)) if ch else []
        prop = PROPOSAL.search(t[BR_RE.search(t).end():] if BR_RE.search(t) else t)
        span = SPAN.search(t)
        typ = None
        if span:                                           # "SPAN : 1 X 3.66M. , R.C.C. BOX, AT CH"
            typ = clean(t[span.end():ch.start()] if ch and ch.start() > span.end() else t[span.end():])
        else:                                              # "BR. NO. 575 DN RCC SLAB - 1 X 1.83 + ..."
            br = BR_RE.search(t)
            after = t[br.end():] if br else t
            span = BARE_SPAN.search(after)
            if span:
                typ = clean(re.sub(r"^\s*(UP|DN)\s*", "", after[:span.start()], flags=re.I))
        if typ and prop and prop.group(0) in typ:
            typ = clean(typ[:typ.find(prop.group(0))])
        return {"span": clean(span.group(1)) if span else None, "type": typ,
                "proposal": clean(prop.group(1)) if prop and not SPAN.search(t) else None, "chainage": n[0] if n else None}

    @property
    def name(self):
        return f"{'EX.' if self.status == 'existing' else 'PROP.' if self.status == 'proposed' else ''} Br. No. {self.num}".strip()


class Curve:
    def __init__(self, num, status, line):
        self.num, self.status, self.line = num, status, line
        self.points = {}          # "TPTC1" -> (chainage, Word)
        self.details = []         # [[Word, ...]] detail blocks (head + Δ / R / TL / CL / Ca / Cd / MSP lines)
        self.hand = None
        self.block_num = None     # the number of a "CURVE No. 248" block found for an unnumbered curve
        self.track = None         # the line it is on, when its block names it ("4TH LINE")

    @property
    def span(self):
        chs = [c for c, _ in self.points.values()]
        return min(chs), max(chs)

    @property
    def name(self):
        num = f"{self.block_num} ({self.num})" if self.block_num else self.num
        return (f"{'existing ' if self.status == 'existing' else ''}curve {num}" + (f" ({self.line} main)" if self.line else "")
                + (f" ({self.track})" if self.track else ""))


class SheetObjects:
    def __init__(self, words):
        self.words = words
        self.bridges, self.curves, self.crossings = [], [], []
        self._bridges()
        self._crossings()
        self._curves()

    def next_line(self, w, ok):
        """The line printed just under w (same direction, one line down, starting or centred where w does)."""
        best, best_a = None, None
        for o in self.words:
            if o is w or o.dir[0] * w.dir[0] + o.dir[1] * w.dir[1] < 0.995 or not ok(o.text):
                continue
            a = _across(w, o.c)
            if not 0.7 * w.h < a < 1.9 * w.h:
                continue
            if abs(_along(w, o.start) - _along(w, w.start)) < 2.5 * w.h or abs(_along(w, o.c)) < 2.5 * w.h:
                if best is None or a < best_a:
                    best, best_a = o, a
        return best

    def lines_from(self, head, ok, stop, most):
        parts = [head]
        while len(parts) < most and not stop(joined(parts).text):
            nxt = self.next_line(parts[-1], ok)
            if nxt is None or nxt in parts:
                break
            parts.append(nxt)
        return parts

    # ------------------------------------------------------------ bridges
    def _bridges(self):
        by = {}
        used = set()
        for w in self.words:
            m = BR_RE.search(w.text)
            if not m:
                continue
            num = bridge_num(m)
            if BR_HEAD.match(w.text):                       # "BR NO. 15" heading a level block
                parts = self.lines_from(w, lambda t: bool(LEVEL_LINE.match(t)), lambda t: False, 8)
                if len(parts) > 1:
                    for s in ("existing", "proposed", None):
                        by.setdefault((num, s), Bridge(num, s))
                    by[(num, None)].levels.append(parts)
                continue
            if id(w) in used:
                continue
            parts = self.lines_from(w, lambda t: not BR_RE.search(t) and not LEVEL_LINE.match(t), lambda t: bool(AT_CH.search(t)), 5)
            used.update(id(p) for p in parts)
            st = status_of(w.text)
            by.setdefault((num, st), Bridge(num, st)).labels.append(parts)
        levels = {k[0]: b.levels for k, b in by.items() if k[1] is None and b.levels}
        for (num, st), b in by.items():
            if b.labels:
                b.levels = levels.get(num, [])
                self.bridges.append(b)
        self.bridges.sort(key=lambda b: (b.fields()["chainage"] or 0, b.status or ""))

    def find_bridges(self, num=None, status=None, kinds=()):
        out = []
        for b in self.bridges:
            if num is not None and b.num != num.upper() and not re.fullmatch(re.escape(num.upper()) + r"(UP|DN)", b.num):
                continue
            if status and b.status != status:
                continue
            f = b.fields()
            t = norm_type((f["type"] or "") + " " + (f["proposal"] or ""))
            if kinds and not all(any(k in t for k in alts) for alts in kinds):
                continue
            out.append(b)
        return out

    # ------------------------------------------------------------ level crossings, ROBs, RUBs, FOBs
    def _crossings(self):
        """Callouts of level crossings and road bridges ("C/L OF EXG. LC 137 BT ROAD", "C/L OF EXG. ROB-15 (NH 552)",
        "C/L OF EXG. BR. NO. LC-226 ...", "'SPL' CLASS L-XING NO. 10 (MANNED)", "PROPOSED ROB AT CH : 11602M.") and
        their level blocks ("EXG. LC 137" / "EXG FL = ..." / "FL = ..."). A numbered bridge that mentions an LC
        ("BR. NO. 578AX(LC-247)") stays a bridge; a note such as "ROB PROPOSED BY IR. SPACE AVAILABLE" is a line of a
        callout, not an item."""
        def head(t):
            if re.match(r"^\s*\(", t):                 # "(RUB/LHS) AT CH: ..." ends a bridge's callout
                return None
            m, x = CROSS_HEAD.match(t), XING_RE.search(t)
            br = BR_RE.search(t)
            if br and re.match(r"\d", br.group(1)):
                return None
            if x:
                return "L-XING", x.group("num").upper(), x.group()
            if m:
                label = re.sub(r"\s+", " ", m.group("label")).strip()
                return m.group("kind").upper(), (m.group("num") or "").upper(), label
            return None

        def is_item(t):                                # starts a new item: not a line of this callout
            h = head(t)
            return bool(h and (h[1] or re.match(r"^\W*C/L\s*OF", t, re.I))) or bool(BR_RE.search(t))

        by, levels, used = {}, {}, set()
        for w in self.words:
            h = head(w.text)
            if not h or id(w) in used:
                continue
            kind, num, label = h
            cl = bool(re.match(r"^\W*C/L\s*OF", w.text, re.I))
            if not cl and not AT_CH.search(w.text):    # a level block: the item's name alone, level lines under it
                parts = self.lines_from(w, lambda t: bool(LEVEL_LINE.match(t)), lambda t: False, 8)
                if len(parts) > 1:
                    levels.setdefault((kind, num), []).append(parts)
                    continue
            if not (num or cl or AT_CH.search(w.text)):
                continue
            parts = self.lines_from(w, lambda t: not is_item(t) and not LEVEL_LINE.match(t), lambda t: bool(AT_CH.search(t)), 5)
            text = joined(parts).text
            ch = AT_CH.search(text)
            if not ch:
                continue
            used.update(id(p) for p in parts)
            pre = text[:text.upper().find(kind) if kind in text.upper() else 0].upper()
            st = "existing" if re.search(r"\bEX(?:G|IST\w*)?\b", pre) else "proposed" if re.search(r"\bPROP", pre) else None
            key = (kind, num or "@" + ch.group(1).strip(), st)
            if key not in by:
                by[key] = Bridge(num or None, st, kind, label)
            by[key].labels.append(parts)
        for (kind, num, st), b in by.items():
            if not num.startswith("@"):
                b.levels = levels.get((kind, num), [])
            else:                                      # a block headed "EXG. BR. NO. LC" with no number: the sheet's one
                lone = [k for k in by if k[0] == kind and k[1].startswith("@")]       # unnumbered item of that kind
                b.levels = levels.get((kind, ""), []) if len(lone) == 1 else []
            self.crossings.append(b)
        self.crossings.sort(key=lambda b: numbers(AT_CH.search(joined(b.best()).text).group(1))[0])

    # ------------------------------------------------------------ curves
    def _curves(self):
        by = {}
        for w in self.words:
            m = CURVE_PT.search(w.text)
            if not m:
                continue
            st = "existing" if m.group("ex") else "proposed"
            lm = LINE_RE.search(w.text)
            line = (lm.group(1) or lm.group(2)).upper() if lm else None
            ch = numbers(m.group("ch"))
            if not ch:
                continue
            key = (m.group("id").upper(), st, line)
            c = by.setdefault(key, Curve(*key))
            pt = m.group("pt").upper() + (m.group("n") or "")
            if pt not in c.points:
                c.points[pt] = (ch[0], w)
        self._unnumbered(by)
        for w in self.words:
            m = CURVE_HEAD.match(w.text)
            if not m:
                continue
            parts = self.lines_from(w, lambda t: bool(CURVE_LINE.match(t)), lambda t: False, 12)
            if len(parts) < 2:
                continue
            num = m.group("id").upper()
            line = (m.group("line") or m.group("line2") or "").upper() or None
            for c in by.values():
                if c.num == num and (line is None or c.line in (None, line)):
                    c.details.append(parts)
                    c.hand = c.hand or m.group("hand").upper()
        self._tpj_curves(by)
        self._blocks(by)
        self.curves = sorted(by.values(), key=lambda c: c.span)

    def _tpj_curves(self, by):
        """Curves printed as one block in the plan: the track above it ("4TH LINE"), "CURVE No. 206", Degree, Δ, R, TL,
        then TP1 / J1 / J2 / TP2 with their chainages, TRL, CCL. Each is a curve of that number and track, its points
        from the block (the same block printed again is one curve)."""
        ok = lambda t: bool(CURVE_LINE.match(t) or TPJ.match(t.strip()) or re.match(r"^\W*PROPOSED\s+SPEED", t, re.I))
        for w in self.words:
            m = CURVE_BLOCK_HEAD.match(w.text)
            if not m:
                continue
            parts = self.lines_from(w, ok, lambda t: False, 16)
            pts = {}
            for p in parts:
                t = TPJ.match(p.text.strip())
                if t:
                    ch = t.group("ch").replace(" ", "")
                    v = (int(ch.split("+")[0]) * 1000 + float(ch.split("+")[1])) if "+" in ch else float(ch)
                    pts[TPJ_AS[re.sub(r"\s", "", t.group("pt")).upper()]] = (v, p)
            if not ("ST1" in pts and "TS2" in pts):
                continue
            num = m.group("id").upper()
            above = [o for o in self.words if TRACK.search(o.text) and abs(o.c[0] - w.c[0]) < 6 * w.h and 0 < w.c[1] - o.c[1] < 3 * w.h]
            track = TRACK.search(min(above, key=lambda o: w.c[1] - o.c[1]).text).group().upper() if above else None
            key = ("@blk" + num, "proposed", track)
            if key in by:
                continue
            c = Curve(num, "proposed", None)
            c.points, c.track = pts, track
            c.details.append(parts)
            by[key] = c

    @property
    def main_track(self):
        """The sheet's own proposed line ("4TH LINE"): the one its band rows name most ("PROP. 4TH LINE FL")."""
        from collections import Counter
        n = Counter(re.sub(r"\s", "", m.group(1)).upper() for w in self.words
                    for m in [re.search(r"PROP\.?\s*(\d\s*(?:ST|ND|RD|TH))\s*LINE", w.text, re.I)] if m)
        return f"{n.most_common(1)[0][0]} LINE" if n else None

    def _blocks(self, by):
        """Detail blocks headed "CURVE No. 248" (Degree / Δ / R / TTL / CCL): to the curve of that number, else to the
        curve its CCL (circular length) is - the CT chainage minus the TC chainage printed for that curve - and, when two
        curves have that length, the one at the block's place along the sheet. A block that fits no curve is left out."""
        for w in self.words:
            m = CURVE_BLOCK_HEAD.match(w.text)
            if not m:
                continue
            parts = self.lines_from(w, lambda t: bool(CURVE_LINE.match(t)), lambda t: False, 12)
            if len(parts) < 2:
                continue
            num = m.group("id").upper()
            named = [c for c in by.values() if c.num == num]
            if named:
                for c in named:
                    c.details.append(parts)
                continue
            ccl = CCL_RE.search(joined(parts).text)
            if not ccl:
                continue

            def circ(c):
                a = next((c.points[p][0] for p in ("TC", "SC", "TPCC1") if p in c.points), None)
                b = next((c.points[p][0] for p in ("CT", "CS", "TPCC2") if p in c.points), None)
                return None if a is None or b is None else b - a
            fit = [c for c in by.values() if circ(c) is not None and abs(circ(c) - float(ccl.group(1))) < 0.02]
            if len(fit) > 1:                          # the same length twice: the one nearest the block along the sheet
                along = lambda c: min(abs(pw.c[0] - w.c[0]) for _, pw in c.points.values())
                fit = [min(fit, key=along)]
            for c in fit:
                c.details.append(parts)
                c.block_num = num

    def find_curves(self, num):
        """The curves numbered num: by their points' number, or by the "CURVE No." block found for them."""
        num = (num or "").upper()
        return [c for c in self.curves if (c.num or "").upper() == num or (c.block_num or "").upper() == num]

    def curve_blocks(self, num):
        """Every "CURVE No. <num>" detail block on the sheet ([[Word]]), whether or not its curve's points are on it."""
        out = []
        for w in self.words:
            m = CURVE_BLOCK_HEAD.match(w.text)
            if m and m.group("id").upper() == (num or "").upper():
                parts = self.lines_from(w, lambda t: bool(CURVE_LINE.match(t)), lambda t: False, 12)
                if len(parts) >= 2:
                    out.append(parts)
        return out

    def degree(self, c):
        """(degree as printed, the block it is in) of a curve, or (None, None) - from its detail block."""
        for parts in c.details:
            m = DEGREE.search(joined(parts).text)
            if m:
                return m.group(1), parts
        return None, None

    def _unnumbered(self, by):
        """Curve points printed without a curve number ("ST AT CH: 1245512.852", "TC AT CH: ...", "CT AT CH: ...") are
        grouped along the line: transition end, start of the circular curve (TC / SC), its end (CT / CS), transition
        end. Each group is one curve, named after its circular part."""
        pts = {}
        for w in self.words:
            m = LONE_PT.match(w.text)
            if m:
                n = numbers(m.group("ch"))
                if n:
                    pts.setdefault((m.group("pt").upper(), round(n[0], 3)), w)
        seq = sorted(((ch, pt, w) for (pt, ch), w in pts.items()), key=lambda t: t[0])
        cur = None
        for ch, pt, w in seq:
            if pt in ("TC", "SC"):                             # the circular part starts: a new curve
                cur = Curve(None, "proposed", None)
                prev = [t for t in seq if t[0] < ch and t[1] not in ("TC", "SC", "CT", "CS")]
                if prev and not any(prev[-1][0] < t[0] < ch for t in seq if t[1] in ("CT", "CS")):
                    cur.points[prev[-1][1] + "1"] = (prev[-1][0], prev[-1][2])
                cur.points[pt] = (ch, w)
                by[("@" + fmt_ch(ch), "proposed", None)] = cur
            elif pt in ("CT", "CS"):
                if cur is None or any(p in cur.points for p in ("CT", "CS")):
                    cur = Curve(None, "proposed", None)        # its start is on the previous sheet
                    by[("@" + fmt_ch(ch), "proposed", None)] = cur
                cur.points[pt] = (ch, w)
            elif cur is not None and any(p in cur.points for p in ("CT", "CS")) and not any(k.endswith("2") for k in cur.points):
                cur.points[pt + "2"] = (ch, w)                 # the transition end after the circular part
                cur = None
        for key, c in list(by.items()):
            if c.num is None:
                circ = [c.points[p][0] for p in ("TC", "SC", "CT", "CS") if p in c.points]
                c.num = f"with circular part from CH {fmt_ch(min(circ))}" + (f" to {fmt_ch(max(circ))}" if len(circ) > 1 else "")

    def curves_near(self, ch):
        """{status: [(distance, Curve)]}: the curves whose extent holds ch (distance 0), else the nearest one."""
        out = {}
        for st in ("proposed", "existing"):
            cs = [c for c in self.curves if c.status == st and c.points]
            if not cs:
                continue
            dist = lambda c: 0 if c.span[0] <= ch <= c.span[1] else min(abs(ch - c.span[0]), abs(ch - c.span[1]))
            on = [c for c in cs if dist(c) == 0]
            out[st] = [(0, c) for c in on] if on else [min(((dist(c), c) for c in cs), key=lambda t: t[0])]
        return out
