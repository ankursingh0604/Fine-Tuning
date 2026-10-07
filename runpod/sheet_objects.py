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

# the number with at most two letters after it (575A, 578AX, 224UP) - a reading without spaces ("575DNRCCSLAB") keeps 575
BR_RE = re.compile(r"\bBR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*(\d+(?:[A-Z]{1,2}(?![A-Z]))?)", re.I)
BR_HEAD = re.compile(r"^\W*(?:EX(?:G|IST\w*)?\.?\s*|PROP\w*\.?\s*)?BR(?:IDGE)?\.?\s*NO\.?\s*[:.\-]?\s*\d+[A-Z]*\s*(?:UP|DN)?\W*$", re.I)
LEVEL_LINE = re.compile(r"^\W*(?:EX(?:G|IST(?:ING)?)?\.?|PROP(?:OSED)?\.?)?\s*(?:FL|F\.L|HFL|H\.F\.L|BL|B\.L|SFL|RL|MIN|BED|DSL|LWL)\b", re.I)
AT_CH = re.compile(r"\bAT\s*CH\.?\s*[:.]?\s*(\d[\d\s+.,]*)", re.I)
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
CURVE_LINE = re.compile(r"^\W*(?:Δ|∆|DELTA|R|TL|CL|TRL|SHIFT|CA|CD|MSP|V|LS|L)\s*[:=]", re.I)
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
    def __init__(self, num, status):
        self.num, self.status = num, status
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

    @property
    def span(self):
        chs = [c for c, _ in self.points.values()]
        return min(chs), max(chs)

    @property
    def name(self):
        return f"{'existing ' if self.status == 'existing' else ''}curve {self.num}" + (f" ({self.line} main)" if self.line else "")


class SheetObjects:
    def __init__(self, words):
        self.words = words
        self.bridges, self.curves = [], []
        self._bridges()
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
            num = m.group(1).upper()
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
        self.curves = sorted(by.values(), key=lambda c: c.span)

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
