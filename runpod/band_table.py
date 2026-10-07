"""The data bands of any L-section sheet, found from where its text is printed (PDF text layer or OCR) - no layout is
assumed. Used by find_crop.py.

    bands = find_bands(sheet.words)        # [Band]: rows (named by their printed headings) and the chainage of every column
    rows = match_rows(bands, "cut/fill")   # the rows a question asks about

A band is a table under the profile: one row per quantity (ground level, formation level, cut / fill, ...), one column
per chainage. It is found as:
  rows      numbers printed in a line across the sheet, many of them (a band row has a value every few metres);
  chainage  the row whose numbers rise steadily from left to right (metres or km+m), the x of each column -> chainage;
  headings  the text printed to the left of the band, each line given to the row it stands level with.
"""
import math
import re

NUM = re.compile(r"^-?\d+(?:\.\d+)?$|^\d+\s*\+\s*\d+(?:\.\d+)?$")
MIN_VALUES = 8                     # a band row has many values; fewer is a stray line of numbers


def value(text):
    t = re.sub(r"\s", "", text)
    if "+" in t:
        a, b = t.split("+")
        return int(a) * 1000 + float(b)
    return float(t)


class Row:
    def __init__(self, words):
        self.words = sorted(words, key=lambda w: w.c[0])
        self.y = sorted(w.c[1] for w in words)[len(words) // 2]
        self.h = sorted(w.h for w in words)[len(words) // 2]
        self.heading = ""
        self.lines = []            # the heading's printed lines (Words)

    @property
    def x0(self):
        return self.words[0].box[0]

    @property
    def x1(self):
        return self.words[-1].box[2]

    def __repr__(self):
        return f"Row({self.heading!r}, y={self.y:.0f}, {len(self.words)} values)"


class Band:
    """Rows sharing one chainage row. `chainage(x)` gives the chainage at a page x (exact on a printed column)."""

    def __init__(self, chain, rows):
        self.chain, self.rows = chain, rows
        self.cols = [(w.c[0], value(w.text)) for w in chain.words]

    def chainage(self, x):
        cols = self.cols
        near = min(cols, key=lambda c: abs(c[0] - x))
        if abs(near[0] - x) < 0.6 * self.chain.h:
            return near[1]
        left = [c for c in cols if c[0] < x]
        right = [c for c in cols if c[0] > x]
        (xa, ca), (xb, cb) = (left[-1], right[0]) if left and right else (cols[0], cols[1]) if not left else (cols[-2], cols[-1])
        return ca + (cb - ca) * (x - xa) / (xb - xa)

    def covers(self, ch):
        lo, hi = self.cols[0][1], self.cols[-1][1]
        return lo - 1e-6 <= ch <= hi + 1e-6

    def values(self, row):
        """(chainage, Word) for every value printed in the row, in chainage order."""
        return sorted(((self.chainage(w.c[0]), w) for w in row.words), key=lambda t: t[0])


def _rows(words):
    nums = sorted((w for w in words if NUM.match(w.text.strip())), key=lambda w: w.c[1])
    groups, cur = [], []
    for w in nums:
        if cur and w.c[1] - cur[-1].c[1] > 0.5 * w.h:
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    rows = []
    for g in groups:
        # one line of numbers can hold two tables side by side; a band row's values stand at most a few text heights apart
        g = sorted(g, key=lambda w: w.c[0])
        run = [g[0]]
        for w in g[1:]:
            if w.c[0] - run[-1].c[0] > 12 * w.h:
                if len(run) >= MIN_VALUES:
                    rows.append(Row(run))
                run = []
            run.append(w)
        if len(run) >= MIN_VALUES:
            rows.append(Row(run))
    return rows


def _is_chainage(row):
    """Values rising steadily left to right, evenly enough to be a chainage scale."""
    try:
        v = [value(w.text) for w in row.words]
    except ValueError:
        return False
    if len(v) < MIN_VALUES or any(b <= a for a, b in zip(v, v[1:])):
        return False
    # on a straight line against x (drawings place columns to the nearest point, so spacing wobbles a little): each
    # value within a third of a column step of the line through the whole row
    x = [w.c[0] for w in row.words]
    n = len(v)
    mx, mv = sum(x) / n, sum(v) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return False
    k = sum((a - mx) * (b - mv) for a, b in zip(x, v)) / sxx
    steps = sorted(b - a for a, b in zip(v, v[1:]))
    step = steps[len(steps) // 2]
    # a level row on a steady grade also rises in a straight line, but by millimetres; columns are metres apart
    return step >= 1 and k > 0 and sum(1 for a, b in zip(x, v) if abs(mv + k * (a - mx) - b) < step / 3) >= 0.9 * n


def find_bands(words):
    rows = _rows(words)
    chains = [r for r in rows if _is_chainage(r)]
    if not chains:
        return []
    # each row belongs to the chainage row it shares the most x with (the nearest one on a tie: two panels on a sheet)
    by = {id(c): [] for c in chains}
    for r in rows:
        if r in chains:
            continue
        best = max(chains, key=lambda c: (min(r.x1, c.x1) - max(r.x0, c.x0), -abs(r.y - c.y)))
        if min(r.x1, best.x1) - max(r.x0, best.x0) > 0:
            by[id(best)].append(r)
    bands = [Band(c, sorted(by[id(c)] + [c], key=lambda r: r.y)) for c in chains]
    _headings(words, bands)
    return bands


def _headings(words, bands):
    """Text printed left of the band, each line given to the row it stands level with (within half the band's row
    spacing - a row with no values printed leaves a wider gap, whose heading must not join its neighbour's)."""
    for b in bands:
        x0 = min(r.x0 for r in b.rows)
        ys = sorted(r.y for r in b.rows)
        span = (min(ys) - 200 * b.chain.h, max(ys) + 200 * b.chain.h)
        heads = [w for w in words if w.box[2] < x0 - 0.5 * w.h and x0 - w.c[0] < 120 * w.h and span[0] < w.c[1] < span[1]
                 and not NUM.match(w.text.strip())]
        gaps = [b2 - a for a, b2 in zip(ys, ys[1:])]
        half = min(gaps) / 2 if gaps else 3 * b.chain.h
        for r in b.rows:
            r.lines = sorted((w for w in heads if abs(w.c[1] - r.y) < half), key=lambda w: (w.c[1], w.c[0]))
            r.heading = " ".join(w.text for w in r.lines).strip()


# ======================================================================== which rows a question asks about
SYN = [  # question words -> words a row heading uses for the same thing
    (r"\bcut\b|\bfill\b|cutting|\bbank\b|embankment|cut\s*/\s*fill", ["cut", "fill", "cutting", "bank", "pfl-ogl", "pfl - ogl"]),
    (r"ground|\bogl\b|\bgl\b|\bngl\b", ["ground", "ogl", "gl", "ngl"]),
    (r"formation|\bf\.?l\b|\bpfl\b|\befl\b", ["formation", "fl", "f.l", "pfl", "efl"]),
    (r"\brail\b|\br\.?l\b|\bprl\b|\berl\b", ["rail", "rl", "r.l", "prl", "erl"]),
    (r"track\s*dist|distance|\bc/c\b|centre", ["track", "distance", "c/c", "centre", "center"]),
    (r"differ|\bdiff\b", ["difference", "differance", "diff"]),
    (r"prop|new", ["prop", "propose", "proposed"]),
    (r"exist|\bexg\b|\bex\b|old", ["exg", "existing", "exist", "ex"]),
    (r"\bup\b", ["up"]),
    (r"\bdn\b|\bdown\b", ["dn", "down"]),
    (r"chainage|\bch\b", ["chainage"]),
    (r"\bkm\b|kilomet", ["km"]),
]
KIND = 6                           # the first SYN entries say WHAT is asked (a row must match one of them)


def _has(heading, word):
    """The heading uses the word; a long word may run into the next one ("RailLevel")."""
    tail = r"(?![a-z0-9])" if len(word) <= 3 else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + tail, heading) is not None


def match_rows(bands, q):
    """(rows, about): `rows` = [(Band, Row)] the question asks about - the rows whose heading says what is asked (cut /
    fill, ground level, formation, rail, ...) AND everything the question specifies (proposed / existing, up / dn, the
    line number; checked against the row's heading and the band's chainage heading, which names the band's line).
    Several rows can tie (FL of DN Main 2 and of UP Main 2): all are returned. `about` = every row of that kind, so
    when nothing on the sheet fits what was specified the answer can say which rows there are."""
    ql = q.lower()
    asked = [(i, words) for i, (pat, words) in enumerate(SYN) if re.search(pat, ql)]
    if not any(i < KIND for i, _ in asked):
        return [], []
    nums = {a or b for a, b in re.findall(r"\b(\d)(?:st|nd|rd|th)?\s*line\b|\bmain\s*(\d)\b", ql)}
    num_in = lambda n, h: re.search(rf"(?<![\d.]){n}(?:st|nd|rd|th)?(?![\d.])", h) is not None
    scored, about = [], []
    for b in bands:
        for r in b.rows:
            h = r.heading.lower()
            if not h or r is b.chain:
                continue
            kind = sum(1 for i, ws in asked if i < KIND and any(_has(h, w) for w in ws))
            if not kind:
                continue
            about.append((b, r))
            ctx = h + " " + b.chain.heading.lower()
            if not all(any(_has(ctx, w) for w in ws) for i, ws in asked if i >= KIND) or not all(num_in(n, ctx) for n in nums):
                continue                                         # asked for something this row is not
            qual = sum(1 for i, ws in asked if i >= KIND and any(_has(h, w) for w in ws)) + sum(1 for n in nums if num_in(n, h))
            # a heading naming things the question did not ask for is a weaker match (e.g. "FL" asked, "(FL - GL)" row)
            extra = sum(1 for i, (pat, ws) in enumerate(SYN[:KIND]) if i not in {j for j, _ in asked} and any(_has(h, w) for w in ws))
            scored.append(((kind, qual, -extra), b, r))
    if not scored:
        return [], about
    top = max(s for s, _, _ in scored)
    return [(b, r) for s, b, r in scored if s == top], about


def asks_row(q):
    """The question asks for something a band row holds (cut / fill, a level, a difference, a track distance)."""
    return any(re.search(pat, q.lower()) for pat, _ in SYN[:KIND])
