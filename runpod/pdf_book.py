"""Every page of a multi-page PDF, asked as one drawing: each question goes to the page(s) that print what it is about.

    book = Book("drawing.pdf", out_dir, ask)       # ask(images, question) -> model answer
    print(book.answer("EXG. BR. NO. 320UP"))       # -> answered from the page that prints that bridge

All pages are indexed from their text layer (no rendering, no model: about a second a page): the bridges and curves
on each (sheet_objects.py), the chainages its data band covers (band_table.py), its grade-point labels. A question is
sent to the pages that hold what it names - a bridge or curve by its number, a chainage by the band that covers it, a
grade point by its chainage label, anything else by the best text match - and only those pages are rendered and read
by the model. Every answer says which page it is from. A bridge printed again at the start of the next sheet is
answered from both pages; a list of bridges covers every page.
"""
import re
from pathlib import Path

import band_table
import find_crop as F


class Book:
    def __init__(self, path, out_dir, ask, log=print):
        import pymupdf
        self.path, self.out = Path(path), Path(out_dir)
        n = len(pymupdf.open(path))
        log(f"{self.path.name}: {n} pages - indexing every page's text (bridges, curves, band chainages) ...")
        self.sheets = [F.open_sheet(path, self.out / f"page{i + 1}", i, lazy=True) for i in range(n)]
        self.finders = [F.Finder(s, ask, self.out / f"page{i + 1}") for i, s in enumerate(self.sheets)]
        self.note = self.summary()

    def summary(self):
        rows = []
        for i, f in enumerate(self.finders):
            bands = f.bands()
            span = (f"CH {F.fmt(min(b.cols[0][1] for b in bands))} - {F.fmt(max(b.cols[-1][1] for b in bands))}"
                    if bands else "no data band found")
            nb = len({b.num for b in f.objects().bridges})
            nc = len(f.objects().curves)
            rows.append(f"  page {i + 1}: {span}; {nb} bridge(s), {nc} curve(s)")
        return f"{self.path.name}: {len(self.finders)} pages\n" + "\n".join(rows)

    # ------------------------------------------------------------ which pages a question is about
    def pages_for(self, q):
        """(pages, why): the 0-based pages to answer from, or [] with the reason nothing on any page fits."""
        ql = q.lower()
        fs = self.finders
        everywhere = range(len(fs))
        nums = [n.upper() for n in F.BRIDGE_REF.findall(q)]
        if nums:
            hit = [i for i in everywhere if any(fs[i].objects().find_bridges(num=n) for n in nums)]
            return hit, f"no bridge {', '.join(nums)} is printed on any of the {len(fs)} pages of this PDF"
        if re.search(r"\b(list|all|how many|which|show|every|count)\b", ql) and re.search(r"\bbr(?:idge)?s?\b", ql):
            return [i for i in everywhere if fs[i].objects().bridges], "no bridge callouts are printed in this PDF"
        cid = F.CURVE_ID.search(q)
        if cid and (re.search(r"\bcurves?\b", ql) or not band_table.asks_row(q)):
            num = cid.group(1).upper()
            hit = [i for i in everywhere if any(c.num == num for c in fs[i].objects().curves)]
            return hit, f"no curve {num} is printed on any of the {len(fs)} pages of this PDF"
        # a chainage: the pages whose band covers it, else whose grade-point / chainage labels carry it
        chs = [ch for ch, w in fs[0].places(q) if w is None] if fs else []
        if chs:
            inside = lambda b, ch: b.cols[0][1] < ch < b.cols[-1][1]
            covers = [i for i in everywhere if any(b.covers(ch) for b in fs[i].bands() for ch in chs)]
            strict = [i for i in covers if any(inside(b, ch) for b in fs[i].bands() for ch in chs)]
            labels = [i for i in everywhere if any(re.search(r"\bCH\b|CH\s*[:.]", w.text, re.I) and
                                                   any(abs(a - b) < 0.6 for a in chs for b in F.numbers(w.text))
                                                   for w in fs[i].s.words)]
            if F.GRADE_WORDS.search(q) and labels:
                return labels, ""
            if strict or covers:
                return strict or covers, ""
            if labels:
                return labels, ""
            return [], (f"CH {', '.join(F.fmt(c) for c in chs)} is not on any of the {len(fs)} pages of this PDF "
                        f"(their bands cover: {self.ranges()})")
        # anything else: the page(s) whose text matches best
        scored = [(max((s for s, _ in fs[i].matches(q)), default=0), i) for i in everywhere]
        top = max((s for s, _ in scored), default=0)
        if top <= 0:
            return [], "nothing in the question is printed on any page of this PDF"
        return [i for s, i in scored if s == top][:3], ""

    def ranges(self):
        out = []
        for i, f in enumerate(self.finders):
            for b in f.bands():
                out.append(f"page {i + 1}: {F.fmt(b.cols[0][1])}-{F.fmt(b.cols[-1][1])}")
        return "; ".join(out) or "no data bands found"

    # ------------------------------------------------------------ answering
    def is_list(self, q):
        ql = q.lower()
        return bool(re.search(r"\b(list|all|how many|which|show|every|count)\b", ql) and re.search(r"\bbr(?:idge)?s?\b", ql)
                     and not F.BRIDGE_REF.search(q) and not re.search(r"\bcurves?\b", ql))

    def list_bridges(self, q):
        """The bridges asked for on every page, in page order: a bridge printed again at the start of the next sheet
        is listed (and read) once, with all the pages it is on."""
        found, what = [], ""
        for i, f in enumerate(self.finders):
            bs, what = f.bridges_asked(q)
            found += [(i, b) for b in bs]
        if not found:
            return f"No {what + ' ' if what else ''}bridges are printed on any of the {len(self.finders)} pages of this PDF."
        pages = {}
        for i, b in found:
            pages.setdefault((b.num, b.status), []).append(i + 1)
        out, done = [], set()
        for i, b in found:
            key = (b.num, b.status)
            if key in done:
                continue
            done.add(key)
            where = pages[key]
            tag = f"page {where[0]}" if len(where) == 1 else f"pages {', '.join(map(str, where))}"
            out.append(f"  [{tag}] {self.finders[i].bridge_line(b)}")
        return (f"{len(out)} {what + ' ' if what else ''}bridge(s) in this PDF ({len(self.finders)} pages), in page "
                f"order (each callout read by the model from its own crop):\n" + "\n".join(out))

    def answer(self, q):
        if self.is_list(q):
            return self.list_bridges(q)
        pages, why = self.pages_for(q)
        if not pages:
            return why[0].upper() + why[1:] + " (nothing is guessed)."
        out = []
        for i in pages:
            a = self.finders[i].answer(q)
            out.append(f"[page {i + 1}]\n{a}")
        return "\n\n".join(out)
