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

HELP = """Things you can ask (each answered from the page that prints it)

  Bridges     list all bridges · list all RCC bridges · list existing pipe bridges
              bridge 224 details in csv (the RCC box template layout) · all bridge details in csv
              existing bridge 320UP · chainage of existing bridge 16 · span of bridge 17 · FL of bridge 575
  Curves      curve 8 · which curve is near bridge 15 · which bridge is near curve 8 · curve near CH 12000
  Gradients   gradient at CH 11540
  Band        FL at 11275 · cut or fill at C 8 TPCC2 · ground level at bridge 15 · track distance at 1246000
  Anything    speed on curve 1 · ROB at CH 11602"""


class Book:
    def __init__(self, path, out_dir, ask, log=print):
        import pymupdf
        self.path, self.out = Path(path), Path(out_dir)
        self.is_pdf = self.path.suffix.lower() == ".pdf"
        n = len(pymupdf.open(path)) if self.is_pdf else 1     # an image is one sheet (its text from the local OCR)
        log(f"{self.path.name}: {n} page{'s' if n > 1 else ''} - indexing every page's text (bridges, curves, band chainages) ...")
        self.sheets = ([F.open_sheet(path, self.out / f"page{i + 1}", i, lazy=True) for i in range(n)] if self.is_pdf
                       else [F.open_sheet(path, self.out / "page1")])
        self.finders = [F.Finder(s, ask, self.out / f"page{i + 1}") for i, s in enumerate(self.sheets)]
        self.note = self.summary()

    def suggestions(self):
        """A few questions this document can answer, made from what it prints (for a chat UI's suggestion chips)."""
        out = []
        bridges = [b for f in self.finders for b in f.objects().bridges]
        curves = [c for f in self.finders for c in f.objects().curves if c.num and not c.num.startswith("with")]
        bands = [b for f in self.finders for b in f.bands()]
        if bridges:                                     # worded without stops or dashes (they are shown as chips)
            out.append("List all bridges")
            b = bridges[min(2, len(bridges) - 1)]
            st = {"existing": "existing ", "proposed": "proposed "}.get(b.status, "")
            out.append(f"Chainage of {st}bridge {b.num}")
            if curves:
                out.append(f"Which curve is near bridge {b.num}")
        if curves:
            out.append(f"Which bridge is near curve {curves[0].num}")
        if bands:
            b = bands[0]
            mid = b.cols[len(b.cols) // 2][1] + (b.cols[1][1] - b.cols[0][1]) / 2 if len(b.cols) > 1 else b.cols[0][1]
            out.append(f"Cut or fill at {F.fmt(mid)}")
        return out[:4]

    @property
    def where(self):
        n = len(self.finders)
        return "on this sheet" if n == 1 else f"on any of the {n} pages of this PDF"

    def summary(self):
        rows = []
        for i, f in enumerate(self.finders):
            bands = f.bands()
            span = (f"CH {F.fmt(min(b.cols[0][1] for b in bands))} - {F.fmt(max(b.cols[-1][1] for b in bands))}"
                    if bands else "no data band found")
            nb = len({b.num for b in f.objects().bridges})
            nc = len(f.objects().curves)
            rows.append(f"  page {i + 1}: {span}; {nb} bridge(s), {nc} curve(s)")
        n = len(self.finders)
        return f"{self.path.name}: {n} page{'s' if n > 1 else ''}\n" + "\n".join(rows)

    # ------------------------------------------------------------ which pages a question is about
    def pages_for(self, q):
        """(pages, why): the 0-based pages to answer from, or [] with the reason nothing on any page fits."""
        ql = q.lower()
        fs = self.finders
        everywhere = range(len(fs))
        nums = F.bridge_nums(q)
        if nums:
            hit = [i for i in everywhere if any(fs[i].objects().find_bridges(num=n) for n in nums)]
            return hit, f"no bridge {', '.join(nums)} is printed {self.where}"
        if re.search(r"\b(list|all|how many|which|show|every|count)\b", ql) and re.search(r"\bbr(?:idge)?s?\b", ql):
            return [i for i in everywhere if fs[i].objects().bridges], "no bridge callouts are printed in this PDF"
        cid = F.CURVE_ID.search(q)
        if cid and (re.search(r"\bcurves?\b", ql) or not band_table.asks_row(q)):
            num = cid.group(1).upper()
            hit = [i for i in everywhere if fs[i].objects().find_curves(num) or fs[i].objects().curve_blocks(num)]
            return hit, f"no curve {num} is printed {self.where}"
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
            return [], (f"CH {', '.join(F.fmt(c) for c in chs)} is not {self.where} "
                        f"(their bands cover: {self.ranges()})")
        # anything else: the page(s) whose text matches best
        scored = [(max((s for s, _ in fs[i].matches(q)), default=0), i) for i in everywhere]
        top = max((s for s, _ in scored), default=0)
        if top <= 0:
            return [], f"nothing in the question is printed {self.where}"
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
        """The JSON list of bridges (bridge_list.py) over every page: an item printed again at the start of the next
        sheet is listed once; readings that differ from the PDF name their page."""
        import bridge_list
        rows, seen = [], set()
        for i, f in enumerate(self.finders):
            for g, w in bridge_list.rows(f, q):
                key = (re.sub(r"\s", "", (w["br_no"] or "").upper()), w["chainage"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append((g, w, f"page {i + 1} · " if len(self.finders) > 1 else ""))
        rows.sort(key=lambda t: bridge_list.chainage_value(t[1]["chainage"]))
        name = Path(getattr(self, "display_name", self.path.name)).stem
        return F.export_bridges(rows, self.out / "exports" / f"{name}_bridges.json", q, self)

    def list_bridges_text(self, q):
        """The bridges asked for on every page as text, in page order: a bridge printed again at the start of the
        next sheet is listed (and read) once, with all the pages it is on."""
        found, what = [], ""
        for i, f in enumerate(self.finders):
            bs, what = f.bridges_asked(q)
            found += [(i, b) for b in bs]
        if not found:
            return f"No {what + ' ' if what else ''}bridges are printed {self.where}."
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

    def curve_near_bridge(self, q):
        """'Which curve is near bridge 264' over every page: the bridge's chainage from the page that prints it, the
        curves from all pages (a bridge's page need not print the curve next to it)."""
        out = []
        for num in F.bridge_nums(q):
            where = [(i, b) for i, f in enumerate(self.finders) for b in f.objects().find_bridges(num=num)]
            if not where:
                out.append(f"No bridge {num} is printed {self.where} (nothing is guessed).")
                continue
            i, b = where[0]
            f = self.finders[i]
            read, _ = f.read_parts(b.best(), f"br_{b.name}")
            got, ok = f.check_fields(b, read, ("chainage",))
            ch = got["chainage"]
            if ch is None:
                out.append(f"{b.name}: its chainage could not be read (\"{read}\"), so the curve near it cannot be found.")
                continue
            # every curve of every page with its distance from the bridge; the nearest first and, at the same distance
            # (two lines curving at one place: "4TH LINE" and "3RD LINE"), the sheet's own line first
            main = f.objects().main_track
            near, seen = [], set()
            for k, g in enumerate(self.finders):
                for st, found in g.objects().curves_near(ch).items():
                    for d, c in found:
                        key = (c.block_num or c.num, c.track, round(c.span[0], 3))
                        if key not in seen:
                            seen.add(key)
                            near.append((d, c.track is not None and c.track != main, k, c))
            near.sort(key=lambda t: (t[0], t[1]))
            head = f"[page {i + 1}] {b.name} at CH {F.fmt(ch)}{ok}:"
            if not near:
                out.append(f"{head}\n  no curve is printed {self.where}.")
                continue
            lines = []
            for d, _, k, c in near[:1] + [t for t in near[1:] if t[0] == 0 and near[0][0] == 0]:
                g = self.finders[k]
                deg, _ = g.objects().degree(c)
                line = g.curve_position(c, ch, d) + (f" [page {k + 1}]" if k != i else "")
                if deg:
                    line += f"; degree of curve {deg}" + (" (the bridge is on this curve)" if d == 0 else "")
                lines.append(("" if not lines else "also ") + line)
            out.append(head + "".join(f"\n  {x}" for x in lines))
        return "\n\n".join(out)

    def bridge_csv(self, q):
        """'Bridge 224 details in csv' over every page: the bridge(s) in the RCC box template layout (bridge_card.py),
        a bridge printed again at the start of the next sheet once; every bridge in the PDF when none is named."""
        import bridge_list
        nums = F.bridge_nums(q)
        items = []
        for i, f in enumerate(self.finders):
            if nums and not any(f.objects().find_bridges(num=n) for n in nums):
                continue
            for c, g, w in f.bridge_cards(nums):
                # the same bridge printed on two pages (once without its chainage): one card, the fuller reading
                no = re.sub(r"\s", "", (w["br_no"] or "").upper())
                same = next((k for k, it in enumerate(items) if re.sub(r"\s", "", (it[2]["br_no"] or "").upper()) == no
                             and (not w["chainage"] or not it[2]["chainage"] or w["chainage"] == it[2]["chainage"])), None)
                full = lambda row: sum(v is not None for v in row.values())
                item = (c, g, w, f"page {i + 1} · " if len(self.finders) > 1 else "")
                if same is None:
                    items.append(item)
                elif full(w) > full(items[same][2]):
                    items[same] = item
        if not items:
            return f"No bridge {', '.join(nums)} is printed {self.where} (nothing is guessed)." if nums else \
                f"No bridge callouts are printed {self.where}."
        items.sort(key=lambda t: bridge_list.chainage_value(t[2]["chainage"]))
        name = Path(getattr(self, "display_name", self.path.name)).stem
        text = F.export_cards(items, self.out / "exports", name, self)
        missing = F.not_found(nums, [w for _, _, w, _ in items])
        return text + (f"\nNot printed {self.where} (so not in the file; nothing is guessed): {', '.join(missing)}" if missing else "")

    def answer(self, q):
        if F.wants_card(q, F.bridge_nums(q)) and re.search(r"\bbr(?:idge)?s?\b|\bbr\.|\b\d{1,4}a?(?:up|dn)\b", q.lower()):
            return self.bridge_csv(q)
        if re.search(r"\bcurves?\b", q, re.I) and F.bridge_nums(q):
            return self.curve_near_bridge(q)                # "which curve is near bridge 264": the curves of every page
        if self.is_list(q):
            return self.list_bridges(q)
        pages, why = self.pages_for(q)
        if not pages:
            return why[0].upper() + why[1:] + " (nothing is guessed)."
        if len(self.finders) == 1:
            return self.finders[0].answer(q)
        return "\n\n".join(f"[page {i + 1}]\n{self.finders[i].answer(q)}" for i in pages)

    def crops_since(self, marks):
        """The crops saved since marks (from crop_marks()) - the ones behind the last answer."""
        return [p for f, m in zip(self.finders, marks) for p in f.saved[m:]]

    def crop_marks(self):
        return [len(f.saved) for f in self.finders]
