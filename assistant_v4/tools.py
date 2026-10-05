"""The v4 tools (docs/tools_v4.json), implemented over the store. The agent conversations in the dataset were generated
with the same output shapes; tests/replay_agent_v4.py replays them through these tools to prove they match.

Hooks (optional; without them the tools still work for vector PDFs and the store):
    image_reader(path) -> list of annotation dicts   reading images / scanned pages with the vision model
    look_model(sheet_id, area, question) -> str       answering 'look' when the store has no text for the area
    web_provider(query) -> [{"title", "snippet", "url"}]   internet search (off unless given)
"""
import ast
import json
import operator
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from store import Store, km, line_of, sha_of     # noqa: E402

INTERP = ["cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl", "ground_level"]


def f3(v):
    """Numbers in messages, written like the training data: 0.041, 5000 (not 5000.0)."""
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


# ---------------------------------------------------------------- safe arithmetic for calc

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only numbers, + - * / and parentheses are allowed")
    return ev(ast.parse(expr, mode="eval"))


# ---------------------------------------------------------------- leak gate for web_search

ALLOWED_WORDS = "railway longitudinal section abbreviation meaning"


def leak_gate(term, denylist):
    t = (term or "").strip()
    if not t:
        return "empty term"
    if len(t) > 30 or len(t.split()) > 4:
        return "a bare term only (at most 4 words / 30 characters)"
    if re.search(r"\d{3,}|\d+\.\d+|\d+\+\d+", t):
        return "it contains a number that could identify the project"
    up = t.upper()
    for d in denylist:
        if up == d or (len(d) >= 4 and d in up) or (len(up) >= 4 and up in d and len(up) / len(d) > 0.6):
            return "it matches something identifying on the sheets"
    return None


class Tools:
    def __init__(self, store=None, image_reader=None, look_model=None, web_provider=None, library_dir=None, log_dir=None):
        self.store = store or Store()
        self.image_reader, self.look_model, self.web_provider = image_reader, look_model, web_provider
        self.library_dir = Path(library_dir or ROOT / "docs" / "library")
        self.log_dir = Path(log_dir or ROOT / "data" / "v4")
        self._index = None
        self._lib = None

    # ------------------------------------------------------------ the sheets, as the tools see them
    def index(self):
        """Bridges, band columns, sheets, curves, TBMs per line from the latest version of every sheet."""
        latest = self.store.latest()
        if self._index and self._index[0] == id(latest):
            return self._index[1]
        idx = {"bridges": defaultdict(dict), "cols": defaultdict(dict), "sheets": defaultdict(list), "curves": defaultdict(dict),
               "tbms": {}, "ann": {}, "line": {}}
        for sid, (a, ver) in latest.items():
            line = line_of(a)
            idx["ann"][sid] = a
            if not line:
                continue
            idx["line"][sid] = line
            info = a["sheet_info"]
            lo, hi = km(info.get("chainage_from")), km(info.get("chainage_to"))
            if lo is not None and hi is not None:
                idx["sheets"][line].append((lo, hi, sid))
            for b in a["bridges"]:
                if b.get("belongs_to") == "this sheet" and b.get("complete") and b.get("chainage_m"):
                    rec = {k: b.get(k) for k in ("bridge_id", "existing_on", "existing_type", "existing_span", "crossing",
                                                 "proposal", "category", "chainage_m")}
                    rec["levels"] = {k: v for k, v in (b.get("lsection_levels") or {}).items() if k != "bbox"} or None
                    rec["sheet_id"] = sid
                    idx["bridges"][line][b["bridge_id"]] = rec
            bd = a.get("bands") if isinstance(a.get("bands"), dict) else {}
            for c in bd.get("columns", []):
                if c["checks_ok"]:
                    idx["cols"][line].setdefault(c["chainage"], (c, sid))
            for c in a["curves"]:
                if c["location"] == "lsection_band" or c["curve_no"] not in idx["curves"][line]:
                    idx["curves"][line][c["curve_no"]] = {"curve_no": c["curve_no"], "hand": c["hand"], **c["params"], "sheet_id": sid}
            for t in a["tbm_benchmarks"]:
                idx["tbms"][t["tbm_id"]] = {**{k: t[k] for k in ("tbm_id", "chainage_m", "easting", "northing", "msl_m", "description")},
                                            "sheet_id": sid}
        idx["xs"] = {line: sorted(cols) for line, cols in idx["cols"].items()}
        self._index = (id(latest), idx)
        return idx

    # ------------------------------------------------------------ query
    def query(self, kind, line=None, bridge_id=None, chainage=None, chainage_from=None, chainage_to=None, curve_no=None,
              sheet_id=None, term=None, tbm_id=None, sheet_no=None):
        idx = self.index()
        if kind == "bridge":
            b = idx["bridges"].get(line, {}).get(str(bridge_id))
            if not b:
                return {"error": f"bridge {bridge_id} not found on the {line}"}
            older = self.store.older_values(b["sheet_id"])
            if older:
                out = json.loads(json.dumps(b))
                out["version"] = f"{self.store.latest()[b['sheet_id']][1]['revision']} (latest)"
                prev = []
                for rev, file, ann in older:
                    ob = next((x for x in ann["bridges"] if x["bridge_id"] == b["bridge_id"]), None)
                    fl = ((ob or {}).get("lsection_levels") or {}).get("proposed_formation_level")
                    prev.append({"revision": rev, "file": file, "proposed_formation_level": fl})
                out["older_versions"] = prev
                return out
            return b
        if kind in ("bridges_in_range", "bridges_on_sheet"):
            bs = [b for b in idx["bridges"].get(line, {}).values()] if kind == "bridges_in_range" else \
                 [b for lb in idx["bridges"].values() for b in lb.values() if b["sheet_id"] == sheet_id]
            if kind == "bridges_in_range":
                bs = [b for b in bs if chainage_from <= b["chainage_m"] <= chainage_to]
                return [{"bridge_id": b["bridge_id"], "chainage_m": b["chainage_m"], "sheet_id": b["sheet_id"], "levels": b["levels"]}
                        for b in sorted(bs, key=lambda b: b["chainage_m"])]
            return sorted(bs, key=lambda b: b["chainage_m"])
        if kind == "sheet_for_chainage":
            return [sid for lo, hi, sid in idx["sheets"].get(line, []) if lo <= chainage <= hi]
        if kind == "curve":
            c = idx["curves"].get(line, {}).get(str(curve_no))
            return c or {"error": f"curve {curve_no} not found on the {line}"}
        if kind == "tbm":
            return idx["tbms"].get(tbm_id) or {"error": f"{tbm_id} not found"}
        if kind == "notes":
            a = idx["ann"].get(sheet_id)
            return [{"no": n["no"], "text": n["text"]} for n in a["notes"]] if a else {"error": f"sheet {sheet_id} not found"}
        if kind == "abbreviation":
            sheets = [sheet_id] if sheet_id else sorted(idx["ann"])
            for sid in sheets:
                m = (idx["ann"].get(sid) or {}).get("abbreviations", {}).get(term)
                if m:
                    return {"term": term, "meaning": m, "source": f"abbreviations table, sheet {sid}"}
            return {"term": term, "meaning": None, "source": f"not in the abbreviations table of sheet {sheet_id}" if sheet_id
                    else "not in the abbreviations table of any sheet read"}
        if kind == "versions":
            return self.store.versions(line, sheet_no)
        return {"error": f"unknown kind {kind}"}

    # ------------------------------------------------------------ band_at (interpolation)
    def band_at(self, line, chainages):
        idx = self.index()
        xs, cols = idx["xs"].get(line, []), idx["cols"].get(line, {})
        out = []
        for ch in chainages:
            res = None
            for x1, x2 in zip(xs, xs[1:] + [None]):
                if x1 == ch or (x2 is not None and x1 < ch < x2):
                    if x1 == ch:
                        x2 = x1
                    if x2 != x1 and x2 - x1 != 20:
                        res = {"chainage": ch, "error": "a printed column between them could not be read; not interpolated"}
                        break
                    c1, s1 = cols[x1]
                    c2, _ = cols[x2]
                    t = 0.0 if x2 == x1 else (ch - x1) / (x2 - x1)
                    y = {k: round(c1["values"][k] + (c2["values"][k] - c1["values"][k]) * t, 3) for k in INTERP
                         if k in c1["values"] and k in c2["values"]}
                    strip = lambda c: {k: v for k, v in c["values"].items() if k != "chainage"}      # noqa: E731
                    res = {"chainage": ch, "sheet_id": s1, "x1": int(x1), "x2": int(x2), "column_x1": strip(c1),
                           "column_x2": strip(c2), "y": y, "rail_level_ok": [c1.get("rail_level_ok", True), c2.get("rail_level_ok", True)]}
                    break
            out.append(res or {"chainage": ch, "error": "not on the data bands of any sheet read for this line"})
        return out

    # ------------------------------------------------------------ calc
    def calc(self, expression):
        try:
            return {"expression": expression, "value": round(safe_eval(expression), 3)}
        except Exception as e:                              # noqa: BLE001
            return {"expression": expression, "error": str(e)}

    # ------------------------------------------------------------ look
    def look(self, sheet_id, area, question):
        a = self.index()["ann"].get(sheet_id)
        if a is None:
            return {"error": f"sheet {sheet_id} not found"}
        ar = (area or "").lower()
        if "legend" in ar:
            return {"text": a["legend"]}
        if "issue" in ar:
            return {"text": issue_lines(a)}
        if "note" in ar:
            return {"text": [f"{n['no']}. {n['text']}" for n in a["notes"]]}
        if "abbreviation" in ar:
            return {"text": [f"{k} = {v}" for k, v in a["abbreviations"].items()]}
        if "bridge details" in ar or "bridge table" in ar:
            return {"text": [" | ".join(str(r.get(k) or "-") for k in ("bridge_id", "chainage_m", "existing_type", "existing_span", "crossing",
                                                                       "proposed_structure", "category", "proposed_span"))
                             for r in a.get("bridge_table") or []]}
        region = next((r for name, r in a.get("regions", {}).items() if name.replace("_", " ") in ar), None)
        if region:
            return {"text": [t["text"] for t in sorted(a["all_text"], key=lambda t: (round(t["bbox"][1]), t["bbox"][0]))
                             if region[0] <= t["bbox"][0] <= region[2] and region[1] <= t["bbox"][1] <= region[3]]}
        if self.look_model:
            return {"text": [self.look_model(sheet_id, area, question)]}
        return {"error": "no stored text for that area; reading it needs the vision model"}

    # ------------------------------------------------------------ search_library / web_search
    def search_library(self, query):
        if self._lib is None:
            self._lib = []
            for f in sorted(self.library_dir.glob("**/*")) if self.library_dir.exists() else []:
                if f.suffix.lower() in (".txt", ".md"):
                    pages = [f.read_text(encoding="utf-8", errors="ignore")]
                elif f.suffix.lower() == ".pdf":
                    import pymupdf
                    pages = [p.get_text() for p in pymupdf.open(f)]
                else:
                    continue
                for i, text in enumerate(pages):
                    for j in range(0, len(text), 800):
                        self._lib.append({"document": f.name, "page": i + 1, "text": text[j:j + 900]})
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1]
        scored = sorted(((sum(p["text"].lower().count(w) for w in words), p) for p in self._lib), key=lambda t: -t[0])
        return {"results": [p for s, p in scored[:3] if s > 0]}

    def denylist(self):
        out = set()
        for a in self.index()["ann"].values():
            info = a["sheet_info"]
            for k in ("drawing_no", "client", "project", "title"):
                if info.get(k):
                    out.add(info[k].upper())
            for s in info.get("stations") or []:
                out.add(str(s.get("name", "")).upper())
            for r in info.get("issue_record") or []:
                for k in ("prepared_by", "checked_by", "approved_by"):
                    if r.get(k):
                        out.add(r[k].upper())
            out.add(a.get("sheet_id", "").upper())
        parts = set()
        for d in out:                         # also each part ("WEST CENTRAL RAILWAY", "KOTA DIVISION", "MATHURA", ...)
            for p in re.split(r"[,;/()\-]+", d):
                p = p.strip(" .:")
                if len(p) >= 4 and not re.fullmatch(r"(LINE|SHEET|PLAN|SECTION|DETAILED|AND|THE|FOR|OF)", p):
                    parts.add(p)
        return {d for d in out | parts if d}

    def web_search(self, term):
        why = leak_gate(term, self.denylist())
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "term": term}
        if why:
            res = {"term": term, "error": f"not searched: {why}"}
        elif not self.web_provider:
            res = {"term": term, "error": "internet search is switched off"}
        else:
            query = f'"{term.strip()}" {ALLOWED_WORDS}'
            rec["query"] = query
            res = {"term": term, "results": [{"snippet": r.get("snippet"), "title": r.get("title"), "url": r.get("url")}
                                             for r in self.web_provider(query)[:3]]}
        rec["result"] = "blocked" if why else ("off" if "error" in res else "ok")
        with open(self.log_dir / "web_search_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return res

    # ------------------------------------------------------------ read_sheet
    def read_sheet(self, file):
        import annotate as A
        import annotate_v4 as AV
        path = Path(file)
        if not path.exists():
            path = ROOT / file
        if not path.exists():
            return {"file": str(file), "error": "file not found"}
        sha = sha_of(path)
        if self.store.known_file(sha):
            pages = [page_summary(a, ver["line"], "already stored (same file)") for sid, (a, ver) in self.store.latest().items()
                     if ver["file"] == path.name or ver["file"] == str(file)]
            for i, p in enumerate(sorted(pages, key=lambda p: p["page"])):
                p["page"] = i + 1
            return {"file": path.name, "pages": pages, "continuity": continuity(pages)}
        pages = []
        if path.suffix.lower() == ".pdf":
            import pymupdf
            doc = pymupdf.open(path)
            for i, page in enumerate(doc):
                if len(page.get_text("words")) < 200:
                    anns = self.image_reader(page) if self.image_reader else None
                    if not anns:
                        pages.append({"page": i + 1, "type": "scanned page", "sheet_id": None,
                                      "note": "no text layer; reading it needs the vision model, which is not loaded."})
                        continue
                else:
                    anns = [A.annotate_page(page, path.name, i)]
                for ann in anns:
                    pages.append(self._store_page(ann, path.name, sha, i))
        else:
            anns = self.image_reader(path) if self.image_reader else None
            if not anns:
                return {"file": path.name, "error": "reading images needs the vision model, which is not loaded"}
            for ann in anns:
                pages.append(self._store_page(ann, path.name, sha, 0))
        self.store.add_file(sha, path.name)
        self.store.db.commit()
        return {"file": path.name, "pages": pages, "continuity": continuity(pages)}

    def _store_page(self, ann, name, sha, i):
        import annotate_v4 as AV
        title = (ann["sheet_info"].get("title") or "").upper()
        bd = ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
        if not (bd.get("columns") or "L SECTION" in title or "L-SECTION" in title):
            text = " ".join(t["text"] for t in ann.get("all_text", [])).upper()
            cover = any(w in text for w in ("INDEX", "CONTENTS", "LIST OF DRAWINGS")) or len(ann.get("all_text", [])) < 150
            return {"page": i + 1, "type": "cover / index page" if cover else "drawing of another kind", "sheet_id": None,
                    "note": "not a Plan & L-Section sheet, so no L-section values were taken from it."}
        ann["sheet_id"] = AV.sheet_id(ann, name, i)
        ann["findings"] = AV.findings(ann)
        status = self.store.add_version(ann, name, f"{sha}:{i}", i + 1)
        return page_summary(ann, line_of(ann), status, page=i + 1)


def issue_lines(a):
    return [f"{r.get('rev')} | {r.get('date')} | prepared by {r.get('prepared_by')} | checked by {r.get('checked_by')} | "
            f"approved by {r.get('approved_by')}" for r in (a["sheet_info"].get("issue_record") or [])]


def page_summary(a, line, status="new", page=None):
    info = a["sheet_info"]
    bd = a.get("bands") if isinstance(a.get("bands"), dict) else {}
    rev = ((info.get("issue_record") or [{}])[-1] or {}).get("rev") or "R0"
    return {"page": page or a["page_index"] + 1, "type": "L-section sheet", "sheet_id": a["sheet_id"], "sheet_no": info.get("sheet_no"),
            "line": line, "chainage_from": info.get("chainage_from"), "chainage_to": info.get("chainage_to"),
            "revision": rev, "read_by": "PDF text layer (exact)", "layout": "known",
            "read": {"bridges": len(a["bridges"]), "level_blocks": sum(1 for b in a["bridges"] if b.get("lsection_levels")),
                     "curves": len(a["curves"]), "band_columns": len(bd.get("columns", [])), "tbms": len(a["tbm_benchmarks"])},
            "findings": a.get("findings") or [], "status": status}


def continuity(pages):
    ls = [p for p in pages if p["type"] == "L-section sheet" and p.get("chainage_from") and p.get("chainage_to")]
    ls.sort(key=lambda p: km(p["chainage_from"]))
    issues = []
    for a, b in zip(ls, ls[1:]):
        gap = km(b["chainage_from"]) - km(a["chainage_to"])
        if gap > 1:
            issues.append(f"gap of {f3(round(gap, 1))} m between sheet {a['sheet_id']} (ends {a['chainage_to']}) and {b['sheet_id']} (starts {b['chainage_from']})")
        elif gap < -1:
            issues.append(f"overlap of {f3(round(-gap, 1))} m between sheet {a['sheet_id']} and {b['sheet_id']}")
    return {"checked_pairs": max(0, len(ls) - 1), "issues": issues}


def call(tools, name, args):
    """Run one tool call by name (as the agent loop does)."""
    fn = getattr(tools, name, None)
    if fn is None or name.startswith("_"):
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
