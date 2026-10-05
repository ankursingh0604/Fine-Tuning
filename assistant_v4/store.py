"""The v4 store: everything read from the sheets, with versions (Phase 1 of V4_PLAN.md).

SQLite file (default data/v4/store.sqlite). One row per sheet version; the data of each version is kept as the
annotation JSON (the same structure scripts/annotate.py writes), so the tools read exactly what was annotated.

Versions (decided): a sheet number already in the store is kept as a separate version, never replaced. The same
file content again (same SHA-256) reuses the stored reading. Answers use the latest revision of a sheet (issue-record
revision; if equal or missing, the latest upload).
"""
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (sha TEXT PRIMARY KEY, name TEXT, stored TEXT);
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sheet_id TEXT, line TEXT, sheet_no TEXT, revision TEXT,
    file TEXT, sha TEXT, page INTEGER, stored TEXT, chainage_from REAL, chainage_to REAL, ann TEXT);
CREATE INDEX IF NOT EXISTS v_sheet ON versions(sheet_id);
CREATE INDEX IF NOT EXISTS v_line ON versions(line);
"""


def km(km_text):
    m = re.match(r"(\d+)\+(\d+(?:\.\d+)?)", km_text or "")
    return int(m.group(1)) * 1000 + float(m.group(2)) if m else None


def line_of(ann):
    bd = ann.get("bands") if isinstance(ann.get("bands"), dict) else {}
    p = (bd.get("lines") or {}).get("proposed")
    if not p:
        t = ann["sheet_info"].get("title") or ""
        p = "4TH LINE" if "4TH LINE" in t else "3RD LINE" if t else None
    return p.lower() if p else None


def revision_of(ann):
    rec = ann["sheet_info"].get("issue_record") or []
    return (rec[-1] or {}).get("rev") if rec else "R0"


def rev_key(rev):
    m = re.match(r"R(\d+)", rev or "")
    return int(m.group(1)) if m else -1


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Store:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "data" / "v4" / "store.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self._cache = None

    # ------------------------------------------------------------ writing
    def known_file(self, sha):
        return self.db.execute("SELECT name FROM files WHERE sha = ?", (sha,)).fetchone()

    def add_file(self, sha, name):
        self.db.execute("INSERT OR IGNORE INTO files VALUES (?, ?, ?)", (sha, name, time.strftime("%Y-%m-%d %H:%M")))

    def add_version(self, ann, file, sha, page):
        """Store one sheet; returns 'new' or 'stored as a new version (Rn); earlier versions kept'."""
        sid, line = ann.get("sheet_id"), line_of(ann)
        rev = revision_of(ann)
        prior = self.db.execute("SELECT revision FROM versions WHERE sheet_id = ?", (sid,)).fetchall()
        info = ann["sheet_info"]
        self.db.execute("INSERT INTO versions (sheet_id, line, sheet_no, revision, file, sha, page, stored, chainage_from, chainage_to, ann) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (sid, line, info.get("sheet_no"), rev, file, sha, page, time.strftime("%Y-%m-%d %H:%M"),
                         km(info.get("chainage_from")), km(info.get("chainage_to")), json.dumps(ann, ensure_ascii=False)))
        self.db.commit()
        self._cache = None
        if prior:
            return f"stored as a new version ({rev}); {', '.join(r for (r,) in prior)} kept"
        return "new"

    def ingest_annotations(self, folder=None):
        """Load data/v4/annotations (the sheets already read) as version R0-of-their-file."""
        folder = Path(folder or ROOT / "data" / "v4" / "annotations")
        n = 0
        for f in sorted(folder.glob("*.json")):
            ann = json.loads(f.read_text(encoding="utf-8"))
            pdf = ROOT / ann["source_pdf"]
            sha = f"{sha_of(pdf)}:{ann['page_index']}" if pdf.exists() else f"ann:{f.stem}"
            if self.db.execute("SELECT 1 FROM versions WHERE sha = ?", (sha,)).fetchone():
                continue
            self.add_version(ann, ann["source_pdf"], sha, ann["page_index"] + 1)
            n += 1
        for pdf in {json.loads(f.read_text(encoding="utf-8"))["source_pdf"] for f in folder.glob("*.json")}:
            if (ROOT / pdf).exists():
                self.add_file(sha_of(ROOT / pdf), pdf)
        self.db.commit()
        return n

    # ------------------------------------------------------------ reading (latest versions)
    def latest(self):
        """sheet_id -> (annotation, version row) of the latest version of every sheet."""
        if self._cache is None:
            best = {}
            for row in self.db.execute("SELECT id, sheet_id, line, sheet_no, revision, file, stored, ann FROM versions ORDER BY id"):
                vid, sid, line, no, rev, file, stored, ann = row
                cur = best.get(sid)
                if cur is None or (rev_key(rev), vid) >= (rev_key(cur[1]["revision"]), cur[1]["id"]):
                    best[sid] = (json.loads(ann), {"id": vid, "line": line, "sheet_no": no, "revision": rev, "file": file, "stored": stored})
            self._cache = best
        return self._cache

    def versions(self, line, sheet_no):
        rows = self.db.execute("SELECT revision, file, stored, id FROM versions WHERE line = ? AND sheet_no = ? ORDER BY id",
                               (line, str(sheet_no))).fetchall()
        if not rows:
            return []
        top = max(rows, key=lambda r: (rev_key(r[0]), r[3]))
        return [{"revision": r[0], "file": r[1], "stored": r[2], **({"latest": True} if r is top else {})} for r in rows]

    def older_values(self, sheet_id):
        """Annotations of the earlier versions of a sheet, oldest first."""
        rows = self.db.execute("SELECT revision, file, ann, id FROM versions WHERE sheet_id = ? ORDER BY id", (sheet_id,)).fetchall()
        latest_id = self.latest().get(sheet_id, (None, {"id": None}))[1]["id"]
        return [(r[0], r[1], json.loads(r[2])) for r in rows if r[3] != latest_id]
