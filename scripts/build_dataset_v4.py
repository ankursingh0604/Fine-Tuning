"""Build the v4 datasets (normal + reasoning) from data/v4/annotations/ (run scripts/annotate_v4.py first).

    .venv\\Scripts\\python scripts\\build_dataset_v4.py

Uses the v3 task generators (build_dataset.py, build_reasoning.py) unchanged in what they ask, with per-sheet facts
set for each sheet: which lines the sheet is about (proposed 3rd line / existing UP line, or proposed 4th line /
existing DN line), which note gives the rail level, and the ruling gradient if a note gives one. v4-only additions:
note questions by topic, the band key `exg_line_fl` (the existing line is UP or DN), TBM rows without a chainage,
and the BRIDGE DETAILS table.

Output: data/v4/dataset/{train,val,test}.jsonl, reasoning_{train,val,test}.jsonl, images/, stats.json.
Answers keep boxes in absolute pixels of the image as given (the v3 format); the training script converts boxes to
the chosen model's own format, so the dataset does not depend on the model.
"""
import io
import contextlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import random

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as D          # noqa: E402
import build_reasoning as R        # noqa: E402

ANN = ROOT / "data" / "v4" / "annotations"
OUT = ROOT / "data" / "v4" / "dataset"

TEST_PDF = "MKN_PnP_1203-1244_4th Line.pdf"          # a whole section the model never sees
TEST = {"MKN-3RD_100", "MKN-3RD_104", "MKN-3RD_110",   # v3's test sheets, for a like-for-like comparison
        "MKN-3RD_080", "MKN-4TH_057", "MKN-4TH_101"}
VAL = {"MKN-3RD_096", "MKN-3RD_066", "MKN-4TH_078", "MKN-4TH_097", "MKN-4TH_063"}
CAP_PER_SHEET = {"reason_gradient": 6}   # most a single sheet may contribute to a task


def cap(rows, seed=5):
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault((r["sheet"], r["task"]), []).append(r)
    keep = set()
    for (sheet, task), rs in by.items():
        n = CAP_PER_SHEET.get(task)
        keep.update(id(r) for r in (rng.sample(rs, n) if n and len(rs) > n else rs))
    return [r for r in rows if id(r) in keep]


def series(sid):
    return sid.split("_")[0]


def sheet_facts(ann):
    lines = (ann.get("bands") or {}).get("lines") if isinstance(ann.get("bands"), dict) else None
    lines = lines or {"proposed": "3RD LINE", "existing": "UP LINE"}
    prop = lines["proposed"].lower()                    # "4th line"
    exg = lines["existing"].replace("LINE", "line")     # "DN line"
    rail = next((n["no"] for n in ann["notes"] if "RAIL LEVEL" in n["text"]), 5)
    ruling = None
    for n in ann["notes"]:
        m = re.search(r"RULING GRADIENT[^0-9]*?1 IN (\d+)", n["text"])
        if m:
            ruling = (int(m.group(1)), n["no"])
            break
    return {"prop": prop, "exg": exg, "rail_note": rail, "ruling": ruling, "v4": True, "interpolate": True}


def rename_lines(ann, facts):
    """The annotator names lines as on the 3rd-line sheets; use this sheet's own lines."""
    old_new = {"proposed 3rd line": f"proposed {facts['prop']}", "existing UP line": f"existing {facts['exg']}"}
    for key in ("curves", "gradient_segments", "vpis", "plan_grade_points", "transition_points"):
        for item in ann.get(key) or []:
            if isinstance(item, dict) and item.get("line") in old_new:
                item["line"] = old_new[item["line"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D.OUT, D.IMG = OUT, OUT / "images"
    R.OUT = OUT
    D.IMG.mkdir(exist_ok=True)
    for old in D.IMG.glob("*.png"):
        old.unlink()
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(ANN.glob("*.json"))]
    split_of = {a["sheet_id"]: "test" if (a["source_pdf"] == TEST_PDF or a["sheet_id"] in TEST) else
                "val" if a["sheet_id"] in VAL else "train" for a in anns}
    # Bridges and band columns near a val/test sheet repeat on its neighbours: hold them out of train, per line
    # (the 3rd and 4th lines share chainages and bridge numbers but are different lines).
    held, held_ch = {}, {}
    for a in anns:
        if split_of[a["sheet_id"]] != "train":
            s = series(a["sheet_id"])
            held.setdefault(s, set()).update(b["bridge_id"] for b in a["bridges"])
            held_ch.setdefault(s, set()).update(c["chainage"] for c in (a.get("bands") or {}).get("columns", []))
    out = {k: [] for k in ("train", "val", "test")}
    rout = {k: [] for k in ("train", "val", "test")}
    pdfs = {}
    for a in anns:
        sid, split, s = a["sheet_id"], split_of[a["sheet_id"]], series(a["sheet_id"])
        facts = sheet_facts(a)
        rename_lines(a, facts)
        D.SHEET.clear()
        D.SHEET.update(facts)
        page = pdfs.setdefault(a["source_pdf"], pymupdf.open(ROOT / a["source_pdf"]))[a["page_index"]]
        bl = held.get(s, set()) if split == "train" else set()
        blc = held_ch.get(s, set()) if split == "train" else set()
        has_bands = isinstance(a.get("bands"), dict) and a["bands"].get("columns")
        rows, rrows, problems = [], [], []
        jobs = [("tiles", lambda: D.tiles(page, a, sid, bl)),
                ("bridges", lambda: D.bridge_crops(page, a, sid, 2 if split == "train" else 1, bl)),
                ("curves", lambda: D.curve_crops(page, a, sid)),
                ("panel", lambda: D.panel_crops(page, a, sid)),
                ("sheet", lambda: D.sheet_view(page, a, sid))]
        if has_bands:
            jobs.append(("bands", lambda: D.band_crops(page, a, sid, blc)))
        rjobs = [("r_level_check", lambda: R.r_level_check(page, a, sid, bl)),
                 ("r_curve_verify", lambda: R.r_curve_verify(page, a, sid)),
                 ("r_gradient", lambda: R.r_gradient(page, a, sid)),
                 ("r_free_board", lambda: R.r_free_board(page, a, sid, bl))]
        if has_bands:
            rjobs = [("r_bridge_bands", lambda: R.r_bridge_bands(page, a, sid, bl, blc))] + rjobs[:1] + \
                    [("r_bands", lambda: R.r_bands(page, a, sid, blc)), ("r_curve_position", lambda: R.r_curve_position(page, a, sid))] +                     rjobs[1:] + [("r_ruling", lambda: R.r_ruling(page, a, sid))]
        for name, job in jobs:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rows += job()
            except Exception as e:                    # noqa: BLE001 - report and carry on with the other tasks
                problems.append(f"{name}: {type(e).__name__}: {e}")
        for name, job in rjobs:
            try:
                rrows += job()
            except Exception as e:                    # noqa: BLE001
                problems.append(f"{name}: {type(e).__name__}: {e}")
        for i, r in enumerate(rows):
            r["id"], r["split"] = f"{sid}_{i:04d}", split
        for i, r in enumerate(rrows):
            r["id"], r["split"] = f"r_{sid}_{i:04d}", split
        out[split] += rows
        rout[split] += cap(rrows)
        print(f"{sid:<13} ({split:<5}) {len(rows):>4} normal  {len(rrows):>3} reasoning" + (f"   PROBLEMS: {problems}" if problems else ""))

    stats = {"normal": {}, "reasoning": {}}
    for split in out:
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in out[split]:
                f.write(json.dumps({k: r[k] for k in ("id", "sheet", "split", "task", "image", "width", "height", "messages")},
                                   ensure_ascii=False) + "\n")
        with open(OUT / f"reasoning_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rout[split]:
                f.write(json.dumps({k: r[k] for k in ("id", "sheet", "split", "task", "image", "images", "sizes", "messages")},
                                   ensure_ascii=False) + "\n")
        stats["normal"][split] = {"rows": len(out[split]), "sheets": len({r["sheet"] for r in out[split]}),
                                  "tasks": dict(sorted(Counter(r["task"] for r in out[split]).items()))}
        stats["reasoning"][split] = {"rows": len(rout[split]), "tasks": dict(sorted(Counter(r["task"] for r in rout[split]).items()))}
    stats["splits"] = {k: sorted(s for s, v in split_of.items() if v == k) for k in ("val", "test")}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in D.IMG.glob("*.png")) / 1e6
    print(json.dumps({k: {s: v["rows"] for s, v in stats[k].items()} for k in ("normal", "reasoning")}, indent=1))
    print(f"{len(list(D.IMG.glob('*.png')))} images, {size:.0f} MB")


if __name__ == "__main__":
    main()
