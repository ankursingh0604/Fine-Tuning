"""A stand-in for the fine-tuned model, for testing runpod/sheet_reader.py without a GPU.

It answers each question the way the training data taught the model to - but from the true annotations of
the area the reader actually cropped. So any geometry mistake in the reader (wrong dpi, offset, a crop that
misses the level block, a band window that misses the bridge's column) shows up as missing or wrong answers.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as D      # noqa: E402

Z = 150 / 72


def visible(box, clip):
    w = max(0, min(box[2], clip[2]) - max(box[0], clip[0]))
    h = max(0, min(box[3], clip[3]) - max(box[1], clip[1]))
    return w * h / max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)


class SimModel:
    wants_meta = True

    def __init__(self, ann):
        self.a = ann
        self.calls = 0

    def ask(self, imgs, question, meta=None):
        self.calls += 1
        a, k = self.a, meta["kind"]
        if k == "find":
            c = meta["clip"]
            out = []
            for b in a["bridges"]:
                for view, lab in (("plan_callout", "plan"), ("lsection_callout", "L-section")):
                    if view in b and visible(b[view]["bbox"], c) >= 0.9:
                        bx = b[view]["bbox"]
                        px = [round(max(0, (bx[0] - c[0]) * Z)), round(max(0, (bx[1] - c[1]) * Z)),
                              round(min(1008, (bx[2] - c[0]) * Z)), round(min(1008, (bx[3] - c[1]) * Z))]
                        out.append({"bbox_2d": px, "label": f"bridge {b['bridge_id']} ({lab} callout)"})
            return D.fenced(out) if out else "There are no bridge or structure callouts in this crop."
        if k == "bridge":
            b = next((b for b in a["bridges"] if b["bridge_id"] == meta["bid"]), None)
            if not b or not b["complete"]:
                return "I cannot read a complete callout here."
            lv = b.get("lsection_levels")
            return D.fenced(D.bridge_fields(b, with_levels=bool(lv) and visible(lv["bbox"], meta["clip"]) >= 0.98))
        cols = [c for c in a["bands"]["columns"] if c["checks_ok"]]
        if k in ("range", "band"):
            x0, x1 = meta["xrange"]
            win = [c for c in cols if x0 <= c["x"] <= x1]
            if not win:
                return "No data-band columns are visible."
            if k == "range":
                return f"This crop covers chainage {D.ch_text(win[0]['chainage'])} to {D.ch_text(win[-1]['chainage'])}: {len(win)} columns, one every 20 m."
            near = min(win, key=lambda c: abs(c["chainage"] - meta["ch"]))
            return D.fenced({"bridge_chainage": meta["ch"], "nearest_column": D.band_record(near),
                             "distance_m": round(abs(near["chainage"] - meta["ch"]), 3)})
        if k in ("title", "tbm") and meta.get("clip"):            # it can only read what the crop shows
            region = (a.get("regions") or {}).get("title_block" if k == "title" else "tbm_table")
            if region and visible(region, meta["clip"]) < 0.9:
                return "I cannot see a title block in this crop." if k == "title" else "There is no TBM table in this crop."
        if k == "title":
            info = a["sheet_info"]
            t = {k2: info.get(k2) for k2 in ("drawing_no", "sheet_no", "title", "chainage_from", "chainage_to", "scale", "date", "client")}
            return D.fenced(t)
        if k == "tbm":
            return D.fenced([{k2: t[k2] for k2 in ("tbm_id", "chainage_m", "easting", "northing", "msl_m", "description")}
                             for t in a["tbm_benchmarks"]])
        return ""
