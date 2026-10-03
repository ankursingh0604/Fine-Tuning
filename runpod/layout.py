"""Find a sheet's layout from the image itself, and judge whether it is the layout the model was trained on.

Step 1 - structure, not fixed positions:
  - long horizontal / vertical rules (table borders), found by the length of their dark runs
  - the data-band table: a stack of horizontal rules whose left label strip is read with a local OCR
    (rapidocr, CPU, offline); each row is identified from its printed label, so a different row order or
    a moved table is still understood
  - the column pitch and the first column, measured from the chainage row
  - the right-hand panel, and inside it the TBM table and the title block, located by their headings
Step 3 - warn instead of guessing:
  - "known": this drawing set's layout -> the reader uses exactly the crops the model was trained on
  - "similar": the same rows, but in other places / order -> the reader works from the detected geometry
  - "unknown": rows missing or unrecognised, or no band table -> parts are skipped and the reasons reported
The reader adds the read-time checks (band arithmetic, title block, bridges found) to this verdict.
"""
import difflib
import re

import numpy as np

DARK = 215
ROW_FIELDS = ["cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl", "ground_level", "chainage"]
KNOWN_ROW_H = [42.7, 49.4, 45.4, 45.3, 49.5, 45.6, 45.1, 45.6]   # pt, this drawing set
KNOWN_PANEL_X = 3175.4
KNOWN_PITCH = 11.34
KNOWN_BAND_TOP = (1845, 1875)    # pt: top rule of the cut/fill row on the trained sheets
# The labels as printed on the trained sheets (upper case, letters and digits only). A row is matched to these
# by similarity first, so OCR slips ("DIEFERENCE") do not matter. "schematic" rows are part of the trained
# layout but carry no numbers, so they are known and skipped.
TRAINED_LABELS = {
    "CUTMFILLMFLGL": "cut_fill",
    "DIFFERENCEBETWEENPROP3RDLINEFLANDEXGUPLINEFL": "fl_difference",
    "PROP3RDLINERL": "prop_rl",
    "PROP3RDLINEFL": "prop_fl",
    "TRACKDISTANCEBETWEENPROP3RDLINEEXGUPLINE": "track_distance",
    "EXGUPLINEFL": "exg_up_fl",
    "GROUNDLEVELMBELOWPROP3RDLINE": "ground_level",
    "PROP3RDLINECHAINAGE": "chainage",
    "PROP3RDLINEVERTICALSCHEMATIC": "schematic",
    "EXGUPLINEVERTICALSCHEMATIC": "schematic",
}
# Labels not seen in training: keywords, most specific first (several labels name other rows inside them,
# e.g. "DIFFERENCE BETWEEN PROP. 3RD LINE FL AND EXG. UP LINE FL").
LABELS = [
    (r"SCHEMATIC", "schematic"),
    (r"D[I1L]\w?FERENCE|DIFF", "fl_difference"),
    (r"TRACK\w*DIST|DISTANCE\w*TRACK|TRACKCENT", "track_distance"),
    (r"CUT|FILL|FLGL", "cut_fill"),
    (r"GROUND|NGL", "ground_level"),
    (r"CHAINAGE", "chainage"),
    (r"(EXG|EXIST)\w*FL", "exg_up_fl"),
    (r"RL$|RAILLEVEL", "prop_rl"),
    (r"FL$|FORMATION", "prop_fl"),
]
HEADINGS = r"NOTE|TBM|BENCH|LEGEND|ABBREV|REFERENCE|ISSUE|CLIENT"

_ocr = None


def ocr():
    """The local OCR engine, or None if rapidocr is not installed (the label check is then skipped)."""
    global _ocr
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr = RapidOCR()
        except Exception:                      # noqa: BLE001 - any import/runtime problem means "no OCR"
            _ocr = False
    return _ocr or None


def ocr_boxes(img):
    """[(text, y_centre_px, x_centre_px)] for an RGB image."""
    eng = ocr()
    if eng is None:
        return None
    res, _ = eng(np.asarray(img))
    return [(t, float(np.mean([p[1] for p in b])), float(np.mean([p[0] for p in b]))) for b, t, _ in (res or [])]


def field_of(label):
    norm = re.sub(r"[^A-Z0-9]", "", label.upper())
    if not norm:
        return None
    best = max(TRAINED_LABELS, key=lambda k: difflib.SequenceMatcher(None, norm, k).ratio())
    if difflib.SequenceMatcher(None, norm, best).ratio() >= 0.8:
        return TRAINED_LABELS[best]
    return next((f for pat, f in LABELS if re.search(pat, norm)), None)


# ---------------------------------------------------------------- rules

def _cluster(idx):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= 2:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _longest_runs(mask):
    """Per row of a 2-D bool array: (length, start, end) of its longest run of True."""
    m = np.pad(mask, ((0, 0), (1, 1))).astype(np.int8)
    d = np.diff(m, axis=1)
    out = []
    for r in range(m.shape[0]):
        s, e = np.flatnonzero(d[r] == 1), np.flatnonzero(d[r] == -1)
        if not len(s):
            out.append((0, 0, 0))
            continue
        i = int(np.argmax(e - s))
        out.append((int(e[i] - s[i]), int(s[i]), int(e[i])))
    return out


def rules(sh, rect, min_len_pt, vertical=False):
    """Horizontal (or vertical) rules inside rect (points) at least min_len_pt long: [(pos, start, end)] in points."""
    x0, y0 = sh.px(rect[0], rect[1])
    x1, y1 = sh.px(rect[2], rect[3])
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(sh.gray.shape[1], int(x1)), min(sh.gray.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return []
    mask = sh.gray[y0:y1, x0:x1] < DARK
    if vertical:
        mask = mask.T
    need = min_len_pt * sh.z
    cand = np.flatnonzero(mask.mean(axis=1) * mask.shape[1] >= need * 0.95)
    runs = dict(zip(cand, _longest_runs(mask[cand]))) if len(cand) else {}
    good = [i for i in cand if runs[i][0] >= need]
    out = []
    for grp in _cluster(good):
        r = max((runs[i] for i in grp), key=lambda t: t[0])
        pos = float(np.mean(grp))
        if vertical:
            out.append(((x0 + pos - sh.dx) / sh.z, (y0 + r[1] - sh.dy) / sh.z, (y0 + r[2] - sh.dy) / sh.z))
        else:
            out.append(((y0 + pos - sh.dy) / sh.z, (x0 + r[1] - sh.dx) / sh.z, (x0 + r[2] - sh.dx) / sh.z))
    return out


# ---------------------------------------------------------------- band table

def _stacks(hr):
    """Group horizontal rules into table stacks: similar extents, regular spacing."""
    stacks, cur = [], []
    for r in sorted(hr):
        if cur:
            gap = r[0] - cur[-1][0]
            same = abs(r[1] - cur[-1][1]) < 20 and abs(r[2] - cur[-1][2]) < 20
            prev = cur[-1][0] - cur[-2][0] if len(cur) > 1 else None
            regular = prev is None or 1 / 1.35 < gap / prev < 1.35 or len(cur) == 2
            if same and 20 <= gap <= 90 and regular:
                cur.append(r)
                continue
            stacks.append(cur)
        cur = [r]
    if cur:
        stacks.append(cur)
    # two rules close together at a stack's top (a heavier border) leave a 2-rule stub: merge rows sensibly
    return [s for s in stacks if len(s) >= 3]


def _pitch(prof, z):
    """Column pitch (points) and phase (px) of a periodic profile."""
    p = prof - prof.mean()
    if not np.any(p):
        return None, None
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    lo, hi = int(5 * z), min(int(80 * z), len(ac) - 2)
    if hi <= lo + 2:
        return None, None
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0]
    if not peaks:
        return None, None
    top = max(seg[i] for i in peaks)
    lag = lo + next(i for i in peaks if seg[i] >= 0.6 * top)
    k = max(1, len(p) // (2 * lag))
    w = max(2, lag // 4)
    a, b = max(0, k * lag - w), min(len(ac), k * lag + w + 1)
    pitch_px = (a + int(np.argmax(ac[a:b]))) / k
    # Every column centre, then a straight-line fit: an error of 1 % in the pitch would put the 250th
    # column 2.5 columns out, so the rough estimate is not good enough on its own.
    sm = np.convolve(prof, np.ones(3) / 3, mode="same")
    thr = sm.mean() + 0.3 * sm.std()
    pk = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > thr]
    keep = []
    for i in pk:                                       # one peak per column: the strongest within 0.6 pitch
        if keep and i - keep[-1] < 0.6 * pitch_px:
            if sm[i] > sm[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    if len(keep) < 4:
        return None, None
    x = np.array(keep, float)
    p0, pp = x[0], pitch_px
    # Fit outwards in stages (15, 30, 60 ... columns): with the rough pitch, numbering all columns at once
    # drifts by several columns at the far end and the fit then confirms the wrong pitch.
    span = 15
    while True:
        near = x[x <= p0 + span * pp]
        idx = np.round((near - p0) / pp)
        ok = np.abs(near - (p0 + idx * pp)) < 0.3 * pp
        if ok.sum() < 4:
            return None, None
        pp, p0 = np.polyfit(idx[ok], near[ok], 1)
        if len(near) == len(x):
            break
        span *= 2
    first = p0 % pp if p0 >= pp else p0
    return pp / z, float(first)


def band_table(sh, area_right):
    """The data-band table found from the image: rows identified by their labels, column pitch measured."""
    hr = rules(sh, (60, 900, area_right, 2330), min_len_pt=0.25 * (area_right - 60))
    best = None
    for st in reversed(_stacks(hr)):                     # bottom-most first: the data bands sit at the bottom
        top, bot = st[0][0], st[-1][0]
        vr = rules(sh, (st[0][1] - 5, top + 2, st[0][2] + 5, bot - 2), min_len_pt=0.9 * (bot - top - 4), vertical=True)
        xs = sorted(v[0] for v in vr)
        if len(xs) < 2:
            continue
        strip = (xs[0], xs[1])
        rows = [[st[i][0], st[i + 1][0], "", None] for i in range(len(st) - 1)]
        boxes = ocr_boxes(sh.crop((strip[0] + 1, top, strip[1] - 1, bot)))
        if boxes is not None:
            for text, yc, _ in sorted(boxes, key=lambda b: (b[1], b[2])):
                y = top + yc / sh.z
                for r in rows:
                    if r[0] <= y < r[1]:
                        r[2] += text + " "
            for r in rows:
                r[3] = field_of(r[2])
        elif len(rows) >= len(ROW_FIELDS):                 # no OCR: assume this drawing set's order, and say so
            for r, f in zip(rows, ["schematic"] * (len(rows) - len(ROW_FIELDS)) + ROW_FIELDS):
                r[3] = f
        n_known = sum(1 for r in rows if r[3] and r[3] != "schematic")
        if not best or n_known > best["n_known"]:
            best = {"rows": [{"top": round(r[0], 1), "bottom": round(r[1], 1), "label": r[2].strip(), "field": r[3]} for r in rows],
                    "strip": [round(strip[0], 1), round(strip[1], 1)], "end": round(xs[-1] if len(xs) > 2 else st[0][2], 1),
                    "n_known": n_known, "labels_checked": boxes is not None}
        if n_known >= 3:
            break
    if not best or best["n_known"] < 3:
        return None
    ch = next((r for r in best["rows"] if r["field"] == "chainage"), best["rows"][-1])
    x0, y0 = sh.px(best["strip"][1] + 2, ch["top"] + 2)
    x1, y1 = sh.px(best["end"] - 2, ch["bottom"] - 2)
    prof = (sh.gray[int(y0):int(y1), int(x0):int(x1)] < DARK).mean(axis=0).astype(float)
    pitch, phase = _pitch(prof, sh.z)
    best["pitch"] = round(pitch, 3) if pitch else None
    best["first_x"] = round(best["strip"][1] + 2 + phase / sh.z, 1) if pitch else None
    return best


# ---------------------------------------------------------------- right panel

def right_panel(sh):
    vr = rules(sh, (60, 100, 3725, 2300), min_len_pt=0.6 * 2200, vertical=True)
    xs = [v[0] for v in vr if 1885 < v[0] < 3700]
    return min(xs) if xs else None


def panel_sections(sh, panel_x, right=3728, top=57, bottom=2328):
    """{'tbm': rect, 'title': rect} located by the headings printed in the right panel (OCR, ~30 s)."""
    boxes = ocr_boxes(sh.crop((panel_x + 3, top, right, bottom)))
    if not boxes:
        return {}
    heads = sorted((top + y / sh.z, t.upper()) for t, y, _ in boxes if re.search(HEADINGS, t.upper()))
    hr = [r[0] for r in rules(sh, (panel_x + 3, top, right, bottom), min_len_pt=0.9 * (right - panel_x - 3))]

    def above(y):
        return max([r for r in hr if r < y - 2], default=top)
    out = {}
    tbm = next((i for i, (y, t) in enumerate(heads) if "TBM" in t or "BENCH" in t), None)
    if tbm is not None:
        nxt = next((y for y, t in heads[tbm + 1:] if not re.search(r"TBM|BENCH", t)), bottom)
        out["tbm"] = (panel_x, above(heads[tbm][0]), right + 8, above(nxt))
    cl = next((i for i, (y, t) in enumerate(heads) if "CLIENT" in t), None)
    if cl is not None:
        nxt = next((y for y, t in heads[cl + 1:] if "CLIENT" not in t), None)
        out["title"] = (panel_x, above(heads[cl][0]), right + 8, above(nxt) if nxt else bottom + 5)
    return out


# ---------------------------------------------------------------- verdict

def analyse(sh):
    """Layout of the sheet image: status, notes, and the detected geometry."""
    notes = []
    panel_x = right_panel(sh)
    if panel_x is None:
        notes.append("No right-hand panel (notes / TBM / title block column) was found.")
    elif abs(panel_x - KNOWN_PANEL_X) > 6:
        notes.append(f"The right-hand panel starts at {panel_x:.0f} pt, not {KNOWN_PANEL_X:.0f} pt as on the trained sheets.")
    bt = band_table(sh, (panel_x or 3725) - 5)
    status = "known"
    if bt is None:
        status = "unknown"
        notes.append("No data-band table with recognisable row labels was found.")
    else:
        fields = [r["field"] for r in bt["rows"] if r["field"] != "schematic"]
        missing = [f for f in ROW_FIELDS if f not in fields]
        extra = [r["label"] or "(blank)" for r in bt["rows"] if not r["field"]]
        if not bt["labels_checked"]:
            notes.append("Row labels were not checked (the OCR package rapidocr_onnxruntime is not installed); "
                         "the rows are assumed to be in the trained order.")
        if missing:
            status = "unknown"
            notes.append("Band rows not found: " + ", ".join(missing) + ".")
        if extra:
            notes.append("Band rows the model was not trained on (not read): " + "; ".join(extra) + ".")
        if fields != ROW_FIELDS:
            if status == "known":
                status = "similar"
            if not missing:
                notes.append("The band rows are in a different order from the trained sheets; they are re-stacked "
                             "into the trained order before reading.")
        data_rows = [r for r in bt["rows"] if r["field"] not in (None, "schematic")]
        heights = [r["bottom"] - r["top"] for r in data_rows]
        if status == "known" and data_rows and not KNOWN_BAND_TOP[0] <= data_rows[0]["top"] <= KNOWN_BAND_TOP[1]:
            status = "similar"
            notes.append(f"The data bands are at a different height on the sheet (top at {data_rows[0]['top']:.0f} pt).")
        if status == "known" and (len(heights) != len(KNOWN_ROW_H) or any(abs(h - k) > 4 for h, k in zip(heights, KNOWN_ROW_H))):
            status = "similar"
            notes.append("The band rows have different heights from the trained sheets.")
        if bt["pitch"] is None:
            status = "unknown"
            notes.append("The band column spacing could not be measured.")
        elif abs(bt["pitch"] - KNOWN_PITCH) > 0.03 * KNOWN_PITCH:
            if status == "known":
                status = "similar"
            notes.append(f"Band columns are {bt['pitch']:.2f} pt apart, not {KNOWN_PITCH} pt as on the trained sheets.")
    if status == "known" and panel_x is not None and abs(panel_x - KNOWN_PANEL_X) > 6:
        status = "similar"
    return {"status": status, "notes": notes, "panel_x": panel_x, "band": bt}
