"""Read a whole Plan & L-Section sheet IMAGE (50-200 dpi) with the fine-tuned model.

A whole A0 sheet is far too large for the model to read in one image (at 1400 px wide its text is ~4 px
tall). So the reader does what a person would do with a magnifier:

  1. measures the image's dpi from the drawing's frame lines and brings it to the training scale (150 dpi),
     then checks the layout (layout.py): this drawing set's layout uses exactly the trained crops; another
     layout is read from the detected table rules, labels and column spacing, or reported as unreadable
  2. tiles the drawing into training-size crops and asks each "find all bridge callouts" (trained task)
  3. crops around each bridge callout + level block and asks "read the callout as JSON" (trained task)
  4. finds the L-section data bands, reads their chainage range once, then for each bridge crops the band
     columns around it (with the row labels, as in training), reads the two printed columns either side of the
     bridge chainage (trained task) and interpolates every row between them (y = y1 + (y2 - y1)/(x2 - x1) * (x - x1))
  5. reads the title block and the TBM table (found by their headings if they are not at the usual place)
  6. checks everything it read (band arithmetic, FL vs MIN FL REQ., level block vs bands) so misreads
     are flagged rather than passed on silently

The coordinates used below are PDF points of these drawings (A0, 3770 x 2384 pt); the image is mapped
onto them from its frame lines, so any dpi and small margins/offsets are handled.
"""
import json
import re

import numpy as np
from PIL import Image, ImageDraw

Z = 150 / 72                     # training scale: px per pt
TILE = 1008
# Tiles overlap by half a tile (242 pt): bigger than the largest callout (about 204 x 230 pt), so every
# callout lies wholly inside at least one tile.
STRIDE = 504
FRAME_L, FRAME_R = 46.8, 3731.8  # pt: outer frame lines of these sheets (left, right)
FRAME_T = 53.0                   # pt: outer frame line (top)
PITCH = 11.34                    # pt between data-band columns (one every 20 m)
COL_HALF = 5.55                  # pt: half a column's width + margin, cut in the gap between columns
ROWS_Y = (1863.0, 2236.0)        # pt: data-band rows (cut/fill ... chainage) incl. margin
DRAW_X, DRAW_Y = (45.0, 3165.0), (55.0, 1650.0)   # pt: plan + L-section profile, where callouts are printed
TITLE = (3175, 1868, 3740, 2335)
TBM = (3175, 490, 3740, 872)
ROW_ORDER = ["cut_fill", "fl_difference", "prop_rl", "prop_fl", "track_distance", "exg_up_fl", "ground_level", "chainage"]
RAIL = 0.762                     # note 5
TOL = 0.0025

Q_FIND = "Find all bridge callouts visible in this drawing crop and give their bounding boxes as JSON."
Q_BRIDGE = "Read the callout for bridge {bid} in this crop and return its details as JSON."
Q_RANGE = "Which chainages do the data bands in this crop cover?"
Q_BAND = "{name} is at chainage {ch}. Give the data-band values at the nearest column as JSON."
Q_TITLE = "Read the title block and return the sheet metadata as JSON."
Q_TBM = "Read the TBM details table and return every row as JSON."


# ---------------------------------------------------------------- parsing answers

def parse_json(text):
    m = re.search(r"```json\s*(.*?)```", text, re.S) or re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def g(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, (int, float)) else str(v)


# ---------------------------------------------------------------- image geometry

def lines(dark_profile, thr):
    xs = np.where(dark_profile > thr)[0]
    out = []
    for x in xs:
        if out and x - out[-1][-1] <= 2:
            out[-1].append(x)
        else:
            out.append([x])
    return [float(np.mean(l)) for l in out]


class Sheet:
    """The input image mapped onto the drawing's point coordinates at the training scale."""

    def __init__(self, img):
        img = img.convert("RGB")
        g0 = np.asarray(img.convert("L"))
        h, w = g0.shape
        v = lines((g0[int(h * 0.3):int(h * 0.7)] < 140).mean(axis=0), 0.95)
        self.warnings = []
        if len(v) >= 2 and v[-1] - v[0] > w * 0.8:
            dpi = (v[-1] - v[0]) / (FRAME_R - FRAME_L) * 72
            left_px = v[0]
        else:                                   # no frame found: assume the whole sheet fills the image
            dpi = w / (3770 / 72)
            left_px = FRAME_L * dpi / 72
            self.warnings.append("The drawing frame was not found; the scale is estimated from the image width.")
        self.dpi = dpi
        s = (150 / dpi)
        self.img = img.resize((round(w * s), round(h * s)), Image.LANCZOS) if abs(s - 1) > 0.01 else img
        gw = np.asarray(self.img.convert("L"))
        hz = lines((gw[:, int(gw.shape[1] * 0.1):int(gw.shape[1] * 0.8)] < 140).mean(axis=1), 0.6)
        top_px = hz[0] if hz else FRAME_T * Z
        self.dx = left_px * s - FRAME_L * Z     # px shift of this image against the drawing's coordinates
        self.dy = top_px - FRAME_T * Z
        if dpi < 100:
            self.warnings.append(f"The image is about {dpi:.0f} dpi: text is only {9.9 * dpi / 72:.0f} px tall, "
                                 f"so expect misread digits (best at 150-200 dpi).")
        self.gray = gw
        self.z = Z                                # px per pt in self.img (the training scale)

    def px(self, x_pt, y_pt):
        return x_pt * Z + self.dx, y_pt * Z + self.dy

    def crop(self, rect, size=None):
        """Crop a rectangle given in points, as the model saw crops in training (white outside the image)."""
        x0, y0 = self.px(rect[0], rect[1])
        x1, y1 = self.px(rect[2], rect[3])
        W, H = (size or (round(x1 - x0), round(y1 - y0)))
        out = Image.new("RGB", (W, H), "white")
        part = self.img.crop((round(x0), round(y0), round(x0) + W, round(y0) + H))
        out.paste(part, (0, 0))
        return pad28(out) if not size else out

    def band_table(self):
        """x of the band label strip's left/right borders and of the table's right end (points)."""
        y0, y1 = self.px(0, 1875)[1], self.px(0, 2215)[1]
        # Thin table rules turn light grey when a low-dpi image is scaled up, so count anything clearly
        # darker than the paper; a rule is a column that is "dark" over nearly the whole band height.
        xs = [(x - self.dx) / Z for x in lines((self.gray[int(y0):int(y1)] < 215).mean(axis=0), 0.9)]
        inner = [x for x in xs if 60 < x < 3170]
        if len(inner) < 3:
            return None
        return inner[0], inner[1], inner[-1]


def pad28(img):
    W, H = -(-img.width // 28) * 28, -(-img.height // 28) * 28
    if (W, H) == img.size:
        return img
    out = Image.new("RGB", (W, H), "white")
    out.paste(img, (0, 0))
    return out


# ---------------------------------------------------------------- reading

class Reader:
    def __init__(self, model, log=print):
        self.model = model          # anything with ask(images, question) -> str (runpod/app.py Model)
        self.log = log

    def ask(self, imgs, q, meta):
        return self.model.ask(imgs, q, meta=meta) if getattr(self.model, "wants_meta", False) else self.model.ask(imgs, q)

    # 1-2: find bridges
    def find_bridges(self, sh, area=(DRAW_X, DRAW_Y)):
        (X0, X1), (Y0, Y1) = area
        side, step = TILE / Z, STRIDE / Z
        xs = np.arange(X0, X1 - side + step, step)
        ys = np.arange(Y0, Y1 - side + step, step)
        found = {}
        n = 0
        for y in ys:
            for x in xs:
                x0, y0 = min(x, X1 - side), min(y, Y1 - side)
                clip = (x0, y0, x0 + side, y0 + side)
                ans = self.ask([sh.crop(clip, (TILE, TILE))], Q_FIND, {"kind": "find", "clip": clip})
                n += 1
                for item in parse_json(ans) or []:
                    if not isinstance(item, dict) or not isinstance(item.get("bbox_2d"), list) or len(item["bbox_2d"]) != 4:
                        continue
                    m = re.match(r"bridge (.+) \((plan|L-section) callout\)", str(item.get("label", "")))
                    if not m:
                        continue
                    bx = [clip[0] + item["bbox_2d"][0] / Z, clip[1] + item["bbox_2d"][1] / Z,
                          clip[0] + item["bbox_2d"][2] / Z, clip[1] + item["bbox_2d"][3] / Z]
                    edge = min(item["bbox_2d"][0], item["bbox_2d"][1], TILE - item["bbox_2d"][2], TILE - item["bbox_2d"][3]) < 4
                    found.setdefault((m.group(1), m.group(2)), []).append((edge, bx))
        self.log(f"  asked {n} tiles, found {len(found)} callouts")
        best = {}
        for key, cands in found.items():
            whole = [b for e, b in cands if not e] or [b for _, b in cands]
            best[key] = max(whole, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return best

    # 3: read each bridge
    def read_bridge(self, sh, bid, box, view):
        side = TILE / Z
        cx = (box[0] + box[2]) / 2
        cy = box[1] + 140 if view == "L-section" else (box[1] + box[3]) / 2     # include the level block below
        clip = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        ans = self.ask([sh.crop(clip, (TILE, TILE))], Q_BRIDGE.format(bid=bid), {"kind": "bridge", "clip": clip, "bid": bid})
        return parse_json(ans) if isinstance(parse_json(ans), dict) else None, ans

    # 4: data bands
    def band_image(self, sh, layout, start_idx):
        """Label strip + 16 columns, as in training. layout["rows"] (other layouts) re-stacks the rows in the
        trained order, so the model sees the arrangement it learnt even when the sheet orders them differently."""
        left, right, _ = layout["table"]
        pitch, first_x = layout.get("pitch", PITCH), layout.get("first_x", right + 10.9)
        x0 = first_x + start_idx * pitch - COL_HALF
        x1 = first_x + (start_idx + 15) * pitch + COL_HALF
        if not layout.get("rows"):
            strip = sh.crop((left + 3, ROWS_Y[0], right + 0.5, ROWS_Y[1]))
            band = sh.crop((x0, ROWS_Y[0], x1, ROWS_Y[1]))
        else:
            parts = []
            rows = layout["rows"]
            for i, f in enumerate(ROW_ORDER):
                r = rows.get(f)
                if r is None:                        # a row the sheet does not have: blank, and its value dropped later
                    h = int(round(45 * Z))
                    parts.append((Image.new("RGB", (1, h), "white"), Image.new("RGB", (1, h), "white")))
                    continue
                y0 = r[0] - (1.8 if i == 0 else 0.5)
                y1 = r[1] + (2.6 if i == len(ROW_ORDER) - 1 else -0.5)
                h = int(round((y1 - y0) * Z))
                parts.append((sh.crop((left + 3, y0, right + 0.5, y1), (int(round((right - left - 2.5) * Z)), h)),
                              sh.crop((x0, y0, x1, y1), (int(round((x1 - x0) * Z)), h))))
            sw = max(a.width for a, _ in parts)
            bw = max(b.width for _, b in parts)
            hh = sum(a.height for a, _ in parts)
            strip, band = Image.new("RGB", (sw, hh), "white"), Image.new("RGB", (bw, hh), "white")
            y = 0
            for a, b in parts:
                strip.paste(a, (0, y))
                band.paste(b, (0, y))
                y += a.height
        img = Image.new("RGB", (strip.width + band.width, max(strip.height, band.height)), "white")
        img.paste(strip, (0, 0))
        img.paste(band, (strip.width, 0))
        return pad28(img), (x0, x1)

    def band_range(self, sh, layout, start_idx):
        img, (x0, x1) = self.band_image(sh, layout, start_idx)
        ans = self.ask([img], Q_RANGE, {"kind": "range", "xrange": (x0, x1)})
        m = re.search(r"chainage\s+([\d.]+)\s+to\s+([\d.]+)", ans)
        if not m:
            return None
        step = re.search(r"one every\s+([\d.]+)\s*m", ans)
        return float(m.group(1)), float(m.group(2)), float(step.group(1)) if step else 20.0

    def band_layout(self, sh, found=None, trained=False):
        """Where the data bands are and the chainage of their first column: {table, start_ch, n_cols, ...} or None.
        found: the band table detected by layout.analyse() (for another layout; trained=True keeps the trained crop
        style when the sheet is this drawing set's layout but the fixed search missed the table)."""
        if found is None:
            table = sh.band_table()
            if not table:
                return None
            lay = {"table": list(table)}
            n_cols = int(round((table[2] - 5.6 - (table[1] + 10.9)) / PITCH)) + 1
        elif trained:
            lay = {"table": [found["strip"][0], found["strip"][1], found["end"]]}
            n_cols = int(round((found["end"] - 5.6 - (found["strip"][1] + 10.9)) / PITCH)) + 1
        else:
            lay = {"table": [found["strip"][0], found["strip"][1], found["end"]], "pitch": found["pitch"],
                   "first_x": found["first_x"],
                   "rows": {r["field"]: (r["top"], r["bottom"]) for r in found["rows"] if r["field"] in ROW_ORDER}}
            n_cols = int((found["end"] - 5.6 - found["first_x"]) / found["pitch"]) + 1
        rng = self.band_range(sh, lay, 0)
        if not rng:
            return None
        lay.update({"start_ch": rng[0], "n_cols": n_cols})
        if rng[2] != 20.0:
            lay["step"] = rng[2]
        return lay

    def read_column(self, sh, layout, i, name):
        """The printed band column number i (0 = first), read by the model, or None."""
        xc = layout["start_ch"] + i * layout.get("step", 20)
        s = max(0, min(i - 8, layout["n_cols"] - 16))
        img, (x0, x1) = self.band_image(sh, layout, s)
        band = parse_json(self.ask([img], Q_BAND.format(name=name, ch=g(xc)), {"kind": "band", "xrange": (x0, x1), "ch": xc}))
        col = band.get("nearest_column") if isinstance(band, dict) else None
        if not isinstance(col, dict) or not isinstance(col.get("chainage"), (int, float)) or abs(col["chainage"] - xc) > 0.5:
            return None                                     # not read, or the model read another column
        if layout.get("rows"):                              # values of rows this sheet does not have were not seen
            for f in ROW_ORDER[:-1]:
                if f not in layout["rows"]:
                    col.pop(f, None)
        return col

    def band_at(self, sh, layout, ch, name):
        """Band values at chainage ch (rule, Ankur 2026-10-05): read the two printed columns either side, x1 and x2,
        and interpolate every row, y = y1 + (y2 - y1) / (x2 - x1) * (x - x1); on a printed column, y is that column.
        The model only reads the columns; the arithmetic is done here. None if ch is off the bands."""
        step = layout.get("step", 20)
        k = (ch - layout["start_ch"]) / step
        exact = abs(k - round(k)) < 1e-6
        idx = [int(round(k))] if exact else [int(np.floor(k)), int(np.floor(k)) + 1]
        if not all(0 <= i < layout["n_cols"] for i in idx):
            return None
        cols = [self.read_column(sh, layout, i, name) for i in idx]
        out = {"chainage": ch, "x1": layout["start_ch"] + idx[0] * step, "x2": layout["start_ch"] + idx[-1] * step}
        if any(c is None for c in cols):
            out["error"] = ("the printed column at CH " + " and ".join(g(out[x]) for x, c in zip(("x1", "x2"), cols) if c is None)
                            + " could not be read, so the value cannot be interpolated")
            out["columns"] = [c for c in cols if c]
            return out
        c1, c2 = cols[0], cols[-1]
        t = 0.0 if exact else (ch - c1["chainage"]) / (c2["chainage"] - c1["chainage"])
        out.update({"t": round(t, 4), "columns": cols if not exact else [c1],
                    "y": {f: round(c1[f] + (c2[f] - c1[f]) * t, 3) for f in ROW_ORDER[:-1] if f in c1 and f in c2}})
        return out

    # 5: title block and TBM table
    def panel_reads(self, sh, lay):
        """Title block and TBM table: at the trained positions first; if the answer is not a valid title block /
        TBM table (or the panel is elsewhere), find them by their printed headings and read them there."""
        from layout import panel_sections
        out, where = {}, {}
        jobs = (("title", TITLE, Q_TITLE, lambda v: isinstance(v, dict) and any(v.get(k) for k in ("sheet_no", "drawing_no", "title"))),
                ("tbm", TBM, Q_TBM, lambda v: isinstance(v, list) and any(isinstance(r, dict) and r.get("tbm_id") for r in v)))
        trained_panel = lay["panel_x"] is not None and abs(lay["panel_x"] - 3175.4) <= 6
        sections = None
        for key, rect, q, valid in jobs:
            val = None
            if trained_panel or lay["panel_x"] is None:
                val = parse_json(self.ask([sh.crop(rect)], q, {"kind": key, "clip": rect}))
            if not valid(val) and lay["panel_x"] is not None:
                if sections is None:
                    self.log("  looking for the title block / TBM table by their headings ...")
                    sections = panel_sections(sh, lay["panel_x"])
                if key in sections:
                    val = parse_json(self.ask([sh.crop(sections[key])], q, {"kind": key, "clip": sections[key]}))
                    if valid(val):
                        where[key] = "found by its heading"
            out[key] = val if valid(val) else None
        return out["title"], out["tbm"], where

    def read(self, image):
        import layout as L
        sh = Sheet(image)
        result = {"dpi": round(sh.dpi), "warnings": list(sh.warnings), "bridges": [], "title": None, "tbm": None, "findings": []}
        self.log(f"image {image.size[0]} x {image.size[1]} px, about {sh.dpi:.0f} dpi")
        self.log("checking the sheet layout ...")
        lay = L.analyse(sh)
        known = lay["status"] == "known"
        self.log(f"  layout: {lay['status']}" + (f" - {'; '.join(lay['notes'])}" if lay["notes"] else ""))
        self.log("finding bridge callouts ...")
        if known:
            calls = self.find_bridges(sh)
        else:
            band_top = min(r["top"] for r in lay["band"]["rows"]) if lay["band"] else 2330.0
            calls = self.find_bridges(sh, ((FRAME_L - 2, (lay["panel_x"] or FRAME_R) - 5), (FRAME_T + 2, band_top)))
        ids = sorted({bid for bid, _ in calls})
        if known:
            layout = self.band_layout(sh)
            if layout is None and lay["band"]:      # the fixed search missed the table: use the one found, trained crop style
                layout = self.band_layout(sh, lay["band"], trained=True)
        elif lay["band"] and lay["status"] == "similar":
            layout = self.band_layout(sh, lay["band"])
        else:
            layout = None
        result["band_layout"] = layout
        if not layout:
            result["warnings"].append("The L-section data bands were not found; band values were not read." if known or not lay["band"]
                                      else "The data-band rows could not all be identified; band values were not read.")
        self.log(f"reading {len(ids)} bridges ...")
        for bid in ids:
            view = "L-section" if (bid, "L-section") in calls else "plan"
            box = calls[(bid, view)]
            data, raw = self.read_bridge(sh, bid, box, view)
            rec = {"bridge_id": bid, "read_from": f"{view} callout", "box_pt": [round(v, 1) for v in box], "data": data,
                   "raw": None if data else raw, "band": None, "checks": []}
            ch = (data or {}).get("chainage_m")
            if data and isinstance(ch, (int, float)) and layout:
                rec["band"] = self.band_at(sh, layout, ch, bid if bid.startswith(("ROB", "LC")) else f"Bridge {bid}")
            rec["checks"] = check_bridge(rec)
            result["bridges"].append(rec)
        self.log("reading title block and TBM table ...")
        result["title"], result["tbm"], where = self.panel_reads(sh, lay)
        result["findings"] = [c for b in result["bridges"] for c in b["checks"]]
        result["layout"] = verdict(lay, result, where)
        # the band rows' printed labels (e.g. "EXG. DN LINE FL"), so answers use the sheet's own wording
        result["band_labels"] = {r["field"]: r["label"] for r in (lay["band"] or {}).get("rows", []) if r.get("field") and r.get("label")}
        if result["layout"]["confidence"] != "high":
            result["warnings"] += result["layout"]["warnings"]
        result["overlay"] = overlay(sh, result)
        return result


# ---------------------------------------------------------------- layout verdict (warn instead of guessing)

LEVELS_OF_TRUST = ["low", "medium", "high"]


def band_adds_up(col):
    try:
        return (abs(col["prop_rl"] - col["prop_fl"] - RAIL) <= TOL and abs(col["prop_fl"] - col["ground_level"] - col["cut_fill"]) <= TOL
                and abs(col["prop_fl"] - col["exg_up_fl"] - col["fl_difference"]) <= TOL)
    except (KeyError, TypeError):
        return False


def verdict(lay, result, where):
    """How far the reading can be trusted, from the layout found and from checks on what was read."""
    notes = list(lay["notes"])
    lower = lambda a, b: min(a, b, key=LEVELS_OF_TRUST.index)      # noqa: E731
    cols = [c for b in result["bridges"] for c in band_columns(b["band"])]
    full = [c for c in cols if all(k in c for k in ROW_ORDER[:-1])]
    ok = sum(1 for c in full if band_adds_up(c))
    conf = {"known": "high", "similar": "medium", "unknown": "low"}[lay["status"]]
    if len(full) >= 3 and ok / len(full) < 0.5:
        conf = "low"
        notes.append(f"Only {ok} of {len(full)} band columns add up (FL - GL = cut/fill, RL - FL = 0.762, FL difference): "
                     "the band rows probably mean something different on this sheet.")
    elif len(full) >= 3 and ok / len(full) < 0.8:
        conf = lower(conf, "medium")
        notes.append(f"{len(full) - ok} of {len(full)} band columns do not add up - several values are probably misread.")
    if result["title"] is None:
        conf = lower(conf, "medium")
        notes.append("The title block was not found or could not be read.")
    if result["tbm"] is None:
        notes.append("The TBM table was not found or could not be read.")
    if not result["bridges"] and result["title"] is None:
        conf = "low"
        notes.append("No bridges and no title block were found: this may not be a Plan & L-Section sheet.")
    for k, w in where.items():
        notes.append(f"The {'title block' if k == 'title' else 'TBM table'} was not at its usual position; it was {w}.")
    lead = {"high": None,
            "medium": "This sheet's layout differs from the sheets the model was trained on; check the values.",
            "low": "This sheet's layout differs from the sheets the model was trained on; treat the values as unreliable."}[conf]
    return {"status": lay["status"], "confidence": conf, "notes": notes,
            "band_columns_checked": len(full), "band_columns_adding_up": ok,
            "warnings": ([lead] if lead else []) + notes}


# ---------------------------------------------------------------- checks on what was read

def band_columns(band):
    """The printed columns behind a band result (new: x1/x2; readings saved by the earlier version: one nearest column)."""
    if not isinstance(band, dict):
        return []
    if "columns" in band:
        return [c for c in band["columns"] if isinstance(c, dict)]
    return [band["nearest_column"]] if isinstance(band.get("nearest_column"), dict) else []


def band_values(band):
    """The band values at the asked chainage: interpolated y (new), or the nearest column (earlier readings)."""
    if not isinstance(band, dict):
        return {}
    if "y" in band:
        return band["y"] or {}
    return band.get("nearest_column") or {} if "nearest_column" in band else {}


def check_bridge(rec):
    out = []
    bid, d, band = rec["bridge_id"], rec["data"] or {}, rec["band"] or {}
    if not rec["data"]:
        return [{"severity": "INFO", "bridge": bid, "message": f"Bridge {bid}: the callout could not be read."}]
    lv = d.get("levels") or {}
    fl, req = lv.get("proposed_formation_level"), lv.get("min_formation_level_required")
    if isinstance(fl, (int, float)) and isinstance(req, (int, float)) and fl < req:
        out.append({"severity": "FLAG", "bridge": bid,
                    "message": f"FLAG: For bridge {bid} min. FL = {g(req)}, and FL = {g(fl)} ({g(req - fl)} m below the minimum required)."})
    if isinstance(band, dict) and band.get("error"):
        out.append({"severity": "INFO", "bridge": bid, "message": f"Bridge {bid}: {band['error']}."})
    for col in band_columns(band):
        try:
            bad = []
            if abs(col["prop_rl"] - col["prop_fl"] - RAIL) > TOL:
                bad.append(f"RL - FL = {g(col['prop_rl'] - col['prop_fl'])} (note 5: 0.762)")
            if abs(col["prop_fl"] - col["ground_level"] - col["cut_fill"]) > TOL:
                bad.append(f"FL - GL = {g(col['prop_fl'] - col['ground_level'])} vs cut/fill {g(col['cut_fill'])}")
            if abs(col["prop_fl"] - col["exg_up_fl"] - col["fl_difference"]) > TOL:
                bad.append(f"FL - existing FL = {g(col['prop_fl'] - col['exg_up_fl'])} vs difference {g(col['fl_difference'])}")
            if bad:
                out.append({"severity": "CHECK", "bridge": bid,
                            "message": f"Bridge {bid}: band column {g(col.get('chainage'))} does not add up ({'; '.join(bad)}) - probably a misread digit."})
        except (KeyError, TypeError):
            out.append({"severity": "INFO", "bridge": bid, "message": f"Bridge {bid}: the band column was read incompletely."})
    y = band_values(band)
    if isinstance(fl, (int, float)) and isinstance(y.get("prop_fl"), (int, float)) and abs(y["prop_fl"] - fl) > 0.05:
        out.append({"severity": "CHECK", "bridge": bid,
                    "message": f"Bridge {bid}: level block FL {g(fl)} vs band FL {g(y['prop_fl'])} at CH {g(band.get('chainage', rec['data'].get('chainage_m')))}."})
    return out


def overlay(sh, result, width=2400):
    img = sh.img.copy()
    d = ImageDraw.Draw(img)
    flagged = {f["bridge"] for f in result["findings"] if f["severity"] in ("FLAG", "CHECK")}
    for b in result["bridges"]:
        x0, y0 = sh.px(b["box_pt"][0], b["box_pt"][1])
        x1, y1 = sh.px(b["box_pt"][2], b["box_pt"][3])
        col = (220, 30, 30) if b["bridge_id"] in flagged else (0, 150, 70)
        d.rectangle([x0, y0, x1, y1], outline=col, width=6)
        d.text((x0 + 4, y0 - 24), b["bridge_id"], fill=col)
    s = width / img.width
    return img.resize((width, round(img.height * s)), Image.LANCZOS)
