"""Read a whole Plan & L-Section sheet IMAGE (50-200 dpi) with the fine-tuned model.

A whole A0 sheet is far too large for the model to read in one image (at 1400 px wide its text is ~4 px
tall). So the reader does what a person would do with a magnifier:

  1. measures the image's dpi from the drawing's frame lines and brings it to the training scale (150 dpi)
  2. tiles the drawing into training-size crops and asks each "find all bridge callouts" (trained task)
  3. crops around each bridge callout + level block and asks "read the callout as JSON" (trained task)
  4. finds the L-section data bands, reads their chainage range once, then for each bridge crops the band
     columns around it (with the row labels, as in training) and asks for the nearest column (trained task)
  5. reads the title block and the TBM table
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
    def find_bridges(self, sh):
        side, step = TILE / Z, STRIDE / Z
        xs = np.arange(DRAW_X[0], DRAW_X[1] - side + step, step)
        ys = np.arange(DRAW_Y[0], DRAW_Y[1] - side + step, step)
        found = {}
        n = 0
        for y in ys:
            for x in xs:
                x0, y0 = min(x, DRAW_X[1] - side), min(y, DRAW_Y[1] - side)
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
    def band_image(self, sh, table, start_idx):
        left, right, _ = table
        first_x = right + 10.9
        x0 = first_x + start_idx * PITCH - COL_HALF
        x1 = first_x + (start_idx + 15) * PITCH + COL_HALF
        strip = sh.crop((left + 3, ROWS_Y[0], right + 0.5, ROWS_Y[1]))
        band = sh.crop((x0, ROWS_Y[0], x1, ROWS_Y[1]))
        img = Image.new("RGB", (strip.width + band.width, max(strip.height, band.height)), "white")
        img.paste(strip, (0, 0))
        img.paste(band, (strip.width, 0))
        return pad28(img), (x0, x1)

    def band_range(self, sh, table, start_idx):
        img, (x0, x1) = self.band_image(sh, table, start_idx)
        ans = self.ask([img], Q_RANGE, {"kind": "range", "xrange": (x0, x1)})
        m = re.search(r"chainage\s+([\d.]+)\s+to\s+([\d.]+)", ans)
        return (float(m.group(1)), float(m.group(2))) if m else None

    def band_layout(self, sh):
        """Where the data bands are and the chainage of their first column: {table, start_ch, n_cols} or None."""
        table = sh.band_table()
        if not table:
            return None
        n_cols = int(round((table[2] - 5.6 - (table[1] + 10.9)) / PITCH)) + 1
        rng = self.band_range(sh, table, 0)
        return {"table": list(table), "start_ch": rng[0], "n_cols": n_cols} if rng else None

    def band_at(self, sh, layout, ch, name):
        """The band column nearest chainage ch (no interpolation), read by the model, or None if ch is off the bands."""
        idx = int(round((ch - layout["start_ch"]) / 20))
        if not 0 <= idx < layout["n_cols"]:
            return None
        s = max(0, min(idx - 8, layout["n_cols"] - 16))
        img, (x0, x1) = self.band_image(sh, layout["table"], s)
        band = parse_json(self.ask([img], Q_BAND.format(name=name, ch=g(ch)), {"kind": "band", "xrange": (x0, x1), "ch": ch}))
        return band if isinstance(band, dict) else None

    def read(self, image):
        sh = Sheet(image)
        result = {"dpi": round(sh.dpi), "warnings": list(sh.warnings), "bridges": [], "title": None, "tbm": None, "findings": []}
        self.log(f"image {image.size[0]} x {image.size[1]} px, about {sh.dpi:.0f} dpi")
        self.log("finding bridge callouts ...")
        calls = self.find_bridges(sh)
        ids = sorted({bid for bid, _ in calls})
        layout = self.band_layout(sh)
        result["band_layout"] = layout
        if not layout:
            result["warnings"].append("The L-section data bands were not found; band values were not read.")
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
        title = parse_json(self.ask([sh.crop(TITLE)], Q_TITLE, {"kind": "title"}))
        result["title"] = title if isinstance(title, dict) else None
        tbm = parse_json(self.ask([sh.crop(TBM)], Q_TBM, {"kind": "tbm"}))
        result["tbm"] = tbm if isinstance(tbm, list) else None
        result["findings"] = [c for b in result["bridges"] for c in b["checks"]]
        result["overlay"] = overlay(sh, result)
        return result


# ---------------------------------------------------------------- checks on what was read

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
    col = band.get("nearest_column") if isinstance(band, dict) else None
    if isinstance(col, dict):
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
            if isinstance(fl, (int, float)) and abs(col["prop_fl"] - fl) > 0.05:
                out.append({"severity": "CHECK", "bridge": bid,
                            "message": f"Bridge {bid}: level block FL {g(fl)} vs band FL {g(col['prop_fl'])} at CH {g(col.get('chainage'))}."})
        except (KeyError, TypeError):
            out.append({"severity": "INFO", "bridge": bid, "message": f"Bridge {bid}: the band column was read incompletely."})
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
