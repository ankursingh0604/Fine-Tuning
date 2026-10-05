"""Agent / tool-use conversations for v4, generated from the annotations (every tool result is real data).

    .venv\\Scripts\\python scripts\\agent_conversations_v4.py

Each conversation: system prompt, user turn(s), assistant turns that call tools (docs/tools_v4.json), tool results
computed from the store of all annotated sheets, and the final answer. Situations (V4_PLAN.md, situation awareness
and reasoning rules): single lookups, interpolated band values, MIN FL checks, multi-step range questions, follow-ups,
ambiguous requests (asking back), bridges not found, values not printed, the rail-level check, curves, TBMs, which
sheet covers a chainage, unfamiliar terms (own guess, then confirm), and out-of-scope engineering decisions.
Assistant turns carry "reasoning" only where the reasoning rules call for thinking.

Writes data/v4/dataset/agent_{train,val,test}.jsonl, split by the sheet the conversation is about.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as D           # noqa: E402
import build_dataset_v4 as V        # noqa: E402

DS = ROOT / "data" / "v4" / "dataset"
TOOLS = json.loads((ROOT / "docs" / "tools_v4.json").read_text(encoding="utf-8"))["tools"]
SEED = 41
rng = random.Random(SEED)
SYSTEM = ("You are an assistant for railway Plan & L-Section drawings (Mathura-Nagda 3rd and 4th lines). "
          "Answer from the sheets through your tools and cite the sheet, chainage and column. The 3rd and 4th lines share "
          "chainages and bridge numbers: if a question does not say which line, ask. Band values between columns are "
          "interpolated, y = y1 + (y2 - y1)/(x2 - x1) * (x - x1); always give y with x1, x2, y1, y2. Use calc for every "
          "calculation. When FL is below MIN FL REQ., say 'FLAG: For bridge X min. FL = .., and FL = ..'. Never invent a "
          "value; say when something is not printed or failed a check. For an unfamiliar term, give your own guess, "
          "labelled as a guess, then confirm it from the sheet, the library or the web. Engineering decisions belong to "
          "the engineers.")
LEVEL_NAMES = {"existing_formation_level": "existing FL", "min_formation_level_required": "MIN FL REQ.",
               "proposed_formation_level": "FL", "bed_level": "bed level", "high_flood_level": "HFL", "free_board": "free board"}
LEVEL_ASK = {"existing_formation_level": ["existing FL", "existing formation level"],
             "min_formation_level_required": ["MIN FL REQ.", "minimum formation level required"],
             "proposed_formation_level": ["FL", "proposed formation level"], "bed_level": ["bed level"],
             "high_flood_level": ["HFL", "high flood level"], "free_board": ["free board"]}
BAND_ASK = {"ground_level": "ground level", "cut_fill": "cut or fill", "track_distance": "track distance",
            "prop_rl": "rail level", "prop_fl": "formation level from the data bands"}
BAND_WORD = {"ground_level": "ground level", "cut_fill": "cut(-)/fill(+)", "track_distance": "track distance",
             "prop_rl": "proposed RL", "prop_fl": "proposed FL", "exg_up_fl": "existing line FL", "fl_difference": "FL difference"}


def f3(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def chs(v):
    return D.ch_text(v)


# ---------------------------------------------------------------- the store (what the tools return)

class Store:
    def __init__(self, anns):
        self.bridges = defaultdict(dict)       # line -> bridge_id -> record
        self.cols = defaultdict(dict)          # line -> chainage -> (column, sheet_id)
        self.sheets = defaultdict(list)        # line -> [(lo, hi, sheet_id)]
        self.curves = defaultdict(dict)        # line -> curve_no -> record
        self.tbms = {}
        self.notes = {}
        self.abbr = {}
        self.rail_note = {}
        for a in anns:
            line = line_of(a)
            if not line:
                continue
            sid = a["sheet_id"]
            self.sheet_ann = getattr(self, "sheet_ann", {})
            self.sheet_ann[sid] = a
            self.sheet_line = getattr(self, "sheet_line", {})
            self.sheet_line[sid] = line
            facts = V.sheet_facts(a)
            V.rename_lines(a, facts)
            self.rail_note[sid] = facts["rail_note"]
            info = a["sheet_info"]
            lo, hi = (D_ch(info.get("chainage_from")), D_ch(info.get("chainage_to")))
            if lo is not None and hi is not None:
                self.sheets[line].append((lo, hi, sid))
            for b in a["bridges"]:
                if b.get("belongs_to") == "this sheet" and b.get("complete") and b.get("chainage_m"):
                    rec = {k: b.get(k) for k in ("bridge_id", "existing_on", "existing_type", "existing_span", "crossing",
                                                 "proposal", "category", "chainage_m")}
                    rec["levels"] = {k: v for k, v in (b.get("lsection_levels") or {}).items() if k != "bbox"} or None
                    rec["sheet_id"] = sid
                    self.bridges[line][b["bridge_id"]] = rec
            bd = a.get("bands") if isinstance(a.get("bands"), dict) else {}
            for c in bd.get("columns", []):
                if c["checks_ok"]:
                    self.cols[line].setdefault(c["chainage"], (c, sid))
            for c in a["curves"]:
                if c["location"] == "lsection_band" or c["curve_no"] not in self.curves[line]:
                    self.curves[line][c["curve_no"]] = {"curve_no": c["curve_no"], "hand": c["hand"], **c["params"], "sheet_id": sid}
            for t in a["tbm_benchmarks"]:
                self.tbms[t["tbm_id"]] = {k: t[k] for k in ("tbm_id", "chainage_m", "easting", "northing", "msl_m", "description")}
                self.tbms[t["tbm_id"]]["sheet_id"] = sid
            self.notes[sid] = a["notes"]
            self.sheet_abbr = getattr(self, "sheet_abbr", {})
            self.sheet_abbr[sid] = {k: v for k, v in a["abbreviations"].items() if complete_meaning(v)}
            for k, v in self.sheet_abbr[sid].items():
                self.abbr.setdefault(k, v)
        for line in self.cols:
            self.sorted_cols = getattr(self, "sorted_cols", {})
            self.sorted_cols[line] = sorted(self.cols[line])

    def band_at(self, line, ch):
        xs = self.sorted_cols.get(line, [])
        for x1, x2 in zip(xs, xs[1:]):
            if x1 == ch or x1 < ch < x2:
                if x1 == ch:
                    x2 = x1
                if x2 != x1 and x2 - x1 != 20:
                    return {"chainage": ch, "error": "a printed column between them could not be read; not interpolated"}
                c1, s1 = self.cols[line][x1]
                c2, _ = self.cols[line][x2]
                y, t = D.interpolate(c1, c2, ch)
                strip = lambda c: {k: v for k, v in c["values"].items() if k != "chainage"}     # noqa: E731
                return {"chainage": ch, "sheet_id": s1, "x1": int(x1), "x2": int(x2), "column_x1": strip(c1),
                        "column_x2": strip(c2), "y": y, "rail_level_ok": [c1.get("rail_level_ok", True), c2.get("rail_level_ok", True)]}
        if xs and xs[-1] == ch:
            c1, s1 = self.cols[line][ch]
            return {"chainage": ch, "sheet_id": s1, "x1": int(ch), "x2": int(ch), "column_x1": c1["values"], "y": c1["values"]}
        return {"chainage": ch, "error": "not on the data bands of any sheet read for this line"}

    def sheet_for(self, line, ch):
        return [sid for lo, hi, sid in self.sheets[line] if lo <= ch <= hi]


def D_ch(km):
    m = re.match(r"(\d+)\+(\d+(?:\.\d+)?)", km or "")
    return int(m.group(1)) * 1000 + float(m.group(2)) if m else None


def line_of(a):
    bd = a.get("bands") if isinstance(a.get("bands"), dict) else {}
    p = (bd.get("lines") or {}).get("proposed")
    if not p:
        p = "4TH LINE" if "4TH LINE" in (a["sheet_info"].get("title") or "") else "3RD LINE" if a["sheet_info"].get("title") else None
    return p.lower() if p else None


def complete_meaning(v):
    return len(v.split()) >= 2 and not re.search(r"\b(of|to|the|and|for)$", v.strip(), re.I)


# ---------------------------------------------------------------- message helpers

def call(name, args, reasoning=None):
    m = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}
    if reasoning:
        m["reasoning"] = reasoning
    return m


def result(name, obj):
    return {"role": "tool", "name": name, "content": json.dumps(obj, ensure_ascii=False)}


def say(text, reasoning=None):
    m = {"role": "assistant", "content": text}
    if reasoning:
        m["reasoning"] = reasoning
    return m


def user(text):
    return {"role": "user", "content": text}


def conv(task, sheet, msgs, thinking):
    return {"task": task, "sheet": sheet, "thinking": thinking, "messages": [{"role": "system", "content": SYSTEM}] + msgs}


def calc_value(expr):
    return round(eval(expr, {"__builtins__": {}}), 3)      # expressions are built here from numbers only


def bname(b):
    return b["bridge_id"] if b["bridge_id"].startswith(("ROB", "LC")) else f"bridge {b['bridge_id']}"


def src(b, line):
    return f"{line}, sheet {b['sheet_id']}, CH {f3(b['chainage_m'])}"


# ---------------------------------------------------------------- situations

def lookup_level(S, line, b):
    lv = b["levels"]
    k = rng.choice(list(LEVEL_ASK))
    q = f"What is the {rng.choice(LEVEL_ASK[k])} at {bname(b)} on the {line}?"
    msgs = [user(q), call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]}), result("query", b)]
    if lv and lv.get(k) is not None:
        msgs.append(say(f"The {LEVEL_NAMES[k]} at {bname(b)} is {f3(lv[k])} m (level block; {src(b, line)})."))
    elif lv:
        msgs.append(say(f"The level block of {bname(b)} does not give a {LEVEL_NAMES[k]}; it is not printed on the sheet "
                        f"({src(b, line)}). Printed there: " + ", ".join(f"{LEVEL_NAMES[x]} {f3(v)}" for x, v in lv.items() if x in LEVEL_NAMES) + "."))
    else:
        msgs.append(say(f"No level block was read for {bname(b)} ({src(b, line)}), so its {LEVEL_NAMES[k]} is not available from the sheets."))
    return conv("agent_lookup", b["sheet_id"], msgs, False)


def band_value(S, line, b):
    k = rng.choice(list(BAND_ASK))
    ba = S.band_at(line, b["chainage_m"])
    q = f"What is the {BAND_ASK[k]} at {bname(b)} on the {line}?"
    msgs = [user(q), call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]}), result("query", b),
            call("band_at", {"line": line, "chainages": [b["chainage_m"]]}), result("band_at", [ba])]
    if "error" in ba:
        msgs.append(say(f"{bname(b).capitalize()} is at CH {f3(b['chainage_m'])}, but the band value there cannot be given: {ba['error']}."))
    else:
        y = ba["y"][k]
        if ba["x1"] == ba["x2"]:
            how = f"on the printed column CH {ba['x1']}"
        else:
            how = (f"interpolated between CH {ba['x1']} (y1 = {f3(ba['column_x1'][k])}) and CH {ba['x2']} "
                   f"(y2 = {f3(ba['column_x2'][k])})")
        extra = ""
        if k == "cut_fill":
            extra = f" - {'fill (embankment)' if y > 0 else 'cut' if y < 0 else 'at ground level'}"
        msgs.append(say(f"At {bname(b)} (CH {f3(b['chainage_m'])}, {line}, sheet {ba['sheet_id']}) the {BAND_WORD[k]} is "
                        f"y = {f3(y)}{' m' if k != 'cut_fill' else ' m'}{extra}, {how}."))
    return conv("agent_band_value", b["sheet_id"], msgs, False)


def min_fl(S, line, b):
    lv = b["levels"]
    fl, req = lv["proposed_formation_level"], lv["min_formation_level_required"]
    expr = f"{fl} - {req}"
    d = calc_value(expr)
    q = rng.choice(["Is the formation level high enough at {n} on the {l}?", "Does {n} ({l}) meet the minimum formation level?",
                    "Check FL against MIN FL REQ. for {n} on the {l}."]).format(n=bname(b), l=line)
    reasoning = (f"I need the level block of {bname(b)} on the {line}: its FL and MIN FL REQ. Then FL - MIN FL REQ.; "
                 f"if it is negative the bridge must be flagged in the agreed wording.")
    msgs = [user(q), call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]}, reasoning), result("query", b),
            call("calc", {"expression": expr}), result("calc", {"expression": expr, "value": d})]
    if d < 0:
        ans = (f"FLAG: For bridge {b['bridge_id']} min. FL = {f3(req)}, and FL = {f3(fl)} ({f3(-d)} m below the minimum required). "
               f"FL - MIN FL REQ. = {f3(fl)} - {f3(req)} = {f3(d)} m ({src(b, line)}).")
    else:
        ans = (f"Yes. FL {f3(fl)} - MIN FL REQ. {f3(req)} = {f3(d)} m, so the FL is {'exactly at' if d == 0 else 'above'} the minimum "
               f"required ({src(b, line)}).")
    return conv("agent_min_fl", b["sheet_id"], msgs + [say(ans, "FL - MIN FL REQ. is " + ("negative: flag it." if d < 0 else "not negative: no flag."))], True)


def range_fill(S, line, bs):
    lo, hi = bs[0]["chainage_m"] - rng.uniform(5, 300), bs[-1]["chainage_m"] + rng.uniform(5, 300)
    lo, hi = round(lo), round(hi)
    thr = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    in_range = [{"bridge_id": b["bridge_id"], "chainage_m": b["chainage_m"], "sheet_id": b["sheet_id"]} for b in bs]
    bands = [S.band_at(line, b["chainage_m"]) for b in bs]
    reasoning = (f"Find the bridges between CH {lo} and {hi} on the {line}, then the interpolated cut/fill at each bridge "
                 f"chainage, and keep those with fill (positive) above {thr} m.")
    msgs = [user(f"Which bridges between CH {lo} and CH {hi} on the {line} are on a fill of more than {thr} m?"),
            call("query", {"kind": "bridges_in_range", "line": line, "chainage_from": lo, "chainage_to": hi}, reasoning),
            result("query", in_range),
            call("band_at", {"line": line, "chainages": [b["chainage_m"] for b in bs]}), result("band_at", bands)]
    rows, hits, skipped = [], [], []
    for b, ba in zip(bs, bands):
        if "error" in ba:
            skipped.append(bname(b))
            continue
        cf = ba["y"]["cut_fill"]
        rows.append(f"{bname(b)} (CH {f3(b['chainage_m'])}): cut/fill y = {f3(cf)} m")
        if cf > thr:
            hits.append(f"{bname(b)} ({f3(cf)} m, sheet {b['sheet_id']})")
    think = "Interpolated cut/fill at each bridge: " + "; ".join(rows) + (f". No band value for: {', '.join(skipped)}." if skipped else ".")
    ans = (f"On the {line} between CH {lo} and {hi}, " + (f"{len(hits)} bridge{'s are' if len(hits) != 1 else ' is'} on a fill of more than "
           f"{thr} m: " + "; ".join(hits) + "." if hits else f"no bridge is on a fill of more than {thr} m.")
           + f" Checked {len(rows)} of {len(bs)} bridges; values interpolated between the band columns either side of each bridge."
           + (f" Not checked (band value not available): {', '.join(skipped)}." if skipped else ""))
    return conv("agent_range_fill", bs[0]["sheet_id"], msgs + [say(ans, think)], True)


def range_flags(S, line, bs):
    lo, hi = round(bs[0]["chainage_m"] - 50), round(bs[-1]["chainage_m"] + 50)
    recs = [{"bridge_id": b["bridge_id"], "chainage_m": b["chainage_m"], "sheet_id": b["sheet_id"], "levels": b["levels"]} for b in bs]
    flags = [b for b in bs if b["levels"] and b["levels"].get("proposed_formation_level") is not None
             and b["levels"].get("min_formation_level_required") is not None
             and b["levels"]["proposed_formation_level"] < b["levels"]["min_formation_level_required"]]
    no_lv = [b for b in bs if not (b["levels"] and b["levels"].get("min_formation_level_required") is not None)]
    reasoning = f"List the bridges between CH {lo} and {hi} on the {line} with their level blocks and compare FL with MIN FL REQ. for each."
    msgs = [user(f"Are any bridges between CH {lo} and {hi} on the {line} below the minimum formation level?"),
            call("query", {"kind": "bridges_in_range", "line": line, "chainage_from": lo, "chainage_to": hi}, reasoning), result("query", recs)]
    if flags:
        ans = "\n".join(f"FLAG: For bridge {b['bridge_id']} min. FL = {f3(b['levels']['min_formation_level_required'])}, and FL = "
                        f"{f3(b['levels']['proposed_formation_level'])} (sheet {b['sheet_id']})." for b in flags)
    else:
        ans = f"No. Every bridge between CH {lo} and {hi} on the {line} with a level block has FL at or above its MIN FL REQ."
    if no_lv:
        ans += f" No MIN FL REQ. was read for: {', '.join(bname(b) for b in no_lv)}."
    think = "; ".join(f"{bname(b)}: FL {f3(b['levels']['proposed_formation_level'])} vs MIN {f3(b['levels']['min_formation_level_required'])}"
                      for b in bs if b not in no_lv)
    return conv("agent_range_flags", bs[0]["sheet_id"], msgs + [say(ans, think)], True)


def follow_up(S, line, b, b2):
    first = lookup_level(S, line, b)["messages"][1:]
    k = rng.choice([x for x in LEVEL_ASK if (b2["levels"] or {}).get(x) is not None] or ["proposed_formation_level"])
    msgs = first + [user(rng.choice([f"And {bname(b2)}?", f"What about {bname(b2)}?"])),
                    call("query", {"kind": "bridge", "line": line, "bridge_id": b2["bridge_id"]}), result("query", b2)]
    k0 = next((x for x in LEVEL_ASK if LEVEL_NAMES[x] in first[-1]["content"]), k)
    val = (b2["levels"] or {}).get(k0)
    msgs.append(say(f"The {LEVEL_NAMES[k0]} at {bname(b2)} is {f3(val)} m ({src(b2, line)})." if val is not None else
                    f"{bname(b2).capitalize()} has no {LEVEL_NAMES[k0]} printed in its level block ({src(b2, line)})."))
    return conv("agent_follow_up", b["sheet_id"], msgs, False)


def ambiguous_line(S, bid):
    b3, b4 = S.bridges["3rd line"][bid], S.bridges["4th line"][bid]
    k = rng.choice(["proposed_formation_level", "min_formation_level_required", "existing_formation_level"])
    msgs = [user(f"What is the {rng.choice(LEVEL_ASK[k])} at bridge {bid}?"),
            say(f"Bridge {bid} is on both drawing sets: the 3rd line (CH {f3(b3['chainage_m'])}, sheet {b3['sheet_id']}) and the "
                f"4th line (CH {f3(b4['chainage_m'])}, sheet {b4['sheet_id']}). Which line do you mean?",
                f"Bridge numbers repeat on the 3rd and 4th lines; bridge {bid} exists on both, so I must ask instead of choosing.")]
    line, b = rng.choice([("3rd line", b3), ("4th line", b4)])
    msgs += [user(f"The {line}."), call("query", {"kind": "bridge", "line": line, "bridge_id": bid}), result("query", b)]
    v = (b["levels"] or {}).get(k)
    msgs.append(say(f"On the {line}, the {LEVEL_NAMES[k]} at bridge {bid} is {f3(v)} m ({src(b, line)})." if v is not None
                    else f"On the {line}, bridge {bid}'s level block does not give a {LEVEL_NAMES[k]} ({src(b, line)})."))
    return conv("agent_ask_back", b["sheet_id"], msgs, True)


def not_found(S, line, b):
    num = re.match(r"\d+", b["bridge_id"])
    if not num:
        return None
    n = int(num.group())
    fake = str(n + rng.choice([700, 900]))
    near = sorted(S.bridges[line], key=lambda i: abs(int(re.match(r"\d+", i).group()) - int(fake)) if re.match(r"\d+", i) else 1e9)[:3]
    msgs = [user(f"Give me the details of bridge {fake} on the {line}."),
            call("query", {"kind": "bridge", "line": line, "bridge_id": fake}), result("query", {"error": f"bridge {fake} not found on the {line}"}),
            say(f"Bridge {fake} is not on any {line} sheet that has been read. The nearest bridge numbers on the {line} are "
                f"{', '.join(near)}. If it is on another sheet, upload that sheet and I will read it.")]
    return conv("agent_not_found", b["sheet_id"], msgs, False)


def rail_check(S, line, col_ch, sid):
    ba = S.band_at(line, col_ch)
    if "error" in ba:
        return None
    note = S.rail_note.get(sid, 5)
    v = ba["column_x1"]
    expr = f"{v['prop_rl']} - {v['prop_fl']}"
    d = calc_value(expr)
    reasoning = f"Read the band column at CH {chs(col_ch)} on the {line}, compute RL - FL and compare with the rail-level note ({note}) of 762 mm."
    msgs = [user(f"Does the rail level agree with the notes at CH {chs(col_ch)} on the {line}?"),
            call("band_at", {"line": line, "chainages": [col_ch]}, reasoning), result("band_at", [ba]),
            call("calc", {"expression": expr}), result("calc", {"expression": expr, "value": d})]
    ok = abs(d - 0.762) < 0.0025
    ans = (f"Yes. At CH {chs(col_ch)} (sheet {ba['sheet_id']}) RL - FL = {f3(v['prop_rl'])} - {f3(v['prop_fl'])} = {f3(d)} m, the 762 mm "
           f"that note {note} requires." if ok else
           f"No - CHECK. At CH {chs(col_ch)} (sheet {ba['sheet_id']}) RL - FL = {f3(v['prop_rl'])} - {f3(v['prop_fl'])} = {f3(d)} m, "
           f"but note {note} requires the rail level 762 mm above formation. The drawing should be checked here.")
    return conv("agent_rail_check", sid, msgs + [say(ans, f"RL - FL = {f3(d)} against 0.762: {'agrees' if ok else 'does not agree'}.")], True)


def curve(S, line, c):
    k = rng.choice(["radius", "transition_length", "cant", "max_speed", "total_tangent_length", "deflection_angle"])
    words = {"radius": "radius", "transition_length": "transition length", "cant": "cant", "max_speed": "maximum speed",
             "total_tangent_length": "total tangent length", "deflection_angle": "deflection angle"}
    msgs = [user(f"What is the {words[k]} of curve no. {c['curve_no']} on the {line}?"),
            call("query", {"kind": "curve", "line": line, "curve_no": c["curve_no"]}), result("query", c),
            say(f"Curve no. {c['curve_no']} ({line}, sheet {c['sheet_id']}, {c['hand']}): {words[k]} {c[k]}.")]
    return conv("agent_curve", c["sheet_id"], msgs, False)


def which_sheet(S, line, ch):
    sids = S.sheet_for(line, ch)
    if not sids:
        return None
    msgs = [user(f"Which sheet covers CH {chs(ch)} on the {line}?"),
            call("query", {"kind": "sheet_for_chainage", "line": line, "chainage": ch}), result("query", sids),
            say(f"CH {chs(ch)} on the {line} is on sheet{'s' if len(sids) > 1 else ''} {', '.join(sids)}."
                + (" (it is at the match line, shown on both)." if len(sids) > 1 else ""))]
    return conv("agent_which_sheet", sids[0], msgs, False)


def tbm(S, t):
    msgs = [user(f"Where is {t['tbm_id']} and what is its MSL?"),
            call("query", {"kind": "tbm", "tbm_id": t["tbm_id"]}), result("query", t),
            say(f"{t['tbm_id']} (sheet {t['sheet_id']}): MSL {f3(t['msl_m'])} m"
                + (f", at CH {f3(t['chainage_m'])}" if t["chainage_m"] is not None else "")
                + f", easting {f3(t['easting'])}, northing {f3(t['northing'])} - {t['description']}")]
    return conv("agent_tbm", t["sheet_id"], msgs, False)


GUESSES = {   # the model's own first guess for each term (labelled as a guess, then confirmed from a source)
    "TTP1": "a transition / tangent point at the start of a curve", "CTP1": "the point where the circular curve starts",
    "CTP2": "the point where the circular curve ends", "TTP2": "a transition / tangent point at the end of a curve",
    "CCL": "the circular curve length", "TRL": "the transition length", "BVC": "the beginning of a vertical curve",
    "EVC": "the end of a vertical curve", "ERL": "an existing rail level", "RTL": "a rail top level", "IP": "the intersection point of two tangents",
    "GP": "a grade (gradient change) point", "SL": "a sleeper or soffit level", "R": "the radius of a curve", "L": "a length",
    "FL": "the formation level", "RL": "the rail level", "TL": "the tangent length", "G": "a gradient", "CL": "a curve length or centre line",
    "PVI": "the point of vertical intersection", "TP1": "the first tangent point of a curve", "TP2": "the second tangent point of a curve",
    "J1": "a junction point", "J2": "a second junction point", "BL": "the bed level", "ST": "straight to transition (curve starts)",
    "TC": "transition to circular curve", "CT": "circular curve to transition", "TS": "transition to straight (curve ends)"}


def term(S, sid, k):
    meaning = S.sheet_abbr[sid][k]          # this sheet's own table defines it
    guess = GUESSES.get(k, "a level or length used on L-sections")
    msgs = [user(f"What does {k} mean on these drawings?"),
            call("query", {"kind": "abbreviation", "term": k, "sheet_id": sid},
                 f"My guess: {k} is {guess}. Confirm it from the sheet's abbreviations table first."),
            result("query", {"term": k, "meaning": meaning, "source": f"abbreviations table, sheet {sid}"}),
            say(f"{k} = {meaning} (from the abbreviations table on sheet {sid}). My first guess was {guess}; the sheet's own table is what counts.")]
    return conv("agent_term", sid, msgs, True)


def term_web(S, sid, k):
    meaning = S.abbr[k]                      # this sheet's table does not define it; another drawing set's table does
    guess = GUESSES.get(k, "a level or length used on L-sections")
    msgs = [user(f"What is {k}?"),
            call("query", {"kind": "abbreviation", "term": k, "sheet_id": sid}, f"My guess: {k} is {guess}. Check the sheet first."),
            result("query", {"term": k, "meaning": None, "source": f"not in the abbreviations table of sheet {sid}"}),
            call("search_library", {"query": f"{k} abbreviation railway L-section"}), result("search_library", {"results": []}),
            call("web_search", {"term": k}), result("web_search", {"term": k, "results": [{"snippet": f"{k}: {meaning}"}]}),
            say(f"{k} most likely means {meaning}. This is from the internet - it is not defined on sheet {sid} or in the library - "
                f"so please confirm it against your drawing set's abbreviations. (My own first guess was {guess}.)")]
    return conv("agent_term_web", sid, msgs, True)


def out_of_scope(S, line, b):
    lv = b["levels"]
    msgs = [user(f"Should we raise the formation level at {bname(b)} on the {line}?"),
            call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]}), result("query", b),
            say("That is a design decision for the engineers, so I won't recommend one. What the drawing gives for it: "
                + ", ".join(f"{LEVEL_NAMES[k]} {f3(v)}" for k, v in lv.items() if k in LEVEL_NAMES)
                + f" ({src(b, line)}). I can check it against MIN FL REQ. or work out the free board if that helps.",
                "Raising the FL is an engineering decision; I can only report the levels and checks.")]
    return conv("agent_out_of_scope", b["sheet_id"], msgs, True)



# ---------------------------------------------------------------- uploads, versions, page types, upload quality, look, reports

def page_summary(S, a, status="new"):
    info = a["sheet_info"]
    bd = a.get("bands") if isinstance(a.get("bands"), dict) else {}
    rev = ((info.get("issue_record") or [{}])[-1] or {}).get("rev") or "R0"
    return {"page": a["page_index"] + 1, "type": "L-section sheet", "sheet_id": a["sheet_id"], "sheet_no": info.get("sheet_no"),
            "line": S.sheet_line.get(a["sheet_id"]), "chainage_from": info.get("chainage_from"), "chainage_to": info.get("chainage_to"),
            "revision": rev, "read_by": "PDF text layer (exact)", "layout": "known",
            "read": {"bridges": len(a["bridges"]), "level_blocks": sum(1 for b in a["bridges"] if b.get("lsection_levels")),
                     "curves": len(a["curves"]), "band_columns": len(bd.get("columns", [])), "tbms": len(a["tbm_benchmarks"])},
            "findings": a.get("findings") or [], "status": status}


def continuity(pages):
    ls = [p for p in pages if p["type"] == "L-section sheet" and p.get("chainage_from") and p.get("chainage_to")]
    ls.sort(key=lambda p: D_ch(p["chainage_from"]))
    issues = []
    for a, b in zip(ls, ls[1:]):
        gap = D_ch(b["chainage_from"]) - D_ch(a["chainage_to"])
        if gap > 1:
            issues.append(f"gap of {f3(round(gap, 1))} m between sheet {a['sheet_id']} (ends {a['chainage_to']}) and {b['sheet_id']} (starts {b['chainage_from']})")
        elif gap < -1:
            issues.append(f"overlap of {f3(round(-gap, 1))} m between sheet {a['sheet_id']} and {b['sheet_id']}")
    return {"checked_pairs": max(0, len(ls) - 1), "issues": issues}


def pdf_groups(S):
    g = defaultdict(list)
    for sid, a in S.sheet_ann.items():
        g[a["source_pdf"]].append(a)
    return {k: sorted(v, key=lambda a: a["page_index"]) for k, v in g.items()}


def upload_reply(res):
    pages = res["pages"]
    ls = [p for p in pages if p["type"] == "L-section sheet"]
    other = [p for p in pages if p["type"] != "L-section sheet"]
    lines_ = sorted({p["line"] for p in ls if p.get("line")})
    head = f"{res['file']}: {len(pages)} page{'s' if len(pages) != 1 else ''}, {len(ls)} L-section sheet{'s' if len(ls) != 1 else ''}"
    if ls:
        head += f" ({ls[0]['sheet_id']} to {ls[-1]['sheet_id']}), {', '.join(lines_)}, CH {ls[0]['chainage_from']} to {ls[-1]['chainage_to']}"
        head += ". Read exactly from the PDF's text layer." if all(p["read_by"].startswith("PDF") for p in ls) else "."
    out = [head]
    for p in other:
        out.append(f"Page {p['page']} is a {p['type']}: {p['note']}")
    tot = Counter()
    for p in ls:
        tot.update(p["read"])
    if ls:
        out.append(f"Read: {tot['bridges']} bridges ({tot['level_blocks']} with level blocks), {tot['curves']} curves, "
                   f"{tot['band_columns']} band columns, {tot['tbms']} TBMs.")
    c = res["continuity"]
    out.append("Continuity: " + ("consecutive sheets join up." if not c["issues"] else "; ".join(c["issues"]) + "."))
    fnd = [f"{p['sheet_id']}: {f}" for p in ls for f in p["findings"]]
    if fnd:
        out.append("Findings on the drawings (worth checking with the engineers): " + "; ".join(fnd[:6])
                   + (f"; and {len(fnd) - 6} more" if len(fnd) > 6 else "") + ".")
    return " ".join(out)


def upload_pdf(S, pdf, anns, renumber=False):
    pages = sorted((page_summary(S, a) for a in anns), key=lambda p: p["page"])
    if renumber:                                   # a file holding only some of the sheets: pages 1..n
        for i, p in enumerate(pages):
            p["page"] = i + 1
    res = {"file": pdf, "pages": pages, "continuity": continuity(pages)}
    q = rng.choice([f"I have uploaded {pdf}. What is in it?", f"Read {pdf} and tell me what it contains.",
                    f"Here is {pdf}. Summarise it."])
    msgs = [user(q), call("read_sheet", {"file": pdf}, "Read the file into the store, then summarise the pages, continuity and findings."),
            result("read_sheet", res), say(upload_reply(res), "Summarise per page type, the chainage covered, continuity and findings.")]
    return msgs, res


def upload_then_ask(S, pdf, anns, renumber=False):
    msgs, res = upload_pdf(S, pdf, anns, renumber)
    cands = [b for a in anns if a["sheet_id"] in S.sheet_line
             for b in S.bridges[S.sheet_line[a["sheet_id"]]].values() if b["sheet_id"] == a["sheet_id"] and b["levels"]]
    if not cands:
        return None
    b = rng.choice(cands)
    line = S.sheet_line[b["sheet_id"]]
    k = rng.choice([x for x in LEVEL_ASK if b["levels"].get(x) is not None])
    msgs += [user(f"What is the {rng.choice(LEVEL_ASK[k])} at bridge {b['bridge_id']}?"),
             call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]},
                  f"The file just uploaded is the {line}, so bridge {b['bridge_id']} means the {line} bridge."),
             result("query", b),
             say(f"The {LEVEL_NAMES[k]} at bridge {b['bridge_id']} is {f3(b['levels'][k])} m ({src(b, line)}, from {pdf}).")]
    return conv("agent_upload_ask", b["sheet_id"], msgs, True)


def reupload(S, pdf, anns):
    pages = [page_summary(S, a, "already stored (same file)") for a in sorted(anns, key=lambda a: a["page_index"])]
    for i, pg in enumerate(pages):
        pg["page"] = i + 1
    res = {"file": pdf, "pages": pages, "continuity": continuity(pages)}
    msgs = [user(f"I have uploaded {pdf} again."), call("read_sheet", {"file": pdf}), result("read_sheet", res),
            say(f"{pdf} is identical to the file already stored (same content), so nothing new was added: the earlier reading of "
                f"its {len(pages)} sheet(s) is reused. If the drawings were revised, upload the revised file and it will be kept as a new version.")]
    return conv("agent_reupload", anns[0]["sheet_id"], msgs, False)


def new_version(S, a):
    """A revised sheet (hypothetical R1 with one changed FL, a labelled scenario) is stored as a new version; R0 is kept."""
    line = S.sheet_line[a["sheet_id"]]
    cands = [b for b in S.bridges[line].values() if b["sheet_id"] == a["sheet_id"] and b["levels"]
             and b["levels"].get("proposed_formation_level") is not None]
    if not cands:
        return []
    b = rng.choice(cands)
    old = b["levels"]["proposed_formation_level"]
    new = round(old + rng.choice([0.05, 0.1, -0.03, 0.12]), 3)
    fname = f"{a['sheet_id']}_R1.pdf"
    page = page_summary(S, a, "stored as a new version (R1); R0 kept")
    page["revision"], page["page"] = "R1", 1
    res = {"file": fname, "pages": [page], "continuity": {"checked_pairs": 0, "issues": []}}
    b1 = json.loads(json.dumps(b))
    b1["levels"]["proposed_formation_level"] = new
    b1["version"] = "R1 (latest)"
    b1["older_versions"] = [{"revision": "R0", "file": a["source_pdf"], "proposed_formation_level": old}]
    msgs = [user(f"I have uploaded {fname}, a revised sheet."), call("read_sheet", {"file": fname}), result("read_sheet", res),
            say(f"Stored sheet {a['sheet_id']} revision R1 as a new version; revision R0 (from {a['source_pdf']}) is kept, nothing was replaced. "
                f"Answers use R1 from now on, and I will say when R0 differs."),
            user(f"What is the FL at bridge {b['bridge_id']} now?"),
            call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]}), result("query", b1),
            say(f"The FL at bridge {b['bridge_id']} is {f3(new)} m in the latest version R1 (sheet {a['sheet_id']}). "
                f"Revision R0 had {f3(old)} m, so it changed by {f3(round(new - old, 3))} m.")]
    vers = [{"revision": "R0", "file": a["source_pdf"], "stored": "2026-10-01"},
            {"revision": "R1", "file": fname, "stored": "2026-10-05", "latest": True}]
    msgs2 = [user(f"Which versions of sheet {a['sheet_info'].get('sheet_no')} on the {line} do we have?"),
             call("query", {"kind": "versions", "line": line, "sheet_no": a["sheet_info"].get("sheet_no")}), result("query", vers),
             say(f"Two versions of sheet {a['sheet_id']}: R0 (from {a['source_pdf']}) and R1 (from {fname}, the latest, used by default). "
                 f"Ask for R0 to see the earlier values.")]
    return [conv("agent_new_version", a["sheet_id"], msgs, False), conv("agent_versions", a["sheet_id"], msgs2, False)]


def mixed_pdf(S, pdf, anns):
    """A PDF that also holds a cover page and a drawing that is not an L-section."""
    shifted = []
    for i, a in enumerate(sorted(anns, key=lambda a: a["page_index"])):
        p = page_summary(S, a)
        p["page"] = i + 2
        shifted.append(p)
    pages = ([{"page": 1, "type": "cover / index page", "sheet_id": None,
               "note": "the title page listing the sheets; it is not an L-section and was not read into the store."}] + shifted +
             [{"page": len(anns) + 2, "type": "drawing of another kind (it looks like a General Arrangement Drawing of a bridge)", "sheet_id": None,
               "note": "not a Plan & L-Section sheet, so no L-section values were taken from it."}])
    fname = pdf.replace(".pdf", "_with_cover.pdf")
    res = {"file": fname, "pages": pages, "continuity": continuity(pages)}
    msgs = [user(f"Please read {fname}."), call("read_sheet", {"file": fname}), result("read_sheet", res),
            say(upload_reply(res), "Two pages are not L-section sheets: report them instead of forcing L-section fields onto them.")]
    return conv("agent_page_types", anns[0]["sheet_id"], msgs, True)


def image_upload(S, a, quality):
    """One sheet uploaded as an image: low DPI, or a layout that differs from the trained sheets (labelled scenarios)."""
    line = S.sheet_line[a["sheet_id"]]
    cands = [b for b in S.bridges[line].values() if b["sheet_id"] == a["sheet_id"] and b["levels"]]
    if not cands:
        return None
    b = rng.choice(cands)
    page = page_summary(S, a)
    page["read_by"], page["page"] = "image + model", 1
    fname = f"sheet_{a['sheet_info'].get('sheet_no')}_{'75dpi' if quality == 'low_dpi' else 'scan'}.png"
    if quality == "low_dpi":
        page["dpi"] = 75
        page["warnings"] = ["the image is about 75 dpi: text is about 10 px tall, so digits may be misread (150-300 dpi recommended)"]
        caveat = "It was read from a 75 dpi image, so check this value on the drawing."
    else:
        page["dpi"] = 150
        page["layout"] = "similar"
        page["layout_notes"] = ["the band rows are in a different order from the trained sheets; they were re-stacked into the trained order before reading"]
        caveat = "This sheet's layout differs from the trained sheets (band rows re-ordered), so check this value on the drawing."
    k = rng.choice([x for x in LEVEL_ASK if b["levels"].get(x) is not None])
    res = {"file": fname, "pages": [page], "continuity": {"checked_pairs": 0, "issues": []}}
    msgs = [user(f"I have uploaded {fname}. What is the {rng.choice(LEVEL_ASK[k])} at bridge {b['bridge_id']}?"),
            call("read_sheet", {"file": fname}), result("read_sheet", res),
            call("query", {"kind": "bridge", "line": line, "bridge_id": b["bridge_id"]}), result("query", b),
            say(f"The {LEVEL_NAMES[k]} at bridge {b['bridge_id']} is {f3(b['levels'][k])} m ({src(b, line)}). {caveat}",
                "The reading carries a warning, so the answer must carry it too.")]
    return conv("agent_upload_quality", a["sheet_id"], msgs, True)


def unknown_layout(S, a):
    if not a["sheet_info"].get("chainage_from"):
        return None
    page = page_summary(S, a)
    page.update({"page": 1, "read_by": "image + model", "dpi": 150, "layout": "unknown",
                 "layout_notes": ["band rows not found: track_distance", "band rows the model was not trained on (not read): BANK HEIGHT (m)"]})
    page["read"]["band_columns"] = 0
    fname = f"sheet_{a['sheet_info'].get('sheet_no')}_other_layout.png"
    res = {"file": fname, "pages": [page], "continuity": {"checked_pairs": 0, "issues": []}}
    ch = round(D_ch(a["sheet_info"]["chainage_from"]) + rng.uniform(300, 3000), 1)
    msgs = [user(f"I uploaded {fname}. What is the track distance at CH {f3(ch)}?"),
            call("read_sheet", {"file": fname}), result("read_sheet", res),
            say("I cannot give the track distance from this sheet: its layout is not one I was trained on, and the band row for "
                "track distance was not found, so no band values were read rather than guessed. It also has a row labelled "
                "'BANK HEIGHT (m)' that I did not read. I can look at the band table directly (that reading would be marked as a "
                "new-layout value to check), or you can upload a sheet in the usual layout.",
                "Layout unknown and the row is missing: do not guess; say what was skipped and offer look.")]
    return conv("agent_layout_unknown", a["sheet_id"], msgs, True)


def look_panel(S, a):
    sid = a["sheet_id"]
    if rng.random() < 0.5 and a["legend"]:
        what, obj = "legend", {"text": a["legend"]}
        reply = f"The legend of sheet {sid} shows: " + "; ".join(a["legend"]) + "."
    else:
        rec = a["sheet_info"].get("issue_record") or []
        if not rec:
            return None
        what, obj = "issue record", {"rows": rec}
        reply = f"Issue record of sheet {sid}: " + "; ".join(
            f"{r.get('rev')} dated {r.get('date')}, prepared by {r.get('prepared_by')}, checked by {r.get('checked_by')}, "
            f"approved by {r.get('approved_by')}" for r in rec) + "."
    msgs = [user(f"What does the {what} on sheet {sid} say?"),
            call("look", {"sheet_id": sid, "area": f"right-hand panel, {what}", "question": f"Read the {what} as printed."}),
            result("look", obj), say(reply)]
    return conv("agent_look", sid, msgs, False)


def unreadable(S, a):
    line = S.sheet_line[a["sheet_id"]]
    cols = [c for c in (a.get("bands") or {}).get("columns", []) if c["checks_ok"]] if isinstance(a.get("bands"), dict) else []
    if len(cols) < 3:
        return None
    c = rng.choice(cols)
    ch = c["chainage"]
    vals = {k: v for k, v in c["values"].items() if k not in ("chainage", "ground_level")}
    res = [{"chainage": ch, "sheet_id": a["sheet_id"], "x1": int(ch), "x2": int(ch), "column_x1": vals, "y": vals,
            "unreadable": ["ground_level"], "note": "read from a 75 dpi image; the ground-level digits could not be read"}]
    msgs = [user(f"What is the ground level at CH {chs(ch)} on the {line}?"),
            call("band_at", {"line": line, "chainages": [ch]}), result("band_at", res),
            say(f"The ground level at CH {chs(ch)} could not be read: its digits are unreadable on the 75 dpi image of sheet {a['sheet_id']}, "
                f"and I will not guess a value. The other rows of that column were read (for example proposed FL {f3(vals.get('prop_fl'))}). "
                f"A 150-300 dpi image or the PDF would give it exactly.",
                "The value is unreadable: say so, never fill it in.")]
    return conv("agent_unreadable", a["sheet_id"], msgs, True)


def report(S, a):
    sid, line = a["sheet_id"], S.sheet_line[a["sheet_id"]]
    bs = sorted((b for b in S.bridges[line].values() if b["sheet_id"] == sid), key=lambda b: b["chainage_m"])
    if len(bs) < 3:
        return None
    rows = ["| Bridge | Chainage | Existing | Crosses | Proposal | FL | MIN FL REQ. | Check |", "|---|---|---|---|---|---|---|---|"]
    flags = []
    for b in bs:
        lv = b["levels"] or {}
        fl, mn = lv.get("proposed_formation_level"), lv.get("min_formation_level_required")
        if fl is not None and mn is not None and fl < mn:
            flags.append(f"FLAG: For bridge {b['bridge_id']} min. FL = {f3(mn)}, and FL = {f3(fl)}.")
        chk = ("FLAG" if fl is not None and mn is not None and fl < mn else "ok" if fl is not None and mn is not None else "no level block")
        rows.append(f"| {b['bridge_id']} | {f3(b['chainage_m'])} | {' '.join(x for x in (b['existing_type'], b['existing_span']) if x) or '-'} | "
                    f"{b['crossing'] or '-'} | {b['proposal'] or '-'} | {f3(fl) if fl is not None else '-'} | {f3(mn) if mn is not None else '-'} | {chk} |")
    msgs = [user(rng.choice([f"Give me the bridge schedule for sheet {sid}.", f"Make a table of all bridges on sheet {sid}."])),
            call("query", {"kind": "bridges_on_sheet", "sheet_id": sid}), result("query", bs),
            say(f"Bridges on sheet {sid} ({line}):\n\n" + "\n".join(rows) + ("\n\n" + "\n".join(flags) if flags else ""))]
    return conv("agent_report", sid, msgs, False)

# ---------------------------------------------------------------- main

def main():
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(V.ANN.glob("*.json"))]
    split_of = {a["sheet_id"]: "test" if (a["source_pdf"] == V.TEST_PDF or a["sheet_id"] in V.TEST) else
                "val" if a["sheet_id"] in V.VAL else "train" for a in anns}
    S = Store(anns)
    convs = []
    for line, bridges in S.bridges.items():
        bs = sorted(bridges.values(), key=lambda b: b["chainage_m"])
        with_lv = [b for b in bs if b["levels"]]
        for b in bs:
            convs.append(band_value(S, line, b))
            if b["levels"]:
                convs.append(lookup_level(S, line, b))
                lv = b["levels"]
                if lv.get("proposed_formation_level") is not None and lv.get("min_formation_level_required") is not None:
                    convs.append(min_fl(S, line, b))
            if rng.random() < 0.15:
                c = not_found(S, line, b)
                if c:
                    convs.append(c)
        for i in range(0, len(bs) - 4, 5):
            grp = bs[i:i + rng.choice([3, 4, 5])]
            if grp[-1]["chainage_m"] - grp[0]["chainage_m"] < 6000 and len({b["sheet_id"] for b in grp}) <= 2:
                convs.append(range_fill(S, line, grp))
                if all(b["levels"] for b in grp):
                    convs.append(range_flags(S, line, grp))
        for b, b2 in zip(with_lv[::4], with_lv[1::4]):
            convs.append(follow_up(S, line, b, b2))
        for c in list(S.curves[line].values())[::2]:
            convs.append(curve(S, line, c))
        for ch in rng.sample(sorted(S.cols[line]), min(120, len(S.cols[line]))):
            c = which_sheet(S, line, ch + rng.choice([0, 7.5, 13.2]))
            if c:
                convs.append(c)
        bad = [ch for ch, (c, sid) in S.cols[line].items() if c.get("rail_level_ok") is False]
        good = rng.sample(sorted(ch for ch, (c, sid) in S.cols[line].items() if c.get("rail_level_ok")), 60)
        for ch in bad[::3] + good:
            c = rail_check(S, line, ch, S.cols[line][ch][1])
            if c:
                convs.append(c)
        for b in rng.sample(with_lv, min(40, len(with_lv))):
            convs.append(out_of_scope(S, line, b))
    for bid in sorted(set(S.bridges["3rd line"]) & set(S.bridges["4th line"]))[::2]:
        convs.append(ambiguous_line(S, bid))
    for t in list(S.tbms.values())[::3]:
        convs.append(tbm(S, t))
    sids = sorted(split_of)
    for k in S.abbr:
        having = [sid for sid in sids if k in S.sheet_abbr.get(sid, {})]
        lacking = [sid for sid in sids if sid in S.sheet_abbr and k not in S.sheet_abbr[sid]]
        for sid in rng.sample(having, min(4, len(having))):
            convs.append(term(S, sid, k))
        for sid in rng.sample(lacking, min(2, len(lacking))):
            convs.append(term_web(S, sid, k))
    # uploads: a whole PDF goes to the strictest split of its sheets, so no test sheet appears in training
    rank = {"train": 0, "val": 1, "test": 2}
    for pdf, anns_p in pdf_groups(S).items():
        sp = max((split_of[a["sheet_id"]] for a in anns_p), key=rank.get)
        for _ in range(2):
            msgs, _r = upload_pdf(S, pdf, anns_p)
            convs.append({**conv("agent_upload", anns_p[0]["sheet_id"], msgs, True), "force_split": sp})
        for _ in range(3):
            c = upload_then_ask(S, pdf, anns_p)
            if c:
                convs.append({**c, "force_split": sp})
        convs.append({**reupload(S, pdf, anns_p), "force_split": sp})
        convs.append({**mixed_pdf(S, pdf, anns_p), "force_split": sp})
        # training versions from the PDF's training sheets only (a file holding some of the sheets)
        tr = [a for a in anns_p if split_of[a["sheet_id"]] == "train"]
        if sp != "train" and len(tr) >= 2:
            for _ in range(3):
                lo = rng.randrange(0, len(tr) - 1)
                part = tr[lo:lo + rng.randint(2, min(6, len(tr) - lo))]
                name = f"{pdf[:-4]}_sheets_{part[0]['sheet_info'].get('sheet_no')}-{part[-1]['sheet_info'].get('sheet_no')}.pdf"
                msgs, _r = upload_pdf(S, name, part, renumber=True)
                convs.append({**conv("agent_upload", part[0]["sheet_id"], msgs, True), "force_split": "train"})
                c = upload_then_ask(S, name, part, renumber=True)
                if c:
                    convs.append({**c, "force_split": "train"})
            convs.append({**reupload(S, name, part), "force_split": "train"})
            convs.append({**mixed_pdf(S, name, part), "force_split": "train"})
    for sid, a in sorted(S.sheet_ann.items()):
        if sid not in S.sheet_line:
            continue
        if rng.random() < 0.3:
            convs += new_version(S, a)
        for q in ("low_dpi", "similar"):
            if rng.random() < 0.3:
                c = image_upload(S, a, q)
                if c:
                    convs.append(c)
        for fn, prob in ((unknown_layout, 0.25), (look_panel, 0.6), (unreadable, 0.45), (report, 0.7)):
            if rng.random() < prob:
                c = fn(S, a)
                if c:
                    convs.append(c)
    out = defaultdict(list)
    for i, c in enumerate(convs):
        c["id"] = f"agent_{i:05d}"
        c["split"] = c.pop("force_split", None) or split_of.get(c["sheet"], "train")
        out[c["split"]].append(c)
    for split in ("train", "val", "test"):
        with open(DS / f"agent_{split}.jsonl", "w", encoding="utf-8") as f:
            for c in out[split]:
                f.write(json.dumps({k: c[k] for k in ("id", "split", "sheet", "task", "thinking", "messages")}, ensure_ascii=False) + "\n")
    print({s: (len(out[s]), dict(Counter(c["task"] for c in out[s]))) for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
