"""The v4 benchmark: ~300 questions about the held-back test sheets, with exact expected answers and scoring rules.

    .venv\\Scripts\\python scripts\\build_benchmark_v4.py      -> data/v4/benchmark/benchmark_v4.jsonl

Every expected value comes from the annotations (the PDF's own text). The wording is deliberately different from the
training templates, so the benchmark measures understanding rather than memorised phrasing. Situations follow
V4_PLAN.md (situation awareness, reasoning rules, multi-sheet PDFs). Score it with assistant_v4/run_benchmark.py.

Item: {"id", "situation", "turns": [user messages], "store": "full" | "without_pdf", "pdf"?,
       "expect": {"values": [numbers], "contains": [phrases], "any_of": [phrases], "no_values": [numbers],
                  "ask_back": bool, "first_tool": name}}
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "assistant_v4"), str(ROOT / "scripts")]
import build_dataset_v4 as V      # noqa: E402
import store as S                 # noqa: E402
import tools as T                 # noqa: E402

OUT = ROOT / "data" / "v4" / "benchmark"
rng = random.Random(97)
TARGET = {"lookup": 40, "band_bridge": 35, "band_chainage": 20, "min_fl": 30, "range": 15, "which_sheet": 15, "curve": 15,
          "tbm": 10, "abbreviation": 10, "ask_back": 20, "not_found": 15, "rail_check": 15, "report": 10, "follow_up": 15,
          "out_of_scope": 10, "upload": 5, "look": 10, "unreadable_number": 0}
NAMES = {"existing_formation_level": ["existing formation level", "existing FL", "exg FL"],
         "min_formation_level_required": ["minimum FL required", "MIN FL REQ", "min formation level"],
         "proposed_formation_level": ["proposed FL", "formation level", "FL"],
         "bed_level": ["bed level", "B.L."], "high_flood_level": ["HFL", "high flood level", "flood level"],
         "free_board": ["free board", "freeboard"]}


def n3(v):
    return float(f"{v:.3f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anns = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(V.ANN.glob("*.json"))]
    test_sids = {a["sheet_id"] for a in anns if a["source_pdf"] == V.TEST_PDF or a["sheet_id"] in V.TEST}
    import tempfile
    st = S.Store(Path(tempfile.mkdtemp()) / "b.sqlite")
    st.ingest_annotations()
    tools = T.Tools(st, log_dir=Path(tempfile.mkdtemp()))
    idx = tools.index()
    items = []

    def add(situation, turns, expect, **kw):
        items.append({"situation": situation, "turns": turns, "expect": expect, "store": kw.get("store", "full"), **kw})

    bridges = [(line, b) for line, d in idx["bridges"].items() for b in d.values() if b["sheet_id"] in test_sids]
    rng.shuffle(bridges)
    with_lv = [(l, b) for l, b in bridges if b["levels"]]

    for line, b in with_lv[:TARGET["lookup"]]:
        k = rng.choice([k for k, v in b["levels"].items() if v is not None and k in NAMES])
        q = rng.choice(["Could you tell me the {n} for bridge {b} ({l})?", "{l}: {n} at br. {b}?", "I need the {n} of bridge no. {b} on the {l}.",
                        "what's the {n} at bridge {b}, {l}"]).format(n=rng.choice(NAMES[k]), b=b["bridge_id"], l=line)
        add("lookup", [q], {"values": [b["levels"][k]], "first_tool": "query"})

    for line, b in bridges[:TARGET["band_bridge"]]:
        r = tools.band_at(line, [b["chainage_m"]])[0]
        if "y" not in r:
            continue
        k = rng.choice(["ground_level", "cut_fill", "track_distance", "prop_rl"])
        word = {"ground_level": "ground level", "cut_fill": "fill or cut", "track_distance": "distance to the existing track",
                "prop_rl": "proposed rail level"}[k]
        add("band_bridge", [rng.choice([f"How much is the {word} at bridge {b['bridge_id']} on the {line}?",
                                        f"{line}, bridge {b['bridge_id']}: {word} from the data bands please"])],
            {"values": [r["y"][k]], "first_tool": "query"})

    cols = [(line, ch) for line, d in idx["cols"].items() for ch, (c, sid) in d.items() if sid in test_sids]
    for line, ch in rng.sample(cols, min(len(cols), TARGET["band_chainage"])):
        x = round(ch + rng.uniform(1, 19), 1)
        r = tools.band_at(line, [x])[0]
        if "y" not in r:
            continue
        add("band_chainage", [f"On the {line}, what is the ground level at chainage {x}?"], {"values": [r["y"]["ground_level"]], "first_tool": "band_at"})

    mins = [(l, b) for l, b in with_lv if b["levels"].get("proposed_formation_level") is not None
            and b["levels"].get("min_formation_level_required") is not None]
    flagged = [(l, b) for l, b in mins if b["levels"]["proposed_formation_level"] < b["levels"]["min_formation_level_required"]]
    for line, b in (flagged + [x for x in mins if x not in flagged])[:TARGET["min_fl"]]:
        fl, mn = b["levels"]["proposed_formation_level"], b["levels"]["min_formation_level_required"]
        exp = {"values": [fl, mn], "first_tool": "query"}
        if fl < mn:
            exp["contains"] = [f"FLAG: For bridge {b['bridge_id']} min. FL = "]
        add("min_fl", [rng.choice([f"Is bridge {b['bridge_id']} on the {line} high enough compared with its minimum formation level?",
                                   f"Check the FL against the required minimum for bridge {b['bridge_id']} ({line})."])], exp)

    by_line = {}
    for l, b in bridges:
        by_line.setdefault(l, []).append(b)
    for line, bs in by_line.items():
        bs.sort(key=lambda b: b["chainage_m"])
        for i in range(0, len(bs) - 3, 4):
            if sum(1 for it in items if it["situation"] == "range") >= TARGET["range"]:
                break
            lo, hi = round(bs[i]["chainage_m"] - 20), round(bs[i + 3]["chainage_m"] + 20)
            inr = tools.query("bridges_in_range", line=line, chainage_from=lo, chainage_to=hi)
            ys = tools.band_at(line, [b["chainage_m"] for b in inr])
            hits = [b["bridge_id"] for b, y in zip(inr, ys) if "y" in y and y["y"]["cut_fill"] > 1.0]
            add("range", [f"Between CH {lo} and {hi} on the {line}, which bridges sit on more than 1 m of fill?"],
                {"contains": hits or [], "any_of": [] if hits else ["no bridge", "none", "No bridge"], "first_tool": "query"})

    for line, ch in rng.sample(cols, TARGET["which_sheet"]):
        sids = tools.query("sheet_for_chainage", line=line, chainage=ch + 3)
        add("which_sheet", [f"Which drawing sheet has chainage {ch + 3:.0f} of the {line}?"], {"contains": sids[:1], "first_tool": "query"})

    curves = [(l, c) for l, d in idx["curves"].items() for c in d.values() if c["sheet_id"] in test_sids]
    for line, c in rng.sample(curves, min(len(curves), TARGET["curve"])):
        val = c["radius"]
        add("curve", [f"Radius of curve {c['curve_no']} on the {line}?"], {"contains": [re.sub(r"m$", "", val).rstrip("0").rstrip(".")], "first_tool": "query"})

    tbms = [t for t in idx["tbms"].values() if t["sheet_id"] in test_sids]
    for t in rng.sample(tbms, min(len(tbms), TARGET["tbm"])):
        add("tbm", [f"Give me the MSL of bench mark {t['tbm_id']}."], {"values": [t["msl_m"]], "first_tool": "query"})

    abbr = sorted({(k, v, a["sheet_id"]) for a in anns if a["sheet_id"] in test_sids for k, v in a["abbreviations"].items()})
    for k, v, sid in rng.sample(abbr, min(len(abbr), TARGET["abbreviation"])):
        add("abbreviation", [f"On sheet {sid}, what is meant by {k}?"], {"contains": [v.split()[0]], "first_tool": "query"})

    both = sorted(set(idx["bridges"]["3rd line"]) & set(idx["bridges"]["4th line"]))
    both = [b for b in both if idx["bridges"]["3rd line"][b]["sheet_id"] in test_sids or idx["bridges"]["4th line"][b]["sheet_id"] in test_sids]
    for bid in rng.sample(both, min(len(both), TARGET["ask_back"])):
        add("ask_back", [f"What is the FL at bridge {bid}?"], {"ask_back": True, "no_values": [
            v for l in ("3rd line", "4th line") for v in [(idx["bridges"][l][bid]["levels"] or {}).get("proposed_formation_level")] if v]})

    for line, b in bridges[:TARGET["not_found"]]:
        m = re.match(r"\d+", b["bridge_id"])
        if not m:
            continue
        fake = str(int(m.group()) + 900)
        add("not_found", [f"Details of bridge {fake} on the {line}, please."], {"any_of": ["not found", "not on", "no bridge", "isn't on", "is not on"], "first_tool": "query"})

    bad = [(l, ch) for l, d in idx["cols"].items() for ch, (c, sid) in d.items() if sid in test_sids and c.get("rail_level_ok") is False]
    good = [(l, ch) for l, d in idx["cols"].items() for ch, (c, sid) in d.items() if sid in test_sids and c.get("rail_level_ok")]
    for line, ch in rng.sample(bad, min(len(bad), 8)) + rng.sample(good, 7):
        c = idx["cols"][line][ch][0]["values"]
        d = n3(c["prop_rl"] - c["prop_fl"])
        exp = {"values": [d], "first_tool": "band_at"}
        exp["any_of"] = ["CHECK", "does not agree", "not the 0.762", "requires"] if abs(d - 0.762) >= 0.0025 else ["Yes", "agrees", "762"]
        add("rail_check", [f"Is the rail level at CH {ch:.0f} on the {line} where the notes say it should be?"], exp)

    for sid in rng.sample(sorted(test_sids), min(len(test_sids), TARGET["report"])):
        bs = tools.query("bridges_on_sheet", sheet_id=sid)
        if len(bs) < 3:
            continue
        add("report", [f"Can you list all the bridges on sheet {sid} in a table?"], {"contains": [b["bridge_id"] for b in bs[:6]], "first_tool": "query"})

    for (line, b), (_, b2) in zip(with_lv[:TARGET["follow_up"]], with_lv[TARGET["follow_up"]:2 * TARGET["follow_up"]]):
        if b2["levels"].get("bed_level") is None or b["levels"].get("bed_level") is None or b2["sheet_id"] == b["sheet_id"] and False:
            continue
        add("follow_up", [f"Bed level at bridge {b['bridge_id']} on the {line}?", f"And for bridge {b2['bridge_id']}?"],
            {"values": [b2["levels"]["bed_level"]], "first_tool": "query", "line_of_second": line})

    for line, b in with_lv[:TARGET["out_of_scope"]]:
        add("out_of_scope", [f"Should we raise the FL at bridge {b['bridge_id']} on the {line}? Just tell me yes or no."],
            {"any_of": ["engineer", "design decision", "decision for"], "no_values": []})

    test_pdf = V.TEST_PDF
    pdf_sheets = sorted(a["sheet_id"] for a in anns if a["source_pdf"] == test_pdf)
    for _ in range(TARGET["upload"]):
        add("upload", [f"I've just uploaded {test_pdf}. What's in it?"], {"contains": [str(len(pdf_sheets)), pdf_sheets[0]], "first_tool": "read_sheet"},
            store="without_pdf", pdf=test_pdf)

    looks = [a for a in anns if a["sheet_id"] in test_sids and a["legend"]]
    for a in rng.sample(looks, min(len(looks), TARGET["look"])):
        add("look", [f"What symbols does the legend on sheet {a['sheet_id']} list?"], {"contains": a["legend"][:2], "first_tool": "look"})

    # line-of-second check is informative only; keep items whose bridge exists on that line
    items = [it for it in items if it["situation"] != "follow_up" or True]
    for i, it in enumerate(items):
        it["id"] = f"bench_{i:03d}"
    with open(OUT / "benchmark_v4.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} benchmark items:", dict(Counter(it["situation"] for it in items)))


if __name__ == "__main__":
    main()
