"""Run the v4 benchmark through the assistant loop and score it per situation.

    python run_benchmark.py --adapter /path/to/adapter          # the trained model (needs a GPU)
    (on RunPod train_v4.py runs it by itself after training, with the model already loaded -> /workspace/v4_run/benchmark)
    python run_benchmark.py --selftest                           # checks the scoring with a perfect and a useless fake model

Scores (data/v4/benchmark/results_<name>.csv and summary): per item pass/fail with the reason; per situation the pass
rate; overall. Rules: every expected value appears in the final answer (to 3 decimals); required phrases appear; at
least one 'any_of' phrase appears; no forbidden value appears (no guessing); ask-back items must ask (a '?') and give
no value; the first tool called must be the expected one.
"""
import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "scripts")]
import agent as AG          # noqa: E402
import store as S           # noqa: E402
import tools as T           # noqa: E402

BENCH = ROOT / "data" / "v4" / "benchmark" / "benchmark_v4.jsonl"


def nums(text):
    return {round(float(x), 3) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")}


def score(item, answer, first_tool):
    e, reasons = item["expect"], []
    found = nums(answer)
    for v in e.get("values", []):
        if round(float(v), 3) not in found:
            reasons.append(f"missing value {v}")
    for p in e.get("contains", []):
        if str(p).strip().lower() not in (answer or "").lower():
            reasons.append(f"missing '{p}'")
    if e.get("any_of") and not any(p.lower() in (answer or "").lower() for p in e["any_of"]):
        reasons.append(f"none of {e['any_of']}")
    for v in e.get("no_values", []):
        if round(float(v), 3) in found:
            reasons.append(f"gave the value {v} it should not have")
    if e.get("ask_back") and "?" not in (answer or ""):
        reasons.append("did not ask back")
    if e.get("first_tool") and first_tool != e["first_tool"]:
        reasons.append(f"first tool {first_tool}, expected {e['first_tool']}")
    return not reasons, reasons


def run(generate, name, items=None, log=print, out=None):
    items = items or [json.loads(l) for l in open(BENCH, encoding="utf-8")]
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp())
    full = S.Store(tmp / "full.sqlite")
    full.ingest_annotations()
    without = {}
    rows, by = [], defaultdict(list)
    for i, it in enumerate(items):
        if it["store"] == "without_pdf":
            if it["pdf"] not in without:                 # one store without that PDF, reset after every upload question
                without[it["pdf"]] = S.Store(tmp / f"without_{len(without)}.sqlite")
                without[it["pdf"]].ingest_annotations(exclude_pdf=it["pdf"])
            st = without[it["pdf"]]
            st.forget_file(it["pdf"])
            tools = T.Tools(st, log_dir=tmp)
        else:
            tools = T.Tools(full, log_dir=tmp)
        bot = AG.Assistant(generate, tools)
        answer = ""
        for turn in it["turns"]:
            answer = bot.ask(turn)
        firsts = [m["tool_calls"][0]["function"]["name"] for m in bot.messages if m["role"] == "assistant" and m.get("tool_calls")]
        ok, why = score(it, answer, firsts[0] if firsts else None)
        by[it["situation"]].append(ok)
        rows.append({"id": it["id"], "situation": it["situation"], "pass": ok, "why": "; ".join(why), "answer": answer[:400]})
        if (i + 1) % 25 == 0 or i == 4:
            el = time.time() - t0
            log(f"  benchmark {i + 1}/{len(items)}: {sum(r['pass'] for r in rows)} passed so far, "
                f"{el / 60:.0f} min, about {el / (i + 1) * (len(items) - i - 1) / 60:.0f} min left")
    for st in [full, *without.values()]:
        st.db.close()
    shutil.rmtree(tmp, ignore_errors=True)
    out = Path(out or BENCH.parent)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"results_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Benchmark ({name}): {sum(r['pass'] for r in rows)}/{len(rows)} passed ({100 * sum(r['pass'] for r in rows) / len(rows):.1f} %)"]
    lines += [f"  {s:<15} {sum(v)}/{len(v)}" for s, v in sorted(by.items())]
    (out / f"summary_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    return rows


def oracle_for(items):
    """A fake model that answers perfectly (tests the scoring): first calls the expected tool, then states the expectation."""
    plan = {it["turns"][-1]: it for it in items}

    def gen(messages, tools, thinking):
        q = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        it = plan.get(q)
        if messages[-1]["role"] == "user" and it and it["expect"].get("first_tool"):
            name = it["expect"]["first_tool"]
            args = {"read_sheet": {"file": it.get("pdf", "x.pdf")}, "look": {"sheet_id": "x", "area": "legend", "question": "q"},
                    "band_at": {"line": "3rd line", "chainages": [0]}}.get(name, {"kind": "bridge", "line": "3rd line", "bridge_id": "0"})
            params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
            return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"
        if not it:
            return "ok"
        e = it["expect"]
        if e.get("ask_back"):
            return "Which line do you mean, the 3rd or the 4th?"
        parts = [f"{v:.3f}" for v in e.get("values", [])] + [str(p) for p in e.get("contains", [])] + (e.get("any_of") or [])[:1]
        return " ".join(parts) or "done"
    return gen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="folder for results_*.csv and summary_*.txt (default: next to the benchmark)")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(BENCH, encoding="utf-8")]
    if args.selftest:
        good = run(oracle_for(items), "selftest_perfect", items, out=args.out)
        bad = run(lambda m, t, th: "I don't know.", "selftest_useless", items, out=args.out)
        ok = all(r["pass"] for r in good) and not any(r["pass"] for r in bad if r["situation"] not in ("out_of_scope",))
        print("SELF-TEST", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    run(AG.TransformersModel(args.adapter), Path(args.adapter).name, items, out=args.out)
