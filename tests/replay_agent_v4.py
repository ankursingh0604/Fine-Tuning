"""Replay every tool call of the v4 agent conversations through the real tools and compare the results.

    .venv\\Scripts\\python tests\\replay_agent_v4.py

Proves that what the model is trained to see from a tool is what the tools will really return. Scenario situations
(upload quality verdicts, the hypothetical R1 revision, unreadable values, unknown layouts, cover/GAD pages, web
results) are labelled scenarios, not real data, and are skipped. Whole-PDF uploads are re-read from the real PDF
into a fresh store and compared page by page.
"""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "assistant_v4"), str(ROOT / "scripts")]
import store as S          # noqa: E402
import tools as T          # noqa: E402

SCENARIO = {"agent_upload_quality", "agent_new_version", "agent_versions", "agent_unreadable", "agent_layout_unknown",
            "agent_page_types", "agent_term_web", "agent_reupload", "agent_gap_column"}


def main():
    tmp = Path(tempfile.mkdtemp())
    st = S.Store(tmp / "store.sqlite")
    print("ingested", st.ingest_annotations(), "sheets")
    tools = T.Tools(st, log_dir=tmp)
    ok, bad = Counter(), Counter()
    examples = {}
    for f in ("agent_train", "agent_val", "agent_test"):
        for line in open(ROOT / "data" / "v4" / "dataset" / f"{f}.jsonl", encoding="utf-8"):
            c = json.loads(line)
            if c["task"] in SCENARIO:
                continue
            msgs = c["messages"]
            for i, m in enumerate(msgs):
                if m["role"] != "assistant" or not m.get("tool_calls"):
                    continue
                tc = m["tool_calls"][0]["function"]
                name, args = tc["name"], tc["arguments"]
                if name == "read_sheet":
                    if "_sheets_" in args["file"] or not (ROOT / args["file"]).exists():
                        continue                       # partial files do not exist on disk
                    n_fresh = ok["_fresh"] = ok["_fresh"] + 1
                    fresh = T.Tools(S.Store(tmp / f"fresh_{n_fresh}.sqlite"), log_dir=tmp)       # a new store per upload
                    got = fresh.read_sheet(args["file"])
                else:
                    got = T.call(tools, name, args)
                want = json.loads(msgs[i + 1]["content"])
                got = json.loads(json.dumps(got, ensure_ascii=False))
                key = (c["task"], name)
                if got == want:
                    ok[key] += 1
                else:
                    bad[key] += 1
                    examples.setdefault(key, (c["id"], json.dumps(want)[:400], json.dumps(got)[:400]))
    ok.pop("_fresh", None)
    print(f"tool calls replayed: {sum(ok.values()) + sum(bad.values())}, identical: {sum(ok.values())}, different: {sum(bad.values())}")
    ok.pop("_fresh", None)
    for key in sorted(set(ok) | set(bad)):
        print(f"  {key[0]:<24} {key[1]:<15} identical {ok[key]:>5}  different {bad[key]:>4}")
    for key, (cid, w, g) in examples.items():
        print(f"\nDIFFERENT {key} in {cid}\n  dataset: {w}\n  tools:   {g}")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
