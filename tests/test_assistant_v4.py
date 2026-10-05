"""Tests for the v4 assistant loop and the web-search leak gate (no model needed).

    .venv\\Scripts\\python tests\\test_assistant_v4.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "assistant_v4"), str(ROOT / "scripts")]
import agent as AG         # noqa: E402
import store as S          # noqa: E402
import tools as T          # noqa: E402

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def tool_call_text(name, args):
    params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
    return f"<think>\n\n</think>\n\n<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def main():
    tmp = Path(tempfile.mkdtemp())
    st = S.Store(tmp / "s.sqlite")
    st.ingest_annotations()
    tools = T.Tools(st, log_dir=tmp)

    # 1. tool-call parsing and argument types
    calls = AG.parse_calls(tool_call_text("query", {"kind": "bridge", "line": "3rd line", "bridge_id": "331"}) +
                           tool_call_text("band_at", {"line": "4th line", "chainages": [1256295.109]}))
    check(calls[0]["arguments"]["bridge_id"] == "331", "bridge id stays a string ('331')")
    check(calls[1]["arguments"]["chainages"] == [1256295.109], "chainages become a list of numbers")

    # 2. the loop with a scripted model: lookup, then a follow-up that needs the conversation
    script = iter([tool_call_text("query", {"kind": "bridge", "line": "4th line", "bridge_id": "594"}),
                   "<think>\n\n</think>\n\nThe FL at bridge 594 is 175.077 m.",
                   tool_call_text("band_at", {"line": "4th line", "chainages": [1256295.109]}),
                   "<think>\n\n</think>\n\nCut/fill y = 2.773 m."])
    seen = []

    def fake(messages, tool_defs, thinking):
        seen.append(len(messages))
        return next(script)
    bot = AG.Assistant(fake, tools)
    a1 = bot.ask("What is the FL at bridge 594 on the 4th line?")
    a2 = bot.ask("And the cut or fill there?")
    check(a1 == "The FL at bridge 594 is 175.077 m.", "final answer returned after the tool call")
    tool_msgs = [m for m in bot.messages if m["role"] == "tool"]
    check(json.loads(tool_msgs[0]["content"])["chainage_m"] == 1256295.109, "query tool result reaches the model")
    check(json.loads(tool_msgs[1]["content"])[0]["y"]["cut_fill"] == 2.773, "band_at interpolates (bridge 594 DN: 2.773)")
    check(seen[2] > seen[0], "the follow-up sees the earlier conversation (memory within a conversation)")

    # 3. calc safety
    check(tools.calc("245.541 - 245.5")["value"] == 0.041, "calc: arithmetic")
    check("error" in tools.calc("__import__('os').system('dir')"), "calc: anything but arithmetic is refused")

    # 4. leak gate: general terms pass, identifying text is blocked, search is off by default
    deny = tools.denylist()
    for term in ("CTP1", "TRL", "PVI", "free board"):
        check(T.leak_gate(term, deny) is None, f"leak gate lets the general term '{term}' through")
    for term in ("1242662.9", "1240+766.8", "MKN-3RD_100", "SIVA THUTA", "WEST CENTRAL RAILWAY", "bridge 560 at chainage 1242662.9 on sheet 100"):
        check(T.leak_gate(term, deny) is not None, f"leak gate blocks '{term}'")
    check(tools.web_search("TRL").get("error") == "internet search is switched off", "web search is off unless a provider is given")
    sent = []
    t2 = T.Tools(st, web_provider=lambda q: sent.append(q) or [{"title": "t", "snippet": "TRL: Transition Length", "url": "u"}], log_dir=tmp)
    r = t2.web_search("TRL")
    check(sent == ['"TRL" railway longitudinal section abbreviation meaning'], "only the fixed query template leaves the machine")
    check(r["results"][0]["snippet"].startswith("TRL"), "web result returned and labelled by term")
    t2.web_search("1242662.9")
    check(len(sent) == 1, "a blocked term never reaches the provider")

    # 5. versions: a revised sheet is kept as a new version, the same file is reused
    a = json.loads((ROOT / "data/v4/annotations/MKN-3RD_100.json").read_text(encoding="utf-8"))
    a["sheet_info"]["issue_record"] = a["sheet_info"]["issue_record"] + [{"rev": "R1", "date": "05-10-2026"}]
    b = next(x for x in a["bridges"] if x["bridge_id"] == "560")
    b["lsection_levels"]["proposed_formation_level"] = 180.824
    status = st.add_version(a, "MKN-3RD_100_R1.pdf", "test-r1", 1)
    check(status.startswith("stored as a new version (R1)"), "revised sheet stored as a new version")
    q = tools.query("bridge", line="3rd line", bridge_id="560")
    check(q["levels"]["proposed_formation_level"] == 180.824 and q["older_versions"][0]["proposed_formation_level"] == 180.724,
          "answers use the latest revision and report the older value")
    check(len(tools.query("versions", line="3rd line", sheet_no="100")) == 2, "both versions listed")

    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
