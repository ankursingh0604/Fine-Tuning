"""The v4 assistant loop: model -> tool calls -> tools -> model ... -> answer.

    from agent import Assistant
    bot = Assistant(generate)          # generate(messages, tools, thinking) -> raw model text (see TransformersModel)
    print(bot.ask("What is the FL at bridge 560 on the 3rd line?"))

Tool calls are read in Qwen3.5's own format:
    <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>...</function>\\n</tool_call>
Argument values are typed from docs/tools_v4.json (so bridge "331" stays a string and chainages become numbers).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "runpod_v4"))

import tools as T                 # noqa: E402

TOOLS = json.loads((ROOT / "docs" / "tools_v4.json").read_text(encoding="utf-8"))["tools"]
SCHEMA = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in TOOLS}
CALL_RE = re.compile(r"<tool_call>\s*<function=([\w_]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w_]+)>\s*(.*?)\s*</parameter>", re.S)
MAX_ROUNDS = 8


def typed(tool, key, raw):
    kind = SCHEMA.get(tool, {}).get(key, {}).get("type")
    if kind == "string":
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_calls(text):
    calls = []
    for name, body in CALL_RE.findall(text):
        calls.append({"name": name, "arguments": {k: typed(name, k, v) for k, v in PARAM_RE.findall(body)}})
    return calls


def split_reply(text):
    think = re.search(r"<think>(.*?)</think>", text, re.S)
    final = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    final = CALL_RE.sub("", final)
    final = re.sub(r"<\|[^|]+\|>", "", final).strip()
    return (think.group(1).strip() if think else ""), final


class Assistant:
    def __init__(self, generate, tools=None, system=None):
        self.generate = generate
        self.tools = tools or T.Tools()
        self.messages = [{"role": "system", "content": system or default_system()}]

    def ask(self, question, thinking=None):
        """One user turn; returns the final answer (the conversation is kept for follow-ups)."""
        self.messages.append({"role": "user", "content": question})
        think = True if thinking is None else thinking
        for _ in range(MAX_ROUNDS):
            raw = self.generate(self.messages, TOOLS, think)
            reasoning, final = split_reply(raw)
            calls = parse_calls(raw)
            msg = {"role": "assistant", "content": final if not calls else ""}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = [{"type": "function", "function": c} for c in calls]
            self.messages.append(msg)
            if not calls:
                return final
            for c in calls:
                res = T.call(self.tools, c["name"], c["arguments"])
                self.messages.append({"role": "tool", "name": c["name"], "content": json.dumps(res, ensure_ascii=False)})
        return "I could not finish this within the allowed number of tool calls."


def default_system():
    sys.path.insert(0, str(ROOT / "scripts"))
    import agent_conversations_v4 as G
    return G.SYSTEM


class TransformersModel:
    """generate() for a fine-tuned v4 adapter loaded with Unsloth / transformers (in-house GPU), or for a model that is
    already loaded (model= and processor=, e.g. at the end of train_v4.py)."""

    def __init__(self, adapter=None, load_in_4bit=False, model=None, processor=None, max_new_tokens=1024):
        import data_v4 as DV
        self.DV, self.max_new_tokens = DV, max_new_tokens
        if model is None:
            from unsloth import FastVisionModel
            model, processor = FastVisionModel.from_pretrained(adapter, load_in_4bit=load_in_4bit)
            FastVisionModel.for_inference(model)
        self.model, self.processor = model, processor

    def __call__(self, messages, tools, thinking):
        import torch
        text = self.DV.render(self.processor, messages, thinking, tools, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
