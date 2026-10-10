"""The L-section v5 model (Qwen3.5-4B + the v5 LoRA adapter) behind the same ask(images, question) as app.Model, so
Railsight (chat_ui.py, read_sheet.py) and ask_lsec.py can use it for reading crops and for answering with facts.

    m = Model("lsec_run/adapter.zip")          # or an unzipped adapter folder; four_bit=True on a shared / small GPU
    m.ask([crop], "What text is written in this crop? Reply with the exact text only.")
"""
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "lsec_kit", HERE.parent / "lsec_bundle" / "lsec_kit"):
    if p.exists():
        sys.path.insert(0, str(p))


def _adapter_dir(path):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path
    out = Path(tempfile.gettempdir()) / f"lsec_adapter_{path.stat().st_mtime_ns}"
    if not out.exists():
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
    # the zip may hold the adapter folder or its files
    inner = [d for d in out.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()]
    return inner[0] if inner else out


class Model:
    def __init__(self, adapter, four_bit=False):
        from unsloth import FastVisionModel
        self.model, self.processor = FastVisionModel.from_pretrained(str(_adapter_dir(adapter)), load_in_4bit=four_bit)
        FastVisionModel.for_inference(self.model)

    def ask(self, imgs, q, max_new=600):
        import torch
        import data_lsec as DL
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": q}]}]
        text = DL.render(self.processor, msgs, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[im.convert("RGB") for im in imgs] or None, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
