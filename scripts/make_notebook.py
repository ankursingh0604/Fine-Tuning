"""Generate the training notebook and the RunPod script from one set of cells.

    python scripts/make_notebook.py

Writes:
    notebooks/train_qwen25vl_qlora.ipynb   Google Colab (T4, data on Google Drive)
    runpod/train.py                        RunPod: the same steps as a plain script, run in the
                                           background by runpod/run.sh so a dropped browser or a
                                           broken Jupyter cannot stop training
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(target):
    colab = target == "colab"
    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

    if colab:
        md(r"""
# QLoRA fine-tune of Qwen2.5-VL-3B on railway Plan & L-Section drawings

Runs on a free Colab **T4** (Runtime → Change runtime type → T4 GPU).

1. Upload `dataset.zip` (from `data/dataset.zip` in the Fine-Tuning folder) to `MyDrive/railway_vlm/`.
2. Run all cells. Order: install → load data → **baseline eval of the untouched model** → QLoRA training → eval again → save.

The baseline matters: without it you cannot tell whether fine-tuning did anything.
""")
    else:
        md(r"""
LoRA / QLoRA fine-tune of Qwen2.5-VL-3B on railway Plan & L-Section drawings (RunPod).

Started by run.sh, which installs Unsloth into its own environment (/workspace/venv) so the
pod's Jupyter is never modified, then runs this file in the background with its output in
/workspace/train.log. Order: load data -> baseline eval of the untouched model -> training ->
eval again -> save. Everything is written to /workspace/railway_vlm.

Safe to re-run after an interruption: it reuses the saved baseline answers and resumes
training from the latest checkpoint.
""")

    code(r"""
%%capture
%pip install unsloth
""")

    if colab:
        code(r"""
from google.colab import drive
drive.mount('/content/drive')

OUT = '/content/drive/MyDrive/railway_vlm'
DATA = '/content/dataset'
!mkdir -p {OUT}
!unzip -q -o {OUT}/dataset.zip -d /content/
!ls {DATA} && head -c 400 {DATA}/train.jsonl
""")
    else:
        code(r"""
import os, zipfile

WORK = '/workspace'                       # the pod's persistent volume
OUT = f'{WORK}/railway_vlm'               # predictions, checkpoints, adapter
DATA = f'{WORK}/dataset'
os.environ.setdefault('HF_HOME', f'{WORK}/hf')   # keep the model download on the volume, so a restart does not fetch it again
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(f'{DATA}/train.jsonl'):
    zipfile.ZipFile(f'{WORK}/dataset.zip').extractall(WORK)
print(sorted(os.listdir(DATA)))
print(open(f'{DATA}/train.jsonl', encoding='utf-8').read(400))
""")

    md(r"""
## Settings
""")

    if colab:
        code(r"""
MODEL_NAME = 'unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit'   # 4-bit base = QLoRA
LOAD_4BIT = True
BATCH, GRAD_ACC = 1, 8      # effective batch 8
MAX_SEQ_LEN = 3072          # samples are ~1,300 tokens; whole-sheet layout answers reach ~2,130
EPOCHS = 2
LR = 2e-4
LORA_R = 16
REASONING_REPEAT = 2        # each reasoning (chain-of-thought) row appears this many times in training
EVAL_PER_TASK = 8           # test rows per task for the before/after comparison (83 rows on sheet 100, ~20 min per pass on a T4)
EVAL_STEPS = 50
ADAPTER_DIR = f'{OUT}/qwen25vl3b-railway-lora'
""")
    else:
        code(r"""
import torch

GPU = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9

# Under 20 GB: 4-bit base (QLoRA). 24 GB and up: 16-bit base (plain LoRA) - same method, slightly better quality.
LOAD_4BIT = VRAM_GB < 20
MODEL_NAME = 'unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit' if LOAD_4BIT else 'unsloth/Qwen2.5-VL-3B-Instruct'
# Always an effective batch of 8, so runs on different GPUs are comparable.
BATCH, GRAD_ACC = (1, 8) if VRAM_GB < 20 else (2, 4) if VRAM_GB < 30 else (4, 2)

MAX_SEQ_LEN = 3072          # samples are ~1,300 tokens; whole-sheet layout answers reach ~2,130
EPOCHS = 1                  # one pass over 5,042 rows: about 2.6 h on an L4; 2 = better, twice as long
LR = 2e-4
LORA_R = 16
REASONING_REPEAT = 2        # each reasoning (chain-of-thought) row appears this many times in training
EVAL_PER_TASK = 6           # test rows per task (31 tasks, about 180 rows from sheets 100, 104, 110); 1000 = every row
EVAL_STEPS = 150            # validation loss every N steps (each check takes a few minutes)
ADAPTER_DIR = f'{OUT}/qwen25vl3b-railway-lora'

print(f'{GPU}, {VRAM_GB:.0f} GB -> {"QLoRA (4-bit)" if LOAD_4BIT else "LoRA (16-bit)"}, batch {BATCH} x {GRAD_ACC}')
""")

    md(r"""
## Load the dataset

Images are loaded once and shared between rows that use the same crop. Every image side is a
multiple of 28, so Qwen2.5-VL does not rescale them and the pixel boxes in the answers stay valid.
""")

    code(r"""
import json, random
from PIL import Image

_cache = {}
def image(path):
    if path not in _cache:
        _cache[path] = Image.open(f'{DATA}/{path}').convert('RGB')
    return _cache[path]

import os

def load(split):
    # direct-answer rows plus, when present, the reasoning (chain-of-thought) rows of the same split
    rows = [json.loads(l) for l in open(f'{DATA}/{split}.jsonl', encoding='utf-8')]
    if os.path.exists(f'{DATA}/reasoning_{split}.jsonl'):
        rows += [json.loads(l) for l in open(f'{DATA}/reasoning_{split}.jsonl', encoding='utf-8')]
    return rows

def images_of(r):
    return [image(p) for p in (r.get('images') or [r['image']])]

def question_of(r):
    return r['messages'][0]['content'][-1]['text']

def to_conversation(r):
    # one image entry per image, in order, then the question (reasoning rows can have several images)
    user, assistant = r['messages']
    return {'messages': [
        {'role': 'user', 'content': [{'type': 'image', 'image': im} for im in images_of(r)]
                                    + [{'type': 'text', 'text': question_of(r)}]},
        assistant]}

train_rows, val_rows, test_rows = load('train'), load('val'), load('test')
# Reasoning rows are a small share of the data: repeat them so the step-by-step style is learned.
train_rows += [r for r in train_rows if r['task'].startswith('reason_')] * (REASONING_REPEAT - 1)
train_ds = [to_conversation(r) for r in train_rows]
val_ds = [to_conversation(r) for r in val_rows]
print(len(train_ds), 'train /', len(val_ds), 'val /', len(test_rows), 'test rows,', len(_cache), 'images in memory')

# Fixed evaluation subset: up to EVAL_PER_TASK rows of each task from the held-out test sheet.
random.seed(0)
by_task = {}
for r in test_rows:
    by_task.setdefault(r['task'], []).append(r)
eval_rows = [r for t in sorted(by_task) for r in random.sample(by_task[t], min(EVAL_PER_TASK, len(by_task[t])))]
print(len(eval_rows), 'eval rows across', len(by_task), 'tasks')
""")

    md(r"""
## Load the model
""")

    code(r"""
from unsloth import FastVisionModel, is_bf16_supported
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=LOAD_4BIT,
    max_seq_length=MAX_SEQ_LEN,          # without this Unsloth caps sequences at 2048
    use_gradient_checkpointing='unsloth',
)

# Check that the processor keeps our image sizes: a 1008 px tile must give a 72 x 72 patch grid (14 px patches).
probe = tokenizer.image_processor(images=image(train_rows[0]['image']), return_tensors='pt')
t, h, w = probe['image_grid_thw'][0].tolist()
print('image', image(train_rows[0]['image']).size, '-> patch grid', (h, w), '=', (w * 14, h * 14), 'px')
assert (w * 14, h * 14) == image(train_rows[0]['image']).size, 'processor is rescaling images: boxes would be wrong'
""")

    md(r"""
## Evaluation helpers

Scores are value-based, so the untouched model is not penalised just for using different JSON key names:

| Task group | Metric |
|---|---|
| boxes (`tile_structures`, `bridge_ground`, `sheet_layout`) | precision / recall of boxes at IoU ≥ 0.5; an empty tile counts as correct when nothing is predicted |
| JSON reading (`bridge_json`, `curve_json`, `tbm_json`, `title_json`) | **field acc**: share of reference fields reproduced exactly under the same key; **value recall**: share of the reference values that appear anywhere in the answer |
| text answers | value recall (numbers in the reference found in the answer) and word recall (content words of the reference found in the answer) |
| `bridge_level_check` | yes/no correct **and** both levels quoted correctly (nearly every reference is "Yes") |
""")

    code(r"""
import re, math, time
import pandas as pd

MAX_NEW = {'tile_structures': 400, 'sheet_layout': 700, 'tbm_json': 700, 'title_json': 500,
           'bridge_json': 300, 'curve_json': 300, 'bridge_explain': 300}

def ask(imgs, question, max_new_tokens=128):
    imgs = imgs if isinstance(imgs, list) else [imgs]
    messages = [{'role': 'user', 'content': [{'type': 'image'} for _ in imgs] + [{'type': 'text', 'text': question}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(images=imgs, text=text, add_special_tokens=False, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def parse_json(text):
    m = re.search(r'```json\s*(.*?)```', text, re.S) or re.search(r'(\[.*\]|\{.*\})', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

def boxes(text):
    obj = parse_json(text)
    items = obj if isinstance(obj, list) else [obj] if isinstance(obj, dict) else []
    return [i['bbox_2d'] for i in items if isinstance(i, dict) and isinstance(i.get('bbox_2d'), list) and len(i['bbox_2d']) == 4]

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w*h
    return w*h / union if union > 0 else 0

def box_match(pred, ref):
    used, tp = set(), 0
    for r in ref:
        best = max(((iou(p, r), i) for i, p in enumerate(pred) if i not in used), default=(0, None))
        if best[0] >= 0.5:
            used.add(best[1]); tp += 1
    return tp, len(pred) - tp, len(ref) - tp

def norm(v):
    s = str(v).strip().lower()
    try:
        return f'{float(re.sub(r"[^0-9.+-]", "", s)):.3f}' if re.fullmatch(r'[-+]?[\d.]+\s*(m|mm|kmph)?', s) else s
    except ValueError:
        return s

def flat(obj, prefix=''):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flat(v, f'{prefix}{k}.'))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            key = v.get('tbm_id', i) if isinstance(v, dict) else i
            out.update(flat(v, f'{prefix}{key}.'))
        return out
    return {prefix[:-1]: obj}

def values(text):
    obj = parse_json(text)
    vals = [v for v in flat(obj).values() if v not in (None, '')] if obj is not None else []
    vals += re.findall(r'\d+\.\d+|\d{2,}', text) if obj is None else []
    return {norm(v) for v in vals}

def value_recall(pred, ref):
    ref_vals = values(ref)
    if not ref_vals:
        return None
    p = pred.lower()
    pred_vals = values(pred) | {norm(x) for x in re.findall(r'\d+\.\d+|\d+', p)}
    hits = sum(1 for v in ref_vals if v in pred_vals or (not re.fullmatch(r'[\d.]+', v) and v in p))
    return hits / len(ref_vals)

STOP = set('the a an of is to and in on at for this that from with as are by be it its sheet drawing crop stands shown'.split())

def words(t):
    return {w for w in re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', t.lower()) if w not in STOP and len(w) > 1}

def word_recall(pred, ref):
    r = words(ref)
    return len(r & words(pred)) / len(r) if r else None

def final_part(text):
    # the answer after </think> (the whole text when there is no reasoning block)
    return text.split('</think>', 1)[1].strip() if '</think>' in text else text.strip()

def score(row, pred):
    task, ref = row['task'], row['messages'][1]['content'][0]['text']
    s = {'task': task}
    if task.startswith('reason_'):
        # Reasoning rows are scored on the final answer; the steps are not required to match word for word.
        fr, fp = final_part(ref), final_part(pred)
        s['think'] = float('<think>' in pred and '</think>' in pred)
        s['value_recall'] = value_recall(fp, fr)
        s['word_recall'] = word_recall(fp, fr)
        if 'FLAG:' in fr:
            s['flag_ok'] = float(fr.split('FLAG:', 1)[1].split('(')[0].strip() in fp)
        return s
    if task in ('tile_structures', 'bridge_ground', 'sheet_layout'):
        rb, pb = boxes(ref), boxes(pred)
        if not rb:
            s['empty_ok'] = float(not pb)
        tp, fp, fn = box_match(pb, rb)
        s.update(tp=tp, fp=fp, fn=fn)
    elif task == 'bridge_level_check':
        # Nearly every reference is "Yes", so the verdict alone rewards always saying yes:
        # also require the two levels (proposed FL, MIN FL REQ.) to be quoted correctly.
        verdict_ok = pred.strip().lower()[:3].startswith(ref.strip().lower()[:2])
        levels = re.findall(r'\d+\.\d+', ref)[:2]
        s['correct'] = float(verdict_ok and all(v in pred for v in levels))
    else:
        s['value_recall'] = value_recall(pred, ref)
        if not task.endswith('_json'):
            s['word_recall'] = word_recall(pred, ref)
    if task.endswith('_json'):
        r, p = flat(parse_json(ref) or {}), flat(parse_json(pred) or {})
        r = {k: v for k, v in r.items() if v is not None}
        s['field_acc'] = sum(norm(p.get(k)) == norm(v) for k, v in r.items()) / max(len(r), 1)
    return s

def evaluate(rows, tag):
    FastVisionModel.for_inference(model)
    results, t0 = [], time.time()
    for i, r in enumerate(rows):
        pred = ask(images_of(r), question_of(r), MAX_NEW.get(r['task'], 700 if r['task'].startswith('reason_') else 128))
        results.append({**score(r, pred), 'id': r['id'], 'pred': pred})
        if i % 20 == 0:
            print(f'{tag}: {i}/{len(rows)}  {time.time()-t0:.0f}s')
    df = pd.DataFrame(results)
    df.to_json(f'{OUT}/predictions_{tag}.jsonl', orient='records', lines=True, force_ascii=False)
    return df

def rescore(df):
    # Re-score saved answers after changing score() without generating them again.
    rows = {r['id']: r for r in eval_rows}
    return pd.DataFrame([{**score(rows[i], p), 'id': i, 'pred': p} for i, p in zip(df['id'], df['pred'])])

def summary(df):
    g = df.groupby('task')
    out = pd.DataFrame({'n': g.size()})
    for col in ('field_acc', 'value_recall', 'word_recall', 'correct', 'empty_ok', 'think', 'flag_ok'):
        if col in df:
            out[col] = g[col].mean()
    if 'tp' in df:
        s = g[['tp', 'fp', 'fn']].sum()
        out['box_precision'] = s.tp / (s.tp + s.fp).where(lambda x: x > 0)
        out['box_recall'] = s.tp / (s.tp + s.fn).where(lambda x: x > 0)
    return out.round(3)
""")

    md(r"""
## Baseline: the untouched model on the held-out test sheet
""")

    code(r"""
import os
if os.path.exists(f'{OUT}/predictions_base.jsonl'):      # re-run after an interruption: reuse the saved answers
    base_df = rescore(pd.read_json(f'{OUT}/predictions_base.jsonl', lines=True, dtype={'id': str}))
    print('reusing saved baseline answers')
else:
    base_df = evaluate(eval_rows, 'base')
base_summary = summary(base_df)
base_summary
""")

    code(r"""
# Look at a few raw answers before training.
for _, r in base_df.groupby('task').head(1).iterrows():
    print(f"--- {r.task} ({r.id})\n{r.pred[:400]}\n")
""")

    md(r"""
## Add LoRA adapters and train

Only the answer tokens are trained on (`train_on_responses_only`). The vision layers get adapters too,
because reading small rotated callout text is a vision problem. If you run out of GPU memory,
set `finetune_vision_layers=False` first.
""")

    code(r"""
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R, lora_alpha=LORA_R, lora_dropout=0, bias='none',
    random_state=3407,
)
FastVisionModel.for_training(model)

collator = UnslothVisionDataCollator(
    model, tokenizer,
    resize='max',                       # the default 'min' shrinks every image and breaks the pixel boxes
    max_seq_length=MAX_SEQ_LEN,
    train_on_responses_only=True,
    instruction_part='<|im_start|>user\n',
    response_part='<|im_start|>assistant\n',
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type='cosine',
        optim='adamw_8bit',
        weight_decay=0.01,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=EVAL_STEPS,
        save_strategy='steps',
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        output_dir=f'{OUT}/checkpoints',
        report_to='none',
        seed=3407,
        max_seq_length=MAX_SEQ_LEN,
        remove_unused_columns=False,
        dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True},
    ),
)

print(f'GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.max_memory_reserved()/1e9:.1f} GB reserved before training')
import glob
resume = bool(glob.glob(f'{OUT}/checkpoints/checkpoint-*'))   # continue from the latest checkpoint after an interruption
print('resuming from checkpoint' if resume else 'starting training')
stats = trainer.train(resume_from_checkpoint=resume or None)
print(f"{stats.metrics['train_runtime']/60:.1f} min, peak {torch.cuda.max_memory_reserved()/1e9:.1f} GB")
# Save the trained adapter now, so it is safe even if the pod stops during the final evaluation.
import shutil
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
shutil.make_archive(f'{OUT}/adapter', 'zip', ADAPTER_DIR)
print('adapter saved:', f'{OUT}/adapter.zip')
""")

    code(r"""
# Train vs validation loss. Validation is a whole unseen sheet, so a rising val loss means overfitting to the training sheets.
hist = pd.DataFrame(trainer.state.log_history)
ax = hist.dropna(subset=['loss']).plot(x='step', y='loss', label='train')
hist.dropna(subset=['eval_loss']).plot(x='step', y='eval_loss', label='val', ax=ax)
""")

    md(r"""
## After training: same test rows, then compare
""")

    code(r"""
ft_df = evaluate(eval_rows, 'finetuned')
ft_summary = summary(ft_df)
comparison = base_summary.join(ft_summary, lsuffix='_base', rsuffix='_ft', how='outer')
cols = sorted([c for c in comparison.columns if c not in ('n_base', 'n_ft')], key=lambda c: c.rsplit('_', 1)[0])
comparison = comparison[['n_base'] + cols]
comparison.to_csv(f'{OUT}/comparison.csv')
comparison
""")

    code(r"""
# Side by side on one bridge crop.
r = next(r for r in eval_rows if r['task'] == 'bridge_explain')
display(image(r['image']).resize((504, 504)))
print('Q:', r['messages'][0]['content'][1]['text'])
print('\nREFERENCE:', r['messages'][1]['content'][0]['text'])
print('\nBASE:', base_df.set_index('id').loc[r['id'], 'pred'])
print('\nFINE-TUNED:', ft_df.set_index('id').loc[r['id'], 'pred'])
""")

    md(r"""
## Save the adapter

Only the LoRA weights are saved (tens of MB). To use them later, load them with
`FastVisionModel.from_pretrained(ADAPTER_DIR, load_in_4bit=LOAD_4BIT)`.
""")

    if colab:
        code(r"""
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
!ls -la {ADAPTER_DIR}
""")
    else:
        code(r"""
import shutil
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
shutil.make_archive(f'{OUT}/adapter', 'zip', ADAPTER_DIR)
# Download these from the Jupyter file browser (right-click -> Download), then stop the pod.
for f in ('adapter.zip', 'comparison.csv', 'predictions_base.jsonl', 'predictions_finetuned.jsonl'):
    p = f'{OUT}/{f}'
    print(f'{p:<55} {os.path.getsize(p)/1e6:6.1f} MB' if os.path.exists(p) else f'{p} MISSING')
""")

    md(r"""
## Try it on your own crop

Crop any part of a sheet at about 150 dpi and ask a question. Keep each side a multiple of 28
(or let the processor resize it, in which case boxes refer to the resized image).
""")

    code(r"""
FastVisionModel.for_inference(model)
test_img = image(test_rows[0]['image'])          # replace with Image.open('your_crop.png').convert('RGB')
print(ask(test_img, 'Find all bridge callouts visible in this drawing crop and give their bounding boxes as JSON.', 400))
""")

    if not colab:
        md(r"""
## Optional: score an earlier adapter on the same test rows

If a previous run's `adapter.zip` was uploaded to `/workspace`, load it and score it on the same
rows, giving one base / previous / new comparison table.
""")
        code(r"""
import gc, shutil
PREV_ZIP = f'{WORK}/adapter.zip'
if os.path.exists(PREV_ZIP):
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    prev_dir = f'{WORK}/adapter_prev'
    shutil.rmtree(prev_dir, ignore_errors=True)
    zipfile.ZipFile(PREV_ZIP).extractall(prev_dir)
    model, tokenizer = FastVisionModel.from_pretrained(prev_dir, load_in_4bit=LOAD_4BIT, max_seq_length=MAX_SEQ_LEN)
    prev_df = evaluate(eval_rows, 'previous')
    three = pd.concat({'base': base_summary, 'previous': summary(prev_df), 'new': ft_summary}, axis=1)
    three.to_csv(f'{OUT}/comparison_base_previous_new.csv')
    print(three.to_string(), flush=True)
else:
    print('no earlier adapter.zip in /workspace - skipped')
""")

    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
    if colab:
        meta.update({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
    for c in cells:
        c["source"] = [line + "\n" for line in c["source"].split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def to_script(nb):
    """Notebook cells -> plain Python: markdown becomes comments, bare expressions are printed,
    the install cell is dropped (run.sh installs into its own venv) and the plot is saved."""
    out = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            if not out:
                out.append('"""' + src + '\n"""\nimport matplotlib\nmatplotlib.use("Agg")')
            else:
                out.append("\n".join("# " + l if l else "#" for l in src.split("\n")))
            continue
        if "%pip" in src:
            continue
        lines = [l for l in src.split("\n") if not l.startswith("display(")]
        if re.fullmatch(r"[A-Za-z_]\w*", lines[-1]):
            lines[-1] = f"print({lines[-1]}.to_string(), flush=True)"
        if "hist = pd.DataFrame" in src:
            lines.append("ax.figure.savefig(f'{OUT}/loss.png')")
        out.append("\n".join(lines))
    return "\n\n\n".join(out) + "\n"


if __name__ == "__main__":
    (ROOT / "notebooks").mkdir(exist_ok=True)
    out = ROOT / "notebooks" / "train_qwen25vl_qlora.ipynb"
    out.write_text(json.dumps(build("colab"), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)
    (ROOT / "runpod").mkdir(exist_ok=True)
    out = ROOT / "runpod" / "train.py"
    out.write_text(to_script(build("runpod")), encoding="utf-8")
    print("wrote", out)
