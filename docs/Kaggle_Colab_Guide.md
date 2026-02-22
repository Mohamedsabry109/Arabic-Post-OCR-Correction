# Kaggle / Google Colab Inference Guide

This guide explains how to run LLM inference for Arabic OCR correction on Kaggle or
Google Colab when no local GPU is available.

## Script Overview

| Script | Where it runs | Key feature |
|--------|--------------|-------------|
| `scripts/run_local.py` | Local machine (GPU or API) | Integrates with project, no HF sync |
| `scripts/kaggle_inference.py` | Kaggle | Self-contained (1 file), HF sync required |
| `scripts/colab_inference.py` | Google Colab | Self-contained (1 file), Drive primary + optional HF sync |

The pipeline has three stages:

| Stage | Runs On | Command |
|-------|---------|---------|
| **1. Export** | Local | `python pipelines/run_phase2.py --mode export` |
| **2. Inference** | Kaggle / Colab / Local GPU | see below |
| **3. Analyze** | Local | `python pipelines/run_phase2.py --mode analyze` |

---

## Stage 1 — Export (Local)

Run this on your local machine to produce the input file for remote inference.

```bash
# Full dataset (all 18 datasets)
python pipelines/run_phase2.py --mode export

# Quick smoke test (50 samples)
python pipelines/run_phase2.py --mode export --limit 50

# One dataset only
python pipelines/run_phase2.py --mode export --datasets KHATT-train
```

**Output**: `results/phase2/inference_input.jsonl`

Each line is one sample:
```json
{"sample_id": "AHTD3A0001_Para2_3", "dataset": "KHATT-train", "ocr_text": "...", "gt_text": "..."}
```

---

## Stage 2a — Inference on Kaggle (`kaggle_inference.py`)

### What to upload

Upload **two files** to a Kaggle dataset:
```
results/phase2/inference_input.jsonl    ← produced by Stage 1
scripts/kaggle_inference.py             ← self-contained, no other files needed
```

### Notebook setup

Create a new Kaggle notebook: **GPU T4 x2** accelerator, **Internet enabled**.

```python
# Cell 1 — install deps
!pip install transformers accelerate huggingface_hub -q
# Only if using --quantize-4bit:
# !pip install bitsandbytes -q
```

```python
# Cell 2 — run (with HF sync — recommended)
!python /kaggle/input/your-dataset/kaggle_inference.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo  YourUsername/arabic-ocr-corrections \
    --hf-token hf_xxxxxxxxxxxx \
    --sync-every 100
```

> **Tip**: Store `hf_xxxxxxxxxxxx` as a Kaggle secret (`HF_TOKEN`) instead of
> passing it on the command line. The script reads `HF_TOKEN` automatically.

```python
# Cell 3 — check progress at any time (no model reload)
!python /kaggle/input/your-dataset/kaggle_inference.py \
    --hf-repo YourUsername/arabic-ocr-corrections \
    --hf-token hf_xxxxxxxxxxxx
```

### Without HF sync (manual download)

```python
!python /kaggle/input/your-dataset/kaggle_inference.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507
```

Download `corrections.jsonl` from the **Output** tab when done.

### Resume after timeout

HF sync is the easiest resume path: re-run the same command and the script pulls
existing progress from HF before starting. Without HF sync, progress is in
`/kaggle/working/corrections.jsonl` — re-run the command and it resumes from the
last written sample automatically.

### Low VRAM (P100 / T4 with 16GB)

```python
!python /kaggle/input/your-dataset/kaggle_inference.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --quantize-4bit
```

### Accessing the model on Kaggle

**Option A** — Download from HuggingFace (Internet enabled required, ~8GB, 10-15 min):
```python
# Model downloads automatically — no extra setup needed
```

**Option B** — Add as a Kaggle model (faster, no internet):
1. Go to [kaggle.com/models](https://www.kaggle.com/models) → search `Qwen3-4B`
2. Add it to the notebook as a model input
3. Pass the local path: `--model /kaggle/input/qwen3-4b/transformers/default/1`

---

## Stage 2b — Inference on Google Colab (`colab_inference.py`)

### What to upload

Upload **two files** to Google Drive (e.g. `MyDrive/arabic-ocr/`):
```
results/phase2/inference_input.jsonl
scripts/colab_inference.py
```

### Notebook setup

```python
# Cell 1 — mount Drive and install deps
from google.colab import drive
drive.mount('/content/drive')
!pip install transformers accelerate huggingface_hub -q
```

```python
# Cell 2 — copy script from Drive and run
import shutil
shutil.copy('/content/drive/MyDrive/arabic-ocr/colab_inference.py', '.')

!python colab_inference.py \
    --input  /content/drive/MyDrive/arabic-ocr/inference_input.jsonl \
    --output /content/drive/MyDrive/arabic-ocr/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507
```

Output goes directly to Drive — it survives session disconnects automatically.

### With HF backup (optional)

```python
!python colab_inference.py \
    --input  /content/drive/MyDrive/arabic-ocr/inference_input.jsonl \
    --output /content/drive/MyDrive/arabic-ocr/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo  YourUsername/arabic-ocr-corrections \
    --hf-token hf_xxxxxxxxxxxx \
    --sync-every 50
```

### Check progress without reloading the model

```python
!python colab_inference.py --summary-only \
    --output /content/drive/MyDrive/arabic-ocr/corrections.jsonl
```

### Resume after disconnect

Re-mount Drive and re-run the same command. The script reads already-completed
sample IDs from the Drive file and skips them.

---

## Stage 2c — Local Inference (`run_local.py`)

If you have a local GPU or API key, skip Kaggle/Colab entirely:

```bash
# Local GPU (set model.backend: "transformers" in config.yaml)
python scripts/run_local.py

# API backend (set model.backend: "api" in config.yaml)
python scripts/run_local.py

# Subset of datasets
python scripts/run_local.py --datasets KHATT-train KHATT-validation

# Smoke test
python scripts/run_local.py --limit 50
```

Output is written directly to `results/phase2/{dataset_key}/corrections.jsonl`.
Then run analysis as normal.

---

## Stage 3 — Download and Analyze (Local)

### Place downloaded files

After downloading `corrections.jsonl` from Kaggle/Colab (or HF), split it
by dataset key and place each piece at the right path:

```python
import json
from pathlib import Path
from collections import defaultdict

records = defaultdict(list)
with open("corrections.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        records[r["dataset"]].append(r)

for ds_key, recs in records.items():
    out = Path(f"results/phase2/{ds_key}/corrections.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(recs)} records to {out}")
```

Or if you used `run_local.py`, files are already in the right place.

### Run analysis

```bash
# All datasets
python pipelines/run_phase2.py --mode analyze

# One dataset (faster for testing)
python pipelines/run_phase2.py --mode analyze --datasets KHATT-train

# Skip error analysis (much faster)
python pipelines/run_phase2.py --mode analyze --no-error-analysis
```

### Output structure after analyze

```
results/phase2/
├── KHATT-train/
│   ├── corrections.jsonl          ← from inference
│   ├── metrics.json               ← CER/WER on corrected text
│   ├── comparison_vs_phase1.json  ← delta vs Qaari baseline
│   └── error_changes.json         ← which error types were fixed/introduced
├── [17 more dataset folders...]
├── metrics.json                   ← aggregated across all datasets
├── comparison.json                ← aggregated comparison
├── phase2.log
└── report.md
```

---

## Troubleshooting

### "corrections.jsonl not found"

You need to download the file from Kaggle/Colab first (or pull from HF).

### "Input file not found" (on Kaggle)

Check the dataset path:
```python
import os; os.listdir("/kaggle/input/your-dataset-name/")
```

### OOM / CUDA out of memory

Add `--quantize-4bit`. Requires `bitsandbytes`:
```bash
pip install bitsandbytes
```

### "enable_thinking is not supported"

Update transformers:
```bash
pip install -U transformers
```

### Kaggle session timed out (with HF sync)

Re-run the same command. The script pulls existing progress from HF first.

### Kaggle session timed out (without HF sync)

Re-run the same command. Progress is read from `/kaggle/working/corrections.jsonl`
and completed samples are skipped. **Do not delete** `corrections.jsonl`.

---

## Quick Reference

```bash
# ─── LOCAL ────────────────────────────────────────────────────────
python pipelines/run_phase2.py --mode export       # produce inference_input.jsonl
python scripts/run_local.py                        # run inference locally (GPU/API)
python pipelines/run_phase2.py --mode analyze      # compute metrics

# ─── KAGGLE ───────────────────────────────────────────────────────
# Upload: inference_input.jsonl + kaggle_inference.py
python kaggle_inference.py \
    --input  /kaggle/input/.../inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo  user/arabic-ocr-corrections \
    --hf-token hf_xxx

# ─── COLAB ────────────────────────────────────────────────────────
# Upload: inference_input.jsonl + colab_inference.py to Drive
python colab_inference.py \
    --input  /content/drive/MyDrive/arabic-ocr/inference_input.jsonl \
    --output /content/drive/MyDrive/arabic-ocr/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507
```
