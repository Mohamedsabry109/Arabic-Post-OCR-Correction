# Kaggle / Google Colab Inference Guide

This guide explains how to run LLM inference for Arabic OCR correction on Kaggle or
Google Colab when no local GPU is available.

## Overview

All inference — local, Kaggle, and Colab — uses a single script: **`scripts/infer.py`**.
On remote environments, the script is run after cloning the repo. No files are manually
copied or inlined.

### Three-stage workflow

| Stage | Runs on | Command |
|-------|---------|---------|
| **1. Export** | Local | `python pipelines/run_phase2.py --mode export` |
| **2. Inference** | Kaggle / Colab / Local GPU | `python scripts/infer.py` |
| **3. Analyze** | Local | `python pipelines/run_phase2.py --mode analyze` |

---

## Stage 1 — Export (Local)

Produce `inference_input.jsonl` from the local OCR + GT data.

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

Push the latest code before running remote inference:
```bash
git push
```

---

## Stage 2a — Inference on Kaggle

### Setup (once per notebook)

Create a Kaggle notebook with **GPU T4 x2** accelerator and **Internet enabled**.

Upload `inference_input.jsonl` to a Kaggle dataset.

```python
# Cell 1 — Install deps
!pip install transformers accelerate huggingface_hub pyyaml tqdm -q
```

```python
# Cell 2 — Clone repo
REPO_URL = "https://github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git"
PROJECT_DIR = "/kaggle/working/project"
!git clone {REPO_URL} {PROJECT_DIR}
```

### Run inference

```python
# Cell 3 — Run (with HF sync — recommended for resume across timeouts)
import os
HF_REPO  = "YOUR_HF_USERNAME/arabic-ocr-corrections"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via Kaggle Secrets

!python {PROJECT_DIR}/scripts/infer.py \
    --input  /kaggle/input/YOUR_DATASET/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo  {HF_REPO} \
    --hf-token {HF_TOKEN} \
    --sync-every 100
```

> **Tip**: Store the HF token as a Kaggle secret (`HF_TOKEN`) instead of pasting it in
> the cell. The script also reads `HF_TOKEN` from the environment automatically.

**Resume after timeout**: Re-run Cell 3. The script pulls completed records from HF
before starting so no work is lost.

### Without HF sync

```python
!python {PROJECT_DIR}/scripts/infer.py \
    --input  /kaggle/input/YOUR_DATASET/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507
```

Download `corrections.jsonl` from the Output tab when done. Re-running the same command
resumes from the last written sample automatically.

### Low VRAM (P100 / T4 with 16 GB)

```python
!python {PROJECT_DIR}/scripts/infer.py \
    --input  /kaggle/input/YOUR_DATASET/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --quantize-4bit
```

> Requires `pip install bitsandbytes -q`.

A ready-to-use notebook is at **`notebooks/kaggle_setup.ipynb`**.

---

## Stage 2b — Inference on Google Colab

Output goes directly to Google Drive, so it survives disconnects without HF sync.

```python
# Cell 1 — Mount Drive and install deps
from google.colab import drive
drive.mount('/content/drive')
!pip install transformers accelerate huggingface_hub pyyaml tqdm -q
```

```python
# Cell 2 — Clone repo
REPO_URL = "https://github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git"
PROJECT_DIR = "/content/project"
!git clone {REPO_URL} {PROJECT_DIR}
```

```python
# Cell 3 — Run inference (output to Drive)
DRIVE_DIR = "/content/drive/MyDrive/arabic-ocr"

!python {PROJECT_DIR}/scripts/infer.py \
    --input  {DRIVE_DIR}/inference_input.jsonl \
    --output {DRIVE_DIR}/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507
```

**Resume after disconnect**: Re-mount Drive and re-run Cell 3. The script reads already-
completed sample IDs from the Drive file and skips them.

A ready-to-use notebook is at **`notebooks/colab_setup.ipynb`**.

---

## Stage 2c — Local Inference (GPU or API)

If you have a local GPU or API key, skip Kaggle/Colab entirely:

```bash
# After export:
python scripts/infer.py

# Subset / smoke test
python scripts/infer.py --datasets KHATT-train --limit 50

# Different model
python scripts/infer.py --model Qwen/Qwen3-4B-Instruct-2507
```

Model backend is set in `configs/config.yaml` (`model.backend: "transformers"` or
`model.backend: "api"`).

---

## Stage 3 — Analyze (Local)

Copy the downloaded `corrections.jsonl` to `results/phase2/corrections.jsonl`, then run:

```bash
python pipelines/run_phase2.py --mode analyze
```

The analyze step **auto-splits** the combined file by dataset key into
`results/phase2/{dataset_key}/corrections.jsonl` before computing metrics.

```bash
# One dataset (faster for testing)
python pipelines/run_phase2.py --mode analyze --datasets KHATT-train

# Skip error analysis (much faster)
python pipelines/run_phase2.py --mode analyze --no-error-analysis
```

### Output structure after analyze

```
results/phase2/
├── corrections.jsonl              <- combined file from inference
├── KHATT-train/
│   ├── corrections.jsonl          <- auto-split from combined
│   ├── metrics.json               <- CER/WER on corrected text
│   ├── comparison_vs_phase1.json  <- delta vs Qaari baseline
│   └── error_changes.json         <- which error types were fixed/introduced
├── [17 more dataset folders...]
├── metrics.json                   <- aggregated across all datasets
├── comparison.json                <- aggregated comparison
├── phase2.log
└── report.md
```

---

## infer.py Reference

```
usage: infer.py [-h] [--input PATH] [--output PATH] [--config PATH]
                [--model MODEL] [--datasets DATASET [DATASET ...]]
                [--limit N] [--force] [--max-retries N]
                [--hf-repo REPO] [--hf-token TOKEN] [--sync-every N]

Key arguments:
  --input         Path to inference_input.jsonl  (default: results/phase2/inference_input.jsonl)
  --output        Path to write corrections.jsonl (default: results/phase2/corrections.jsonl)
  --model         Override model name from config.yaml
  --datasets      Process only these dataset keys (space-separated)
  --limit         Max samples per dataset (for testing)
  --force         Re-run all samples, ignoring existing output
  --hf-repo       HuggingFace dataset repo for cross-session sync (user/name)
  --hf-token      HuggingFace token (or set HF_TOKEN env var)
  --sync-every    Push to HF every N samples (default: 100)
```

---

## Troubleshooting

### "Input file not found"

Run the export step first:
```bash
python pipelines/run_phase2.py --mode export
```

### "Input file not found" on Kaggle

Check the dataset path:
```python
import os; os.listdir("/kaggle/input/YOUR_DATASET_NAME/")
```

### OOM / CUDA out of memory

Add `--quantize-4bit` and install bitsandbytes:
```bash
pip install bitsandbytes -q
```

### "enable_thinking is not supported"

Update transformers:
```bash
pip install -U transformers
```

### HF push fails with 401

Check that `HF_TOKEN` is set correctly and has write access to the dataset repo.
Create the dataset repo at huggingface.co/new-dataset if it does not exist yet.

### Kaggle session timed out (with HF sync)

Re-run the inference cell. The script pulls existing progress from HF first.

### Kaggle session timed out (without HF sync)

Re-run the inference cell. The script reads completed sample IDs from
`/kaggle/working/corrections.jsonl` and skips them.

---

## Quick Reference

```bash
# ── LOCAL ──────────────────────────────────────────────────────────
python pipelines/run_phase2.py --mode export   # produce inference_input.jsonl
python scripts/infer.py                        # run inference (local GPU/API)
python pipelines/run_phase2.py --mode analyze  # compute metrics

# ── KAGGLE ─────────────────────────────────────────────────────────
# In Kaggle notebook (see notebooks/kaggle_setup.ipynb):
git clone <repo_url> project
python project/scripts/infer.py \
    --input  /kaggle/input/.../inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --hf-repo user/arabic-ocr-corrections --hf-token hf_xxx

# ── COLAB ──────────────────────────────────────────────────────────
# In Colab notebook (see notebooks/colab_setup.ipynb):
git clone <repo_url> project
python project/scripts/infer.py \
    --input  /content/drive/MyDrive/arabic-ocr/inference_input.jsonl \
    --output /content/drive/MyDrive/arabic-ocr/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507
```
