# Kaggle / Google Colab Inference Guide

This guide explains how to run LLM inference for Arabic OCR correction on Kaggle or
Google Colab when no local GPU is available.

The pipeline is split into three stages:

| Stage | Runs On | Command |
|-------|---------|---------|
| **1. Export** | Local machine | `python pipelines/run_phase2.py --mode export` |
| **2. Inference** | Kaggle / Colab | `python run_inference.py` |
| **3. Analyze** | Local machine | `python pipelines/run_phase2.py --mode analyze` |

---

## Stage 1 — Export (Local)

Run this on your local machine to produce the input file for remote inference.

```bash
# Full dataset
python pipelines/run_phase2.py --mode export

# Quick smoke test (50 samples)
python pipelines/run_phase2.py --mode export --limit 50

# One dataset only
python pipelines/run_phase2.py --mode export --dataset KHATT-train
```

**Output**: `results/phase2/inference_input.jsonl`

Each line is one sample:
```json
{"sample_id": "AHTD3A0001_Para2_3", "dataset": "KHATT-train", "ocr_text": "...", "gt_text": "..."}
```

---

## Stage 2 — Inference on Kaggle

### 2.1 Files to Upload

Upload these four items to your Kaggle notebook (as a dataset or directly):

```
results/phase2/inference_input.jsonl    ← produced by Stage 1
src/core/prompt_builder.py
src/core/llm_corrector.py
scripts/run_inference.py
```

The inference script only needs these four files. No database, GT files, or
full project structure is required on the remote machine.

### 2.2 Kaggle Notebook Setup

Create a new Kaggle notebook with **GPU T4 x2** accelerator and **Internet enabled**.

#### Cell 1 — Install dependencies
```python
!pip install transformers accelerate -q
# Only needed if using --quantize-4bit:
# !pip install bitsandbytes -q
```

#### Cell 2 — Set paths (adjust to your Kaggle dataset paths)
```python
INPUT_JSONL  = "/kaggle/input/your-dataset-name/inference_input.jsonl"
OUTPUT_JSONL = "/kaggle/working/corrections.jsonl"
MODEL_NAME   = "Qwen/Qwen3-4B-Instruct-2507"
```

#### Cell 3 — Run inference
```python
!python /kaggle/input/your-dataset-name/run_inference.py \
    --input  {INPUT_JSONL} \
    --output {OUTPUT_JSONL} \
    --model  {MODEL_NAME}
```

Or if you uploaded the scripts directly into the working directory:
```python
import subprocess
result = subprocess.run([
    "python", "run_inference.py",
    "--input",  INPUT_JSONL,
    "--output", OUTPUT_JSONL,
    "--model",  MODEL_NAME,
], capture_output=False)
```

#### Cell 4 — Verify output
```python
import json
with open(OUTPUT_JSONL) as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(f"Total records: {len(lines)}")
print(f"Successful   : {sum(1 for r in lines if r['success'])}")
print(f"Failed       : {sum(1 for r in lines if not r['success'])}")
print("\nExample:")
print(json.dumps(lines[0], ensure_ascii=False, indent=2))
```

### 2.3 Running Per Dataset

The full dataset (~7,000 samples) may take 4–6 hours on a T4 GPU. Run one dataset
at a time if your session might time out:

```bash
# Export one dataset
python pipelines/run_phase2.py --mode export --dataset KHATT-train

# Run inference for that dataset on Kaggle
python run_inference.py --input inference_input.jsonl --output KHATT-train_corrections.jsonl

# Then repeat for other datasets
```

**Resume after timeout**: The script writes results line-by-line. If the Kaggle session
expires, restart and re-run the same command. Already-completed sample IDs are detected
and skipped automatically.

### 2.4 Low VRAM GPUs (P100 / T4 with 16GB)

The Qwen3-4B model in float16 uses approximately 8GB VRAM. If you hit OOM errors:

```python
!python run_inference.py \
    --input  {INPUT_JSONL} \
    --output {OUTPUT_JSONL} \
    --model  {MODEL_NAME} \
    --quantize-4bit   # enables 4-bit NF4 via bitsandbytes (~4GB VRAM)
```

### 2.5 Uploading the Project as a Kaggle Dataset

**Option A** (recommended for ease): Upload only the four required files as a Kaggle dataset.

**Option B** (for reusability): Upload the entire project `src/` directory as a dataset so
the full package is available for all phases. Add this to your notebook:

```python
import sys
sys.path.insert(0, "/kaggle/input/your-project-dataset/")
```

### 2.6 Accessing the Model on Kaggle

**Option A** — Download from HuggingFace (requires Internet enabled in Settings):
```python
# The model downloads automatically on first run
# ~8GB download — allow 10-15 minutes
```

**Option B** — Use as a Kaggle model (faster, no internet needed):
1. Go to [kaggle.com/models](https://www.kaggle.com/models) and search for `Qwen3-4B`
2. Add it to your notebook as a model input
3. Pass the local path to `--model`:
```python
!python run_inference.py \
    --input  {INPUT_JSONL} \
    --output {OUTPUT_JSONL} \
    --model  /kaggle/input/qwen3-4b/transformers/default/1
```

---

## Stage 2 — Inference on Google Colab

### 2A.1 Mount Google Drive (recommended for large datasets)

```python
from google.colab import drive
drive.mount('/content/drive')

INPUT_JSONL  = "/content/drive/MyDrive/arabic-ocr/inference_input.jsonl"
OUTPUT_JSONL = "/content/drive/MyDrive/arabic-ocr/corrections.jsonl"
```

Saving to Drive means output is preserved if the Colab session disconnects.

### 2A.2 Install Dependencies

```python
!pip install transformers accelerate -q
```

### 2A.3 Upload Project Files

Upload via the Files panel (left sidebar) or from Drive:

```python
# From Drive (if you uploaded them there)
import shutil
shutil.copy("/content/drive/MyDrive/arabic-ocr/run_inference.py", ".")
shutil.copy("/content/drive/MyDrive/arabic-ocr/prompt_builder.py", "src/core/")
shutil.copy("/content/drive/MyDrive/arabic-ocr/llm_corrector.py", "src/core/")
```

Or use the Colab file upload widget:
```python
from google.colab import files
uploaded = files.upload()  # select inference_input.jsonl + the three .py files
```

### 2A.4 Run Inference

```python
!python run_inference.py \
    --input  {INPUT_JSONL} \
    --output {OUTPUT_JSONL} \
    --model  Qwen/Qwen3-4B-Instruct-2507
```

### 2A.5 Download Output

```python
from google.colab import files
files.download(OUTPUT_JSONL)
```

Or it's already in Drive if you used Drive paths.

### 2A.6 Colab Session Limits

Free Colab sessions run for up to 12 hours and may disconnect. Resume support is built in:
- If the session disconnects, re-mount Drive and re-run the inference command
- The script reads already-completed lines from `corrections.jsonl` and skips them

**Tip**: Use Colab Pro for longer sessions and GPU priority.

---

## Stage 3 — Download and Analyze (Local)

### 3.1 Place Downloaded Files

After downloading `corrections.jsonl` from Kaggle/Colab, place it at:

```
results/phase2/PATS-A01-Akhbar/corrections.jsonl
results/phase2/PATS-A01-Andalus/corrections.jsonl
results/phase2/KHATT-train/corrections.jsonl
results/phase2/KHATT-validation/corrections.jsonl
```

If you ran all datasets in one JSONL, split by the `"dataset"` field:

```python
import json
from pathlib import Path
from collections import defaultdict

records = defaultdict(list)
with open("corrections.jsonl") as f:
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

### 3.2 Run Analysis

```bash
# All datasets
python pipelines/run_phase2.py --mode analyze

# One dataset (faster for testing)
python pipelines/run_phase2.py --mode analyze --dataset KHATT-train

# Skip error analysis (much faster)
python pipelines/run_phase2.py --mode analyze --no-error-analysis

# Specific results directory
python pipelines/run_phase2.py --mode analyze --results-dir results/phase2
```

### 3.3 Output Files

After a successful analyze run:

```
results/phase2/
├── KHATT-train/
│   ├── corrections.jsonl          ← downloaded from Kaggle/Colab
│   ├── metrics.json               ← CER/WER on corrected text
│   ├── comparison_vs_phase1.json  ← delta vs Qaari baseline
│   └── error_changes.json         ← which error types were fixed/introduced
├── [other datasets...]
├── metrics.json                   ← aggregated across all datasets
├── comparison.json                ← aggregated comparison
├── phase2.log                     ← full run log
└── report.md                      ← human-readable summary
```

---

## Troubleshooting

### "corrections.jsonl not found"

```
FileNotFoundError: corrections.jsonl not found: results/phase2/KHATT-train/corrections.jsonl
```

You need to download the file from Kaggle/Colab first. See Stage 2 above.

### "Input file not found" (on Kaggle)

The `inference_input.jsonl` was not uploaded correctly. Check the Kaggle dataset path:
```python
import os
os.listdir("/kaggle/input/your-dataset-name/")
```

### OOM / CUDA out of memory

Add `--quantize-4bit` to the inference command. Requires `bitsandbytes`:
```bash
pip install bitsandbytes
python run_inference.py --quantize-4bit ...
```

### Model outputs wrong language or empty text

This is handled automatically: `TransformersCorrector._extract_corrected_text()` checks for
Arabic characters and falls back to the original OCR text if the output is empty or non-Arabic.
These samples are marked `"success": false` in `corrections.jsonl`.

### Kaggle session timed out

Re-run the same inference command — the script resumes from the last completed sample.
Do not delete `corrections.jsonl` between runs.

### "enable_thinking is not supported"

Older versions of `transformers` may not support `enable_thinking=False` for Qwen3. Update:
```bash
pip install -U transformers
```

---

## Quick Reference

```bash
# ─── LOCAL ────────────────────────────────────────────────────
# Prepare data for remote inference
python pipelines/run_phase2.py --mode export

# After downloading corrections.jsonl:
python pipelines/run_phase2.py --mode analyze

# ─── KAGGLE/COLAB ──────────────────────────────────────────────
# Full dataset
python run_inference.py --model Qwen/Qwen3-4B-Instruct-2507

# One dataset, 4-bit quantization, resume-safe
python run_inference.py \
    --input  inference_input.jsonl \
    --output corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --quantize-4bit

# Smoke test (100 samples)
python run_inference.py --limit 100
```
