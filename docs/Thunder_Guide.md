# Thunder Compute -- Full Setup & Inference Guide

End-to-end instructions for running the Arabic OCR correction pipeline on
Thunder Compute (A100 80 GB). Covers instance creation, environment setup,
data transfer, inference for all phases, and getting results back to your
local machine.

---

## Table of Contents

1. [Prerequisites (Local Machine)](#1-prerequisites-local-machine)
2. [Create a Thunder Instance](#2-create-a-thunder-instance)
3. [Connect to the Instance](#3-connect-to-the-instance)
4. [One-Time Environment Setup](#4-one-time-environment-setup)
5. [Transfer Data to Thunder](#5-transfer-data-to-thunder)
6. [Run Inference -- Phase 2 (Zero-Shot)](#6-run-inference----phase-2-zero-shot)
7. [Run Inference -- Phase 3 (OCR-Aware)](#7-run-inference----phase-3-ocr-aware)
8. [Run Inference -- Phase 4 (Self-Reflective)](#8-run-inference----phase-4-self-reflective)
9. [Run Inference -- Phase 6 (Combinations)](#9-run-inference----phase-6-combinations)
10. [Run Inference -- Phase 7 (DSPy)](#10-run-inference----phase-7-dspy)
11. [Run Qaari OCR (Re-OCR Images)](#11-run-qaari-ocr-re-ocr-images)
12. [Transfer Results Back](#12-transfer-results-back)
13. [Analyze Results (Local Machine)](#13-analyze-results-local-machine)
14. [Resume After Crash / Disconnect](#14-resume-after-crash--disconnect)
15. [Cost & Performance Notes](#15-cost--performance-notes)
16. [Troubleshooting](#16-troubleshooting)
17. [Quick Reference Card](#17-quick-reference-card)

---

## 1. Prerequisites (Local Machine)

Before touching Thunder, prepare everything locally.

### 1a. Export inference inputs for all phases you plan to run

Each phase produces its own `inference_input.jsonl` that the Thunder scripts
consume. Export them all up front so you only need one data transfer.

```bash
# Phase 2 (zero-shot) -- always needed first
python pipelines/run_phase2.py --mode export

# Phase 3 (OCR-aware prompting)
python pipelines/run_phase3.py --mode export

# Phase 4 (self-reflective)
python pipelines/run_phase4.py --mode export

# Phase 6 (combinations -- exports all combos)
python pipelines/run_phase6.py --mode export --combo all

# Phase 7 (DSPy)
python pipelines/run_phase7.py --mode export
```

After exporting, you should have files like:
```
results/phase2/inference_input.jsonl
results/phase3/inference_input.jsonl
results/phase4/inference_input.jsonl
results/phase6/conf_only/inference_input.jsonl
results/phase6/self_only/inference_input.jsonl
results/phase6/conf_self/inference_input.jsonl
results/phase7/inference_input.jsonl
```

### 1b. Push your code to GitHub

Thunder will clone the repo, so make sure everything is committed and pushed:

```bash
git add -A
git commit -m "export inference inputs for Thunder run"
git push
```

### 1c. (Optional) Create a HuggingFace dataset repo

If you want automatic cloud backup of results during inference:

1. Go to https://huggingface.co/new-dataset
2. Create a **private** dataset repo (e.g. `your-username/arabic-ocr-results`)
3. Generate a write token at https://huggingface.co/settings/tokens

---

## 2. Create a Thunder Instance

1. Go to https://www.thundercompute.com and sign in
2. Click **Create Instance**
3. Select:
   - **GPU**: A100 80 GB (required for full vLLM throughput; A100 40 GB works
     with `--gpu-memory-util 0.85 --max-model-len 16384`)
   - **Image**: Ubuntu 22.04 + CUDA 12.1 (or whatever CUDA 12.x is available)
   - **Disk**: 100 GB minimum (models ~12 GB + data + working space)
   - **Region**: whatever is cheapest / available
4. Click **Launch** and wait for the instance to reach `Running` state
5. Note the **SSH command** or **IP address** shown in the dashboard

---

## 3. Connect to the Instance

### Option A -- SSH from terminal

```bash
# Thunder typically provides a command like:
ssh -i ~/.ssh/thunder_key root@<INSTANCE_IP>

# Or via Thunder CLI:
thunder ssh <instance-id>
```

### Option B -- Thunder web terminal

Click the **Terminal** button in the Thunder dashboard for a browser-based shell.

### Verify GPU access

Once connected:

```bash
nvidia-smi
```

You should see the A100 80 GB with CUDA driver loaded.

---

## 4. One-Time Environment Setup

### 4a. Clone the project

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git
cd Arabic-Post-OCR-Correction
```

> For private repos:
> ```bash
> git clone https://YOUR_GITHUB_TOKEN@github.com/YOUR_USERNAME/Arabic-Post-OCR-Correction.git
> ```

### 4b. Run the setup script

This installs all Python packages, downloads both models (~12 GB total), and
verifies GPU access. Takes ~10-15 minutes on a fresh instance.

```bash
bash thunder/setup.sh
```

What it does (in order):
1. Installs system packages (git, wget, curl, build-essential)
2. Installs Python packages: torch (CUDA 12.1), vLLM, transformers, accelerate,
   qwen-vl-utils, flash-attn, camel-tools, dspy-ai, scipy, etc.
3. Downloads CAMeL Tools morphology database
4. Pre-downloads Qaari OCR model (~4 GB) to HuggingFace cache
5. Pre-downloads Qwen3-4B-Instruct-2507 (~8 GB) to HuggingFace cache
6. Verifies GPU is accessible and reports VRAM

### 4c. Verify the setup

```bash
# Check vLLM is installed
python -c "import vllm; print('vLLM', vllm.__version__)"

# Check models are cached (should complete instantly, no download)
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', trust_remote_code=True)
print('Qwen3 tokenizer OK')
"

# Check project imports work
python -c "from src.core.prompt_builder import PromptBuilder; print('Imports OK')"
```

---

## 5. Transfer Data to Thunder

The inference input JSONL files need to be on the Thunder instance. Three
approaches, from simplest to most robust:

### Option A -- Git (simplest, if files are committed)

If you committed the inference_input.jsonl files:

```bash
cd /workspace/Arabic-Post-OCR-Correction
git pull
```

### Option B -- scp from local machine

From your **local** terminal:

```bash
# Transfer all results (inference inputs) at once
scp -r results/ root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/results/

# Or transfer specific phases
scp results/phase2/inference_input.jsonl \
    root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/results/phase2/

scp results/phase3/inference_input.jsonl \
    root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/results/phase3/
```

### Option C -- HuggingFace (best for large files / resume across instances)

From local:
```bash
export HF_TOKEN=hf_xxx
python scripts/hf_sync.py push --paths results
```

On Thunder:
```bash
export HF_TOKEN=hf_xxx
python scripts/hf_sync.py pull --paths results
```

### For Qaari OCR (if re-running OCR on images)

Upload the image dataset. This is large (~several GB), so scp or rsync is best:

```bash
# From local machine
scp -r data/images/ root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/data/images/
```

---

## 6. Run Inference -- Phase 2 (Zero-Shot)

Phase 2 is the baseline LLM correction. Run this first.

### Smoke test (20 samples, ~2 minutes)

```bash
cd /workspace/Arabic-Post-OCR-Correction

python thunder/qwen_infer.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl \
    --limit 20
```

Check that:
- The **prompt preview** looks correct (should show the zero-shot prompt)
- Inference runs without errors
- Output file has 20 lines: `wc -l results/phase2/corrections.jsonl`

### Full run (all samples, vLLM)

```bash
python thunder/qwen_infer.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl
```

Expected throughput on A100 80 GB: **5-15x faster than sequential** (~50-200
samples/sec depending on prompt length).

### Full run (transformers fallback, if vLLM has issues)

```bash
python thunder/qwen_infer.py \
    --backend transformers \
    --batch-size 16 \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl
```

### (Optional) Push results to HF after each phase

```bash
export HF_TOKEN=hf_xxx
python scripts/hf_sync.py push --paths results/phase2
```

---

## 7. Run Inference -- Phase 3 (OCR-Aware)

```bash
python thunder/qwen_infer.py \
    --input  results/phase3/inference_input.jsonl \
    --output results/phase3/corrections.jsonl
```

The prompt preview should show the OCR-aware prompt with confusion matrix
context injected.

---

## 8. Run Inference -- Phase 4 (Self-Reflective)

```bash
python thunder/qwen_infer.py \
    --input  results/phase4/inference_input.jsonl \
    --output results/phase4/corrections.jsonl
```

The prompt preview should show training artifact insights and overcorrection
warnings.

---

## 9. Run Inference -- Phase 6 (Combinations)

Phase 6 has multiple combos, each with its own input/output directory.

```bash
# Confusion only
python thunder/qwen_infer.py \
    --input  results/phase6/conf_only/inference_input.jsonl \
    --output results/phase6/conf_only/corrections.jsonl

# Self-reflective only
python thunder/qwen_infer.py \
    --input  results/phase6/self_only/inference_input.jsonl \
    --output results/phase6/self_only/corrections.jsonl

# Confusion + self-reflective
python thunder/qwen_infer.py \
    --input  results/phase6/conf_self/inference_input.jsonl \
    --output results/phase6/conf_self/corrections.jsonl
```

Or run them back-to-back in one command:

```bash
for combo in conf_only self_only conf_self; do
    echo "=== Running Phase 6: $combo ==="
    python thunder/qwen_infer.py \
        --input  results/phase6/$combo/inference_input.jsonl \
        --output results/phase6/$combo/corrections.jsonl
done
```

> **Note**: Phase 6's `best_camel` combo does NOT need inference -- it applies
> CAMeL validation locally during the analyze step.

---

## 10. Run Inference -- Phase 7 (DSPy)

Phase 7 uses DSPy for automated prompt optimization. This runs a different
script:

```bash
python scripts/dspy_optimize.py \
    --input  results/phase7/inference_input.jsonl \
    --output results/phase7/corrections.jsonl
```

Or use the standard Qwen inference if you have a pre-optimized prompt:

```bash
python thunder/qwen_infer.py \
    --input  results/phase7/inference_input.jsonl \
    --output results/phase7/corrections.jsonl
```

---

## 11. Run Qaari OCR (Re-OCR Images)

Only needed if you want to re-run OCR on the source images (e.g., to reproduce
results from scratch or test a different OCR model).

### Smoke test

```bash
python thunder/qaari_infer.py \
    --image-root  data/images \
    --output-root data/ocr-results/qaari-results \
    --dry-run
```

### Full run (vLLM)

```bash
python thunder/qaari_infer.py \
    --image-root  data/images \
    --output-root data/ocr-results/qaari-results
```

### Full run (transformers fallback)

```bash
python thunder/qaari_infer.py \
    --backend transformers \
    --batch-size 16 \
    --image-root  data/images \
    --output-root data/ocr-results/qaari-results
```

---

## 12. Transfer Results Back

After inference completes, get the results back to your local machine.

### Option A -- scp (direct)

From your **local** terminal:

```bash
# Pull all results
scp -r root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/results/ ./results/

# Pull a specific phase
scp root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/results/phase2/corrections.jsonl \
    ./results/phase2/corrections.jsonl

# Pull Qaari OCR results too
scp -r root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/data/ocr-results/qaari-results/ \
    ./data/ocr-results/qaari-results/
```

### Option B -- HuggingFace sync

On Thunder (push):
```bash
export HF_TOKEN=hf_xxx
python scripts/hf_sync.py push
```

On local (pull):
```bash
export HF_TOKEN=hf_xxx
python scripts/hf_sync.py pull
```

### Option C -- rsync (best for large transfers / partial updates)

From your **local** terminal:

```bash
rsync -avz --progress \
    root@<INSTANCE_IP>:/workspace/Arabic-Post-OCR-Correction/results/ \
    ./results/
```

---

## 13. Analyze Results (Local Machine)

Once `corrections.jsonl` files are on your local machine, run the analyze step
for each phase. This computes CER/WER metrics, comparisons, and error analysis.

```bash
# Phase 2
python pipelines/run_phase2.py --mode analyze

# Phase 3
python pipelines/run_phase3.py --mode analyze

# Phase 4
python pipelines/run_phase4.py --mode analyze

# Phase 5 (CAMeL validation -- runs locally, no GPU needed)
python pipelines/run_phase5.py --mode validate

# Phase 6 (all combos)
python pipelines/run_phase6.py --mode analyze --combo all
python pipelines/run_phase6.py --mode validate --combo best_camel
python pipelines/run_phase6.py --mode summarize --combo all

# Phase 7
python pipelines/run_phase7.py --mode analyze
```

---

## 14. Resume After Crash / Disconnect

All Thunder scripts support automatic resume. They read completed `sample_id`s
(Qwen) or check for existing non-empty `.txt` files (Qaari) and skip them.

**Just re-run the exact same command.** Already-processed samples are skipped
automatically:

```bash
# Re-run after crash -- picks up where it left off
python thunder/qwen_infer.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl
```

To force reprocessing everything:

```bash
python thunder/qwen_infer.py \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl \
    --force
```

### If the Thunder instance itself is lost

If you were using HF sync, pull the partial results on a new instance:

```bash
export HF_TOKEN=hf_xxx
python scripts/hf_sync.py pull
# Then re-run inference -- it resumes from the pulled progress
python thunder/qwen_infer.py \
    --input results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl
```

---

## 15. Cost & Performance Notes

### Expected throughput (A100 80 GB, Qwen3-4B FP16)

| Backend | Throughput | Notes |
|---------|-----------|-------|
| vLLM (default) | ~50-200 samples/sec | Continuous batching, all prompts at once |
| Transformers batch (bs=16) | ~5-15 samples/sec | Left-padded batched generate |
| Sequential (reference) | ~1-3 samples/sec | One sample at a time |

### Estimated wall-clock time (full dataset, ~3300 val samples)

| Backend | Phase 2 | Phase 3/4 (longer prompts) |
|---------|---------|---------------------------|
| vLLM | ~30-60 sec | ~1-3 min |
| Transformers batch | ~5-10 min | ~10-20 min |

### Cost optimization tips

- **Run setup.sh first** -- model downloads during setup are free (no GPU billing
  on some providers) or at least don't waste inference time.
- **Run all phases back-to-back** in a single session -- vLLM keeps the model in
  GPU memory, so switching between phase inputs is instant.
- **Use HF sync** -- if the instance crashes, you don't lose all progress.
- **Stop the instance when done** -- Thunder bills by the hour. Don't forget!

---

## 16. Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvidia-smi` shows no GPU | Instance may not have started properly. Restart it from the Thunder dashboard. |
| `vllm` import error | Run `pip install vllm>=0.6.3` or re-run `bash thunder/setup.sh`. |
| `CUDA out of memory` (vLLM) | Reduce context: `--gpu-memory-util 0.85 --max-model-len 16384`. |
| `CUDA out of memory` (transformers) | Reduce batch size: `--batch-size 4`. Or use the `_seq` script. |
| Model download stalls | Check internet. Pre-download with `python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507')"`. |
| `Input file not found` | The inference_input.jsonl is not on the instance. Transfer it (Section 5). |
| Wrong prompt in preview | The prompt is derived from `inference_input.jsonl` fields. Re-export locally from the correct phase, then re-transfer. |
| `ModuleNotFoundError: src.core...` | You are not in the project directory. `cd /workspace/Arabic-Post-OCR-Correction`. |
| `enable_thinking is not supported` | Update transformers: `pip install -U transformers`. |
| SSH connection drops | Use `tmux` or `screen` (Section 16a below) so inference survives disconnects. |
| `flash-attn` install fails | Not critical. vLLM has its own attention; HF path works without it (just slower for Qaari). |
| Empty corrections for some samples | Normal for a few edge cases. The script falls back to OCR text. Check logs for warnings. |

### 16a. Using tmux (recommended for long runs)

Always run inference inside tmux so it survives SSH disconnects:

```bash
# Start a new tmux session
tmux new -s inference

# Run your commands inside tmux...
python thunder/qwen_infer.py --input ... --output ...

# Detach: press Ctrl+B, then D
# Reconnect later:
tmux attach -t inference
```

---

## 17. Quick Reference Card

```bash
# ============================================================
# THUNDER QUICK REFERENCE
# ============================================================

# --- LOCAL (before Thunder) ---
python pipelines/run_phase2.py --mode export       # export phase 2
python pipelines/run_phase3.py --mode export       # export phase 3
python pipelines/run_phase4.py --mode export       # export phase 4
python pipelines/run_phase6.py --mode export --combo all  # export phase 6
python pipelines/run_phase7.py --mode export       # export phase 7
git push                                           # push code

# --- THUNDER (one-time setup) ---
cd /workspace
git clone https://github.com/USER/Arabic-Post-OCR-Correction.git
cd Arabic-Post-OCR-Correction
bash thunder/setup.sh                              # install deps + models
tmux new -s inference                              # start tmux session

# --- THUNDER (transfer data) ---
# Option A: git pull (if inputs are committed)
# Option B: scp from local
# Option C: HF_TOKEN=hf_xxx python scripts/hf_sync.py pull

# --- THUNDER (inference -- run all phases) ---
python thunder/qwen_infer.py \
    --input results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl

python thunder/qwen_infer.py \
    --input results/phase3/inference_input.jsonl \
    --output results/phase3/corrections.jsonl

python thunder/qwen_infer.py \
    --input results/phase4/inference_input.jsonl \
    --output results/phase4/corrections.jsonl

for combo in conf_only self_only conf_self; do
    python thunder/qwen_infer.py \
        --input  results/phase6/$combo/inference_input.jsonl \
        --output results/phase6/$combo/corrections.jsonl
done

python thunder/qwen_infer.py \
    --input results/phase7/inference_input.jsonl \
    --output results/phase7/corrections.jsonl

# --- THUNDER (push results) ---
HF_TOKEN=hf_xxx python scripts/hf_sync.py push

# --- LOCAL (pull results & analyze) ---
HF_TOKEN=hf_xxx python scripts/hf_sync.py pull    # or scp
python pipelines/run_phase2.py --mode analyze
python pipelines/run_phase3.py --mode analyze
python pipelines/run_phase4.py --mode analyze
python pipelines/run_phase5.py --mode validate     # local only, no GPU
python pipelines/run_phase6.py --mode analyze --combo all
python pipelines/run_phase6.py --mode validate --combo best_camel
python pipelines/run_phase6.py --mode summarize --combo all
python pipelines/run_phase7.py --mode analyze
```
