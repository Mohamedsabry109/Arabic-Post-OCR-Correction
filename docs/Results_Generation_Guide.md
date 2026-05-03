# Results Generation Guide

This document is the single source of truth for generating all paper results.
Follow every step in the order listed. Do not skip ahead.

---

## Overview

| Split | Purpose | Phases |
|-------|---------|--------|
| **Training** | Build all knowledge sources used to enrich validation prompts | Phase 1 (confusion matrices → `results/phase1-training/`), Phase 2 (LLM corrections → `results/phase2-training/`) |
| **Validation** | Produce final evaluation numbers reported in the paper | Phases 1–8 (evaluation only) |

**Leakage-free design.** Every piece of knowledge injected into a validation prompt must come exclusively from training data:

| Knowledge artifact | Source | Used by |
|-------------------|--------|---------|
| Character confusion matrix | `results/phase1-training/{train_key}/confusion_matrix.json` | Phases 3, 6 |
| LLM failure word pairs / overcorrections | `results/phase2-training/analysis/word_pairs_llm_failures.txt` | Phases 3, 4, 5, 6 |
| Self-reflective insights + few-shot examples | `results/phase2-training/` corrections | Phase 4 |
| BM25 RAG index | `results/phase2-training/` corrections | Phase 8 |

**Inference rule:** Every phase that requires an LLM follows the same three-step pattern:
`export` (local) → `infer.py` (Kaggle or Thunder) → `analyze` (local).
Phase 5 and the Phase 6 `best_camel` combo are local-only (no inference step).

---

## Current Status

| Step | Status |
|------|--------|
| Phase 1 — training split → `results/phase1/` (to rename → `results/phase1-training/`) | ✓ Done |
| Phase 2 — training split → `results/phase2/` (to rename → `results/phase2-training/`) | ✓ Done |
| Rename Phase 1 and Phase 2 training folders | ✗ Pending (Part A) |
| Training artifact generation | ✗ Pending (Part A) |
| Phase 4 analyze-train | ✗ Pending (Part A) |
| Phase 8 RAG index build | ✗ Pending (Part A) |
| All validation phases | ✗ Pending (Part B) |

---

## Part A — Training Folder Rename + Artifact Generation

### A0 — Rename Training Output Folders

Both Phase 1 and Phase 2 were run with the training config and wrote to the default
`results/phase1/` and `results/phase2/` directories. Rename them so the final layout is:

- `results/phase1-training/` — training Phase 1 results (confusion matrices for Phases 3 and 6)
- `results/phase1/` — free for validation Phase 1 (step B1)
- `results/phase2-training/` — training Phase 2 corrections (for A1–A3 and all artifact paths)
- `results/phase2/` — free for validation Phase 2 (step B2)

```bash
mv results/phase1 results/phase1-training
mv results/phase2 results/phase2-training
```

> `results/phase1-training/baseline_metrics.json` holds the training aggregate.
> It will not be overwritten by the validation B1 run (which writes to `results/phase1/`).

**After the rename, switch config to validation datasets (see Part B config switch below)
before running B1 and B2. You can then run A1–A3 in parallel once the training
`corrections.jsonl` lands in `results/phase2-training/`.**

---

### A1 — Generate Training Artifacts

**Prerequisite:** `results/phase2-training/corrections.jsonl` exists.

`scripts/analyze_training.py` reads the combined Phase 2 training corrections and
produces the knowledge files that Phases 3, 4, 5, and 6 embed into their prompts.

```bash
python scripts/analyze_training.py --input results/phase2-training/corrections.jsonl --output-dir results/phase2-training/analysis
```

**Outputs produced:**

| File | Used by |
|------|---------|
| `results/phase2-training/analysis/word_pairs_llm_failures.txt` | Phases 3, 4, 5, 6 |
| `results/phase2-training/analysis/sample_classification.json` | Phase 4 |
| `results/phase2-training/analysis/word_pairs_ocr.txt` | Reference |
| `results/phase2-training/analysis/error_stats.json` | Reference |
| `results/phase2-training/analysis/summary.md` | Reference |

---

### A2 — Phase 4 Analyze-Train

Reads Phase 2 training corrections (per-dataset files) and extracts per-type LLM
performance insights (fix rates, error weaknesses). Phase 4 export will fail without these.

**Prerequisite:** A1 done. Per-dataset files at `results/phase2-training/{train_key}/corrections.jsonl`.

The pipeline automatically derives training dataset keys even when `config.yaml` lists
only validation datasets — no config switch needed to run this step.

```bash
python pipelines/run_phase4.py --mode analyze-train --source-phase phase2-training
```

**Outputs produced:**

| File | Used by |
|------|---------|
| `results/phase4/insights/PATS-A01_insights.json` | Phase 4 export |
| `results/phase4/insights/KHATT_insights.json` | Phase 4 export |

---

### A3 — Build Phase 8 RAG Index

Builds the BM25 retrieval index from Phase 2 training corrections. Phase 8 defaults
to `phase2-training` as its source — no extra flags needed after the rename.

**Prerequisite:** Per-dataset training corrections at `results/phase2-training/{train_key}/corrections.jsonl`.

```bash
python pipelines/run_phase8.py --mode build-index
```

**Outputs produced:** BM25 index files in `results/phase8/` (checked by Phase 8 export).

---

## Part B — Switch Config + Run Validation

### Config Switch

Do this immediately after A0, before running B1.
Edit `configs/config.yaml`: comment out the active training `datasets:` block and
uncomment the validation block. This is the **only config change needed** — all
artifact paths already point to training results.

```yaml
# Comment this out:
# datasets:
#   - name: "PATS-A01-Akhbar-train"
#     ...

# Uncomment this:
datasets:
  - name: "PATS-A01-Akhbar-val"
    font: "Akhbar"
    type: "PATS-A01"
    pats_split: "validation"
  - name: "PATS-A01-Andalus-val"
    ...
  # (all 8 val fonts + KHATT-validation)
```

**Verify** the active block has exactly 9 entries (8 PATS fonts + KHATT-validation)
before proceeding.

---

## Part B — Validation Phases

### Inference Environments

Every LLM phase follows: **export (local) → infer (Kaggle or Thunder) → analyze (local)**.
Choose one environment per phase run — both produce identical output files.

---

#### Kaggle (`scripts/infer.py`)

- Backend: HuggingFace Transformers on T4/P100
- Crash recovery: HF sync every N samples (`--sync-every 10`)
- Session is ephemeral — token setup must be in the same cell as inference

```python
# Kaggle notebook cell — token + inference (always together)
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_WRITE")

PROJECT_DIR = "/kaggle/working/project"
INPUT_FILE  = "/kaggle/input/<your-kaggle-dataset>/inference_input.jsonl"
HF_PATH     = "results/<phase>/corrections.jsonl"   # ← change per phase

!python {PROJECT_DIR}/scripts/infer.py --input {INPUT_FILE} --output /kaggle/working/corrections.jsonl --model Qwen/Qwen3-4B-Instruct-2507 --config {PROJECT_DIR}/configs/config.yaml --hf-repo Mohamed109/ocr-results --hf-path {HF_PATH} --sync-every 10
```

After Kaggle finishes, pull corrections to local:
```bash
python scripts/hf_sync.py pull --paths results/<phase>
```

---

#### Thunder (`scripts/infer.py`)

- Backend: HuggingFace Transformers on A100 80 GB — **faster than Kaggle**
- Writes directly to disk; sync via `hf_sync.py` before and after
- Session is persistent — no crash-recovery concern for typical runs
- Project is at `/workspace/Arabic-Post-OCR-Correction` on Thunder

**Step 1 — push inference input from local:**
```bash
python scripts/hf_sync.py push --paths results/<phase>
```

**Step 2 — on Thunder: pull, infer, push:**
```bash
python scripts/hf_sync.py pull --paths results/<phase>
python scripts/infer.py --input results/<phase>/inference_input.jsonl --output results/<phase>/corrections.jsonl --model Qwen/Qwen3-4B-Instruct-2507 --config configs/config.yaml
python scripts/hf_sync.py push --paths results/<phase>
```

**Step 3 — pull corrections back to local:**
```bash
python scripts/hf_sync.py pull --paths results/<phase>
```

> **Phase 7 (DSPy)** runs `scripts/dspy_optimize.py` instead of `scripts/infer.py` —
> see B7 for exact commands.

---

### B1 — Phase 1: Validation Baseline (Local)

Generates OCR baseline metrics and per-dataset confusion matrices for validation datasets.
Outputs to `results/phase1/` (default).

> **Phases 3 and 6 do NOT read from `results/phase1/`.** They read training-split
> confusion matrices from `results/phase1-training/` (produced in A0). The validation
> confusion matrices here are for Phase 1 reporting only.

```bash
python pipelines/run_phase1.py
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase1/{val_key}/metrics.json` | Per-dataset CER/WER baseline |
| `results/phase1/{val_key}/confusion_matrix.json` | Confusion pairs (B1 analysis only) |
| `results/phase1/{val_key}/error_taxonomy.json` | Error type breakdown |
| `results/phase1/baseline_metrics.json` | **Paper number: OCR baseline aggregated** |

---

### B2 — Phase 2: Zero-Shot LLM (Validation)

#### B2.1 — Export (Local)

```bash
python pipelines/run_phase2.py --mode export --force
```

**Output:** `results/phase2/inference_input.jsonl`

#### B2.2 — Inference

**Kaggle:** upload `results/phase2/inference_input.jsonl` as a Kaggle dataset, run the Kaggle template with `HF_PATH = "results/phase2/corrections.jsonl"`, then pull: `python scripts/hf_sync.py pull --paths results/phase2`

**Thunder:** use `<phase> = phase2` in the Thunder template above.

#### B2.5 — Analyze (Local)

```bash
python pipelines/run_phase2.py --mode analyze
```

Auto-splits the combined file into per-dataset files
(`results/phase2/{val_key}/corrections.jsonl`) required by Phase 5.

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase2/{val_key}/metrics.json` | Per-dataset CER/WER |
| `results/phase2/{val_key}/comparison_vs_phase1.json` | Delta vs OCR baseline |
| `results/phase2/{val_key}/error_changes.json` | Error type shifts |
| `results/phase2/metrics.json` | **Paper number: Zero-shot LLM aggregated** |

---

### B3 — Phase 3: OCR-Aware Prompting (Validation)

**Prerequisites:**
- A0 done — `results/phase1-training/{train_key}/confusion_matrix.json` exists
  (e.g. `results/phase1-training/PATS-A01-Akhbar-train/confusion_matrix.json`).
- A1 done — `results/phase2-training/analysis/word_pairs_llm_failures.txt` exists

#### B3.1 — Export (Local)

```bash
python pipelines/run_phase3.py --mode export --phase1-dir results/phase1-training
```

**Output:** `results/phase3/inference_input.jsonl`

#### B3.2 — Inference

**Kaggle:** upload `results/phase3/inference_input.jsonl` as a Kaggle dataset, run the Kaggle template with `HF_PATH = "results/phase3/corrections.jsonl"`, then pull: `python scripts/hf_sync.py pull --paths results/phase3`

**Thunder:** use `<phase> = phase3` in the Thunder template above.

#### B3.5 — Analyze (Local)

```bash
python pipelines/run_phase3.py --mode analyze
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase3/{val_key}/metrics.json` | Per-dataset CER/WER |
| `results/phase3/{val_key}/comparison_vs_phase2.json` | Delta vs zero-shot |
| `results/phase3/metrics.json` | **Paper number: OCR-Aware aggregated** |

---

### B4 — Phase 4: Self-Reflective (Validation)

**Prerequisites:**
- A1 done — `results/phase2-training/analysis/word_pairs_llm_failures.txt` exists
- A2 done — `results/phase4/insights/` files exist (generated from training corrections)

#### B4.1 — Export (Local)

Reads training artifacts and insights. Only validation-split datasets are exported.
Few-shot examples are drawn from `results/phase2-training/` (`phase4.few_shot.source_phase = "phase2-training"` in config).

```bash
python pipelines/run_phase4.py --mode export
```

**Output:** `results/phase4/inference_input.jsonl`

#### B4.2 — Inference

**Kaggle:** upload `results/phase4/inference_input.jsonl` as a Kaggle dataset, run the Kaggle template with `HF_PATH = "results/phase4/corrections.jsonl"`, then pull: `python scripts/hf_sync.py pull --paths results/phase4`

**Thunder:** use `<phase> = phase4` in the Thunder template above.

#### B4.5 — Analyze (Local)

```bash
python pipelines/run_phase4.py --mode analyze
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase4/{val_key}/metrics.json` | Per-dataset CER/WER |
| `results/phase4/{val_key}/comparison_vs_phase2.json` | Delta vs zero-shot |
| `results/phase4/metrics.json` | **Paper number: Self-Reflective aggregated** |

---

### B5 — Phase 5: CAMeL Morphological Validation (Local Only)

**Prerequisites:**
- B2.5 done — per-dataset corrections at `results/phase2/{val_key}/corrections.jsonl`
- A1 done — `results/phase2-training/analysis/word_pairs_llm_failures.txt` exists
- CAMeL Tools installed: `pip install camel-tools && camel_data -i morphology-db-msa-r13`

```bash
python pipelines/run_phase5.py --mode validate
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase5/{val_key}/metrics.json` | Per-dataset CER/WER after CAMeL validation |
| `results/phase5/{val_key}/comparison_vs_phase2.json` | Delta vs zero-shot |
| `results/phase5/metrics.json` | **Paper number: CAMeL Validation aggregated** |

---

### B6 — Phase 6: True Combination & Ablation

Phase 6 runs **one new inference** — `conf_self`, the true combination of all Phase 3 and Phase 4 signals — then uses Phase 3 and Phase 4 results directly as the ablation baselines.
Phase 3 = confusion-only baseline. Phase 4 = self-reflective-only baseline. No new inference needed for those.

**Prerequisites:**
- B3.5 done — `results/phase3/{val_key}/metrics.json` exists (ablation conf_only baseline)
- B4.5 done — `results/phase4/{val_key}/metrics.json` exists (ablation self_only baseline)
- A0 done — `results/phase1-training/{train_key}/confusion_matrix.json` exists
- A1 done — `results/phase2-training/analysis/word_pairs_llm_failures.txt` exists
- A2 done — `results/phase4/insights/` files exist

---

#### B6a — conf_self (True combination: Phase 3 full + Phase 4 full)

Injects ALL Phase 3 signals (confusion pairs + word-level failure examples) AND ALL Phase 4 signals (insights + word pairs + overcorrection warnings + few-shot examples) into one prompt.

**Export:**
```bash
python pipelines/run_phase6.py --combo conf_self --mode export --phase1-dir results/phase1-training
```
Output: `results/phase6/conf_self/inference_input.jsonl`

**Inference:**
Kaggle: upload `results/phase6/conf_self/inference_input.jsonl`, run template with `HF_PATH = "results/phase6/conf_self/corrections.jsonl"`, pull: `python scripts/hf_sync.py pull --paths results/phase6/conf_self`
Thunder: use `<phase> = phase6/conf_self` in the Thunder template above.

**Analyze:**
```bash
python pipelines/run_phase6.py --combo conf_self --mode analyze
```

---

#### B6b — Pick Best Combo and Set Config

Review `results/phase3/metrics.json` (conf_only), `results/phase4/metrics.json` (self_only),
and `results/phase6/conf_self/metrics.json`. Identify the lowest aggregated CER.

Edit `configs/config.yaml`:
```yaml
phase6:
  best_combo: "conf_self"   # or "phase3" or "phase4" — whichever won
```

---

#### B6c — best_camel (Best combo + CAMeL, Local Only)

```bash
python pipelines/run_phase6.py --combo best_camel --mode validate
```

**Output:** `results/phase6/best_camel/{val_key}/metrics.json`

---

#### B6d — Summarize (Ablation + Synergy)

```bash
python pipelines/run_phase6.py --mode summarize
```

**Output:** `results/phase6/metrics.json` — **Paper numbers: ablation + synergy across all combos**

Ablation uses Phase 3 as conf_only and Phase 4 as self_only. Synergy measures whether conf_self beats the sum of individual improvements.

---

### B7 — Phase 7: DSPy Prompt Optimization

DSPy automatically discovers optimal prompts using training examples. Uses
`scripts/dspy_optimize.py` on Kaggle (not `infer.py`).

**Prerequisite:** Phase 2 training corrections at `results/phase2-training/{train_key}/corrections.jsonl`.

#### B7.1 — Export (Local)

```bash
python pipelines/run_phase7.py --mode export
```

**Outputs:**

| File | Description |
|------|-------------|
| `results/phase7/dspy_trainset.jsonl` | Training examples for DSPy optimizer |
| `results/phase7/dspy_devset.jsonl` | Dev examples for metric evaluation |
| `results/phase7/inference_input.jsonl` | Full val set for inference |

#### B7.2 — Optimize + Infer

**Kaggle**

Upload all three files from `results/phase7/` as a Kaggle dataset, then:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_WRITE")

PROJECT_DIR  = "/kaggle/working/project"
KAGGLE_INPUT = "/kaggle/input/<your-dataset>"

!python {PROJECT_DIR}/scripts/dspy_optimize.py --trainset {KAGGLE_INPUT}/dspy_trainset.jsonl --devset {KAGGLE_INPUT}/dspy_devset.jsonl --input {KAGGLE_INPUT}/inference_input.jsonl --output /kaggle/working/corrections.jsonl --model Qwen/Qwen3-4B-Instruct-2507 --config {PROJECT_DIR}/configs/config.yaml --hf-repo Mohamed109/ocr-results --hf-path results/phase7/corrections.jsonl
```

Pull locally after: `python scripts/hf_sync.py pull --paths results/phase7`

**Thunder**

```bash
python scripts/hf_sync.py push --paths results/phase7
```
On Thunder:
```bash
python scripts/hf_sync.py pull --paths results/phase7
python scripts/dspy_optimize.py --trainset results/phase7/dspy_trainset.jsonl --devset results/phase7/dspy_devset.jsonl --input results/phase7/inference_input.jsonl --output results/phase7/corrections.jsonl --model Qwen/Qwen3-4B-Instruct-2507 --config configs/config.yaml
python scripts/hf_sync.py push --paths results/phase7
```
Pull back locally:
```bash
python scripts/hf_sync.py pull --paths results/phase7
```

#### B7.5 — Analyze (Local)

```bash
python pipelines/run_phase7.py --mode analyze
```

**Outputs:**

| File | Description |
|------|-------------|
| `results/phase7/{val_key}/metrics.json` | Per-dataset CER/WER |
| `results/phase7/metrics.json` | **Paper number: DSPy aggregated** |
| `results/phase7/comparison.json` | Delta vs zero-shot |
| `results/phase7/paper_tables.md` | Ready-to-use LaTeX-style table |

---

### B8 — Phase 8: RAG Retrieval-Augmented Correction

**Prerequisites:**
- A3 done — BM25 index built at `results/phase8/`
- B2.5 done — Phase 2 val corrections exist (used as baseline)

#### B8.1 — Export (Local)

```bash
python pipelines/run_phase8.py --mode export
```

**Output:** `results/phase8/inference_input.jsonl`

#### B8.2 — Inference

**Kaggle:** upload `results/phase8/inference_input.jsonl` as a Kaggle dataset, run the Kaggle template with `HF_PATH = "results/phase8/corrections.jsonl"`, then pull: `python scripts/hf_sync.py pull --paths results/phase8`

**Thunder:** use `<phase> = phase8` in the Thunder template above.

#### B8.5 — Analyze (Local)

```bash
python pipelines/run_phase8.py --mode analyze
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase8/{val_key}/metrics.json` | Per-dataset CER/WER |
| `results/phase8/{val_key}/comparison_vs_phase2.json` | Delta vs zero-shot |
| `results/phase8/metrics.json` | **Paper number: RAG aggregated** |

---

### B9 — Phase 9: Error-Signature RAG

Retrieves training samples by error structural similarity (CAMeL invalid words + confusion-matrix character profiles) rather than text similarity. Only one new inference run needed — index is built locally.

**Prerequisites:**
- A4 done — `results/phase9/index/phase9_index.json` exists
- A0 done — `results/phase1-training/{train_key}/confusion_matrix.json` exists (used to derive high-confusion chars)
- B2.5 done — Phase 2 val corrections exist (baseline)

#### B9.0 — Build Index (Local)

```bash
python pipelines/run_phase9.py --mode build-index
```

**Output:** `results/phase9/index/phase9_index.json` + `index_meta.json`

#### B9.1 — Export (Local)

```bash
python pipelines/run_phase9.py --mode export
```

**Output:** `results/phase9/inference_input.jsonl`

#### B9.2 — Inference

**Kaggle:** upload `results/phase9/inference_input.jsonl` as a Kaggle dataset, run the Kaggle template with `HF_PATH = "results/phase9/corrections.jsonl"`, then pull: `python scripts/hf_sync.py pull --paths results/phase9`

**Thunder:** use `<phase> = phase9` in the Thunder template above.

#### B9.5 — Analyze (Local)

```bash
python pipelines/run_phase9.py --mode analyze
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase9/{val_key}/metrics.json` | Per-dataset CER/WER |
| `results/phase9/{val_key}/comparison_vs_phase2.json` | Delta vs zero-shot |
| `results/phase9/metrics.json` | **Paper number: Error-Signature RAG aggregated** |

---

## Part C — Paper Results Reference

All primary paper numbers are in the aggregated `metrics.json` at the phase root.
Look for `group_aggregates_no_diacritics` for the primary reported metric.

| Phase | Description | Primary Results File |
|-------|-------------|----------------------|
| Phase 1 (val) | OCR baseline | `results/phase1/baseline_metrics.json` |
| Phase 2 (val) | Zero-shot LLM | `results/phase2/metrics.json` |
| Phase 3 (val) | OCR-Aware Prompting | `results/phase3/metrics.json` |
| Phase 4 (val) | Self-Reflective | `results/phase4/metrics.json` |
| Phase 5 (val) | CAMeL Validation | `results/phase5/metrics.json` |
| Phase 6 (val) | Combinations + Ablation | `results/phase6/metrics.json` |
| Phase 7 (val) | DSPy Optimization | `results/phase7/metrics.json` |
| Phase 8 (val) | RAG (BM25) | `results/phase8/metrics.json` |
| Phase 9 (val) | RAG (Error-Signature) | `results/phase9/metrics.json` |

**Training split results** (used for knowledge characterization, not the main comparison):

| Phase | File |
|-------|------|
| Phase 1 (train) | `results/phase1-training/baseline_metrics.json` |
| Phase 2 (train) | `results/phase2-training/metrics.json` |
| Training analysis | `results/phase2-training/analysis/summary.md` |

---

## Execution Checklist

```
Part A — rename and generate training artifacts

  [ ] A0a: mv results/phase1 results/phase1-training
  [ ] A0b: mv results/phase2 results/phase2-training

  Switch config.yaml datasets to validation:
  [ ] Comment out training datasets block
  [ ] Uncomment validation datasets block
  [ ] Verify 9 entries: 8 PATS fonts (-val) + KHATT-validation

  (once results/phase2-training/corrections.jsonl is ready)
  [ ] A1: python scripts/analyze_training.py --input results/phase2-training/corrections.jsonl --output-dir results/phase2-training/analysis
  [ ] A2: python pipelines/run_phase4.py --mode analyze-train --source-phase phase2-training
  [ ] A3: python pipelines/run_phase8.py --mode build-index
  [ ] A4: python pipelines/run_phase9.py --mode build-index

Part B — validation (B1 + B2 run immediately after A0 + config switch)

  [ ] B1: python pipelines/run_phase1.py
  [ ] B2.1: python pipelines/run_phase2.py --mode export --force
  [ ] B2.2: infer (phase2) → pull
  [ ] B2.5: python pipelines/run_phase2.py --mode analyze

  (requires A1–A3 complete)
  [ ] B3.1: python pipelines/run_phase3.py --mode export --phase1-dir results/phase1-training
  [ ] B3.2: infer (phase3) → pull
  [ ] B3.5: python pipelines/run_phase3.py --mode analyze

  [ ] B4.1: python pipelines/run_phase4.py --mode export
  [ ] B4.2: infer (phase4) → pull
  [ ] B4.5: python pipelines/run_phase4.py --mode analyze

  [ ] B5:   python pipelines/run_phase5.py --mode validate

  (requires B3.5 + B4.5 + A0–A2 done)
  [ ] B6a export: python pipelines/run_phase6.py --combo conf_self --mode export --phase1-dir results/phase1-training
  [ ] B6a infer:  infer (phase6/conf_self) → pull
  [ ] B6a analyze: python pipelines/run_phase6.py --combo conf_self --mode analyze
  [ ] B6b: set phase6.best_combo in config.yaml (phase3 | phase4 | conf_self)
  [ ] B6c: python pipelines/run_phase6.py --combo best_camel --mode validate
  [ ] B6d: python pipelines/run_phase6.py --mode summarize

  [ ] B7.1: python pipelines/run_phase7.py --mode export
  [ ] B7.2: dspy_optimize.py (phase7) → pull
  [ ] B7.5: python pipelines/run_phase7.py --mode analyze

  [ ] B8.1: python pipelines/run_phase8.py --mode export
  [ ] B8.2: infer (phase8) → pull
  [ ] B8.5: python pipelines/run_phase8.py --mode analyze

  [ ] B9.0: python pipelines/run_phase9.py --mode build-index
  [ ] B9.1: python pipelines/run_phase9.py --mode export
  [ ] B9.2: infer (phase9) → pull
  [ ] B9.5: python pipelines/run_phase9.py --mode analyze
```
