# Results Generation Guide

This document is the single source of truth for generating all paper results.
Follow every step in the order listed. Do not skip ahead.

---

## Overview

| Split | Purpose | Phases |
|-------|---------|--------|
| **Training** | Build knowledge sources (confusion matrices, LLM failure patterns, RAG index) | 1, 2, then artifact generation |
| **Validation** | Produce final evaluation numbers reported in the paper | 1–8 |

**Inference rule:** Every phase that requires an LLM follows the same three-step pattern:
`export` (local) → `infer.py` (Kaggle or Thunder) → `analyze` (local).
Phase 5 and the Phase 6 `best_camel` combo are local-only (no inference step).

---

## Current Status

| Step | Status |
|------|--------|
| Phase 1 — training split | ✓ Done |
| Phase 2 — training split (Kaggle inference) | ✓ Done |
| Rename Phase 2 folder | ✗ Pending (Part A) |
| Training artifact generation | ✗ Pending (Part A) |
| Phase 4 analyze-train | ✗ Pending (Part A) |
| Phase 8 RAG index build | ✗ Pending (Part A) |
| All validation phases | ✗ Pending (Part B) |

---

## Part A — Training Folder Rename + Artifact Generation

### A0 — Rename Training Output Folders

Rename the training Phase 2 results folder so that:
- `results/phase2-training/` holds all training corrections (what A1–A3 and config paths expect)
- `results/phase2/` is free for validation corrections

```bash
mv results/phase2 results/phase2-training
```

> **Phase 1** does not need renaming. Training and validation per-dataset subdirectories
> coexist naturally (`PATS-A01-Akhbar-train/` vs `PATS-A01-Akhbar-val/` are separate
> folders inside `results/phase1/`). The only caveat: `results/phase1/baseline_metrics.json`
> will be overwritten when Phase 1 runs on validation (step B1). Copy it somewhere safe
> before running B1 if you need the training aggregate numbers separately.

**After the rename, switch config to validation datasets (see Part B config switch below)
before running B1 and B2. You can then run A1–A3 in parallel once the training
`corrections.jsonl` lands in `results/phase2-training/`.**

---

### A1 — Generate Training Artifacts

**Prerequisite:** `results/phase2-training/corrections.jsonl` exists (paste it here once
the Kaggle training inference is complete).

`scripts/analyze_training.py` reads the combined Phase 2 training corrections and
produces the knowledge files that Phases 3, 4, 5, and 6 embed into their prompts.

```bash
python scripts/analyze_training.py \
  --input      results/phase2-training/corrections.jsonl \
  --output-dir results/phase2-training/analysis
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

**Prerequisite:** A1 done. Per-dataset files exist at
`results/phase2-training/{train_key}/corrections.jsonl`.

Note: `--source-phase phase2-training` is required because the default is `phase2`
and the folder has been renamed.

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

Builds the BM25 retrieval index from Phase 2 training corrections. Phase 8 already
defaults to `phase2-training` as its source — no extra flags needed after the rename.

**Prerequisite:** Per-dataset training corrections exist at
`results/phase2-training/{train_key}/corrections.jsonl`.

```bash
python pipelines/run_phase8.py --mode build-index
```

**Outputs produced:** BM25 index files in `results/phase8/` (checked by Phase 8 export).

---

## Part B — Switch Config + Run Validation

### Config Switch

Do this immediately after the A0 rename, before running B1.
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

!python {PROJECT_DIR}/scripts/infer.py \
    --input  {INPUT_FILE} \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --config {PROJECT_DIR}/configs/config.yaml \
    --hf-repo Mohamed109/ocr-results \
    --hf-path {HF_PATH} \
    --sync-every 10
```

After Kaggle finishes, pull corrections to local:
```bash
# On local machine
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

python scripts/infer.py \
    --input  results/<phase>/inference_input.jsonl \
    --output results/<phase>/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507 \
    --config configs/config.yaml

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

Generates OCR baseline metrics and per-dataset confusion matrices for each validation
dataset. Phase 3 reads these confusion matrices when building prompts.

```bash
python pipelines/run_phase1.py
```

**Outputs (per val dataset):**

| File | Description |
|------|-------------|
| `results/phase1/{val_key}/metrics.json` | Per-dataset CER/WER baseline |
| `results/phase1/{val_key}/confusion_matrix.json` | Character confusion pairs (used by Phase 3) |
| `results/phase1/{val_key}/error_taxonomy.json` | Error type breakdown |
| `results/phase1/baseline_metrics.json` | **Paper number: OCR baseline aggregated** |

---

### B2 — Phase 2: Zero-Shot LLM (Validation)

#### B2.1 — Export (Local)

Generates inference input for validation datasets. Use `--force` so the file is
regenerated with only validation samples (it currently contains training entries
from the training run).

```bash
python pipelines/run_phase2.py --mode export --force
```

**Output:** `results/phase2/inference_input.jsonl`

#### B2.2 — Inference

**Kaggle:** upload `results/phase2/inference_input.jsonl` as a Kaggle dataset, then run
the Kaggle template with `HF_PATH = "results/phase2/corrections.jsonl"`.
Pull locally after: `python scripts/hf_sync.py pull --paths results/phase2`

**Thunder:** use `<phase> = phase2` in the Thunder template above.

#### B2.5 — Analyze (Local)

```bash
python pipelines/run_phase2.py --mode analyze
```

This step auto-splits the combined file into per-dataset files
(`results/phase2/{val_key}/corrections.jsonl`). Those per-dataset files are
required by Phase 5.

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
- B1 done — confusion matrices exist at `results/phase1/{val_key}/confusion_matrix.json`
- A1 done — `results/phase2-training/analysis/word_pairs_llm_failures.txt` exists

#### B3.1 — Export (Local)

```bash
python pipelines/run_phase3.py --mode export
```

**Output:** `results/phase3/inference_input.jsonl`

#### B3.2 — Inference

**Kaggle:** upload `results/phase3/inference_input.jsonl` as a Kaggle dataset, then run
the Kaggle template with `HF_PATH = "results/phase3/corrections.jsonl"`.
Pull locally after: `python scripts/hf_sync.py pull --paths results/phase3`

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
- A2 done — `results/phase4/insights/` files exist

#### B4.1 — Export (Local)

Reads training artifacts and insights. Automatically skips training-split datasets;
only validation-split datasets are included in the output.

```bash
python pipelines/run_phase4.py --mode export
```

**Output:** `results/phase4/inference_input.jsonl`

#### B4.2 — Inference

**Kaggle:** upload `results/phase4/inference_input.jsonl` as a Kaggle dataset, then run
the Kaggle template with `HF_PATH = "results/phase4/corrections.jsonl"`.
Pull locally after: `python scripts/hf_sync.py pull --paths results/phase4`

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

Applies CAMeL morphological validation on top of Phase 2 validation corrections.
No Kaggle step — runs entirely on local machine.

**Prerequisites:**
- B2.5 done — per-dataset corrections split at `results/phase2/{val_key}/corrections.jsonl`
- A1 done — `results/phase2-training/analysis/word_pairs_llm_failures.txt` exists
  (for known-overcorrection revert)
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

### B6 — Phase 6: Combinations & Ablation

Phase 6 tests three LLM prompt combos (conf_only, self_only, conf_self) and one
CAMeL combo (best_camel). Each LLM combo is a separate Kaggle inference run.
The CAMeL combo is local-only.

**Prerequisites:**
- B1 done (confusion matrices for conf_only, conf_self)
- A1 done (training artifacts for self_only, conf_self)
- A2 done (insights for self_only, conf_self)

---

#### B6a — conf_only (Phase 3 prompt alone)

**Export:**
```bash
python pipelines/run_phase6.py --combo conf_only --mode export
```
Output: `results/phase6/conf_only/inference_input.jsonl`

**Inference:**
Kaggle: upload `results/phase6/conf_only/inference_input.jsonl`, run template with `HF_PATH = "results/phase6/conf_only/corrections.jsonl"`, pull: `python scripts/hf_sync.py pull --paths results/phase6/conf_only`
Thunder: `<phase> = phase6/conf_only` in the Thunder template above.

**Analyze:**
```bash
python pipelines/run_phase6.py --combo conf_only --mode analyze
```

---

#### B6b — self_only (Phase 4 prompt alone)

**Export:**
```bash
python pipelines/run_phase6.py --combo self_only --mode export
```
Output: `results/phase6/self_only/inference_input.jsonl`

**Inference:**
Kaggle: upload `results/phase6/self_only/inference_input.jsonl`, run template with `HF_PATH = "results/phase6/self_only/corrections.jsonl"`, pull: `python scripts/hf_sync.py pull --paths results/phase6/self_only`
Thunder: `<phase> = phase6/self_only` in the Thunder template above.

**Analyze:**
```bash
python pipelines/run_phase6.py --combo self_only --mode analyze
```

---

#### B6c — conf_self (Phase 3 + Phase 4 combined)

**Export:**
```bash
python pipelines/run_phase6.py --combo conf_self --mode export
```
Output: `results/phase6/conf_self/inference_input.jsonl`

**Inference:**
Kaggle: upload `results/phase6/conf_self/inference_input.jsonl`, run template with `HF_PATH = "results/phase6/conf_self/corrections.jsonl"`, pull: `python scripts/hf_sync.py pull --paths results/phase6/conf_self`
Thunder: `<phase> = phase6/conf_self` in the Thunder template above.

**Analyze:**
```bash
python pipelines/run_phase6.py --combo conf_self --mode analyze
```

---

#### B6d — Pick Best Combo and Set Config

Review `results/phase6/conf_only/metrics.json`, `self_only/metrics.json`,
`conf_self/metrics.json`. Identify the combo with the lowest aggregated CER.

Edit `configs/config.yaml`:
```yaml
phase6:
  best_combo: "conf_self"   # replace with whichever won
```

---

#### B6e — best_camel (Best combo + CAMeL, Local Only)

```bash
python pipelines/run_phase6.py --combo best_camel --mode validate
```

**Output:** `results/phase6/best_camel/{val_key}/metrics.json`

---

#### B6f — Summarize All Combos

```bash
python pipelines/run_phase6.py --mode summarize
```

**Output:** `results/phase6/metrics.json` — **Paper numbers: all 4 combos compared**

Includes ablation analysis (contribution of each component) and synergy analysis
(whether combining phases is super-additive).

---

### B7 — Phase 7: DSPy Prompt Optimization

DSPy automatically discovers optimal prompts using training examples. Phase 7 uses
`scripts/dspy_optimize.py` on Kaggle (not `infer.py`) — the optimizer runs first,
then the compiled program is applied to validation data.

**Prerequisite:** Phase 2 training corrections must exist at
`results/phase2-training/{train_key}/corrections.jsonl` (they do — training is done).

#### B7.1 — Export (Local)

Samples training and dev sets for the DSPy optimizer, and exports the full validation
set for post-optimization inference.

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

Phase 7 uses `scripts/dspy_optimize.py` (not `scripts/infer.py`) on both environments. It runs DSPy optimization first then applies the compiled program to the
full validation set.

**Kaggle**

Upload all three files from `results/phase7/` (`dspy_trainset.jsonl`, `dspy_devset.jsonl`,
`inference_input.jsonl`) as a Kaggle dataset, then:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_WRITE")

PROJECT_DIR  = "/kaggle/working/project"
KAGGLE_INPUT = "/kaggle/input/<your-dataset>"

!python {PROJECT_DIR}/scripts/dspy_optimize.py \
    --trainset {KAGGLE_INPUT}/dspy_trainset.jsonl \
    --devset   {KAGGLE_INPUT}/dspy_devset.jsonl \
    --input    {KAGGLE_INPUT}/inference_input.jsonl \
    --output   /kaggle/working/corrections.jsonl \
    --model    Qwen/Qwen3-4B-Instruct-2507 \
    --config   {PROJECT_DIR}/configs/config.yaml \
    --hf-repo  Mohamed109/ocr-results \
    --hf-path  results/phase7/corrections.jsonl
```

Pull locally after: `python scripts/hf_sync.py pull --paths results/phase7`

**Thunder**

```bash
# Local: push all phase7 files
python scripts/hf_sync.py push --paths results/phase7

# Thunder: pull, optimize+infer, push
python scripts/hf_sync.py pull --paths results/phase7

python scripts/dspy_optimize.py \
    --trainset results/phase7/dspy_trainset.jsonl \
    --devset   results/phase7/dspy_devset.jsonl \
    --input    results/phase7/inference_input.jsonl \
    --output   results/phase7/corrections.jsonl \
    --model    Qwen/Qwen3-4B-Instruct-2507 \
    --config   configs/config.yaml

python scripts/hf_sync.py push --paths results/phase7

# Local: pull results
python scripts/hf_sync.py pull --paths results/phase7
```

> Check `python scripts/dspy_optimize.py --help` for any additional flags.

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

Retrieves similar training corrections for each validation sample and embeds them
in the inference input.

```bash
python pipelines/run_phase8.py --mode export
```

**Output:** `results/phase8/inference_input.jsonl`

#### B8.2 — Inference

**Kaggle:** upload `results/phase8/inference_input.jsonl` as a Kaggle dataset, then run
the Kaggle template with `HF_PATH = "results/phase8/corrections.jsonl"`.
Pull locally after: `python scripts/hf_sync.py pull --paths results/phase8`

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
| Phase 8 (val) | RAG | `results/phase8/metrics.json` |

**Training split results** (used for knowledge characterization, not the main comparison):

| Phase | File |
|-------|------|
| Phase 1 (train) | `results/phase1/baseline_metrics.json` (already computed) |
| Phase 2 (train) | `results/phase2/metrics.json` (already computed) |
| Training analysis | `results/phase2-training/analysis/summary.md` |

---

## Execution Checklist

```
A0: mv results/phase2 results/phase2-training
    (backup results/phase1/baseline_metrics.json if you need training aggregate separately)

Switch config.yaml datasets to validation
  [ ] Comment out training datasets block
  [ ] Uncomment validation datasets block
  [ ] Verify 9 datasets listed (8 PATS fonts + KHATT-validation)

B1 + B2 can run immediately after A0 and config switch.
A1–A3 run once training corrections.jsonl lands in results/phase2-training/.
B3 onward requires A1–A3 to be complete.

Part B — validation (B1 and B2 unblock immediately)
  [ ] B1: python pipelines/run_phase1.py
  [ ] B2: export --force → infer (phase2) → pull → analyze

Part A — training artifacts (run once training corrections.jsonl is ready)
  [ ] A1: python scripts/analyze_training.py \
            --input results/phase2-training/corrections.jsonl \
            --output-dir results/phase2-training/analysis
  [ ] A2: python pipelines/run_phase4.py --mode analyze-train --source-phase phase2-training
  [ ] A3: python pipelines/run_phase8.py --mode build-index

Part B — validation continued (requires A1–A3 complete)
  [ ] B3: export → infer (phase3) → pull → analyze
  [ ] B4: export → infer (phase4) → pull → analyze
  [ ] B5: python pipelines/run_phase5.py --mode validate   (local only)
  [ ] B6a: conf_only  export → infer (phase6/conf_only)  → pull → analyze
  [ ] B6b: self_only  export → infer (phase6/self_only)  → pull → analyze
  [ ] B6c: conf_self  export → infer (phase6/conf_self)  → pull → analyze
  [ ] B6d: set phase6.best_combo in config.yaml
  [ ] B6e: python pipelines/run_phase6.py --combo best_camel --mode validate  (local only)
  [ ] B6f: python pipelines/run_phase6.py --mode summarize
  [ ] B7: export → dspy_optimize.py (phase7) → pull → analyze
  [ ] B8: export → infer (phase8) → pull → analyze
```
