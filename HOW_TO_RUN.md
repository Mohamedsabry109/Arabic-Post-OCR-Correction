# How to Run — Arabic Post-OCR Correction Pipeline

This guide covers how to run Phases 1, 2, and 3 end-to-end.

**Workflow overview:**
- **Phase 1** runs entirely locally (no LLM needed).
- **Phases 2 & 3** use a 3-stage pipeline: `export` (local) → `infer` (Kaggle/Colab) → `analyze` (local).

---

## Prerequisites

```bash
pip install -r requirements.txt
```

For remote inference (Kaggle/Colab):
- Clone the repo there, or upload the `scripts/` and `src/` directories.
- The inference script has no dependency on local data files — everything needed is embedded in the JSONL export.

---

## Phase 1 — Baseline & Error Taxonomy

**Purpose**: Measures raw Qaari OCR quality (CER/WER) and builds character-level confusion matrices used in Phase 3.

### Run all datasets

```bash
python pipelines/run_phase1.py
```

### Run a subset (for testing)

```bash
python pipelines/run_phase1.py --datasets KHATT-train KHATT-validation
```

### Smoke test (50 samples per dataset)

```bash
python pipelines/run_phase1.py --limit 50
```

### Force re-run (ignore cached results)

```bash
python pipelines/run_phase1.py --force
```

### Outputs

```
results/phase1/
└── {dataset_name}/
    ├── metrics.json            # CER, WER per dataset
    ├── confusion_matrix.json   # Character confusion pairs (used by Phase 3)
    ├── error_taxonomy.json     # Error type breakdown
    └── examples/               # Sample error examples per type
```

---

## Phase 2 — Zero-Shot LLM Correction

Phase 2 uses three stages: export on your local machine, infer on Kaggle/Colab, then analyze locally.

### Stage 1 — Export (local)

Reads OCR + GT data, writes `results/phase2/inference_input.jsonl`.

```bash
python pipelines/run_phase2.py --mode export
```

Subset export:

```bash
python pipelines/run_phase2.py --mode export --datasets KHATT-train --limit 50
```

### Stage 2 — Inference (Kaggle / Colab / local GPU)

Copy (or clone) the repo to your remote environment, then run:

```bash
# Default paths (works after cloning the repo)
python scripts/infer.py

# Explicit paths (Kaggle example)
python scripts/infer.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --model  Qwen/Qwen3-4B-Instruct-2507

# With HuggingFace sync (for resuming across Kaggle sessions)
python scripts/infer.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --hf-repo  your-username/arabic-ocr-corrections \
    --hf-token $HF_TOKEN \
    --sync-every 100

# Smoke test (50 samples from KHATT-train only)
python scripts/infer.py --datasets KHATT-train --limit 50

# Force re-run all samples
python scripts/infer.py --force
```

Copy the output `corrections.jsonl` to `results/phase2/corrections.jsonl` before running analyze.

### Stage 3 — Analyze (local)

Reads `results/phase2/corrections.jsonl`, computes CER/WER, runs error analysis.

```bash
python pipelines/run_phase2.py --mode analyze
```

Subset or smoke test:

```bash
python pipelines/run_phase2.py --mode analyze --datasets KHATT-train
python pipelines/run_phase2.py --mode analyze --limit 50
```

### Outputs

```
results/phase2/
├── inference_input.jsonl       # Export: input to infer.py
├── corrections.jsonl           # Inference output (place here before analyze)
└── {dataset_name}/
    ├── metrics.json            # CER, WER after LLM correction
    ├── comparison_vs_phase1.json
    └── error_changes.json      # Per-error-type fix/worsen counts
```

---

## Phase 3 — OCR-Aware Prompting

Phase 3 injects Qaari's character confusion statistics into the LLM system prompt.
It is an **isolated experiment**: results are compared against Phase 2 only, not Phase 1.

**Prerequisite**: Phase 1 must be complete (confusion matrices are read from `results/phase1/`).

### Stage 1 — Export (local)

Reads Phase 1 confusion matrices, embeds the confusion context in each record, and writes `results/phase3/inference_input.jsonl`.

```bash
python pipelines/run_phase3.py --mode export
```

Options:

```bash
# Use top-15 confusion pairs instead of the default 10
python pipelines/run_phase3.py --mode export --top-n 15

# Use grouped (by character category) format instead of flat list
python pipelines/run_phase3.py --mode export --format grouped_arabic

# Subset / smoke test
python pipelines/run_phase3.py --mode export --datasets KHATT-train --limit 50

# Force re-export (overwrite existing records)
python pipelines/run_phase3.py --mode export --force
```

### Stage 2 — Inference (Kaggle / Colab / local GPU)

Same script as Phase 2; `prompt_type` is embedded in the JSONL so no extra flags needed.

```bash
# Default paths
python scripts/infer.py \
    --input  results/phase3/inference_input.jsonl \
    --output results/phase3/corrections.jsonl

# Kaggle example with HF sync
python scripts/infer.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --hf-repo  your-username/arabic-ocr-corrections-phase3 \
    --hf-token $HF_TOKEN \
    --sync-every 100
```

Copy the output `corrections.jsonl` to `results/phase3/corrections.jsonl` before analyze.

### Stage 3 — Analyze (local)

Computes CER/WER, compares against Phase 2, and produces confusion impact statistics.

```bash
python pipelines/run_phase3.py --mode analyze
```

Options:

```bash
# Subset analyze
python pipelines/run_phase3.py --mode analyze --datasets KHATT-train

# Skip optional analyses (faster)
python pipelines/run_phase3.py --mode analyze --no-error-analysis
python pipelines/run_phase3.py --mode analyze --no-confusion-impact

# Point to non-default Phase 2 results
python pipelines/run_phase3.py --mode analyze --phase2-dir results/phase2
```

### Outputs

```
results/phase3/
├── inference_input.jsonl       # Export: input to infer.py (with confusion_context)
├── corrections.jsonl           # Inference output (place here before analyze)
└── {dataset_name}/
    ├── metrics.json            # CER, WER after OCR-aware correction
    ├── comparison_vs_phase2.json   # Delta vs Phase 2 (isolated comparison)
    ├── error_changes.json      # Per-error-type fix/worsen counts
    └── confusion_impact.json   # Per-injected-pair fix-rate improvement vs Phase 2
```

---

## Phase 4 — Linguistic Knowledge Enhancement

Phase 4 has three isolated sub-phases (4A, 4B, 4C).
**Prerequisite**: Phase 2 must be complete (used as baseline for all comparisons).

### Phase 4A — Rule-Augmented Prompting

#### Stage 1 — Export (local)

```bash
python pipelines/run_phase4.py --sub-phase 4a --mode export
```

Options:
```bash
# Subset / smoke test
python pipelines/run_phase4.py --sub-phase 4a --mode export --datasets KHATT-train --limit 50

# Limit to specific rule categories
python pipelines/run_phase4.py --sub-phase 4a --mode export --rule-categories taa_marbuta hamza

# Force re-export
python pipelines/run_phase4.py --sub-phase 4a --mode export --force
```

#### Stage 2 — Inference (Kaggle / Colab / local GPU)

```bash
python scripts/infer.py \
    --input  results/phase4a/inference_input.jsonl \
    --output results/phase4a/corrections.jsonl
```

#### Stage 3 — Analyze (local)

```bash
python pipelines/run_phase4.py --sub-phase 4a --mode analyze
```

### Phase 4B — Few-Shot Prompting (QALB)

#### Stage 1 — Export (local)

```bash
python pipelines/run_phase4.py --sub-phase 4b --mode export
```

Options:
```bash
# Use 10 examples instead of default 5
python pipelines/run_phase4.py --sub-phase 4b --mode export --num-examples 10

# Use random selection instead of diverse
python pipelines/run_phase4.py --sub-phase 4b --mode export --selection random

# Subset / smoke test
python pipelines/run_phase4.py --sub-phase 4b --mode export --datasets KHATT-train --limit 50
```

#### Stage 2 — Inference (Kaggle / Colab / local GPU)

```bash
python scripts/infer.py \
    --input  results/phase4b/inference_input.jsonl \
    --output results/phase4b/corrections.jsonl
```

#### Stage 3 — Analyze (local)

```bash
python pipelines/run_phase4.py --sub-phase 4b --mode analyze
```

### Phase 4C — CAMeL Tools Validation (local only)

Phase 4C is a **local post-processing step** — no GPU or Kaggle needed.
It reads Phase 2 corrections and applies morphological revert strategy.

```bash
python pipelines/run_phase4.py --sub-phase 4c --mode validate
```

Options:
```bash
# Subset
python pipelines/run_phase4.py --sub-phase 4c --mode validate --datasets KHATT-train

# Point to non-default Phase 2 results
python pipelines/run_phase4.py --sub-phase 4c --mode validate --phase2-dir results/phase2
```

### Outputs

```
results/phase4a/
├── inference_input.jsonl       # Export: input to infer.py (with rules_context)
├── corrections.jsonl           # Inference output (place here before analyze)
└── {dataset_name}/
    ├── metrics.json
    ├── comparison_vs_phase2.json
    └── error_changes.json

results/phase4b/
├── inference_input.jsonl       # Export: input to infer.py (with examples_context)
├── corrections.jsonl           # Inference output (place here before analyze)
└── {dataset_name}/
    ├── metrics.json
    ├── comparison_vs_phase2.json
    └── error_changes.json

results/phase4c/
└── {dataset_name}/
    ├── corrections.jsonl       # Post-validation corrected texts
    ├── metrics.json
    ├── comparison_vs_phase2.json
    └── validation_stats.json   # Word-level revert statistics
```

---

## Phase 5 — RAG with OpenITI Corpus

Phase 5 tests whether retrieving similar correct Arabic sentences from the OpenITI corpus helps the LLM. It has **four** stages: `build` (one-time) → `export` → inference → `analyze`.

**Prerequisite**: `data/OpenITI/` must be populated. Install dependencies:

```bash
pip install sentence-transformers faiss-cpu
```

### Stage 0 — Build corpus + FAISS index (one-time, local)

Extracts 200K sentences from OpenITI (stratified by era), embeds them, and saves a FAISS index. Takes 20–60 min on first run; subsequent runs reuse the index.

```bash
python pipelines/run_phase5.py --mode build

# Smoke test (1000 sentences only — fast)
python pipelines/run_phase5.py --mode build --max-sentences 1000

# Force rebuild even if index exists
python pipelines/run_phase5.py --mode build --force
```

### Stage 1 — Export (local)

Retrieves top-3 similar sentences for each OCR sample and writes `results/phase5/inference_input.jsonl`.

```bash
python pipelines/run_phase5.py --mode export

# Subset / smoke test
python pipelines/run_phase5.py --mode export --datasets KHATT-train --limit 50

# Use top-5 instead of default top-3
python pipelines/run_phase5.py --mode export --top-k 5

# Force re-export
python pipelines/run_phase5.py --mode export --force
```

### Stage 2 — Inference (Kaggle / Colab / local GPU)

Same script as all other phases. `prompt_type` is embedded in the JSONL.

```bash
python scripts/infer.py \
    --input  results/phase5/inference_input.jsonl \
    --output results/phase5/corrections.jsonl

# With HF sync
python scripts/infer.py \
    --input  /kaggle/input/your-dataset/inference_input.jsonl \
    --output /kaggle/working/corrections.jsonl \
    --hf-repo  your-username/arabic-ocr-corrections-phase5 \
    --hf-token $HF_TOKEN \
    --sync-every 100
```

Copy the output `corrections.jsonl` to `results/phase5/corrections.jsonl` before running analyze.

### Stage 3 — Analyze (local)

Reads `results/phase5/corrections.jsonl`, computes CER/WER, comparison vs Phase 2, retrieval quality statistics.

```bash
python pipelines/run_phase5.py --mode analyze

# Subset analyze
python pipelines/run_phase5.py --mode analyze --datasets KHATT-train

# Skip error analysis (faster)
python pipelines/run_phase5.py --mode analyze --no-error-analysis

# Point to non-default Phase 2 results
python pipelines/run_phase5.py --mode analyze --phase2-dir results/phase2
```

### Outputs

```
results/phase5/
├── corpus.jsonl                  # Extracted OpenITI sentences
├── faiss.index                   # FAISS binary index
├── faiss.index.sentences.jsonl   # Sentence lookup list
├── inference_input.jsonl         # Export: one record per OCR sample
├── corrections.jsonl             # Inference output (place here before analyze)
├── {dataset_name}/
│   ├── metrics.json
│   ├── comparison_vs_phase2.json
│   ├── error_changes.json
│   └── retrieval_analysis.json   # Retrieval quality statistics
├── metrics.json                  # Aggregated across all datasets
├── comparison.json               # Aggregated comparison vs Phase 2
└── report.md
```

---

## Common Patterns

### Run all three phases end-to-end (local inference)

```bash
python pipelines/run_phase1.py
python pipelines/run_phase2.py --mode export
python scripts/infer.py --input results/phase2/inference_input.jsonl --output results/phase2/corrections.jsonl
python pipelines/run_phase2.py --mode analyze
python pipelines/run_phase3.py --mode export
python scripts/infer.py --input results/phase3/inference_input.jsonl --output results/phase3/corrections.jsonl
python pipelines/run_phase3.py --mode analyze
```

### Quick smoke test across all phases

```bash
python pipelines/run_phase1.py --limit 50 --datasets KHATT-train
python pipelines/run_phase2.py --mode export --limit 50 --datasets KHATT-train
python scripts/infer.py --datasets KHATT-train --limit 50
python pipelines/run_phase2.py --mode analyze --datasets KHATT-train
python pipelines/run_phase3.py --mode export --limit 50 --datasets KHATT-train
python scripts/infer.py --input results/phase3/inference_input.jsonl --output results/phase3/corrections.jsonl --datasets KHATT-train --limit 50
python pipelines/run_phase3.py --mode analyze --datasets KHATT-train
```

### Resume after interruption

All stages support automatic resume — they skip already-processed samples.
Use `--force` to re-run from scratch.

```bash
# Resume Phase 1 (skips datasets with existing metrics.json)
python pipelines/run_phase1.py

# Resume inference (skips sample_ids already in corrections.jsonl)
python scripts/infer.py

# Resume with HF sync (merges remote progress before continuing)
python scripts/infer.py --hf-repo user/repo --hf-token $HF_TOKEN
```

---

## Configuration

All settings are in `configs/config.yaml`. Key knobs:

| Section | Key | Effect |
|---------|-----|--------|
| `model` | `name` | Model to load (override with `--model` flag) |
| `phase1` | `top_confusions_n` | How many confusion pairs to store |
| `phase3` | `top_n` | How many pairs to inject into prompts (default: 10) |
| `phase3` | `format_style` | `flat_arabic` or `grouped_arabic` |
| `phase3` | `min_substitutions` | Sparsity threshold for per-dataset matrix |
| `processing` | `limit_per_dataset` | Global sample limit (overridden by `--limit`) |

---

## Troubleshooting

**`FileNotFoundError: inference_input.jsonl`**
Run the export stage first: `python pipelines/run_phase3.py --mode export`

**`FileNotFoundError: confusion_matrix.json`**
Run Phase 1 first: `python pipelines/run_phase1.py`

**`corrections.jsonl not found` during analyze**
Copy the file from Kaggle/Colab to `results/phase3/corrections.jsonl` before running analyze.

**Arabic text garbled in terminal (Windows)**
This is a display issue only — files are always written with UTF-8. Safe to ignore.

**Empty `confusion_context` in export log**
The dataset's confusion matrix either doesn't exist or has fewer than 200 substitutions. The script falls back to a pooled matrix. If the pooled matrix is also missing, the record uses `prompt_type: zero_shot` (identical to Phase 2 behaviour).
