# End-to-End Smoke Test — PATS-A01-Akhbar-train, limit 200

Tests all 6 phases with a single dataset and small sample count to verify the full
pipeline before running on all data.

**Dataset**: `PATS-A01-Akhbar-train`
**Limit**: 200 samples
**Pattern**: export (local) → infer (Kaggle) → analyze (local)

---

## Phase 1 — Baseline (local only)

```bash
venv/Scripts/python pipelines/run_phase1.py \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --no-camel
```

---

## Phase 2 — Zero-Shot LLM

**Export:**
```bash
venv/Scripts/python pipelines/run_phase2.py \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --mode export
```

**Kaggle** — input/output:
```
--input  results/phase2/inference_input.jsonl
--output results/phase2/corrections.jsonl
```

**Analyze:**
```bash
venv/Scripts/python pipelines/run_phase2.py \
    --datasets PATS-A01-Akhbar-train \
    --mode analyze
```

---

## Phase 3 — OCR-Aware Prompting

> Note: 200 samples will fall below the `min_substitutions: 200` threshold, so Phase 3
> falls back to the pooled (sparse) confusion matrix. Pipeline completes fine, hints will be thin.

**Export:**
```bash
venv/Scripts/python pipelines/run_phase3.py \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --mode export
```

**Kaggle** — input/output:
```
--input  results/phase3/inference_input.jsonl
--output results/phase3/corrections.jsonl
```

**Analyze:**
```bash
venv/Scripts/python pipelines/run_phase3.py \
    --datasets PATS-A01-Akhbar-train \
    --mode analyze
```

---

## Phase 4A — Rule-Augmented

**Export:**
```bash
venv/Scripts/python pipelines/run_phase4.py \
    --sub-phase 4a \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --mode export
```

**Kaggle** — input/output:
```
--input  results/phase4a/inference_input.jsonl
--output results/phase4a/corrections.jsonl
```

**Analyze:**
```bash
venv/Scripts/python pipelines/run_phase4.py \
    --sub-phase 4a \
    --datasets PATS-A01-Akhbar-train \
    --mode analyze
```

---

## Phase 4B — Few-Shot

**Export:**
```bash
venv/Scripts/python pipelines/run_phase4.py \
    --sub-phase 4b \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --mode export
```

**Kaggle** — input/output:
```
--input  results/phase4b/inference_input.jsonl
--output results/phase4b/corrections.jsonl
```

**Analyze:**
```bash
venv/Scripts/python pipelines/run_phase4.py \
    --sub-phase 4b \
    --datasets PATS-A01-Akhbar-train \
    --mode analyze
```

---

## Phase 4C — CAMeL Validation (local only, no Kaggle)

Post-processes Phase 2 corrections. No inference step.

```bash
venv/Scripts/python pipelines/run_phase4.py \
    --sub-phase 4c \
    --datasets PATS-A01-Akhbar-train \
    --mode validate
```

---

## Phase 4D — Self-Reflective

**Step 1 — analyze-train** (reads Phase 2 train corrections, generates insights):
```bash
venv/Scripts/python pipelines/run_phase4d.py \
    --mode analyze-train
```

**Step 2 — export:**
```bash
venv/Scripts/python pipelines/run_phase4d.py \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --mode export
```

**Kaggle** — input/output:
```
--input  results/phase4d/inference_input.jsonl
--output results/phase4d/corrections.jsonl
```

**Analyze:**
```bash
venv/Scripts/python pipelines/run_phase4d.py \
    --datasets PATS-A01-Akhbar-train \
    --mode analyze
```

---

## Phase 5 — RAG

**Step 0 — build index** (one-time, skip if already built):
```bash
venv/Scripts/python pipelines/run_phase5.py \
    --mode build \
    --max-sentences 10000
```

**Export:**
```bash
venv/Scripts/python pipelines/run_phase5.py \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --mode export
```

**Kaggle** — input/output:
```
--input  results/phase5/inference_input.jsonl
--output results/phase5/corrections.jsonl
```

**Analyze:**
```bash
venv/Scripts/python pipelines/run_phase5.py \
    --datasets PATS-A01-Akhbar-train \
    --mode analyze
```

---

## Phase 6 — Combinations & Ablation

### Export all inference combos

```bash
venv/Scripts/python pipelines/run_phase6.py \
    --datasets PATS-A01-Akhbar-train \
    --limit 200 \
    --combo all \
    --mode export
```

### Kaggle — infer each combo

200 samples x 12 combos = 2400 samples total, fits in one Kaggle session run sequentially.

```
--input  results/phase6/pair_conf_rules/inference_input.jsonl
--output results/phase6/pair_conf_rules/corrections.jsonl

--input  results/phase6/pair_conf_fewshot/inference_input.jsonl
--output results/phase6/pair_conf_fewshot/corrections.jsonl

--input  results/phase6/pair_conf_rag/inference_input.jsonl
--output results/phase6/pair_conf_rag/corrections.jsonl

--input  results/phase6/pair_rules_fewshot/inference_input.jsonl
--output results/phase6/pair_rules_fewshot/corrections.jsonl

--input  results/phase6/full_prompt/inference_input.jsonl
--output results/phase6/full_prompt/corrections.jsonl

--input  results/phase6/abl_no_confusion/inference_input.jsonl
--output results/phase6/abl_no_confusion/corrections.jsonl

--input  results/phase6/abl_no_rules/inference_input.jsonl
--output results/phase6/abl_no_rules/corrections.jsonl

--input  results/phase6/abl_no_fewshot/inference_input.jsonl
--output results/phase6/abl_no_fewshot/corrections.jsonl

--input  results/phase6/abl_no_rag/inference_input.jsonl
--output results/phase6/abl_no_rag/corrections.jsonl

--input  results/phase6/self_reflective/inference_input.jsonl
--output results/phase6/self_reflective/corrections.jsonl

--input  results/phase6/pair_self_conf/inference_input.jsonl
--output results/phase6/pair_self_conf/corrections.jsonl

--input  results/phase6/full_with_self/inference_input.jsonl
--output results/phase6/full_with_self/corrections.jsonl
```

### Analyze all inference combos

```bash
venv/Scripts/python pipelines/run_phase6.py \
    --datasets PATS-A01-Akhbar-train \
    --combo all \
    --mode analyze
```

### CAMeL combos (local only, no Kaggle)

First set `phase6.pair_best` in `configs/config.yaml` to whichever combo had the best
CER from the analyze step above (e.g. `full_prompt`), then:

```bash
venv/Scripts/python pipelines/run_phase6.py \
    --datasets PATS-A01-Akhbar-train \
    --combo full_system \
    --mode validate

venv/Scripts/python pipelines/run_phase6.py \
    --datasets PATS-A01-Akhbar-train \
    --combo pair_best_camel \
    --mode validate
```

### Final summary

```bash
venv/Scripts/python pipelines/run_phase6.py --mode summarize
```
