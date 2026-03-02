# Local Development Guide — Running Without a GPU

This guide explains how to run the entire pipeline end-to-end on your local
machine without Kaggle or Colab. Use this to verify all pipeline stages work,
tune prompts and parameters, and iterate quickly before spending GPU compute.

---

## How it works

The `MockCorrector` backend returns OCR text unchanged (identity correction).
It loads no model and runs instantly. Every pipeline stage — export, infer,
analyze, summarize — produces valid output files with correct structure.

**Expected metrics**: CER/WER will equal the OCR baseline (no improvement),
because no actual correction is performed. This is intentional — you are
testing the pipeline, not the model.

When you switch to `backend: "transformers"` on Kaggle/Colab, the only thing
that changes is the `corrected_text` field in `corrections.jsonl`. All analysis
and reporting code is identical.

---

## Quick start (all phases, single command each)

```bash
# 1. Phase 1 — baseline analysis (always local, no LLM)
python pipelines/run_phase1.py --config configs/config_dev.yaml

# 2. Phase 2 — zero-shot (full mode: export + mock infer + analyze in one step)
python pipelines/run_phase2.py --config configs/config_dev.yaml --mode full

# 3. Phase 3 — OCR-aware prompting
python pipelines/run_phase3.py --config configs/config_dev.yaml --mode export
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase3/inference_input.jsonl \
    --output results/phase3/corrections.jsonl
python pipelines/run_phase3.py --config configs/config_dev.yaml --mode analyze

# 4. Phase 4A — rule-augmented
python pipelines/run_phase4.py --config configs/config_dev.yaml --sub-phase 4a --mode export
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase4a/inference_input.jsonl \
    --output results/phase4a/corrections.jsonl
python pipelines/run_phase4.py --config configs/config_dev.yaml --sub-phase 4a --mode analyze

# 5. Phase 4B — few-shot
python pipelines/run_phase4.py --config configs/config_dev.yaml --sub-phase 4b --mode export
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase4b/inference_input.jsonl \
    --output results/phase4b/corrections.jsonl
python pipelines/run_phase4.py --config configs/config_dev.yaml --sub-phase 4b --mode analyze

# 6. Phase 4C — CAMeL validation (local post-processing, no infer.py needed)
python pipelines/run_phase4.py --config configs/config_dev.yaml --sub-phase 4c --mode validate

# 7. Phase 4D — self-reflective
python pipelines/run_phase4d.py --config configs/config_dev.yaml --mode analyze-train
python pipelines/run_phase4d.py --config configs/config_dev.yaml --mode export
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase4d/inference_input.jsonl \
    --output results/phase4d/corrections.jsonl
python pipelines/run_phase4d.py --config configs/config_dev.yaml --mode analyze

# 8. Phase 5 — RAG (build index first — takes a few minutes even with 1000 sentences)
python pipelines/run_phase5.py --config configs/config_dev.yaml --mode build
python pipelines/run_phase5.py --config configs/config_dev.yaml --mode export
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase5/inference_input.jsonl \
    --output results/phase5/corrections.jsonl
python pipelines/run_phase5.py --config configs/config_dev.yaml --mode analyze

# 9. Phase 6 — combinations (export all 12 combos at once)
python pipelines/run_phase6.py --config configs/config_dev.yaml --mode export --combo all
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase6/full_prompt/inference_input.jsonl \
    --output results/phase6/full_prompt/corrections.jsonl
# ... repeat infer for each combo, or use the loop below
python pipelines/run_phase6.py --config configs/config_dev.yaml --mode analyze --combo all
python pipelines/run_phase6.py --config configs/config_dev.yaml --mode summarize
```

---

## Faster: loop all Phase 6 combos in one command

```bash
for combo in pair_conf_rules pair_conf_fewshot pair_conf_rag pair_rules_fewshot \
             full_prompt abl_no_confusion abl_no_rules abl_no_fewshot abl_no_rag \
             self_reflective pair_self_conf full_with_self; do
    python scripts/infer.py \
        --config configs/config_dev.yaml \
        --backend mock \
        --input  results/phase6/${combo}/inference_input.jsonl \
        --output results/phase6/${combo}/corrections.jsonl
done
```

---

## Development workflow

### Iterate on a single phase

After changing a prompt in `prompt_builder.py` or logic in a pipeline, re-run
only the affected phase. You don't need to re-run earlier phases.

```bash
# Example: tweak Phase 3 prompt and re-test
python pipelines/run_phase3.py --config configs/config_dev.yaml --mode export --force
python scripts/infer.py --config configs/config_dev.yaml --backend mock \
    --input results/phase3/inference_input.jsonl \
    --output results/phase3/corrections.jsonl --force
python pipelines/run_phase3.py --config configs/config_dev.yaml --mode analyze --force
```

`--force` skips the resume check and re-processes all samples.

### Add more datasets to the dev loop

Edit `configs/config_dev.yaml` under `datasets:` to add more fonts or splits.
The full list is in `configs/config.yaml`. Adding 1-2 more fonts exercises
more code paths (e.g. different confusion matrices per font).

### Use a different sample limit

Override the per-dataset limit on the command line:

```bash
python pipelines/run_phase2.py --config configs/config_dev.yaml --mode full --limit 5
```

This overrides `processing.limit_per_dataset` in the dev config. Use `5` for
a lightning-fast sanity check, `50` for a more thorough test.

### Test specific datasets only

```bash
python pipelines/run_phase1.py --config configs/config_dev.yaml \
    --datasets PATS-A01-Akhbar-val
```

---

## What each stage produces

After a full dev run you should have:

```
results/
├── phase1/
│   ├── PATS-A01-Akhbar-train/  metrics.json, confusion_matrix.json, error_taxonomy.json
│   ├── PATS-A01-Akhbar-val/    metrics.json, confusion_matrix.json, error_taxonomy.json
│   ├── KHATT-train/            metrics.json, confusion_matrix.json, error_taxonomy.json
│   └── KHATT-validation/       metrics.json, confusion_matrix.json, error_taxonomy.json
├── phase2/
│   ├── inference_input.jsonl
│   ├── PATS-A01-Akhbar-train/  corrections.jsonl, metrics.json, comparison_vs_phase1.json
│   └── ...
├── phase3/   (same structure)
├── phase4a/  (same structure)
├── phase4b/  (same structure)
├── phase4c/  (same structure, no inference_input.jsonl)
├── phase4d/
│   ├── insights/  PATS-A01_insights.json, KHATT_insights.json
│   └── ...
├── phase5/
│   ├── corpus.jsonl, faiss.index, faiss.index.sentences.jsonl
│   └── ...
└── phase6/
    ├── full_prompt/  inference_input.jsonl, corrections.jsonl, {dataset}/...
    ├── ...
    ├── combinations_summary.json
    ├── ablation_summary.json
    └── paper_tables.md
```

All JSON files have the same schema as production. You can inspect them to
verify the pipeline is building prompts, parsing outputs, and computing metrics
correctly.

---

## Switching to real inference on Kaggle/Colab

Once you're satisfied the pipeline is correct, switch to real inference:

### Step 1 — Re-export with production config

```bash
# Export all phases using production config (all 18 datasets, no limit)
python pipelines/run_phase2.py  --mode export --force
python pipelines/run_phase3.py  --mode export --force
python pipelines/run_phase4.py  --sub-phase 4a --mode export --force
python pipelines/run_phase4.py  --sub-phase 4b --mode export --force
python pipelines/run_phase4d.py --mode analyze-train --force
python pipelines/run_phase4d.py --mode export --force
python pipelines/run_phase5.py  --mode build   # one-time, reuses if index exists
python pipelines/run_phase5.py  --mode export --force
python pipelines/run_phase6.py  --mode export --combo all --force
```

### Step 2 — Upload each `inference_input.jsonl` to Kaggle/Colab

See `docs/Kaggle_Colab_Guide.md` for upload and run instructions.

### Step 3 — Download `corrections.jsonl` and run analyze

```bash
python pipelines/run_phase2.py  --mode analyze
python pipelines/run_phase3.py  --mode analyze
# ...etc
```

### Optional — clean dev results before production run

Dev runs write mock corrections to the same `results/` directory.
If you want a clean slate before production:

```bash
# Remove mock corrections (keeps Phase 1 data which is real)
rm -rf results/phase2 results/phase3 results/phase4a results/phase4b \
       results/phase4c results/phase4d results/phase5 results/phase6
```

Phase 1 data is safe to keep — it doesn't depend on the LLM backend.

---

## Troubleshooting

**`MockCorrector` output looks correct but CER is unchanged from Phase 1**
This is correct. MockCorrector returns OCR text unchanged, so the corrected CER
equals the OCR CER. This proves the pipeline runs end-to-end without errors.

**Phase 3 export warns "pooled confusion matrix used"**
With 20 samples per dataset, per-dataset confusion matrices are sparse.
`config_dev.yaml` sets `min_substitutions: 1` to allow sparse matrices.
The pooled fallback is also tested, which is a valid code path.

**Phase 4D `analyze-train` produces empty insights**
With 20 samples there may be too few errors per ErrorType. `config_dev.yaml`
sets `min_sample_size: 1` to capture statistics even from small samples.
If insights are still empty, the Phase 4D export will fall back to zero-shot
prompts, which is the correct fallback behavior to verify.

**Phase 5 `build` fails with embedding model errors**
`sentence-transformers` is required. Install it:
```bash
pip install sentence-transformers faiss-cpu
```
The dev config uses only 1000 sentences from OpenITI, so the build completes
in under a minute.

**Phase 6 summarize has no stats (skipped)**
Statistical tests require at least 10 paired samples per combo.
With 20 samples per dataset × 4 datasets = 80 samples, tests will run.
If you used `--limit 5`, increase it to at least 10 for the summarize step.

**CAMeL Tools not installed (Phase 4C warning)**
Phase 4C degrades gracefully if CAMeL is missing. To install:
```bash
pip install camel-tools
camel_data -i morphology-db-msa-r13
```

**I want to test with a real (small) model locally**
Use a tiny HuggingFace model instead of mock:
```bash
# Override backend and model on the command line
python scripts/infer.py \
    --backend transformers \
    --model "Qwen/Qwen3-0.6B-Instruct" \
    --input  results/phase2/inference_input.jsonl \
    --output results/phase2/corrections.jsonl
```
A 0.6B model runs on CPU (slowly, ~5-30 sec/sample) but produces real corrections.

---

## Config reference: dev vs production

| Setting | `config_dev.yaml` | `config.yaml` (production) |
|---------|-------------------|---------------------------|
| `model.backend` | `"mock"` | `"transformers"` |
| Datasets | 4 (Akhbar + KHATT) | 18 (all fonts + KHATT) |
| `limit_per_dataset` | 20 | null (unlimited) |
| `phase5.corpus.max_sentences` | 1000 | 200000 |
| `phase5.index.batch_size` | 64 | 256 |
| `phase6.stats.n_bootstrap` | 100 | 1000 |
| `phase*.max_retries` | 0 | 2 |
| `phase1.min_confusion_count` | 1 | 2 |
| `phase3.min_substitutions` | 1 | 200 |
| `phase4d.insights.min_sample_size` | 1 | 10 |
