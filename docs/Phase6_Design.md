# Phase 6: Combinations & Ablation Study — Design Document

## 1. Overview

### 1.1 Purpose

Phase 6 is the synthesis phase of the research. Where Phases 3–5 test each knowledge source
in isolation, Phase 6 tests them together, answering three questions:

1. **Synergy**: Do combinations outperform the sum of their isolated improvements?
2. **Optimality**: What is the best-performing combination?
3. **Necessity**: What does each component contribute to the full system (ablation)?

| Aspect | Detail |
|--------|--------|
| **Research Questions** | What combination is optimal? Which components synergize? What does each contribute? |
| **Comparison** | Each combo vs Phase 2 (primary); also vs each other |
| **Prompt Version** | `p6v1` (combined prompt) |
| **New Method** | `PromptBuilder.build_combined()` |
| **New Pipeline** | `pipelines/run_phase6.py` |
| **New Analysis Module** | `src/analysis/stats_tester.py` (StatsTester) |

### 1.2 Design Principle: Hierarchical, Not Exhaustive

With 5 components, 2^5 = 32 combinations are possible. We test a principled subset:

| Level | Experiments | Type |
|-------|-------------|------|
| **Level 1** | Phases 3–5 (done) | Isolated individual effects |
| **Level 2** | 5 pre-defined pairs | Pairwise synergy |
| **Level 3** | Full prompt system | All 4 prompt components |
| **Level 4** | Full system | Full prompt + CAMeL post-processing |
| **Level 5** | 4 ablations (−1 each) | Component necessity |

> **Comparison**: Every experiment compares against **Phase 2 (zero-shot baseline)**.
> Ablations additionally compare against the full system.

### 1.3 Downstream Use

- **Paper**: The final results table (Table X) comparing all phases and key combinations
- **Practical guidance**: Minimal effective combination for Arabic OCR post-processing
- **Publication**: `paper_tables.md` + figures for direct inclusion

---

## 2. Component Taxonomy

### 2.1 Categories

Five components from Phases 3–5 form the building blocks of Phase 6:

| Component | Source Phase | Mechanism | Context Field | Per-Sample? |
|-----------|-------------|-----------|---------------|-------------|
| **Confusion** (C) | Phase 3 | Inject OCR error pairs | `confusion_context` | No (global per dataset) |
| **Rules** (R) | Phase 4A | Inject Arabic spelling rules | `rules_context` | No (global) |
| **Few-Shot** (F) | Phase 4B | Inject correction examples | `examples_context` | No (global) |
| **RAG** (G) | Phase 5 | Inject similar correct sentences | `retrieval_context` | **Yes** (per-sample) |
| **CAMeL** (M) | Phase 4C | Post-process LLM output | N/A | Yes (per-word) |

### 2.2 How Components Combine

**Prompt-based components** (C, R, F, G) modify the LLM system prompt. They compete for
context window but are otherwise independent.

**Post-processing** (CAMeL) applies after LLM output. It is additive — can be layered on
top of any prompt-based combination without changing the inference step.

```
OCR Text
   │
   ▼
┌─────────────────────────────────────────────┐
│              LLM System Prompt              │
│  ┌─────────┐ ┌────────┐ ┌─────────────────┐ │
│  │Confusion│ │  Rules │ │   Few-Shot Exs  │ │
│  │ (opt.)  │ │ (opt.) │ │     (opt.)      │ │
│  └─────────┘ └────────┘ └─────────────────┘ │
│  ┌────────────────────────────────────────┐  │
│  │       Retrieved Sentences (opt.)       │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
   │
   ▼  (LLM corrects OCR text)
   │
   ▼
┌─────────────────────────────────────────────┐
│     CAMeL Morphological Validation (opt.)   │
│  Revert invalid-word corrections to OCR    │
└─────────────────────────────────────────────┘
   │
   ▼
Corrected Text
```

---

## 3. Experiment Matrix

### 3.1 Full Experiment List

| Combo ID | Components | Type | Inference Needed? |
|----------|------------|------|-------------------|
| `pair_conf_rules` | C + R | Pair | New Kaggle run |
| `pair_conf_fewshot` | C + F | Pair | New Kaggle run |
| `pair_conf_rag` | C + G | Pair | New Kaggle run |
| `pair_rules_fewshot` | R + F | Pair | New Kaggle run |
| `pair_best_camel` | Best pair + M | Pair (CAMeL) | Local only (post-process best pair) |
| `full_prompt` | C + R + F + G | Full (no CAMeL) | New Kaggle run |
| `full_system` | C + R + F + G + M | Full | Local only (post-process full_prompt) |
| `abl_no_confusion` | R + F + G | Ablation | New Kaggle run |
| `abl_no_rules` | C + F + G | Ablation | New Kaggle run |
| `abl_no_fewshot` | C + R + G | Ablation | New Kaggle run |
| `abl_no_rag` | C + R + F | Ablation | New Kaggle run |

**Key observation**: `abl_no_camel` = `full_prompt` — these are identical. No extra run needed.

**Total**: 9 new Kaggle inference runs + 2 local CAMeL post-processing steps.

> **Selecting `pair_best_camel`**: After reviewing pair results, pick the top-performing
> pair. If scores are close, prefer `pair_conf_rules` (most interpretable). Update
> `config.yaml phase6.pair_best` before running `--combo pair_best_camel --mode validate`.

### 3.2 Inference Run Summary

**Kaggle inference runs** (each needs: export → Kaggle → analyze):
1. `pair_conf_rules`
2. `pair_conf_fewshot`
3. `pair_conf_rag`
4. `pair_rules_fewshot`
5. `full_prompt`
6. `abl_no_confusion`
7. `abl_no_rules`
8. `abl_no_fewshot`
9. `abl_no_rag`

**Local post-processing only**:
1. `pair_best_camel` — load best pair's `corrections.jsonl`, apply CAMeL
2. `full_system` — load `full_prompt/corrections.jsonl`, apply CAMeL

---

## 4. `build_combined()` — Prompt Builder Design

### 4.1 Combined System Prompt

**Version**: `COMBINED_PROMPT_VERSION = "p6v1"`

The prompt builds context sections dynamically — only non-empty contexts are included:

```python
COMBINED_SECTION_CONFUSION: str = (
    "أولاً: أخطاء شائعة في هذا النظام:\n{confusion_context}"
)

COMBINED_SECTION_RULES: str = (
    "ثانياً: قواعد إملائية مهمة:\n{rules_context}"
)

COMBINED_SECTION_EXAMPLES: str = (
    "ثالثاً: أمثلة على التصحيح:\n{examples_context}"
)

COMBINED_SECTION_RETRIEVAL: str = (
    "رابعاً: نصوص مرجعية صحيحة مشابهة:\n{retrieval_context}"
)

COMBINED_HEADER: str = (
    "أنت مصحح نصوص عربية متخصص. استخدم المعلومات التالية لتصحيح أخطاء التعرف الضوئي:\n\n"
)

COMBINED_FOOTER: str = (
    "\n\nصحح النص التالي. أعد النص المصحح فقط بدون أي شرح أو تعليق."
)
```

### 4.2 `build_combined()` Signature

```python
def build_combined(
    self,
    ocr_text: str,
    confusion_context: str = "",
    rules_context: str = "",
    examples_context: str = "",
    retrieval_context: str = "",
) -> list[dict]:
    """Build a combined correction prompt including any subset of context types.

    Includes only non-empty contexts, in a fixed order:
      1. Confusion context (OCR error pairs)
      2. Rules context (Arabic spelling rules)
      3. Examples context (few-shot correction pairs)
      4. Retrieval context (similar correct sentences from OpenITI)

    Falls back to zero-shot if all context strings are empty or whitespace.

    Args:
        ocr_text: OCR text to correct.
        confusion_context: Pre-formatted confusion matrix context (Phase 3).
        rules_context: Pre-formatted rules context (Phase 4A).
        examples_context: Pre-formatted few-shot examples (Phase 4B).
        retrieval_context: Pre-formatted retrieved sentences (Phase 5).

    Returns:
        List of message dicts [{"role": "system", ...}, {"role": "user", ...}].
    """

@property
def combined_prompt_version(self) -> str:
    return self.COMBINED_PROMPT_VERSION
```

### 4.3 Context Ordering Rationale

The fixed order (confusion → rules → examples → retrieved sentences) is:

1. **Confusion first**: Most specific to Qaari's failure modes. Sets the correction frame.
2. **Rules second**: General Arabic orthography. Applies across all corrections.
3. **Examples third**: Demonstration of the correction task format.
4. **Retrieval last**: Per-sample context, placed closest to the OCR input.

### 4.4 Prompt Version Table (All Phases)

| Phase | `prompt_type` | `prompt_version` | Contexts |
|-------|--------------|-----------------|---------|
| Phase 2 | `zero_shot` | `v1` | None |
| Phase 3 | `ocr_aware` | `p3v1` | Confusion |
| Phase 4A | `rule_augmented` | `p4av1` | Rules |
| Phase 4B | `few_shot` | `p4bv1` | Examples |
| Phase 5 | `rag` | `p5v1` | Retrieval |
| Phase 6 | `combined` | `p6v1` | Any subset |

---

## 5. JSONL Record Schema (Phase 6)

Each record in `results/phase6/{combo_id}/inference_input.jsonl`:

```json
{
  "sample_id": "KHATT-train_0042",
  "dataset": "KHATT-train",
  "ocr_text": "...",
  "gt_text": "...",
  "prompt_type": "combined",
  "combo_id": "pair_conf_rules",
  "prompt_version": "p6v1",
  "confusion_context": "أخطاء شائعة:\n1. ا → ن ...",
  "rules_context": "قواعد:\n1. التاء المربوطة ...",
  "examples_context": null,
  "retrieval_context": null
}
```

**Context fields**:
- Present as non-null string if the component is active in this combo.
- `null` if the component is not part of this combo.
- `retrieval_context` is always per-sample when RAG is active; other fields are the same for all records in a dataset.

**CAMeL combos** have no JSONL — they process corrections in memory using the base combo's `corrections.jsonl`.

---

## 6. CAMeL Post-Processing

### 6.1 Approach

CAMeL-based combos (`pair_best_camel`, `full_system`) reuse the "revert" strategy from Phase 4C:

1. Load base combo's `corrections.jsonl` (e.g., `full_prompt/corrections.jsonl`)
2. For each sample, compare corrected text vs OCR text word-by-word
3. If a corrected word fails `WordValidator.is_valid_arabic_word()`:
   - Revert to the OCR word
4. Save result as `{combo_id}/corrections.jsonl`
5. Compute metrics normally

This avoids any new LLM inference — pure local computation, fast.

### 6.2 "Best Pair" Selection

`pair_best_camel` needs a configured base. After running all pairs:

1. Compare `pair_conf_rules`, `pair_conf_fewshot`, `pair_conf_rag`, `pair_rules_fewshot`
2. Select best by average CER across all datasets
3. Set in `configs/config.yaml`:
   ```yaml
   phase6:
     pair_best: "pair_conf_rules"  # update after reviewing pair results
   ```
4. Run: `python pipelines/run_phase6.py --combo pair_best_camel --mode validate`

---

## 7. Pipeline: `pipelines/run_phase6.py`

### 7.1 Modes

| Mode | Stage | Description |
|------|-------|-------------|
| `export` | 1 (local) | Build JSONL with combined contexts |
| *(inference)* | 2 (Kaggle) | Run `scripts/infer.py` |
| `analyze` | 3 (local) | CER/WER, comparison vs Phase 2 |
| `validate` | — (local) | CAMeL post-processing combos only |
| `summarize` | — (local) | Cross-combo analysis, ablation summary |

### 7.2 CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--combo` | str | required | Combo ID (see Section 3.1) or `all` |
| `--mode` | str | required | `export` \| `analyze` \| `validate` \| `summarize` |
| `--datasets` | str+ | None | Subset of dataset keys |
| `--limit` | int | None | Max samples per dataset |
| `--force` | flag | False | Re-run even if output exists |
| `--no-error-analysis` | flag | False | Skip error_changes.json |
| `--phase2-dir` | path | `results/phase2` | Phase 2 baseline directory |
| `--config` | path | `configs/config.yaml` | Config file |
| `--results-dir` | path | `results/phase6` | Output directory |

### 7.3 Export Mode Logic

```
run_phase6.py --combo pair_conf_rules --mode export
│
├── 1. Load config; determine which contexts are active for combo_id
│
├── 2. Load active contexts (from prior phase results):
│       confusion_context: KnowledgeBase.load_confusion_matrix() → ConfusionFormatter
│       rules_context:     KnowledgeBase.load_rules()           → RulesFormatter
│       examples_context:  KnowledgeBase.load_qalb_examples()   → FewShotFormatter
│
├── 3. If RAG active: RAGRetriever.load_index()
│
├── 4. For each dataset × sample:
│       If RAG active: chunks = retriever.retrieve(ocr_text)
│                      retrieval_context = retriever.format_for_prompt(chunks)
│       Write JSONL record with combo_id + active contexts
│
└── 5. Log summary: N records written
```

**Context reuse**: Confusion, rules, and few-shot contexts are identical to Phases 3/4.
The export step re-reads from `results/phase1/` and `data/` sources directly — same
logic as individual phase exports, no need to read prior inference_input.jsonl files.

### 7.4 Analyze Mode Logic

```
run_phase6.py --combo pair_conf_rules --mode analyze
│
├── 1. Load corrections.jsonl (combined for all datasets)
├── 2. Load inference_input.jsonl (for sample metadata)
├── 3. Split by dataset; for each dataset:
│       compute_metrics() → metrics.json
│       compare_vs_phase2() → comparison_vs_phase2.json
│       optional: compute_error_changes() → error_changes.json
└── 4. Write aggregated metrics + report
```

### 7.5 Validate Mode Logic (CAMeL combos)

```
run_phase6.py --combo full_system --mode validate
│
├── 1. Determine base combo (full_prompt for full_system; config.phase6.pair_best for pair_best_camel)
├── 2. Load base corrections.jsonl
├── 3. Load WordValidator (CAMeL Tools)
├── 4. For each sample:
│       apply_camel_revert(ocr_text, corrected_text, validator) → validated_text
├── 5. Save corrections.jsonl under combo_id/
└── 6. Compute metrics (run analyze logic inline)
```

### 7.6 Summarize Mode Logic

```
run_phase6.py --mode summarize
│
├── 1. Load all combo metrics.json files (across all combos + Phase 2 baseline)
├── 2. Build combinations_summary.json (all pairs + full)
├── 3. Build ablation_summary.json (full vs each ablation)
├── 4. compute_synergy(): for each pair, synergy = Δpair − (Δcomp_a + Δcomp_b)
├── 5. compute_error_type_breakdown(): aggregate error_changes.json across combos
├── 6. build_final_comparison.json(): all phases + all combos in one table
├── 7. run_statistical_tests() using StatsTester
│       paired t-tests, Cohen's d, Bonferroni correction
├── 8. generate_paper_tables(): paper_tables.md with LaTeX-ready tables
└── 9. optional: generate_figures() (requires matplotlib)
```

### 7.7 Typical Workflow

```bash
# For each LLM-based combo (repeat per combo):
python pipelines/run_phase6.py --combo pair_conf_rules --mode export
python scripts/infer.py \
    --input  results/phase6/pair_conf_rules/inference_input.jsonl \
    --output results/phase6/pair_conf_rules/corrections.jsonl
python pipelines/run_phase6.py --combo pair_conf_rules --mode analyze

# CAMeL combos (local only):
# (After reviewing pair results, update phase6.pair_best in config.yaml)
python pipelines/run_phase6.py --combo full_system --mode validate
python pipelines/run_phase6.py --combo pair_best_camel --mode validate

# Final cross-combo analysis:
python pipelines/run_phase6.py --mode summarize

# Smoke test:
python pipelines/run_phase6.py --combo pair_conf_rules --mode export --datasets KHATT-train --limit 50
python scripts/infer.py --input results/phase6/pair_conf_rules/inference_input.jsonl \
    --output results/phase6/pair_conf_rules/corrections.jsonl --datasets KHATT-train --limit 50
python pipelines/run_phase6.py --combo pair_conf_rules --mode analyze --datasets KHATT-train
```

---

## 8. `scripts/infer.py` Addition

One new `prompt_type` dispatch case:

```python
elif prompt_type == "combined":
    combo_id = record.get("combo_id", "unknown")
    messages = builder.build_combined(
        ocr_text=record["ocr_text"],
        confusion_context=record.get("confusion_context") or "",
        rules_context=record.get("rules_context") or "",
        examples_context=record.get("examples_context") or "",
        retrieval_context=record.get("retrieval_context") or "",
    )
    prompt_ver = builder.combined_prompt_version
```

No other changes to `scripts/infer.py` are needed.

---

## 9. Cross-Combo Analysis

### 9.1 Synergy Analysis

For each pair, synergy measures whether the combination beats the sum of individual gains:

```
Δ_pair     = Phase2_CER − pair_CER
Δ_comp_a   = Phase2_CER − isolated_CER_a   (from Phase 3/4/5 results)
Δ_comp_b   = Phase2_CER − isolated_CER_b

synergy    = Δ_pair − (Δ_comp_a + Δ_comp_b)
```

- `synergy > 0`: Components amplify each other (super-additive)
- `synergy ≈ 0`: Components are independent
- `synergy < 0`: Components interfere (one dilutes the other)

### 9.2 Ablation Interpretation

For each ablation (full − component):

```
Δ_drop = full_CER − ablation_CER
```

- `Δ_drop large (negative)`: Component is essential — removing it hurts significantly
- `Δ_drop ≈ 0`: Component is redundant — others compensate
- `Δ_drop > 0`: Component was hurting — removing it helps (interference)

### 9.3 Statistical Analysis (StatsTester)

**Location**: `src/analysis/stats_tester.py` (new file)

Statistical significance testing for all key comparisons:

```python
class StatsTester:
    """Paired statistical tests for CER/WER comparison across correction methods."""

    def paired_ttest(
        self,
        scores_a: list[float],
        scores_b: list[float],
        alpha: float = 0.05,
    ) -> dict:
        """Paired two-tailed t-test. Returns t-stat, p-value, significant flag."""

    def cohens_d(self, scores_a: list[float], scores_b: list[float]) -> float:
        """Cohen's d effect size for paired samples."""

    def bonferroni_correct(self, p_values: list[float], alpha: float = 0.05) -> list[bool]:
        """Bonferroni correction for multiple comparisons. Returns list of significance flags."""

    def bootstrap_ci(
        self,
        scores_a: list[float],
        scores_b: list[float],
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """95% bootstrap confidence interval for the mean CER difference."""

    def compare_all(
        self,
        baseline: list[float],
        systems: dict[str, list[float]],
        alpha: float = 0.05,
    ) -> dict:
        """Run Bonferroni-corrected paired t-tests for all systems vs baseline.

        Args:
            baseline: CER scores for Phase 2 (one per sample, all datasets).
            systems: Dict mapping system name → per-sample CER scores.
            alpha: Family-wise error rate after Bonferroni correction.

        Returns:
            Dict with per-system: t_stat, p_value, p_corrected, significant, cohens_d.
        """
```

**Dependencies**: `scipy` (for `scipy.stats.ttest_rel`), `numpy`. Both are available in the
inference environment. For local analysis, add `scipy` to `requirements.txt`.

**Key tests to run**:

| Comparison | Test | Correction |
|------------|------|-----------|
| Each combo vs Phase 2 | Paired t-test | Bonferroni (N combos) |
| Full system vs each ablation | Paired t-test | Bonferroni (5 ablations) |
| Best combo vs best isolated phase | Paired t-test | None (single test) |

---

## 10. Output Structure

```
results/phase6/
├── combinations/
│   ├── pair_conf_rules/
│   │   ├── inference_input.jsonl
│   │   ├── corrections.jsonl
│   │   └── {dataset_name}/
│   │       ├── metrics.json
│   │       ├── comparison_vs_phase2.json
│   │       └── error_changes.json
│   ├── pair_conf_fewshot/    (same structure)
│   ├── pair_conf_rag/        (same structure)
│   ├── pair_rules_fewshot/   (same structure)
│   ├── pair_best_camel/      (no inference_input.jsonl)
│   │   └── {dataset_name}/
│   │       ├── metrics.json
│   │       └── comparison_vs_phase2.json
│   └── combinations_summary.json
├── full_prompt/
│   ├── inference_input.jsonl
│   ├── corrections.jsonl
│   └── {dataset_name}/
│       ├── metrics.json
│       ├── comparison_vs_phase2.json
│       └── error_changes.json
├── full_system/
│   └── {dataset_name}/
│       ├── metrics.json
│       └── comparison_vs_phase2.json
├── ablation/
│   ├── no_confusion/         (same structure as full_prompt)
│   ├── no_rules/
│   ├── no_fewshot/
│   ├── no_rag/
│   └── ablation_summary.json
├── analysis/
│   ├── synergy_analysis.json
│   ├── redundancy_matrix.json
│   └── error_type_breakdown.json
├── statistical_tests.json
├── final_comparison.json
├── figures/                  (optional, requires matplotlib)
│   ├── improvement_chart.png
│   ├── combination_heatmap.png
│   ├── ablation_chart.png
│   └── error_breakdown.png
├── paper_tables.md
└── report.md
```

---

## 11. Output Schemas

### 11.1 `metrics.json` (per dataset, per combo)

Same structure as Phases 3–5, with `phase = "phase6"` and `combo_id` in meta:

```json
{
  "meta": {
    "phase": "phase6",
    "combo_id": "pair_conf_rules",
    "components": ["confusion", "rules"],
    "dataset": "KHATT-train",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt_type": "combined",
    "prompt_version": "p6v1",
    "generated_at": "...",
    "num_samples": 1400
  },
  "corrected": {
    "cer": 0.065,
    "wer": 0.181
  }
}
```

### 11.2 `combinations_summary.json`

```json
{
  "meta": {"generated_at": "...", "n_combos": 5, "n_datasets": 18},
  "phase2_baseline": {"avg_cer": 0.089, "avg_wer": 0.234},
  "combinations": {
    "pair_conf_rules": {
      "components": ["confusion", "rules"],
      "avg_cer": 0.071,
      "avg_wer": 0.198,
      "delta_cer": -0.018,
      "delta_wer": -0.036,
      "cer_relative_pct": -20.2,
      "synergy_cer": 0.003
    }
  },
  "best_combo": "pair_conf_rules"
}
```

### 11.3 `ablation_summary.json`

```json
{
  "meta": {"generated_at": "...", "n_datasets": 18},
  "full_system": {"avg_cer": 0.060, "avg_wer": 0.175},
  "ablations": {
    "no_confusion": {
      "components_remaining": ["rules", "fewshot", "rag"],
      "avg_cer": 0.068,
      "delta_from_full": 0.008,
      "interpretation": "Confusion matrix is essential: CER rises 13.3% without it."
    },
    "no_rules": {...},
    "no_fewshot": {...},
    "no_rag": {...},
    "no_camel": {
      "components_remaining": ["confusion", "rules", "fewshot", "rag"],
      "avg_cer": 0.062,
      "delta_from_full": 0.002,
      "note": "Same as full_prompt — no new inference needed."
    }
  }
}
```

### 11.4 `synergy_analysis.json`

```json
{
  "meta": {"generated_at": "..."},
  "methodology": "synergy = delta_pair - (delta_a + delta_b)",
  "pairs": {
    "pair_conf_rules": {
      "delta_pair":    -0.018,
      "delta_conf":    -0.012,
      "delta_rules":   -0.007,
      "sum_individual": -0.019,
      "synergy":        0.001,
      "interpretation": "Near-additive. Components do not strongly interfere."
    }
  }
}
```

### 11.5 `statistical_tests.json`

```json
{
  "meta": {
    "test": "paired_ttest",
    "baseline": "phase2",
    "correction": "bonferroni",
    "alpha_family": 0.05,
    "n_tests": 11
  },
  "results": {
    "pair_conf_rules": {
      "t_stat": -5.42,
      "p_value": 0.00001,
      "p_corrected": 0.00011,
      "significant": true,
      "cohens_d": -0.31,
      "ci_95": [-0.021, -0.015]
    }
  }
}
```

### 11.6 `final_comparison.json`

All phases and combos in one structure — used to generate the paper's main results table:

```json
{
  "meta": {"generated_at": "..."},
  "systems": {
    "phase1_ocr": {"avg_cer": 0.113, "avg_wer": 0.289},
    "phase2_zero_shot": {"avg_cer": 0.089, "avg_wer": 0.234},
    "phase3_confusion": {"avg_cer": 0.081, "avg_wer": 0.217},
    "phase4a_rules":    {"avg_cer": 0.083, "avg_wer": 0.221},
    "phase4b_fewshot":  {"avg_cer": 0.079, "avg_wer": 0.209},
    "phase4c_camel":    {"avg_cer": 0.087, "avg_wer": 0.231},
    "phase5_rag":       {"avg_cer": 0.085, "avg_wer": 0.227},
    "pair_conf_rules":  {"avg_cer": 0.071, "avg_wer": 0.198},
    "full_prompt":      {"avg_cer": 0.062, "avg_wer": 0.179},
    "full_system":      {"avg_cer": 0.060, "avg_wer": 0.175}
  }
}
```

---

## 12. Configuration (`configs/config.yaml` Additions)

```yaml
# ---------------------------------------------------------------------------
# Phase 6 specific
# ---------------------------------------------------------------------------
phase6:
  pair_best: null              # Set after reviewing pair results (e.g. "pair_conf_rules")
  analyze_errors: true
  max_retries: 2
  stats:
    alpha: 0.05                # Family-wise error rate
    n_bootstrap: 1000          # Bootstrap iterations for CIs
  figures:
    enabled: false             # Set true if matplotlib is installed
    dpi: 150
```

---

## 13. New and Modified Files

### 13.1 New Files

| File | Purpose |
|------|---------|
| `pipelines/run_phase6.py` | Full pipeline: export / analyze / validate / summarize |
| `src/analysis/stats_tester.py` | `StatsTester`: paired t-test, Cohen's d, Bonferroni |

### 13.2 Modified Files

| File | Changes |
|------|---------|
| `src/core/prompt_builder.py` | Add `build_combined()`, `COMBINED_SECTION_*` constants, `COMBINED_PROMPT_VERSION = "p6v1"`, `combined_prompt_version` property |
| `scripts/infer.py` | Add `elif prompt_type == "combined":` dispatch branch |
| `configs/config.yaml` | Add `phase6:` block |
| `HOW_TO_RUN.md` | Add Phase 6 section |
| `requirements.txt` | Add `scipy>=1.10.0` (for stats_tester) |

### 13.3 No Modifications to Phases 1–5

All existing pipeline scripts, prompt builders, and output files are untouched.

---

## 14. Prerequisites

| Prerequisite | Why |
|-------------|-----|
| Phase 1 complete | Confusion matrices in `results/phase1/` |
| Phase 2 complete | Baseline metrics in `results/phase2/` |
| Phase 3 complete | Confirms confusion_context format; needed for reading confusion matrices |
| Phase 4A complete | Confirms rules_context format |
| Phase 4B complete | Confirms examples_context format |
| Phase 5 FAISS index built | RAG retrieval needed in export for RAG-containing combos |
| `camel-tools` installed | CAMeL validate mode |
| `sentence-transformers`, `faiss-cpu` installed | RAG export |
| `scipy` installed | `stats_tester.py` |

---

## 15. Implementation Order

| Step | Action | Notes |
|------|--------|-------|
| 1 | `src/analysis/stats_tester.py` — `StatsTester` | New analysis module |
| 2 | `src/core/prompt_builder.py` — add `build_combined()` | Minimal additions |
| 3 | `scripts/infer.py` — add `elif prompt_type == "combined":` | One new elif |
| 4 | `configs/config.yaml` — add `phase6:` block | |
| 5 | `pipelines/run_phase6.py` — `export` mode | Reuse context loaders from Phases 3-5 |
| 6 | Smoke test export: `--combo pair_conf_rules --mode export --datasets KHATT-train --limit 50` | |
| 7 | Run Kaggle inference for `pair_conf_rules` (smoke) | |
| 8 | `pipelines/run_phase6.py` — `analyze` mode | Reuse metrics logic from prior phases |
| 9 | Smoke test analyze | |
| 10 | Export + infer all 9 LLM-based combos (can parallelize on Kaggle) | |
| 11 | Analyze all 9 combos | |
| 12 | `pipelines/run_phase6.py` — `validate` mode | CAMeL post-processing |
| 13 | Run validate for `full_system` and `pair_best_camel` | |
| 14 | `pipelines/run_phase6.py` — `summarize` mode | Cross-combo analysis |
| 15 | Run summarize | Generate `final_comparison.json`, `paper_tables.md`, stats |
| 16 | Update `HOW_TO_RUN.md` | |

---

## 16. Context Loading Strategy

The Phase 6 export step re-derives contexts from the same sources as Phases 3–5.
It does **not** read from prior phase `inference_input.jsonl` files.

| Component | Source | Class/Function Used |
|-----------|--------|---------------------|
| Confusion | `results/phase1/{dataset}/confusion_matrix.json` | `KnowledgeBase.load_confusion_matrix()` + Phase 3 formatter |
| Rules | `data/rules/` | `KnowledgeBase.load_rules()` + Phase 4A formatter |
| Few-Shot | `data/QALB-*/` | `KnowledgeBase.load_qalb_examples()` + Phase 4B formatter |
| RAG | `results/phase5/faiss.index` | `RAGRetriever.load_index()` + `retrieve()` |

This keeps the export step self-contained and reproducible, even if prior phase JSONL files are deleted.

---

## 17. Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **Context window overflow**: 4 combined contexts may exceed Qwen3-4B's 32K context | Measure combined prompt token count during export. If >4096 tokens, shorten each section (e.g., top-5 confusion pairs instead of 10, top-3 rules, 3 examples, 2 retrieved sentences). |
| **Component interference**: Adding more context may confuse the model | This is a valid research finding. Report if full system underperforms best pair. |
| **Phase 5 FAISS index not built**: RAG-containing combos fail export | Check for index before export; exit with clear error + "Run `python pipelines/run_phase5.py --mode build` first". |
| **pair_best not set in config**: `pair_best_camel` fails | Exit with clear error: "Set `phase6.pair_best` in config.yaml after reviewing pair results." |
| **scipy not installed**: stats_tester.py fails | Graceful degradation: skip `statistical_tests.json` in summarize mode, log a warning. |
| **matplotlib not installed**: figure generation fails | `figures.enabled: false` by default; skip silently. |
| **9 separate Kaggle runs**: Logistically complex | Pipeline uses the same `scripts/infer.py` pattern as all prior phases. Can batch on Kaggle by chaining `infer.py` calls or using one notebook per combo. |

---

## 18. Research Value & Paper Contribution

### 18.1 Key Research Findings Phase 6 Produces

| Finding | Source |
|---------|--------|
| Best overall system CER/WER | `full_system/metrics.json` → `final_comparison.json` |
| Whether combinations beat isolated phases | `synergy_analysis.json` |
| Which component is most essential | `ablation_summary.json` (largest Δ_drop) |
| Which components interfere | `synergy_analysis.json` (negative synergy) |
| Minimal effective combination | `combinations_summary.json` (best pair) |
| Statistical confidence in improvements | `statistical_tests.json` |

### 18.2 Paper Table Structure

`paper_tables.md` will generate a LaTeX-ready table covering all phases and key combos:

```
System                | Avg CER | Δ CER | Avg WER | Δ WER | Sig.
─────────────────────────────────────────────────────────────────
Phase 1 (OCR only)    | 11.3%   |  —    | 28.9%   |  —    |
Phase 2 (zero-shot)   |  8.9%   | -2.4% | 23.4%   | -5.5% | *
Phase 3 (+Confusion)  |  8.1%   | -0.8% | 21.7%   | -1.7% | *
Phase 4A (+Rules)     |  8.3%   | -0.6% | 22.1%   | -1.3% | *
Phase 4B (+Few-Shot)  |  7.9%   | -1.0% | 20.9%   | -2.5% | *
Phase 4C (+CAMeL)     |  8.7%   | -0.2% | 23.1%   | -0.3% |
Phase 5 (+RAG)        |  8.5%   | -0.4% | 22.7%   | -0.7% |
Pair: C+R             |  7.1%   | -1.8% | 19.8%   | -3.6% | *
Full Prompt (C+R+F+G) |  6.2%   | -2.7% | 17.9%   | -5.5% | *
Full System (+CAMeL)  |  6.0%   | -2.9% | 17.5%   | -5.9% | *
─────────────────────────────────────────────────────────────────
* p < 0.05 (Bonferroni-corrected, vs Phase 2)
```
