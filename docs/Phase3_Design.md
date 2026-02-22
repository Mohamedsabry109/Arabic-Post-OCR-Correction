# Phase 3: OCR-Aware Prompting — Design Document

## 1. Overview

### 1.1 Purpose

Phase 3 tests whether injecting **Qaari's specific character-confusion patterns** into the
correction prompt improves LLM performance. The LLM in Phase 2 knew it was correcting Arabic OCR
text but had no knowledge of *which* errors Qaari actually makes. Phase 3 gives it that knowledge
by embedding the top-N confusion pairs from the Phase 1 confusion matrix directly into the system
prompt.

This is the simplest possible form of domain adaptation — no model fine-tuning, no retrieval,
just informed prompting.

### 1.2 Research Question

**Does telling the LLM about Qaari's specific character confusion patterns improve correction
accuracy over zero-shot prompting?**

### 1.3 Isolated Comparison Design

> **CRITICAL**: Phase 3 compares **only against Phase 2** (zero-shot LLM baseline).
> It does NOT compare against Phase 1 (OCR baseline). This isolates the effect of the
> confusion matrix injection from the general LLM correction ability.
>
> The key metric is: Δ CER = Phase3 CER − Phase2 CER

### 1.4 Research Questions Answered

| Question | Output Artefact |
|----------|----------------|
| Does OCR-error knowledge improve correction? | `comparison_vs_phase2.json` (Δ CER/WER) |
| Which confusion pairs does the LLM learn to fix? | `confusion_impact.json` |
| Is the improvement consistent across datasets? | `comparison_vs_phase2.json` per dataset |
| Does improvement vary with number of confusions injected? | `variant_summary.json` (optional) |
| Does the LLM introduce new errors when OCR-aware? | `error_changes.json` |

### 1.5 Downstream Use

- **Phase 6**: Uses `metrics.json` as the "confusion matrix" component in combinations/ablation
- **Paper**: "Phase 3: + Confusion Matrix" row in the main results table
- **Phase 6 combinations**: Pair A (Confusion + Rules), Pair B (Confusion + Few-shot), etc.

### 1.6 Compute Environment

Same as Phase 2. No local GPU. LLM inference runs on **Kaggle** or **Google Colab**.
Pipeline is split into three stages: export → inference → analyze.

See `docs/Guidelines.md §12` for the full compute environment policy.

---

## 2. Data

### 2.1 Datasets

Phase 3 processes all 18 datasets identically to Phase 2. No change to data loading.

### 2.2 Knowledge Source: Phase 1 Confusion Matrices

Each Phase 1 dataset produces a `confusion_matrix.json`. Phase 3 loads these to inject
confusion information into prompts.

**Location**:
```
results/phase1/{dataset_key}/confusion_matrix.json
```

**Key fields used from this file**:
- `confusions`: Dict mapping `gt_char → { ocr_char → { count, probability } }`
- `top_20`: Pre-sorted list of top-20 substitution pairs (used directly)
- `metadata.total_substitutions`: Guards against sparse matrices

### 2.3 Confusion Matrix Selection Strategy

Phase 3 uses **per-dataset confusion matrices** — the confusion matrix from Phase 1 for
`KHATT-train` is used when processing `KHATT-train`, etc.

**Sparsity fallback**: If a dataset's confusion matrix has fewer than
`config['phase3']['min_substitutions_for_dataset_matrix']` (default: 200) total
substitutions, fall back to the **pooled confusion matrix** for the same dataset type:

| Dataset type | Pooled fallback |
|---|---|
| PATS-A01 (any font/split) | Pool all 16 PATS confusion matrices |
| KHATT | Pool KHATT-train + KHATT-validation |

The pooled matrices are pre-built during the export step if any individual matrix is sparse.

**Why per-dataset?** Different PATS fonts may have different error distributions. Handwritten
KHATT has different confusion patterns than typewritten PATS. Per-dataset injection is most
accurate.

### 2.4 Data Flow

```
LOCAL (export stage)
────────────────────────────────────────────────────────────────────────────
DataLoader.iter_samples(dataset_key)
       │
       │  OCRSample(ocr_text, gt_text, sample_id, ...)
       ▼
ConfusionMatrixLoader.load(results/phase1/{key}/confusion_matrix.json)
       │
       │  top-N confusion pairs
       ▼
ConfusionMatrixLoader.format_for_prompt(confusions, n=10)
       │
       │  confusion_context string (Arabic text)
       ▼
inference_input.jsonl ← {sample_id, ocr_text, gt_text, dataset,
                          prompt_type="ocr_aware", confusion_context="..."}
       │
       │  [pushed to git / uploaded to Kaggle]
       ▼

REMOTE (inference stage — Kaggle/Colab)
────────────────────────────────────────────────────────────────────────────
scripts/infer.py reads record["prompt_type"] → "ocr_aware"
       │
       ▼
PromptBuilder.build_ocr_aware(ocr_text, confusion_context)
       │
       ▼
TransformersCorrector.correct(...)
       │
       ▼
corrections.jsonl ← {sample_id, corrected_text, gt_text, prompt_type, ...}
       │
       │  [downloaded from Kaggle]
       ▼

LOCAL (analyze stage)
────────────────────────────────────────────────────────────────────────────
Load corrections.jsonl → list[CorrectedSample]
       │
       ├──► calculate_metrics(text_field="corrected_text")  → metrics.json
       ├──► compare_metrics(phase2_baseline, phase3_corrected) → comparison_vs_phase2.json
       ├──► run_error_change_analysis()                      → error_changes.json
       └──► run_confusion_impact_analysis()                  → confusion_impact.json
```

---

## 3. Module Design

### 3.1 New Modules

```
src/
└── data/
    └── knowledge_base.py       ← NEW: ConfusionMatrixLoader (Phase 3)
                                        (stub for RulesLoader, QALBLoader — Phases 4A/4B)

pipelines/
└── run_phase3.py               ← NEW: Phase 3 export/analyze pipeline
```

### 3.2 Modified Modules

```
src/core/prompt_builder.py      ← ADD: build_ocr_aware() method + OCR_AWARE_SYSTEM_V1 constant
scripts/infer.py                ← ADD: prompt_type dispatch (zero_shot vs ocr_aware)
```

### 3.3 Reused Modules (No Modifications)

```
src/data/data_loader.py         ← Unchanged
src/data/text_utils.py          ← Unchanged
src/analysis/metrics.py         ← Unchanged
src/analysis/error_analyzer.py  ← Unchanged
src/core/llm_corrector.py       ← Unchanged
pipelines/_utils.py             ← Unchanged (resolve_datasets used as-is)
```

---

## 4. `src/data/knowledge_base.py` — ConfusionMatrixLoader

This is a new file that will grow across phases:
- Phase 3: `ConfusionMatrixLoader` (confusion matrix loading and formatting)
- Phase 4A: `RulesLoader` (stub at Phase 3 time)
- Phase 4B: `QALBLoader` (stub at Phase 3 time)

### 4.1 Data Structures

```python
from dataclasses import dataclass


@dataclass
class ConfusionPair:
    """A single character-level confusion pair from Phase 1."""
    gt_char: str          # The correct character (ground truth)
    ocr_char: str         # What Qaari produces instead
    count: int            # Number of occurrences in Phase 1 corpus
    probability: float    # count / total substitutions for this gt_char
```

### 4.2 `ConfusionMatrixLoader` Class

```python
import json
from pathlib import Path
from typing import Optional


class ConfusionMatrixLoader:
    """Load and format Phase 1 confusion matrices for prompt injection.

    Usage::

        loader = ConfusionMatrixLoader()
        pairs = loader.load(Path("results/phase1/KHATT-train/confusion_matrix.json"))
        text = loader.format_for_prompt(pairs, n=10)

    The formatted text is embedded into inference_input.jsonl records during the
    Phase 3 export step and read by infer.py on the remote machine.
    """

    # Minimum substitutions required to use a dataset-specific matrix.
    # Datasets below this threshold fall back to the pooled matrix.
    MIN_SUBSTITUTIONS: int = 200

    def load(self, path: Path) -> list[ConfusionPair]:
        """Load confusion pairs from a Phase 1 confusion_matrix.json file.

        Uses the pre-sorted 'top_20' list if present (faster); otherwise
        flattens and sorts the full 'confusions' dict.

        Args:
            path: Path to confusion_matrix.json produced by Phase 1.

        Returns:
            List of ConfusionPair sorted by count descending.

        Raises:
            FileNotFoundError: If confusion matrix file does not exist.
            ValueError: If file is malformed or has unexpected structure.
        """

    def load_pooled(self, matrix_paths: list[Path]) -> list[ConfusionPair]:
        """Pool confusion pairs from multiple confusion_matrix.json files.

        Counts are summed across datasets; probabilities are recomputed
        from the pooled counts.

        Args:
            matrix_paths: List of paths to confusion_matrix.json files.

        Returns:
            List of ConfusionPair sorted by pooled count descending.
        """

    def has_enough_data(self, path: Path, min_substitutions: int = MIN_SUBSTITUTIONS) -> bool:
        """Check if a confusion matrix has enough data to be useful.

        Args:
            path: Path to confusion_matrix.json.
            min_substitutions: Minimum total substitutions threshold.

        Returns:
            True if the matrix has >= min_substitutions total substitutions.
            False if file is missing, malformed, or below threshold.
        """

    def format_for_prompt(
        self,
        pairs: list[ConfusionPair],
        n: int = 10,
        style: str = "flat_arabic",
    ) -> str:
        """Format top-N confusion pairs into Arabic text for prompt injection.

        Args:
            pairs: Sorted list of ConfusionPair (from load() or load_pooled()).
            n: Number of top pairs to include.
            style: Formatting style. See §4.3 for options.

        Returns:
            A multi-line Arabic string describing the confusion patterns.
            Returns empty string if pairs is empty.
        """
```

### 4.3 Format Styles

Two styles are supported. The **primary** style is `flat_arabic`.

#### Style A: `flat_arabic` (Primary — used in main experiment)

Lists each confusion pair as a simple Arabic sentence. Readable by the model
and easy to parse if needed.

**Example output (N=5)**:
```
أخطاء Qaari الشائعة في التعرف على الحروف:
- يستبدل (ة) بـ (ه) في 85% من الحالات
- يستبدل (ب) بـ (ت) في 32% من الحالات
- يستبدل (أ) بـ (ا) في 28% من الحالات
- يستبدل (ي) بـ (ى) في 21% من الحالات
- يستبدل (ح) بـ (ج) في 14% من الحالات
```

**Template per line**:
```
- يستبدل ({gt_char}) بـ ({ocr_char}) في {round(probability*100)}% من الحالات
```

#### Style B: `grouped_arabic` (Optional variant)

Groups confusion pairs by error category (using the same character groups as
`ErrorAnalyzer`). More compact, but requires classifying each pair into a group.

**Example output**:
```
أخطاء Qaari الشائعة في التعرف على الحروف:
- التاء المربوطة والهاء: يخلط بين (ة) و (ه)
- الهمزات: يخلط بين (أ) و (ا) و (إ)
- النقاط: يخلط بين (ب) و (ت)، وبين (ح) و (ج)
- الألف المقصورة: يخلط بين (ى) و (ي)
```

Groups mirror `ErrorAnalyzer.DOT_GROUPS`, `HAMZA_GROUP`, `TAA_GROUP`, `ALEF_MAKSURA_GROUP`.

**Note**: `grouped_arabic` is used in optional sub-experiments only. The main Phase 3
result uses `flat_arabic` for simplicity and reproducibility.

### 4.4 Stub Placeholders for Future Phases

At the bottom of `knowledge_base.py`, add empty stub classes to establish the file's scope:

```python
class RulesLoader:
    """Load Arabic orthographic rules for Phase 4A. (Not yet implemented.)"""
    ...


class QALBLoader:
    """Load QALB error-correction pairs for Phase 4B. (Not yet implemented.)"""
    ...
```

---

## 5. `src/core/prompt_builder.py` Additions

### 5.1 New Prompt Constants

Add to the `PromptBuilder` class:

```python
# -----------------------------------------------------------------------
# Phase 3 prompt constants — OCR-Aware Prompting
# -----------------------------------------------------------------------

OCR_AWARE_SYSTEM_V1: str = (
    "أنت مصحح نصوص عربية متخصص في تصحيح مخرجات نظام Qaari للتعرف الضوئي. "
    "فيما يلي أبرز الأخطاء الشائعة التي يرتكبها هذا النظام:\n\n"
    "{confusion_context}\n\n"
    "صحح النص التالي مع الانتباه بشكل خاص لهذه الأخطاء. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)

OCR_AWARE_PROMPT_VERSION: str = "p3v1"
```

**Why a separate version string?** `PROMPT_VERSION = "v1"` belongs to the zero-shot Phase 2
prompt. Phase 3 uses `"p3v1"` so `corrections.jsonl` records are unambiguously attributable to
the right phase and prompt wording.

### 5.2 New Method: `build_ocr_aware()`

```python
def build_ocr_aware(
    self,
    ocr_text: str,
    confusion_context: str,
) -> list[dict]:
    """Build OCR-aware correction prompt (Phase 3).

    Injects Qaari's top-N character confusion pairs into the system prompt.
    The LLM receives:
      - A system prompt naming Qaari as the OCR engine + formatted confusion list
      - The raw OCR text to correct (user turn)

    Args:
        ocr_text: OCR prediction text to correct.
        confusion_context: Pre-formatted Arabic string describing confusion
            pairs (produced by ConfusionMatrixLoader.format_for_prompt()).
            If empty string, falls back gracefully to zero-shot system prompt.

    Returns:
        Two-element messages list::

            [
              {"role": "system", "content": <system with confusion_context>},
              {"role": "user",   "content": ocr_text}
            ]
    """
    if not confusion_context.strip():
        # Graceful fallback: no confusions available → use zero-shot
        return self.build_zero_shot(ocr_text)

    system = self.OCR_AWARE_SYSTEM_V1.format(confusion_context=confusion_context)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": ocr_text},
    ]

@property
def ocr_aware_prompt_version(self) -> str:
    """Return the Phase 3 prompt version string."""
    return self.OCR_AWARE_PROMPT_VERSION
```

### 5.3 Design Notes

- `confusion_context` is **pre-computed** during the export step and embedded in
  `inference_input.jsonl`. The remote inference machine does not need to load any files.
- The `{confusion_context}` placeholder is inside the system prompt (not the user turn).
  This frames the confusion knowledge as part of the LLM's role/identity, not as
  additional instructions layered on top of the text.
- All samples from the same dataset share the same `confusion_context` string; it is
  computed once per dataset during export.
- If `confusion_context` is empty (sparse matrix + no pooled fallback), `build_ocr_aware()`
  falls back to `build_zero_shot()` and logs a warning. The record in `corrections.jsonl`
  will carry `"prompt_type": "zero_shot_fallback"` to flag this.

---

## 6. `scripts/infer.py` Modifications

### 6.1 Changes Required

The inference loop currently hardcodes `builder.build_zero_shot()`. Phase 3 requires it to
dispatch based on the `prompt_type` field in each record.

**Current code (line ~339)**:
```python
messages = builder.build_zero_shot(record["ocr_text"])
```

**New code**:
```python
prompt_type = record.get("prompt_type", "zero_shot")

if prompt_type == "ocr_aware":
    confusion_context = record.get("confusion_context", "")
    messages = builder.build_ocr_aware(record["ocr_text"], confusion_context)
    prompt_ver = builder.ocr_aware_prompt_version
elif prompt_type == "zero_shot":
    messages = builder.build_zero_shot(record["ocr_text"])
    prompt_ver = builder.prompt_version
else:
    logger.warning("Unknown prompt_type '%s' for %s — falling back to zero_shot.",
                   prompt_type, record["sample_id"])
    messages = builder.build_zero_shot(record["ocr_text"])
    prompt_ver = builder.prompt_version
    prompt_type = "zero_shot_fallback"
```

The `out_record` dict gains a `"prompt_type"` field:
```python
out_record = {
    ...
    "prompt_type":    prompt_type,
    "prompt_version": prompt_ver,
    ...
}
```

### 6.2 Backward Compatibility

Phase 2 `inference_input.jsonl` records do **not** have a `prompt_type` field.
`record.get("prompt_type", "zero_shot")` defaults to `"zero_shot"`, so existing
Phase 2 records are processed identically to before. No re-running Phase 2 needed.

### 6.3 What Stays the Same

- All HF sync logic (`--hf-repo`, `--hf-token`, `--sync-every`) is unchanged
- All resume logic (reading completed sample_ids) is unchanged
- Default `--input` / `--output` paths still point to `results/phase2/` — Phase 3
  passes its own paths via CLI args
- `--model`, `--limit`, `--force`, `--datasets` flags all work identically

---

## 7. `pipelines/run_phase3.py`

### 7.1 Entry Point

```python
#!/usr/bin/env python3
"""Phase 3: OCR-Aware Prompting (Confusion Matrix Injection).

Isolated experiment: compares against Phase 2 (zero-shot) baseline only.
Each dataset uses its own Phase 1 confusion matrix; sparse datasets fall
back to a pooled confusion matrix for the same dataset type.

Three-stage pipeline (no local GPU required):

  --mode export   → Build confusion-injected inference_input.jsonl
  --mode analyze  → Load corrections.jsonl, compute metrics and reports
  --mode full     → End-to-end with API backend (requires model.backend='api')

Usage
-----
    python pipelines/run_phase3.py --mode export
    python pipelines/run_phase3.py --mode export  --limit 50
    python pipelines/run_phase3.py --mode export  --top-n 5
    python pipelines/run_phase3.py --mode export  --top-n 20  --format grouped_arabic
    python pipelines/run_phase3.py --mode export  --datasets KHATT-train
    python pipelines/run_phase3.py --mode analyze
    python pipelines/run_phase3.py --mode analyze --datasets KHATT-train
    python pipelines/run_phase3.py --mode analyze --no-error-analysis
"""
```

### 7.2 CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | `export` \| `analyze` \| `full` |
| `--top-n` | int | 10 | Number of confusion pairs to inject (5, 10, or 20) |
| `--format` | str | `flat_arabic` | Confusion format: `flat_arabic` \| `grouped_arabic` |
| `--limit` | int | None | Max samples per dataset (testing) |
| `--datasets` | str+ | None | Subset of dataset keys to process |
| `--force` | flag | False | Re-export / re-analyze even if files exist |
| `--no-error-analysis` | flag | False | Skip error_changes.json (faster) |
| `--no-confusion-impact` | flag | False | Skip confusion_impact.json (faster) |
| `--config` | path | `configs/config.yaml` | Config file path |
| `--results-dir` | path | `results/phase3` | Output directory |
| `--phase1-dir` | path | `results/phase1` | Phase 1 results (confusion matrices) |
| `--phase2-dir` | path | `results/phase2` | Phase 2 results (comparison baseline) |

### 7.3 Execution Flow — Export Mode

```
run_phase3.py --mode export
│
├─ 1. parse_args() + setup_logging() + load_config()
│
├─ 2. Initialise DataLoader(config)
│       Initialise ConfusionMatrixLoader()
│
├─ 3. Pre-build pooled confusion matrices (one per dataset type):
│     │
│     ├─ PATS-pooled: load all 16 PATS phase1 confusion matrices → pool
│     └─ KHATT-pooled: load KHATT-train + KHATT-validation matrices → pool
│       (Done once upfront; used as fallback for sparse individual matrices)
│
├─ 4. For each dataset_key in active_datasets:
│     │
│     ├─ 4a. Check resume: if dataset_key already in inference_input.jsonl → skip
│     │
│     ├─ 4b. Load Phase 1 confusion matrix:
│     │       path = results/phase1/{key}/confusion_matrix.json
│     │       if path not found OR not has_enough_data():
│     │           use pooled fallback for dataset type
│     │           log WARNING: "Using pooled matrix for {key}"
│     │
│     ├─ 4c. Format confusion context:
│     │       confusion_context = loader.format_for_prompt(pairs, n=args.top_n,
│     │                                                    style=args.format)
│     │
│     ├─ 4d. Iterate loader.iter_samples(dataset_key, limit) → OCRSample
│     │
│     └─ 4e. Write each sample to results/phase3/inference_input.jsonl:
│             {
│               "sample_id":        ...,
│               "dataset":          ...,
│               "ocr_text":         ...,
│               "gt_text":          ...,
│               "prompt_type":      "ocr_aware",
│               "confusion_context": confusion_context,
│               "top_n":            args.top_n,
│               "format_style":     args.format
│             }
│
└─ 5. Log summary:
       "inference_input.jsonl ready: {N} samples across {K} datasets.
        Next: run scripts/infer.py --input results/phase3/inference_input.jsonl
                                   --output results/phase3/corrections.jsonl"
```

**Resume in export mode**: Before writing a dataset's samples, check whether `dataset_key`
already appears in `results/phase3/inference_input.jsonl`. If present and `--force` is not
set, skip. This mirrors Phase 2's resume convention.

### 7.4 Execution Flow — Analyze Mode

```
run_phase3.py --mode analyze
│
├─ 1. parse_args() + setup_logging() + load_config()
│
├─ 2. Auto-split combined corrections.jsonl (if present at results/phase3/corrections.jsonl)
│     into per-dataset files at results/phase3/{dataset_key}/corrections.jsonl
│     (Same logic as Phase 2's analyze step)
│
├─ 3. Load Phase 2 baseline from results/phase2/metrics.json
│     If missing: warn + set phase2_metrics = None (comparison skipped)
│
├─ 4. For each dataset_key in active_datasets:
│     │
│     ├─ 4a. Check resume: if results/phase3/{key}/metrics.json exists → skip (unless --force)
│     │
│     ├─ 4b. Load results/phase3/{key}/corrections.jsonl → list[CorrectedSample]
│     │       Abort with clear error if file missing
│     │
│     ├─ 4c. calculate_metrics(corrected_samples, text_field="corrected_text")
│     │       save: results/phase3/{key}/metrics.json
│     │
│     ├─ 4d. compare_metrics(phase2_baseline[key], phase3_metrics)
│     │       save: results/phase3/{key}/comparison_vs_phase2.json
│     │
│     ├─ 4e. [if not --no-error-analysis]
│     │       run_error_change_analysis(corrected_samples, key)
│     │       save: results/phase3/{key}/error_changes.json
│     │
│     └─ 4f. [if not --no-confusion-impact]
│             run_confusion_impact_analysis(corrected_samples, key,
│                                           phase2_error_changes, confusion_pairs)
│             save: results/phase3/{key}/confusion_impact.json
│
├─ 5. Aggregate → results/phase3/metrics.json
│
├─ 6. Aggregate comparisons → results/phase3/comparison.json
│
├─ 7. generate_report() → results/phase3/report.md
│
└─ 8. Print summary table
```

### 7.5 Key Functions

```python
def run_export(
    config: dict,
    results_dir: Path,
    phase1_dir: Path,
    active_datasets: list[str],
    top_n: int,
    format_style: str,
    limit: int | None,
    force: bool,
) -> None:
    """Build confusion-injected inference_input.jsonl for all active datasets."""


def run_analyze(
    config: dict,
    results_dir: Path,
    phase1_dir: Path,
    phase2_dir: Path,
    active_datasets: list[str],
    analyze_errors: bool,
    analyze_impact: bool,
    force: bool,
) -> None:
    """Load corrections, compute metrics and reports. No LLM or GPU required."""


def build_pooled_matrices(
    phase1_dir: Path,
    loader: "ConfusionMatrixLoader",
) -> dict[str, list["ConfusionPair"]]:
    """Pre-build pooled confusion matrices for PATS and KHATT dataset types.

    Returns:
        Dict with keys "PATS-A01" and "KHATT", each mapping to a sorted
        list of pooled ConfusionPair objects.
    """


def resolve_confusion_matrix(
    dataset_key: str,
    phase1_dir: Path,
    pooled: dict[str, list["ConfusionPair"]],
    loader: "ConfusionMatrixLoader",
) -> tuple[list["ConfusionPair"], str]:
    """Return (confusion_pairs, source_label) for a dataset.

    source_label is "dataset_specific" or "pooled_{type}" — stored in
    inference_input.jsonl for traceability.

    Resolution order:
    1. Load results/phase1/{dataset_key}/confusion_matrix.json
    2. If missing or sparse: use pooled["PATS-A01"] or pooled["KHATT"]
    3. If pooled is also empty: return ([], "none") — triggers zero_shot fallback

    Args:
        dataset_key: e.g. "KHATT-train" or "PATS-A01-Akhbar-train"
        phase1_dir: Path to results/phase1/
        pooled: Pre-built pooled matrices from build_pooled_matrices()
        loader: ConfusionMatrixLoader instance

    Returns:
        Tuple of (pairs list, source_label string).
    """


def run_confusion_impact_analysis(
    corrected_samples: list,          # list[CorrectedSample]
    dataset_name: str,
    phase2_error_changes_path: Path,  # results/phase2/{key}/error_changes.json
    confusion_pairs: list["ConfusionPair"],
    results_dir: Path,
) -> dict:
    """Measure impact of confusion injection per injected confusion pair.

    Compares Phase 2 and Phase 3 error_changes for each confusion pair that
    was injected into the prompt. Determines whether the LLM fixed more
    errors of each type after being told about them.

    Args:
        corrected_samples: Phase 3 corrected samples.
        dataset_name: Dataset key label.
        phase2_error_changes_path: Path to Phase 2 error_changes.json for comparison.
        confusion_pairs: The top-N pairs that were injected.
        results_dir: Where to write confusion_impact.json.

    Returns:
        confusion_impact dict (see §10.5 for full schema).
    """


def generate_report(
    all_metrics: dict[str, "MetricResult"],
    all_comparisons: dict[str, dict],
    all_error_changes: dict[str, dict],
    all_confusion_impacts: dict[str, dict],
    model_name: str,
    top_n: int,
    results_dir: Path,
) -> None:
    """Write human-readable Markdown report to results_dir/report.md."""
```

---

## 8. Sub-Experiments: N Variants

### 8.1 Primary Experiment (N=10, format=flat_arabic)

The **main Phase 3 result** uses default settings: N=10, `flat_arabic`.
This is the number that appears in the paper's main results table.

```bash
python pipelines/run_phase3.py --mode export
python scripts/infer.py --input results/phase3/inference_input.jsonl \
                        --output results/phase3/corrections.jsonl \
                        --model Qwen/Qwen3-4B-Instruct-2507
python pipelines/run_phase3.py --mode analyze
```

### 8.2 Optional Variant Experiments

Run separately to explore the effect of N. Results go into sub-directories to avoid
overwriting the primary experiment.

| Variant | N | Format | Output dir |
|---------|---|--------|-----------|
| Primary | 10 | flat_arabic | `results/phase3/` |
| Variant A | 5 | flat_arabic | `results/phase3_n5/` |
| Variant B | 20 | flat_arabic | `results/phase3_n20/` |
| Variant C | 10 | grouped_arabic | `results/phase3_grouped/` |

```bash
# Variant A (N=5)
python pipelines/run_phase3.py --mode export --top-n 5 --results-dir results/phase3_n5
python scripts/infer.py --input results/phase3_n5/inference_input.jsonl \
                        --output results/phase3_n5/corrections.jsonl
python pipelines/run_phase3.py --mode analyze --results-dir results/phase3_n5 \
                               --phase3-dir results/phase3_n5
```

### 8.3 Variant Summary Report

After running variants, generate a comparison with:

```bash
python pipelines/run_phase3.py --mode variant-summary \
    --variant-dirs results/phase3 results/phase3_n5 results/phase3_n20
```

Produces `results/phase3/variant_summary.json` — a table comparing CER/WER across N values,
to show whether more confusions always helps or if there's a diminishing-returns point.

**This is optional** — if time is limited, skip variants and use N=10 as the sole result.

---

## 9. Output Structure

```
results/phase3/
├── inference_input.jsonl             ← upload to Kaggle/Colab (export stage)
│
├── PATS-A01-Akhbar-train/
│   ├── corrections.jsonl             ← download from Kaggle/Colab
│   ├── metrics.json                  ← Phase 3 CER/WER
│   ├── comparison_vs_phase2.json     ← Δ vs Phase 2 baseline (ISOLATED comparison)
│   ├── error_changes.json            ← Fixed/introduced errors by type
│   └── confusion_impact.json         ← Per-confusion-pair improvement analysis
├── PATS-A01-Akhbar-val/
│   └── [same structure]
├── *(14 more dataset folders)*
├── KHATT-train/
│   └── [same structure]
├── KHATT-validation/
│   └── [same structure]
│
├── metrics.json                      ← Aggregated Phase 3 CER/WER
├── comparison.json                   ← Aggregated comparison vs Phase 2
├── variant_summary.json              ← (optional) comparison across N=5/10/20 variants
├── phase3.log
└── report.md
```

---

## 10. Output Schemas

### 10.1 `inference_input.jsonl` (Phase 3 — export stage)

Extends the Phase 2 format with three new fields.

```json
{
  "sample_id":         "AHTD3A0001_Para2_3",
  "dataset":           "KHATT-train",
  "ocr_text":          "... raw OCR prediction ...",
  "gt_text":           "... ground truth ...",
  "prompt_type":       "ocr_aware",
  "confusion_context": "أخطاء Qaari الشائعة في التعرف على الحروف:\n- يستبدل (ة) بـ (ه) في 85% من الحالات\n...",
  "top_n":             10,
  "format_style":      "flat_arabic",
  "confusion_source":  "dataset_specific"
}
```

**`confusion_source`** values:
- `"dataset_specific"` — used the dataset's own Phase 1 confusion matrix
- `"pooled_PATS-A01"` — fell back to pooled PATS matrix (sparse dataset)
- `"pooled_KHATT"` — fell back to pooled KHATT matrix
- `"none"` — no confusion data available; `prompt_type` will be `"zero_shot_fallback"`

### 10.2 `corrections.jsonl` (inference stage output)

Same schema as Phase 2, with the addition of `prompt_type`:

```json
{
  "sample_id":         "AHTD3A0001_Para2_3",
  "dataset":           "KHATT-train",
  "ocr_text":          "...",
  "corrected_text":    "...",
  "gt_text":           "...",
  "model":             "Qwen/Qwen3-4B-Instruct-2507",
  "prompt_type":       "ocr_aware",
  "prompt_version":    "p3v1",
  "prompt_tokens":     187,
  "output_tokens":     94,
  "latency_s":         2.48,
  "success":           true,
  "error":             null
}
```

`prompt_version: "p3v1"` distinguishes Phase 3 corrections from Phase 2 (`"v1"`) in any
combined JSONL files.

### 10.3 `metrics.json` (per dataset)

```json
{
  "meta": {
    "phase": "phase3",
    "dataset": "KHATT-train",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt_type": "ocr_aware",
    "prompt_version": "p3v1",
    "top_n": 10,
    "format_style": "flat_arabic",
    "confusion_source": "dataset_specific",
    "generated_at": "...",
    "git_commit": "...",
    "num_samples": 1400,
    "limit_applied": null,
    "failed_samples": 2,
    "zero_shot_fallback_samples": 0
  },
  "corrected": {
    "cer": 0.071,
    "wer": 0.198,
    "cer_std": 0.058,
    "wer_std": 0.171,
    "cer_median": 0.055,
    "wer_median": 0.162,
    "cer_p95": 0.198,
    "wer_p95": 0.589,
    "num_samples": 1400,
    "num_chars_ref": 158432,
    "num_words_ref": 28901
  }
}
```

**`zero_shot_fallback_samples`**: Count of samples where `confusion_context` was empty and
`build_ocr_aware()` silently fell back to the zero-shot prompt. If non-zero, the result is
partially contaminated — log a prominent warning in the report.

### 10.4 `comparison_vs_phase2.json` (per dataset)

```json
{
  "meta": {
    "comparison": "phase3_vs_phase2",
    "dataset": "KHATT-train",
    "generated_at": "...",
    "top_n": 10,
    "format_style": "flat_arabic"
  },
  "phase2_baseline": {
    "cer": 0.089,
    "wer": 0.234,
    "source": "results/phase2/KHATT-train/metrics.json"
  },
  "phase3_corrected": {
    "cer": 0.071,
    "wer": 0.198
  },
  "delta": {
    "cer_absolute": -0.018,
    "wer_absolute": -0.036,
    "cer_relative_pct": -20.2,
    "wer_relative_pct": -15.4
  },
  "interpretation": "CER reduced by 20.2% relative vs Phase 2 (8.9% → 7.1%). WER reduced by 15.4% (23.4% → 19.8%).",
  "significant": null
}
```

Convention: **negative delta = improvement** (consistent with Phase 2 convention).

`"significant": null` — populated by Phase 6 statistical tests (t-test, effect size). Left
null in Phase 3 output since statistical testing is deferred to Phase 6.

### 10.5 `confusion_impact.json` (per dataset)

The most Phase-3-specific output. Measures whether injecting each confusion pair actually
helped the LLM fix that error type.

```json
{
  "meta": {
    "dataset": "KHATT-train",
    "top_n_injected": 10,
    "phase2_error_changes_source": "results/phase2/KHATT-train/error_changes.json",
    "generated_at": "..."
  },
  "injected_pairs": [
    {"gt": "ة", "ocr": "ه", "count": 312, "probability": 0.85, "rank": 1},
    {"gt": "ب", "ocr": "ت", "count": 245, "probability": 0.32, "rank": 2}
  ],
  "impact_by_pair": {
    "ة→ه": {
      "phase1_error_count":    312,
      "phase2_remaining":       89,
      "phase3_remaining":       51,
      "phase2_fix_rate":       0.715,
      "phase3_fix_rate":       0.836,
      "marginal_improvement":  0.121,
      "direction":             "improved"
    },
    "ب→ت": {
      "phase1_error_count":    245,
      "phase2_remaining":      134,
      "phase3_remaining":      127,
      "phase2_fix_rate":       0.453,
      "phase3_fix_rate":       0.482,
      "marginal_improvement":  0.029,
      "direction":             "improved"
    }
  },
  "summary": {
    "total_pairs_injected":    10,
    "pairs_improved":           8,
    "pairs_unchanged":          1,
    "pairs_worsened":           1,
    "avg_marginal_improvement": 0.067,
    "most_improved_pair":      "ة→ه",
    "least_improved_pair":     "ص→ض"
  }
}
```

**`direction`** values:
- `"improved"` — Phase 3 fix rate > Phase 2 fix rate
- `"unchanged"` — Δ < 0.01 (noise threshold)
- `"worsened"` — Phase 3 fix rate < Phase 2 fix rate

**Note on error type mapping**: `confusion_impact.json` works at the confusion-pair level
(e.g., `ة→ه`) while `error_changes.json` works at the error-type level (e.g., `taa_marbuta`).
For the impact analysis, the pair `ة→ه` maps to `taa_marbuta` errors. The mapping is done
using the same character groups as `ErrorAnalyzer` (`TAA_GROUP`, `DOT_GROUPS`, etc.).

If a confusion pair maps to multiple error types (e.g., `ب→ت` is both `dot_confusion` and
potentially `deletion`), the analysis uses the **substitution** count only (not
deletions/insertions).

### 10.6 `error_changes.json` (per dataset)

Identical schema to Phase 2 `error_changes.json`. Shows fixed/introduced errors by `ErrorType`.

### 10.7 `variant_summary.json` (optional, at phase3 root)

```json
{
  "meta": {
    "variants": ["results/phase3_n5", "results/phase3", "results/phase3_n20"],
    "generated_at": "..."
  },
  "by_dataset": {
    "KHATT-train": {
      "phase2_cer": 0.089,
      "n5_cer":  0.078,
      "n10_cer": 0.071,
      "n20_cer": 0.069
    }
  },
  "aggregated": {
    "phase2_mean_cer": 0.089,
    "n5_mean_cer":  0.080,
    "n10_mean_cer": 0.073,
    "n20_mean_cer": 0.071
  },
  "finding": "Diminishing returns beyond N=10. N=20 improves by only 0.002 CER vs N=10."
}
```

### 10.8 `report.md`

```markdown
# Phase 3 Report: OCR-Aware Prompting

Generated: <timestamp>
Model: Qwen/Qwen3-4B-Instruct-2507
Confusion injection: Top-10, flat_arabic format

## Summary: Phase 3 vs Phase 2 (Isolated Comparison)

| Dataset         | Phase 2 CER | Phase 3 CER | Δ CER  | Phase 2 WER | Phase 3 WER | Δ WER  |
|-----------------|-------------|-------------|--------|-------------|-------------|--------|
| KHATT-train     | 8.9%        | 7.1%        | -1.8%  | 23.4%       | 19.8%       | -3.6%  |
| KHATT-validation| X.X%        | X.X%        | -X.X%  | X.X%        | X.X%        | -X.X%  |
| PATS-A01-Akhbar | X.X%        | X.X%        | -X.X%  | X.X%        | X.X%        | -X.X%  |
| ...             |             |             |        |             |             |        |

## Top Injected Confusions and Their Impact

| Rank | GT Char | OCR Char | Count | Phase 2 Fix Rate | Phase 3 Fix Rate | Improvement |
|------|---------|----------|-------|-----------------|-----------------|-------------|
| 1    | ة       | ه        | 312   | 71.5%           | 83.6%           | +12.1%      |
| 2    | ب       | ت        | 245   | 45.3%           | 48.2%           | +2.9%       |
| ...  |         |          |       |                 |                 |             |

## Error Type Changes

[Table from error_changes.json: fixed/introduced per ErrorType]

## Confusion Matrix Source per Dataset

| Dataset | Source | Total Substitutions |
|---------|--------|---------------------|
| KHATT-train | dataset_specific | 4832 |
| PATS-A01-Akhbar-train | dataset_specific | 2341 |
| ...    |        |                     |

## Key Findings

- [Bullet: overall CER improvement vs Phase 2]
- [Bullet: which error types improved most]
- [Bullet: any zero_shot_fallback datasets]
- [Bullet: whether improvement is consistent across PATS fonts]
```

---

## 11. Configuration (`configs/config.yaml` Additions)

Add a `phase3:` block:

```yaml
# ---------------------------------------------------------------------------
# Phase 3: OCR-Aware Prompting
# ---------------------------------------------------------------------------
phase3:
  top_n: 10                      # Number of confusion pairs to inject
  format_style: "flat_arabic"    # "flat_arabic" | "grouped_arabic"
  min_substitutions_for_dataset_matrix: 200  # Sparsity fallback threshold
  analyze_errors: true           # Compute error_changes.json (adds ~10% runtime)
  analyze_impact: true           # Compute confusion_impact.json (requires phase2 error_changes)
```

These values are used as defaults and can be overridden by CLI flags (`--top-n`, `--format`,
etc.).

---

## 12. The Only Modifications to Existing Phase 2 Code

Phase 3 does not modify any Phase 1 or Phase 2 analysis code. The only changes to
**previously shipped files** are:

### 12.1 `src/core/prompt_builder.py`

**Add** two constants and one method:
- `OCR_AWARE_SYSTEM_V1` (str constant)
- `OCR_AWARE_PROMPT_VERSION` (str constant)
- `build_ocr_aware(ocr_text, confusion_context)` (method)
- `ocr_aware_prompt_version` (property)

All existing Phase 2 code (`build_zero_shot`, `ZERO_SHOT_SYSTEM_V1`, `PROMPT_VERSION`,
`prompt_version`) is **untouched**. Phase 2 pipeline behaviour is unchanged.

### 12.2 `scripts/infer.py`

**Modify** the inference loop dispatch (currently ~line 339):
- Replace hardcoded `builder.build_zero_shot(record["ocr_text"])` with a
  `prompt_type`-based dispatch block (see §6.1)
- Add `"prompt_type"` to the `out_record` dict

The `--input`, `--output`, and all other CLI arguments are unchanged. No new required
arguments. Phase 2 JSONL files without a `prompt_type` field continue to work identically.

---

## 13. Dependencies

No new dependencies beyond what Phase 1 and Phase 2 already require.

| Package | Already Required By | Notes |
|---------|--------------------|----|
| `pyyaml` | Phase 1 | Config loading |
| `editdistance` | Phase 1 | CER calculation |
| `jiwer` | Phase 1 | WER calculation |
| `tqdm` | Phase 1 | Progress bars |
| `transformers`, `torch`, `accelerate` | Phase 2 | Remote inference |

No new packages needed for Phase 3.

---

## 14. Testing

### 14.1 Test File Locations

```
tests/
├── test_knowledge_base.py          ← NEW
├── test_prompt_builder_phase3.py   ← NEW (extends Phase 2 test)
└── fixtures/
    ├── sample_confusion_matrix.json  ← NEW: small confusion matrix for unit tests
    └── sample_corrections_p3.jsonl   ← NEW: 5 phase3 correction records
```

### 14.2 `test_knowledge_base.py` — Required Test Cases

**ConfusionMatrixLoader loading:**
- `test_load_returns_confusion_pairs_sorted_by_count`
- `test_load_raises_on_missing_file`
- `test_load_uses_top20_precomputed_list_when_available`
- `test_load_pooled_sums_counts_across_matrices`
- `test_load_pooled_recomputes_probability`
- `test_has_enough_data_returns_false_below_threshold`
- `test_has_enough_data_returns_true_above_threshold`
- `test_has_enough_data_returns_false_on_missing_file`

**ConfusionMatrixLoader formatting:**
- `test_format_flat_arabic_produces_expected_header`
- `test_format_flat_arabic_limits_to_top_n`
- `test_format_flat_arabic_includes_probability_percentage`
- `test_format_returns_empty_string_for_empty_pairs`
- `test_format_grouped_arabic_groups_dot_confusion_pairs`

### 14.3 `test_prompt_builder_phase3.py` — Required Test Cases

- `test_build_ocr_aware_returns_two_messages`
- `test_build_ocr_aware_system_contains_confusion_context`
- `test_build_ocr_aware_user_contains_ocr_text`
- `test_build_ocr_aware_empty_context_falls_back_to_zero_shot`
- `test_build_ocr_aware_whitespace_only_context_falls_back`
- `test_ocr_aware_prompt_version_is_p3v1`
- `test_zero_shot_prompt_version_unchanged` (regression: still "v1")
- `test_build_zero_shot_unchanged_after_phase3_additions` (regression)

### 14.4 `infer.py` Dispatch Tests

Add to `tests/test_infer.py` (or create if absent):

- `test_infer_dispatches_ocr_aware_when_prompt_type_ocr_aware`
- `test_infer_dispatches_zero_shot_when_prompt_type_missing`
- `test_infer_dispatches_zero_shot_when_prompt_type_zero_shot`
- `test_infer_logs_warning_on_unknown_prompt_type`
- `test_out_record_includes_prompt_type_field`

---

## 15. Implementation Order

| Step | File | Depends On | Notes |
|------|------|-----------|-------|
| 1 | `src/data/knowledge_base.py` | Nothing | `ConfusionMatrixLoader` only; stub RulesLoader/QALBLoader |
| 2 | `tests/test_knowledge_base.py` | Step 1 | Use `fixtures/sample_confusion_matrix.json` |
| 3 | `src/core/prompt_builder.py` (add `build_ocr_aware`) | Step 1 | New method + constants only; zero regression risk |
| 4 | `tests/test_prompt_builder_phase3.py` | Step 3 | Confirm no regression on Phase 2 tests too |
| 5 | `scripts/infer.py` (add dispatch) | Step 3 | Small, backward-compatible change |
| 6 | Tests for `infer.py` dispatch | Step 5 | No GPU required |
| 7 | `pipelines/run_phase3.py` | Steps 1–5 | Export + Analyze modes |
| 8 | Smoke test — export mode | Step 7 + Phase 1 results | `--mode export --limit 5` |
| 9 | Remote inference on Kaggle/Colab | Step 5 + GPU | `scripts/infer.py --input phase3/... --output phase3/...` |
| 10 | Smoke test — analyze mode | Step 9 output | `--mode analyze --datasets KHATT-train --no-error-analysis` |
| 11 | Full run — all 18 datasets | Step 10 confirmed | Collect paper numbers |
| 12 | Optional variants (N=5, N=20) | Step 11 | Only if time permits |

**Smoke test commands**:

```bash
# Step 8: export (requires results/phase1/ to exist)
python pipelines/run_phase3.py --mode export --limit 5

# Step 10: analyze (after downloading corrections from Kaggle)
python pipelines/run_phase3.py --mode analyze \
    --datasets KHATT-train \
    --no-error-analysis \
    --no-confusion-impact
```

---

## 16. Known Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Phase 1 confusion matrix not found for a dataset | Medium | `resolve_confusion_matrix()` falls back to pooled; logs WARNING; records `confusion_source` |
| Pooled matrix also empty (Phase 1 never ran) | Low | `build_ocr_aware()` gets `confusion_context=""` → falls back to zero-shot; `zero_shot_fallback_samples` counter warns in report |
| LLM ignores the confusion context entirely | Possible | This is a valid experimental result — report it; do not force-filter |
| LLM over-corrects a character mentioned in confusion list | Possible | `confusion_impact.json` will show `"direction": "worsened"` for that pair — report it |
| Context window exceeded by long OCR + long confusion list | Low | `MAX_INPUT_TOKENS=512` guard in `TransformersCorrector` still applies; N=10 confusion text is ~300 chars |
| Phase 2 `error_changes.json` missing (impact analysis fails) | Medium | `run_confusion_impact_analysis()` checks path exists; if missing, skip impact analysis and log WARNING rather than crashing |
| Different `top_n` values produce incomparable results | Low | Each variant writes to its own results dir; `variant_summary.json` provides the cross-N comparison |
| PATS font-specific vs pooled confusion matrix gives misleading results | Low | `confusion_source` field in every record makes this traceable; report lists source per dataset |
| Kaggle session timeout mid-run | Medium | `infer.py` write-per-line + resume logic unchanged from Phase 2 |

---

## 17. Appendix A: Prompt Text

### System Message (OCR-Aware, v1 — N=10, flat_arabic, KHATT example)

The final system prompt sent to the model is the template with `{confusion_context}` filled in:

```
أنت مصحح نصوص عربية متخصص في تصحيح مخرجات نظام Qaari للتعرف الضوئي.
فيما يلي أبرز الأخطاء الشائعة التي يرتكبها هذا النظام:

أخطاء Qaari الشائعة في التعرف على الحروف:
- يستبدل (ة) بـ (ه) في 85% من الحالات
- يستبدل (ب) بـ (ت) في 32% من الحالات
- يستبدل (أ) بـ (ا) في 28% من الحالات
- يستبدل (ي) بـ (ى) في 21% من الحالات
- يستبدل (ح) بـ (ج) في 14% من الحالات
- يستبدل (ن) بـ (ب) في 11% من الحالات
- يستبدل (إ) بـ (ا) في 10% من الحالات
- يستبدل (ذ) بـ (د) في 9% من الحالات
- يستبدل (ز) بـ (ر) في 8% من الحالات
- يستبدل (ث) بـ (ت) في 7% من الحالات

صحح النص التالي مع الانتباه بشكل خاص لهذه الأخطاء.
أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي.
```

### User Message

```
[The raw OCR output text]
```

### Why This Prompt Design

- **"نظام Qaari"**: Names the specific OCR system. The LLM knows this is not random
  noise — these are systematic patterns from one engine.
- **"أبرز الأخطاء الشائعة"**: "Most notable common errors" — frames this as a briefing,
  not an exhaustive list. Avoids the model over-fixating on listed pairs.
- **Flat list with percentages**: Probabilities give the model a sense of how much to
  prioritize each confusion. High-probability pairs (85%) deserve more attention than
  low-probability ones (7%).
- **"مع الانتباه بشكل خاص"**: "with special attention to" — reinforces targeted awareness
  without constraining the model to only fix listed errors.
- **"أعد النص المصحح فقط"**: Same as Phase 2 — prevents explanation output.

### Comparison to Phase 2 Zero-Shot System Prompt

| Aspect | Phase 2 | Phase 3 |
|--------|---------|---------|
| Mentions Qaari | No | Yes |
| Injects confusion pairs | No | Yes (top N) |
| Instruction to return only text | Yes | Yes |
| Prompt length (approx.) | ~25 tokens | ~150–200 tokens |
| Prompt version | `v1` | `p3v1` |
