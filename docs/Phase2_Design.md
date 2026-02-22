# Phase 2: Zero-Shot LLM Correction — Design Document

## 1. Overview

### 1.1 Purpose

Phase 2 applies a **vanilla LLM to Arabic OCR correction with no task-specific knowledge injected**.
It answers: *"What can a 3B-parameter model do out-of-the-box?"*

This phase is the **critical hub of the entire experiment**. Every subsequent phase (3, 4A, 4B, 4C, 5)
compares against Phase 2's CER/WER numbers. Getting Phase 2 right — deterministic, reproducible,
comparable — is more important than for any other phase.

### 1.2 Research Questions Answered

| Question | Output Artefact |
|----------|----------------|
| Can a vanilla LLM correct Arabic OCR errors? | `comparison.json` (Δ CER/WER vs Phase 1) |
| How much CER/WER does zero-shot correction achieve? | `metrics.json` |
| Which error types does the LLM fix without guidance? | `error_changes.json` |
| Does the LLM introduce new errors? | `error_changes.json` (`errors_introduced`) |
| Is correction better on PATS-A01 or KHATT? | `comparison.json` per dataset |

### 1.3 Downstream Use

Phase 2 outputs are consumed by every subsequent phase:

- **Phases 3, 4A, 4B, 4C, 5**: Load `comparison.json` as the comparison baseline
- **Phase 4C**: Can load `corrections.jsonl` directly to apply CAMeL validation on top
- **Phase 6**: Uses Phase 2 as lower bound in the ablation table
- **Paper**: "Zero-shot LLM" row in the main results table

### 1.4 Compute Environment

> **No local GPU is available.** LLM inference runs on **Kaggle** or **Google Colab**.
> See `docs/Guidelines.md §12` for the full compute environment policy.

The pipeline is therefore split into three stages:

| Stage | Runs On | What It Does |
|-------|---------|--------------|
| `--mode export` | Local | Exports OCR texts → `inference_input.jsonl` |
| Inference script | Kaggle / Colab | Loads model, corrects texts → `corrections.jsonl` |
| `--mode analyze` | Local | Loads corrections, computes metrics, generates reports |

For future API-based inference (e.g., OpenAI, Anthropic), `--mode full` runs all three
stages locally with an `APICorrector` backend.

---

## 2. Data

### 2.1 Datasets (Same as Phase 1)

Phase 2 processes the same four datasets as Phase 1. No changes to data loading logic.

| Dataset Key | Type | Approx. Samples |
|-------------|------|-----------------|
| PATS-A01-Akhbar | Typewritten/Synthetic | ~2,766 |
| PATS-A01-Andalus | Typewritten/Synthetic | ~2,599 |
| KHATT-train | Handwritten/Real | ~1,400 |
| KHATT-validation | Handwritten/Real | ~233 |

### 2.2 Data Flow

```
LOCAL (export stage)
────────────────────────────────────────────────
DataLoader.iter_samples(dataset_key)
        │
        │  OCRSample(ocr_text, gt_text, sample_id, ...)
        ▼
PromptBuilder.build_zero_shot(ocr_text)  [optional: pre-build prompts for export]
        │
        ▼
inference_input.jsonl  ← {sample_id, ocr_text, gt_text, dataset}
        │
        │  [uploaded to Kaggle/Colab]
        ▼

REMOTE (inference stage — Kaggle/Colab)
────────────────────────────────────────────────
scripts/run_inference.py reads inference_input.jsonl
        │
        ▼
PromptBuilder.build_zero_shot(ocr_text)
        │
        ▼
TransformersCorrector.correct(sample_id, ocr_text, messages)
        │
        ▼
corrections.jsonl  ← {sample_id, corrected_text, gt_text, success, ...}
        │
        │  [downloaded from Kaggle/Colab]
        ▼

LOCAL (analyze stage)
────────────────────────────────────────────────
Load corrections.jsonl → list[CorrectedSample]
        │
        ├──► calculate_metrics(corrected_samples, text_field="corrected_text")
        ├──► compare_metrics(phase1_baseline, corrected_metrics)
        └──► run_error_change_analysis(corrected_samples)
```

---

## 3. Module Design

### 3.1 New Modules

```
src/
└── core/
    ├── llm_corrector.py       ← NEW: BaseLLMCorrector (ABC), TransformersCorrector,
    │                                 CorrectionResult, CorrectedSample, get_corrector()
    ├── api_corrector.py       ← NEW: APICorrector (future — stub provided, not implemented)
    └── prompt_builder.py      ← NEW: PromptBuilder (zero-shot for Phase 2; extensible for 3–5)

scripts/
└── run_inference.py           ← NEW: Standalone inference script for Kaggle/Colab
```

### 3.2 Reused Modules (No Modifications)

```
src/data/data_loader.py         ← DataLoader.iter_samples() — unchanged
src/data/text_utils.py          ← normalise_arabic() — unchanged
src/analysis/error_analyzer.py  ← ErrorAnalyzer — unchanged
```

### 3.3 Modules Requiring Minor Modification

```
src/analysis/metrics.py         ← add optional text_field parameter to calculate_metrics()
```

### 3.4 Pipeline Entry Point

```
pipelines/run_phase2.py         ← NEW: orchestrates export + analyze stages
```

---

## 4. `src/core/prompt_builder.py`

The `PromptBuilder` constructs chat message lists in OpenAI format. Phase 2 uses only
`build_zero_shot()`. Later phases will add more `build_*` methods to this class.

**Design principle**: No knowledge injection in Phase 2. The system prompt instructs the
model to correct OCR text and return **only** the corrected text — no explanations.

### 4.1 Full API

```python
class PromptBuilder:
    """Build phase-specific chat message lists for LLM correction.

    Returns list[dict] in OpenAI chat format. These are passed to any
    BaseLLMCorrector implementation — the builder is backend-agnostic.

    Phase 2 usage:
        builder = PromptBuilder()
        messages = builder.build_zero_shot(ocr_text="النص...")
        result = corrector.correct(sample_id, ocr_text, messages)

    Extension: Later phases add build_ocr_aware(), build_rule_augmented(), etc.
    All methods return the same list[dict] format regardless of backend.
    """

    # -----------------------------------------------------------------------
    # Prompt constants — versioned so changes are detectable in results
    # -----------------------------------------------------------------------

    ZERO_SHOT_SYSTEM_V1: str = (
        "أنت مصحح نصوص عربية متخصص. "
        "مهمتك تصحيح أخطاء التعرف الضوئي (OCR) في النص العربي. "
        "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
    )

    PROMPT_VERSION: str = "v1"

    def build_zero_shot(self, ocr_text: str) -> list[dict]:
        """Build zero-shot correction prompt (Phase 2).

        No task-specific knowledge injected. The LLM receives:
          - A general Arabic OCR correction instruction (system)
          - The raw OCR text to correct (user)

        Args:
            ocr_text: OCR prediction text to correct.

        Returns:
            Messages list for tokenizer.apply_chat_template() or API calls:
            [
              {"role": "system", "content": ZERO_SHOT_SYSTEM_V1},
              {"role": "user",   "content": ocr_text}
            ]
        """

    @property
    def prompt_version(self) -> str:
        """Return the current prompt version string (for metadata logging)."""
        return self.PROMPT_VERSION
```

### 4.2 Design Notes

- Messages are in **OpenAI chat format** — compatible with both HuggingFace
  `tokenizer.apply_chat_template()` and OpenAI-compatible API calls. This is the key
  abstraction that makes `PromptBuilder` backend-agnostic.
- `ZERO_SHOT_SYSTEM_V1` is a class constant so prompt changes are visible in code review.
  Increment `PROMPT_VERSION` whenever the wording changes between runs.

---

## 5. `src/core/llm_corrector.py` — Corrector Backend Architecture

This module defines the **abstract interface** (`BaseLLMCorrector`) and the primary
implementation (`TransformersCorrector`) used on Kaggle/Colab. A factory function
(`get_corrector`) selects the backend from config.

### 5.1 Data Structures

```python
from dataclasses import dataclass
from typing import Optional

from src.data.data_loader import OCRSample


@dataclass
class CorrectionResult:
    """Result of LLM correction for a single sample."""
    sample_id: str
    ocr_text: str
    corrected_text: str        # LLM output (or ocr_text if failed)
    prompt_tokens: int         # tokens consumed by the prompt
    output_tokens: int         # tokens generated by the model
    latency_s: float           # wall-clock seconds for this call
    success: bool              # False if generation failed or output was empty
    error: Optional[str]       # error message if success=False, else None


@dataclass
class CorrectedSample:
    """OCRSample extended with LLM correction result.

    Used throughout the Phase 2 analysis pipeline.
    The sample.gt_text field provides ground truth for metric calculation.
    """
    sample: OCRSample          # original Phase 1 sample (ocr_text, gt_text, etc.)
    corrected_text: str        # LLM corrected output
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    success: bool
    error: Optional[str] = None
```

### 5.2 `BaseLLMCorrector` — Abstract Interface

```python
from abc import ABC, abstractmethod


class BaseLLMCorrector(ABC):
    """Abstract base class for all LLM corrector backends.

    Any backend (local transformers, OpenAI API, Anthropic API, etc.) must
    implement this interface. The pipeline code depends only on this class,
    never on a specific implementation.

    Implementations provided:
        TransformersCorrector   — HuggingFace transformers (runs on Kaggle/Colab)
        APICorrector            — OpenAI-compatible REST API (future)

    Usage:
        corrector = get_corrector(config)   # factory selects backend
        result = corrector.correct(sample_id, ocr_text, messages)
    """

    @abstractmethod
    def correct(
        self,
        sample_id: str,
        ocr_text: str,
        messages: list[dict],
        max_retries: int = 2,
    ) -> CorrectionResult:
        """Correct a single OCR text using the provided chat messages.

        Implementations must:
          - Return a CorrectionResult in all cases (never raise on failure)
          - Fall back to ocr_text as corrected_text when success=False
          - Set error to a descriptive string when success=False

        Args:
            sample_id: Identifier for logging and result attribution.
            ocr_text: Original OCR text (stored in result for reference).
            messages: Chat messages list in OpenAI format (system + user).
            max_retries: Retry count on empty/failed generation.

        Returns:
            CorrectionResult with corrected_text always populated.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a string identifying the model (for metadata and logging)."""
```

### 5.3 `TransformersCorrector` — HuggingFace Backend

This is the primary implementation, designed to run on Kaggle/Colab kernels.

```python
import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TransformersCorrector(BaseLLMCorrector):
    """LLM corrector backed by HuggingFace transformers.

    Loads model and tokenizer ONCE at initialisation. Use correct() for
    repeated inference across all samples.

    Designed to run on Kaggle/Colab GPU kernels. Supports optional 4-bit
    quantization (bitsandbytes) for GPUs with limited VRAM.

    Config keys read:
        config['model']['name']          → HuggingFace model ID
        config['model']['temperature']   → float (default 0.1)
        config['model']['max_tokens']    → int (default 1024)
        config['model']['device']        → "auto" | "cuda" | "cpu"
        config['model']['quantize_4bit'] → bool (default False)
    """

    MAX_INPUT_TOKENS: int = 512    # guard against OOM on unexpectedly long inputs

    def __init__(self, config: dict) -> None:
        """Load model and tokenizer from config.

        Raises:
            RuntimeError: If model loading fails.
        """

    def correct(
        self,
        sample_id: str,
        ocr_text: str,
        messages: list[dict],
        max_retries: int = 2,
    ) -> CorrectionResult:
        """Correct a single OCR text. Retries on empty output. Never raises."""

    @property
    def model_name(self) -> str:
        """Return the loaded model's HuggingFace ID."""

    # -----------------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------------

    def _load_model(self, config: dict) -> None:
        """Load tokenizer and model. Called once in __init__.

        Loading steps:
          1. AutoTokenizer.from_pretrained(model_name)
          2. If quantize_4bit=True: BitsAndBytesConfig(load_in_4bit=True, nf4)
             Otherwise: torch_dtype=torch.float16, device_map=device
          3. model.eval()
        """

    def _generate(self, messages: list[dict]) -> tuple[str, int, int]:
        """Tokenise, generate, decode. Returns (text, prompt_tokens, output_tokens).

        Steps:
          1. tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
          2. tokenizer(formatted_prompt, return_tensors="pt") → input_ids
          3. Truncate input_ids to MAX_INPUT_TOKENS if needed (warn if truncated)
          4. model.generate(input_ids, max_new_tokens, do_sample=True,
                            temperature, pad_token_id=eos_token_id)
          5. Extract only new tokens: output_ids[:, input_ids.shape[1]:]
          6. tokenizer.decode(new_tokens, skip_special_tokens=True)
        """

    def _extract_corrected_text(self, raw_output: str, ocr_text: str) -> str:
        """Strip whitespace, validate Arabic content. Fall back to ocr_text if needed.

        Edge cases handled:
          - Empty string after strip → fallback + warning
          - No Arabic characters in output → fallback + warning
            (guards against the model replying in English or refusing)
        """
```

### 5.4 `APICorrector` — Future API Backend (Stub)

Defined in `src/core/api_corrector.py`. Provided as a stub now so the interface contract
is established and the factory function can reference it. Implementation is left for when
API access is needed.

```python
# src/core/api_corrector.py

import logging
from typing import Optional

from src.core.llm_corrector import BaseLLMCorrector, CorrectionResult

logger = logging.getLogger(__name__)


class APICorrector(BaseLLMCorrector):
    """LLM corrector backed by an OpenAI-compatible REST API.

    Works with any service that implements the OpenAI chat completions API:
    - OpenAI (GPT-4o, GPT-3.5, etc.)
    - Anthropic Claude (via compatibility layer)
    - Local vLLM or Ollama server
    - Any provider with an OpenAI-compatible endpoint

    Config keys read:
        config['model']['name']        → model name for the API (e.g., "gpt-4o")
        config['api']['base_url']      → API base URL
        config['api']['api_key_env']   → environment variable holding the API key
        config['api']['timeout_s']     → request timeout in seconds (default 30)
        config['api']['requests_per_minute'] → rate limit (default 60)

    Security: The API key is NEVER stored in config files. It is read from
    the environment variable named by config['api']['api_key_env'].

    Status: STUB — not yet implemented. The correct() method raises
    NotImplementedError. Implement when API access is available.
    """

    def __init__(self, config: dict) -> None:
        """Initialise API client from config.

        Raises:
            ValueError: If the API key environment variable is not set.
            NotImplementedError: Always (stub not yet implemented).
        """
        raise NotImplementedError(
            "APICorrector is not yet implemented. "
            "Use backend='transformers' and run inference on Kaggle/Colab."
        )

    def correct(
        self,
        sample_id: str,
        ocr_text: str,
        messages: list[dict],
        max_retries: int = 2,
    ) -> CorrectionResult:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError
```

### 5.5 `get_corrector` — Factory Function

Located at the bottom of `src/core/llm_corrector.py`:

```python
def get_corrector(config: dict) -> BaseLLMCorrector:
    """Factory: instantiate the corrector backend specified in config.

    Config key: config['model']['backend']
        "transformers"  → TransformersCorrector (default)
        "api"           → APICorrector

    Args:
        config: Parsed config.yaml dict.

    Returns:
        Initialised BaseLLMCorrector implementation.

    Raises:
        ValueError: If backend value is not recognised.
    """
    backend = config.get("model", {}).get("backend", "transformers")

    if backend == "transformers":
        return TransformersCorrector(config)
    elif backend == "api":
        from src.core.api_corrector import APICorrector
        return APICorrector(config)
    else:
        raise ValueError(
            f"Unknown corrector backend '{backend}'. "
            f"Valid values: 'transformers', 'api'."
        )
```

---

## 6. Remote Inference Workflow

### 6.1 `scripts/run_inference.py` — Kaggle/Colab Inference Script

This is a **standalone script** that runs entirely on Kaggle/Colab. It does not depend on
`DataLoader` or any other module that requires local data files. Its only inputs are:

- `inference_input.jsonl` (uploaded to Kaggle/Colab)
- The model (downloaded from HuggingFace on the Kaggle kernel)
- `src/core/prompt_builder.py` and `src/core/llm_corrector.py` (uploaded to Kaggle)

```python
#!/usr/bin/env python3
"""Standalone inference script for Kaggle/Colab.

Reads inference_input.jsonl, corrects each OCR text using a local HuggingFace
model, and writes corrections.jsonl.

This script is designed to be uploaded to and run on Kaggle/Colab kernels.
No database, local GT files, or project-level DataLoader is needed.

Usage (on Kaggle/Colab):
    python run_inference.py
    python run_inference.py --input inference_input.jsonl \
                            --output corrections.jsonl \
                            --model Qwen/Qwen2.5-3B-Instruct \
                            --limit 100
"""

import argparse
import json
import logging
import time
from pathlib import Path

# These two modules are the only project imports needed on the remote machine
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import TransformersCorrector, CorrectionResult


def parse_args() -> argparse.Namespace: ...

def main() -> None:
    """
    Flow:
    1. parse_args()
    2. Load inference_input.jsonl → list of dicts
    3. Initialise TransformersCorrector from a minimal config dict
    4. Initialise PromptBuilder
    5. For each record (tqdm progress):
       a. Skip if already in output file (resume support)
       b. Build zero-shot messages
       c. corrector.correct(sample_id, ocr_text, messages)
       d. Append to corrections.jsonl (one line per sample, written immediately)
    6. Print summary: N corrected, N failed, avg latency
    """
```

**Key properties of `run_inference.py`:**

- Writes to `corrections.jsonl` **line by line** (not all at end) — so partial results
  are preserved if the Kaggle session times out
- Supports **resume**: reads already-written output lines on startup, skips those sample IDs
- Accepts a minimal config dict inline (no `configs/config.yaml` needed on remote machine)
- Single file — easy to upload; no complex project structure required on Kaggle

### 6.2 Export Step (`--mode export`)

Produces `results/phase2/inference_input.jsonl` for upload to Kaggle/Colab.

Each line contains everything the remote machine needs to perform one correction:

```json
{
  "sample_id":  "AHTD3A0001_Para2_3",
  "dataset":    "KHATT-train",
  "ocr_text":   "... raw OCR prediction ...",
  "gt_text":    "... ground truth text ..."
}
```

**Why include `gt_text`?**
The Kaggle kernel does not have access to the GT file directory. Including `gt_text` in the
export means `corrections.jsonl` can carry it forward, so the local analysis step has all it
needs without re-loading the DataLoader.

### 6.3 Analyse Step (`--mode analyze`)

Reads `corrections.jsonl` from the per-dataset output directory and runs all metric/analysis
steps locally. No LLM or GPU is needed.

Input assumed at: `results/phase2/{dataset_key}/corrections.jsonl`

---

## 7. `pipelines/run_phase2.py`

### 7.1 Entry Point

```python
#!/usr/bin/env python3
"""Phase 2: Zero-Shot LLM Correction.

Three-stage pipeline:
  export  → prepare inference_input.jsonl for upload to Kaggle/Colab
  analyze → load corrections.jsonl, compute metrics and reports (local)
  full    → run end-to-end using API backend (future)

Usage:
    # Stage 1: prepare data for Kaggle/Colab
    python pipelines/run_phase2.py --mode export

    # Stage 2: run on Kaggle/Colab (see scripts/run_inference.py)

    # Stage 3: analyze downloaded corrections
    python pipelines/run_phase2.py --mode analyze

    # Future: end-to-end with API backend
    python pipelines/run_phase2.py --mode full

    # Useful flags:
    python pipelines/run_phase2.py --mode export --limit 50
    python pipelines/run_phase2.py --mode analyze --dataset KHATT-train
    python pipelines/run_phase2.py --mode analyze --no-error-analysis
"""
```

### 7.2 CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | **required** | `export` \| `analyze` \| `full` |
| `--limit` | int | None | Max samples per dataset (testing) |
| `--dataset` | str | None | Run only one dataset |
| `--force` | flag | False | Re-export / re-analyze even if files exist |
| `--no-error-analysis` | flag | False | Skip error_changes.json (faster) |
| `--config` | path | `configs/config.yaml` | Config file path |
| `--results-dir` | path | `results/phase2` | Output directory |
| `--phase1-dir` | path | `results/phase1` | Phase 1 results for comparison |

### 7.3 Execution Flow — Export Mode

```
run_phase2.py --mode export
│
├─ 1. parse_args() + setup_logging() + load_config()
│
├─ 2. Initialise DataLoader(config)
│
├─ 3. For each dataset_key:
│     │
│     ├─ loader.iter_samples(dataset_key, limit=limit) → samples
│     │
│     └─ Write each sample to results/phase2/inference_input.jsonl:
│           {"sample_id": ..., "dataset": ..., "ocr_text": ..., "gt_text": ...}
│
└─ 4. Log: "Upload results/phase2/inference_input.jsonl to Kaggle/Colab.
           Then run: python scripts/run_inference.py
           Then download corrections.jsonl to results/phase2/{dataset_key}/"
```

### 7.4 Execution Flow — Analyze Mode

```
run_phase2.py --mode analyze
│
├─ 1. parse_args() + setup_logging() + load_config()
│
├─ 2. Load Phase 1 baseline (results/phase1/baseline_metrics.json)
│     If missing: warn, set phase1_metrics = None
│
├─ 3. For each dataset_key:
│     │
│     ├─ 3a. Load results/phase2/{dataset_key}/corrections.jsonl
│     │       → list[CorrectedSample]
│     │       Abort with clear error if file missing (user must download it)
│     │
│     ├─ 3b. calculate_metrics(corrected_samples, text_field="corrected_text")
│     │       save: results/phase2/{dataset_key}/metrics.json
│     │
│     ├─ 3c. compare_metrics(phase1_baseline, corrected_metrics)
│     │       save: results/phase2/{dataset_key}/comparison_vs_phase1.json
│     │
│     └─ 3d. [if not --no-error-analysis]
│             run_error_change_analysis(corrected_samples)
│             save: results/phase2/{dataset_key}/error_changes.json
│
├─ 4. Aggregate → results/phase2/metrics.json
│
├─ 5. Aggregate comparisons → results/phase2/comparison.json
│
├─ 6. generate_report() → results/phase2/report.md
│
└─ 7. Print summary table
```

### 7.5 Execution Flow — Full Mode (Future / API)

```
run_phase2.py --mode full
│
├─ 1–3. Same as export mode setup
│
├─ 4. get_corrector(config)   ← must be backend="api"; raises if "transformers"
│       (TransformersCorrector is not supported in full mode — requires remote GPU)
│
├─ 5. PromptBuilder()
│
├─ 6. For each dataset_key:
│     │
│     ├─ 6a. loader.iter_samples() → samples
│     │
│     ├─ 6b. For each sample:
│     │       - Build prompt: builder.build_zero_shot(ocr_text)
│     │       - corrector.correct(sample_id, ocr_text, messages)
│     │       - Append to corrections.jsonl immediately (resume support)
│     │
│     └─ 6c. Same as analyze steps 3b–3d
│
└─ 7–9. Same as analyze mode steps 4–7
```

### 7.6 Key Functions

```python
def run_export(
    config: dict,
    results_dir: Path,
    limit: int | None,
    dataset_filter: str | None,
) -> None:
    """Export OCR texts to inference_input.jsonl for upload to Kaggle/Colab."""


def run_analyze(
    config: dict,
    results_dir: Path,
    phase1_dir: Path,
    dataset_filter: str | None,
    analyze_errors: bool,
) -> None:
    """Load corrections.jsonl, calculate metrics, generate reports."""


def load_corrections(
    corrections_path: Path,
) -> list[CorrectedSample]:
    """Read corrections.jsonl into CorrectedSample objects.

    Args:
        corrections_path: Path to corrections.jsonl file.

    Returns:
        List of CorrectedSample objects.

    Raises:
        FileNotFoundError: With clear message instructing user to download
            the file from Kaggle/Colab before running --mode analyze.
    """


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
) -> dict:
    """Compare per-type error counts before (OCR) and after (corrected) correction.

    Runs ErrorAnalyzer on (gt, ocr) and (gt, corrected) for each sample.
    Aggregates per-ErrorType fixed/introduced counts across all samples.
    """


def generate_report(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    all_error_changes: dict[str, dict],
    model_name: str,
    results_dir: Path,
) -> None:
    """Write human-readable Markdown report to results_dir/report.md."""
```

---

## 8. Output Structure

```
results/phase2/
├── inference_input.jsonl            # ← UPLOAD THIS to Kaggle/Colab (export stage)
│
├── PATS-A01-Akhbar/
│   ├── corrections.jsonl            # ← DOWNLOAD THIS from Kaggle/Colab (inference output)
│   ├── metrics.json                 # Post-correction CER/WER (analyze stage)
│   ├── comparison_vs_phase1.json    # Delta vs Phase 1 OCR baseline
│   └── error_changes.json           # Fixed vs introduced errors by type
├── PATS-A01-Andalus/
│   └── [same structure]
├── KHATT-train/
│   └── [same structure]
├── KHATT-validation/
│   └── [same structure]
│
├── metrics.json                     # Aggregated CER/WER across all datasets
├── comparison.json                  # Aggregated comparison across all datasets
├── phase2.log                       # Full run log
└── report.md                        # Human-readable summary
```

**Note on individual `.txt` files**: The original design saved individual corrected text files
(`corrected/{sample_id}.txt`). With the two-stage workflow, `corrections.jsonl` is the primary
output and individual file saving is dropped for simplicity. The JSONL already contains all
corrected texts in a machine-readable format.

---

## 9. Output Schemas

### 9.1 Standard `meta` Block (Phase 2)

```json
{
  "meta": {
    "phase": "phase2",
    "dataset": "KHATT-train",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "backend": "transformers",
    "prompt_type": "zero_shot",
    "prompt_version": "v1",
    "generated_at": "2026-02-21T10:00:00Z",
    "git_commit": "abc1234",
    "num_samples": 1400,
    "limit_applied": null
  }
}
```

### 9.2 `inference_input.jsonl` (export stage output)

One JSON object per line. This file is uploaded to Kaggle/Colab.

```json
{"sample_id": "AHTD3A0001_Para2_3", "dataset": "KHATT-train", "ocr_text": "...", "gt_text": "..."}
```

### 9.3 `corrections.jsonl` (inference stage output — the contract file)

One JSON object per line. This file is downloaded from Kaggle/Colab.

```json
{
  "sample_id": "AHTD3A0001_Para2_3",
  "dataset": "KHATT-train",
  "ocr_text": "... raw OCR prediction ...",
  "corrected_text": "... LLM corrected output ...",
  "gt_text": "... ground truth ...",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "prompt_version": "v1",
  "prompt_tokens": 89,
  "output_tokens": 92,
  "latency_s": 2.31,
  "success": true,
  "error": null
}
```

`gt_text` is included so the analyze stage has all it needs from this one file.

### 9.4 `metrics.json` (per dataset, analyze stage)

```json
{
  "meta": {
    "phase": "phase2",
    "dataset": "KHATT-train",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "backend": "transformers",
    "prompt_type": "zero_shot",
    "prompt_version": "v1",
    "generated_at": "...",
    "git_commit": "...",
    "num_samples": 1400,
    "limit_applied": null,
    "total_prompt_tokens": 124800,
    "total_output_tokens": 131200,
    "total_latency_s": 3240.5,
    "avg_latency_per_sample_s": 2.31,
    "failed_samples": 3
  },
  "corrected": {
    "cer": 0.089,
    "wer": 0.234,
    "cer_std": 0.067,
    "wer_std": 0.189,
    "cer_median": 0.071,
    "wer_median": 0.198,
    "cer_p95": 0.234,
    "wer_p95": 0.623,
    "num_samples": 1400,
    "num_chars_ref": 158432,
    "num_words_ref": 28901
  }
}
```

### 9.5 `comparison_vs_phase1.json` (per dataset)

```json
{
  "meta": { "..." },
  "phase1_ocr": {
    "cer": 0.142,
    "wer": 0.387,
    "source": "results/phase1/KHATT-train/metrics.json"
  },
  "phase2_corrected": {
    "cer": 0.089,
    "wer": 0.234
  },
  "delta": {
    "cer_absolute": -0.053,
    "wer_absolute": -0.153,
    "cer_relative_pct": -37.3,
    "wer_relative_pct": -39.5
  },
  "interpretation": "CER reduced by 37.3% (14.2% → 8.9%). WER reduced by 39.5% (38.7% → 23.4%)."
}
```

Convention: **negative delta = improvement** (lower error = better).

### 9.6 `error_changes.json` (per dataset)

```json
{
  "meta": {
    "note": "Compares ErrorAnalyzer output on (gt, ocr) vs (gt, corrected) per sample.",
    "..."
  },
  "summary": {
    "total_ocr_char_errors": 7023,
    "total_corrected_char_errors": 4412,
    "net_reduction": 2611,
    "errors_fixed": 3156,
    "errors_introduced": 545,
    "fix_rate": 0.449,
    "introduction_rate": 0.078,
    "net_fix_rate": 0.372
  },
  "by_type": {
    "taa_marbuta": {
      "phase1_count": 312,
      "phase2_count": 89,
      "fixed": 223,
      "introduced": 0,
      "fix_rate": 0.715
    },
    "hamza": { "phase1_count": 278, "phase2_count": 134, "fixed": 187,
               "introduced": 43, "fix_rate": 0.673 },
    "dot_confusion":       { "..." },
    "alef_maksura":        { "..." },
    "deletion":            { "..." },
    "insertion":           { "..." },
    "similar_shape":       { "..." },
    "merged_words":        { "..." },
    "split_word":          { "..." },
    "other_substitution":  { "..." }
  }
}
```

### 9.7 `metrics.json` and `comparison.json` (aggregated, at phase2 root)

Same structure as Phase 1's `baseline_metrics.json` but with Phase 2 corrected results.
Full schema omitted — see per-dataset schemas above.

---

## 10. `configs/config.yaml` Additions

Add these keys (some already exist — annotated):

```yaml
# ---------------------------------------------------------------------------
# Model settings (already present — add backend and quantize_4bit)
# ---------------------------------------------------------------------------
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  backend: "transformers"    # NEW: "transformers" | "api"
  temperature: 0.1
  max_tokens: 1024
  device: "auto"
  quantize_4bit: false       # NEW: Enable for Kaggle GPUs with <8GB VRAM

# ---------------------------------------------------------------------------
# API backend settings (used when model.backend = "api") — NEW
# ---------------------------------------------------------------------------
api:
  base_url: null             # e.g., "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"   # environment variable name (key never hardcoded)
  timeout_s: 30
  requests_per_minute: 60   # rate limiting to avoid quota errors

# ---------------------------------------------------------------------------
# Phase 2 specific — NEW
# ---------------------------------------------------------------------------
phase2:
  prompt_version: "v1"      # Track prompt version; update if wording changes
  analyze_errors: true       # Run error_changes.json analysis (adds ~10% runtime)
  max_retries: 2             # Inference retry count on empty/failed output
```

---

## 11. The Only Modification to Existing Phase 1 Code

Phase 2 **must not change Phase 1 modules** to preserve Phase 1 reproducibility. The single
required change is adding an optional `text_field` parameter to `calculate_metrics()`:

```python
# src/analysis/metrics.py — add one optional parameter with a backward-compatible default

def calculate_metrics(
    samples: list,
    dataset_name: str,
    text_field: str = "ocr_text",   # NEW — default preserves Phase 1 behaviour exactly
    normalise: bool = True,
) -> MetricResult:
    """
    For Phase 1 (text_field="ocr_text"): reads sample.ocr_text as hypothesis.
    For Phase 2 (text_field="corrected_text"): reads corrected_sample.corrected_text.

    Resolves the attribute by checking both the sample directly and sample.sample
    (for CorrectedSample which nests the original OCRSample).
    """
```

All existing Phase 1 calls (`calculate_metrics(samples, "KHATT-train")`) work unchanged
because `text_field` defaults to `"ocr_text"`.

---

## 12. Dependencies

### 12.1 Local Machine (export + analyze stages)

No new dependencies beyond what Phase 1 already requires (`pyyaml`, `editdistance`,
`jiwer`, `tqdm`).

### 12.2 Kaggle/Colab Kernel (inference stage)

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | ≥4.37 | Qwen2.5 model and tokenizer |
| `torch` | ≥2.1 | GPU inference (pre-installed on Kaggle/Colab) |
| `accelerate` | ≥0.25 | `device_map="auto"` |
| `bitsandbytes` | ≥0.41 | 4-bit quantization (optional, for <8GB VRAM) |

Install on Kaggle/Colab kernel:

```bash
pip install transformers accelerate
# Optional:
pip install bitsandbytes
```

Model access on Kaggle: Add `Qwen/Qwen2.5-3B-Instruct` as a Kaggle dataset or let
`AutoModelForCausalLM.from_pretrained()` download it via HuggingFace (requires internet
access, which Kaggle kernels have when enabled).

### 12.3 Future: API Backend

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.0 | OpenAI-compatible API client |
| `tenacity` | ≥8.0 | Retry logic with exponential backoff |

---

## 13. Testing

### 13.1 Test File Locations

```
tests/
├── test_prompt_builder.py      ← NEW
├── test_llm_corrector.py       ← NEW (uses MockTransformersCorrector — no GPU needed)
└── fixtures/
    └── sample_corrections.jsonl  ← NEW: 5 pre-made correction records for analyze tests
```

### 13.2 `MockTransformersCorrector`

LLM tests use a deterministic stand-in that implements `BaseLLMCorrector`:

```python
class MockTransformersCorrector(BaseLLMCorrector):
    """Deterministic BaseLLMCorrector for tests. No model loaded."""

    def __init__(self, return_text: str = "نص مصحح", fail: bool = False):
        self._return_text = return_text
        self._fail = fail

    def correct(self, sample_id, ocr_text, messages, max_retries=2) -> CorrectionResult:
        if self._fail:
            return CorrectionResult(
                sample_id=sample_id, ocr_text=ocr_text,
                corrected_text=ocr_text,  # fallback to original
                prompt_tokens=10, output_tokens=0,
                latency_s=0.0, success=False, error="Mock failure"
            )
        return CorrectionResult(
            sample_id=sample_id, ocr_text=ocr_text,
            corrected_text=self._return_text,
            prompt_tokens=10, output_tokens=5,
            latency_s=0.1, success=True, error=None
        )

    @property
    def model_name(self) -> str:
        return "mock-model"
```

### 13.3 Required Test Cases

**`test_prompt_builder.py`**:
- `test_zero_shot_returns_two_messages`
- `test_zero_shot_system_is_arabic`
- `test_zero_shot_user_contains_ocr_text`
- `test_prompt_version_is_string`
- `test_empty_ocr_text_still_builds_valid_messages`

**`test_llm_corrector.py`** (uses `MockTransformersCorrector`):
- `test_base_corrector_cannot_be_instantiated_directly`
- `test_get_corrector_returns_transformers_by_default`
- `test_get_corrector_raises_on_unknown_backend`
- `test_correct_returns_corrected_text_on_success`
- `test_correct_falls_back_to_ocr_on_failure`
- `test_correction_result_dataclass_has_all_fields`
- `test_corrected_sample_nests_ocr_sample`
- `test_extract_corrected_text_strips_whitespace` (unit test on the private method)
- `test_extract_corrected_text_falls_back_on_empty_output`
- `test_extract_corrected_text_falls_back_on_non_arabic_output`

**`test_metrics.py` additions** (backward-compat check):
- `test_calculate_metrics_default_text_field_uses_ocr_text`
- `test_calculate_metrics_corrected_text_field_uses_corrected`

---

## 14. Implementation Order

| Step | File | Depends On | Notes |
|------|------|-----------|-------|
| 1 | `src/core/prompt_builder.py` | Nothing | Pure logic; testable without GPU |
| 2 | `tests/test_prompt_builder.py` | Step 1 | Run immediately |
| 3 | `src/analysis/metrics.py` (add `text_field`) | Phase 1 metrics | Run existing tests to confirm no regression |
| 4 | `src/core/llm_corrector.py` | `transformers`, `torch` | Define ABC + TransformersCorrector |
| 5 | `src/core/api_corrector.py` | Step 4 | Stub only; raises NotImplementedError |
| 6 | `tests/test_llm_corrector.py` | Steps 4–5 | Uses MockTransformersCorrector; no GPU |
| 7 | `scripts/run_inference.py` | Steps 1, 4 | Standalone Kaggle/Colab script |
| 8 | `pipelines/run_phase2.py` | All above | Export + Analyze modes |
| 9 | Smoke test (export mode) | Step 8 + real data | `--mode export --limit 5` |
| 10 | Upload to Kaggle/Colab + inference | Step 7 + GPU | Run `run_inference.py` on remote |
| 11 | Smoke test (analyze mode) | Step 10 output | `--mode analyze --limit 5 --no-error-analysis` |
| 12 | Full run — all datasets | Step 11 confirmed | Collect paper numbers |

**Smoke test commands:**

```bash
# Step 9: export
python pipelines/run_phase2.py --mode export --limit 5

# Step 11: analyze (after downloading corrections from Kaggle)
python pipelines/run_phase2.py --mode analyze --dataset KHATT-train --no-error-analysis
```

---

## 15. Known Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| LLM outputs explanation instead of corrected text | Medium | Explicit "return only corrected text" in system prompt; `_extract_corrected_text()` is defensive |
| LLM output is in English or non-Arabic | Low | Check for Arabic chars in output; fallback to ocr_text |
| LLM refuses to process (safety filter) | Low | Detect empty output; fallback to ocr_text; log as failed |
| GPU OOM on long inputs | Low | `MAX_INPUT_TOKENS=512` guard; truncate with warning |
| Kaggle session times out mid-run | Medium | `run_inference.py` writes JSONL line-by-line; resumes from last written sample |
| Kaggle dataset/model access not available | Medium | Pre-download model weights as a Kaggle dataset; document setup in §16 |
| Phase 1 baseline not found for comparison | Medium | Load gracefully (return None); skip comparison; log warning |
| `corrections.jsonl` format mismatch between inference and analyze stages | Low | Both stages share the schema defined in §9.3; validate on load |
| API rate limits (future APICorrector) | Medium | `requests_per_minute` config key; retry with exponential backoff (tenacity) |
| API key accidentally committed to git | Low | Key always read from env var; `config.yaml` never contains key value |
| Model produces identical text to OCR (no correction) | Medium | Log as "no change" sample; counted in metrics; not treated as failure |
| Phase 2 CER higher than Phase 1 (LLM makes things worse) | Possible | Report honestly; valid experimental finding |

---

## 16. Appendix A: Prompt Text

### System Message (Zero-Shot, v1)

```
أنت مصحح نصوص عربية متخصص. مهمتك تصحيح أخطاء التعرف الضوئي (OCR) في النص العربي. أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي.
```

**English**: "You are a specialized Arabic text corrector. Your task is to correct OCR errors in
Arabic text. Return only the corrected text without any explanation or additional comment."

### User Message

```
[The raw OCR output text]
```

### Why This Prompt Design

- **"متخصص"**: frames the task as domain expertise, not general assistance
- **"أخطاء التعرف الضوئي (OCR)"**: explicitly names the error source — model knows these are
  OCR artifacts, not typos or intentional word choices
- **"أعد النص المصحح فقط"**: critical — prevents the model from explaining itself, which would
  require complex output parsing
- **No additional context**: deliberate — this is zero-shot; any extra knowledge invalidates the
  Phase 2 baseline that later phases compare against

---

## 17. Appendix B: Kaggle Setup Guide

### B.1 Required Files to Upload

Upload these files/directories to your Kaggle notebook as input data:

```
From local results/:
  results/phase2/inference_input.jsonl   → contains OCR texts + GT texts

From project src/:
  src/core/prompt_builder.py
  src/core/llm_corrector.py
  scripts/run_inference.py
```

### B.2 Kaggle Notebook Setup

```python
# Cell 1: Install dependencies
!pip install transformers accelerate -q

# Cell 2: Set paths
INPUT_JSONL  = "/kaggle/input/your-dataset/inference_input.jsonl"
OUTPUT_JSONL = "/kaggle/working/corrections.jsonl"
MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"

# Cell 3: Run inference
!python run_inference.py \
    --input  {INPUT_JSONL} \
    --output {OUTPUT_JSONL} \
    --model  {MODEL_NAME}
```

### B.3 Kaggle GPU Settings

- **Accelerator**: GPU T4 x2 or P100 (free tier)
- **Session length**: up to 12 hours (free) — enough for ~4,000 samples at ~2–3s/sample
- **Internet**: Enable in Settings > Internet (required to download model from HuggingFace)
- **Persistence**: `corrections.jsonl` is written line-by-line; if session expires, restart
  and the script resumes from the last completed sample

### B.4 After Inference

Download `corrections.jsonl` from Kaggle output and place at:

```
results/phase2/PATS-A01-Akhbar/corrections.jsonl
results/phase2/PATS-A01-Andalus/corrections.jsonl
results/phase2/KHATT-train/corrections.jsonl
results/phase2/KHATT-validation/corrections.jsonl
```

One separate inference run per dataset is recommended to keep file sizes manageable.
Alternatively, combine all datasets in one `inference_input.jsonl` and split the output by
the `"dataset"` field.

Then run the local analysis:

```bash
python pipelines/run_phase2.py --mode analyze
```
