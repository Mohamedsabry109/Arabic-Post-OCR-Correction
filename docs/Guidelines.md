# Development Guidelines

## 1. Core Principles

### 1.1 KISS (Keep It Simple, Stupid)

- Write the simplest code that solves the problem
- Avoid premature optimization
- No unnecessary abstractions
- If a function does one thing, it's probably right

### 1.2 Research-First Mindset

This is a research project. Every design decision should consider:

1. **Reproducibility**: Can someone else run this and get the same results?
2. **Measurability**: Does this produce numbers we can put in a paper?
3. **Comparability**: Can we compare this with other approaches?

### 1.3 Maintainability

- Code should be readable 6 months from now
- Clear variable names over comments
- Consistent patterns across modules

---

## 2. Code Standards

### 2.1 Python Style

- **Python Version**: 3.8+
- **Style Guide**: PEP 8
- **Line Length**: 100 characters max
- **Imports**: Standard library → Third-party → Local (separated by blank lines)

```python
# Good
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src.data.data_loader import DataLoader
from src.analysis.metrics import calculate_cer
```

### 2.2 Type Hints

Use type hints for all function signatures:

```python
# Good
def calculate_cer(reference: str, hypothesis: str) -> float:
    ...

def load_dataset(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    ...

# Bad
def calculate_cer(reference, hypothesis):
    ...
```

### 2.3 Docstrings

Use Google-style docstrings for public functions:

```python
def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth text.
        hypothesis: Predicted/corrected text.

    Returns:
        CER value between 0.0 and 1.0+.

    Example:
        >>> calculate_cer("مرحبا", "مرحيا")
        0.2
    """
```

### 2.4 Error Handling

- Fail fast with clear error messages
- Use custom exceptions for domain errors
- Always include context in error messages

```python
# Good
if not file_path.exists():
    raise FileNotFoundError(f"Ground truth file not found: {file_path}")

# Bad
if not file_path.exists():
    raise Exception("File not found")
```

---

## 3. Project Conventions

### 3.1 File Naming

| Type | Convention | Example |
|------|------------|---------|
| Python modules | `snake_case.py` | `data_loader.py` |
| Classes | `PascalCase` | `DataLoader` |
| Functions/variables | `snake_case` | `load_dataset` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_MODEL` |
| Config files | `snake_case.yaml` | `config.yaml` |

### 3.2 Directory Structure

```
src/
├── data/           # Data loading and processing
├── core/           # Main correction logic
└── analysis/       # Metrics and evaluation

pipelines/          # Executable scripts for each phase
configs/            # Configuration files
results/            # Generated outputs (gitignored)
docs/               # Documentation
tests/              # Unit tests
```

### 3.3 Configuration

- All configurable values go in `configs/config.yaml`
- No hardcoded paths in source code
- Use environment variables for machine-specific settings only

### 3.4 Results Organization

Each phase outputs to its own directory:

```
results/phase1/
├── metrics.json          # Machine-readable metrics
├── report.md             # Human-readable summary
└── [other outputs]       # Phase-specific files
```

---

## 4. Data Handling

### 4.1 Text Encoding

- Always use UTF-8
- Explicitly specify encoding in file operations:

```python
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()
```

### 4.2 Arabic Text Processing

- Preserve diacritics unless explicitly removing them
- Handle RTL text properly (no string reversal hacks)
- Use regex with Unicode support for Arabic patterns:

```python
import re
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]+')
```

### 4.3 Data Paths

- Use `pathlib.Path` for all file operations
- Data is stored externally: `../data/`
- Never commit data files to git

---

## 5. LLM Integration

### 5.1 Model Loading

- Load model once, reuse for all predictions
- Use appropriate quantization for available hardware
- Handle CUDA out-of-memory gracefully

### 5.2 Prompts

- Store prompts as constants or templates
- Document what each prompt is designed to do
- Version prompts (don't silently change them between runs)

### 5.3 Determinism

- Use low temperature (0.1) for reproducibility
- Set random seeds where possible
- Document any sources of non-determinism

---

## 6. Metrics & Evaluation

### 6.1 Metric Calculation

- Always calculate on normalized text (consistent preprocessing)
- Report both mean and standard deviation
- Include sample counts

### 6.2 Statistical Rigor

- Use paired tests (same samples across conditions)
- Report p-values and effect sizes
- Use 95% confidence intervals

### 6.3 Reproducibility

Each results JSON should include:
- Timestamp
- Configuration used
- Git commit hash (if available)
- Sample count

---

## 7. Documentation

### 7.1 Required Documentation

| File | Purpose | Update When |
|------|---------|-------------|
| `CLAUDE.md` | Project context for AI assistants | Structure changes |
| `CHANGELOG.md` | Version history | Any significant change |
| `docs/Architecture.md` | System design | Architecture changes |
| `docs/Guidelines.md` | This file | Convention changes |
| `README.md` | Project overview | Major milestones |

### 7.2 Code Comments

- Explain "why", not "what"
- Comment complex algorithms
- No commented-out code (use git history)

```python
# Good: Explains why
# Normalize before comparison to handle diacritic variations
normalized_ref = normalize_arabic(reference)

# Bad: Explains what (obvious from code)
# Calculate the length
length = len(text)
```

---

## 8. Git Workflow

### 8.1 Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `docs`: Documentation
- `test`: Tests
- `config`: Configuration changes

Examples:
```
feat: add zero-shot LLM correction pipeline
fix: handle empty text in CER calculation
docs: update architecture diagram
```

### 8.2 Branches

- `main`: Stable, working code
- `phase-N`: Development for specific phase
- Feature branches for complex features

### 8.3 What to Commit

**Do commit:**
- Source code
- Configuration files
- Documentation
- Requirements

**Don't commit:**
- Data files
- Model weights
- Results (unless specifically needed)
- Virtual environment
- Cache files

---

## 9. Testing

### 9.1 Test Structure

```
tests/
├── test_data_loader.py
├── test_metrics.py
├── test_llm_corrector.py
└── fixtures/           # Test data
```

### 9.2 Test Requirements

- Test edge cases (empty strings, single characters)
- Test with Arabic text specifically
- Test error conditions

### 9.3 Running Tests

```bash
pytest tests/
pytest tests/test_metrics.py -v  # Verbose single file
```

---

## 10. Performance Considerations

### 10.1 Memory

- Process files in streaming fashion for large datasets
- Clear GPU memory between batches if needed
- Monitor memory usage during long runs

### 10.2 Speed

- Use progress bars for long operations
- Cache expensive computations
- Profile before optimizing

### 10.3 Logging

- Use `logging` module, not `print`
- Log to both console and file
- Include timestamps

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

---

## 11. Checklist for New Code

Before committing new code, verify:

- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] No hardcoded paths
- [ ] UTF-8 encoding specified
- [ ] Error handling with clear messages
- [ ] Tests for new functionality
- [ ] CHANGELOG.md updated
- [ ] Works with both datasets (PATS-A01 and KHATT)
- [ ] Pipeline uses `--datasets` flag for dataset selection (no hardcoded list)
- [ ] Pipeline reads active dataset list from `config['datasets']` by default
- [ ] Pipeline skips already-completed datasets unless `--force` is set

---

## 12. Compute Environment

### 12.1 No Local GPU — Remote Inference Required

LLM inference (Phases 2–6) requires a GPU. This project uses **Kaggle** or **Google Colab**
for all LLM inference because no local GPU is available.

**Rule**: Never assume local GPU access for any LLM-related code.

### 12.2 Two-Stage Pipeline Design

All LLM-dependent phases are split into two clearly separated stages:

| Stage | Runs On | Input | Output |
|-------|---------|-------|--------|
| **Export** | Local | OCR texts from DataLoader | `inference_input.jsonl` |
| **Inference** | Kaggle / Colab | `inference_input.jsonl` | `corrections.jsonl` |
| **Analyze** | Local | `corrections.jsonl` | metrics, reports |

The pipeline script exposes this split via a `--mode` flag:

```bash
python pipelines/run_phase2.py --mode export    # local: prepare data for upload
python pipelines/run_phase2.py --mode analyze   # local: process downloaded corrections
python pipelines/run_phase2.py --mode full      # future/API: end-to-end in one run
```

The inference stage runs from a separate script designed to be self-contained on Kaggle/Colab:

```bash
python scripts/run_inference.py                 # executed on Kaggle/Colab kernel
```

### 12.3 The Contract File: `corrections.jsonl`

The only coupling between the inference stage and the analysis stage is a JSONL file.
Each line is one completed correction:

```json
{
  "sample_id": "AHTD3A0001_Para2_3",
  "dataset":   "KHATT-train",
  "ocr_text":  "... raw OCR ...",
  "corrected_text": "... LLM output ...",
  "gt_text":   "... ground truth ...",
  "model":     "Qwen/Qwen2.5-3B-Instruct",
  "prompt_version": "v1",
  "prompt_tokens": 89,
  "output_tokens": 92,
  "latency_s": 2.31,
  "success": true,
  "error": null
}
```

`gt_text` is included in the export so the Kaggle kernel does not need access to the
full ground-truth dataset — the inference stage is self-contained from this JSONL alone.

### 12.4 Corrector Abstraction Layer

All LLM backends implement the same abstract interface (`BaseLLMCorrector`). This keeps
the pipeline code backend-agnostic.

```
BaseLLMCorrector (ABC)
├── TransformersCorrector   — HuggingFace transformers; used on Kaggle/Colab
└── APICorrector            — OpenAI-compatible REST API; used for future API access
```

The active backend is selected by config, not by hardcoded imports:

```yaml
model:
  backend: "transformers"   # switch to "api" for API-based inference
```

The pipeline (`run_phaseN.py`) calls `get_corrector(config)` which returns the correct
implementation. **No pipeline code changes when switching backends.**

### 12.5 Adding a New LLM Backend

To add a new backend (e.g., Anthropic Claude API, a local vLLM server, Ollama):

1. Create a new class inheriting `BaseLLMCorrector` in `src/core/`
2. Implement `correct()` and the `model_name` property
3. Add one `elif` case in the `get_corrector(config)` factory function
4. Add any backend-specific config keys under a new top-level key in `config.yaml`

The pipeline, PromptBuilder, and all analysis modules require **zero changes**.

### 12.6 What Runs Where

| Code | Runs On | Notes |
|------|---------|-------|
| `src/data/data_loader.py` | Local | Reads local OCR and GT files |
| `src/data/text_utils.py` | Local + Kaggle | No external deps; safe to upload |
| `src/core/prompt_builder.py` | Local + Kaggle | No external deps; safe to upload |
| `src/core/llm_corrector.py` | Local + Kaggle | TransformersCorrector runs on Kaggle |
| `src/core/api_corrector.py` | Local | API calls go from local machine |
| `src/analysis/` | Local | Metrics and analysis always run locally |
| `pipelines/run_phase*.py` | Local | Orchestration always on local machine |
| `scripts/run_inference.py` | Kaggle / Colab | Standalone inference script |

### 12.7 Kaggle/Colab Inference Workflow (Step by Step)

```
1. LOCAL   → python pipelines/run_phase2.py --mode export
             → produces: results/phase2/inference_input.jsonl

2. UPLOAD  → upload to Kaggle/Colab:
             - results/phase2/inference_input.jsonl
             - src/core/prompt_builder.py
             - src/core/llm_corrector.py   (TransformersCorrector)
             - scripts/run_inference.py

3. REMOTE  → on Kaggle/Colab kernel, run:
             python scripts/run_inference.py \
               --input inference_input.jsonl \
               --output corrections.jsonl \
               --model Qwen/Qwen2.5-3B-Instruct

4. DOWNLOAD → download corrections.jsonl from Kaggle/Colab output
              → place at: results/phase2/KHATT-train/corrections.jsonl

5. LOCAL   → python pipelines/run_phase2.py --mode analyze
             → reads corrections.jsonl, computes metrics, writes report
```

### 12.8 API Extension (Future)

When API-based inference is needed:

```yaml
# configs/config.yaml
model:
  backend: "api"
  name: "gpt-4o"             # or any OpenAI-compatible model name

api:
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"   # read from environment, never hardcoded
  timeout_s: 30
  requests_per_minute: 60         # rate limiting
```

With `backend: "api"`, the pipeline runs fully locally in `--mode full` (no Kaggle needed):

```bash
OPENAI_API_KEY=sk-... python pipelines/run_phase2.py --mode full
```

**Security**: API keys are **never** stored in config files or committed to git.
Always read from environment variables.

### 12.9 Checklist for LLM-Dependent Code

In addition to the general checklist (§11), verify for any LLM phase:

- [ ] Pipeline supports `--mode export`, `--mode analyze` (and `--mode full` for API)
- [ ] `inference_input.jsonl` includes `gt_text` so Kaggle is self-contained
- [ ] `corrections.jsonl` includes all fields specified in §12.3
- [ ] Analysis stage reads from `corrections.jsonl`, not from LLM directly
- [ ] New LLM feature uses `BaseLLMCorrector`; does not hardcode `TransformersCorrector`
- [ ] API keys read from environment variables only

---

## 13. Dataset Selection and Resume Policy

### 13.1 Configurable Dataset Selection

All processing scripts support flexible dataset selection through two mechanisms:

**Default (all datasets from config)**:
```bash
python pipelines/run_phase1.py            # processes all datasets in config['datasets']
python pipelines/run_phase2.py --mode export
```

**Subset via `--datasets` flag**:
```bash
python pipelines/run_phase1.py --datasets KHATT-train KHATT-validation
python pipelines/run_phase2.py --mode export --datasets PATS-A01-Akhbar
```

**Rules:**
- `config.yaml` `datasets:` list is the **single source of truth** for what "all" means.
- The `--datasets` CLI flag overrides the config list for that run only.
- No `choices=` restriction on `--datasets` — any dataset key supported by `DataLoader` works.
- Never hardcode a dataset list inside a pipeline script; always use `resolve_datasets()`.

### 13.2 The `resolve_datasets()` Helper

All pipeline scripts use the shared `pipelines/_utils.py::resolve_datasets()` function:

```python
from pipelines._utils import resolve_datasets

active_datasets = resolve_datasets(config, args.datasets)
# Returns args.datasets if provided; otherwise reads names from config['datasets'].
```

### 13.3 Full Dataset Coverage

The default `config['datasets']` list covers all 10 datasets:
- 8 PATS-A01 fonts: Akhbar, Andalus, Arial, Naskh, Simplified, Tahoma, Thuluth, Traditional
- 2 KHATT splits: train, validation

`DataLoader.iter_samples()` dynamically resolves any `PATS-A01-{font}` or `KHATT-{split}` key.
No code changes are needed to add a new PATS font — just add it to `config.yaml`.

### 13.4 Resume-After-Break Policy

Every processing step must be re-runnable without re-doing completed work.

**Convention**: a dataset step is "complete" when its primary output JSON exists:
- Phase 1: `results/phase1/{key}/metrics.json`
- Phase 2 export: dataset key present in `results/phase2/inference_input.jsonl`
- Phase 2 analyze: `results/phase2/{key}/metrics.json`

**Behavior**:
- On restart, completed datasets are detected and skipped automatically.
- Use `--force` to override resume and re-process regardless.
- Inference scripts (`run_inference.py`) write line-by-line and read completed IDs on restart.

```bash
# Resume example: run already partially completed phase1
python pipelines/run_phase1.py             # skips KHATT-train if metrics.json exists

# Force re-run
python pipelines/run_phase1.py --force     # re-processes all datasets

# Force re-run for one dataset
python pipelines/run_phase1.py --datasets KHATT-train --force
```
