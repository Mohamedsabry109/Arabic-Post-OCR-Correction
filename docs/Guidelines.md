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
