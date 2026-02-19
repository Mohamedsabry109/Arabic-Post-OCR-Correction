# Phase 1: LLM Baseline for Arabic Post-OCR Correction

This phase implements a zero-shot LLM-based approach to correct OCR errors in Arabic text using instruction-tuned language models.

## Overview

Phase 1 establishes a baseline for Arabic post-OCR correction using:
- **Zero-shot prompting** with Arabic system prompts
- **Qwen2.5-3B-Instruct** or **Qwen3-4B-Instruct** models
- **Character Error Rate (CER)** and **Word Error Rate (WER)** metrics

## Project Structure

```
Arabic-Post-OCR-Correction/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── utils.py              # Arabic text processing utilities
│   ├── data_loader.py        # Dataset loading and validation
│   ├── metrics.py            # CER/WER metrics calculation
│   └── llm_corrector.py      # LLM-based correction
├── data/
│   ├── Original/             # Ground truth text files
│   │   ├── PATS-A01/
│   │   └── KHATT/
│   └── Predictions/          # OCR prediction files
│       ├── PATS-A01/
│       └── KHATT/
├── results/
│   └── phase1/               # Phase 1 output files
├── run_phase1.py             # Main pipeline script
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README_phase1.md          # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- At least 8GB GPU memory (or use quantization for less)

### Setup

1. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Prepare your data:**

Organize your data in the following structure:

```
data/
├── Original/           # Ground truth files
│   ├── PATS-A01/
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   └── ...
│   └── KHATT/
│       ├── 001.txt
│       └── ...
└── Predictions/        # OCR predictions (Qaari output)
    ├── PATS-A01/
    │   ├── 001.txt
    │   ├── 002.txt
    │   └── ...
    └── KHATT/
        ├── 001.txt
        └── ...
```

Each text file should contain one paragraph of Arabic text in UTF-8 encoding.

## Usage

### Basic Usage

Run the complete Phase 1 pipeline:

```bash
python run_phase1.py
```

### Command Line Options

```bash
# Use a custom configuration file
python run_phase1.py --config my_config.yaml

# Process specific datasets
python run_phase1.py --datasets PATS-A01

# Limit samples per dataset (useful for testing)
python run_phase1.py --limit 50

# Skip LLM correction (baseline metrics only)
python run_phase1.py --skip-correction

# Disable progress bars
python run_phase1.py --no-progress

# Combine options
python run_phase1.py --datasets PATS-A01 KHATT --limit 100
```

### Configuration

Edit `config.yaml` to customize:

```yaml
# Data paths
data:
  ground_truth_base: "data/Original"
  predictions_base: "data/Predictions"
  datasets:
    - "PATS-A01"
    - "KHATT"

# Model settings
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  temperature: 0.1
  load_in_4bit: false  # Enable for low GPU memory

# Processing
processing:
  limit_per_dataset: null  # Set to number for testing
```

## Output Files

After running, results are saved to `results/phase1/`:

| File | Description |
|------|-------------|
| `metrics.json` | Complete metrics in JSON format |
| `patsa_corrected.txt` | Corrected texts for PATS-A01 dataset |
| `khatt_corrected.txt` | Corrected texts for KHATT dataset |
| `sample_corrections.txt` | First 20 examples with before/after |
| `report.md` | Markdown report with analysis |

## Metrics

### Character Error Rate (CER)

```
CER = (Substitutions + Deletions + Insertions) / Reference Length
```

Measures character-level edit distance normalized by reference length.

### Word Error Rate (WER)

```
WER = (Word Substitutions + Word Deletions + Word Insertions) / Reference Word Count
```

Measures word-level edit distance normalized by reference word count.

## Python API

### Data Loading

```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader(
    ground_truth_base="data/Original",
    predictions_base="data/Predictions"
)

# List available datasets
datasets = loader.list_datasets()
print(f"Available: {datasets}")

# Load a dataset
pairs = loader.load_dataset("PATS-A01", limit=100)
for ocr_text, ground_truth in pairs[:5]:
    print(f"OCR: {ocr_text[:50]}...")
    print(f"GT:  {ground_truth[:50]}...")
```

### Metrics Calculation

```python
from src.metrics import calculate_cer, calculate_wer, calculate_metrics

# Single pair
reference = "مرحباً بالعالم"
hypothesis = "مرحبا بالعالم"

cer = calculate_cer(reference, hypothesis)
wer = calculate_wer(reference, hypothesis)
print(f"CER: {cer:.2%}, WER: {wer:.2%}")

# Multiple pairs
predictions = ["مرحبا", "العالم"]
ground_truths = ["مرحباً", "العالم"]

metrics = calculate_metrics(predictions, ground_truths)
print(f"Aggregate CER: {metrics['cer']:.2%}")
print(f"Aggregate WER: {metrics['wer']:.2%}")
```

### LLM Correction

```python
from src.llm_corrector import LLMCorrector

# Initialize corrector
corrector = LLMCorrector(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    temperature=0.1,
    load_in_4bit=True  # For limited GPU memory
)

# Correct single text
ocr_text = "مرحيا بالعالم"
corrected = corrector.correct(ocr_text)
print(f"Corrected: {corrected}")

# Batch correction
texts = ["نص اول", "نص ثاني"]
corrected_texts = corrector.correct_batch(texts)
```

### Text Utilities

```python
from src.utils import (
    normalize_arabic,
    clean_text,
    remove_diacritics,
    is_arabic,
    get_text_stats
)

text = "  مَرْحَباً   بالعالم  "

# Clean whitespace
cleaned = clean_text(text)
print(cleaned)  # "مَرْحَباً بالعالم"

# Remove diacritics
no_diacritics = remove_diacritics(text)
print(no_diacritics)  # "مرحبا بالعالم"

# Normalize Arabic characters
normalized = normalize_arabic(text, remove_diacritics=True)
print(normalized)  # "مرحبا بالعالم"

# Check if text is Arabic
print(is_arabic(text))  # True

# Get text statistics
stats = get_text_stats(text)
print(stats)  # {'char_count': 20, 'word_count': 2, ...}
```

## Troubleshooting

### Out of Memory Error

If you encounter GPU memory errors:

1. Enable 4-bit quantization in `config.yaml`:
   ```yaml
   model:
     load_in_4bit: true
   ```

2. Or use 8-bit quantization:
   ```yaml
   model:
     load_in_8bit: true
   ```

3. Reduce batch processing or use CPU (slower):
   ```python
   corrector = LLMCorrector(device="cpu")
   ```

### Encoding Issues

If you see garbled Arabic text:

1. Ensure all input files are UTF-8 encoded
2. The data loader automatically tries alternative encodings (cp1256, iso-8859-6)
3. Check the log file (`phase1.log`) for encoding warnings

### Missing Data

If datasets aren't loading:

1. Verify the directory structure matches the expected format
2. Check that file names match between `Original` and `Predictions` folders
3. Use `loader.validate_alignment("DATASET_NAME")` to check for mismatches

## Model Recommendations

| Model | Size | Quality | Speed | Memory |
|-------|------|---------|-------|--------|
| Qwen2.5-3B-Instruct | 3B | Good | Fast | ~8GB |
| Qwen3-4B-Instruct-2507 | 4B | Better | Medium | ~10GB |
| Qwen2.5-3B (4-bit) | 3B | Good | Fast | ~4GB |

## Next Steps

After completing Phase 1:

1. **Analyze Results**: Review `sample_corrections.txt` to understand error patterns
2. **Error Analysis**: Identify cases where LLM correction degraded quality
3. **Phase 2**: Implement specialized approaches based on Phase 1 insights:
   - Fine-tuning on Arabic OCR correction data
   - Retrieval-augmented generation (RAG)
   - Ensemble methods

## License

See the LICENSE file in the root directory.
