# Knowledge Base Construction for Arabic Post-OCR Correction

This document describes the knowledge base construction scripts used to build the data resources for Phase 2 and Phase 3 of the Arabic Post-OCR correction system.

## Overview

The knowledge bases provide structured information about:
1. **OCR Error Patterns** - What errors Qaari OCR commonly makes
2. **Arabic Vocabulary** - Valid Arabic words for spell checking
3. **N-gram Statistics** - Common word sequences for context validation
4. **Error-Correction Pairs** - Real examples from the QALB corpus
5. **Grammar Rules** - Arabic spelling and grammar rules

## Quick Start

### Build All Knowledge Bases

```bash
# Build everything (requires all data sources)
python scripts/build_knowledge_bases.py --all

# Build with specific paths
python scripts/build_knowledge_bases.py --all \
    --openiti-path /path/to/openiti \
    --qalb-path /path/to/qalb \
    --output data/
```

### Build Individual Components

```bash
# Build only confusion matrix (requires OCR data)
python scripts/build_knowledge_bases.py --confusion-matrix

# Build only vocabulary (requires OpenITI)
python scripts/build_knowledge_bases.py --vocab --openiti-path /path/to/openiti

# Build only n-grams (requires OpenITI)
python scripts/build_knowledge_bases.py --ngrams --openiti-path /path/to/openiti

# Build only QALB corrections
python scripts/build_knowledge_bases.py --qalb --qalb-path /path/to/qalb

# Build only grammar rules (default rules, no LLM needed)
python scripts/build_knowledge_bases.py --rules --default-rules-only
```

## Scripts

### 1. Qaari Error Analyzer (`scripts/analyze_qaari_errors.py`)

Analyzes OCR errors from Qaari by comparing predictions with ground truth.

**Usage:**
```bash
python scripts/analyze_qaari_errors.py \
    --datasets PATS-A01 KHATT \
    --ground-truth data/Original \
    --predictions data/Predictions \
    --output data/
```

**Output Files:**
- `confusion_matrix.json` - Character-level confusion statistics
- `error_statistics.json` - Aggregate error statistics

**Example Output (confusion_matrix.json):**
```json
{
  "ب": {
    "ت": {
      "count": 245,
      "probability": 0.32,
      "positions": {"start": 50, "middle": 150, "end": 45},
      "examples": ["كتاب -> كتات", ...]
    },
    "ن": {
      "count": 89,
      "probability": 0.12,
      ...
    }
  }
}
```

**Runtime:** ~1-2 minutes for 1000 samples

---

### 2. Vocabulary Builder (`scripts/build_vocabulary.py`)

Builds Arabic word frequency vocabulary from the OpenITI corpus.

**Usage:**
```bash
python scripts/build_vocabulary.py \
    --corpus /path/to/openiti \
    --output data/vocab_100k.json \
    --top-n 100000 \
    --min-count 2
```

**Output File:** `vocab_100k.json`

**Example Output:**
```json
{
  "metadata": {
    "total_words": 50000000,
    "unique_words": 500000,
    "vocabulary_size": 100000,
    "created_at": "2024-01-15T10:30:00"
  },
  "vocabulary": {
    "الله": 123456,
    "من": 100000,
    "في": 98765,
    ...
  }
}
```

**Runtime:** ~5-15 minutes for 1GB of text

---

### 3. N-gram Builder (`scripts/build_ngrams.py`)

Extracts word bigrams and trigrams from the OpenITI corpus.

**Usage:**
```bash
python scripts/build_ngrams.py \
    --corpus /path/to/openiti \
    --output data/ngrams.json \
    --top-bigrams 50000 \
    --top-trigrams 50000
```

**Output File:** `ngrams.json`

**Example Output:**
```json
{
  "metadata": {
    "total_bigrams": 45000000,
    "total_trigrams": 40000000,
    "ngram_separator": "_"
  },
  "bigrams": {
    "من_الله": 12345,
    "في_ذلك": 11234,
    ...
  },
  "trigrams": {
    "في_سبيل_الله": 5678,
    "بسم_الله_الرحمن": 4567,
    ...
  }
}
```

**Runtime:** ~10-20 minutes for 1GB of text

---

### 4. QALB Processor (`scripts/extract_qalb.py`)

Extracts error-correction pairs from the QALB parallel corpus.

**Usage:**
```bash
python scripts/extract_qalb.py \
    --qalb-path /path/to/qalb \
    --output-dir data/ \
    --few-shot 20 \
    --context-window 2
```

**Output Files:**
- `qalb_corrections.json` - All extracted corrections
- `qalb_few_shot.json` - Selected diverse examples

**Example Output (qalb_few_shot.json):**
```json
{
  "metadata": {
    "num_examples": 20,
    "error_types": ["hamza", "spelling", "taa_marbuta", ...]
  },
  "examples": [
    {
      "source": "انا",
      "target": "أنا",
      "error_type": "hamza",
      "context_before": "قال",
      "context_after": "ذهبت إلى",
      "source_sentence": "قال انا ذهبت إلى المدرسة",
      "target_sentence": "قال أنا ذهبت إلى المدرسة"
    },
    ...
  ]
}
```

**Error Types Classified:**
- `hamza` - Hamza/Alef confusion (أ/إ/ا/آ)
- `taa_marbuta` - Taa Marbuta confusion (ة/ه)
- `alef_maksura` - Alef Maksura/Ya confusion (ى/ي)
- `diacritics` - Diacritic errors
- `spacing` - Word spacing issues
- `punctuation` - Punctuation errors
- `spelling` - General spelling errors
- `word_boundary` - Word segmentation errors

**Runtime:** ~1-3 minutes

---

### 5. Rule Extractor (`scripts/extract_rules.py`)

Extracts grammar/spelling rules from Arabic grammar books using LLM or provides default rules.

**Usage:**
```bash
# With LLM extraction from books
python scripts/extract_rules.py \
    --books /path/to/grammar_books \
    --output data/rules \
    --model Qwen/Qwen2.5-3B-Instruct

# Default rules only (no LLM needed)
python scripts/extract_rules.py \
    --output data/rules \
    --default-only
```

**Output Files:**
- `rules/morphology_rules.json` - Morphological rules
- `rules/syntax_rules.json` - Syntactic rules
- `rules/orthography_rules.json` - Spelling/orthographic rules

**Example Output (orthography_rules.json):**
```json
{
  "metadata": {
    "category": "orthography",
    "num_rules": 7
  },
  "rules": [
    {
      "rule_name": "همزة الوصل والقطع",
      "description": "التمييز بين همزة الوصل (ا) وهمزة القطع (أ/إ)",
      "category": "orthography",
      "correct_examples": ["أَكَلَ", "استَغفَرَ", "إنّ"],
      "incorrect_examples": ["اكل", "إستغفر", "ان"],
      "when_to_apply": "في بداية الكلمات العربية"
    },
    ...
  ]
}
```

**Default Rules Include:**
1. Hamza al-Wasl vs Hamza al-Qat'
2. Taa Marbuta vs Taa Maftuha
3. Alef Maksura vs Ya
4. Lam Shamsiyya and Qamariyya
5. Tanwin rules
6. Alef Faariqa (after Waw al-Jama'a)
7. Hamzat al-Madd

**Runtime:** ~5-30 minutes (with LLM), ~1 second (default only)

---

### 6. Master Builder (`scripts/build_knowledge_bases.py`)

Orchestrates the construction of all knowledge bases.

**Usage:**
```bash
# Build all with defaults
python scripts/build_knowledge_bases.py --all

# Build with configuration file
python scripts/build_knowledge_bases.py --config kb_config.yaml

# Force rebuild (ignore existing files)
python scripts/build_knowledge_bases.py --all --force
```

**Configuration (kb_config.yaml):**
```yaml
# Data paths
ground_truth_path: "data/Original"
predictions_path: "data/Predictions"
openiti_path: "data/openiti"
qalb_path: "data/qalb"
grammar_books_path: "data/grammar_books"

# Output
output_dir: "data"

# Datasets
datasets:
  - "PATS-A01"
  - "KHATT"

# Vocabulary settings
vocab_top_n: 100000
vocab_min_count: 2

# N-gram settings
ngrams_top_bigrams: 50000
ngrams_top_trigrams: 50000

# QALB settings
qalb_few_shot_n: 20

# Rules settings
rules_default_only: false

# Processing
skip_existing: true
force_rebuild: false
```

## Output Structure

After running all scripts, the data directory will contain:

```
data/
├── confusion_matrix.json      # Qaari OCR error patterns
├── error_statistics.json      # Aggregate error statistics
├── vocab_100k.json            # Top 100K Arabic words
├── ngrams.json                # Bigram and trigram frequencies
├── qalb_corrections.json      # All QALB error-correction pairs
├── qalb_few_shot.json         # Selected few-shot examples
├── build_report.txt           # Build summary report
└── rules/
    ├── morphology_rules.json  # Morphological rules
    ├── syntax_rules.json      # Syntactic rules
    └── orthography_rules.json # Spelling rules
```

## Data Sources

### Required Data

1. **OCR Data** (for confusion matrix)
   - Ground truth: `data/Original/<dataset>/*.txt`
   - Predictions: `data/Predictions/<dataset>/*.txt`

2. **OpenITI Corpus** (for vocabulary and n-grams)
   - Download from: https://github.com/OpenITI
   - Text files in `.txt`, `.ara`, or `.mARkdown` format

3. **QALB Corpus** (for error-correction pairs)
   - Parallel files: `source.txt` / `target.txt` or `.src` / `.trg`
   - Or combined TSV format

4. **Grammar Books** (optional, for rule extraction)
   - Arabic text files containing grammar rules
   - Used with LLM for structured extraction

## API Usage

### Loading Knowledge Bases in Python

```python
# Load vocabulary
from scripts.build_vocabulary import load_vocabulary
vocab, meta = load_vocabulary("data/vocab_100k.json")
print(f"Vocabulary size: {len(vocab)}")
print(f"Is 'مرحبا' valid? {'مرحبا' in vocab}")

# Load n-grams
from scripts.build_ngrams import load_ngrams, split_ngram
bigrams, trigrams, meta = load_ngrams("data/ngrams.json")
print(f"Top bigram: {split_ngram(list(bigrams.keys())[0])}")

# Load QALB examples
import json
with open("data/qalb_few_shot.json", "r", encoding="utf-8") as f:
    qalb = json.load(f)
for ex in qalb["examples"][:5]:
    print(f"{ex['source']} -> {ex['target']} ({ex['error_type']})")

# Load confusion matrix
with open("data/confusion_matrix.json", "r", encoding="utf-8") as f:
    confusion = json.load(f)
# Get most common confusions for character 'ب'
if 'ب' in confusion:
    for ocr_char, stats in confusion['ب'].items():
        print(f"ب -> {ocr_char}: {stats['count']} times")
```

### Using with LLM Corrector

```python
from src.llm_corrector import LLMCorrector
import json

# Load few-shot examples for prompting
with open("data/qalb_few_shot.json", "r", encoding="utf-8") as f:
    few_shot = json.load(f)["examples"]

# Build few-shot prompt
examples_text = "\n".join([
    f"خطأ: {ex['source']} -> صحيح: {ex['target']}"
    for ex in few_shot[:5]
])

# Use in correction
corrector = LLMCorrector(
    system_prompt=f"""أنت مصحح نصوص عربية. إليك بعض الأمثلة:
{examples_text}

صحح النص التالي:"""
)
```

## Runtime Estimates

| Component | Data Size | Estimated Time |
|-----------|-----------|----------------|
| Confusion Matrix | 1000 samples | 1-2 minutes |
| Vocabulary | 1GB corpus | 5-15 minutes |
| N-grams | 1GB corpus | 10-20 minutes |
| QALB | 10K sentences | 1-3 minutes |
| Rules (default) | - | < 1 second |
| Rules (LLM) | 10 books | 5-30 minutes |
| **Total** | **Full build** | **~30-60 minutes** |

## Troubleshooting

### Memory Issues with Large Corpora

For very large corpora, process in chunks:

```bash
# Process vocabulary with lower top-n first
python scripts/build_vocabulary.py --corpus /path/to/openiti --top-n 50000
```

### Encoding Errors

The scripts try multiple encodings (UTF-8, CP1256, ISO-8859-6). If issues persist:

```python
# Convert files to UTF-8 first
import codecs
with codecs.open(file, 'r', 'cp1256') as f:
    text = f.read()
with codecs.open(file, 'w', 'utf-8') as f:
    f.write(text)
```

### Missing QALB Data

If QALB corpus is not available, the few-shot examples will be empty. Consider:

1. Creating manual examples in `qalb_few_shot.json`
2. Using the confusion matrix to generate synthetic examples

### LLM Rule Extraction Failures

If LLM extraction fails:

```bash
# Use default rules only
python scripts/extract_rules.py --default-only --output data/rules
```

## Contributing

When adding new knowledge base types:

1. Create a new script in `scripts/`
2. Follow the existing pattern (main function, save/load functions)
3. Add to `build_knowledge_bases.py` orchestrator
4. Update this documentation
