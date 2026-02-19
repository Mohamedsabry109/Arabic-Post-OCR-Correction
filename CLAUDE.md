# Arabic Post-OCR Correction

## Project Overview

Research project for Arabic post-OCR text correction using Large Language Models (LLMs). The system improves OCR output quality through a phased approach combining zero-shot LLM correction with linguistic knowledge integration.

## Architecture

```
Phase 1: Zero-shot LLM Baseline
    └── Qwen2.5-3B-Instruct → CER/WER metrics

Phase 2: LLM + Smart Candidates
    └── + Confusion matrix + Vocabulary → Improved metrics

Phase 3: Full Knowledge Integration
    └── + Morphology + Rules + N-grams + QALB → Best metrics
```

## Project Structure

```
Arabic-Post-OCR-Correction/
├── src/                      # Core source code
│   ├── __init__.py
│   ├── utils.py              # Arabic text processing utilities
│   ├── data_loader.py        # Dataset loading and validation
│   ├── metrics.py            # CER/WER calculation
│   └── llm_corrector.py      # LLM-based correction
├── scripts/                  # Knowledge base construction
│   ├── analyze_qaari_errors.py
│   ├── build_vocabulary.py
│   ├── build_ngrams.py
│   ├── extract_qalb.py
│   ├── extract_rules.py
│   └── build_knowledge_bases.py
├── configs/                  # Configuration files
│   ├── config.yaml           # Phase 1 config
│   └── kb_config.yaml        # Knowledge base config
├── docs/                     # Documentation
├── results/                  # Output directory
├── run_phase1.py             # Phase 1 pipeline
└── requirements.txt
```

## Data Location

Data is stored in `../data` relative to project root:
```
../data/
├── Original/           # Ground truth text files
│   ├── PATS-A01/
│   └── KHATT/
└── Predictions/        # OCR predictions (Qaari output)
    ├── PATS-A01/
    └── KHATT/
```

## Key Commands

```bash
# Run Phase 1 pipeline
python run_phase1.py

# Run with specific datasets
python run_phase1.py --datasets PATS-A01

# Limit samples for testing
python run_phase1.py --limit 50

# Build knowledge bases
python scripts/build_knowledge_bases.py --all
```

## Metrics

- **CER (Character Error Rate)**: `(Substitutions + Deletions + Insertions) / Reference Length`
- **WER (Word Error Rate)**: Word-level edit distance normalized by reference word count

## Development Guidelines

1. **KISS Principle**: Keep implementations simple and focused
2. **Maintainability**: Code should be extensible for future phases
3. **Research Focus**: All metrics should be calculated for research paper reporting
4. **SWE Best Practices**: Type hints, docstrings, error handling
5. **Documentation**: Update CLAUDE.md and CHANGELOG.md when making changes

## Configuration

Main configuration in `configs/config.yaml`:
- Model: `Qwen/Qwen2.5-3B-Instruct` (or `Qwen3-4B-Instruct`)
- Temperature: 0.1 (deterministic output)
- Datasets: PATS-A01, KHATT

## Knowledge Bases (Phase 2/3)

- `confusion_matrix.json` - Qaari OCR error patterns
- `vocab_100k.json` - Arabic word frequencies from OpenITI
- `ngrams.json` - Bigram/trigram statistics
- `qalb_corrections.json` - Error-correction pairs
- `rules/` - Arabic grammar/spelling rules

## Dependencies

Core: `torch`, `transformers`, `accelerate`, `python-Levenshtein`, `tqdm`, `PyYAML`

Optional: `bitsandbytes` (quantization), `pyarabic` (text normalization)

## Changelog

See CHANGELOG.md for version history.
