# Arabic Post-OCR Correction

## Research Question

**Can LLMs effectively fix OCR outputs and bridge the performance gap between open-source and closed-source VLMs?**

## Quick Context

- **Project Type**: Master's thesis + research paper
- **Problem**: Qaari (open-source Arabic OCR) produces errors; can LLMs correct them?
- **Datasets**: PATS-A01 (typewritten/synthetic), KHATT (handwritten/real)
- **Metrics**: CER (Character Error Rate), WER (Word Error Rate)
- **Model**: Qwen3-4B-Instruct-2507 (primary)

## Experimental Structure (8 Phases)

| Phase | Name | Research Question | Comparison |
|-------|------|-------------------|------------|
| **1** | Baseline & Error Taxonomy | How bad is Qaari? What errors? | N/A (no LLM) |
| **2** | Zero-Shot LLM | Can vanilla LLM fix OCR? | vs Phase 1 |
| **3** | OCR-Aware Prompting | Does OCR-specific knowledge help? (enhanced: filtered by LLM failures) | vs Phase 2 (**isolated**) |
| **4** | Self-Reflective | Do error patterns + overcorrection warnings help? (reads training artifacts) | vs Phase 2 (**isolated**) |
| **5** | CAMeL Validation | Does morphological post-processing help? (enhanced: known-overcorrection revert) | vs Phase 2 (**isolated**) |
| **6** | Combinations | What's optimal? What contributes? | conf_only, self_only, conf_self, best_camel |
| **7** | DSPy Prompt Optimization | Can automated optimization beat hand-crafted prompts? | vs Phase 2 |
| **8** | RAG (Retrieval-Augmented) | Does per-sample retrieval of similar corrections help? | vs Phase 2 (**isolated**) |

**Key Design**: Phases 3-5, 8, 9 are **isolated experiments** comparing to Phase 2 baseline. Phase 6 runs one new inference (true combination of Phase 3+4 full signals) and uses Phase 3/4 results directly as ablation baselines. Phase 7 uses DSPy to automatically discover optimal prompts. Phase 8 retrieves similar OCR-GT pairs via BM25 character n-grams. Phase 9 retrieves by error-signature similarity (CAMeL invalid words + confusion-matrix character profiles).

**Phase 4 pipeline**: Reads pre-computed training artifacts from `results/phase2-training/analysis/` — `word_pairs_llm_failures.txt` (UNFIXED + INTRODUCED sections) and `sample_classification.json`. No circular re-analysis.

**Removed phases**: Phase 4A (Rules) and Phase 4B (Few-Shot QALB) were removed — rules duplicated the base prompt, QALB taught grammar not OCR correction.

## Knowledge Sources

| Source | Location | Used In |
|--------|----------|---------|
| Confusion Matrix | Generated in Phase 1 | Phase 3, 6 |
| Training Artifacts | `results/phase2-training/analysis/` | Phase 3, 4, 5, 6 |
| CAMeL Tools | `pip install camel-tools` | Phase 1, 5, 6 |
| RAG Index (BM25) | Built from Phase 2 training corrections | Phase 8 |
| Error-Signature Index | Built from Phase 2 training corrections + CAMeL + confusion matrix | Phase 9 |

### CAMeL Tools (Morphological Analysis)

[CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) provides Arabic NLP utilities:
- **Morphological Analyzer**: Validate word existence, extract root/pattern
- **Disambiguator**: Context-aware analysis
- **Use Cases**: Error categorization (Phase 1), **Validation + known-overcorrection revert (Phase 5)**, Combined system (Phase 6)

## Project Structure

```
Arabic-Post-OCR-Correction/
├── src/
│   ├── data/           # DataLoader, KnowledgeBase, TextUtils
│   ├── linguistic/     # CAMeL Tools wrappers (MorphAnalyzer, WordValidator)
│   ├── core/           # LLMCorrector, PromptBuilder
│   └── analysis/       # Metrics, ErrorAnalyzer, StatsTester, Visualizer
├── pipelines/          # run_phase1.py ... run_phase5.py, run_phase4d.py, run_phase7.py, run_phase8.py, run_phase9.py
├── configs/            # config.yaml
├── results/            # Phase outputs (gitignored)
├── docs/               # Architecture.md, Guidelines.md
├── scripts/            # Utility scripts
└── tests/
```

## Data Location

```
./data/
├── ocr-results/                    # OCR predictions (one sub-folder per model)
│   └── qaari-results/              # Qaari outputs (active model)
│       ├── pats-a01-data/A01-Akhbar/
│       └── khatt-data/{train,validation}/
├── ocr-raw-data/                   # Original ground-truth texts
│   ├── PATS_A01_Dataset/
│   └── KHATT/
```

To switch OCR models: set `data.ocr_model` in `configs/config.yaml`.

## Key Commands

```bash
# Run specific phase
python pipelines/run_phase1.py
python pipelines/run_phase2.py

# Run all phases
python pipelines/run_all.py

# Run with sample limit (for testing)
python pipelines/run_phase1.py --limit 50
```

## Development Guidelines

1. **KISS**: Simple code that works
2. **Research-first**: Everything should produce paper-ready numbers
3. **Type hints**: All functions must have type annotations
4. **UTF-8**: Always specify encoding for Arabic text
5. **Documentation**: Update CHANGELOG.md on changes

See `docs/Guidelines.md` for full standards.

## Key Files

| File | Purpose |
|------|---------|
| `docs/Architecture.md` | Full system design and phase details |
| `docs/Guidelines.md` | Coding standards and conventions |
| `configs/config.yaml` | Runtime configuration |
| `CHANGELOG.md` | Version history |

## Current Status

- [x] Architecture document created (7 phases)
- [x] Guidelines established
- [x] CAMeL Tools integration designed
- [x] Phase 1: Baseline & Error Taxonomy
- [x] Phase 2: Zero-Shot LLM
- [x] Phase 3: OCR-Aware Prompting (enhanced: cross-referenced with LLM failures)
- [x] Phase 4: Self-Reflective (enhanced: reads training artifacts, overcorrection warnings)
- [x] Phase 5: CAMeL Validation (enhanced: known-overcorrection revert)
- [x] Phase 6: Combinations (redesigned: 1 inference conf_self + ablation via Phase 3/4 results + 1 CAMeL combo)
- [x] Phase 7: DSPy Prompt Optimization
- [x] Phase 8: RAG (BM25 character n-gram retrieval)
- [x] Phase 9: Error-Signature RAG (CAMeL + confusion matrix + error-structural similarity)
