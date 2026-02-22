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
| **3** | OCR-Aware Prompting | Does OCR-specific knowledge help? | vs Phase 2 (**isolated**) |
| **4A** | Rule-Augmented | Do spelling rules help? | vs Phase 2 (**isolated**) |
| **4B** | Few-Shot Learning | Do correction examples help? | vs Phase 2 (**isolated**) |
| **4C** | CAMeL Validation | Does morphological post-processing help? | vs Phase 2 (**isolated**) |
| **5** | RAG | Does corpus grounding help? | vs Phase 2 (**isolated**) |
| **6** | Combinations + Ablation | What's optimal? What contributes? | Pairs + Full + Ablation |

**Key Design**: Phases 3-5 (including 4C) are **isolated experiments** comparing to Phase 2 baseline. Phase 6 tests meaningful combinations before ablation.

## Knowledge Sources

| Source | Location | Used In |
|--------|----------|---------|
| Confusion Matrix | Generated in Phase 1 | Phase 3, 6 |
| Arabic Rules | `./data/rules/` | Phase 4A, 6 |
| QALB Corpus | `./data/QALB-*/` | Phase 4B, 6 |
| CAMeL Tools | `pip install camel-tools` | Phase 1, 4C, 6 |
| OpenITI | `./data/OpenITI/` | Phase 5, 6 |

### CAMeL Tools (Morphological Analysis)

[CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) provides Arabic NLP utilities:
- **Morphological Analyzer**: Validate word existence, extract root/pattern
- **Disambiguator**: Context-aware analysis
- **Use Cases**: Error categorization (Phase 1), **Isolated validation test (Phase 4C)**, Combined system (Phase 6)

## Project Structure

```
Arabic-Post-OCR-Correction/
├── src/
│   ├── data/           # DataLoader, KnowledgeBase, TextUtils
│   ├── linguistic/     # CAMeL Tools wrappers (MorphAnalyzer, WordValidator)
│   ├── core/           # LLMCorrector, PromptBuilder, RAGRetriever
│   └── analysis/       # Metrics, ErrorAnalyzer, StatsTester, Visualizer
├── pipelines/          # run_phase1.py ... run_phase6.py
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
├── OpenITI/                        # Arabic corpus (RAG)
├── QALB-*/                         # Error-correction pairs
└── rules/                          # Arabic spelling rules
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

- [x] Architecture document created (8 phases)
- [x] Guidelines established
- [x] CAMeL Tools integration designed
- [x] Phase 1: Baseline & Error Taxonomy
- [x] Phase 2: Zero-Shot LLM
- [x] Phase 3: OCR-Aware Prompting
- [x] Phase 4A: Rule-Augmented
- [x] Phase 4B: Few-Shot (QALB)
- [x] Phase 4C: CAMeL Validation (isolated)
- [ ] Phase 5: RAG (OpenITI)
- [ ] Phase 6: Combinations + Ablation
