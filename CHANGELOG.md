# Changelog

All notable changes to the Arabic Post-OCR Correction project.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.0] - 2026-02-20 — Phase 1 Implementation

### Added
- `docs/Phase1_Design.md` — detailed implementation plan for Phase 1 (data flow, schemas, module specs)
- `configs/config.yaml` — full runtime configuration for all phases
- `src/data/text_utils.py` — Arabic text normalisation, repetition stripping, tokenisation
- `src/data/data_loader.py` — `DataLoader`, `OCRSample`, `DataError`; KHATT filename pairing; PATS GT handling
- `src/analysis/metrics.py` — `calculate_cer()`, `calculate_wer()`, `calculate_metrics()`, `calculate_metrics_split()`, `MetricResult`
- `src/analysis/error_analyzer.py` — `ErrorAnalyzer`, `ErrorType`, `ErrorPosition`, character alignment, confusion matrix, error taxonomy
- `src/linguistic/morphology.py` — `MorphAnalyzer` CAMeL Tools wrapper with LRU cache and graceful fallback
- `src/linguistic/validator.py` — `WordValidator`, `ValidationResult`
- `pipelines/run_phase1.py` — end-to-end Phase 1 pipeline with CLI, logging, progress bars, JSON outputs, Markdown report
- `tests/test_text_utils.py` — 28 tests for text utilities
- `tests/test_metrics.py` — 15 tests for CER/WER/MetricResult
- `tests/test_error_analyzer.py` — 25 tests for alignment, classification, confusion matrix, taxonomy
- `tests/test_data_loader.py` — 17 tests for loading, pairing, empty-file handling
- `requirements.txt` updated with `editdistance>=0.6.0`, `jiwer>=3.0.0`, `pytest>=7.0.0`

### Phase 1 Results (All Datasets — Qaari OCR Baseline)
| Dataset | CER (all) | CER (normal) | WER (normal) | Runaway% | N |
|---------|-----------|--------------|--------------|----------|---|
| PATS-A01-Akhbar | 32.62% | 2.05% | 5.11% | 1.3% | 2766 |
| PATS-A01-Andalus | 107.90% | 13.71% | 36.20% | 4.1% | 2764 |
| KHATT-train | 553.56% | 65.92% | 93.29% | 19.0% | 1393 |
| KHATT-validation | 434.85% | 60.03% | 90.13% | 14.8% | 230 |

- **CER (normal)**: excludes samples where Qaari output is >5× GT length (runaway repetition bug)
- **PATS-A01-Akhbar**: very clean — CER(normal)=2.05%; top confusions: ي↔ب (dot), ط↔ظ (dot)
- **PATS-A01-Andalus**: moderate errors — CER(normal)=13.71%; top confusions: م↔ه (shape), ى↔ي (alef-maksura)
- **KHATT**: handwritten — much harder; top confusions: Arabic chars confused with diacritics (ت→fatha, ي→kasra)
- **PATS GT**: cp1256-encoded line files at `../data/train/PATS_A01_Dataset/A01-{Font}Text.txt`; pairing: GT line N ↔ `{Font}_N.txt`
- **85/85 tests passing**

### Added
- `docs/Architecture.md` - Comprehensive 8-phase experimental architecture
- `docs/Guidelines.md` - Extended development guidelines and coding standards
- **Phase 4C: CAMeL Validation** - Isolated test of morphological post-processing
  - Tests whether CAMeL validation alone improves zero-shot results
  - Consistent with isolated comparison design of Phases 3-5
  - Compares prompt-based knowledge vs post-processing approaches
- **Phase 6 Combination Testing** - Hierarchical combination experiments
  - Top pairs testing (not just ablation)
  - Tests interaction effects between components
  - Answers "which components synergize?"
- **CAMeL Tools integration** - Arabic morphological analysis for:
  - Enhanced error categorization (non-word vs valid-but-wrong)
  - Post-LLM correction validation
  - Hybrid neural-symbolic approach
- New `src/linguistic/` layer planned:
  - `morphology.py` - MorphAnalyzer wrapper
  - `validator.py` - WordValidator for correction quality
  - `features.py` - Linguistic feature extraction
- Innovative knowledge integration strategy using:
  - Confusion Matrix (Phase 3) - OCR-specific error patterns
  - Arabic Rules (Phase 4A) - Orthographic rules injection
  - QALB Corpus (Phase 4B) - Few-shot learning examples
  - CAMeL Tools (Phase 4C) - Isolated morphological validation
  - OpenITI Corpus (Phase 5) - Retrieval-Augmented Generation

### Changed
- Restructured from 6-phase to 8-phase experimental design
  - Added Phase 4C for isolated CAMeL testing (methodological consistency)
  - Expanded Phase 6 with combination experiments (beyond just ablation)
- Each phase now tests a specific research hypothesis
- Updated `CLAUDE.md` with new architecture and phase overview
- Data paths updated to reflect actual directory structure
- **Research design clarification**: Phases 3-5 (including 4C) are now explicitly **isolated comparisons** to Phase 2 baseline
  - Phase 2 is the hub: all knowledge-enhanced phases compare to it
  - This enables measuring individual contribution of each knowledge type
  - Phase 6 tests combinations then performs ablation studies
- Updated experimental flow diagram to show Phase 2 as comparison hub
- Added explicit research questions and comparison statements to each phase
- Phase 6 now includes:
  - Level 1: Individual effects (Phases 3-5) ✓
  - Level 2: Top pairs (synergy detection)
  - Level 3: Full system
  - Level 4: Ablation (component necessity)

### Removed
- Old documentation files (`README_phase1.md`, `README_knowledge_bases.md`, `prompts.txt`)
- Previous source code (`src/`, `configs/`) - starting fresh implementation

## [0.0.1] - Initial Planning

### Added
- Project initialization
- Basic directory structure
- Research question: "Can LLMs bridge the gap between open-source and closed-source VLMs?"
