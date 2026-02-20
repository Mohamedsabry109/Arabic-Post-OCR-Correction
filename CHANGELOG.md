# Changelog

All notable changes to the Arabic Post-OCR Correction project.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
