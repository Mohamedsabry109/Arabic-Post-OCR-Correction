# Changelog

All notable changes to the Arabic Post-OCR Correction project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial project structure with Phase 1 implementation
- Core modules: `data_loader.py`, `metrics.py`, `llm_corrector.py`, `utils.py`
- Knowledge base construction scripts in `scripts/`
- Configuration files in `configs/`
- Documentation in `docs/`
- Project guidelines in `CLAUDE.md`

### Phase 1 Components
- Zero-shot LLM correction using Qwen2.5-3B-Instruct
- CER/WER metrics calculation
- Support for PATS-A01 and KHATT datasets
- Arabic text normalization utilities

### Knowledge Base Scripts
- `analyze_qaari_errors.py` - OCR error analysis and confusion matrix
- `build_vocabulary.py` - Vocabulary extraction from OpenITI
- `build_ngrams.py` - N-gram statistics builder
- `extract_qalb.py` - QALB corpus processor
- `extract_rules.py` - Arabic grammar rules extractor
- `build_knowledge_bases.py` - Master orchestrator

## [0.1.0] - Initial Setup

### Added
- Project initialization
- Basic directory structure
- Requirements specification
