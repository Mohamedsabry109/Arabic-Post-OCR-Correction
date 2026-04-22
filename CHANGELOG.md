# Changelog

All notable changes to the Arabic Post-OCR Correction project.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.1.0] - 2026-04-18 — Phase 8: RAG (Retrieval-Augmented Correction)

### Added
- `src/data/rag_index.py` — RAG index and retriever for OCR correction
  - `RAGIndexBuilder`: builds BM25 indices over character n-grams from Phase 2 training corrections
  - Sentence-level store (OCR-GT pairs) and word-level store (error word pairs via difflib)
  - Optional dense retrieval: sentence-transformers + FAISS (hybrid mode)
  - `save()`/`load()` for index serialization (JSONL + pickle + FAISS)
  - `RAGRetriever`: per-sample retrieval with BM25, dense, or hybrid scoring
  - `retrieve_sentences()` / `retrieve_words()` with formatted prompt output
- `pipelines/run_phase8.py` — 3-mode pipeline (build-index / export / analyze)
  - `build-index`: discovers Phase 2 training corrections, builds and saves RAG index
  - `export`: retrieves per-sample context, writes inference_input.jsonl with prompt_type="rag"
  - `analyze`: CER/WER metrics, Phase 2 comparison, error change analysis, Markdown report
  - Retrieval diagnostics saved per dataset (hit rate, avg scores, coverage)
- `src/core/prompt_builder.py` — `build_rag()` method with `<retrieved_corrections>` and
  `<retrieved_word_fixes>` XML sections; prompt version `p8v1`
- `scripts/infer.py` — `"rag"` prompt_type dispatch branch
- `configs/config.yaml` — `phase8:` config block (retrieval_mode, alpha, n-gram sizes,
  top_k settings, embedding config, index filters)
- `requirements.txt` — added `rank-bm25>=0.2.2`; optional `faiss-cpu`, `sentence-transformers`

### Changed
- `CLAUDE.md` — updated to 8-phase structure, added Phase 8 to table and status

## [1.0.0] - 2026-04-09 — Phase refactoring: remove 4A/4B, enhance 3/4/5, redesign combos

### Removed
- **Phase 4A (Rule-Augmented)**: rules duplicated base prompt, no measurable gain
- **Phase 4B (Few-Shot QALB)**: QALB corpus teaches grammar not OCR correction
- `ArabicRule`, `RulesLoader`, `QALBPair`, `QALBLoader` from `src/data/knowledge_base.py`
- `build_rule_augmented()`, `build_few_shot()` from `src/core/prompt_builder.py`
- `rule_augmented`, `few_shot` dispatch from `scripts/infer.py`
- Related tests from `test_prompt_builder.py`, `test_knowledge_base.py`
- Old 10-combo structure from Phase 5/6 (pair_conf_rules, abl_no_*, full_prompt, etc.)

### Changed
- **Phase 3 (OCR-Aware)**: enhanced with training cross-reference
  - `ConfusionMatrixLoader.filter_by_llm_failures()` filters confusion pairs by LLM failure data
  - `format_word_examples_for_prompt()` adds concrete word-level failure examples
  - Config: `phase3.use_training_failures`, `phase3.training_failures_path`
- **Phase 4 (Self-Reflective)**: reads pre-computed training artifacts (no circular re-analysis)
  - Loads UNFIXED/INTRODUCED sections from `word_pairs_llm_failures.txt`
  - New `overcorrection_context` warns LLM about known bad corrections
  - `PromptBuilder.build_self_reflective()` accepts `overcorrection_context` param
  - Config: `phase4.training_artifacts_dir`, `phase4.overcorrection_n`
- **Phase 5 (CAMeL Validation)**: enhanced with known-overcorrection revert
  - `WordValidator.validate_correction()` accepts `known_overcorrections` set
  - Reverts known bad LLM corrections BEFORE morphological checks
  - Fixed position-indexed revert bug (was keyed by word, not position)
- **Phase 6 (Combinations)**: redesigned from 10+ combos to 3+1
  - Inference combos: `conf_only`, `self_only`, `conf_self`
  - CAMeL combo: `best_camel` (best inference combo + CAMeL post-processing)
  - Updated summarize: simplified ablation, synergy analysis for 2-component structure
- **New phase numbering**: 1, 2, 3, 4, 5, 6, 7 (was 1, 2, 3, 4A-4D, 5, 7)
- `CLAUDE.md`: updated to 7-phase structure
- `run_all.py`: updated phase dispatch, removed 4A/4B entries
- `configs/config.yaml`: removed `data.qalb`/`data.rules`, old phase4 sections

### Added
- `load_unfixed_word_pairs()`, `load_introduced_word_pairs()` in `knowledge_base.py`
- `format_word_examples_for_prompt()`, `format_overcorrection_warnings()` in `knowledge_base.py`
- `ConfusionMatrixLoader._char_confusions_from_word_pair()` for difflib-based extraction

## [0.9.0] - 2026-03-26 — Rename Phase 6 to Phase 5; add Phase 7 (DSPy)

### Added
- `pipelines/run_phase7.py` — Phase 7: DSPy automated prompt optimization pipeline
  - `export` mode: samples train/dev sets for DSPy, exports full val inference_input.jsonl
  - `analyze` mode: CER/WER analysis of DSPy-optimized corrections vs Phase 2
- `scripts/dspy_optimize.py` — Kaggle script for DSPy BootstrapFewShot optimization
  - `LocalTransformersLM` adapter: wraps Qwen3 for DSPy compatibility
  - Runs optimization on small train/dev sets, then full inference with compiled program
  - Outputs: `dspy_compiled.json`, `optimized_prompt.txt`, `corrections.jsonl`
- `configs/config.yaml` — added `phase7:` config block (n_train, n_dev, optimizer settings)

### Changed
- **Phase 6 renamed to Phase 5**: all files, paths, config keys updated
  - `pipelines/run_phase6.py` -> `pipelines/run_phase5.py`
  - `results/phase6/` -> `results/phase5/`
  - `config.yaml`: `phase6:` -> `phase5:`
  - `PromptBuilder.COMBINED_PROMPT_VERSION`: `p6v2` -> `p5v1`
- `pipelines/run_all.py` — updated `_ALL_PHASES`, `_PHASE_STEPS`, `_INFERENCE_IO` for Phase 5 rename + Phase 7
- `src/core/prompt_builder.py` — Phase 6 references updated to Phase 5
- `CLAUDE.md` — 8-phase table (added Phase 7: DSPy), updated knowledge sources

### Removed
- `pipelines/run_phase6.py` — replaced by `run_phase5.py`

## [0.8.0] - 2026-03-25 — Remove Phase 5 (RAG); fuse Phase 4D + 4E

### Removed
- `pipelines/run_phase5.py` — RAG pipeline (OpenITI corpus retrieval) dropped; conceptually weak fit for character-level OCR correction
- `src/core/rag_retriever.py` — RAGRetriever, FAISS index builder
- `pipelines/run_phase4e.py` — standalone Word-Error Pairs pipeline; functionality merged into Phase 4D
- `PromptBuilder.build_rag()`, `build_word_error_pairs()` — superseded methods removed
- `configs/config.yaml` — removed `rag:`, `phase5:`, `phase4e:` sections; removed `openiti:` data path
- Phase 6 combos `pair_conf_rag` and `abl_no_rag` removed; `full_prompt` now = conf+rules+fewshot

### Changed
- `PromptBuilder.build_self_reflective()` — now accepts optional `word_pairs_context`; injects both signals as numbered rules 7+8; prompt version bumped to `p4dv3`
- `PromptBuilder.build_combined()` — replaced `retrieval_context` param with `word_pairs_context`
- `pipelines/run_phase4d.py` — export mode loads `results/phase4d/word_error_pairs.txt` (auto-generated by analyze-train) and injects word pairs alongside insights; `analyze-train` mode is where word pairs extraction logic should be added
- `pipelines/run_phase6.py` — COMBO_COMPONENTS 5-tuple → 4-tuple; `use_self=True` combos now also load word pairs from Phase 4D outputs; `phase4d_self` added to summarize paper table
- `scripts/infer.py` — removed `rag` and `word_error_pairs` dispatch; updated `self_reflective` and `combined` to pass `word_pairs_context`
- `CLAUDE.md` — updated phase table (7 phases), knowledge sources, project structure

## [0.7.0] - 2026-03-02 — Phase 4D: Self-Reflective Prompting

### Added
- `src/analysis/llm_error_analyzer.py` — `LLMErrorAnalyzer` class: analyses LLM
  predictions on training splits, computes per-ErrorType fixed/introduced counts,
  and aggregates into dataset-level insights JSON
- `src/data/knowledge_base.py` — `LLMInsightsLoader` class: loads Phase 4D insights
  JSON and formats Arabic weakness/over-correction sections for prompt injection
- `src/core/prompt_builder.py` — `build_self_reflective()` with prompt version `p4dv1`;
  extended `build_combined()` with `insights_context` parameter for Phase 6 combos
- `pipelines/run_phase4d.py` — 3-mode pipeline:
  - `analyze-train`: analyses LLM training predictions vs GT, saves insights JSON
  - `export`: builds val-split inference JSONL with self-reflective prompts
  - `analyze`: post-Kaggle CER/WER metrics, comparison vs Phase 2, paper_tables.md
- `configs/config.yaml` — added `phase4d:` config block (source_phase, insights thresholds)

### Changed
- `pipelines/run_phase6.py` — extended `COMBO_COMPONENTS` to 5-tuples
  `(use_confusion, use_rules, use_fewshot, use_rag, use_self)`; added 3 new combos:
  `self_reflective`, `pair_self_conf`, `full_with_self`; updated `run_export()` to
  load Phase 4D insights and pass `insights_context` to `build_combined()`
- `scripts/infer.py` — added `"self_reflective"` prompt_type dispatch;
  extended `"combined"` dispatch to pass `insights_context`

## [0.6.0] - 2026-02-23 — Phase 6: Combinations & Ablation Study

### Added
- `pipelines/run_phase6.py` — 4-mode pipeline (export/analyze/validate/summarize)
  - 9 inference combos (pair_conf_rules, pair_conf_fewshot, pair_conf_rag,
    pair_rules_fewshot, full_prompt, abl_no_confusion, abl_no_rules,
    abl_no_fewshot, abl_no_rag)
  - 2 CAMeL post-processing combos (pair_best_camel, full_system) — local only
  - Summarize mode: synergy analysis, ablation impact, paper_tables.md
- `src/analysis/stats_tester.py` — `StatsTester` class with paired t-test,
  Cohen's d effect size, Bonferroni correction, bootstrap CI; scipy optional
  with normal-approximation fallback
- `src/core/prompt_builder.py` — `build_combined()` merging confusion/rules/
  few-shot/RAG contexts in fixed order; prompt version `p6v1`
- `docs/Phase6_Design.md` — full design document for Phase 6
- `pipelines/run_all.py` — sequential orchestrator for all phases (export/analyze/full modes)
- `tests/test_stats_tester.py` — 33 tests covering all StatsTester methods
- `tests/test_prompt_builder.py` — 36 tests covering all build_* methods
  including build_combined context inclusion/exclusion/ordering

### Changed
- `scripts/infer.py` — added `combined` prompt_type dispatch for Phase 6
- `configs/config.yaml` — added `phase6:` config block (pair_best, stats alpha/n_bootstrap)
- `requirements.txt` — added `scipy>=1.10.0`
- `HOW_TO_RUN.md` — added full Phase 6 section with combo table, all modes, and workflow

## [0.5.0] - 2026-02-22 — Phase 5: RAG with OpenITI Corpus

### Added
- `pipelines/run_phase5.py` — 3-mode pipeline (build/export/analyze)
  - Build mode: extracts up to 200K sentences from OpenITI, embeds with
    sentence-transformers, saves FAISS index
  - Export mode: retrieves top-k similar sentences per OCR sample
  - Analyze mode: CER/WER comparison vs Phase 2, retrieval quality statistics
- `src/core/rag_retriever.py` — `RAGRetriever` with build_index, load_index,
  retrieve, format_for_prompt; lazy numpy/faiss imports
- `src/data/knowledge_base.py` — `OpenITILoader`, `CorpusSentence` for corpus
  extraction with stratified era sampling
- `src/core/prompt_builder.py` — `build_rag()` with prompt version `p5v1`
- `docs/Phase5_Design.md` — full design document for Phase 5
- `configs/config.yaml` — added `phase5:` config block

## [0.4.0] - 2026-02-22 — Phase 4: Linguistic Knowledge Enhancement

### Added
- `pipelines/run_phase4.py` — unified pipeline for sub-phases 4A, 4B, 4C
  - Phase 4A (rule_augmented): Arabic orthographic rules injected into prompt
  - Phase 4B (few_shot): QALB error-correction examples injected into prompt
  - Phase 4C (camel_validation): CAMeL morphological post-processing of Phase 2
    corrections using revert strategy (no inference needed)
- `src/data/knowledge_base.py` — `RulesLoader` (Phase 4A) and `QALBLoader`
  (Phase 4B) with OCR-error filtering and diverse example selection
- `src/linguistic/validator.py` — `WordValidator.validate_correction()` revert
  strategy: reverts LLM-introduced invalid words back to OCR words
- `src/core/prompt_builder.py` — `build_rule_augmented()` (p4av1) and
  `build_few_shot()` (p4bv1)
- `configs/config.yaml` — added `phase4:` config block with rules, few-shot,
  and camel_validation settings

## [0.3.0] - 2026-02-22 — Phase 3: OCR-Aware Prompting

### Added
- `pipelines/run_phase3.py` — 3-mode pipeline (export/analyze/full)
  - Injects top-N character confusion pairs from Phase 1 into the system prompt
  - Per-dataset confusion matrix with pooled fallback for sparse datasets
  - Confusion impact analysis: per-injected-pair fix-rate vs Phase 2
- `src/data/knowledge_base.py` — `ConfusionMatrixLoader` with
  `format_for_prompt()` supporting flat_arabic and grouped_arabic styles
- `src/core/prompt_builder.py` — `build_ocr_aware()` with prompt version `p3v1`
- `docs/Phase3_Design.md` — full design document for Phase 3
- `configs/config.yaml` — added `phase3:` config block

## [0.2.0] - 2026-02-22 — Phase 2: Zero-Shot LLM Correction

### Added
- `pipelines/run_phase2.py` — export/analyze pipeline for zero-shot LLM correction
- `scripts/infer.py` — unified inference script (local/Kaggle/Colab) with
  HuggingFace cross-session sync and automatic resume
- `src/core/llm_corrector.py` — `BaseLLMCorrector` ABC, `TransformersCorrector`
  (HuggingFace), `APICorrector` stub; `get_corrector()` factory
- `src/core/prompt_builder.py` — `PromptBuilder` with `build_zero_shot()` (v1)
- `notebooks/kaggle_setup.ipynb` — Kaggle notebook: clone repo + run infer.py
- `notebooks/colab_setup.ipynb` — Colab notebook: mount Drive + run infer.py
- `docs/Kaggle_Colab_Guide.md` — full remote inference workflow documentation
- `pipelines/_utils.py` — `resolve_datasets()` helper used by all pipelines
- `configs/config.yaml` — added `phase2:` config block and `model:` settings

### Changed
- Established 3-stage pipeline pattern (export → infer → analyze) used by all
  subsequent phases; `enable_thinking=False` in Qwen3 chat template to suppress
  `<think>` scratchpad tokens

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
- **PATS GT**: cp1256-encoded line files at `./data/ocr-raw-data/PATS_A01_Dataset/A01-{Font}Text.txt`; pairing: GT line N ↔ `{Font}_N.txt`
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
