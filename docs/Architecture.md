# Arabic Post-OCR Correction: Software Architecture

## 1. Research Overview

### 1.1 Research Question

**Can Large Language Models effectively correct OCR errors in Arabic text, bridging the performance gap between open-source and closed-source Vision-Language Models?**

### 1.2 Sub-Questions (Tested by Each Phase)

| Phase | Research Sub-Question | Comparison |
|-------|----------------------|------------|
| Phase 1 | What errors does Qaari make? How severe? | N/A (No LLM) |
| Phase 2 | Can vanilla LLM correct Arabic OCR errors? | vs Phase 1 (OCR baseline) |
| Phase 3 | Does OCR-specific error knowledge help? (enhanced: filtered by LLM failures) | vs Phase 2 (isolated) |
| Phase 4 | Do error patterns + overcorrection warnings help? (self-reflective) | vs Phase 2 (isolated) |
| Phase 5 | Does morphological post-processing help? (enhanced: known-overcorrection revert) | vs Phase 2 (isolated) |
| Phase 6 | What combination is optimal? What contributes? | conf_only, self_only, conf_self, best_camel |
| Phase 7 | Can automated prompt optimization beat hand-crafted prompts? | vs Phase 2 |

**Key Design Principle**: Phases 3–5 are **isolated experiments** comparing to Phase 2 baseline. Phase 6 tests 3 inference combos (confusion only, self-reflective only, both) plus 1 CAMeL combo. Phase 7 uses DSPy to automatically discover optimal prompts.

**Removed phases**: Phase 4A (Rules) and Phase 4B (Few-Shot QALB) were removed — rules duplicated the base prompt, QALB taught grammar not OCR correction.

### 1.3 Datasets

| Dataset | Type | Characteristics | Research Value |
|---------|------|-----------------|----------------|
| **PATS-A01** | Typewritten/Synthetic | Various fonts, controlled | Clean baseline, font variation effects |
| **KHATT** | Handwritten/Real | Natural variation | Real-world performance |

### 1.4 Knowledge Sources

| Source | Contents | How We Use It | Used In |
|--------|----------|---------------|---------|
| **Confusion Matrix** | Qaari's character errors (Phase 1 output) | Tell LLM what to watch for | Phase 3, 6 |
| **Training Artifacts** | LLM failure word pairs + sample classification | Overcorrection warnings, failure context | Phase 3, 4, 5, 6 |
| **CAMeL Tools** | Morphological analyzer, disambiguator | Validate corrections, error categorization | Phase 1, 5, 6 |

### 1.5 CAMeL Tools Integration

[CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) is an Arabic NLP toolkit from NYU Abu Dhabi providing morphological analysis, disambiguation, and text utilities.

**Rationale for Integration:**
- **Morphological Validation**: Detect if LLM corrections produce valid Arabic words
- **Enhanced Error Analysis**: Categorize OCR errors linguistically (root vs surface errors)
- **Hybrid Approach**: Combine neural (LLM) with symbolic (morphology) methods
- **Quality Assurance**: Catch LLM hallucinations that produce non-words

**Key Components Used:**

| Component | Module | Usage |
|-----------|--------|-------|
| Morphological Analyzer | `camel_tools.morphology` | Validate word existence, get word features |
| Disambiguator | `camel_tools.disambig` | Context-aware analysis |
| Text Cleaner | `camel_arclean` | Consistent preprocessing |
| Tokenizer | `camel_tools.tokenizers` | Handle Arabic segmentation |

**Integration Points:**
1. **Phase 1**: Enhanced error categorization (morphologically invalid vs valid-but-wrong)
2. **Phase 5**: Post-LLM validation with known-overcorrection revert strategy
3. **Phase 6**: CAMeL combo (`best_camel`) applies morphological post-processing on best inference combo

---

## 2. Experimental Phases

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              EXPERIMENTAL FLOW                                        │
│                                                                                       │
│  ┌─────────────┐                                                                     │
│  │   PHASE 1   │  Baseline & Error Taxonomy (NO LLM)                                 │
│  │  Analysis   │  → Problem quantification, confusion matrix                         │
│  └──────┬──────┘                                                                     │
│         │                                                                             │
│         ▼                                                                             │
│  ┌─────────────┐                                                                     │
│  │   PHASE 2   │  Zero-Shot LLM ════════════════════════════════════════╗            │
│  │  BASELINE   │  → BASELINE FOR ALL COMPARISONS                        ║            │
│  └──────┬──────┘                                                        ║            │
│         │                                                               ║            │
│         ├─────────────────────┬─────────────────────┐                  ║            │
│         │                     │                     │                  ║            │
│         ▼                     ▼                     ▼                  ▼            │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │   PHASE 3    │  │    PHASE 4       │  │   PHASE 5    │  │    PHASE 7      │     │
│  │  OCR-Aware   │  │  Self-Reflective │  │    CAMeL     │  │  DSPy Prompt    │     │
│  │ (+confusion  │  │ (+error patterns │  │  Validation  │  │  Optimization   │     │
│  │  +LLM fails) │  │  +overcorrection │  │ (+known-bad  │  │  (auto-tuned)   │     │
│  │  vs Ph2 ▲   │  │   warnings)      │  │   revert)    │  │  vs Ph2 ▲      │     │
│  └──────┬───────┘  │  vs Ph2 ▲       │  │  vs Ph2 ▲   │  └─────────────────┘     │
│         │          └────────┬─────────┘  └──────┬───────┘                          │
│         │                   │                   │                                   │
│         └───────────────────┴───────────────────┘                                   │
│                                     │                                               │
│                                     ▼                                               │
│                          ┌─────────────────────┐                                   │
│                          │      PHASE 6        │                                   │
│                          │   Combinations:     │                                   │
│                          │  conf_only          │                                   │
│                          │  self_only          │                                   │
│                          │  conf_self          │                                   │
│                          │  best_camel         │                                   │
│                          └─────────────────────┘                                   │
│                                                                                       │
│  Legend: ▲ = Isolated comparison to Phase 2 baseline                                 │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Phase 2 is the hub**: All knowledge-enhanced phases (3–5) compare to Phase 2 baseline
2. **Isolated experiments**: Each of phases 3–5 tests ONE knowledge addition in isolation
3. **Training artifacts**: Phases 3, 4, 5, 6 read pre-computed artifacts from `results/phase2-training/analysis/` — no circular re-analysis
4. **Final synthesis**: Phase 6 tests 3 inference combos + 1 CAMeL combo to find optimal configuration

---

## Phase 1: Baseline & Error Taxonomy

### Purpose
Establish the problem severity and deeply understand Qaari's error patterns.

### Process
1. Load all OCR predictions and ground truth
2. Calculate baseline CER/WER for each dataset
3. Build character-level confusion matrix
4. Categorize errors into types
5. Analyze error positions (start/middle/end of word)
6. **[CAMeL]** Morphological error analysis (optional enhancement)

### Error Categories to Identify

| Category | Arabic Term | Example |
|----------|-------------|---------|
| Dot confusion | نقط | ب↔ت↔ث↔ن |
| Hamza errors | همزة | أ↔ا↔إ↔آ |
| Taa Marbuta | تاء مربوطة | ة↔ه |
| Alef Maksura | ألف مقصورة | ى↔ي |
| Similar shapes | تشابه | ر↔ز، د↔ذ |
| Merged words | دمج | كلمتين←كلمة |
| Split words | فصل | كلمة←كلمتين |
| Missing chars | حذف | حروف ناقصة |
| Extra chars | زيادة | حروف زائدة |

### Morphological Error Analysis (CAMeL Tools)

When CAMeL Tools is enabled, errors are further categorized:

| Category | Description | Detection Method |
|----------|-------------|------------------|
| **Non-word** | OCR output is not a valid Arabic word | `MorphAnalyzer.analyze()` returns empty |
| **Valid-but-wrong** | OCR output is valid but incorrect | Compare with ground truth |
| **Root-level** | Error affects the word root | Extract root, compare |
| **Affix-level** | Error in prefix/suffix only | Compare morphological breakdown |
| **Diacritic-level** | Only diacritics are wrong | Compare after diacritic removal |

This deeper categorization helps understand:
- How many errors are "obviously wrong" (non-words) vs "plausible errors"
- Whether LLM correction is needed or simple dictionary lookup suffices

### Outputs
```
results/phase1/
├── baseline_metrics.json      # CER/WER per dataset
├── confusion_matrix.json      # {true_char: {ocr_char: count, ...}, ...}
├── error_taxonomy.json        # Categorized error statistics
├── error_examples.json        # Sample errors per category
├── morphological_analysis.json  # [CAMeL] Word validity stats
└── report.md                  # Human-readable analysis
```

### Research Value
- Quantifies the problem
- Informs prompt design for later phases
- Error taxonomy appears in paper's analysis section
- **[CAMeL]** Morphological breakdown shows percentage of "obvious" vs "subtle" errors

---

## Phase 2: Zero-Shot LLM Correction (BASELINE)

> **CRITICAL**: Phase 2 is the **baseline for ALL subsequent comparisons**.
> Phases 3, 4, and 5 are isolated experiments comparing to Phase 2.

### Research Question
**Can a vanilla LLM correct Arabic OCR errors without task-specific guidance?**

### Purpose
Establish what an unguided LLM can achieve — this becomes the baseline for measuring knowledge contributions.

### Prompt Design
```
System: أنت مصحح نصوص عربية. صحح أخطاء التعرف الضوئي في النص التالي.
User: [OCR text]
```

### Process
1. Load OCR predictions
2. Send each text to LLM with simple correction prompt
3. Collect corrected outputs
4. Calculate post-correction CER/WER
5. Compare with Phase 1 (OCR baseline)

### Analysis
- **Improvement Rate**: How much did CER/WER decrease from OCR?
- **Error Changes**: Which errors were fixed? Which new errors introduced?
- **Per-Category**: Which error types does vanilla LLM handle well?

### Outputs
```
results/phase2/
├── corrected/                 # Corrected text files
│   ├── PATS-A01/
│   └── KHATT/
├── metrics.json               # Post-correction CER/WER
├── comparison.json            # vs Phase 1 baseline
├── error_changes.json         # Fixed vs introduced errors
└── report.md
```

### Research Value
- Shows LLM's inherent Arabic correction capability
- **BASELINE for all knowledge-enhanced experiments (Phases 3-5)**

---

## Phase 3: OCR-Aware Prompting (Confusion Matrix Injection)

> **Comparison**: Phase 3 vs Phase 2 (isolated effect of confusion matrix + LLM failure context)

### Research Question
**Does telling the LLM about Qaari's specific error patterns improve correction?**

### Enhancement (v1.0)
Phase 3 is enhanced with training cross-reference:
- `ConfusionMatrixLoader.filter_by_llm_failures()` filters confusion pairs to those where Phase 2 also failed
- `format_word_examples_for_prompt()` adds concrete word-level failure examples
- Config: `phase3.use_training_failures`, `phase3.training_failures_path`

### Prompt Design
```
System: أنت مصحح نصوص عربية متخصص في تصحيح مخرجات نظام Qaari للتعرف الضوئي.

أخطاء Qaari الشائعة (مُصفّاة بحالات فشل النموذج):
- يخلط بين ب و ت و ث (نقط)
- يخلط بين ة و ه
[... top N confusions filtered by LLM failures ...]

أمثلة كلمات لم يُصحَّح فيها بشكل صحيح:
[... word-level failure examples from training ...]

صحح النص التالي مع الانتباه لهذه الأخطاء:

User: [OCR text]
```

### Outputs
```
results/phase3/
├── corrected/
├── metrics.json
├── comparison_vs_phase2.json  # ISOLATED comparison to Phase 2
└── report.md
```

### Research Value
- **Measures isolated effect of OCR-specific knowledge cross-referenced with LLM failures**
- Answers: Does knowing where both Qaari AND the LLM fail help correction?

---

## Phase 4: Self-Reflective Prompting

> **Comparison**: Phase 4 vs Phase 2 (isolated effect of self-reflective error context)

### Research Question
**Do error patterns and overcorrection warnings from training data help the LLM avoid repeating mistakes?**

### Design
Phase 4 reads pre-computed training artifacts — no circular re-analysis:
- Reads `results/phase2-training/analysis/word_pairs_llm_failures.txt` (UNFIXED + INTRODUCED sections)
- Reads `results/phase2-training/analysis/sample_classification.json`
- Builds `overcorrection_context` from INTRODUCED word pairs (words the LLM wrongly changed)
- Config: `phase4.training_artifacts_dir`, `phase4.overcorrection_n`

### Prompt Design
```
System: أنت مصحح نصوص عربية.

أخطاء شائعة لم يُصلَّح فيها في التدريب (UNFIXED):
[... word pairs the LLM failed to fix ...]

تحذير: تجنب هذه التصحيحات الخاطئة (INTRODUCED - كلمات غيّرت بشكل خاطئ):
[... word pairs the LLM wrongly introduced ...]

صحح النص التالي:

User: [OCR text]
```

### Outputs
```
results/phase4/
├── corrected/
├── metrics.json
├── comparison_vs_phase2.json  # ISOLATED comparison to Phase 2
└── report.md
```

### Research Value
- **Tests self-reflective prompting**: Can a model improve by knowing its own failure patterns?
- Novel application: using training-phase error statistics to guide inference
- Pipeline: `pipelines/run_phase4.py`

---

## Phase 5: CAMeL Morphological Validation

> **Comparison**: Phase 5 vs Phase 2 (isolated effect of morphological post-processing)

### Research Question
**Does morphological validation with known-overcorrection revert improve OCR correction?**

### Enhancement (v1.0)
Phase 5 is enhanced beyond simple morphological validation:
- `WordValidator.validate_correction()` accepts `known_overcorrections` set
- Reverts known bad LLM corrections BEFORE morphological checks
- This prevents CAMeL from accepting incorrect but morphologically valid words
- Fixed position-indexed revert bug (was keyed by word, not position)

### Process
1. Run Phase 2 LLM inference (no new inference needed)
2. Load known overcorrections from training artifacts
3. For each corrected text: revert known-bad corrections, then apply CAMeL validation
4. Words failing morphological validation revert to OCR original

### Outputs
```
results/phase5/
├── corrected/
├── metrics.json
├── comparison_vs_phase2.json  # ISOLATED comparison to Phase 2
└── report.md
```

### Research Value
- **Measures isolated effect of CAMeL morphological post-processing**
- Answers: Does symbolic validation catch LLM hallucinations?
- The known-overcorrection revert prevents false positives from morphologically-valid-but-wrong corrections

---

## Phase 6: Combinations

### Research Questions
1. **What is the optimal combination of knowledge sources?**
2. **Do combinations synergize?** (interaction effects)

### Design: 3 Inference Combos + 1 CAMeL Combo

| Combo | Components | Tests |
|-------|------------|-------|
| `conf_only` | Confusion matrix only | OCR-specific knowledge alone |
| `self_only` | Self-reflective only | Error pattern awareness alone |
| `conf_self` | Confusion + Self-reflective | Both prompt enhancements together |
| `best_camel` | Best inference combo + CAMeL post-processing | Optimal prompt + morphological validation |

**Selection strategy**: `best_camel` applies CAMeL on whichever of the 3 inference combos performed best.

### Analysis
- All combos vs Phase 2 (primary comparison)
- Synergy analysis: does `conf_self` outperform `conf_only` + `self_only` improvements summed?
- CAMeL contribution: `best_camel` vs its base inference combo

### Outputs
```
results/phase6/
├── conf_only/
│   ├── corrected/
│   └── metrics.json
├── self_only/
│   ├── corrected/
│   └── metrics.json
├── conf_self/
│   ├── corrected/
│   └── metrics.json
├── best_camel/
│   ├── corrected/
│   └── metrics.json
├── combinations_summary.json
├── statistical_tests.json
└── report.md
```

### Research Value
- **Combination insights**: Which components work well together?
- **Practical guidance**: Minimal effective combination for deployment
- **Publication-ready**: Final comparison numbers for paper

---

---

## 3. System Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SYSTEM COMPONENTS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DATA LAYER                                                      │
│  ───────────                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ DataLoader   │ │ KnowledgeBase│ │ TextUtils    │             │
│  │ - load OCR   │ │ - confusion  │ │ - normalize  │             │
│  │ - load GT    │ │ - word pairs │ │ - clean      │             │
│  │ - align      │ │ - artifacts  │ │ - tokenize   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  LINGUISTIC LAYER (CAMeL Tools)                                  │
│  ──────────────────────────────                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Morphology   │ │ Disambig     │ │ Validator    │             │
│  │ - analyze    │ │ - context    │ │ - is_valid   │             │
│  │ - features   │ │ - disambig   │ │ - revert     │             │
│  │ - lemma/root │ │ - tag        │ │ - validate   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  CORE ENGINE                                                     │
│  ───────────                                                     │
│  ┌──────────────┐ ┌──────────────┐                               │
│  │ LLMCorrector │ │ PromptBuilder│                               │
│  │ - inference  │ │ - zero-shot  │                               │
│  │ - batch      │ │ - ocr_aware  │                               │
│  │ - retry      │ │ - combined   │                               │
│  └──────────────┘ └──────────────┘                               │
│                                                                  │
│  ANALYSIS LAYER                                                  │
│  ──────────────                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Metrics      │ │ ErrorAnalyzer│ │ Visualizer   │             │
│  │ - CER/WER    │ │ - confusion  │ │ - charts     │             │
│  │ - compare    │ │ - categorize │ │ - tables     │             │
│  │ - aggregate  │ │ - diff       │ │ - export     │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  PIPELINE LAYER                                                  │
│  ──────────────                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ run_phase1  run_phase2  run_phase3  run_phase4  ...        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Specifications

#### Data Layer (`src/data/`)

| Module | Classes/Functions | Responsibility |
|--------|-------------------|----------------|
| `data_loader.py` | `DataLoader` | Load and align OCR/GT pairs |
| `knowledge_base.py` | `ConfusionMatrixLoader`, `LLMInsightsLoader`, `WordErrorPairsLoader` | Load knowledge sources |
| `text_utils.py` | `normalize_arabic()`, `clean_text()` | Text preprocessing |

#### Linguistic Layer (`src/linguistic/`) — CAMeL Tools Wrapper

| Module | Classes/Functions | Responsibility |
|--------|-------------------|----------------|
| `morphology.py` | `MorphAnalyzer` | Wrap CAMeL morphological analyzer |
| `validator.py` | `WordValidator`, `validate_correction()` | Check if words are valid Arabic |
| `features.py` | `extract_features()`, `get_word_type()` | Extract linguistic features |

**MorphAnalyzer** wraps `camel_tools.morphology.analyzer` with caching and batch processing.

**WordValidator** provides:
- `is_valid_word(word: str) -> bool` - Check morphological validity
- `validate_text(text: str) -> list[ValidationResult]` - Validate all words
- `score_correction(original: str, corrected: str) -> float` - Score quality

#### Core Engine (`src/core/`)

| Module | Classes/Functions | Responsibility |
|--------|-------------------|----------------|
| `llm_corrector.py` | `LLMCorrector`, `TransformersCorrector` | LLM inference wrapper |
| `prompt_builder.py` | `PromptBuilder` | Construct phase-specific prompts (zero_shot, ocr_aware, self_reflective, combined) |

#### Analysis Layer (`src/analysis/`)

| Module | Classes/Functions | Responsibility |
|--------|-------------------|----------------|
| `metrics.py` | `calculate_cer()`, `calculate_wer()` | Metric calculation |
| `error_analyzer.py` | `ErrorAnalyzer`, `ErrorType` | Build confusion matrix, categorize |
| `llm_error_analyzer.py` | `LLMErrorAnalyzer` | Analyse LLM vs GT per ErrorType (Phase 4D / training analysis) |
| `stats_tester.py` | `StatsTester` | Statistical significance tests |
| `visualizer.py` | `Visualizer` | Generate charts and tables |

---

## 4. Data Flow

### 4.1 Directory Structure

```
Arabic-Post-OCR-Correction/
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── knowledge_base.py
│   │   └── text_utils.py
│   ├── linguistic/              # CAMeL Tools wrappers
│   │   ├── morphology.py        # MorphAnalyzer
│   │   ├── validator.py         # WordValidator
│   │   └── features.py          # Feature extraction
│   ├── core/
│   │   ├── llm_corrector.py
│   │   ├── prompt_builder.py
│   │   └── rag_retriever.py
│   └── analysis/
│       ├── metrics.py
│       ├── error_analyzer.py
│       ├── stats_tester.py
│       └── visualizer.py
├── pipelines/
│   ├── run_phase1.py
│   ├── run_phase2.py
│   ├── run_phase3.py        # Phase 3: OCR-Aware
│   ├── run_phase4.py        # Phase 4: Self-Reflective
│   ├── run_phase4d.py       # Training artifact generator
│   ├── run_phase5.py        # Phase 5 (CAMeL) + Phase 6 (Combinations --combo flag)
│   ├── run_phase7.py        # Phase 7: DSPy Optimization
│   └── run_all.py
├── configs/
│   └── config.yaml
├── results/
│   ├── phase1/
│   ├── phase2/
│   ├── phase2-training/     # Training-split artifacts
│   │   └── analysis/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   └── phase7/
├── docs/
├── tests/
├── scripts/
│   └── hf_download_dataset.py
└── data/                           # All data (consolidated)
    ├── ocr-results/                # OCR predictions, one sub-folder per model
    │   └── qaari-results/          # Active model (change via config.data.ocr_model)
    ├── ocr-raw-data/               # Original ground-truth texts
    │   ├── PATS_A01_Dataset/
    │   └── KHATT/
    └── pats_splits.json            # PATS train/val split file
```

### 4.2 Data & Comparison Dependencies

```
Phase 1 ──────────────────────────────────────────────────────────┐
    │                                                             │
    │ confusion_matrix.json, error_taxonomy.json                  │
    ▼                                                             │
Phase 2 training-split ──────────────────────────────────────────►│
    │                                                             │
    │ results/phase2-training/analysis/                           │
    │   word_pairs_llm_failures.txt                               │
    │   sample_classification.json                                │
    ▼                                                             │
Phase 2 (BASELINE) ════════════════════════════════════════════╗  │
    ║                                                          ║  │
    ║  Isolated experiments: compare each to Phase 2          ║  │
    ║                                                          ║  │
    ╠══════════╦══════════╦══════════╣                         ║  │
    ▼          ▼          ▼          ║                         ║  │
  Ph3        Ph4        Ph5         ║                         ║  │
(+Conf      (+Self-    (+CAMeL      ║                         ║  │
 +LLM fails) reflective) +known-bad) ║                         ║  │
    │          │          │          ║                         ║  │
  P1 CM +    training   training     ║                         ║  │
  training   artifacts  artifacts    ║                         ║  │
    │          │          │          ║                         ║  │
    └──────────┴──────────┘          ║                         ║  │
                    │                ║                         ║  │
                    ▼                ▼                         ║  │
              Phase 6 (Combinations) ◄════════════════════════╝  │
              conf_only | self_only | conf_self | best_camel      │
                                                                  │
              Phase 7 (DSPy Optimization) ◄═══════════════════════╝
```

**Key**: `═══` indicates comparison dependency (not data dependency)

---

## 5. Configuration

### 5.1 Main Configuration (`configs/config.yaml`)

```yaml
# Data paths
data:
  ocr_root: "./data/ocr-results"   # root for all OCR model outputs
  ocr_model: "qaari-results"       # active model sub-folder
  ground_truth: "./data/ocr-raw-data"
  pats_splits_file: "./data/ocr-raw-data/PATS_A01_Dataset/pats_splits.json"

# Datasets (list; one entry per font x split)
datasets:
  - key: "PATS-A01-Akhbar-train"
    type: "pats"
    font: "Akhbar"
    pats_split: "train"
  - key: "KHATT-train"
    type: "khatt"
    split: "train"
  # ... (see config.yaml for full list)

# Model settings
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"
  backend: "transformers"          # or "mock" for local testing
  temperature: 0.1
  max_new_tokens: 1024
  device: "auto"

# Phase-specific settings
phase3:
  use_training_failures: true
  training_failures_path: "results/phase2-training/analysis/word_pairs_llm_failures.txt"
  top_n_confusions: 10

phase4:
  training_artifacts_dir: "results/phase2-training/analysis/"
  overcorrection_n: 10

# CAMeL Tools settings
camel:
  enabled: true
  morphology:
    db: "calima-msa-r13"     # Morphological database
    cache_size: 10000        # Cache analyzed words

# Evaluation
evaluation:
  strip_diacritics: true
  report_both: true            # report both with and without diacritics

# Processing
processing:
  limit_per_dataset: null  # Set for testing
  batch_size: 1
```

---

## 6. Expected Results

### 6.1 Isolated Phase Results Table

| Phase | Method | PATS-A01 CER | PATS-A01 WER | KHATT CER | KHATT WER |
|-------|--------|--------------|--------------|-----------|-----------|
| 1 | Baseline (Qaari) | X.XX% | X.XX% | X.XX% | X.XX% |
| 2 | Zero-shot LLM | X.XX% | X.XX% | X.XX% | X.XX% |
| 3 | + OCR-Aware (confusion + LLM failures) | X.XX% | X.XX% | X.XX% | X.XX% |
| 4 | + Self-Reflective (error patterns + overcorrection) | X.XX% | X.XX% | X.XX% | X.XX% |
| 5 | + CAMeL Validation (+ known-overcorrection revert) | X.XX% | X.XX% | X.XX% | X.XX% |
| 7 | DSPy Optimized | X.XX% | X.XX% | X.XX% | X.XX% |

### 6.2 Combination Results Table (Phase 6)

| Combo | Components | PATS-A01 CER | KHATT CER | vs Phase 2 |
|-------|------------|--------------|-----------|------------|
| conf_only | Confusion only | X.XX% | X.XX% | ΔX.XX% |
| self_only | Self-reflective only | X.XX% | X.XX% | ΔX.XX% |
| conf_self | Confusion + Self-reflective | X.XX% | X.XX% | ΔX.XX% |
| best_camel | Best combo + CAMeL | X.XX% | X.XX% | ΔX.XX% |

---

## 7. Implementation Roadmap

### Phase Implementation Order

| Order | Phase | Dependencies | Complexity |
|-------|-------|--------------|------------|
| 1 | Phase 1 | None | Medium |
| 2 | Phase 2 (training splits) | Phase 1 (for comparison) | Low |
| 3 | Phase 2 (validation splits) | Phase 1 | Low |
| 4 | Phase 4D training analysis | Phase 2 train corrections | Medium |
| 5 | Phase 3 | Phase 1 confusion matrix + Phase 4D artifacts | Low |
| 6 | Phase 4 | Phase 4D artifacts (training artifacts) | Medium |
| 7 | Phase 5 | CAMeL Tools + Phase 4D artifacts | Medium |
| 8 | Phase 6 | Phases 3, 4, 5 | Medium |
| 9 | Phase 7 | Phase 2 | High |

### Shared Components to Build First

1. `DataLoader` - needed by all phases
2. `Metrics` - needed by all phases
3. `LLMCorrector` / `TransformersCorrector` - needed by phases 2–7
4. `TextUtils` - needed by all phases
5. `MorphAnalyzer` - CAMeL wrapper (needed for Phase 1, 5, 6)

---

## Appendix A: Knowledge Base Formats

### A.1 Confusion Matrix (Phase 1 Output)

```json
{
  "metadata": {
    "dataset": "PATS-A01",
    "total_errors": 5000,
    "unique_confusions": 45
  },
  "confusions": {
    "ب": {
      "ت": {"count": 245, "probability": 0.32},
      "ث": {"count": 89, "probability": 0.12},
      "ن": {"count": 67, "probability": 0.09}
    },
    "ة": {
      "ه": {"count": 312, "probability": 0.85}
    }
  }
}
```

### A.2 Training Artifacts (`word_pairs_llm_failures.txt`)

```
SECTION: UNFIXED
OCR: كتابه
GT: كتابة

SECTION: INTRODUCED
OCR: ذلك
LLM: ذالك
GT: ذلك
```

---

## Appendix B: Research Paper Mapping

| Paper Section | Data Source | Phase |
|---------------|-------------|-------|
| Introduction (problem) | baseline_metrics.json | Phase 1 |
| Related Work | - | - |
| Methodology | Architecture.md | - |
| Baseline Results | baseline_metrics.json | Phase 1 |
| Zero-shot Results | metrics.json | Phase 2 |
| Knowledge-Enhanced Results | metrics.json | Phase 3, 4, 5 |
| Combination Results | metrics.json | Phase 6 |
| DSPy Results | metrics.json | Phase 7 |
| Error Analysis | error_taxonomy.json | Phase 1 |
