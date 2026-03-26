# Arabic Post-OCR Correction: Software Architecture

## 1. Research Overview

### 1.1 Research Question

**Can Large Language Models effectively correct OCR errors in Arabic text, bridging the performance gap between open-source and closed-source Vision-Language Models?**

### 1.2 Sub-Questions (Tested by Each Phase)

| Phase | Research Sub-Question | Comparison |
|-------|----------------------|------------|
| Phase 1 | What errors does Qaari make? How severe? | N/A (No LLM) |
| Phase 2 | Can vanilla LLM correct Arabic OCR errors? | vs Phase 1 (OCR baseline) |
| Phase 3 | Does OCR-specific error knowledge help? | vs Phase 2 (isolated) |
| Phase 4A | Do explicit linguistic rules help? | vs Phase 2 (isolated) |
| Phase 4B | Do correction examples help? | vs Phase 2 (isolated) |
| Phase 4C | Does morphological validation help? | vs Phase 2 (isolated) |
| Phase 4D | Do the LLM's own error patterns help (self-reflection)? | vs Phase 2 (isolated) |
| Phase 5 | Does corpus grounding help? | vs Phase 2 (isolated) |
| Phase 5 | What combination is optimal? What contributes? | Combinations + ablation |

**Key Design Principle**: Phases 3-5 (including 4A–4D) are **isolated experiments** comparing to Phase 2 baseline. This measures each knowledge type's independent contribution before combining them in Phase 5.

### 1.3 Datasets

| Dataset | Type | Characteristics | Research Value |
|---------|------|-----------------|----------------|
| **PATS-A01** | Typewritten/Synthetic | Various fonts, controlled | Clean baseline, font variation effects |
| **KHATT** | Handwritten/Real | Natural variation | Real-world performance |

### 1.4 Knowledge Sources

| Source | Contents | How We Use It |
|--------|----------|---------------|
| **Confusion Matrix** | Qaari's character errors | Tell LLM what to watch for |
| **QALB Corpus** | Human error→correction pairs | Few-shot examples |
| **Arabic Rules** | Orthographic rules (hamza, taa marbuta, etc.) | Inject into prompts |
| **OpenITI Corpus** | Large Arabic text corpus | RAG retrieval + vocabulary |
| **CAMeL Tools** | Morphological analyzer, disambiguator | Validate corrections, error categorization |

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
2. **Phase 4A**: Morphology-aware rule application
3. **Phase 5**: Post-LLM validation layer (optional)

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
│  │   PHASE 2   │  Zero-Shot LLM ═══════════════════════════════════════════╗         │
│  │  BASELINE   │  → BASELINE FOR ALL COMPARISONS                           ║         │
│  └──────┬──────┘                                                           ║         │
│         │                                                                  ║         │
│         ├────────────────┬────────────────┬────────────────┬───────────────╫───┐     │
│         │                │                │                │               ║   │     │
│         ▼                ▼                ▼                ▼               ▼   ▼     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ PHASE 3  │  │ PHASE 4A │  │ PHASE 4B │  │ PHASE 4C │  │ PHASE 4D │  │ PHASE 5  │  │
│  │Confusion │  │  Rules   │  │ Few-Shot │  │  CAMeL   │  │  Self-   │  │   RAG    │  │
│  │ Matrix   │  │(Symbolic)│  │  (QALB)  │  │ (Morph.) │  │Reflective│  │(OpenITI) │  │
│  │vs Ph2 ▲  │  │vs Ph2 ▲  │  │vs Ph2 ▲  │  │vs Ph2 ▲  │  │vs Ph2 ▲  │  │vs Ph2 ▲  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘         │
│                                          │                                            │
│                                          ▼                                            │
│                               ┌─────────────────────┐                                │
│                               │      PHASE 6        │                                │
│                               │  Combinations +     │                                │
│                               │  Ablation Study     │                                │
│                               └─────────────────────┘                                │
│                                                                                       │
│  Legend: ▲ = Isolated comparison to Phase 2 baseline                                 │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Phase 2 is the hub**: All knowledge-enhanced phases (3-5, including 4A–4D) compare to Phase 2
2. **Isolated experiments**: Each phase tests ONE knowledge addition
3. **Incremental complexity**: Simple injection (3,4A) → Examples (4B) → Post-processing (4C) → Self-reflection (4D) → Retrieval (5)
4. **Final synthesis**: Phase 5 combines all and measures individual contributions

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
> Phases 3, 4A, 4B, and 5 are isolated experiments comparing to Phase 2.

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

> **Comparison**: Phase 3 vs Phase 2 (isolated effect of confusion matrix)

### Research Question
**Does telling the LLM about Qaari's specific error patterns improve correction?**

### Hypothesis
If the LLM knows "Qaari often mistakes ب for ت", it can pay special attention to these cases.

### Prompt Design
```
System: أنت مصحح نصوص عربية متخصص في تصحيح مخرجات نظام Qaari للتعرف الضوئي.

أخطاء Qaari الشائعة:
- يخلط بين ب و ت و ث (نقط)
- يخلط بين ة و ه
- يخلط بين أ و ا و إ
[... top N confusions from Phase 1 ...]

صحح النص التالي مع الانتباه لهذه الأخطاء:

User: [OCR text]
```

### Variables to Test
- Number of confusions to include (5, 10, 20)
- Format of confusion information (list, examples, statistics)

### Outputs
```
results/phase3/
├── corrected/
├── metrics.json
├── comparison_vs_phase2.json  # ISOLATED comparison to Phase 2
├── confusion_impact.json      # Which confusions were addressed
└── report.md
```

### Research Value
- **Measures isolated effect of OCR-specific knowledge**
- Answers: Does knowing Qaari's error patterns help the LLM?
- Comparison: Phase 3 CER vs Phase 2 CER

---

## Phase 4: Linguistic Knowledge Enhancement

> **Comparisons**:
> - Phase 4A vs Phase 2 (isolated effect of rules)
> - Phase 4B vs Phase 2 (isolated effect of examples)
> - Phase 4A vs Phase 4B (symbolic vs data-driven approaches)

### Research Question
**Does linguistic knowledge improve LLM correction? Which type works better?**

### Sub-Experiments

#### Phase 4A: Arabic Orthographic Rules (Symbolic Approach)

Inject explicit Arabic spelling rules into the prompt.

> **Comparison**: Phase 4A vs Phase 2 — Does telling the LLM rules help?

**Rules to Include:**
1. همزة القطع vs همزة الوصل (Hamza types)
2. التاء المربوطة vs الهاء (Taa Marbuta vs Ha)
3. الألف المقصورة vs الياء (Alef Maksura vs Ya)
4. ال الشمسية والقمرية (Sun/Moon letters)
5. التنوين (Tanwin rules)

**Prompt Design:**
```
System: أنت مصحح نصوص عربية. راعِ القواعد الإملائية التالية:

1. همزة القطع تُكتب في أول الأفعال الرباعية: أَكْرَمَ، أَحْسَنَ
2. همزة الوصل تُكتب في أول الأفعال الخماسية والسداسية: استَغْفَرَ، انْطَلَقَ
3. التاء المربوطة (ة) تُنطق هاءً عند الوقف: مدرسة، جامعة
[... more rules ...]

صحح النص التالي:

User: [OCR text]
```

#### Phase 4B: QALB Few-Shot Examples (Data-Driven Approach)

Use real error-correction pairs from QALB corpus as few-shot examples.

> **Comparison**: Phase 4B vs Phase 2 — Do correction examples help?

**Process:**
1. Extract error-correction pairs from QALB
2. Categorize by error type
3. Select diverse, representative examples
4. Include in prompt as demonstrations

**Prompt Design:**
```
System: أنت مصحح نصوص عربية. إليك أمثلة على التصحيح:

خطأ: انا ذاهب الى المدرسه
صحيح: أنا ذاهب إلى المدرسة

خطأ: هذة الكتاب جميل
صحيح: هذا الكتاب جميل

[... more examples ...]

صحح النص التالي بنفس الطريقة:

User: [OCR text]
```

**Variables to Test:**
- Number of examples (1, 3, 5, 10)
- Selection strategy (random vs error-type-matched)

### Outputs
```
results/phase4/
├── phase4a_rules/
│   ├── corrected/
│   ├── metrics.json
│   ├── comparison_vs_phase2.json  # ISOLATED comparison
│   └── report.md
├── phase4b_fewshot/
│   ├── corrected/
│   ├── metrics.json
│   ├── comparison_vs_phase2.json  # ISOLATED comparison
│   ├── example_impact.json        # Which examples helped
│   └── report.md
├── comparison_4a_vs_4b.json       # Rules vs Examples
└── report.md
```

### Research Value
- **Measures isolated effect of linguistic knowledge**
- **Compares symbolic (rules) vs data-driven (examples) approaches**
- Answers: Which type of linguistic knowledge helps more?
- Note: QALB has human typing errors, not OCR errors — interesting to see if it transfers

#### Phase 4C: CAMeL Morphological Validation (Post-Processing Approach)

Apply morphological validation as a post-processing step after zero-shot LLM correction.

> **Comparison**: Phase 4C vs Phase 2 — Does morphological validation alone improve results?

**Rationale:**
Unlike Phases 3, 4A, 4B (prompt modifications), CAMeL validation is applied **after** the LLM generates output. This tests whether symbolic validation can catch and correct LLM errors.

**Process:**
1. Run zero-shot LLM correction (same as Phase 2)
2. For each word in LLM output:
   - Check if morphologically valid using CAMeL analyzer
   - If invalid, attempt correction:
     - Option A: Flag for human review
     - Option B: Use CAMeL suggestions
     - Option C: Revert to OCR original (if it was valid)
3. Produce validated output

**Validation Strategies:**
| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| Flag only | Mark invalid words | Conservative, no new errors |
| Suggest | Use CAMeL's suggestions | May improve, may introduce errors |
| Revert | Use original if LLM broke it | Prevents LLM hallucination |

**Outputs:**
```
results/phase4/
├── phase4c_camel/
│   ├── corrected/
│   ├── metrics.json
│   ├── comparison_vs_phase2.json  # ISOLATED comparison
│   ├── validation_stats.json      # % words validated, rejected, etc.
│   └── report.md
```

**Research Value:**
- **Tests morphological validation in isolation** (consistent with Phases 3-5 design)
- Answers: Can symbolic post-processing improve neural correction?
- Compares: Prompt-based knowledge (3, 4A, 4B) vs post-processing (4C)
- Provides baseline for CAMeL's contribution before Phase 5 combines everything

#### Phase 4D: Self-Reflective Prompting

Inject the LLM's own known failure patterns (derived from training-split predictions) into the validation prompt.

> **Comparison**: Phase 4D vs Phase 2 — Does self-knowledge of failure patterns improve correction?

**Rationale:**
Unlike Phases 3, 4A, 4B (external knowledge), Phase 4D feeds the model information about its *own* systematic errors on training data. This is a form of meta-learning without fine-tuning.

**Process:**
1. Run Phase 2 (or any source phase) on **training splits** to get corrected texts where GT is available
2. Run `LLMErrorAnalyzer` to compare LLM outputs vs GT, computing per-ErrorType fix_rate and introduction_rate
3. Aggregate by dataset type (PATS-A01, KHATT) to get statistically robust insight
4. Format weaknesses (low fix_rate) and over-corrections (high introduction_rate) as Arabic text
5. Inject as self-reflective context into validation-split prompts

**Prompt Design:**
```
System: أنت مصحح نصوص عربية متخصص. بناءً على تحليل أخطائك السابقة في تصحيح نصوص عربية
مشابهة، إليك ملاحظات مهمة لتحسين أدائك:

{insights_context}

صحح النص التالي مع مراعاة هذه الملاحظات.
أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي.

User: [OCR text]
```

**Outputs:**
```
results/phase4d/
├── insights/
│   ├── PATS-A01_insights.json   # Per-ErrorType fix/intro rates (pooled across fonts)
│   └── KHATT_insights.json      # KHATT-specific insights
├── inference_input.jsonl        # Val-split export with self-reflective prompts
├── corrections.jsonl            # Inference output (place here before analyze)
└── {dataset_name}/
    ├── metrics.json
    ├── comparison_vs_phase2.json
    └── error_changes.json
```

**Research Value:**
- **Tests self-reflective prompting in isolation** (consistent with Phase 4A/4B/4C design)
- Novel application: using the model's own error statistics to improve future corrections
- Answers: Can a model improve by knowing where it previously failed?
- Implemented in `pipelines/run_phase4d.py` with prompt version `p4dv1`

---

## Phase 5: Retrieval-Augmented Generation (RAG)

> **Comparison**: Phase 5 vs Phase 2 (isolated effect of corpus grounding)

### Research Question
**Does grounding in a large Arabic corpus improve OCR correction?**

### Hypothesis
If the LLM sees similar correct sentences from OpenITI, it has better context for correction.

### Process

#### 5.1 Build Retrieval Index
1. Process OpenITI corpus
2. Split into sentences/chunks
3. Create embeddings (using Arabic embedding model)
4. Build vector index (FAISS or similar)

#### 5.2 Retrieval-Augmented Correction
1. For each OCR text, retrieve top-K similar sentences from OpenITI
2. Include retrieved sentences in prompt as context
3. LLM corrects with awareness of similar correct text

### Prompt Design
```
System: أنت مصحح نصوص عربية.

نصوص مشابهة صحيحة من المكتبة العربية:
1. [Retrieved sentence 1]
2. [Retrieved sentence 2]
3. [Retrieved sentence 3]

استخدم هذه النصوص كمرجع لتصحيح النص التالي:

User: [OCR text]
```

### Alternative: Vocabulary Validation
Instead of full RAG, simpler approach:
1. Build vocabulary from OpenITI
2. After LLM correction, validate words against vocabulary
3. Flag or re-correct words not in vocabulary

### Outputs
```
results/phase5/
├── retrieval_index/           # Built index (for reuse)
├── corrected/
├── metrics.json
├── comparison_vs_phase2.json  # ISOLATED comparison
├── retrieval_analysis.json    # How often retrieval helped
└── report.md
```

### Research Value
- **Measures isolated effect of corpus grounding**
- Novel application of RAG to OCR correction
- Answers: Does seeing correct Arabic text help correction?
- Most technically complex phase (embeddings, vector index)

---

## Phase 5: Combinations & Ablation Study

### Research Questions
1. **What is the optimal combination of knowledge sources?**
2. **Which components synergize?** (interaction effects)
3. **What does each component contribute to the full system?** (ablation)

### Purpose
Phase 5 goes beyond simple ablation to test meaningful combinations, answering both "what works best together?" and "what does each part contribute?"

### Experimental Design Rationale

**Why not test all 32 combinations?**
- 5 components → 2^5 = 32 combinations
- Computationally expensive
- Many combinations are uninteresting

**Hierarchical approach (what we do):**

| Level | What We Test | Research Question |
|-------|--------------|-------------------|
| Level 1 | Isolated (Phases 3-5) | Individual effects ✓ |
| Level 2 | Top pairs | Which components synergize? |
| Level 3 | Best triple (optional) | Diminishing returns? |
| Level 4 | Full system | Maximum performance |
| Level 5 | Ablation (Full - 1) | Component necessity |

### 6.1 Component Categories

Components fall into two categories:

| Category | Components | How Applied |
|----------|------------|-------------|
| **Prompt-based** | Confusion, Rules, Few-shot, RAG | Modify LLM input |
| **Post-processing** | CAMeL Validation | Modify LLM output |

This distinction matters: prompt-based components may compete for context window, while CAMeL is additive.

### 6.2 Combination Experiments

#### Pair Combinations (select based on Phase 3-5 results)

Test pairs of top-performing isolated components:

| Experiment | Components | Tests |
|------------|------------|-------|
| Pair A | Confusion + Rules | OCR-specific + linguistic |
| Pair B | Confusion + Few-shot | OCR-specific + examples |
| Pair C | Confusion + RAG | OCR-specific + grounding |
| Pair D | Rules + Few-shot | Symbolic + data-driven |
| Pair E | Best prompt + CAMeL | Prompt + post-processing |

**Selection Strategy:** After running Phases 3-5, select top 3-5 pairs based on:
1. Top 2 isolated performers paired together
2. Complementary approaches (e.g., symbolic + data-driven)
3. Prompt-based + post-processing combinations

#### Full System

```
┌─────────────────────────────────────────────────────┐
│                 FULL PIPELINE                        │
│                                                      │
│  OCR Text                                           │
│      │                                              │
│      ▼                                              │
│  ┌─────────────┐                                    │
│  │  Retrieve   │ ← OpenITI (Phase 5)               │
│  │  Similar    │                                    │
│  └──────┬──────┘                                    │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────────────────────────────┐       │
│  │           LLM Prompt                     │       │
│  │  • Confusion matrix (Phase 3)           │       │
│  │  • Rules (Phase 4A)                     │       │
│  │  • Few-shot examples (Phase 4B)         │       │
│  │  • Retrieved context (Phase 5)          │       │
│  └──────────────────┬──────────────────────┘       │
│                     │                               │
│                     ▼                               │
│  ┌─────────────────────────────────────────┐       │
│  │     Morphological Validation (Phase 4C) │       │
│  │  • Validate corrected words              │       │
│  │  • Flag/correct non-words                │       │
│  └──────────────────┬──────────────────────┘       │
│                     │                               │
│                     ▼                               │
│              Corrected Text                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 6.3 Ablation Studies

Remove one component at a time from the full system:

| Experiment | Components | Measures |
|------------|------------|----------|
| Full | All 5 | Upper bound |
| −Confusion | Rules + QALB + RAG + CAMeL | Confusion matrix necessity |
| −Rules | Confusion + QALB + RAG + CAMeL | Rules necessity |
| −QALB | Confusion + Rules + RAG + CAMeL | Few-shot necessity |
| −RAG | Confusion + Rules + QALB + CAMeL | Retrieval necessity |
| −CAMeL | Confusion + Rules + QALB + RAG | Morphological validation necessity |
| None | Zero-shot (Phase 2) | Lower bound |

**Interpretation:**
- Large Δ from Full = component is essential
- Small Δ from Full = component is redundant (other components compensate)
- Negative Δ = component hurts when combined (interference)

### 6.4 Statistical Analysis

For all comparisons:
- Paired t-tests (same samples)
- 95% confidence intervals
- Cohen's d effect sizes
- Per-dataset breakdown (PATS-A01 vs KHATT)
- Bonferroni correction for multiple comparisons

### 6.5 Key Analyses

| Analysis | Purpose |
|----------|---------|
| Synergy detection | Do pairs outperform sum of individuals? |
| Redundancy analysis | Which components overlap in what they fix? |
| Error-type breakdown | Which combinations fix which error types? |
| Efficiency analysis | Performance vs computational cost |

### Outputs
```
results/phase5/
├── combinations/
│   ├── pair_confusion_rules/
│   ├── pair_confusion_fewshot/
│   ├── pair_confusion_rag/
│   ├── pair_rules_fewshot/
│   ├── pair_best_camel/
│   └── combinations_summary.json
├── full_system/
│   ├── corrected/
│   └── metrics.json
├── ablation/
│   ├── no_confusion/
│   ├── no_rules/
│   ├── no_qalb/
│   ├── no_rag/
│   ├── no_camel/
│   └── ablation_summary.json
├── analysis/
│   ├── synergy_analysis.json
│   ├── redundancy_matrix.json
│   └── error_type_breakdown.json
├── statistical_tests.json
├── final_comparison.json
├── figures/
│   ├── improvement_chart.png
│   ├── combination_heatmap.png
│   ├── ablation_chart.png
│   ├── error_breakdown.png
│   └── dataset_comparison.png
├── paper_tables.md
└── report.md
```

### Research Value
- **Combination insights**: Which components work well together?
- **Ablation insights**: What's necessary vs redundant?
- **Practical guidance**: Minimal effective combination for deployment
- **Publication-ready**: Final numbers, figures, tables

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
│  │ - load GT    │ │ - rules      │ │ - clean      │             │
│  │ - align      │ │ - QALB       │ │ - tokenize   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  LINGUISTIC LAYER (CAMeL Tools)                                  │
│  ──────────────────────────────                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Morphology   │ │ Disambig     │ │ Validator    │             │
│  │ - analyze    │ │ - context    │ │ - is_valid   │             │
│  │ - features   │ │ - disambig   │ │ - suggest    │             │
│  │ - lemma/root │ │ - tag        │ │ - score      │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  CORE ENGINE                                                     │
│  ───────────                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ LLMCorrector │ │ PromptBuilder│ │ RAGRetriever │             │
│  │ - inference  │ │ - zero-shot  │ │ - index      │             │
│  │ - batch      │ │ - few-shot   │ │ - search     │             │
│  │ - retry      │ │ - combined   │ │ - embed      │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
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
| `knowledge_base.py` | `ConfusionMatrixLoader`, `RulesLoader`, `QALBLoader`, `OpenITILoader`, `LLMInsightsLoader` | Load knowledge sources |
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
| `llm_corrector.py` | `LLMCorrector` | LLM inference wrapper |
| `prompt_builder.py` | `PromptBuilder` | Construct phase-specific prompts |
| `rag_retriever.py` | `RAGRetriever` | OpenITI retrieval system |

#### Analysis Layer (`src/analysis/`)

| Module | Classes/Functions | Responsibility |
|--------|-------------------|----------------|
| `metrics.py` | `calculate_cer()`, `calculate_wer()` | Metric calculation |
| `error_analyzer.py` | `ErrorAnalyzer`, `ErrorType` | Build confusion matrix, categorize |
| `llm_error_analyzer.py` | `LLMErrorAnalyzer` | Analyse LLM vs GT per ErrorType (Phase 4D) |
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
│   ├── run_phase3.py
│   ├── run_phase4.py
│   ├── run_phase4d.py
│   ├── run_phase5.py
│   ├── run_phase5.py
│   └── run_all.py
├── configs/
│   └── config.yaml
├── results/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4a/
│   ├── phase4b/
│   ├── phase4c/
│   ├── phase4d/
│   ├── phase5/
│   └── phase5/
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
    ├── OpenITI/                    # Arabic corpus for RAG
    ├── QALB-0.9.1-Dec03-2021-SharedTasks/  # Error-correction pairs
    └── rules/                      # Arabic spelling rules
```

### 4.2 Data & Comparison Dependencies

```
Phase 1 ────────────────────────────────────────────────────────────────────────┐
    │                                                                           │
    │ confusion_matrix.json, error_taxonomy.json                                │
    ▼                                                                           │
Phase 2 (BASELINE) ═════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║  All phases below compare to Phase 2 (isolated experiments)               ║
    ║                                                                           ║
    ╠══════╦══════╦══════╦══════╦══════╦══════╣
    ▼      ▼      ▼      ▼      ▼      ▼      ║
  Ph3    Ph4A   Ph4B   Ph4C   Ph4D   Ph5     ║
(+Conf) (+Rul) (+FS) (+CAMeL)(+Self)(+RAG)  ║
    │      │      │      │      │      │      ║
 P1 CM  rules/ QALB/ CAMeL  P2 train OpenIT ║
    │      │      │      │   insights  │      ║
    └──────┴──────┴──────┴──────┴──────┘      ║
                                        │                                       ║
                                        ▼                                       ║
                                  Phase 5 ◄═════════════════════════════════════╝
                                  (Combinations + Ablation)
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
  openiti: "./data/OpenITI"
  qalb: "./data/QALB-0.9.1-Dec03-2021-SharedTasks/QALB-0.9.1-Dec03-2021-SharedTasks"
  rules: "./data/rules"

# Datasets
datasets:
  - name: "PATS-A01"
    ocr_path: "pats-a01-data/A01-Akhbar"
    gt_path: "PATS_A01_Dataset"
  - name: "KHATT"
    ocr_path: "khatt-data"
    gt_path: "KHATT"

# Model settings
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  temperature: 0.1
  max_tokens: 1024
  device: "auto"

# RAG settings (Phase 5)
rag:
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  top_k: 3
  chunk_size: 200

# Few-shot settings (Phase 4B)
few_shot:
  num_examples: 5
  selection: "diverse"  # or "random"

# CAMeL Tools settings
camel:
  enabled: true
  morphology:
    db: "calima-msa-r13"     # Morphological database
    cache_size: 10000        # Cache analyzed words
  validation:
    enabled: true            # Enable post-LLM validation
    re_correct: false        # Re-send invalid words to LLM
    min_confidence: 0.5      # Minimum validation score

# Output
output:
  results_dir: "results"
  save_corrected: true
  save_comparisons: true

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
| 3 | + Confusion Matrix | X.XX% | X.XX% | X.XX% | X.XX% |
| 4A | + Arabic Rules | X.XX% | X.XX% | X.XX% | X.XX% |
| 4B | + QALB Few-shot | X.XX% | X.XX% | X.XX% | X.XX% |
| 4C | + CAMeL Validation | X.XX% | X.XX% | X.XX% | X.XX% |
| 4D | + Self-Reflective | X.XX% | X.XX% | X.XX% | X.XX% |
| 5 | + RAG (OpenITI) | X.XX% | X.XX% | X.XX% | X.XX% |

### 6.2 Combination Results Table

| Combination | Components | PATS-A01 CER | KHATT CER | vs Best Isolated |
|-------------|------------|--------------|-----------|------------------|
| Pair A | Confusion + Rules | X.XX% | X.XX% | ΔX.XX% |
| Pair B | Confusion + Few-shot | X.XX% | X.XX% | ΔX.XX% |
| Pair C | Confusion + RAG | X.XX% | X.XX% | ΔX.XX% |
| Pair D | Best prompt + CAMeL | X.XX% | X.XX% | ΔX.XX% |
| Full | All components | X.XX% | X.XX% | ΔX.XX% |

### 6.3 Ablation Results Table

| Configuration | PATS-A01 CER | KHATT CER | Δ from Full |
|---------------|--------------|-----------|-------------|
| Full System | X.XX% | X.XX% | - |
| − Confusion | X.XX% | X.XX% | +X.XX% |
| − Rules | X.XX% | X.XX% | +X.XX% |
| − QALB | X.XX% | X.XX% | +X.XX% |
| − RAG | X.XX% | X.XX% | +X.XX% |
| − CAMeL | X.XX% | X.XX% | +X.XX% |

---

## 7. Implementation Roadmap

### Phase Implementation Order

| Order | Phase | Dependencies | Complexity |
|-------|-------|--------------|------------|
| 1 | Phase 1 | None | Medium |
| 2 | Phase 2 | Phase 1 (for comparison) | Low |
| 3 | Phase 3 | Phase 1 (confusion matrix) | Low |
| 4 | Phase 4A | Rules files | Low |
| 5 | Phase 4B | QALB corpus | Medium |
| 6 | Phase 4C | CAMeL Tools, Phase 2 output | Medium |
| 7 | Phase 4D | Phase 2 train-split corrections | Medium |
| 8 | Phase 5 | OpenITI, embedding model | High |
| 9 | Phase 5 | All previous phases | High |

### Shared Components to Build First

1. `DataLoader` - needed by all phases
2. `Metrics` - needed by all phases
3. `LLMCorrector` - needed by phases 2-6
4. `TextUtils` - needed by all phases
5. `MorphAnalyzer` - CAMeL wrapper (needed for Phase 1, 4C, 6)

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

### A.2 QALB Few-Shot Examples

```json
{
  "examples": [
    {
      "source": "انا ذاهب الى المدرسه",
      "target": "أنا ذاهب إلى المدرسة",
      "error_types": ["hamza", "taa_marbuta"],
      "context": "sentence about going to school"
    }
  ]
}
```

### A.3 Arabic Rules

```json
{
  "rules": [
    {
      "name": "همزة القطع",
      "name_en": "Hamza al-Qat",
      "description": "تُكتب في أول الأفعال الرباعية",
      "examples": {
        "correct": ["أَكْرَمَ", "أَحْسَنَ"],
        "incorrect": ["اكرم", "احسن"]
      },
      "pattern": "^[اأإآ]"
    }
  ]
}
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
| Knowledge-Enhanced Results | metrics.json | Phase 3-5 |
| Ablation Study | ablation_summary.json | Phase 5 |
| Error Analysis | error_taxonomy.json | Phase 1, 6 |
| Figures | figures/*.png | Phase 5 |
| Tables | paper_tables.md | Phase 5 |
