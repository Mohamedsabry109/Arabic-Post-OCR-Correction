# Phase 4: Linguistic Knowledge Enhancement — Design Document

## 1. Overview

### 1.1 Purpose

Phase 4 tests three independent ways to inject **Arabic linguistic knowledge** into the correction process. Each sub-phase is a self-contained isolated experiment, identical in design to Phase 3:

| Sub-Phase | Knowledge Source | Mechanism | New Modules |
|-----------|-----------------|-----------|-------------|
| **4A** | Arabic spelling rules (`data/rules/`) | Prompt injection | `RulesLoader` |
| **4B** | QALB error-correction pairs (`data/QALB-*/`) | Few-shot examples | `QALBLoader` |
| **4C** | CAMeL morphological analyzer | Post-processing validation | `WordValidator` |

### 1.2 Research Questions

| Sub-Phase | Research Question |
|-----------|------------------|
| 4A | Does injecting explicit Arabic orthographic rules improve correction? |
| 4B | Do real error-correction examples improve correction? |
| 4C | Does morphological validation of LLM output improve correction? |

### 1.3 Isolated Comparison Design

> **CRITICAL**: Each sub-phase compares **only against Phase 2** (zero-shot baseline).
> No sub-phase compares against Phase 1, Phase 3, or each other.
> This isolates each knowledge type's independent contribution.
>
> Key metric: Δ CER = Phase4x CER − Phase2 CER (negative = improvement)

4A and 4B follow the three-stage pipeline (export → inference → analyze) identical to Phase 3.
4C is different: it is **entirely local** — no new LLM inference needed. It reads Phase 2
`corrections.jsonl` and applies CAMeL validation locally.

### 1.4 Downstream Use

- **Phase 6**: 4A, 4B, 4C results feed into combination experiments
- **Paper**: Rows "4A: +Rules", "4B: +Few-shot", "4C: +CAMeL Validation" in the main results table
- **Ablation (Phase 6)**: Each sub-phase's knowledge source is removed one at a time from the full system

---

## 2. Data Sources

### 2.1 Phase 4A: Arabic Spelling Rules

**Location**: `data/rules/`

```
data/rules/
├── orthography.jsonl       # Full Arabic orthography textbook (by page, UTF-8)
├── orthography.txt         # Same book, plain text
├── orthography_small.txt   # Abbreviated extracted version
├── morphology.jsonl        # Arabic morphology textbook (by page)
├── morphology.txt          # Same book, plain text
└── syntax.jsonl            # Arabic syntax textbook (by page)
```

**Format of `.jsonl` files**: One JSON object per page:
```json
{"book_id": 7818, "page_no": 1, "text": "...Arabic text..."}
```

**Design Decision**: The rules books are full textbooks (hundreds of pages of prose). They are
too verbose to inject directly into prompts. Phase 4A uses a **curated, hardcoded rule set**
(`RulesLoader.CORE_RULES`) focusing specifically on OCR-relevant orthographic errors. The books
are the authoritative source used to write these rules; the loader does not parse the books
at runtime.

**Rationale**: The book content is descriptive pedagogy (teaching methodology, historical
context), not a structured rule list. Extracting machine-usable rules requires human
curation. The curated rules focus on the same error categories as Phase 1's confusion matrix:
taa marbuta / ha, hamza forms, alef maksura / ya, dot-bearing pairs, and similar-shape letters.

### 2.2 Phase 4B: QALB Corpus

**Location**: `data/QALB-0.9.1-Dec03-2021-SharedTasks/QALB-0.9.1-Dec03-2021-SharedTasks/`

```
data/QALB-.../
├── data/
│   ├── 2014/
│   │   ├── train/
│   │   │   ├── QALB-2014-L1-Train.sent     # Original (erroneous) sentences
│   │   │   ├── QALB-2014-L1-Train.cor      # Corrected sentences
│   │   │   └── QALB-2014-L1-Train.m2       # M2 annotation format
│   │   ├── dev/
│   │   └── test/
│   └── 2015/
└── docs/
```

**File formats**:

`.sent` (one sentence per line):
```
aj_0001844_0000004.ar الى التعليق رقم 2 اكيد ان لحكام العرب ...
```
The sentence ID is the first token (ends with `.ar`). The rest of the line is the text.

`.cor` (one corrected sentence per line):
```
S إلى التعليق رقم 2 : أكيد أن للحكام العرب ...
```
The `S` prefix is stripped. Lines align 1:1 with `.sent`.

**Key Limitation**: QALB contains **human typing errors** (Arabic online comments), not OCR errors.
The overlap with OCR error types is partial:

| Error Type | In QALB? | In OCR? | Usable for Phase 4B? |
|------------|----------|---------|----------------------|
| Hamza errors (أ/ا/إ/آ/ء) | Yes | Yes | Yes |
| Taa marbuta / Ha (ة/ه) | Yes | Yes | Yes |
| Alef maksura / Ya (ى/ي) | Yes | Yes | Yes |
| Dot confusion (ب/ت/ث) | Rare | Yes | Limited |
| Merged/split words | Yes | Yes | Yes |
| Grammar/agreement errors | Yes | No | No |
| Punctuation errors | Yes | No | No |

**Design Decision**: Filter QALB pairs to include only those where the changes are
character-level substitutions matching OCR error patterns. Grammar and punctuation
corrections are excluded.

### 2.3 Phase 4C: CAMeL Tools

**Package**: `camel-tools` (installed via pip)
**Database**: `calima-msa-r13` (configured in `config.yaml`)

CAMeL's morphological analyzer accepts an Arabic word and returns its possible morphological
analyses. A word with zero analyses is morphologically invalid — likely a non-word error.

**Phase 2 dependency**: Phase 4C reads `results/phase2/corrections.jsonl` as input — the
LLM-corrected texts from Phase 2. It does not perform new LLM inference.

---

## 3. Phase 4A: Arabic Orthographic Rules

### 3.1 Purpose and Scope

Inject a curated set of Arabic orthographic rules into the LLM system prompt. The LLM already
knows Arabic grammar; the rules are meant to remind it of specific patterns that overlap with
OCR errors — the same patterns that appear in Qaari's confusion matrix.

### 3.2 `RulesLoader` — Module Design

**Location**: `src/data/knowledge_base.py` (replaces the current stub)

```python
from dataclasses import dataclass


@dataclass
class ArabicRule:
    """A single Arabic orthographic rule."""
    name: str           # Short name, e.g., "taa_marbuta"
    category: str       # Category: "hamza" | "taa_marbuta" | "alef_maksura" |
                        #           "dots" | "similar_shapes" | "other"
    description: str    # One-sentence Arabic description of the rule
    correct_examples: list[str]   # Words showing correct usage
    incorrect_examples: list[str] # Common incorrect forms
    prompt_text: str    # Ready-to-inject Arabic text for LLM prompt
```

```python
class RulesLoader:
    """Load and format Arabic orthographic rules for Phase 4A prompt injection.

    The primary source is the hardcoded CORE_RULES list — a curated set of
    OCR-relevant orthographic rules. The rules.jsonl books are the authoritative
    reference used to write CORE_RULES; they are not parsed at runtime.

    Usage::

        loader = RulesLoader()
        rules = loader.load()                        # All CORE_RULES
        rules = loader.load(categories=["hamza"])    # Subset by category
        text = loader.format_for_prompt(rules, n=5)  # Format for prompt injection
    """

    CATEGORIES: list[str] = [
        "hamza",         # همزة الوصل / همزة القطع / همزة على الألف / السطر
        "taa_marbuta",   # التاء المربوطة vs الهاء
        "alef_maksura",  # الألف المقصورة (ى) vs الياء (ي)
        "alef_forms",    # أ / ا / إ / آ / ء
        "dots",          # الأحرف المنقوطة (ب/ت/ث/ن)
        "similar_shapes",# ر/ز، د/ذ، س/ش، ص/ض، ف/ق
    ]

    CORE_RULES: list[ArabicRule] = [...]  # See §3.4

    def load(
        self,
        categories: list[str] | None = None,
    ) -> list[ArabicRule]:
        """Return rules filtered by category.

        Args:
            categories: If None, returns all CORE_RULES.
                Otherwise, returns only rules whose category is in the list.

        Returns:
            List of ArabicRule objects.
        """

    def format_for_prompt(
        self,
        rules: list[ArabicRule],
        n: int | None = None,
        style: str = "compact_arabic",
    ) -> str:
        """Format rules into Arabic text for prompt injection.

        Args:
            rules: List of ArabicRule (from load()).
            n: Maximum number of rules to include. None = all.
            style: Formatting style:
                - "compact_arabic" (default): one line per rule with examples
                - "detailed_arabic": full description + examples

        Returns:
            Multi-line Arabic string. Empty string if rules is empty.
        """
```

### 3.3 Prompt Design

**Version**: `RULES_PROMPT_VERSION = "p4av1"`

```python
RULES_SYSTEM_V1: str = (
    "أنت مصحح نصوص عربية متخصص. "
    "راعِ القواعد الإملائية التالية عند تصحيح النص:\n\n"
    "{rules_context}\n\n"
    "صحح النص التالي مع الانتباه بشكل خاص لهذه القواعد. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)
```

**`PromptBuilder` addition**:
```python
def build_rule_augmented(self, ocr_text: str, rules_context: str) -> list[dict]:
    """Build rule-augmented correction prompt (Phase 4A)."""
```

### 3.4 Core Rules (Curated)

The `CORE_RULES` list contains exactly these rule categories, with one to three rules per category.
Rules are written for **OCR error correction** context — they emphasize what to watch for when
the source text may be an OCR scan, not keyboard input.

**Category: `taa_marbuta`** (highest overlap with OCR errors):
```
- التاء المربوطة (ة) تُكتب في نهاية الأسماء المؤنثة: مدرسة، جامعة، معلمة
  الهاء (ه) تُكتب في نهاية الأفعال وأسماء الإشارة: يكتبه، هذه
  خطأ شائع في الطباعة: "مدرسه" بدلاً من "مدرسة"
```

**Category: `hamza`**:
```
- همزة الوصل تُكتب ألفاً بدون همزة: استغفر، انطلق، ابن، اسم
- همزة القطع تُكتب بهمزة: أكرم، أحسن، إسلام، أنا
- خطأ شائع: حذف الهمزة فوق الألف أو كتابة همزة وصل مكان همزة قطع
```

**Category: `alef_maksura`**:
```
- الألف المقصورة (ى) تُكتب في نهاية الأفعال الماضية الثلاثية: رأى، مشى، جرى
  وفي بعض الأسماء: أخرى، عيسى، موسى
- الياء (ي) تُكتب في نهاية المضارع والمضاف إليه: يمشي، يجري، قاضي
- خطأ شائع في التعرف الضوئي: الخلط بين (ى) و (ي) في نهاية الكلمات
```

**Category: `alef_forms`**:
```
- الهمزة على الألف (أ) للمفتوح والمضموم: أَكَل، أُكِل
- الهمزة تحت الألف (إ) للمكسور: إسلام، إنسان
- الألف المد (آ) للهمزة المفتوحة بعد ألف: آمن، آية
```

**Category: `dots`** (dot-bearing letters most confused by OCR):
```
- ب تحتها نقطة، ت فوقها نقطتان، ث فوقها ثلاث نقاط، ن فوقها نقطة
- ج تحتها نقطة، ح بلا نقطة، خ فوقها نقطة
- خطأ شائع في التعرف الضوئي: إسقاط النقاط أو إضافتها خطأً
```

### 3.5 Execution Flow

Phase 4A follows the same three-stage pipeline as Phase 3:

**Export mode** (`--mode export`):
1. Load all active datasets via `DataLoader`
2. Load rules via `RulesLoader.load()` → `format_for_prompt()`
3. For each sample, write record to `results/phase4a/inference_input.jsonl`:
   ```json
   {
     "sample_id": "...", "dataset": "...", "ocr_text": "...", "gt_text": "...",
     "prompt_type": "rule_augmented",
     "rules_context": "...",
     "num_rules": 6,
     "rule_categories": ["taa_marbuta", "hamza", "alef_maksura", "alef_forms", "dots"]
   }
   ```
4. Unlike Phase 3, the same `rules_context` is shared by all datasets (rules are global, not per-dataset)

**Inference**: Run `scripts/infer.py` with `--input results/phase4a/inference_input.jsonl --output results/phase4a/corrections.jsonl`

**Analyze mode** (`--mode analyze`):
- Computes CER/WER on Phase 4A corrections
- Compares against Phase 2 baseline (isolated)
- Produces `error_changes.json` and `rules_impact.json`

### 3.6 `rules_impact.json`

Measures whether injecting each rule category helped fix the corresponding error type:

```json
{
  "meta": {"dataset": "KHATT-train", "rule_categories_injected": ["taa_marbuta", ...]},
  "impact_by_category": {
    "taa_marbuta": {
      "phase2_fix_rate": 0.715,
      "phase4a_fix_rate": 0.821,
      "marginal_improvement": 0.106,
      "direction": "improved"
    },
    "hamza": {
      "phase2_fix_rate": 0.612,
      "phase4a_fix_rate": 0.598,
      "marginal_improvement": -0.014,
      "direction": "worsened"
    }
  },
  "summary": {
    "categories_improved": 4,
    "categories_worsened": 1,
    "categories_unchanged": 1,
    "avg_marginal_improvement": 0.052
  }
}
```

---

## 4. Phase 4B: QALB Few-Shot Examples

### 4.1 Purpose and Scope

Inject real error-correction pairs from the QALB corpus as few-shot examples. The LLM sees
concrete before/after corrections and generalises from them.

**Key challenge**: QALB contains human typing errors; OCR errors are systematic character-level
substitutions. The overlap is real but partial. Phase 4B answers: *"Do human correction
examples transfer to the OCR correction task?"* — this is itself a research finding.

### 4.2 `QALBLoader` — Module Design

**Location**: `src/data/knowledge_base.py` (replaces the current stub)

```python
from dataclasses import dataclass


@dataclass
class QALBPair:
    """A single error-correction pair from QALB."""
    source: str             # Original erroneous text
    corrected: str          # Human-corrected version
    error_types: list[str]  # Detected error types overlapping with OCR errors
    source_file: str        # e.g., "QALB-2014-L1-Train"


class QALBLoader:
    """Load and select few-shot correction examples from the QALB corpus.

    Phase 4B uses QALB pairs that contain OCR-relevant error types:
    hamza errors, taa marbuta/ha, alef maksura/ya, and similar-shape
    character confusions. Grammar and punctuation-only changes are excluded.

    Usage::

        loader = QALBLoader(config)
        pairs = loader.load(splits=["train"])
        pairs = loader.filter_ocr_relevant(pairs)
        selected = loader.select(pairs, n=5, strategy="diverse")
        text = loader.format_for_prompt(selected)
    """

    # OCR-relevant character substitution patterns to look for
    OCR_SUBSTITUTION_PAIRS: frozenset[tuple[str, str]] = frozenset([
        ("ة", "ه"), ("ه", "ة"),        # taa marbuta / ha
        ("أ", "ا"), ("ا", "أ"),         # hamza on alef
        ("إ", "ا"), ("ا", "إ"),         # hamza below alef
        ("ي", "ى"), ("ى", "ي"),         # ya / alef maksura
        ("أ", "إ"), ("إ", "أ"),         # hamza position
        ("ب", "ت"), ("ت", "ب"),         # dot confusion
        ("ن", "ب"), ("ب", "ن"),
        ("ح", "ج"), ("ج", "ح"),
        ("د", "ذ"), ("ذ", "د"),
        ("ر", "ز"), ("ز", "ر"),
    ])

    def load(
        self,
        splits: list[str] | None = None,
        years: list[str] | None = None,
    ) -> list[QALBPair]:
        """Load QALB pairs from .sent/.cor file pairs.

        Args:
            splits: Which splits to load: "train" | "dev" | "test".
                    Default: ["train"] (dev/test reserved for evaluation).
            years: Which years to load: "2014" | "2015".
                    Default: ["2014"].

        Returns:
            List of QALBPair. Pairs where .cor differs from .sent are included.
            Pairs where source == corrected (no change) are excluded.
        """

    def filter_ocr_relevant(
        self,
        pairs: list[QALBPair],
        require_char_substitution: bool = True,
        max_length: int = 100,
        min_length: int = 10,
    ) -> list[QALBPair]:
        """Keep only pairs where changes match OCR error patterns.

        Filtering criteria:
        1. At least one character-level change matching OCR_SUBSTITUTION_PAIRS
        2. No word count change > 2 (avoids pairs with extensive restructuring)
        3. Text length within [min_length, max_length] characters
        4. Not more than 3 total words changed (keeps examples focused)

        Args:
            pairs: Full QALB pairs from load().
            require_char_substitution: If True, require at least one
                OCR-pattern substitution (default True).
            max_length: Maximum character length for source text.
            min_length: Minimum character length (skip very short texts).

        Returns:
            Filtered list of QALBPair with error_types populated.
        """

    def select(
        self,
        pairs: list[QALBPair],
        n: int = 5,
        strategy: str = "diverse",
        seed: int = 42,
    ) -> list[QALBPair]:
        """Select N pairs using the specified strategy.

        Args:
            pairs: Filtered pairs from filter_ocr_relevant().
            n: Number of examples to select.
            strategy: Selection strategy:
                - "diverse" (default): Select to maximise error type coverage.
                  At most ceil(n / num_categories) examples per error category.
                - "random": Uniform random selection (reproducible with seed).
                - "most_common": Select pairs with the most frequent error types
                  (those that appear most in the Phase 1 confusion matrix).
            seed: Random seed for reproducibility.

        Returns:
            List of exactly n QALBPair (or fewer if fewer pairs are available).
        """

    def format_for_prompt(
        self,
        pairs: list[QALBPair],
        style: str = "inline_arabic",
    ) -> str:
        """Format selected pairs as Arabic few-shot examples for prompt injection.

        Args:
            pairs: Selected pairs from select().
            style: Formatting style:
                - "inline_arabic" (default): inline before/after format
                - "numbered_arabic": numbered list of examples

        Returns:
            Multi-line Arabic string. Empty string if pairs is empty.
        """
```

### 4.3 Parsing QALB Files

**`.sent` file format**: Each line is `{sentence_id}.ar {source_text}`
- Strip everything up to and including the first `.ar ` token to get `source_text`
- Lines that don't match this pattern are skipped

**`.cor` file format**: Each line is `S {corrected_text}`
- Strip the leading `S ` prefix
- Lines are aligned 1:1 with `.sent` lines

**Pairing algorithm**:
```
For each (sent_line, cor_line) in zip(sent_file, cor_file):
    source = sent_line.split(".ar ", 1)[1].strip()
    corrected = cor_line.lstrip("S ").strip()
    if source != corrected:
        yield QALBPair(source, corrected, ...)
```

### 4.4 OCR-Relevant Filtering Algorithm

```python
def _detect_error_types(source: str, corrected: str) -> list[str]:
    """Detect which OCR-relevant error types appear in this pair."""
    types = []
    # Align source and corrected character by character (simple diff)
    # Look for character substitutions matching OCR_SUBSTITUTION_PAIRS
    # Look for merged/split words (word count difference ≤ 2)
    # Classify each difference into an error category
    return types
```

The filtering is conservative: if the dominant change is grammatical restructuring
(verb form changes, preposition choices, etc.), the pair is excluded even if it also
has a taa marbuta fix. This keeps examples focused on OCR-relevant corrections.

### 4.5 Prompt Design

**Version**: `FEWSHOT_PROMPT_VERSION = "p4bv1"`

```python
FEWSHOT_SYSTEM_V1: str = (
    "أنت مصحح نصوص عربية متخصص. "
    "فيما يلي أمثلة على تصحيح أخطاء الكتابة العربية:\n\n"
    "{examples_context}\n\n"
    "صحح النص التالي بنفس الأسلوب. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)
```

**Formatted examples (inline_arabic style)**:
```
أمثلة على التصحيح:
- خطأ: الى التعليق رقم 2 اكيد ان الحكام العرب
  صحيح: إلى التعليق رقم 2 أكيد أن الحكام العرب

- خطأ: نحن ببالغ الاسى نعزي ضحايا الحادث الاليم
  صحيح: نحن ببالغ الأسى نعزي ضحايا الحادث الأليم

- خطأ: مدرسه جديده في حي الجامعه
  صحيح: مدرسة جديدة في حي الجامعة
```

**`PromptBuilder` addition**:
```python
def build_few_shot(self, ocr_text: str, examples_context: str) -> list[dict]:
    """Build few-shot correction prompt (Phase 4B)."""
```

### 4.6 Execution Flow

**Export mode**: Same as Phase 4A except:
- The `examples_context` is computed once globally (same for all datasets)
- OR (optional): computed per-dataset type (PATS vs KHATT) using `strategy="most_common"`
  filtered against each dataset's Phase 1 confusion matrix

```json
{
  "sample_id": "...", "dataset": "...", "ocr_text": "...", "gt_text": "...",
  "prompt_type": "few_shot",
  "examples_context": "أمثلة على التصحيح:\n- خطأ: ...\n  صحيح: ...\n...",
  "num_examples": 5,
  "selection_strategy": "diverse"
}
```

**Inference**: Same `scripts/infer.py` with `prompt_type="few_shot"` dispatch

**Analyze**: Same structure as Phase 4A — produces `metrics.json`, `comparison_vs_phase2.json`,
`error_changes.json`, and `example_impact.json`

### 4.7 `example_impact.json`

```json
{
  "meta": {
    "dataset": "KHATT-train",
    "num_examples_injected": 5,
    "selection_strategy": "diverse",
    "example_error_types": ["taa_marbuta", "hamza", "alef_maksura", "dot_confusion", "hamza"]
  },
  "impact_by_error_type": {
    "taa_marbuta": {
      "num_examples_with_this_type": 2,
      "phase2_fix_rate": 0.715,
      "phase4b_fix_rate": 0.768,
      "marginal_improvement": 0.053,
      "direction": "improved"
    }
  },
  "summary": {
    "types_improved": 3,
    "types_worsened": 1,
    "types_unchanged": 1
  }
}
```

---

## 5. Phase 4C: CAMeL Morphological Validation

### 5.1 Purpose and Scope

Phase 4C is architecturally different from 4A and 4B: it is a **post-processing step**, not
a prompt modification. The LLM generates output identically to Phase 2 (zero-shot), then
CAMeL Tools validates each word. Invalid words are handled according to the validation
strategy.

**No new LLM inference needed.** Phase 4C takes Phase 2's `corrections.jsonl` as input
and processes it locally. This is entirely CPU-bound (no GPU).

### 5.2 `WordValidator` — Module Design

**Location**: `src/linguistic/validator.py` (new file)

```python
from dataclasses import dataclass


@dataclass
class WordValidationResult:
    """Validation result for a single word."""
    word: str               # The word being validated
    is_valid: bool          # True if CAMeL finds >= 1 morphological analysis
    analyses_count: int     # Number of analyses found (0 = invalid)
    suggested: str | None   # CAMeL's top suggestion (if is_valid=False)
    source: str             # "llm" | "ocr" | "original" (after strategy applied)


@dataclass
class TextValidationResult:
    """Validation result for a full text."""
    original_llm: str           # LLM-corrected text (from Phase 2)
    original_ocr: str           # Original OCR text (pre-LLM)
    validated: str              # Final text after applying validation strategy
    words_total: int
    words_invalid: int
    words_reverted: int         # Words that were reverted to OCR
    words_kept_invalid: int     # Invalid words kept (OCR was also invalid)
    strategy: str


class WordValidator:
    """Validate Arabic words using CAMeL Tools morphological analyzer.

    Wraps camel_tools.morphology.analyzer.Analyzer with caching and
    a configurable validation strategy.

    Usage::

        validator = WordValidator(config)
        result = validator.validate_text(
            llm_text="نص LLM المصحح",
            ocr_text="نص OCR الأصلي",
        )
        print(result.validated)  # Text after applying validation strategy

    Strategies:
        "flag_only": Mark invalid words but keep LLM output unchanged.
                     Use for analysis only.
        "revert":    If a word is invalid in LLM output AND valid in OCR original,
                     revert to the OCR word. Otherwise keep LLM word.
                     Most conservative — prevents LLM hallucinations.
        "suggest":   Replace invalid words with CAMeL's top suggestion.
                     Most aggressive — may introduce new errors.
    """

    STRATEGY_REVERT = "revert"      # Primary strategy for Phase 4C
    STRATEGY_FLAG = "flag_only"     # For analysis
    STRATEGY_SUGGEST = "suggest"    # Optional

    def __init__(self, config: dict) -> None:
        """Initialise with CAMeL database and cache settings from config."""

    def validate_word(self, word: str) -> WordValidationResult:
        """Check if a single Arabic word is morphologically valid.

        Uses an LRU cache to avoid re-analyzing the same word.

        Args:
            word: Arabic word string. Non-Arabic tokens (numbers, punctuation,
                  Latin characters) are always returned as valid (is_valid=True).

        Returns:
            WordValidationResult.
        """

    def validate_text(
        self,
        llm_text: str,
        ocr_text: str,
        strategy: str = STRATEGY_REVERT,
    ) -> TextValidationResult:
        """Validate all words in llm_text and apply the specified strategy.

        Word-level alignment between llm_text and ocr_text is required for
        the "revert" strategy. If word counts differ significantly (>20%),
        alignment is skipped and the strategy falls back to "flag_only" for
        that sample, with a warning logged.

        Args:
            llm_text: LLM-corrected text (from Phase 2 corrections.jsonl).
            ocr_text: Original OCR text (source for LLM correction).
            strategy: Validation strategy (see class docstring).

        Returns:
            TextValidationResult with .validated containing the final text.
        """
```

### 5.3 Validation Strategy: `revert` (Primary)

```
For each word w_llm in LLM output:
  result = validate_word(w_llm)
  if result.is_valid:
    keep w_llm
  else:
    Find aligned OCR word w_ocr (by position)
    if w_ocr is morphologically valid:
      use w_ocr  (revert to OCR)
      words_reverted += 1
    else:
      keep w_llm  (both invalid — keep LLM choice as it may be closer to gt)
      words_kept_invalid += 1
```

**Alignment**: Simple token-index alignment. If `len(llm_words) != len(ocr_words)` by more
than 20%, skip alignment and default to `flag_only` for that sample.

**Non-Arabic tokens**: Numbers, punctuation, and Latin characters are always marked valid
and never reverted.

### 5.4 Execution Flow

Phase 4C uses a **single-stage pipeline**: no export, no Kaggle.

```
run_phase4c.py --mode validate
│
├─ 1. parse_args() + load_config()
│
├─ 2. Initialise WordValidator(config)
│       Load CAMeL database (once, slow first time)
│
├─ 3. Load Phase 2 corrections.jsonl → list[dict]
│       Each record has: sample_id, ocr_text, corrected_text, gt_text, dataset, ...
│
├─ 4. For each dataset_key in active_datasets:
│     │
│     ├─ 4a. Check resume: if results/phase4c/{key}/metrics.json exists → skip
│     │
│     ├─ 4b. For each sample in dataset:
│     │       result = validator.validate_text(
│     │           llm_text=sample["corrected_text"],
│     │           ocr_text=sample["ocr_text"],
│     │           strategy=args.strategy,
│     │       )
│     │       Write to results/phase4c/{key}/corrections.jsonl
│     │
│     ├─ 4c. calculate_metrics() → metrics.json
│     │
│     ├─ 4d. compare_metrics(phase2_baseline) → comparison_vs_phase2.json
│     │
│     ├─ 4e. [if not --no-error-analysis]
│     │       run_error_change_analysis() → error_changes.json
│     │
│     └─ 4f. Save validation stats → validation_stats.json
│
├─ 5. Aggregate metrics → results/phase4c/metrics.json
├─ 6. Aggregate comparisons → results/phase4c/comparison.json
└─ 7. generate_report() → results/phase4c/report.md
```

### 5.5 `validation_stats.json`

```json
{
  "meta": {
    "dataset": "KHATT-train",
    "strategy": "revert",
    "camel_db": "calima-msa-r13",
    "generated_at": "..."
  },
  "totals": {
    "samples_processed":      1400,
    "samples_alignment_ok":   1312,
    "samples_alignment_skip":   88,
    "words_total":           28901,
    "words_invalid_in_llm":   1247,
    "words_reverted":          892,
    "words_kept_invalid":      355,
    "revert_rate_pct":         3.09
  },
  "by_error_type": {
    "taa_marbuta": {
      "invalid_count": 312,
      "reverted_count": 289,
      "revert_rate_pct": 92.6
    },
    "hamza": {
      "invalid_count": 198,
      "reverted_count": 156,
      "revert_rate_pct": 78.8
    }
  }
}
```

### 5.6 Phase 4C `corrections.jsonl` Schema

Extends Phase 2 records with validation metadata:

```json
{
  "sample_id":         "AHTD3A0001_Para2_3",
  "dataset":           "KHATT-train",
  "ocr_text":          "...",
  "corrected_text":    "...",   ← Phase 4C validated text
  "llm_text":          "...",   ← Phase 2 LLM text (before validation)
  "gt_text":           "...",
  "model":             "Qwen/Qwen3-4B-Instruct-2507",
  "prompt_type":       "zero_shot_camel_validated",
  "prompt_version":    "v1",    ← same as Phase 2 (no new inference)
  "strategy":          "revert",
  "words_reverted":    3,
  "words_kept_invalid": 1,
  "alignment_ok":      true,
  "success":           true,
  "error":             null
}
```

---

## 6. Shared Pipeline Infrastructure

### 6.1 Single Pipeline File

All three sub-phases are implemented in a **single file** `pipelines/run_phase4.py` with a
required `--sub-phase` argument:

```
python pipelines/run_phase4.py --sub-phase 4a --mode export
python pipelines/run_phase4.py --sub-phase 4a --mode analyze
python pipelines/run_phase4.py --sub-phase 4b --mode export
python pipelines/run_phase4.py --sub-phase 4b --mode analyze
python pipelines/run_phase4.py --sub-phase 4c --mode validate
```

**Why a single file**: All three sub-phases share: config loading, dataset resolution,
`resolve_datasets()`, metrics computation, comparison against Phase 2, error analysis, and
report generation. Sub-phase-specific logic is encapsulated in separate functions.

### 6.2 CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sub-phase` | str | **required** | `4a` \| `4b` \| `4c` |
| `--mode` | str | **required** | `export` \| `analyze` (4a/4b), `validate` (4c) |
| `--limit` | int | None | Max samples per dataset |
| `--datasets` | str+ | None | Subset of dataset keys |
| `--force` | flag | False | Re-run even if output exists |
| `--no-error-analysis` | flag | False | Skip error_changes.json |
| `--config` | path | `configs/config.yaml` | Config file |
| `--results-dir` | path | `results/phase4{sub}` | Output directory |
| `--phase2-dir` | path | `results/phase2` | Phase 2 baseline directory |
| `--phase1-dir` | path | `results/phase1` | Phase 1 results (used by 4b for confusion-guided selection) |

**4A-specific**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rule-categories` | str+ | all | Rule categories to inject |
| `--num-rules` | int | None (all) | Max rules to inject |

**4B-specific**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-examples` | int | 5 | Number of few-shot examples |
| `--selection` | str | `diverse` | `diverse` \| `random` \| `most_common` |
| `--qalb-splits` | str+ | `train` | Which QALB splits to load |

**4C-specific**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--strategy` | str | `revert` | `revert` \| `flag_only` \| `suggest` |
| `--phase2-corrections` | path | `results/phase2/corrections.jsonl` | Phase 2 input for 4C |

### 6.3 `scripts/infer.py` Additions

Two new `prompt_type` values are added to the dispatch block:

```python
elif prompt_type == "rule_augmented":
    rules_context = record.get("rules_context", "")
    messages = builder.build_rule_augmented(record["ocr_text"], rules_context)
    prompt_ver = builder.rules_prompt_version

elif prompt_type == "few_shot":
    examples_context = record.get("examples_context", "")
    messages = builder.build_few_shot(record["ocr_text"], examples_context)
    prompt_ver = builder.few_shot_prompt_version
```

Fallback behavior: if `rules_context` or `examples_context` is empty, both builders fall back
to `build_zero_shot()` and log a warning (same pattern as Phase 3's `confusion_context`).

---

## 7. `src/core/prompt_builder.py` Additions

```python
# ---------------------------------------------------------------------------
# Phase 4A — Rule-Augmented Prompting
# ---------------------------------------------------------------------------

RULES_SYSTEM_V1: str = (
    "أنت مصحح نصوص عربية متخصص. "
    "راعِ القواعد الإملائية التالية عند تصحيح النص:\n\n"
    "{rules_context}\n\n"
    "صحح النص التالي مع الانتباه بشكل خاص لهذه القواعد. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)

RULES_PROMPT_VERSION: str = "p4av1"

def build_rule_augmented(self, ocr_text: str, rules_context: str) -> list[dict]:
    """Build rule-augmented correction prompt (Phase 4A).
    Falls back to zero-shot if rules_context is empty.
    """

@property
def rules_prompt_version(self) -> str: ...

# ---------------------------------------------------------------------------
# Phase 4B — Few-Shot Prompting
# ---------------------------------------------------------------------------

FEWSHOT_SYSTEM_V1: str = (
    "أنت مصحح نصوص عربية متخصص. "
    "فيما يلي أمثلة على تصحيح أخطاء الكتابة العربية:\n\n"
    "{examples_context}\n\n"
    "صحح النص التالي بنفس الأسلوب. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)

FEWSHOT_PROMPT_VERSION: str = "p4bv1"

def build_few_shot(self, ocr_text: str, examples_context: str) -> list[dict]:
    """Build few-shot correction prompt (Phase 4B).
    Falls back to zero-shot if examples_context is empty.
    """

@property
def few_shot_prompt_version(self) -> str: ...
```

---

## 8. `src/linguistic/validator.py` (New File)

```
src/
└── linguistic/
    ├── __init__.py       (create if not exists)
    └── validator.py      ← NEW: WordValidator, WordValidationResult, TextValidationResult
```

The `src/linguistic/` package is introduced in Phase 4C. It will also house future CAMeL
wrappers (`morphology.py`, `features.py`) in later phases.

---

## 9. Output Structure

```
results/
├── phase4a/
│   ├── inference_input.jsonl     ← upload to Kaggle/Colab
│   ├── corrections.jsonl         ← download from Kaggle/Colab
│   ├── {dataset_name}/
│   │   ├── corrections.jsonl     ← per-dataset split
│   │   ├── metrics.json
│   │   ├── comparison_vs_phase2.json
│   │   ├── error_changes.json
│   │   └── rules_impact.json
│   ├── metrics.json              ← aggregated
│   ├── comparison.json
│   ├── report.md
│   └── phase4a.log
│
├── phase4b/
│   ├── inference_input.jsonl
│   ├── corrections.jsonl
│   ├── {dataset_name}/
│   │   ├── corrections.jsonl
│   │   ├── metrics.json
│   │   ├── comparison_vs_phase2.json
│   │   ├── error_changes.json
│   │   └── example_impact.json
│   ├── metrics.json
│   ├── comparison.json
│   ├── report.md
│   └── phase4b.log
│
└── phase4c/
    ├── {dataset_name}/
    │   ├── corrections.jsonl     ← locally generated (no Kaggle step)
    │   ├── metrics.json
    │   ├── comparison_vs_phase2.json
    │   ├── error_changes.json
    │   └── validation_stats.json
    ├── metrics.json
    ├── comparison.json
    ├── report.md
    └── phase4c.log
```

---

## 10. Output Schemas

### 10.1 `metrics.json` (per dataset, all sub-phases)

Same structure as Phase 3, with `phase` and `prompt_type` updated per sub-phase:

```json
{
  "meta": {
    "phase": "phase4a",
    "dataset": "KHATT-train",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt_type": "rule_augmented",
    "prompt_version": "p4av1",
    "num_rules": 6,
    "rule_categories": ["taa_marbuta", "hamza", "alef_maksura", "alef_forms", "dots"],
    "generated_at": "...",
    "num_samples": 1400
  },
  "corrected": {
    "cer": 0.073,
    "wer": 0.201,
    ...
  }
}
```

For Phase 4B: `prompt_type: "few_shot"`, `num_examples: 5`, `selection_strategy: "diverse"`.
For Phase 4C: `prompt_type: "zero_shot_camel_validated"`, `strategy: "revert"`.

### 10.2 `comparison_vs_phase2.json` (per dataset, all sub-phases)

```json
{
  "meta": {
    "comparison": "phase4a_vs_phase2",
    "dataset": "KHATT-train",
    "generated_at": "..."
  },
  "phase2_baseline": {"cer": 0.089, "wer": 0.234},
  "phase4_corrected": {"cer": 0.073, "wer": 0.201},
  "delta": {
    "cer_absolute": -0.016,
    "wer_absolute": -0.033,
    "cer_relative_pct": -18.0,
    "wer_relative_pct": -14.1
  },
  "interpretation": "CER reduced by 18.0% relative vs Phase 2.",
  "significant": null
}
```

---

## 11. Configuration (`configs/config.yaml` Additions)

```yaml
# ---------------------------------------------------------------------------
# Phase 4 specific
# ---------------------------------------------------------------------------
phase4:
  # Phase 4A — Rule-Augmented
  rules:
    categories: null         # null = all categories; or list e.g. ["hamza", "taa_marbuta"]
    num_rules: null          # null = all rules in selected categories
    format_style: "compact_arabic"   # "compact_arabic" | "detailed_arabic"
    analyze_impact: true

  # Phase 4B — Few-Shot
  few_shot:
    num_examples: 5
    selection: "diverse"     # "diverse" | "random" | "most_common"
    qalb_splits:             # QALB splits to load examples from
      - "train"
    qalb_years:
      - "2014"
    max_source_length: 100   # Max character length for a QALB example
    analyze_impact: true

  # Phase 4C — CAMeL Validation
  camel_validation:
    strategy: "revert"       # "revert" | "flag_only" | "suggest"
    phase2_corrections: "results/phase2/corrections.jsonl"
    analyze_errors: true
```

---

## 12. Module Summary: What's New vs Modified

### 12.1 New Files

| File | Phase | Purpose |
|------|-------|---------|
| `src/linguistic/__init__.py` | 4C | Package init |
| `src/linguistic/validator.py` | 4C | `WordValidator`, validation results |

### 12.2 Modified Files

| File | Phase | Changes |
|------|-------|---------|
| `src/data/knowledge_base.py` | 4A, 4B | Replace `RulesLoader` stub with full implementation; replace `QALBLoader` stub with full implementation |
| `src/core/prompt_builder.py` | 4A, 4B | Add `build_rule_augmented()`, `build_few_shot()`, two new constants, two new version properties |
| `scripts/infer.py` | 4A, 4B | Add `rule_augmented` and `few_shot` cases to `prompt_type` dispatch |
| `configs/config.yaml` | 4A/B/C | Add `phase4:` block |

### 12.3 New Pipeline Files

| File | Purpose |
|------|---------|
| `pipelines/run_phase4.py` | Full pipeline for 4A, 4B, 4C (`--sub-phase 4a|4b|4c`) |

### 12.4 No Modifications to Phase 2 / Phase 3

Existing Phase 2 and Phase 3 pipeline files and output files are untouched.

---

## 13. Dependencies

### 13.1 Phase 4A and 4B

No new packages required. `pyyaml`, `tqdm`, `editdistance`, `jiwer` already present.

### 13.2 Phase 4C (CAMeL Tools)

```bash
pip install camel-tools
# On first run, download the morphological database:
camel_data -i morphology-db-msa-s31
```

`camel-tools` is already noted in `configs/config.yaml` and `docs/Architecture.md`.
The database download is a one-time setup. The validator handles `ImportError` gracefully:
if `camel_tools` is not installed, Phase 4C exits with a clear error message.

---

## 14. Testing

### 14.1 New Test Files

```
tests/
├── test_rules_loader.py         ← Phase 4A
├── test_qalb_loader.py          ← Phase 4B
├── test_word_validator.py       ← Phase 4C
├── test_prompt_builder_phase4.py
└── fixtures/
    ├── sample_qalb.sent         ← 10 QALB sent lines (anonymised)
    ├── sample_qalb.cor          ← 10 corresponding corrected lines
    └── sample_corrections_p2.jsonl  ← 5 Phase 2 correction records for 4C test
```

### 14.2 `test_rules_loader.py`

- `test_load_returns_all_core_rules_by_default`
- `test_load_filters_by_category`
- `test_load_unknown_category_returns_empty`
- `test_format_compact_arabic_produces_header`
- `test_format_limits_to_n_rules`
- `test_format_returns_empty_string_for_empty_list`

### 14.3 `test_qalb_loader.py`

- `test_load_parses_sent_cor_pairs`
- `test_load_skips_unchanged_pairs`
- `test_filter_keeps_taa_marbuta_errors`
- `test_filter_excludes_grammar_only_changes`
- `test_select_diverse_covers_multiple_categories`
- `test_select_random_is_reproducible_with_seed`
- `test_format_inline_arabic_produces_bullet_pairs`
- `test_format_returns_empty_string_for_empty_list`

### 14.4 `test_word_validator.py`

- `test_valid_arabic_word_returns_is_valid_true`
- `test_invalid_arabic_word_returns_is_valid_false`
- `test_non_arabic_token_always_returns_valid`
- `test_revert_strategy_reverts_invalid_word_when_ocr_valid`
- `test_revert_strategy_keeps_llm_word_when_both_invalid`
- `test_flag_only_strategy_keeps_llm_output_unchanged`
- `test_validate_text_alignment_skipped_when_word_counts_differ`
- `test_cache_hit_avoids_reanalysis`

### 14.5 `test_prompt_builder_phase4.py`

- `test_build_rule_augmented_returns_two_messages`
- `test_build_rule_augmented_contains_rules_in_system`
- `test_build_rule_augmented_empty_context_falls_back_to_zero_shot`
- `test_rules_prompt_version_is_p4av1`
- `test_build_few_shot_returns_two_messages`
- `test_build_few_shot_contains_examples_in_system`
- `test_build_few_shot_empty_context_falls_back_to_zero_shot`
- `test_few_shot_prompt_version_is_p4bv1`
- `test_phase3_build_ocr_aware_unchanged` (regression)
- `test_phase2_build_zero_shot_unchanged` (regression)

---

## 15. Implementation Order

| Step | File | Notes |
|------|------|-------|
| 1 | `src/data/knowledge_base.py` — `RulesLoader` | Replace stub; `CORE_RULES` list |
| 2 | `tests/test_rules_loader.py` | Verify rule loading and formatting |
| 3 | `src/data/knowledge_base.py` — `QALBLoader` | Replace stub; parse .sent/.cor |
| 4 | `tests/test_qalb_loader.py` | Verify parsing, filtering, selection |
| 5 | `src/core/prompt_builder.py` — add 4A/4B methods | Minimal, additive changes |
| 6 | `tests/test_prompt_builder_phase4.py` | Including regression tests |
| 7 | `scripts/infer.py` — add dispatch cases | Two new elif branches |
| 8 | `src/linguistic/__init__.py` + `validator.py` | CAMeL wrapper |
| 9 | `tests/test_word_validator.py` | Requires camel-tools installed |
| 10 | `pipelines/run_phase4.py` — 4A export + analyze | First sub-phase |
| 11 | Smoke test 4A export: `--sub-phase 4a --mode export --limit 5` | |
| 12 | Run 4A inference on Kaggle | `scripts/infer.py --input results/phase4a/...` |
| 13 | Smoke test 4A analyze | `--sub-phase 4a --mode analyze --datasets KHATT-train` |
| 14 | `pipelines/run_phase4.py` — 4B export + analyze | Second sub-phase |
| 15 | Smoke test 4B export + inference + analyze | Same pattern |
| 16 | `pipelines/run_phase4.py` — 4C validate | Third sub-phase (no Kaggle) |
| 17 | Smoke test 4C: `--sub-phase 4c --mode validate --limit 50` | Local only |
| 18 | Full runs for all three sub-phases (all datasets) | Paper numbers |

---

## 16. Known Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| QALB pairs have too little OCR overlap — few examples pass filter | Medium | Lower the filter threshold; in the worst case, use all pairs with ≥1 character change (relax `require_char_substitution`). If QALB pairs are still too few, log a warning and fall back to zero-shot. |
| LLM ignores injected rules or examples | Possible | This is a valid research finding; report it. Also consider the `grouped_arabic` rule format as an optional variant. |
| CAMeL Tools `calima-msa-r13` database not downloaded | Medium | Phase 4C exits with a clear `ImportError` or `FileNotFoundError` with a one-line download command. The validator wraps the import in a try/except. |
| Word alignment fails for many samples (word count mismatch) | Medium | `validate_text()` falls back to `flag_only` for misaligned samples. Log count of fallbacks in `validation_stats.json`. |
| CAMeL's morphological database only covers MSA — KHATT/PATS may have dialectal forms | Low | Dialectal words may be incorrectly flagged as invalid. Log `camel_db` version in `validation_stats.json`. Consider adding an allowlist of common dialectal words. |
| `revert` strategy may hurt: if LLM correctly fixed a word but CAMeL marks the correction invalid | Medium | Log `words_reverted` count; compare Phase 4C CER vs Phase 2 CER by error type to detect this. If it hurts, `flag_only` is a safe fallback. |
| QALB `.sent` parsing fails on edge cases (multi-space, no `.ar` token) | Low | `QALBLoader.load()` skips malformed lines with a WARNING log. |
| `rules_context` string too long for context window | Low | `RulesLoader.format_for_prompt(n=...)` limits the count. Default `num_rules=None` with 6 categories × ~2 rules = ~12 rules = ~600 Arabic characters. Well within limits. |

---

## 17. Appendix: Prompt Text Examples

### Phase 4A System Prompt (compact_arabic, all categories)

```
أنت مصحح نصوص عربية متخصص. راعِ القواعد الإملائية التالية عند تصحيح النص:

• التاء المربوطة (ة) للأسماء المؤنثة: مدرسة، جامعة — والهاء (ه) للأفعال وأسماء الإشارة: يكتبه، هذه
• همزة الوصل بلا همزة: استغفر، انطلق، اسم — وهمزة القطع بهمزة: أكرم، أحسن، إسلام
• الألف المقصورة (ى) في نهاية الأفعال: رأى، مشى — والياء (ي) في المضارع والمضاف إليه: يمشي، قاضي
• الهمزة على الألف (أ) للمفتوح والمضموم، وتحتها (إ) للمكسور، والمد (آ): آمن، آية
• الأحرف المنقوطة: ب نقطة واحدة، ت نقطتان، ث ثلاث نقاط، ن نقطة فوق — تحقق من النقاط بعناية

صحح النص التالي مع الانتباه بشكل خاص لهذه القواعد.
أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي.
```

### Phase 4B System Prompt (5 examples, inline_arabic)

```
أنت مصحح نصوص عربية متخصص. فيما يلي أمثلة على تصحيح أخطاء الكتابة العربية:

أمثلة على التصحيح:
- خطأ: الى التعليق رقم 2 اكيد ان الحكام العرب مسؤولية
  صحيح: إلى التعليق رقم 2 أكيد أن الحكام العرب مسؤولية

- خطأ: مدرسه جديده في حي الجامعه والمنطقه التعليميه
  صحيح: مدرسة جديدة في حي الجامعة والمنطقة التعليمية

- خطأ: نحن ببالغ الاسى نعزي ضحايا الحادث الاليم
  صحيح: نحن ببالغ الأسى نعزي ضحايا الحادث الأليم

- خطأ: ذاهب الى المدرسه اليوم للاجتماع
  صحيح: ذاهب إلى المدرسة اليوم للاجتماع

- خطأ: واصل الثوار انتصاراتهم وحررو كل حقول
  صحيح: واصل الثوار انتصاراتهم وحرروا كل حقول

صحح النص التالي بنفس الأسلوب.
أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي.
```

### Comparison: Phase 2 vs 4A vs 4B System Prompts

| Aspect | Phase 2 | Phase 3 | Phase 4A | Phase 4B |
|--------|---------|---------|----------|----------|
| Mentions Qaari | No | Yes | No | No |
| Injects confusion pairs | No | Yes | No | No |
| Injects rules | No | No | Yes | No |
| Injects examples | No | No | No | Yes |
| Prompt length (approx.) | ~25 tokens | ~150–200 | ~100–150 | ~200–350 |
| Prompt version | `v1` | `p3v1` | `p4av1` | `p4bv1` |
