"""Phase-specific prompt construction for LLM correction.

All methods return OpenAI chat format (list[dict]) so they are compatible
with both HuggingFace tokenizer.apply_chat_template() and API calls.

Architecture
------------
The base prompt (``configs/crafted_system_prompt.txt``) uses an XML-like
tag structure: ``<system>``, ``<task>``, ``<rules>``, ``<examples>``,
``<output_format>``.  Phase-specific knowledge is injected as additional
XML sections following the same convention.

Prompt language
~~~~~~~~~~~~~~~
Section wrappers and labels are in English.  Arabic-specific content
(confusion pairs, word-error examples, insights) remains
in Arabic — it is the knowledge payload, not the meta-language.

Injection strategy
~~~~~~~~~~~~~~~~~~
* **Knowledge sections** (phases 3, 4) — injected as XML tags
  immediately *before* ``<examples>`` so the model sees them before the
  few-shot pairs:

    - Phase 3 → ``<confusion_patterns>``
    - Phase 4 → ``<self_analysis>`` (bundles insights + word pairs + overcorrection warnings)

* **Phase 6** (combined) injects all of the above in order; each section
  is only included if its context is non-empty.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build phase-specific chat message lists for LLM correction.

    Each ``build_*`` method returns a list[dict] in OpenAI chat format and
    can be passed directly to any ``BaseLLMCorrector`` implementation.

    All phases share the crafted system prompt as their base.  Phase-specific
    knowledge is injected as XML sections via :meth:`_inject_knowledge`
    (before ``<examples>``) or :meth:`_append_examples` (before
    ``<output_format>``).

    Usage::

        builder = PromptBuilder()                          # default path
        builder = PromptBuilder("path/to/system.txt")     # custom path
        messages = builder.build_ocr_aware(ocr_text, confusion_context)
        result = corrector.correct(sample_id, ocr_text, messages)
    """

    # Default path to the crafted system prompt (relative to project root).
    DEFAULT_CRAFTED_PROMPT_PATH: str = (
        "configs/crafted_system_prompt.txt"
    )

    # -----------------------------------------------------------------------
    # Prompt version strings — bump whenever wording changes between runs.
    # -----------------------------------------------------------------------

    PROMPT_VERSION: str = "p2v2"                   # Phase 2 — crafted base (unchanged)
    OCR_AWARE_PROMPT_VERSION: str = "p3v2"          # Phase 3 — improved section labels
    SELF_REFLECTIVE_PROMPT_VERSION: str = "p4v2"    # Phase 4 — improved labels + few-shot
    COMBINED_PROMPT_VERSION: str = "p6v2"           # Phase 6 — improved labels + few-shot
    RAG_PROMPT_VERSION: str = "p8v1"                # Phase 8 — RAG retrieval-augmented
    CRAFTED_PROMPT_VERSION: str = "crafted_v1"      # standalone crafted
    META_PROMPT_VERSION: str = "meta_v1"            # meta-prompting

    # -----------------------------------------------------------------------
    # Legacy system prompts — retained for backward compatibility when
    # re-running old exported JSONL files that carry version="v1"/"v2".
    # -----------------------------------------------------------------------

    ZERO_SHOT_SYSTEM_V1: str = (
        "أنت مصحح نصوص عربية متخصص. "
        "مهمتك تصحيح أخطاء التعرف الضوئي (OCR) في النص العربي. "
        "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
    )

    ZERO_SHOT_SYSTEM_V2: str = (
        "أنت مدقق نصوص عربية. "
        "النص التالي ناتج عن نظام التعرف الضوئي وقد يكون صحيحاً بالكامل أو يحتوي على أخطاء قليلة. "
        "مهمتك: إذا كان النص صحيحاً، أعده كما هو بالضبط. "
        "إذا وجدت خطأ واضحاً فقط، صححه. لا تغير ما هو صحيح. "
        "أعد النص فقط بدون أي شرح."
    )

    _ZERO_SHOT_VERSIONS: dict[str, str] = {
        "v1": ZERO_SHOT_SYSTEM_V1,
        "v2": ZERO_SHOT_SYSTEM_V2,
    }

    # -----------------------------------------------------------------------
    # Anchors for structured injection into the crafted base prompt.
    # The prompt uses XML-like section tags; injections follow that pattern.
    # -----------------------------------------------------------------------

    # Knowledge sections (phases 3, 4) are inserted immediately before
    # the <examples> block so the model sees them before the few-shot pairs.
    _INJECT_ANCHOR: str = "<examples>"

    # Dynamic few-shot examples are appended inside the <examples> block,
    # after any static examples already in the prompt file.
    _EXAMPLES_END_ANCHOR: str = "</examples>"

    # -----------------------------------------------------------------------
    # Per-phase XML section templates.
    # Section wrappers / labels are in English; Arabic content stays Arabic.
    # -----------------------------------------------------------------------

    _OCR_AWARE_SECTION: str = (
        "<confusion_patterns>\n"
        "This OCR system makes these systematic character errors. "
        "When you see the LEFT form in the input, correct it to the RIGHT form:\n\n"
        "{confusion_context}\n"
        "</confusion_patterns>"
    )

    # Phase 8 — RAG retrieval-augmented sections.
    _RAG_SENTENCES_SECTION: str = (
        "<retrieved_corrections>\n"
        "These are corrections from similar OCR texts in the training data. "
        "Use them as reference for fixing similar errors in the input:\n\n"
        "{retrieved_sentences}\n"
        "</retrieved_corrections>"
    )

    _RAG_WORDS_SECTION: str = (
        "<retrieved_word_fixes>\n"
        "Word-level corrections from similar OCR contexts "
        "(OCR form \u2192 correct form):\n\n"
        "{retrieved_words}\n"
        "</retrieved_word_fixes>"
    )

    # Combined-phase section body for confusion (same as standalone).
    _COMBINED_CONFUSION: str = (
        "<confusion_patterns>\n"
        "This OCR system makes these systematic character errors. "
        "When you see the LEFT form in the input, correct it to the RIGHT form:\n\n"
        "{confusion_context}\n"
        "</confusion_patterns>"
    )

    # Self-analysis section is built dynamically (bundles insights + word pairs).
    # _COMBINED_INSIGHTS and _COMBINED_WORD_PAIRS are inlined in build_* methods.

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, crafted_prompt_path: str | None = None) -> None:
        """
        Args:
            crafted_prompt_path: Path to the crafted system prompt text file.
                Defaults to ``DEFAULT_CRAFTED_PROMPT_PATH``.  A warning is
                logged and ``ZERO_SHOT_SYSTEM_V2`` is used as fallback if the
                file cannot be found.
        """
        self._crafted_prompt_path: str = (
            crafted_prompt_path or self.DEFAULT_CRAFTED_PROMPT_PATH
        )
        self._crafted_base: str | None = None  # lazy-loaded + cached

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _load_crafted_base(self) -> str:
        """Return the crafted system prompt text, loading from disk on first call."""
        if self._crafted_base is None:
            path = Path(self._crafted_prompt_path)
            # If relative path not found from CWD, try resolving from project root.
            # prompt_builder.py lives at src/core/ — project root is 3 levels up.
            # This handles Kaggle/Colab where CWD != project root.
            if not path.is_absolute() and not path.exists():
                project_root = Path(__file__).resolve().parent.parent.parent
                alt = project_root / path
                if alt.exists():
                    path = alt
            try:
                with open(path, encoding="utf-8") as f:
                    self._crafted_base = f.read()
            except FileNotFoundError:
                logger.warning(
                    "Crafted prompt not found at '%s' — falling back to zero-shot v2.",
                    self._crafted_prompt_path,
                )
                self._crafted_base = self.ZERO_SHOT_SYSTEM_V2
        return self._crafted_base

    def _inject_knowledge(self, base: str, section: str) -> str:
        """Insert *section* immediately before the ``<examples>`` block.

        Falls back to injecting before ``<output_format>`` when the base
        prompt has no ``<examples>`` anchor (e.g. the clean base prompt).
        This guarantees all knowledge sections appear before the output
        instruction regardless of whether the base file has an examples block.

        Multiple calls chain correctly: each successive injection finds the
        ``<output_format>`` tag further right (after previously injected sections),
        so the natural ordering is preserved — first call closest to rules,
        last call closest to output_format.
        """
        if self._INJECT_ANCHOR in base:
            return base.replace(
                self._INJECT_ANCHOR,
                section + "\n\n" + self._INJECT_ANCHOR,
                1,
            )
        if "<output_format>" in base:
            return base.replace("<output_format>", section + "\n\n<output_format>", 1)
        return base.rstrip() + "\n\n" + section

    def _inject_dynamic_examples(self, base: str, examples_text: str) -> str:
        """Append *examples_text* inside the ``<examples>`` block.

        Dynamic few-shot examples are placed after any static examples already
        in the prompt file.  If no ``</examples>`` anchor is found, a new
        ``<examples>`` block is inserted just before ``<output_format>``
        (or appended at the end as a last resort).

        Args:
            base: The system prompt string (after knowledge injection).
            examples_text: Pre-formatted INPUT/OUTPUT pairs from
                ``FewShotExampleSelector.format_for_prompt()``.

        Returns:
            Updated system prompt string with dynamic examples embedded.
        """
        if not examples_text.strip():
            return base

        if self._EXAMPLES_END_ANCHOR in base:
            return base.replace(
                self._EXAMPLES_END_ANCHOR,
                "\n" + examples_text + "\n" + self._EXAMPLES_END_ANCHOR,
                1,
            )

        # No </examples> — try to insert a new block before <output_format>
        output_anchor = "<output_format>"
        if output_anchor in base:
            return base.replace(
                output_anchor,
                "<examples>\n" + examples_text + "\n</examples>\n\n" + output_anchor,
                1,
            )

        # Last resort: append at the end
        return base.rstrip() + "\n\n<examples>\n" + examples_text + "\n</examples>"

    def _messages(self, system: str, ocr_text: str) -> list[dict]:
        """Return standard two-element OpenAI chat messages."""
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    # -----------------------------------------------------------------------
    # Phase 2 — Zero-Shot
    # -----------------------------------------------------------------------

    def build_zero_shot(
        self,
        ocr_text: str,
        version: str = "crafted",
        few_shot_examples: str = "",
    ) -> list[dict]:
        """Build zero-shot correction prompt (Phase 2).

        Uses the crafted system prompt as-is by default.  Pass ``version="v1"``
        or ``version="v2"`` to use legacy prompts when re-running old exports.

        Args:
            ocr_text: OCR prediction text to correct.
            version: ``"crafted"`` (default), ``"v1"``, or ``"v2"``.
            few_shot_examples: Optional pre-formatted INPUT/OUTPUT examples from
                ``FewShotExampleSelector.format_for_prompt()``.  Appended inside
                the ``<examples>`` block of the crafted prompt.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if version in self._ZERO_SHOT_VERSIONS:
            system = self._ZERO_SHOT_VERSIONS[version]
        else:
            system = self._load_crafted_base()
        if few_shot_examples.strip():
            system = self._inject_dynamic_examples(system, few_shot_examples)
        return self._messages(system, ocr_text)

    @property
    def prompt_version(self) -> str:
        """Phase 2 prompt version string."""
        return self.PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 3 — OCR-Aware Prompting
    # -----------------------------------------------------------------------

    def build_ocr_aware(
        self,
        ocr_text: str,
        confusion_context: str,
        word_examples: str = "",
        few_shot_examples: str = "",
    ) -> list[dict]:
        """Build OCR-aware correction prompt (Phase 3).

        Injects Qaari's top-N character confusion pairs into the crafted
        system prompt.  If *word_examples* is provided (Phase 3 enhanced),
        appends concrete word-level failure examples inside the same
        ``<confusion_patterns>`` section.

        Falls back to zero-shot if *confusion_context* is empty.

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted Arabic confusion-pairs string
                produced by ``ConfusionMatrixLoader.format_for_prompt()``.
            word_examples: Optional pre-formatted word-level failure examples
                from training cross-reference.
            few_shot_examples: Optional pre-formatted INPUT/OUTPUT examples from
                ``FewShotExampleSelector.format_for_prompt()``.  Appended inside
                the ``<examples>`` block.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not confusion_context.strip():
            return self.build_zero_shot(ocr_text, few_shot_examples=few_shot_examples)
        base = self._load_crafted_base()
        section = self._OCR_AWARE_SECTION.format(confusion_context=confusion_context)
        if word_examples.strip():
            # Insert word examples inside the confusion_patterns section
            section = section.replace(
                "</confusion_patterns>",
                "\n\nWORD-LEVEL FAILURES — common whole-word misreads from training data "
                "(OCR form → correct form):\n" + word_examples + "\n</confusion_patterns>",
            )
        system = self._inject_knowledge(base, section)
        if few_shot_examples.strip():
            system = self._inject_dynamic_examples(system, few_shot_examples)
        return self._messages(system, ocr_text)

    @property
    def ocr_aware_prompt_version(self) -> str:
        """Phase 3 prompt version string."""
        return self.OCR_AWARE_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 4 — Self-Reflective Prompting
    # -----------------------------------------------------------------------

    def build_self_reflective(
        self,
        ocr_text: str,
        insights_context: str,
        word_pairs_context: str = "",
        overcorrection_context: str = "",
        few_shot_examples: str = "",
    ) -> list[dict]:
        """Build Phase 4 correction prompt: self-reflective insights + word-error pairs.

        Injects all signals as a single ``<self_analysis>`` section before
        ``<examples>``.  Dynamic few-shot examples are appended inside
        ``<examples>``.  Falls back to zero-shot if all contexts are empty.

        Injection order in the final system prompt::

            <error_patterns>    (from crafted base)
            <self_analysis>     (insights + word pairs + overcorrection warnings)
            <examples>
              [static examples from crafted base]
              [dynamic few-shot examples — appended here]
            </examples>
            <output_format>

        Args:
            ocr_text: OCR prediction text to correct.
            insights_context: Pre-formatted Arabic insights string from
                training analysis.  May be empty.
            word_pairs_context: Pre-formatted Arabic word-pair string from
                training analysis (UNFIXED errors).  Optional.
            overcorrection_context: Pre-formatted warnings about common
                over-corrections the LLM makes (INTRODUCED errors).  Optional.
            few_shot_examples: Optional pre-formatted INPUT/OUTPUT examples from
                ``FewShotExampleSelector.format_for_prompt()``.  Appended inside
                the ``<examples>`` block after the static examples.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        has_insights = bool(insights_context.strip())
        has_pairs = bool(word_pairs_context.strip())
        has_overcorrections = bool(overcorrection_context.strip())
        if not has_insights and not has_pairs and not has_overcorrections:
            return self.build_zero_shot(ocr_text, few_shot_examples=few_shot_examples)
        base = self._load_crafted_base()
        inner_parts: list[str] = []
        if has_insights:
            inner_parts.append(
                "SYSTEMATIC WEAKNESSES — error types this model frequently misses "
                "(apply extra caution to these):\n" + insights_context
            )
        if has_pairs:
            inner_parts.append(
                "KNOWN WORD-LEVEL ERRORS from training data "
                "(OCR form → correct form):\n" + word_pairs_context
            )
        if has_overcorrections:
            inner_parts.append(
                "OVER-CORRECTION TRAPS — DO NOT make these changes "
                "(this model incorrectly applies them):\n" + overcorrection_context
            )
        section = (
            "<self_analysis>\n"
            "Based on analysis of this model's corrections on similar training texts:\n\n"
            + "\n\n".join(inner_parts)
            + "\n</self_analysis>"
        )
        system = self._inject_knowledge(base, section)
        if few_shot_examples.strip():
            system = self._inject_dynamic_examples(system, few_shot_examples)
        return self._messages(system, ocr_text)

    @property
    def self_reflective_prompt_version(self) -> str:
        """Phase 4 prompt version string."""
        return self.SELF_REFLECTIVE_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 6 — Combined Prompting (was Phase 5)
    # -----------------------------------------------------------------------

    def build_combined(
        self,
        ocr_text: str,
        confusion_context: str = "",
        insights_context: str = "",
        word_pairs_context: str = "",
        overcorrection_context: str = "",
        few_shot_examples: str = "",
        word_examples: str = "",
    ) -> list[dict]:
        """Build a combined correction prompt (Phase 6).

        Uses the crafted system prompt as the base and injects any non-empty
        context sources as XML sections, following the prompt's tag structure.

        Falls back to zero-shot if all contexts are empty.

        Injection order in the final system prompt::

            <error_patterns>       (from crafted base)
            <confusion_patterns>   (Phase 3 confusion matrix + word-level failures — if provided)
            <self_analysis>        (Phase 4 insights + word pairs + warnings — if provided)
            <examples>
              [static examples from crafted base]
              [dynamic few-shot examples — appended here if provided]
            </examples>
            <output_format>

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted confusion pairs (Phase 3 format).
            insights_context: Pre-formatted self-reflective insights (Phase 4).
            word_pairs_context: Pre-formatted word-error pairs (Phase 4).
            overcorrection_context: Pre-formatted over-correction warnings (Phase 4).
            few_shot_examples: Optional pre-formatted INPUT/OUTPUT examples from
                ``FewShotExampleSelector.format_for_prompt()``.
            word_examples: Optional pre-formatted word-level failure examples
                from training cross-reference.  When non-empty and
                *confusion_context* is also non-empty, injected inside the
                ``<confusion_patterns>`` section (same as Phase 3 does).

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        all_empty = not any([
            confusion_context.strip(),
            insights_context.strip(),
            word_pairs_context.strip(),
            overcorrection_context.strip(),
        ])
        if all_empty:
            return self.build_zero_shot(ocr_text, few_shot_examples=few_shot_examples)

        base = self._load_crafted_base()

        # Inject knowledge sections before <examples> (in order).
        if confusion_context.strip():
            section = self._COMBINED_CONFUSION.format(confusion_context=confusion_context)
            if word_examples.strip():
                section = section.replace(
                    "</confusion_patterns>",
                    "\n\nWORD-LEVEL FAILURES — common whole-word misreads from training data "
                    "(OCR form → correct form):\n" + word_examples + "\n</confusion_patterns>",
                )
            base = self._inject_knowledge(base, section)
        # Bundle Phase 4 signals into one <self_analysis> section.
        has_insights = bool(insights_context.strip())
        has_pairs = bool(word_pairs_context.strip())
        has_overcorrections = bool(overcorrection_context.strip())
        if has_insights or has_pairs or has_overcorrections:
            inner: list[str] = []
            if has_insights:
                inner.append(
                    "SYSTEMATIC WEAKNESSES — error types this model frequently misses "
                    "(apply extra caution to these):\n" + insights_context
                )
            if has_pairs:
                inner.append(
                    "KNOWN WORD-LEVEL ERRORS from training data "
                    "(OCR form → correct form):\n" + word_pairs_context
                )
            if has_overcorrections:
                inner.append(
                    "OVER-CORRECTION TRAPS — DO NOT make these changes "
                    "(this model incorrectly applies them):\n" + overcorrection_context
                )
            section = (
                "<self_analysis>\n"
                "Based on analysis of this model's corrections on similar training texts:\n\n"
                + "\n\n".join(inner)
                + "\n</self_analysis>"
            )
            base = self._inject_knowledge(base, section)

        if few_shot_examples.strip():
            base = self._inject_dynamic_examples(base, few_shot_examples)

        return self._messages(base, ocr_text)

    @property
    def combined_prompt_version(self) -> str:
        """Phase 6 prompt version string."""
        return self.COMBINED_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 8 — RAG (Retrieval-Augmented Generation)
    # -----------------------------------------------------------------------

    def build_rag(
        self,
        ocr_text: str,
        retrieved_sentences: str = "",
        retrieved_words: str = "",
        few_shot_examples: str = "",
    ) -> list[dict]:
        """Build Phase 8 RAG correction prompt.

        Injects per-sample retrieved corrections (sentences and/or word
        fixes) into the crafted system prompt.  Falls back to zero-shot
        if both retrieval contexts are empty.

        Injection order in the final system prompt::

            <error_patterns>          (from crafted base)
            <retrieved_corrections>   (similar sentence corrections)
            <retrieved_word_fixes>    (similar word-level corrections)
            <examples>
              [static examples from crafted base]
              [dynamic few-shot examples — appended here if provided]
            </examples>
            <output_format>

        Args:
            ocr_text: OCR prediction text to correct.
            retrieved_sentences: Pre-formatted INPUT/OUTPUT pairs from
                ``RAGRetriever.format_sentences_for_prompt()``.
            retrieved_words: Pre-formatted word corrections from
                ``RAGRetriever.format_words_for_prompt()``.
            few_shot_examples: Optional pre-formatted INPUT/OUTPUT examples
                from ``FewShotExampleSelector.format_for_prompt()``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        has_sentences = bool(retrieved_sentences.strip())
        has_words = bool(retrieved_words.strip())

        if not has_sentences and not has_words:
            return self.build_zero_shot(ocr_text, few_shot_examples=few_shot_examples)

        base = self._load_crafted_base()

        if has_sentences:
            section = self._RAG_SENTENCES_SECTION.format(
                retrieved_sentences=retrieved_sentences,
            )
            base = self._inject_knowledge(base, section)

        if has_words:
            section = self._RAG_WORDS_SECTION.format(
                retrieved_words=retrieved_words,
            )
            base = self._inject_knowledge(base, section)

        if few_shot_examples.strip():
            base = self._inject_dynamic_examples(base, few_shot_examples)

        return self._messages(base, ocr_text)

    @property
    def rag_prompt_version(self) -> str:
        """Phase 8 prompt version string."""
        return self.RAG_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Crafted Prompt (standalone — used by scripts/craft_prompt.py)
    # -----------------------------------------------------------------------

    def build_crafted(self, ocr_text: str, system_prompt: str) -> list[dict]:
        """Build correction prompt using an externally supplied system prompt.

        Used for LLM-designed prompts produced by ``scripts/craft_prompt.py``.
        Falls back to zero-shot v2 if *system_prompt* is empty.

        Args:
            ocr_text: OCR prediction text to correct.
            system_prompt: The crafted Arabic system prompt text.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not system_prompt.strip():
            return self.build_zero_shot(ocr_text, version="v2")
        return self._messages(system_prompt, ocr_text)

    @property
    def crafted_prompt_version(self) -> str:
        """Crafted-prompt version string."""
        return self.CRAFTED_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Meta-prompting (prompt generation via scripts/craft_prompt.py)
    # -----------------------------------------------------------------------

    def build_meta_prompt(self, user_prompt: str) -> list[dict]:
        """Build messages for a meta-prompting task (e.g. prompt generation).

        No system message — the full meta-prompt is sent as the user turn so
        the model follows the embedded instructions directly.

        Args:
            user_prompt: The full meta-prompt text (from craft_prompt.py).

        Returns:
            Single-element messages list in OpenAI chat format.
        """
        return [{"role": "user", "content": user_prompt}]

    @property
    def meta_prompt_version(self) -> str:
        """Meta-prompt version string."""
        return self.META_PROMPT_VERSION
