"""Phase-specific prompt construction for LLM correction.

All methods return OpenAI chat format (list[dict]) so they are compatible
with both HuggingFace tokenizer.apply_chat_template() and API calls.

Architecture
------------
Every phase uses ``results/prompt_craft/crafted_system_prompt.txt`` as its
base system prompt.  Phase-specific knowledge (confusion matrix, rules,
few-shot pairs, RAG passages, self-reflective insights) is injected into
that base rather than replacing it.

Injection strategy
~~~~~~~~~~~~~~~~~~
* **Knowledge sections** (phases 3, 4A, 4D, 5, 6): inserted as a new
  numbered rule (7, 8, …) immediately before the
  ``القيود الصارمة للمخرجات`` (output constraints) block.
* **Extra examples** (phase 4B, and the examples slot of phase 6):
  appended after the trailing ``---`` examples section.
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
    context is layered on top via :meth:`_inject_knowledge` (rules block) or
    :meth:`_append_examples` (examples block).

    Usage::

        builder = PromptBuilder()                          # default path
        builder = PromptBuilder("path/to/system.txt")     # custom path
        messages = builder.build_ocr_aware(ocr_text, confusion_context)
        result = corrector.correct(sample_id, ocr_text, messages)
    """

    # Default path to the crafted system prompt (relative to project root).
    DEFAULT_CRAFTED_PROMPT_PATH: str = (
        "data/crafted_system_prompt.txt"
    )

    # -----------------------------------------------------------------------
    # Prompt version strings — bump whenever wording changes between runs.
    # -----------------------------------------------------------------------

    PROMPT_VERSION: str = "p2v2"                  # Phase 2 — crafted base
    OCR_AWARE_PROMPT_VERSION: str = "p3v2"         # Phase 3
    RULES_PROMPT_VERSION: str = "p4av2"            # Phase 4A
    FEWSHOT_PROMPT_VERSION: str = "p4bv2"          # Phase 4B
    SELF_REFLECTIVE_PROMPT_VERSION: str = "p4dv2"  # Phase 4D
    RAG_PROMPT_VERSION: str = "p5v2"               # Phase 5
    COMBINED_PROMPT_VERSION: str = "p6v2"          # Phase 6
    CRAFTED_PROMPT_VERSION: str = "crafted_v1"     # standalone crafted
    META_PROMPT_VERSION: str = "meta_v1"           # meta-prompting

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
    # -----------------------------------------------------------------------

    # Text that starts the "output constraints" block — knowledge sections
    # are inserted immediately before this.
    _INJECT_ANCHOR: str = "القيود الصارمة للمخرجات"

    # Separator between the rules block and the few-shot examples block.
    _EXAMPLES_SEPARATOR: str = "\n---\n"

    # -----------------------------------------------------------------------
    # Per-phase section headers (Arabic, matching the crafted prompt's style)
    # -----------------------------------------------------------------------

    _OCR_AWARE_SECTION: str = (
        "7. أخطاء الارتباك الخاصة بنظام Qaari (مستخلصة من مصفوفة الارتباك):\n"
        "{confusion_context}"
    )

    _RULES_SECTION: str = (
        "7. قواعد إملائية عربية إضافية يجب مراعاتها:\n"
        "{rules_context}"
    )

    _SELF_REFLECTIVE_SECTION: str = (
        "7. ملاحظات مستخلصة من تحليل أخطاء سابقة في التصحيح:\n"
        "{insights_context}"
    )

    _RAG_SECTION: str = (
        "7. نصوص عربية مرجعية مشابهة (مسترجعة من مصادر عربية موثوقة):\n"
        "{retrieval_context}"
    )

    _FEWSHOT_EXTRA_HEADER: str = (
        "أمثلة إضافية من بيانات التدريب (QALB):\n\n{examples_context}"
    )

    # Combined-phase section bodies (numbered dynamically in build_combined).
    _COMBINED_CONFUSION: str = (
        "أخطاء ارتباك نظام Qaari:\n{confusion_context}"
    )
    _COMBINED_RULES: str = (
        "قواعد إملائية عربية إضافية:\n{rules_context}"
    )
    _COMBINED_RETRIEVAL: str = (
        "نصوص مرجعية مشابهة:\n{retrieval_context}"
    )
    _COMBINED_INSIGHTS: str = (
        "ملاحظات من تحليل أخطاء سابقة:\n{insights_context}"
    )
    _COMBINED_EXAMPLES_HEADER: str = (
        "أمثلة إضافية من بيانات التدريب (QALB):\n\n{examples_context}"
    )

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
            try:
                with open(self._crafted_prompt_path, encoding="utf-8") as f:
                    self._crafted_base = f.read()
            except FileNotFoundError:
                logger.warning(
                    "Crafted prompt not found at '%s' — falling back to zero-shot v2.",
                    self._crafted_prompt_path,
                )
                self._crafted_base = self.ZERO_SHOT_SYSTEM_V2
        return self._crafted_base

    def _inject_knowledge(self, base: str, section: str) -> str:
        """Insert *section* just before the output-constraints block.

        Falls back to injecting before the ``---`` separator, or appending
        at the end, if the anchor text is absent.
        """
        if self._INJECT_ANCHOR in base:
            return base.replace(
                self._INJECT_ANCHOR,
                section + "\n\n" + self._INJECT_ANCHOR,
                1,
            )
        if self._EXAMPLES_SEPARATOR in base:
            return base.replace(
                self._EXAMPLES_SEPARATOR,
                "\n\n" + section + self._EXAMPLES_SEPARATOR,
                1,
            )
        return base + "\n\n" + section

    def _append_examples(self, base: str, extra: str) -> str:
        """Append *extra* to the very end of the prompt."""
        return base.rstrip() + "\n\n" + extra

    def _messages(self, system: str, ocr_text: str) -> list[dict]:
        """Return standard two-element OpenAI chat messages."""
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    # -----------------------------------------------------------------------
    # Phase 2 — Zero-Shot
    # -----------------------------------------------------------------------

    def build_zero_shot(self, ocr_text: str, version: str = "crafted") -> list[dict]:
        """Build zero-shot correction prompt (Phase 2).

        Uses the crafted system prompt as-is by default.  Pass ``version="v1"``
        or ``version="v2"`` to use legacy prompts when re-running old exports.

        Args:
            ocr_text: OCR prediction text to correct.
            version: ``"crafted"`` (default), ``"v1"``, or ``"v2"``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if version in self._ZERO_SHOT_VERSIONS:
            system = self._ZERO_SHOT_VERSIONS[version]
        else:
            system = self._load_crafted_base()
        return self._messages(system, ocr_text)

    @property
    def prompt_version(self) -> str:
        """Phase 2 prompt version string."""
        return self.PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 3 — OCR-Aware Prompting
    # -----------------------------------------------------------------------

    def build_ocr_aware(self, ocr_text: str, confusion_context: str) -> list[dict]:
        """Build OCR-aware correction prompt (Phase 3).

        Injects Qaari's top-N character confusion pairs as a new numbered rule
        (rule 7) into the crafted system prompt, immediately before the output
        constraints block.  Falls back to zero-shot if *confusion_context* is
        empty.

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted Arabic confusion-pairs string
                produced by ``ConfusionMatrixLoader.format_for_prompt()``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not confusion_context.strip():
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        section = self._OCR_AWARE_SECTION.format(confusion_context=confusion_context)
        return self._messages(self._inject_knowledge(base, section), ocr_text)

    @property
    def ocr_aware_prompt_version(self) -> str:
        """Phase 3 prompt version string."""
        return self.OCR_AWARE_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 4A — Rule-Augmented Prompting
    # -----------------------------------------------------------------------

    def build_rule_augmented(self, ocr_text: str, rules_context: str) -> list[dict]:
        """Build rule-augmented correction prompt (Phase 4A).

        Injects Arabic orthographic rules as rule 7 into the crafted system
        prompt.  Falls back to zero-shot if *rules_context* is empty.

        Args:
            ocr_text: OCR prediction text to correct.
            rules_context: Pre-formatted Arabic rules string produced by
                ``RulesLoader.format_for_prompt()``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not rules_context.strip():
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        section = self._RULES_SECTION.format(rules_context=rules_context)
        return self._messages(self._inject_knowledge(base, section), ocr_text)

    @property
    def rules_prompt_version(self) -> str:
        """Phase 4A prompt version string."""
        return self.RULES_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 4B — Few-Shot Learning (QALB)
    # -----------------------------------------------------------------------

    def build_few_shot(self, ocr_text: str, examples_context: str) -> list[dict]:
        """Build few-shot correction prompt (Phase 4B).

        Appends QALB error-correction examples after the crafted prompt's
        existing examples section (i.e. after the ``---`` separator).  Falls
        back to zero-shot if *examples_context* is empty.

        Args:
            ocr_text: OCR prediction text to correct.
            examples_context: Pre-formatted QALB correction pairs produced by
                ``QALBLoader.format_for_prompt()``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not examples_context.strip():
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        extra = self._FEWSHOT_EXTRA_HEADER.format(examples_context=examples_context)
        return self._messages(self._append_examples(base, extra), ocr_text)

    @property
    def few_shot_prompt_version(self) -> str:
        """Phase 4B prompt version string."""
        return self.FEWSHOT_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 4D — Self-Reflective Prompting
    # -----------------------------------------------------------------------

    def build_self_reflective(
        self, ocr_text: str, insights_context: str
    ) -> list[dict]:
        """Build self-reflective correction prompt (Phase 4D).

        Injects insights about the model's own failure patterns (derived from
        training-split predictions) as rule 7.  Falls back to zero-shot if
        *insights_context* is empty.

        Args:
            ocr_text: OCR prediction text to correct.
            insights_context: Pre-formatted Arabic insights string produced by
                ``LLMInsightsLoader.format_for_prompt()``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not insights_context.strip():
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        section = self._SELF_REFLECTIVE_SECTION.format(insights_context=insights_context)
        return self._messages(self._inject_knowledge(base, section), ocr_text)

    @property
    def self_reflective_prompt_version(self) -> str:
        """Phase 4D prompt version string."""
        return self.SELF_REFLECTIVE_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 5 — RAG (OpenITI corpus retrieval)
    # -----------------------------------------------------------------------

    def build_rag(self, ocr_text: str, retrieval_context: str) -> list[dict]:
        """Build RAG-augmented correction prompt (Phase 5).

        Injects retrieved OpenITI sentences as rule 7 into the crafted system
        prompt.  Falls back to zero-shot if *retrieval_context* is empty.

        Args:
            ocr_text: OCR prediction text to correct.
            retrieval_context: Pre-formatted Arabic retrieved-sentences string
                produced by ``RAGRetriever.format_for_prompt()``.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not retrieval_context.strip():
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        section = self._RAG_SECTION.format(retrieval_context=retrieval_context)
        return self._messages(self._inject_knowledge(base, section), ocr_text)

    @property
    def rag_prompt_version(self) -> str:
        """Phase 5 prompt version string."""
        return self.RAG_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 6 — Combined Prompting
    # -----------------------------------------------------------------------

    def build_combined(
        self,
        ocr_text: str,
        confusion_context: str = "",
        rules_context: str = "",
        examples_context: str = "",
        retrieval_context: str = "",
        insights_context: str = "",
    ) -> list[dict]:
        """Build a combined correction prompt (Phase 6).

        Uses the crafted system prompt as the base and injects any non-empty
        context sources.  Knowledge sections (confusion, rules, retrieval,
        insights) are numbered 7, 8, … and inserted before the output
        constraints block.  QALB examples are appended at the end.

        Falls back to zero-shot if all contexts are empty.

        Injection order:
            1. Confusion matrix (Phase 3)
            2. Spelling rules   (Phase 4A)
            3. RAG passages     (Phase 5)
            4. Self-reflective  (Phase 4D)
            5. QALB examples    (Phase 4B — appended, not injected)

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted confusion pairs (Phase 3 format).
            rules_context: Pre-formatted Arabic rules (Phase 4A format).
            examples_context: Pre-formatted few-shot pairs (Phase 4B format).
            retrieval_context: Pre-formatted retrieved sentences (Phase 5 format).
            insights_context: Pre-formatted self-reflective insights (Phase 4D format).

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        all_empty = not any([
            confusion_context.strip(),
            rules_context.strip(),
            examples_context.strip(),
            retrieval_context.strip(),
            insights_context.strip(),
        ])
        if all_empty:
            return self.build_zero_shot(ocr_text)

        base = self._load_crafted_base()

        # Build numbered knowledge sections (starting at 7).
        parts: list[str] = []
        num = 7
        if confusion_context.strip():
            body = self._COMBINED_CONFUSION.format(confusion_context=confusion_context)
            parts.append(f"{num}. {body}")
            num += 1
        if rules_context.strip():
            body = self._COMBINED_RULES.format(rules_context=rules_context)
            parts.append(f"{num}. {body}")
            num += 1
        if retrieval_context.strip():
            body = self._COMBINED_RETRIEVAL.format(retrieval_context=retrieval_context)
            parts.append(f"{num}. {body}")
            num += 1
        if insights_context.strip():
            body = self._COMBINED_INSIGHTS.format(insights_context=insights_context)
            parts.append(f"{num}. {body}")

        if parts:
            base = self._inject_knowledge(base, "\n\n".join(parts))

        # QALB examples go after the trailing examples section.
        if examples_context.strip():
            extra = self._COMBINED_EXAMPLES_HEADER.format(
                examples_context=examples_context
            )
            base = self._append_examples(base, extra)

        return self._messages(base, ocr_text)

    @property
    def combined_prompt_version(self) -> str:
        """Phase 6 prompt version string."""
        return self.COMBINED_PROMPT_VERSION

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
