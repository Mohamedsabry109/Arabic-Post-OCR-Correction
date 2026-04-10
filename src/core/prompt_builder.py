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

    PROMPT_VERSION: str = "p2v2"                  # Phase 2 — crafted base
    OCR_AWARE_PROMPT_VERSION: str = "p3v2"         # Phase 3
    SELF_REFLECTIVE_PROMPT_VERSION: str = "p4v1"    # Phase 4 (insights + word pairs)
    COMBINED_PROMPT_VERSION: str = "p6v1"          # Phase 6
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
    # The prompt uses XML-like section tags; injections follow that pattern.
    # -----------------------------------------------------------------------

    # Knowledge sections (phases 3, 4) are inserted immediately before
    # the <examples> block so the model sees them before the few-shot pairs.
    _INJECT_ANCHOR: str = "<examples>"

    # -----------------------------------------------------------------------
    # Per-phase XML section templates.
    # Section wrappers / labels are in English; Arabic content stays Arabic.
    # -----------------------------------------------------------------------

    _OCR_AWARE_SECTION: str = (
        "<confusion_patterns>\n"
        "Character confusions specific to this OCR system (corrupted → correct):\n"
        "{confusion_context}\n"
        "</confusion_patterns>"
    )

    # Combined-phase section body for confusion (same as standalone).
    _COMBINED_CONFUSION: str = (
        "<confusion_patterns>\n"
        "Character confusions specific to this OCR system:\n"
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

        Falls back to appending at the end if the anchor is absent (e.g. when
        the crafted base prompt is not available and zero-shot v2 is used as
        fallback).
        """
        if self._INJECT_ANCHOR in base:
            return base.replace(
                self._INJECT_ANCHOR,
                section + "\n\n" + self._INJECT_ANCHOR,
                1,
            )
        return base.rstrip() + "\n\n" + section

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

    def build_ocr_aware(
        self,
        ocr_text: str,
        confusion_context: str,
        word_examples: str = "",
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

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not confusion_context.strip():
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        section = self._OCR_AWARE_SECTION.format(confusion_context=confusion_context)
        if word_examples.strip():
            # Insert word examples inside the confusion_patterns section
            section = section.replace(
                "</confusion_patterns>",
                "\n\n" + word_examples + "\n</confusion_patterns>",
            )
        return self._messages(self._inject_knowledge(base, section), ocr_text)

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
    ) -> list[dict]:
        """Build Phase 4 correction prompt: self-reflective insights + word-error pairs.

        Injects all signals as a single ``<self_analysis>`` section before
        ``<examples>``.  Falls back to zero-shot if all contexts are empty.

        Args:
            ocr_text: OCR prediction text to correct.
            insights_context: Pre-formatted Arabic insights string from
                training analysis.  May be empty.
            word_pairs_context: Pre-formatted Arabic word-pair string from
                training analysis (UNFIXED errors).  Optional.
            overcorrection_context: Pre-formatted warnings about common
                over-corrections the LLM makes (INTRODUCED errors).  Optional.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        has_insights = bool(insights_context.strip())
        has_pairs = bool(word_pairs_context.strip())
        has_overcorrections = bool(overcorrection_context.strip())
        if not has_insights and not has_pairs and not has_overcorrections:
            return self.build_zero_shot(ocr_text)
        base = self._load_crafted_base()
        inner_parts: list[str] = []
        if has_insights:
            inner_parts.append(
                "Analysis of previous correction patterns on similar texts:\n"
                + insights_context
            )
        if has_pairs:
            inner_parts.append(
                "Common word-level OCR errors found in training data:\n"
                + word_pairs_context
            )
        if has_overcorrections:
            inner_parts.append(
                "WARNING - Common over-corrections to AVOID (do NOT make these changes):\n"
                + overcorrection_context
            )
        section = (
            "<self_analysis>\n"
            + "\n\n".join(inner_parts)
            + "\n</self_analysis>"
        )
        return self._messages(self._inject_knowledge(base, section), ocr_text)

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
    ) -> list[dict]:
        """Build a combined correction prompt (Phase 6).

        Uses the crafted system prompt as the base and injects any non-empty
        context sources as XML sections, following the prompt's tag structure.

        Falls back to zero-shot if all contexts are empty.

        Injection order (all appear before ``<examples>``):
            1. ``<confusion_patterns>``  — Phase 3 confusion matrix
            2. ``<self_analysis>``       — Phase 4 insights + word pairs + overcorrection warnings

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted confusion pairs (Phase 3 format).
            insights_context: Pre-formatted self-reflective insights (Phase 4).
            word_pairs_context: Pre-formatted word-error pairs (Phase 4).
            overcorrection_context: Pre-formatted over-correction warnings (Phase 4).

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
            return self.build_zero_shot(ocr_text)

        base = self._load_crafted_base()

        # Inject knowledge sections before <examples> (in order).
        if confusion_context.strip():
            section = self._COMBINED_CONFUSION.format(confusion_context=confusion_context)
            base = self._inject_knowledge(base, section)
        # Bundle Phase 4 signals into one <self_analysis> section.
        has_insights = bool(insights_context.strip())
        has_pairs = bool(word_pairs_context.strip())
        has_overcorrections = bool(overcorrection_context.strip())
        if has_insights or has_pairs or has_overcorrections:
            inner: list[str] = []
            if has_insights:
                inner.append(
                    "Analysis of previous correction patterns on similar texts:\n"
                    + insights_context
                )
            if has_pairs:
                inner.append(
                    "Common word-level OCR errors found in training data:\n"
                    + word_pairs_context
                )
            if has_overcorrections:
                inner.append(
                    "WARNING - Common over-corrections to AVOID (do NOT make these changes):\n"
                    + overcorrection_context
                )
            section = (
                "<self_analysis>\n"
                + "\n\n".join(inner)
                + "\n</self_analysis>"
            )
            base = self._inject_knowledge(base, section)

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
