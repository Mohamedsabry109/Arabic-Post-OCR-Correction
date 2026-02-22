"""Phase-specific prompt construction for LLM correction.

All methods return OpenAI chat format (list[dict]) so they are compatible
with both HuggingFace tokenizer.apply_chat_template() and API calls.
"""


class PromptBuilder:
    """Build phase-specific chat message lists for LLM correction.

    Each build_* method returns list[dict] in OpenAI chat format and can be
    passed directly to any BaseLLMCorrector implementation.

    Phase 2 uses build_zero_shot() only.
    Later phases add build_ocr_aware(), build_rule_augmented(), etc.

    Usage::

        builder = PromptBuilder()
        messages = builder.build_zero_shot(ocr_text="النص المدخل...")
        result = corrector.correct(sample_id, ocr_text, messages)
    """

    # -----------------------------------------------------------------------
    # Prompt constants — versioned so changes are detectable in results.
    # Increment PROMPT_VERSION whenever any prompt wording changes between runs.
    # -----------------------------------------------------------------------

    ZERO_SHOT_SYSTEM_V1: str = (
        "أنت مصحح نصوص عربية متخصص. "
        "مهمتك تصحيح أخطاء التعرف الضوئي (OCR) في النص العربي. "
        "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
    )

    PROMPT_VERSION: str = "v1"

    def build_zero_shot(self, ocr_text: str) -> list[dict]:
        """Build zero-shot correction prompt (Phase 2).

        No task-specific knowledge injected. The LLM receives a general
        Arabic OCR correction instruction and the raw text to correct.

        Args:
            ocr_text: OCR prediction text to correct.

        Returns:
            Two-element messages list::

                [
                  {"role": "system", "content": <Arabic system prompt>},
                  {"role": "user",   "content": ocr_text}
                ]
        """
        return [
            {"role": "system", "content": self.ZERO_SHOT_SYSTEM_V1},
            {"role": "user",   "content": ocr_text},
        ]

    @property
    def prompt_version(self) -> str:
        """Return the current prompt version string (included in all metadata)."""
        return self.PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 3 — OCR-Aware Prompting
    # -----------------------------------------------------------------------

    OCR_AWARE_SYSTEM_V1: str = (
        "أنت مصحح نصوص عربية متخصص في تصحيح مخرجات نظام Qaari للتعرف الضوئي. "
        "فيما يلي أبرز الأخطاء الشائعة التي يرتكبها هذا النظام:\n\n"
        "{confusion_context}\n\n"
        "صحح النص التالي مع الانتباه بشكل خاص لهذه الأخطاء. "
        "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
    )

    OCR_AWARE_PROMPT_VERSION: str = "p3v1"

    def build_ocr_aware(self, ocr_text: str, confusion_context: str) -> list[dict]:
        """Build OCR-aware correction prompt (Phase 3).

        Injects Qaari's top-N character confusion pairs into the system prompt.
        If confusion_context is empty, falls back to the zero-shot prompt.

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted Arabic string describing confusion
                pairs (produced by ConfusionMatrixLoader.format_for_prompt()).
                Pass empty string to trigger zero-shot fallback.

        Returns:
            Two-element messages list in OpenAI chat format::

                [
                  {"role": "system", "content": <system with confusion list>},
                  {"role": "user",   "content": ocr_text}
                ]
        """
        if not confusion_context.strip():
            # No confusion data available — fall back gracefully to zero-shot
            return self.build_zero_shot(ocr_text)

        system = self.OCR_AWARE_SYSTEM_V1.format(confusion_context=confusion_context)
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    @property
    def ocr_aware_prompt_version(self) -> str:
        """Return the Phase 3 prompt version string."""
        return self.OCR_AWARE_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 4A — Rule-Augmented Prompting
    # -----------------------------------------------------------------------

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

        Injects Arabic orthographic rules into the system prompt. If
        rules_context is empty, falls back to the zero-shot prompt.

        Args:
            ocr_text: OCR prediction text to correct.
            rules_context: Pre-formatted Arabic rules string
                (produced by RulesLoader.format_for_prompt()).
                Pass empty string to trigger zero-shot fallback.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not rules_context.strip():
            return self.build_zero_shot(ocr_text)

        system = self.RULES_SYSTEM_V1.format(rules_context=rules_context)
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    @property
    def rules_prompt_version(self) -> str:
        """Return the Phase 4A prompt version string."""
        return self.RULES_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 4B — Few-Shot Prompting (QALB)
    # -----------------------------------------------------------------------

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

        Injects QALB error-correction examples into the system prompt. If
        examples_context is empty, falls back to the zero-shot prompt.

        Args:
            ocr_text: OCR prediction text to correct.
            examples_context: Pre-formatted few-shot examples string
                (produced by QALBLoader.format_for_prompt()).
                Pass empty string to trigger zero-shot fallback.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not examples_context.strip():
            return self.build_zero_shot(ocr_text)

        system = self.FEWSHOT_SYSTEM_V1.format(examples_context=examples_context)
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    @property
    def few_shot_prompt_version(self) -> str:
        """Return the Phase 4B prompt version string."""
        return self.FEWSHOT_PROMPT_VERSION
