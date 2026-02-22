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

    # -----------------------------------------------------------------------
    # Phase 5 — RAG (OpenITI corpus retrieval)
    # -----------------------------------------------------------------------

    RAG_SYSTEM_V1: str = (
        "أنت مصحح نصوص عربية متخصص. "
        "فيما يلي نصوص عربية صحيحة مشابهة للنص المراد تصحيحه، استخدمها كمرجع:\n\n"
        "{retrieval_context}\n\n"
        "صحح النص التالي مستعيناً بهذه النصوص المرجعية. "
        "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
    )

    RAG_PROMPT_VERSION: str = "p5v1"

    def build_rag(self, ocr_text: str, retrieval_context: str) -> list[dict]:
        """Build RAG-augmented correction prompt (Phase 5).

        Injects retrieved OpenITI sentences as reference context. If
        retrieval_context is empty, falls back to the zero-shot prompt.

        Args:
            ocr_text: OCR prediction text to correct.
            retrieval_context: Pre-formatted Arabic string of retrieved
                sentences (produced by RAGRetriever.format_for_prompt()).
                Pass empty string to trigger zero-shot fallback.

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        if not retrieval_context.strip():
            return self.build_zero_shot(ocr_text)

        system = self.RAG_SYSTEM_V1.format(retrieval_context=retrieval_context)
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    @property
    def rag_prompt_version(self) -> str:
        """Return the Phase 5 prompt version string."""
        return self.RAG_PROMPT_VERSION

    # -----------------------------------------------------------------------
    # Phase 6 — Combined Prompting
    # -----------------------------------------------------------------------

    # Section headers and footers for the combined system prompt.
    # Only non-empty context sections are included in the final prompt.

    _COMBINED_HEADER: str = (
        "أنت مصحح نصوص عربية متخصص. "
        "استخدم المعلومات التالية لتصحيح أخطاء التعرف الضوئي:\n\n"
    )

    _COMBINED_SECTION_CONFUSION: str = (
        "أولاً: أخطاء شائعة في هذا النظام:\n{confusion_context}"
    )

    _COMBINED_SECTION_RULES: str = (
        "ثانياً: قواعد إملائية مهمة:\n{rules_context}"
    )

    _COMBINED_SECTION_EXAMPLES: str = (
        "ثالثاً: أمثلة على التصحيح:\n{examples_context}"
    )

    _COMBINED_SECTION_RETRIEVAL: str = (
        "رابعاً: نصوص مرجعية صحيحة مشابهة:\n{retrieval_context}"
    )

    _COMBINED_FOOTER: str = (
        "\n\nصحح النص التالي. أعد النص المصحح فقط بدون أي شرح أو تعليق."
    )

    COMBINED_PROMPT_VERSION: str = "p6v1"

    def build_combined(
        self,
        ocr_text: str,
        confusion_context: str = "",
        rules_context: str = "",
        examples_context: str = "",
        retrieval_context: str = "",
    ) -> list[dict]:
        """Build a combined correction prompt including any subset of context types.

        Includes only non-empty contexts in a fixed order:
        1. Confusion context  (OCR error pairs from Phase 3)
        2. Rules context      (Arabic spelling rules from Phase 4A)
        3. Examples context   (few-shot correction pairs from Phase 4B)
        4. Retrieval context  (similar correct sentences from Phase 5 / OpenITI)

        Falls back to zero-shot if all context strings are empty or whitespace.

        Args:
            ocr_text: OCR prediction text to correct.
            confusion_context: Pre-formatted confusion pairs (Phase 3 format).
            rules_context: Pre-formatted Arabic rules (Phase 4A format).
            examples_context: Pre-formatted few-shot examples (Phase 4B format).
            retrieval_context: Pre-formatted retrieved sentences (Phase 5 format).

        Returns:
            Two-element messages list in OpenAI chat format.
        """
        sections: list[str] = []

        if confusion_context.strip():
            sections.append(
                self._COMBINED_SECTION_CONFUSION.format(confusion_context=confusion_context)
            )
        if rules_context.strip():
            sections.append(
                self._COMBINED_SECTION_RULES.format(rules_context=rules_context)
            )
        if examples_context.strip():
            sections.append(
                self._COMBINED_SECTION_EXAMPLES.format(examples_context=examples_context)
            )
        if retrieval_context.strip():
            sections.append(
                self._COMBINED_SECTION_RETRIEVAL.format(retrieval_context=retrieval_context)
            )

        if not sections:
            return self.build_zero_shot(ocr_text)

        system = (
            self._COMBINED_HEADER
            + "\n\n".join(sections)
            + self._COMBINED_FOOTER
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": ocr_text},
        ]

    @property
    def combined_prompt_version(self) -> str:
        """Return the Phase 6 combined prompt version string."""
        return self.COMBINED_PROMPT_VERSION
