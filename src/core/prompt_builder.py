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
