"""Arabic word morphological validation using CAMeL Tools.

Provides WordValidator which checks individual tokens, aggregates
validity rates across full texts, and applies a revert strategy for
Phase 4C post-processing correction.
"""

import logging
from dataclasses import dataclass, field

from src.data.text_utils import tokenise_arabic, is_arabic_word
from src.linguistic.morphology import MorphAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Morphological validation result for a single word token."""

    word: str
    is_valid: bool           # has >= 1 morphological analysis
    analyses_count: int      # number of analyses (0 means invalid/unknown)


@dataclass
class TextCorrectionResult:
    """Outcome of applying the revert strategy to one LLM-corrected text.

    Used by WordValidator.validate_correction() in Phase 4C.
    """

    final_text: str              # Text after applying the revert strategy
    total_words: int             # Arabic tokens compared
    reverted_count: int          # Words reverted from LLM back to OCR
    kept_count: int              # Words kept as LLM correction
    unchanged_count: int         # Words identical in both OCR and LLM output
    revert_rate: float           # reverted_count / total_words (0.0 if no words)
    reverted_words: list[str] = field(default_factory=list)  # words that were reverted


class WordValidator:
    """Validate Arabic word tokens using morphological analysis.

    Usage::

        analyzer = MorphAnalyzer()
        validator = WordValidator(analyzer)
        results = validator.validate_text("الكتاب على المكتبة")
        rate = validator.validity_rate("الكتاب على المكتبة")
    """

    def __init__(self, analyzer: MorphAnalyzer) -> None:
        """Initialise with a MorphAnalyzer instance.

        Args:
            analyzer: Initialised MorphAnalyzer. Can be disabled (no-op).
        """
        self._analyzer = analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_word(self, word: str) -> ValidationResult:
        """Validate a single word token.

        Args:
            word: A single Arabic word string.

        Returns:
            ValidationResult with is_valid and analyses_count.
        """
        analyses = self._analyzer.analyse(word)

        if analyses is None:
            # Analyser disabled — treat as unknown
            return ValidationResult(word=word, is_valid=False, analyses_count=0)

        return ValidationResult(
            word=word,
            is_valid=len(analyses) > 0,
            analyses_count=len(analyses),
        )

    def validate_text(self, text: str) -> list[ValidationResult]:
        """Tokenise *text* and validate each Arabic word token.

        Non-Arabic tokens (digits, Latin characters, punctuation) are skipped.

        Args:
            text: Arabic text string (may contain mixed content).

        Returns:
            List of ValidationResult, one per Arabic token found.
        """
        tokens = tokenise_arabic(text)
        results: list[ValidationResult] = []

        for token in tokens:
            if not is_arabic_word(token):
                continue
            results.append(self.validate_word(token))

        return results

    def validity_rate(self, text: str) -> float:
        """Return the fraction of Arabic tokens that are morphologically valid.

        Args:
            text: Arabic text string.

        Returns:
            Float in [0.0, 1.0].
            Returns 1.0 if no Arabic tokens are found (vacuously true).
            Returns 0.0 if the analyser is disabled (no information).
        """
        if not self._analyzer.enabled:
            return 0.0

        results = self.validate_text(text)
        if not results:
            return 1.0

        valid_count = sum(1 for r in results if r.is_valid)
        return valid_count / len(results)

    def validate_correction(
        self,
        llm_text: str,
        ocr_text: str,
        strategy: str = "revert",
    ) -> TextCorrectionResult:
        """Apply post-processing validation to a single LLM-corrected text.

        Compares each Arabic word in *llm_text* against the corresponding word
        in *ocr_text* (aligned by position after tokenisation).  For words that
        differ, applies *strategy* to decide which version to keep in the final
        output.

        Strategy ``"revert"``:
            If the LLM word is morphologically *invalid* **and** the OCR word is
            morphologically *valid*, revert to the OCR word.  In all other cases
            keep the LLM word (LLM is trusted to correct OCR noise).

        Words that are identical in both texts are counted as ``unchanged`` and
        always kept as-is.

        Tokens with mismatched counts (LLM tokenisation differs from OCR) are
        passed through unchanged — the LLM output is kept verbatim for those
        positions to avoid mis-alignment.

        Args:
            llm_text: Corrected text produced by the LLM.
            ocr_text: Original OCR text (used as fallback source).
            strategy: Revert strategy to apply.  Currently only ``"revert"``
                is supported.

        Returns:
            TextCorrectionResult with the final text and word-level statistics.
        """
        if strategy != "revert":
            raise ValueError(f"Unsupported strategy: '{strategy}'. Only 'revert' is supported.")

        if not self._analyzer.enabled:
            # Analyser disabled — pass LLM text through unchanged
            llm_tokens = tokenise_arabic(llm_text)
            n = sum(1 for t in llm_tokens if is_arabic_word(t))
            return TextCorrectionResult(
                final_text=llm_text,
                total_words=n,
                reverted_count=0,
                kept_count=n,
                unchanged_count=0,
                revert_rate=0.0,
            )

        llm_tokens = tokenise_arabic(llm_text)
        ocr_tokens = tokenise_arabic(ocr_text)

        # Only consider Arabic tokens for the revert decision; non-Arabic
        # tokens (punctuation, digits, spaces) are kept from the LLM output.
        llm_arabic = [t for t in llm_tokens if is_arabic_word(t)]
        ocr_arabic = [t for t in ocr_tokens if is_arabic_word(t)]

        if len(llm_arabic) != len(ocr_arabic):
            # Token count mismatch — keep LLM text verbatim
            n = len(llm_arabic)
            return TextCorrectionResult(
                final_text=llm_text,
                total_words=n,
                reverted_count=0,
                kept_count=n,
                unchanged_count=0,
                revert_rate=0.0,
            )

        # Build word-level revert decisions
        revert_map: dict[str, str] = {}  # llm_word -> ocr_word (for reverted tokens)
        reverted_words: list[str] = []
        reverted_count = 0
        kept_count = 0
        unchanged_count = 0

        for llm_word, ocr_word in zip(llm_arabic, ocr_arabic):
            if llm_word == ocr_word:
                unchanged_count += 1
                continue

            llm_valid = self.validate_word(llm_word).is_valid
            ocr_valid = self.validate_word(ocr_word).is_valid

            if not llm_valid and ocr_valid:
                revert_map[llm_word] = ocr_word
                reverted_words.append(llm_word)
                reverted_count += 1
            else:
                kept_count += 1

        # Rebuild the final text by replacing reverted words
        final_tokens: list[str] = []
        arabic_idx = 0
        for token in llm_tokens:
            if is_arabic_word(token) and arabic_idx < len(llm_arabic):
                llm_word = llm_arabic[arabic_idx]
                arabic_idx += 1
                if llm_word in revert_map:
                    final_tokens.append(revert_map[llm_word])
                else:
                    final_tokens.append(token)
            else:
                final_tokens.append(token)

        final_text = "".join(final_tokens)
        total = reverted_count + kept_count + unchanged_count
        revert_rate = reverted_count / total if total > 0 else 0.0

        return TextCorrectionResult(
            final_text=final_text,
            total_words=total,
            reverted_count=reverted_count,
            kept_count=kept_count,
            unchanged_count=unchanged_count,
            revert_rate=revert_rate,
            reverted_words=reverted_words,
        )
