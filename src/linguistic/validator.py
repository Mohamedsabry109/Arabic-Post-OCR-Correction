"""Arabic word morphological validation using CAMeL Tools.

Provides WordValidator which checks individual tokens and aggregates
validity rates across full texts.
"""

import logging
from dataclasses import dataclass

from src.data.text_utils import tokenise_arabic, is_arabic_word
from src.linguistic.morphology import MorphAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Morphological validation result for a single word token."""

    word: str
    is_valid: bool           # has ≥ 1 morphological analysis
    analyses_count: int      # number of analyses (0 means invalid/unknown)


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
