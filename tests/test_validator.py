"""Tests for src/linguistic/validator.py and src/linguistic/morphology.py

MorphAnalyzer tests cover the disabled / graceful-fallback path only —
no camel_tools installation required.

WordValidator tests use a lightweight mock analyzer so the revert logic
can be exercised without a real morphological database.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.linguistic.morphology import MorphAnalyzer
from src.linguistic.validator import WordValidator, ValidationResult, TextCorrectionResult


# ===========================================================================
# Mock MorphAnalyzer — controls which words are "valid"
# ===========================================================================


class _MockAnalyzer:
    """Minimal MorphAnalyzer stand-in with a configurable valid-word set."""

    def __init__(self, valid_words: set[str], enabled: bool = True) -> None:
        self.enabled = enabled
        self._valid_words = valid_words

    def analyse(self, word: str):
        if not self.enabled:
            return None
        return [{"analysis": "mock"}] if word in self._valid_words else []


# ===========================================================================
# MorphAnalyzer (disabled path — no camel_tools needed)
# ===========================================================================


class TestMorphAnalyzerDisabled:
    def test_disabled_flag_suppresses_analysis(self):
        analyzer = MorphAnalyzer(enabled=False)
        assert analyzer.enabled is False
        assert analyzer.analyse("كتب") is None

    def test_disabled_is_analysable_returns_false(self):
        analyzer = MorphAnalyzer(enabled=False)
        assert analyzer.is_analysable("كتب") is False

    def test_disabled_analyse_batch_all_none(self):
        analyzer = MorphAnalyzer(enabled=False)
        result = analyzer.analyse_batch(["كتب", "مدرسة"])
        assert all(v is None for v in result.values())

    def test_camel_unavailable_sets_enabled_false(self):
        """When camel_tools is not installed, enabled is set to False."""
        # This works whether camel_tools is installed or not:
        # if not installed, init sets enabled=False automatically.
        analyzer = MorphAnalyzer(enabled=True)
        if not analyzer.enabled:
            # camel_tools not installed — graceful fallback verified
            assert analyzer.analyse("كتب") is None
        else:
            # camel_tools IS installed — just verify analyse returns a list
            result = analyzer.analyse("كتب")
            assert isinstance(result, list)


# ===========================================================================
# WordValidator — validate_word
# ===========================================================================


class TestValidateWord:
    def test_valid_word_returns_is_valid_true(self):
        mock = _MockAnalyzer({"كتاب"})
        v = WordValidator(mock)
        result = v.validate_word("كتاب")
        assert result.is_valid is True
        assert result.analyses_count == 1

    def test_invalid_word_returns_is_valid_false(self):
        mock = _MockAnalyzer({"كتاب"})
        v = WordValidator(mock)
        result = v.validate_word("ببببب")
        assert result.is_valid is False
        assert result.analyses_count == 0

    def test_disabled_analyzer_returns_is_valid_false(self):
        mock = _MockAnalyzer(set(), enabled=False)
        v = WordValidator(mock)
        result = v.validate_word("كتاب")
        assert result.is_valid is False

    def test_returns_validation_result(self):
        mock = _MockAnalyzer({"مدرسة"})
        v = WordValidator(mock)
        result = v.validate_word("مدرسة")
        assert isinstance(result, ValidationResult)
        assert result.word == "مدرسة"


# ===========================================================================
# WordValidator — validate_text
# ===========================================================================


class TestValidateText:
    def test_validate_text_skips_non_arabic(self):
        mock = _MockAnalyzer(set())
        v = WordValidator(mock)
        results = v.validate_text("hello 123 مرحبا")
        # Only the Arabic word should be validated
        assert len(results) == 1
        assert results[0].word == "مرحبا"

    def test_validate_text_multiple_arabic_words(self):
        mock = _MockAnalyzer({"الكتاب", "على", "الطاولة"})
        v = WordValidator(mock)
        results = v.validate_text("الكتاب على الطاولة")
        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_validate_text_empty_string(self):
        mock = _MockAnalyzer(set())
        v = WordValidator(mock)
        assert v.validate_text("") == []

    def test_validate_text_mixed_validity(self):
        mock = _MockAnalyzer({"كتاب"})
        v = WordValidator(mock)
        results = v.validate_text("كتاب ببببب")
        valid_words   = [r for r in results if r.is_valid]
        invalid_words = [r for r in results if not r.is_valid]
        assert len(valid_words) == 1
        assert len(invalid_words) == 1


# ===========================================================================
# WordValidator — validity_rate
# ===========================================================================


class TestValidityRate:
    def test_disabled_analyzer_returns_zero(self):
        mock = _MockAnalyzer(set(), enabled=False)
        v = WordValidator(mock)
        assert v.validity_rate("الكتاب على الطاولة") == 0.0

    def test_all_valid_returns_one(self):
        mock = _MockAnalyzer({"الكتاب", "على", "الطاولة"})
        v = WordValidator(mock)
        rate = v.validity_rate("الكتاب على الطاولة")
        assert rate == 1.0

    def test_half_valid_returns_half(self):
        mock = _MockAnalyzer({"الكتاب"})   # only 1 of 2 words valid
        v = WordValidator(mock)
        rate = v.validity_rate("الكتاب ببببب")
        assert abs(rate - 0.5) < 1e-9

    def test_empty_text_returns_one(self):
        mock = _MockAnalyzer(set())
        v = WordValidator(mock)
        # No Arabic tokens → vacuously valid
        assert v.validity_rate("") == 1.0

    def test_no_arabic_tokens_returns_one(self):
        mock = _MockAnalyzer(set())
        v = WordValidator(mock)
        assert v.validity_rate("hello world 123") == 1.0


# ===========================================================================
# WordValidator — validate_correction (revert strategy)
# ===========================================================================


class TestValidateCorrection:
    def test_unsupported_strategy_raises(self):
        mock = _MockAnalyzer(set())
        v = WordValidator(mock)
        with pytest.raises(ValueError, match="revert"):
            v.validate_correction("النص", "النص", strategy="other")

    def test_identical_texts_no_revert(self):
        mock = _MockAnalyzer({"الكتاب", "على", "الطاولة"})
        v = WordValidator(mock)
        result = v.validate_correction("الكتاب على الطاولة", "الكتاب على الطاولة")
        assert isinstance(result, TextCorrectionResult)
        assert result.reverted_count == 0
        assert result.unchanged_count == 3
        # final_text is built by joining tokens (no spaces guaranteed);
        # just verify all original words appear in it
        assert "الكتاب" in result.final_text
        assert "على" in result.final_text
        assert "الطاولة" in result.final_text

    def test_revert_invalid_llm_word_when_ocr_valid(self):
        # ocr: "كتاب" (valid), llm: "ككككك" (invalid) → should revert to "كتاب"
        mock = _MockAnalyzer({"كتاب"})   # "ككككك" not in valid set
        v = WordValidator(mock)
        result = v.validate_correction("ككككك", "كتاب")
        assert result.reverted_count == 1
        assert result.revert_rate == 1.0
        assert "كتاب" in result.final_text

    def test_keep_valid_llm_word_even_if_ocr_also_valid(self):
        # Both are valid → keep LLM word (no revert)
        mock = _MockAnalyzer({"مدرسة", "مدرسه"})
        v = WordValidator(mock)
        result = v.validate_correction("مدرسة", "مدرسه")
        assert result.reverted_count == 0
        assert result.kept_count == 1
        assert "مدرسة" in result.final_text

    def test_keep_invalid_llm_word_when_ocr_also_invalid(self):
        # Both invalid → keep LLM word (can't revert to another invalid word)
        mock = _MockAnalyzer(set())   # nothing valid
        v = WordValidator(mock)
        result = v.validate_correction("ككككك", "ببببب")
        assert result.reverted_count == 0
        assert result.kept_count == 1

    def test_disabled_analyzer_passthrough(self):
        mock = _MockAnalyzer(set(), enabled=False)
        v = WordValidator(mock)
        result = v.validate_correction("النص المصحح", "النص الأصلي")
        # Disabled → returns LLM text unchanged
        assert result.final_text == "النص المصحح"
        assert result.reverted_count == 0

    def test_token_count_mismatch_passthrough(self):
        # LLM adds an extra word → token counts differ → passthrough
        mock = _MockAnalyzer({"كلمة", "إضافية", "اختبار"})
        v = WordValidator(mock)
        result = v.validate_correction("كلمة إضافية", "كلمة اختبار فقط")
        assert result.reverted_count == 0
        assert result.final_text == "كلمة إضافية"

    def test_revert_rate_calculation(self):
        # 1 revert out of 3 total differing words
        valid = {"صواب1", "صواب2", "صواب3"}
        mock = _MockAnalyzer(valid)
        v = WordValidator(mock)
        # All 3 LLM words invalid (not in valid set); OCR words valid
        result = v.validate_correction("خطا1 خطا2 خطا3", "صواب1 صواب2 صواب3")
        assert result.reverted_count == 3
        assert abs(result.revert_rate - 1.0) < 1e-9

    def test_unchanged_words_excluded_from_revert_rate(self):
        # 2 same + 1 diff (llm invalid, ocr valid)
        mock = _MockAnalyzer({"على", "الطاولة", "الكتاب"})
        v = WordValidator(mock)
        result = v.validate_correction("على الطاولة ككككك", "على الطاولة الكتاب")
        assert result.unchanged_count == 2
        assert result.reverted_count == 1
        total = result.reverted_count + result.kept_count + result.unchanged_count
        assert total == 3

    def test_result_has_reverted_words_list(self):
        mock = _MockAnalyzer({"كتاب"})
        v = WordValidator(mock)
        result = v.validate_correction("ككككك", "كتاب")
        assert "ككككك" in result.reverted_words
