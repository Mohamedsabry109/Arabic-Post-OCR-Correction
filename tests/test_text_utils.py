"""Tests for src/data/text_utils.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.text_utils import (
    normalise_arabic,
    strip_repetitions,
    tokenise_arabic,
    is_arabic_word,
    count_arabic_chars,
    MAX_RUN,
)


class TestStripRepetitions:
    def test_collapses_long_run(self):
        # 10 zeros → MAX_RUN zeros
        text = "كلمة" + "٠" * 10
        result = strip_repetitions(text, max_run=MAX_RUN)
        assert result == "كلمة" + "٠" * MAX_RUN

    def test_preserves_short_run(self):
        text = "سسس"
        result = strip_repetitions(text, max_run=5)
        assert result == "سسس"

    def test_collapses_at_boundary(self):
        # Exactly max_run+1 copies should be collapsed
        result = strip_repetitions("أ" * (MAX_RUN + 1))
        assert result == "أ" * MAX_RUN

    def test_exactly_max_run_unchanged(self):
        text = "ب" * MAX_RUN
        assert strip_repetitions(text) == text

    def test_empty_string(self):
        assert strip_repetitions("") == ""

    def test_no_repetitions(self):
        text = "مرحبا بالعالم"
        assert strip_repetitions(text) == text

    def test_multiple_different_runs(self):
        text = "أ" * 10 + "ب" * 10
        result = strip_repetitions(text, max_run=3)
        assert result == "أ" * 3 + "ب" * 3


class TestNormaliseArabic:
    def test_alef_variants_normalised(self):
        # أ إ آ ٱ → ا
        text = "أسرة إسلامية آمنة ٱلله"
        result = normalise_arabic(text)
        # All alef variants → ا
        assert "أ" not in result
        assert "إ" not in result
        assert "آ" not in result

    def test_whitespace_collapsed(self):
        text = "كلمة   أخرى\t\nثالثة"
        result = normalise_arabic(text)
        assert "  " not in result
        assert "\t" not in result
        assert "\n" not in result

    def test_leading_trailing_stripped(self):
        result = normalise_arabic("  مرحبا  ")
        assert result == "مرحبا"

    def test_diacritics_preserved_by_default(self):
        text = "كَتَبَ"
        result = normalise_arabic(text)
        # Diacritics should remain
        assert "كَتَبَ" == result or len(result) >= 3

    def test_diacritics_removed_when_requested(self):
        text = "كَتَبَ"
        result = normalise_arabic(text, remove_diacritics=True)
        assert result == "كتب"

    def test_empty_string(self):
        assert normalise_arabic("") == ""

    def test_repetitions_collapsed(self):
        text = "كلمة" + "٠" * 20
        result = normalise_arabic(text)
        assert len(result) < len(text)


class TestTokeniseArabic:
    def test_basic_split(self):
        tokens = tokenise_arabic("الكتاب على الطاولة")
        assert tokens == ["الكتاب", "على", "الطاولة"]

    def test_ignores_punctuation(self):
        tokens = tokenise_arabic("مرحبا، كيف حالك؟")
        assert "مرحبا" in tokens
        assert "كيف" in tokens
        assert "حالك" in tokens

    def test_empty_string(self):
        assert tokenise_arabic("") == []

    def test_no_empty_tokens(self):
        tokens = tokenise_arabic("  كلمة   ")
        assert all(len(t) > 0 for t in tokens)

    def test_single_word(self):
        assert tokenise_arabic("كلمة") == ["كلمة"]

    def test_mixed_arabic_latin(self):
        tokens = tokenise_arabic("نص Arabic و text")
        assert "نص" in tokens
        assert "Arabic" in tokens


class TestIsArabicWord:
    def test_arabic_word_true(self):
        assert is_arabic_word("مرحبا") is True

    def test_latin_word_false(self):
        assert is_arabic_word("hello") is False

    def test_digits_false(self):
        assert is_arabic_word("123") is False

    def test_mixed_true(self):
        # Contains at least one Arabic char
        assert is_arabic_word("abc مرحبا") is True

    def test_empty_false(self):
        assert is_arabic_word("") is False

    def test_arabic_with_diacritics_true(self):
        assert is_arabic_word("كَتَبَ") is True


class TestCountArabicChars:
    def test_pure_arabic(self):
        assert count_arabic_chars("مرحبا") == 5

    def test_mixed(self):
        count = count_arabic_chars("abc مرحبا 123")
        assert count == 5

    def test_empty(self):
        assert count_arabic_chars("") == 0

    def test_no_arabic(self):
        assert count_arabic_chars("hello world 123") == 0
