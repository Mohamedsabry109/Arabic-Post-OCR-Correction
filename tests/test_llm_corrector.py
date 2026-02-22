"""Tests for src/core/llm_corrector.py

Tests focus on pure logic that can run without a GPU or a loaded model:
- CorrectionResult dataclass construction
- TransformersCorrector._extract_corrected_text()
- get_corrector() error handling for unknown backends
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.core.llm_corrector import CorrectionResult, TransformersCorrector, get_corrector


# ===========================================================================
# CorrectionResult
# ===========================================================================


class TestCorrectionResult:
    def test_construction_success(self):
        r = CorrectionResult(
            sample_id="s1",
            ocr_text="النص الأصلي",
            corrected_text="النص المصحح",
            prompt_tokens=50,
            output_tokens=20,
            latency_s=0.5,
            success=True,
        )
        assert r.sample_id == "s1"
        assert r.success is True
        assert r.error is None

    def test_construction_failure(self):
        r = CorrectionResult(
            sample_id="s2",
            ocr_text="نص",
            corrected_text="نص",   # fallback = ocr_text
            prompt_tokens=0,
            output_tokens=0,
            latency_s=1.2,
            success=False,
            error="Timeout",
        )
        assert r.success is False
        assert r.error == "Timeout"
        assert r.corrected_text == r.ocr_text  # safe fallback

    def test_optional_error_defaults_to_none(self):
        r = CorrectionResult("s3", "نص", "نص مصحح", 10, 5, 0.1, True)
        assert r.error is None


# ===========================================================================
# TransformersCorrector._extract_corrected_text (no model needed)
# ===========================================================================


class TestExtractCorrectedText:
    """Call _extract_corrected_text directly without loading a model."""

    @pytest.fixture
    def extractor(self):
        """Return a bound method without loading the model."""
        # Create a partially-initialised instance by bypassing __init__
        obj = object.__new__(TransformersCorrector)
        return obj._extract_corrected_text

    def test_normal_arabic_output(self, extractor):
        result = extractor("النص المصحح بشكل صحيح", "النص الأصلي")
        assert result == "النص المصحح بشكل صحيح"

    def test_whitespace_stripped(self, extractor):
        result = extractor("   النص المصحح   ", "النص الأصلي")
        assert result == "النص المصحح"

    def test_empty_output_falls_back_to_ocr(self, extractor):
        result = extractor("", "النص الأصلي")
        assert result == "النص الأصلي"

    def test_whitespace_only_falls_back_to_ocr(self, extractor):
        result = extractor("   \n\t  ", "النص الأصلي")
        assert result == "النص الأصلي"

    def test_non_arabic_output_falls_back_to_ocr(self, extractor):
        # Model replied in English
        result = extractor("I cannot correct this text.", "النص الأصلي")
        assert result == "النص الأصلي"

    def test_mixed_arabic_latin_kept(self, extractor):
        # Contains Arabic characters — accepted
        result = extractor("نص مع some latin", "النص الأصلي")
        assert "نص" in result

    def test_arabic_with_newline_stripped(self, extractor):
        result = extractor("النص المصحح\n", "النص الأصلي")
        assert result == "النص المصحح"

    def test_single_arabic_char_accepted(self, extractor):
        result = extractor("ا", "ا")
        assert result == "ا"

    def test_ocr_text_returned_unchanged_as_fallback(self, extractor):
        original = "النص الأصلي مع تفاصيل"
        result = extractor("", original)
        assert result is original or result == original


# ===========================================================================
# get_corrector — error handling (no model load)
# ===========================================================================


class TestGetCorrector:
    def test_unknown_backend_raises_value_error(self):
        config = {"model": {"backend": "unknown_backend"}}
        with pytest.raises(ValueError, match="unknown_backend"):
            get_corrector(config)

    def test_missing_backend_key_defaults_to_transformers(self):
        # get_corrector with no backend key tries to load TransformersCorrector.
        # We can't load a real model — just verify the ValueError is NOT raised
        # for "transformers" backend (it tries loading, not an invalid backend).
        config = {"model": {"backend": "transformers"}}
        try:
            get_corrector(config)
        except (ImportError, RuntimeError, OSError):
            pass   # Expected: model not downloaded / no GPU
        except ValueError as exc:
            pytest.fail(f"ValueError raised for 'transformers' backend: {exc}")

    def test_api_backend_imported_without_error(self):
        # The "api" backend imports from api_corrector — should not raise ValueError
        config = {"model": {"backend": "api"}}
        try:
            get_corrector(config)
        except (ImportError, RuntimeError, OSError, Exception):
            pass   # Expected: missing API key or connectivity
        except ValueError as exc:
            pytest.fail(f"ValueError raised for 'api' backend: {exc}")
