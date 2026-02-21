"""Tests for src/analysis/metrics.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.analysis.metrics import calculate_cer, calculate_wer, calculate_metrics, MetricResult
from src.data.data_loader import OCRSample


def _make_sample(gt: str, ocr: str, sid: str = "test") -> OCRSample:
    return OCRSample(
        sample_id=sid,
        dataset="TEST",
        font=None,
        split=None,
        ocr_text=ocr,
        gt_text=gt,
        ocr_path=Path("/dev/null"),
        gt_path=None,
    )


class TestCalculateCER:
    def test_identical_strings_is_zero(self):
        assert calculate_cer("مرحبا", "مرحبا") == 0.0

    def test_completely_different_strings(self):
        # 5 chars reference, all wrong
        cer = calculate_cer("مرحبا", "كلمة!")
        assert cer > 0.0

    def test_empty_reference_returns_zero(self):
        assert calculate_cer("", "أي شيء") == 0.0

    def test_empty_hypothesis(self):
        # All chars deleted → CER = 1.0
        assert calculate_cer("مرحبا", "") == 1.0

    def test_single_char_substitution(self):
        # "مرحبا" vs "مرحيا" — 1 substitution in 5 chars = 0.2
        cer = calculate_cer("مرحبا", "مرحيا")
        assert abs(cer - 0.2) < 1e-9

    def test_can_exceed_one(self):
        # Short ref, long hypothesis with many insertions
        cer = calculate_cer("ا", "بتثنجحخدذرزسشصضطظ")
        assert cer > 1.0

    def test_normalisation_applied(self):
        # أ and ا should be treated as the same after normalisation
        cer = calculate_cer("أهلا", "اهلا")
        assert cer == 0.0


class TestCalculateWER:
    def test_identical_texts_zero(self):
        assert calculate_wer("الكتاب على الطاولة", "الكتاب على الطاولة") == 0.0

    def test_single_word_substitution(self):
        wer = calculate_wer("الكتاب على الطاولة", "الكتب على الطاولة")
        assert abs(wer - 1 / 3) < 1e-9

    def test_empty_reference_returns_zero(self):
        assert calculate_wer("", "كلمة") == 0.0

    def test_word_insertion(self):
        # 3 ref words, 1 extra → WER = 1/3
        wer = calculate_wer("كلمة واحدة فقط", "كلمة واحدة إضافية فقط")
        assert abs(wer - 1 / 3) < 1e-9

    def test_complete_mismatch(self):
        wer = calculate_wer("كلمة واحدة", "شيء مختلف تماماً")
        assert wer > 0.0


class TestCalculateMetrics:
    def test_empty_samples_returns_zero(self):
        result = calculate_metrics([], dataset_name="TEST")
        assert result.cer == 0.0
        assert result.wer == 0.0
        assert result.num_samples == 0

    def test_single_perfect_sample(self):
        samples = [_make_sample("مرحبا بالعالم", "مرحبا بالعالم")]
        result = calculate_metrics(samples, "TEST")
        assert result.cer == 0.0
        assert result.wer == 0.0
        assert result.num_samples == 1

    def test_aggregates_correctly(self):
        samples = [
            _make_sample("مرحبا", "مرحيا", "s1"),   # 0.2 CER
            _make_sample("كلمة", "كلمة", "s2"),     # 0.0 CER
        ]
        result = calculate_metrics(samples, "TEST")
        assert result.num_samples == 2
        assert abs(result.cer - 0.1) < 1e-9
        assert result.num_chars_ref > 0
        assert result.num_words_ref > 0

    def test_to_dict_has_required_keys(self):
        samples = [_make_sample("مرحبا", "مرحبا")]
        result = calculate_metrics(samples, "TEST")
        d = result.to_dict()
        for key in ("cer", "wer", "cer_std", "wer_std", "cer_median", "wer_median",
                    "cer_p95", "wer_p95", "num_samples", "num_chars_ref", "num_words_ref"):
            assert key in d, f"Missing key: {key}"

    def test_std_is_zero_for_single_sample(self):
        samples = [_make_sample("مرحبا", "مرحيا")]
        result = calculate_metrics(samples, "TEST")
        assert result.cer_std == 0.0
