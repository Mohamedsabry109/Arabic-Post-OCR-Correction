"""Tests for src/analysis/error_analyzer.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.analysis.error_analyzer import (
    ErrorAnalyzer,
    ErrorType,
    ErrorPosition,
    SampleError,
)
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


@pytest.fixture
def analyzer() -> ErrorAnalyzer:
    return ErrorAnalyzer()


class TestAlignChars:
    def test_substitution(self, analyzer):
        pairs = analyzer._align_chars("ب", "ت")
        assert ("ب", "ت") in pairs

    def test_deletion(self, analyzer):
        # "كب" vs "ك" — "ب" deleted
        pairs = analyzer._align_chars("كب", "ك")
        assert ("ب", "") in pairs

    def test_insertion(self, analyzer):
        # "ك" vs "كب" — "ب" inserted
        pairs = analyzer._align_chars("ك", "كب")
        assert ("", "ب") in pairs

    def test_identical_no_errors(self, analyzer):
        pairs = analyzer._align_chars("مرحبا", "مرحبا")
        # All pairs should be matches (gt_char == hyp_char)
        mismatches = [(g, h) for g, h in pairs if g != h]
        assert mismatches == []

    def test_empty_strings(self, analyzer):
        pairs = analyzer._align_chars("", "")
        assert pairs == []


class TestClassifyErrorType:
    def test_taa_marbuta(self, analyzer):
        assert analyzer._classify_error_type("ة", "ه") == ErrorType.TAA_MARBUTA
        assert analyzer._classify_error_type("ه", "ة") == ErrorType.TAA_MARBUTA

    def test_hamza(self, analyzer):
        assert analyzer._classify_error_type("أ", "ا") == ErrorType.HAMZA
        assert analyzer._classify_error_type("إ", "ا") == ErrorType.HAMZA

    def test_dot_confusion_ba_ta(self, analyzer):
        assert analyzer._classify_error_type("ب", "ت") == ErrorType.DOT_CONFUSION
        assert analyzer._classify_error_type("ت", "ث") == ErrorType.DOT_CONFUSION

    def test_alef_maksura(self, analyzer):
        assert analyzer._classify_error_type("ى", "ي") == ErrorType.ALEF_MAKSURA

    def test_deletion(self, analyzer):
        assert analyzer._classify_error_type("ب", "") == ErrorType.DELETION

    def test_insertion(self, analyzer):
        assert analyzer._classify_error_type("", "ت") == ErrorType.INSERTION

    def test_other_substitution(self, analyzer):
        # م vs ع — not in any special group
        result = analyzer._classify_error_type("م", "ع")
        # Could be OTHER_SUB or SIMILAR_SHAPE
        assert result in (ErrorType.OTHER_SUB, ErrorType.SIMILAR_SHAPE)


class TestAnalyseSample:
    def test_perfect_match_no_errors(self, analyzer):
        sample = _make_sample("مرحبا بالعالم", "مرحبا بالعالم")
        result = analyzer.analyse_sample(sample)
        assert result.cer == 0.0
        assert result.wer == 0.0
        assert result.char_errors == []

    def test_single_substitution(self, analyzer):
        sample = _make_sample("مرحبا", "مرحيا")
        result = analyzer.analyse_sample(sample)
        assert len(result.char_errors) == 1
        assert result.char_errors[0].gt_char == "ب"
        assert result.char_errors[0].ocr_char == "ي"

    def test_taa_marbuta_error_classified(self, analyzer):
        sample = _make_sample("مدرسة", "مدرسه")
        result = analyzer.analyse_sample(sample)
        assert any(
            e.error_type == ErrorType.TAA_MARBUTA
            for e in result.char_errors
        )

    def test_deletion_classified(self, analyzer):
        sample = _make_sample("كتاب", "كتب")
        result = analyzer.analyse_sample(sample)
        deletions = [e for e in result.char_errors if e.error_type == ErrorType.DELETION]
        assert len(deletions) >= 1

    def test_returns_sample_error(self, analyzer):
        sample = _make_sample("نص", "نص")
        result = analyzer.analyse_sample(sample)
        assert isinstance(result, SampleError)
        assert result.sample_id == "test"


class TestBuildConfusionMatrix:
    def test_format_has_required_keys(self, analyzer):
        sample = _make_sample("مدرسة على الحائط", "مدرسه عل الجائط")
        errors = [analyzer.analyse_sample(sample)]
        matrix = analyzer.build_confusion_matrix(errors, dataset="TEST", min_count=1)
        assert "meta" in matrix
        assert "confusions" in matrix
        assert "top_20" in matrix

    def test_meta_fields_present(self, analyzer):
        errors = [analyzer.analyse_sample(_make_sample("ة", "ه"))]
        matrix = analyzer.build_confusion_matrix(errors, dataset="TEST", min_count=1)
        meta = matrix["meta"]
        assert "dataset" in meta
        assert "total_substitutions" in meta
        assert "generated_at" in meta

    def test_taa_marbuta_in_matrix(self, analyzer):
        # Run the same substitution multiple times to exceed min_count
        samples = [_make_sample("مدرسة", "مدرسه", sid=str(i)) for i in range(3)]
        errors = [analyzer.analyse_sample(s) for s in samples]
        matrix = analyzer.build_confusion_matrix(errors, dataset="TEST", min_count=2)
        # ة → ه should appear
        assert "ة" in matrix["confusions"]
        assert "ه" in matrix["confusions"]["ة"]

    def test_min_count_filters_rare(self, analyzer):
        sample = _make_sample("ب", "ت")  # single occurrence
        errors = [analyzer.analyse_sample(sample)]
        matrix = analyzer.build_confusion_matrix(errors, dataset="TEST", min_count=5)
        # Should be filtered out
        assert matrix["confusions"] == {}


class TestBuildTaxonomy:
    def test_sums_consistent(self, analyzer):
        samples = [
            _make_sample("مدرسة", "مدرسه"),
            _make_sample("أسرة", "اسرة"),
            _make_sample("كتب", "كتب"),
        ]
        errors = [analyzer.analyse_sample(s) for s in samples]
        taxonomy = analyzer.build_taxonomy(errors, dataset="TEST")

        total_from_types = sum(
            v["count"] for v in taxonomy["by_type"].values()
        )
        assert total_from_types == taxonomy["meta"]["total_char_errors"]

    def test_has_required_sections(self, analyzer):
        errors = [analyzer.analyse_sample(_make_sample("ب", "ت"))]
        taxonomy = analyzer.build_taxonomy(errors, dataset="TEST")
        assert "meta" in taxonomy
        assert "by_type" in taxonomy
        assert "by_position" in taxonomy
        assert "word_level" in taxonomy

    def test_all_error_types_present(self, analyzer):
        errors = [analyzer.analyse_sample(_make_sample("كلمة", "كلمة"))]
        taxonomy = analyzer.build_taxonomy(errors, dataset="TEST")
        for et in ErrorType:
            assert et.value in taxonomy["by_type"]
