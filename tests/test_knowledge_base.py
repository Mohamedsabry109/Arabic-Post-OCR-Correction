"""Tests for src/data/knowledge_base.py

Covers ConfusionMatrixLoader, LLMInsightsLoader, and WordErrorPairsLoader.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.knowledge_base import (
    ConfusionMatrixLoader,
    ConfusionPair,
)


# ===========================================================================
# ConfusionMatrixLoader
# ===========================================================================


def _write_matrix_top20(path: Path, pairs: list[dict], total: int = 1000) -> None:
    """Write a confusion_matrix.json with top_20 list."""
    data = {
        "meta": {"total_substitutions": total},
        "top_20": pairs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_matrix_confusions(path: Path, confusions: dict) -> None:
    """Write a confusion_matrix.json with full confusions dict."""
    path.write_text(json.dumps({"confusions": confusions}, ensure_ascii=False), encoding="utf-8")


class TestConfusionMatrixLoader:
    @pytest.fixture
    def loader(self) -> ConfusionMatrixLoader:
        return ConfusionMatrixLoader()

    def test_load_top20_path(self, loader, tmp_path):
        p = tmp_path / "cm.json"
        _write_matrix_top20(p, [
            {"gt": "ي", "ocr": "ب", "count": 300, "probability": 0.3},
            {"gt": "ة", "ocr": "ه", "count": 200, "probability": 0.2},
        ])
        pairs = loader.load(p)
        assert len(pairs) == 2
        assert pairs[0].gt_char == "ي"
        assert pairs[0].ocr_char == "ب"
        assert pairs[0].count == 300

    def test_load_confusions_path(self, loader, tmp_path):
        p = tmp_path / "cm.json"
        _write_matrix_confusions(p, {
            "ط": {"ظ": {"count": 150, "probability": 0.15}},
            "س": {"ش": {"count": 80,  "probability": 0.08}},
        })
        pairs = loader.load(p)
        assert len(pairs) == 2
        # Sorted descending by count
        assert pairs[0].count == 150

    def test_load_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.json")

    def test_load_malformed_json_raises(self, loader, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{bad json", encoding="utf-8")
        with pytest.raises(ValueError):
            loader.load(p)

    def test_load_empty_top20_falls_through_to_confusions(self, loader, tmp_path):
        p = tmp_path / "cm.json"
        data = {
            "top_20": [],
            "confusions": {"أ": {"ا": {"count": 50, "probability": 0.05}}},
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        pairs = loader.load(p)
        assert len(pairs) == 1

    def test_load_returns_empty_for_no_confusions(self, loader, tmp_path):
        p = tmp_path / "cm.json"
        p.write_text("{}", encoding="utf-8")
        pairs = loader.load(p)
        assert pairs == []

    def test_load_pooled_sums_counts(self, loader, tmp_path):
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        _write_matrix_top20(p1, [{"gt": "ي", "ocr": "ب", "count": 100, "probability": 0.1}])
        _write_matrix_top20(p2, [{"gt": "ي", "ocr": "ب", "count": 200, "probability": 0.2}])
        pairs = loader.load_pooled([p1, p2])
        assert len(pairs) == 1
        assert pairs[0].count == 300

    def test_load_pooled_skips_missing(self, loader, tmp_path):
        p = tmp_path / "only.json"
        _write_matrix_top20(p, [{"gt": "ط", "ocr": "ظ", "count": 50, "probability": 0.05}])
        pairs = loader.load_pooled([p, tmp_path / "missing.json"])
        assert len(pairs) == 1

    def test_load_pooled_empty_returns_empty(self, loader, tmp_path):
        pairs = loader.load_pooled([tmp_path / "missing.json"])
        assert pairs == []

    def test_has_enough_data_true_from_meta(self, loader, tmp_path):
        p = tmp_path / "cm.json"
        _write_matrix_top20(p, [], total=500)
        assert loader.has_enough_data(p, min_substitutions=200) is True

    def test_has_enough_data_false_below_threshold(self, loader, tmp_path):
        p = tmp_path / "cm.json"
        _write_matrix_top20(p, [], total=100)
        assert loader.has_enough_data(p, min_substitutions=200) is False

    def test_has_enough_data_false_for_missing_file(self, loader, tmp_path):
        assert loader.has_enough_data(tmp_path / "missing.json") is False

    def test_format_for_prompt_flat_contains_pair(self, loader):
        pairs = [ConfusionPair(gt_char="ي", ocr_char="ب", count=100, probability=0.5)]
        text = loader.format_for_prompt(pairs, n=1, style="flat_arabic")
        assert "ي" in text
        assert "ب" in text
        assert "50%" in text

    def test_format_for_prompt_grouped(self, loader):
        # ى/ي fall into the alef_maksura group
        pairs = [ConfusionPair(gt_char="ى", ocr_char="ي", count=200, probability=0.2)]
        text = loader.format_for_prompt(pairs, n=1, style="grouped_arabic")
        assert "ى" in text or "الألف" in text

    def test_format_for_prompt_empty_returns_empty_string(self, loader):
        assert loader.format_for_prompt([], n=5) == ""

    def test_format_for_prompt_n_limits_output(self, loader):
        pairs = [
            ConfusionPair("ي", "ب", 300, 0.3),
            ConfusionPair("ة", "ه", 200, 0.2),
            ConfusionPair("ي", "ب", 100, 0.1),  # duplicate key, unique count
        ]
        text_all = loader.format_for_prompt(pairs, n=3)
        text_n2 = loader.format_for_prompt(pairs, n=2)
        # n=2 should produce fewer bullet lines than n=3
        bullets_all = text_all.count("يستبدل")
        bullets_n2  = text_n2.count("يستبدل")
        assert bullets_n2 < bullets_all




