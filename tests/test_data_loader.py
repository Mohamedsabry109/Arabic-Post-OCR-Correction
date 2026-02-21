"""Tests for src/data/data_loader.py"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.data_loader import (
    DataLoader,
    DataError,
    OCRSample,
    _pair_by_stem,
    _pats_line_number,
    _read_ocr_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_khatt(tmp_path: Path):
    """Create a minimal fake KHATT directory structure with 3 sample pairs."""
    ocr_dir = tmp_path / "ocr" / "khatt-data" / "train" / "Training"
    gt_dir = tmp_path / "gt" / "KHATT" / "data" / "train" / "Training"
    ocr_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    samples = [
        ("AHTD3A0001_Para2_3", "من العذاب في الآخرة وأفضل ما رزقهم", "من العذاب في الاخرة واضل ما رزقهم"),
        ("AHTD3A0002_Para2_1", "الكتاب على الطاولة", "الكتب على الطاوله"),
        ("AHTD3A0003_Para1_1", "مرحبا بالعالم العربي", "مرحبا بالعالم العربي"),
    ]

    for stem, gt_text, ocr_text in samples:
        (gt_dir / f"{stem}.txt").write_text(gt_text, encoding="utf-8")
        (ocr_dir / f"{stem}.txt").write_text(ocr_text, encoding="utf-8")

    config = {
        "data": {
            "ocr_results": str(tmp_path / "ocr"),
            "ground_truth": str(tmp_path / "gt"),
            "pats_gt_file": None,
        },
        "processing": {"limit_per_dataset": None},
    }
    return config, ocr_dir, gt_dir, samples


# ---------------------------------------------------------------------------
# Tests: _pair_by_stem
# ---------------------------------------------------------------------------


class TestPairByStem:
    def test_matches_correctly(self, tmp_path):
        ocr_dir = tmp_path / "ocr"
        gt_dir = tmp_path / "gt"
        ocr_dir.mkdir()
        gt_dir.mkdir()

        for name in ["file_a", "file_b", "file_c"]:
            (ocr_dir / f"{name}.txt").write_text("ocr", encoding="utf-8")
            (gt_dir / f"{name}.txt").write_text("gt", encoding="utf-8")

        # Extra file only in OCR
        (ocr_dir / "ocr_only.txt").write_text("ocr only", encoding="utf-8")

        pairs = _pair_by_stem(ocr_dir, gt_dir)
        assert len(pairs) == 3
        stems = {p[0].stem for p in pairs}
        assert stems == {"file_a", "file_b", "file_c"}

    def test_returns_sorted(self, tmp_path):
        ocr_dir = tmp_path / "ocr"
        gt_dir = tmp_path / "gt"
        ocr_dir.mkdir()
        gt_dir.mkdir()

        for name in ["z_file", "a_file", "m_file"]:
            (ocr_dir / f"{name}.txt").write_text("x", encoding="utf-8")
            (gt_dir / f"{name}.txt").write_text("x", encoding="utf-8")

        pairs = _pair_by_stem(ocr_dir, gt_dir)
        stems = [p[0].stem for p in pairs]
        assert stems == sorted(stems)


# ---------------------------------------------------------------------------
# Tests: DataLoader.load_khatt
# ---------------------------------------------------------------------------


class TestLoadKHATT:
    def test_returns_ocr_samples(self, tmp_khatt):
        config, *_ = tmp_khatt
        loader = DataLoader(config)
        samples = loader.load_khatt(split="train")
        assert len(samples) == 3
        assert all(isinstance(s, OCRSample) for s in samples)

    def test_sample_fields_populated(self, tmp_khatt):
        config, _, _, expected = tmp_khatt
        loader = DataLoader(config)
        samples = loader.load_khatt(split="train")
        sample = next(s for s in samples if s.sample_id == "AHTD3A0001_Para2_3")
        assert sample.dataset == "KHATT"
        assert sample.split == "train"
        assert sample.font is None
        assert len(sample.gt_text) > 0
        assert len(sample.ocr_text) > 0

    def test_limit_respected(self, tmp_khatt):
        config, *_ = tmp_khatt
        loader = DataLoader(config)
        samples = loader.load_khatt(split="train", limit=2)
        assert len(samples) == 2

    def test_nonexistent_dir_raises_data_error(self, tmp_khatt):
        config, *_ = tmp_khatt
        loader = DataLoader(config)
        with pytest.raises(DataError, match="not found"):
            loader.load_khatt(split="validation")

    def test_empty_ocr_file_skipped(self, tmp_khatt):
        config, ocr_dir, gt_dir, _ = tmp_khatt
        # Add an empty OCR file with a corresponding GT
        (ocr_dir / "empty_sample.txt").write_text("", encoding="utf-8")
        (gt_dir / "empty_sample.txt").write_text("ground truth text", encoding="utf-8")
        loader = DataLoader(config)
        samples = loader.load_khatt(split="train")
        # Empty OCR file should be skipped
        assert all(s.sample_id != "empty_sample" for s in samples)


# ---------------------------------------------------------------------------
# Tests: _pats_line_number
# ---------------------------------------------------------------------------


class TestPatsLineNumber:
    def test_extracts_number(self):
        assert _pats_line_number(Path("Akhbar_123.txt")) == 123
        assert _pats_line_number(Path("Andalus_7.txt")) == 7
        assert _pats_line_number(Path("Akhbar_1000.txt")) == 1000

    def test_returns_zero_for_unrecognised(self):
        assert _pats_line_number(Path("ReadMe.txt")) == 0


# ---------------------------------------------------------------------------
# Tests: DataLoader missing config keys
# ---------------------------------------------------------------------------


class TestDataLoaderConfig:
    def test_missing_ocr_results_raises(self):
        bad_config = {"data": {"ground_truth": "/some/path", "pats_gt_file": None}}
        with pytest.raises(DataError, match="config"):
            DataLoader(bad_config)

    def test_pats_without_gt_file_raises_data_error(self, tmp_khatt):
        config, *_ = tmp_khatt
        # config already has pats_gt_file: None
        loader = DataLoader(config)
        with pytest.raises(DataError, match="PATS"):
            loader.load_pats(font="Akhbar")


# ---------------------------------------------------------------------------
# Tests: _read_ocr_file
# ---------------------------------------------------------------------------


class TestReadOCRFile:
    def test_strips_repetitions(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("كلمة" + "٠" * 100, encoding="utf-8")
        result = _read_ocr_file(f)
        assert len(result) < 20  # much shorter than 104 chars

    def test_empty_file_returns_empty_string(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = _read_ocr_file(f)
        assert result == ""

    def test_normal_file_returned(self, tmp_path):
        f = tmp_path / "normal.txt"
        f.write_text("مرحبا بالعالم", encoding="utf-8")
        result = _read_ocr_file(f)
        assert result == "مرحبا بالعالم"
