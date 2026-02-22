"""Tests for src/data/knowledge_base.py

Covers ConfusionMatrixLoader, RulesLoader, QALBLoader, and
OpenITILoader (static / pure methods only; full corpus load skipped).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.knowledge_base import (
    ConfusionMatrixLoader,
    ConfusionPair,
    RulesLoader,
    QALBLoader,
    QALBPair,
    OpenITILoader,
    CorpusSentence,
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


# ===========================================================================
# RulesLoader
# ===========================================================================


class TestRulesLoader:
    @pytest.fixture
    def loader(self) -> RulesLoader:
        return RulesLoader()

    def test_load_all_returns_all_core_rules(self, loader):
        rules = loader.load()
        assert len(rules) == len(RulesLoader.CORE_RULES)

    def test_load_filter_by_category(self, loader):
        rules = loader.load(categories=["hamza"])
        assert all(r.category == "hamza" for r in rules)
        assert len(rules) > 0

    def test_load_multiple_categories(self, loader):
        rules = loader.load(categories=["hamza", "taa_marbuta"])
        cats = {r.category for r in rules}
        assert cats == {"hamza", "taa_marbuta"}

    def test_load_unknown_category_returns_empty(self, loader):
        rules = loader.load(categories=["nonexistent_category"])
        assert rules == []

    def test_load_returns_independent_copy(self, loader):
        # Modifying the returned list should not affect the class constant
        rules = loader.load()
        rules.clear()
        assert len(loader.CORE_RULES) > 0

    def test_format_for_prompt_compact_contains_arabic(self, loader):
        rules = loader.load()
        text = loader.format_for_prompt(rules, style="compact_arabic")
        assert len(text) > 0
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        assert arabic_chars > 10

    def test_format_for_prompt_detailed_contains_more_text(self, loader):
        rules = loader.load()
        compact = loader.format_for_prompt(rules, style="compact_arabic")
        detailed = loader.format_for_prompt(rules, style="detailed_arabic")
        assert len(detailed) >= len(compact)

    def test_format_for_prompt_n_limits_rules(self, loader):
        rules = loader.load()
        text_all = loader.format_for_prompt(rules)
        text_2 = loader.format_for_prompt(rules, n=2)
        assert len(text_2) < len(text_all)

    def test_format_for_prompt_empty_returns_empty_string(self, loader):
        assert loader.format_for_prompt([]) == ""

    def test_all_categories_present(self, loader):
        rules = loader.load()
        found_cats = {r.category for r in rules}
        for cat in RulesLoader.CATEGORIES:
            assert cat in found_cats, f"Category '{cat}' missing from CORE_RULES"


# ===========================================================================
# QALBLoader (pure logic — no file I/O)
# ===========================================================================


def _make_qalb_pair(source: str, corrected: str, error_types=None) -> QALBPair:
    return QALBPair(
        source=source,
        corrected=corrected,
        error_types=error_types or [],
        source_file="test",
    )


class TestQALBLoader:
    @pytest.fixture
    def loader(self) -> QALBLoader:
        return QALBLoader()  # No config — uses defaults

    # ---- _detect_ocr_error_types ----

    def test_detect_taa_marbuta(self, loader):
        types = loader._detect_ocr_error_types("مدرسه", "مدرسة")
        assert "taa_marbuta" in types

    def test_detect_hamza(self, loader):
        types = loader._detect_ocr_error_types("اكرم", "أكرم")
        assert "hamza" in types

    def test_detect_alef_maksura(self, loader):
        types = loader._detect_ocr_error_types("موسي", "موسى")
        assert "alef_maksura" in types

    def test_detect_dot_confusion(self, loader):
        types = loader._detect_ocr_error_types("كتاب", "كتاب".replace("ب", "ت"))
        # ب→ت is dot_confusion if lengths match
        types2 = loader._detect_ocr_error_types("كتاب", "كتات")
        assert "dot_confusion" in types2

    def test_detect_no_substitution_returns_empty(self, loader):
        types = loader._detect_ocr_error_types("مرحبا", "مرحبا")
        assert types == []

    def test_detect_non_ocr_substitution_returns_empty(self, loader):
        # ل→م is not in OCR_SUBSTITUTION_PAIRS
        types = loader._detect_ocr_error_types("كلمة", "كممة")
        assert types == []

    # ---- filter_ocr_relevant ----

    def test_filter_keeps_ocr_relevant_pairs(self, loader):
        pairs = [
            _make_qalb_pair("مدرسه في المدينة", "مدرسة في المدينة"),   # taa_marbuta error
            _make_qalb_pair("جملة عادية جداً لا يوجد بها أي خطأ", "جملة عادية جداً لا يوجد بها أي خطأ"),  # identical
        ]
        # The second pair is already removed by load() (identical), but filter also handles it
        filtered = loader.filter_ocr_relevant([pairs[0]])
        assert len(filtered) == 1

    def test_filter_rejects_too_short(self, loader):
        pairs = [_make_qalb_pair("مدرسه", "مدرسة")]  # 5 chars < min_length=10
        filtered = loader.filter_ocr_relevant(pairs, min_length=10)
        assert filtered == []

    def test_filter_rejects_too_long(self, loader):
        long_text = "مدرسه " * 30   # very long
        pairs = [_make_qalb_pair(long_text, long_text.replace("ه", "ة"))]
        filtered = loader.filter_ocr_relevant(pairs, max_length=100)
        assert filtered == []

    def test_filter_rejects_too_many_changed_words(self, loader):
        # More than max_words_changed=4 positions differ
        src = "كلمه وجملة ومثال ونص وأكثر من ذلك"
        cor = src.replace("ه", "ة").replace("جملة", "جمله")
        pairs = [_make_qalb_pair(src, cor)]
        filtered = loader.filter_ocr_relevant(pairs, max_words_changed=1)
        assert filtered == []

    def test_filter_populates_error_types(self, loader):
        pairs = [_make_qalb_pair("مدرسه في المدينة", "مدرسة في المدينة")]
        filtered = loader.filter_ocr_relevant(pairs)
        if filtered:
            assert filtered[0].error_types != []

    # ---- select ----

    def test_select_random_returns_n(self, loader):
        pairs = [
            _make_qalb_pair(f"جملة رقم {i} للاختبار", f"جملة صحيحة {i}", ["taa_marbuta"])
            for i in range(20)
        ]
        selected = loader.select(pairs, n=5, strategy="random")
        assert len(selected) == 5

    def test_select_returns_all_if_fewer_than_n(self, loader):
        pairs = [_make_qalb_pair("جملة للاختبار", "جملة صحيحة", ["hamza"])]
        selected = loader.select(pairs, n=5)
        assert len(selected) == 1

    def test_select_empty_returns_empty(self, loader):
        assert loader.select([], n=5) == []

    def test_select_diverse_covers_multiple_types(self, loader):
        types = ["taa_marbuta", "hamza", "alef_maksura", "dot_confusion"]
        pairs = [
            _make_qalb_pair(f"جملة للاختبار {i}", f"جملة صحيحة {i}", [t])
            for i, t in enumerate(types * 3)
        ]
        selected = loader.select(pairs, n=4, strategy="diverse")
        found_types = {t for p in selected for t in p.error_types}
        assert len(found_types) >= 2  # diverse selection spans types

    def test_select_reproducible_with_same_seed(self, loader):
        pairs = [
            _make_qalb_pair(f"جملة {i} للاختبار", f"صحيح {i}", ["hamza"])
            for i in range(10)
        ]
        s1 = loader.select(pairs, n=3, strategy="random", seed=99)
        s2 = loader.select(pairs, n=3, strategy="random", seed=99)
        assert [p.source for p in s1] == [p.source for p in s2]

    # ---- format_for_prompt ----

    def test_format_inline_style(self, loader):
        pairs = [_make_qalb_pair("مدرسه في المدينة", "مدرسة في المدينة")]
        text = loader.format_for_prompt(pairs, style="inline_arabic")
        assert "مدرسه" in text
        assert "مدرسة" in text

    def test_format_numbered_style(self, loader):
        pairs = [_make_qalb_pair("مدرسه", "مدرسة", ["taa_marbuta"])]
        text = loader.format_for_prompt(pairs, style="numbered_arabic")
        assert "1." in text

    def test_format_empty_returns_empty_string(self, loader):
        assert loader.format_for_prompt([]) == ""

    def test_format_multiple_pairs(self, loader):
        pairs = [
            _make_qalb_pair("مدرسه", "مدرسة"),
            _make_qalb_pair("اكرم", "أكرم"),
        ]
        text = loader.format_for_prompt(pairs)
        assert "مدرسه" in text
        assert "اكرم" in text


# ===========================================================================
# OpenITILoader (static methods + save/load corpus)
# ===========================================================================


class TestOpenITILoaderStatics:
    def test_is_content_line_page_marker(self):
        assert OpenITILoader._is_content_line("PageV01P001") is False

    def test_is_content_line_section_marker(self):
        assert OpenITILoader._is_content_line("### Section") is False
        assert OpenITILoader._is_content_line("##") is False
        assert OpenITILoader._is_content_line("#") is False

    def test_is_content_line_meta(self):
        assert OpenITILoader._is_content_line("#META#Header#End#") is False

    def test_is_content_line_arabic_prefix(self):
        assert OpenITILoader._is_content_line("# كتب العرب في العصر الحديث") is True

    def test_is_content_line_poem_prefix(self):
        assert OpenITILoader._is_content_line("~~بيت شعر عربي جميل") is True

    def test_is_content_line_bare_arabic(self):
        # Enough Arabic chars — treated as content
        assert OpenITILoader._is_content_line("كتب العرب في العصر الحديث") is True

    def test_is_content_line_too_few_arabic(self):
        assert OpenITILoader._is_content_line("abc") is False

    def test_clean_content_line_removes_prefix(self):
        cleaned = OpenITILoader._clean_content_line("# كتب العرب في الكتاب")
        assert not cleaned.startswith("# ")
        assert "كتب العرب" in cleaned

    def test_clean_content_line_removes_poem_prefix(self):
        cleaned = OpenITILoader._clean_content_line("~~بيت شعر جميل")
        assert not cleaned.startswith("~~")

    def test_clean_content_line_removes_page_markers(self):
        cleaned = OpenITILoader._clean_content_line("# نص PageV01P005 عربي")
        assert "PageV01P005" not in cleaned

    def test_clean_content_line_normalises_whitespace(self):
        cleaned = OpenITILoader._clean_content_line("# كلمة   كلمة   كلمة")
        assert "  " not in cleaned

    def test_clean_content_line_removes_trailing_verse_number(self):
        cleaned = OpenITILoader._clean_content_line("# بيت شعر جميل 42")
        assert not cleaned.strip().endswith("42")


class TestOpenITILoaderParseFile:
    def test_parse_file_extracts_content_after_header(self, tmp_path):
        content = "\n".join([
            "##METADATA",
            "#META#Header#End#",
            "# كتب العرب في العصر الحديث وما تلاه من عصور",
            "# نص عربي آخر مناسب للاختبار يحتوي على كلمات كثيرة",
            "",
        ])
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")
        lines = OpenITILoader.parse_file(p)
        assert len(lines) >= 1
        assert any("كتب العرب" in line for line in lines)

    def test_parse_file_skips_header(self, tmp_path):
        content = "\n".join([
            "##METADATA",
            "title: some title",
            "#META#Header#End#",
            "# النص الفعلي للكتاب العربي يبدأ هنا",
        ])
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")
        lines = OpenITILoader.parse_file(p)
        assert all("METADATA" not in line for line in lines)

    def test_parse_file_skips_page_markers(self, tmp_path):
        content = "\n".join([
            "#META#Header#End#",
            "PageV01P001",
            "# نص عربي جيد للاختبار يحتوي على محتوى مفيد",
        ])
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")
        lines = OpenITILoader.parse_file(p)
        assert all("PageV" not in line for line in lines)

    def test_parse_file_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        lines = OpenITILoader.parse_file(p)
        assert lines == []


class TestOpenITILoaderCorpus:
    @pytest.fixture
    def sentences(self) -> list[CorpusSentence]:
        return [
            CorpusSentence("جملة عربية صحيحة للاختبار", "uri_1", 400, 25, 0),
            CorpusSentence("نص عربي آخر للاختبار والتحقق", "uri_2", 500, 28, 1),
        ]

    def test_save_and_load_corpus_roundtrip(self, tmp_path, sentences):
        loader = OpenITILoader()
        p = tmp_path / "corpus.jsonl"
        loader.save_corpus(sentences, p)
        loaded = loader.load_corpus(p)
        assert len(loaded) == len(sentences)
        assert loaded[0].text == sentences[0].text
        assert loaded[1].source_uri == sentences[1].source_uri

    def test_load_corpus_missing_file_raises(self, tmp_path):
        loader = OpenITILoader()
        with pytest.raises(FileNotFoundError):
            loader.load_corpus(tmp_path / "missing.jsonl")

    def test_save_creates_parent_dirs(self, tmp_path, sentences):
        loader = OpenITILoader()
        p = tmp_path / "subdir" / "corpus.jsonl"
        loader.save_corpus(sentences, p)
        assert p.exists()

    def test_load_corpus_skips_malformed_lines(self, tmp_path):
        loader = OpenITILoader()
        p = tmp_path / "bad.jsonl"
        p.write_text(
            '{"text": "جملة صحيحة"}\n'
            'bad json line\n'
            '{"text": "جملة أخرى"}\n',
            encoding="utf-8",
        )
        loaded = loader.load_corpus(p)
        assert len(loaded) == 2
