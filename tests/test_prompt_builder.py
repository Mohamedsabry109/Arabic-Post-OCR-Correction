"""Tests for src/core/prompt_builder.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.core.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


OCR_TEXT = "النص المدخل للاختبار"
CONFUSION_CTX = "ي → ب (300 مرة)"
RULES_CTX = "التاء المربوطة تُكتب هكذا"
EXAMPLES_CTX = "مثال: خطأ → صواب"
RETRIEVAL_CTX = "1. نص مرجعي صحيح"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_chat_format(messages: list[dict]) -> None:
    """Assert messages is a valid 2-element OpenAI chat list."""
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert isinstance(messages[0]["content"], str)
    assert isinstance(messages[1]["content"], str)


def _user_text(messages: list[dict]) -> str:
    return messages[1]["content"]


def _system_text(messages: list[dict]) -> str:
    return messages[0]["content"]


# ---------------------------------------------------------------------------
# build_zero_shot
# ---------------------------------------------------------------------------

class TestBuildZeroShot:
    def test_returns_two_messages(self, builder):
        msgs = builder.build_zero_shot(OCR_TEXT)
        _assert_chat_format(msgs)

    def test_user_content_is_ocr_text(self, builder):
        msgs = builder.build_zero_shot(OCR_TEXT)
        assert _user_text(msgs) == OCR_TEXT

    def test_system_content_is_arabic(self, builder):
        system = _system_text(builder.build_zero_shot(OCR_TEXT))
        assert len(system) > 0
        # System prompt should contain Arabic instruction keywords
        assert "مصحح" in system or "تصحيح" in system

    def test_prompt_version_property(self, builder):
        assert builder.prompt_version == "v1"

    def test_empty_ocr_text_accepted(self, builder):
        msgs = builder.build_zero_shot("")
        assert _user_text(msgs) == ""


# ---------------------------------------------------------------------------
# build_ocr_aware
# ---------------------------------------------------------------------------

class TestBuildOcrAware:
    def test_returns_two_messages(self, builder):
        msgs = builder.build_ocr_aware(OCR_TEXT, CONFUSION_CTX)
        _assert_chat_format(msgs)

    def test_user_content_is_ocr_text(self, builder):
        msgs = builder.build_ocr_aware(OCR_TEXT, CONFUSION_CTX)
        assert _user_text(msgs) == OCR_TEXT

    def test_confusion_context_in_system(self, builder):
        msgs = builder.build_ocr_aware(OCR_TEXT, CONFUSION_CTX)
        assert CONFUSION_CTX in _system_text(msgs)

    def test_empty_confusion_falls_back_to_zero_shot(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        aware = builder.build_ocr_aware(OCR_TEXT, "")
        assert aware == zero

    def test_whitespace_only_confusion_falls_back(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        aware = builder.build_ocr_aware(OCR_TEXT, "   ")
        assert aware == zero

    def test_prompt_version_property(self, builder):
        assert builder.ocr_aware_prompt_version == "p3v1"


# ---------------------------------------------------------------------------
# build_rule_augmented
# ---------------------------------------------------------------------------

class TestBuildRuleAugmented:
    def test_returns_two_messages(self, builder):
        msgs = builder.build_rule_augmented(OCR_TEXT, RULES_CTX)
        _assert_chat_format(msgs)

    def test_rules_context_in_system(self, builder):
        msgs = builder.build_rule_augmented(OCR_TEXT, RULES_CTX)
        assert RULES_CTX in _system_text(msgs)

    def test_empty_rules_falls_back_to_zero_shot(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        rule = builder.build_rule_augmented(OCR_TEXT, "")
        assert rule == zero

    def test_prompt_version_property(self, builder):
        assert builder.rules_prompt_version == "p4av1"


# ---------------------------------------------------------------------------
# build_few_shot
# ---------------------------------------------------------------------------

class TestBuildFewShot:
    def test_returns_two_messages(self, builder):
        msgs = builder.build_few_shot(OCR_TEXT, EXAMPLES_CTX)
        _assert_chat_format(msgs)

    def test_examples_context_in_system(self, builder):
        msgs = builder.build_few_shot(OCR_TEXT, EXAMPLES_CTX)
        assert EXAMPLES_CTX in _system_text(msgs)

    def test_empty_examples_falls_back_to_zero_shot(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        fs = builder.build_few_shot(OCR_TEXT, "")
        assert fs == zero

    def test_prompt_version_property(self, builder):
        assert builder.few_shot_prompt_version == "p4bv1"


# ---------------------------------------------------------------------------
# build_rag
# ---------------------------------------------------------------------------

class TestBuildRag:
    def test_returns_two_messages(self, builder):
        msgs = builder.build_rag(OCR_TEXT, RETRIEVAL_CTX)
        _assert_chat_format(msgs)

    def test_retrieval_context_in_system(self, builder):
        msgs = builder.build_rag(OCR_TEXT, RETRIEVAL_CTX)
        assert RETRIEVAL_CTX in _system_text(msgs)

    def test_empty_retrieval_falls_back_to_zero_shot(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        rag = builder.build_rag(OCR_TEXT, "")
        assert rag == zero

    def test_prompt_version_property(self, builder):
        assert builder.rag_prompt_version == "p5v1"


# ---------------------------------------------------------------------------
# build_combined
# ---------------------------------------------------------------------------

class TestBuildCombined:
    def test_returns_two_messages_all_contexts(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, CONFUSION_CTX, RULES_CTX, EXAMPLES_CTX, RETRIEVAL_CTX
        )
        _assert_chat_format(msgs)

    def test_user_content_is_ocr_text(self, builder):
        msgs = builder.build_combined(OCR_TEXT, confusion_context=CONFUSION_CTX)
        assert _user_text(msgs) == OCR_TEXT

    def test_all_non_empty_contexts_included(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, CONFUSION_CTX, RULES_CTX, EXAMPLES_CTX, RETRIEVAL_CTX
        )
        system = _system_text(msgs)
        assert CONFUSION_CTX in system
        assert RULES_CTX in system
        assert EXAMPLES_CTX in system
        assert RETRIEVAL_CTX in system

    def test_empty_contexts_excluded(self, builder):
        # Only confusion provided; rules/examples/retrieval should not appear
        msgs = builder.build_combined(
            OCR_TEXT,
            confusion_context=CONFUSION_CTX,
            rules_context="",
            examples_context="",
            retrieval_context="",
        )
        system = _system_text(msgs)
        assert CONFUSION_CTX in system
        assert RULES_CTX not in system
        assert EXAMPLES_CTX not in system
        assert RETRIEVAL_CTX not in system

    def test_all_empty_falls_back_to_zero_shot(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        combined = builder.build_combined(OCR_TEXT, "", "", "", "")
        assert combined == zero

    def test_whitespace_only_contexts_treated_as_empty(self, builder):
        zero = builder.build_zero_shot(OCR_TEXT)
        combined = builder.build_combined(OCR_TEXT, "  ", "\t", "  \n  ", "")
        assert combined == zero

    def test_single_context_confusion_only(self, builder):
        msgs = builder.build_combined(OCR_TEXT, confusion_context=CONFUSION_CTX)
        system = _system_text(msgs)
        assert CONFUSION_CTX in system

    def test_single_context_rag_only(self, builder):
        msgs = builder.build_combined(OCR_TEXT, retrieval_context=RETRIEVAL_CTX)
        system = _system_text(msgs)
        assert RETRIEVAL_CTX in system

    def test_context_order_confusion_before_rules(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, confusion_context=CONFUSION_CTX, rules_context=RULES_CTX
        )
        system = _system_text(msgs)
        assert system.index(CONFUSION_CTX) < system.index(RULES_CTX)

    def test_context_order_rules_before_examples(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, rules_context=RULES_CTX, examples_context=EXAMPLES_CTX
        )
        system = _system_text(msgs)
        assert system.index(RULES_CTX) < system.index(EXAMPLES_CTX)

    def test_context_order_examples_before_retrieval(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, examples_context=EXAMPLES_CTX, retrieval_context=RETRIEVAL_CTX
        )
        system = _system_text(msgs)
        assert system.index(EXAMPLES_CTX) < system.index(RETRIEVAL_CTX)

    def test_pair_confusion_rules(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, confusion_context=CONFUSION_CTX, rules_context=RULES_CTX
        )
        system = _system_text(msgs)
        assert CONFUSION_CTX in system
        assert RULES_CTX in system
        assert EXAMPLES_CTX not in system
        assert RETRIEVAL_CTX not in system

    def test_pair_rules_fewshot(self, builder):
        msgs = builder.build_combined(
            OCR_TEXT, rules_context=RULES_CTX, examples_context=EXAMPLES_CTX
        )
        system = _system_text(msgs)
        assert RULES_CTX in system
        assert EXAMPLES_CTX in system
        assert CONFUSION_CTX not in system
        assert RETRIEVAL_CTX not in system

    def test_prompt_version_property(self, builder):
        assert builder.combined_prompt_version == "p6v1"

    def test_combined_system_differs_from_zero_shot(self, builder):
        zero = _system_text(builder.build_zero_shot(OCR_TEXT))
        combined = _system_text(
            builder.build_combined(OCR_TEXT, confusion_context=CONFUSION_CTX)
        )
        assert combined != zero
