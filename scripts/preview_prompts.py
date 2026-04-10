#!/usr/bin/env python3
"""Preview prompts for all phases with sample Arabic text.

Usage:
    python scripts/preview_prompts.py                  # all phases
    python scripts/preview_prompts.py --phase 2 3 4a   # specific phases
    python scripts/preview_prompts.py --phase 5 --full # show full prompt text
    python scripts/preview_prompts.py --phase 3 --dataset PATS-A01-Akhbar-train
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Sample OCR text used for all previews
# ---------------------------------------------------------------------------

SAMPLE_OCR_TEXT = (
    "وقال رئيس الوزراء ان الحكومه ستعمل على تطوبر"
    " التعليم والصحه فى جميع المحافظات"
)

# ---------------------------------------------------------------------------
# Placeholder contexts (used when real data is unavailable)
# ---------------------------------------------------------------------------

PLACEHOLDER_CONFUSION = (
    "- يستبدل (ه) بـ (ة) في 23% من الحالات\n"
    "- يستبدل (ي) بـ (ى) في 18% من الحالات\n"
    "- يستبدل (ا) بـ (أ) في 12% من الحالات\n"
    "- يستبدل (س) بـ (ش) في 8% من الحالات\n"
    "- يحذف (ء) في 6% من الحالات"
)


PLACEHOLDER_INSIGHTS = (
    "نقاط القوة:\n"
    "- يصحح أخطاء التاء المربوطة/المفتوحة بنسبة 78%\n"
    "- يصحح أخطاء الألف بأشكالها بنسبة 65%\n"
    "نقاط الضعف:\n"
    "- يصحح أخطاء النقاط (ب/ت/ث/ن) بنسبة 23% فقط\n"
    "- يصحح أخطاء الحروف المتشابهة (س/ش) بنسبة 31% فقط\n"
    "تصحيحات زائدة:\n"
    "- يدخل أخطاء همزة جديدة بنسبة 8%"
)

PLACEHOLDER_WORD_PAIRS = (
    "الحكومه -> الحكومة\n"
    "تطوبر -> تطوير\n"
    "فى -> في\n"
    "المحافظه -> المحافظة\n"
    "الصحه -> الصحة"
)

# ---------------------------------------------------------------------------
# Loaders for real data (used when available)
# ---------------------------------------------------------------------------


def _load_real_confusion(dataset: str | None) -> str | None:
    """Try to load real confusion context from Phase 1 results."""
    from src.data.knowledge_base import ConfusionMatrixLoader

    if dataset:
        search = [dataset]
    else:
        search = ["PATS-A01-Akhbar-train", "KHATT-train"]

    loader = ConfusionMatrixLoader()
    for ds in search:
        path = PROJECT_ROOT / "results" / "phase1" / ds / "confusion_matrix.json"
        if path.exists():
            try:
                pairs = loader.load(path)
                if pairs:
                    return loader.format_for_prompt(pairs, n=10)
            except Exception:
                pass
    return None


def _load_real_insights(dataset_type: str | None = None) -> str | None:
    """Try to load real Phase 4D insights."""
    from src.data.knowledge_base import LLMInsightsLoader

    ds_type = dataset_type or "PATS-A01"
    insights_dir = PROJECT_ROOT / "results" / "phase4d" / "insights"
    path = insights_dir / f"{ds_type}_insights.json"
    if not path.exists():
        return None
    try:
        loader = LLMInsightsLoader()
        data = loader.load(path)
        if data:
            return loader.format_for_prompt(data)
    except Exception:
        pass
    return None


def _load_real_word_pairs() -> str | None:
    """Try to load real Phase 4D word-error pairs."""
    from src.data.knowledge_base import WordErrorPairsLoader

    path = PROJECT_ROOT / "results" / "phase4d" / "word_error_pairs.txt"
    if not path.exists():
        return None
    try:
        loader = WordErrorPairsLoader()
        pairs = loader.load(path)
        if pairs:
            selected = loader.select(pairs, n=15, strategy="random", seed=42)
            if selected:
                return loader.format_for_prompt(selected)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 70
THIN_SEP = "-" * 70


def _print_messages(
    messages: list[dict],
    phase_label: str,
    version: str,
    full: bool = False,
    real_data: bool = False,
) -> None:
    """Pretty-print a chat message list."""
    data_tag = "[real data]" if real_data else "[placeholder]"
    print(f"\n{SEPARATOR}")
    print(f"  {phase_label}  (version: {version})  {data_tag}")
    print(SEPARATOR)

    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]

        print(f"\n--- {role} ---")
        if full:
            print(content)
        else:
            # Truncated preview: first 600 chars + last 200 chars
            if len(content) > 900:
                print(content[:600])
                print(f"\n  [...truncated {len(content) - 800} chars...]\n")
                print(content[-200:])
            else:
                print(content)

    print(f"\n  System prompt length: {len(messages[0]['content'])} chars")
    print(f"  User message length:  {len(messages[1]['content'])} chars")
    print(THIN_SEP)


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------


def preview_phase2(pb: PromptBuilder, full: bool, **_: object) -> None:
    msgs = pb.build_zero_shot(SAMPLE_OCR_TEXT)
    _print_messages(msgs, "Phase 2: Zero-Shot", pb.prompt_version, full)


def preview_phase3(
    pb: PromptBuilder, full: bool, dataset: str | None = None, **_: object,
) -> None:
    real = _load_real_confusion(dataset)
    ctx = real if real else PLACEHOLDER_CONFUSION
    msgs = pb.build_ocr_aware(SAMPLE_OCR_TEXT, ctx)
    _print_messages(
        msgs, "Phase 3: OCR-Aware", pb.ocr_aware_prompt_version, full,
        real_data=real is not None,
    )


def preview_phase4d(pb: PromptBuilder, full: bool, **_: object) -> None:
    real_insights = _load_real_insights()
    real_pairs = _load_real_word_pairs()
    ins = real_insights if real_insights else PLACEHOLDER_INSIGHTS
    wp = real_pairs if real_pairs else PLACEHOLDER_WORD_PAIRS
    msgs = pb.build_self_reflective(SAMPLE_OCR_TEXT, ins, wp)
    has_real = real_insights is not None or real_pairs is not None
    _print_messages(
        msgs, "Phase 4D: Self-Reflective + Word Pairs",
        pb.self_reflective_prompt_version, full, real_data=has_real,
    )


def preview_phase5(
    pb: PromptBuilder, full: bool, dataset: str | None = None, **_: object,
) -> None:
    real_conf = _load_real_confusion(dataset)
    real_insights = _load_real_insights()
    real_pairs = _load_real_word_pairs()

    msgs = pb.build_combined(
        SAMPLE_OCR_TEXT,
        confusion_context=real_conf or PLACEHOLDER_CONFUSION,
        insights_context=real_insights or PLACEHOLDER_INSIGHTS,
        word_pairs_context=real_pairs or PLACEHOLDER_WORD_PAIRS,
    )
    has_real = any(x is not None for x in [
        real_conf, real_insights, real_pairs,
    ])
    _print_messages(
        msgs, "Phase 6: Combined (all contexts)",
        pb.combined_prompt_version, full, real_data=has_real,
    )


# Phase 7 uses DSPy-generated prompts — show the base zero-shot that DSPy optimizes from.
def preview_phase7(pb: PromptBuilder, full: bool, **_: object) -> None:
    msgs = pb.build_zero_shot(SAMPLE_OCR_TEXT)
    _print_messages(
        msgs,
        "Phase 7: DSPy (base prompt — DSPy optimizes this at runtime)",
        pb.prompt_version, full,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PHASES: dict[str, callable] = {
    "2":  preview_phase2,
    "3":  preview_phase3,
    "4":  preview_phase4d,
    "6":  preview_phase5,
    "7":  preview_phase7,
}

ALL_PHASES = list(PHASES.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    # Fix Windows console encoding for Arabic text
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )

    parser = argparse.ArgumentParser(
        description="Preview prompts for all experimental phases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/preview_prompts.py                    # all phases, truncated
              python scripts/preview_prompts.py --phase 2 3        # phases 2 and 3
              python scripts/preview_prompts.py --phase 5 --full   # phase 5, full text
              python scripts/preview_prompts.py --dataset KHATT-train  # use KHATT confusion matrix
        """),
    )
    parser.add_argument(
        "--phase", nargs="+", default=ALL_PHASES,
        choices=ALL_PHASES, metavar="PHASE",
        help=f"Phases to preview (choices: {', '.join(ALL_PHASES)}). Default: all.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show full prompt text instead of truncated preview.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset key for loading real confusion matrix (e.g. PATS-A01-Akhbar-train).",
    )
    args = parser.parse_args()

    pb = PromptBuilder()

    print(f"Sample OCR text: {SAMPLE_OCR_TEXT}")
    print(f"Previewing phases: {', '.join(args.phase)}")

    for phase_key in args.phase:
        PHASES[phase_key](pb, full=args.full, dataset=args.dataset)

    print(f"\n{'=' * 70}")
    print("  Prompt editing guide:")
    print(f"{'=' * 70}")
    print("  Base system prompt : configs/crafted_system_prompt.txt")
    print("  Prompt builder     : src/core/prompt_builder.py")
    print("  Knowledge loaders  : src/data/knowledge_base.py")
    print("  Config knobs       : configs/config.yaml")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
