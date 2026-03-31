#!/usr/bin/env python3
"""LLM-assisted prompt crafting for Arabic OCR correction.

Uses the same LLM (via ``infer.py``) to analyze real OCR error patterns
and design an optimal correction prompt, grounded in actual data.

Workflow
-------
    # Round 1: Generate the crafted prompt
    python scripts/craft_prompt.py --mode prepare          # local
    python scripts/infer.py \
        --input  results/prompt_craft/meta_prompt_input.jsonl \
        --output results/prompt_craft/meta_prompt_output.jsonl  # Kaggle
    python scripts/craft_prompt.py --mode extract           # local

    # Round 2: Evaluate the crafted prompt
    python scripts/craft_prompt.py --mode export            # local
    python scripts/infer.py \
        --input  results/prompt_craft/inference_input.jsonl \
        --output results/prompt_craft/corrections.jsonl     # Kaggle
    python scripts/craft_prompt.py --mode analyze           # local

    # Round 3 (optional): Refine based on failures
    python scripts/craft_prompt.py --mode refine            # local
    python scripts/infer.py \
        --input  results/prompt_craft/refine_input.jsonl \
        --output results/prompt_craft/refine_output.jsonl   # Kaggle
    python scripts/craft_prompt.py --mode extract \
        --corrections results/prompt_craft/refine_output.jsonl  # local
    # Then repeat Round 2
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError, OCRSample
from src.analysis.metrics import calculate_cer, calculate_wer
from src.analysis.report_formatter import write_corrections_report
from src.data.text_utils import normalise_arabic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "results" / "prompt_craft"

# CER boundaries for stratified sampling
_CER_EASY_MAX = 0.05
_CER_MEDIUM_MAX = 0.25

# Default sample counts per stratum
_N_CORRECT = 5
_N_EASY = 5
_N_MEDIUM = 10
_N_HARD = 5

_SEED = 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-assisted prompt crafting for Arabic OCR correction."
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["prepare", "extract", "refine", "export", "analyze"],
        help=(
            "prepare: build meta-prompt JSONL for infer.py | "
            "extract: pull generated prompt from infer.py output | "
            "refine: build refinement JSONL from failures | "
            "export: create evaluation JSONL using crafted prompt | "
            "analyze: compare results vs baselines"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "config.yaml",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Directory for prompt_craft outputs (default: results/prompt_craft).",
    )
    p.add_argument(
        "--crafted-prompt",
        type=Path,
        default=None,
        dest="crafted_prompt",
        help="Path to crafted_system_prompt.txt.",
    )
    p.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help=(
            "Path to infer.py output JSONL. "
            "extract: meta_prompt_output.jsonl or refine_output.jsonl | "
            "refine/analyze: corrections.jsonl"
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset keys to process (default: validation datasets for export).",
    )
    p.add_argument(
        "--sample-list",
        type=Path,
        default=None,
        dest="sample_list",
        help="Path to test_samples.json to filter samples (export mode).",
    )
    p.add_argument(
        "--include-khatt",
        action="store_true",
        dest="include_khatt",
        help="Include KHATT-train in sample selection (prepare mode).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset (export mode).",
    )
    p.add_argument("--seed", type=int, default=_SEED)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_output_dir(args: argparse.Namespace, config: dict) -> Path:
    """Resolve the output directory from args or config."""
    if args.output_dir:
        return args.output_dir
    craft_cfg = config.get("prompt_craft", {})
    return Path(craft_cfg.get("results_dir", "results/prompt_craft"))


def _resolve_crafted_path(args: argparse.Namespace, output_dir: Path) -> Path:
    """Resolve the crafted prompt file path."""
    return args.crafted_prompt or (output_dir / "crafted_system_prompt.txt")


def _get_train_datasets(config: dict, include_khatt: bool = False) -> list[str]:
    """Return train dataset keys from config."""
    datasets_cfg = config.get("datasets", [])
    train: list[str] = []
    for ds in datasets_cfg:
        name = ds["name"]
        ds_type = ds.get("type", "")
        if ds_type == "PATS-A01" and ds.get("pats_split") == "train":
            train.append(name)
        elif ds_type == "KHATT" and ds.get("split") == "train" and include_khatt:
            train.append(name)
    return train


def _get_val_datasets(config: dict) -> list[str]:
    """Return validation dataset keys from config."""
    datasets_cfg = config.get("datasets", [])
    val: list[str] = []
    for ds in datasets_cfg:
        name = ds["name"]
        ds_type = ds.get("type", "")
        if ds_type == "PATS-A01" and ds.get("pats_split") == "validation":
            val.append(name)
        elif ds_type == "KHATT" and ds.get("split") == "validation":
            val.append(name)
    return val


def _write_meta_jsonl(
    output_path: Path,
    user_prompt: str,
    sample_id: str = "meta_prompt_001",
    prompt_version: str = "meta_v1",
) -> None:
    """Write a single-record JSONL for meta-prompting via infer.py."""
    record = {
        "sample_id": sample_id,
        "dataset": "prompt_craft",
        "ocr_text": user_prompt,
        "gt_text": "",
        "prompt_type": "meta_prompt",
        "prompt_version": prompt_version,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def select_diverse_samples(
    loader: DataLoader,
    config: dict,
    include_khatt: bool = False,
    seed: int = _SEED,
    n_correct: int = _N_CORRECT,
    n_easy: int = _N_EASY,
    n_medium: int = _N_MEDIUM,
    n_hard: int = _N_HARD,
) -> dict[str, list[dict]]:
    """Select stratified samples from train data by CER difficulty.

    Returns:
        Dict with keys ``"correct"``, ``"easy"``, ``"medium"``, ``"hard"``.
        Each value is a list of dicts with sample info including OCR/GT text.
    """
    rng = random.Random(seed)
    train_datasets = _get_train_datasets(config, include_khatt)
    logger.info(
        "Scanning %d train datasets for diverse samples ...", len(train_datasets)
    )

    buckets: dict[str, list[dict]] = {
        "correct": [],
        "easy": [],
        "medium": [],
        "hard": [],
    }

    for ds_key in train_datasets:
        try:
            samples = list(loader.iter_samples(ds_key))
        except DataError as exc:
            logger.warning("Skipping %s: %s", ds_key, exc)
            continue

        for s in samples:
            gt_norm = normalise_arabic(s.gt_text, remove_diacritics=True)
            ocr_norm = normalise_arabic(s.ocr_text, remove_diacritics=True)

            # Skip near-empty or runaway
            if len(gt_norm) < 5:
                continue
            if len(ocr_norm) / max(len(gt_norm), 1) > 5.0:
                continue

            cer = calculate_cer(s.gt_text, s.ocr_text, strip_diacritics=True)

            entry = {
                "sample_id": s.sample_id,
                "dataset": ds_key,
                "font": s.font,
                "cer": round(cer, 6),
                "ocr_text": s.ocr_text,
                "gt_text": s.gt_text,
            }

            if cer == 0.0:
                buckets["correct"].append(entry)
            elif cer <= _CER_EASY_MAX:
                buckets["easy"].append(entry)
            elif cer <= _CER_MEDIUM_MAX:
                buckets["medium"].append(entry)
            else:
                buckets["hard"].append(entry)

    # Stratified selection with font diversity
    targets = {
        "correct": n_correct,
        "easy": n_easy,
        "medium": n_medium,
        "hard": n_hard,
    }

    selected: dict[str, list[dict]] = {}
    for bucket_name, target_n in targets.items():
        pool = buckets[bucket_name]
        if not pool:
            selected[bucket_name] = []
            continue

        rng.shuffle(pool)
        picked: list[dict] = []
        seen_fonts: set[str] = set()

        # First pass: one sample per font for diversity
        for entry in pool:
            if len(picked) >= target_n:
                break
            font = entry.get("font") or entry.get("dataset")
            if font not in seen_fonts:
                picked.append(entry)
                seen_fonts.add(font)

        # Second pass: fill remaining slots
        for entry in pool:
            if len(picked) >= target_n:
                break
            if entry not in picked:
                picked.append(entry)

        selected[bucket_name] = picked

    total = sum(len(v) for v in selected.values())
    logger.info(
        "Selected %d samples: %d correct, %d easy, %d medium, %d hard",
        total,
        len(selected["correct"]),
        len(selected["easy"]),
        len(selected["medium"]),
        len(selected["hard"]),
    )
    return selected


# ---------------------------------------------------------------------------
# Meta-prompt construction
# ---------------------------------------------------------------------------


_META_PROMPT_HEADER = """\
# Task: Design an Optimal Arabic OCR Correction System Prompt

## Context
You are designing a system prompt for an Arabic LLM (Qwen3-4B-Instruct) \
that corrects OCR errors from Qaari, an open-source Arabic OCR system.

The correction pipeline works as follows:
- The LLM receives a **system prompt** (what you will design) and the \
**OCR text** as the user message.
- The LLM must output **ONLY** the corrected Arabic text -- no \
explanations, comments, or formatting.
- If the text is already correct, it must be returned unchanged.

## Representative OCR Samples

Below are real OCR outputs paired with their ground truth (GT), grouped \
by error severity.  Study these carefully to understand the error \
patterns before designing the prompt."""

_BUCKET_LABELS: dict[str, tuple[str, str]] = {
    "correct": (
        "Correct samples (CER = 0.0)",
        "The OCR produced these correctly.  "
        "The prompt must teach the LLM NOT to change correct text.",
    ),
    "easy": (
        "Easy errors (CER < 0.05)",
        "Minor mistakes -- typically one or two character errors.",
    ),
    "medium": (
        "Medium errors (0.05 < CER < 0.25)",
        "Typical OCR errors -- multiple character confusions per sample.",
    ),
    "hard": (
        "Hard errors (CER > 0.25)",
        "Significant corruption -- many errors throughout the text.",
    ),
}

_META_PROMPT_REQUIREMENTS = """\

## Requirements for the System Prompt

Design an Arabic system prompt that:

1. **Defines the task clearly**: The LLM corrects Arabic OCR output \
from the Qaari system.
2. **Balances correction vs. conservation**:
   - Correct clear OCR errors (character confusions, missing/extra chars).
   - Do NOT change text that is already correct.
   - When uncertain, prefer leaving text unchanged over risking a wrong \
correction.
3. **Addresses specific Qaari error patterns**: Reference the character \
confusions and error types you observed in the samples above.
4. **Output format**: Return ONLY the corrected text -- no explanations.
5. **Conciseness**: Under 200 Arabic words.
6. **Language**: The entire prompt must be in Arabic.
7. **No examples**: Do NOT include correction examples (a separate \
pipeline phase handles few-shot examples).

## Output Format

Return ONLY the Arabic system prompt text.  No English text, no \
markdown, no explanations -- just the raw Arabic system message."""


def build_meta_prompt(samples: dict[str, list[dict]]) -> str:
    """Build the meta-prompt that asks the LLM to design a correction prompt."""
    parts: list[str] = [_META_PROMPT_HEADER]

    for bucket_name in ("correct", "easy", "medium", "hard"):
        bucket = samples.get(bucket_name, [])
        if not bucket:
            continue
        label, desc = _BUCKET_LABELS[bucket_name]
        parts.append(f"\n### {label}")
        parts.append(f"*{desc}*\n")
        for i, s in enumerate(bucket, 1):
            cer_str = f"{s['cer']:.4f}"
            font_str = f", Font: {s['font']}" if s.get("font") else ""
            parts.append(f"**Sample {i}** (CER: {cer_str}{font_str})")
            parts.append(f"OCR: {s['ocr_text']}")
            parts.append(f"GT:  {s['gt_text']}")
            parts.append("")

    parts.append(_META_PROMPT_REQUIREMENTS)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Refine-prompt construction
# ---------------------------------------------------------------------------


def build_refine_prompt(crafted_prompt: str, failures: list[dict]) -> str:
    """Build a refinement meta-prompt showing where the crafted prompt failed."""
    parts: list[str] = []

    parts.append("# Task: Refine the Arabic OCR Correction System Prompt\n")
    parts.append("## Current Prompt")
    parts.append(
        "The following Arabic system prompt was used for OCR correction, "
        "but it caused regressions on some samples (made them worse than "
        "the original OCR):\n"
    )
    parts.append("---")
    parts.append(crafted_prompt)
    parts.append("---\n")

    n_shown = min(len(failures), 15)
    parts.append(
        f"## Failures ({len(failures)} total, showing worst {n_shown})\n"
    )
    parts.append(
        "For each failure:\n"
        "- **OCR**: original OCR text\n"
        "- **LLM**: what the LLM output using the current prompt\n"
        "- **GT**: the correct ground truth\n"
        "- **CER**: character error rate (higher = worse)\n"
    )

    for i, f in enumerate(failures[:n_shown], 1):
        parts.append(
            f"**Failure {i}** "
            f"(CER: {f['cer_before']:.4f} -> {f['cer_after']:.4f})"
        )
        parts.append(f"OCR: {f['ocr_text']}")
        parts.append(f"LLM: {f['corrected_text']}")
        parts.append(f"GT:  {f['gt_text']}")
        parts.append("")

    parts.append(
        "## Instructions\n\n"
        "Analyze WHY the prompt failed on these samples.  Common issues:\n"
        "- Over-correction: changing words that were already correct\n"
        "- Hallucination: adding text not in the original\n"
        "- Missing errors: failing to fix obvious OCR mistakes\n"
        "- Wrong corrections: replacing with the wrong character\n\n"
        "Produce an improved Arabic system prompt that addresses these "
        "failure patterns.\n"
        "Same constraints: Arabic only, under 200 words, no examples.\n\n"
        "Return ONLY the revised Arabic system prompt text, nothing else."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Mode: prepare
# ---------------------------------------------------------------------------


def run_prepare(args: argparse.Namespace, config: dict) -> None:
    """Select diverse samples and build meta-prompt JSONL for infer.py."""
    output_dir = _resolve_output_dir(args, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    craft_cfg = config.get("prompt_craft", {})
    loader = DataLoader(config)

    samples = select_diverse_samples(
        loader,
        config,
        include_khatt=args.include_khatt,
        seed=args.seed,
        n_correct=craft_cfg.get("n_correct", _N_CORRECT),
        n_easy=craft_cfg.get("n_easy", _N_EASY),
        n_medium=craft_cfg.get("n_medium", _N_MEDIUM),
        n_hard=craft_cfg.get("n_hard", _N_HARD),
    )

    # Save sample details for reproducibility
    samples_path = output_dir / "samples_used.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("Saved sample details to %s", samples_path)

    # Build meta-prompt text
    meta_prompt_text = build_meta_prompt(samples)

    # Save human-readable copy
    meta_txt_path = output_dir / "meta_prompt.txt"
    with open(meta_txt_path, "w", encoding="utf-8") as f:
        f.write(meta_prompt_text)
    logger.info("Saved meta-prompt (readable) to %s", meta_txt_path)

    # Write JSONL for infer.py
    jsonl_path = output_dir / "meta_prompt_input.jsonl"
    _write_meta_jsonl(jsonl_path, meta_prompt_text, sample_id="meta_prompt_001")
    logger.info("Saved meta-prompt JSONL to %s", jsonl_path)

    output_jsonl = output_dir / "meta_prompt_output.jsonl"
    print("\n" + "=" * 60)
    print("Meta-prompt ready!")
    print(f"  JSONL : {jsonl_path}")
    print(f"  Text  : {meta_txt_path}")
    print()
    print("Next step -- run inference on Kaggle/Colab:")
    print(f"  python scripts/infer.py \\")
    print(f"    --input  {jsonl_path} \\")
    print(f"    --output {output_jsonl}")
    print()
    print("Then: python scripts/craft_prompt.py --mode extract")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Mode: extract
# ---------------------------------------------------------------------------


def run_extract(args: argparse.Namespace, config: dict) -> None:
    """Extract the generated prompt from infer.py output."""
    output_dir = _resolve_output_dir(args, config)

    # Determine input: --corrections flag, or default meta_prompt_output.jsonl
    corrections_path = args.corrections or (output_dir / "meta_prompt_output.jsonl")
    if not corrections_path.exists():
        logger.error("Inference output not found: %s", corrections_path)
        logger.error(
            "Run infer.py with meta_prompt_input.jsonl first."
        )
        sys.exit(1)

    # Read the first (and usually only) record
    generated_prompt = None
    with open(corrections_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_type") == "meta_prompt" or r.get("dataset") == "prompt_craft":
                generated_prompt = r.get("corrected_text", "").strip()
                break

    if not generated_prompt:
        # Fallback: use the first record's corrected_text
        with open(corrections_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                generated_prompt = r.get("corrected_text", "").strip()
                break

    if not generated_prompt:
        logger.error("No generated prompt found in %s", corrections_path)
        sys.exit(1)

    # Save to crafted_system_prompt.txt
    crafted_path = _resolve_crafted_path(args, output_dir)
    crafted_path.parent.mkdir(parents=True, exist_ok=True)
    with open(crafted_path, "w", encoding="utf-8") as f:
        f.write(generated_prompt)

    logger.info(
        "Extracted crafted prompt (%d chars) to %s",
        len(generated_prompt), crafted_path,
    )

    print("\n" + "=" * 60)
    print("Crafted prompt extracted!")
    print(f"  File: {crafted_path}")
    print(f"  Length: {len(generated_prompt)} chars")
    print()
    print("Review the prompt, then run:")
    print("  python scripts/craft_prompt.py --mode export")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Mode: refine
# ---------------------------------------------------------------------------


def run_refine(args: argparse.Namespace, config: dict) -> None:
    """Build a refinement JSONL from the crafted prompt + regression failures."""
    output_dir = _resolve_output_dir(args, config)

    # Load the crafted prompt
    crafted_path = _resolve_crafted_path(args, output_dir)
    if not crafted_path.exists():
        logger.error("Crafted prompt not found: %s", crafted_path)
        logger.error("Run --mode extract first.")
        sys.exit(1)

    crafted_prompt = crafted_path.read_text(encoding="utf-8").strip()
    logger.info(
        "Loaded crafted prompt (%d chars) from %s", len(crafted_prompt), crafted_path
    )

    # Load corrections (evaluation results)
    corrections_path = args.corrections or (output_dir / "corrections.jsonl")
    if not corrections_path.exists():
        logger.error("Corrections not found: %s", corrections_path)
        logger.error("Run --mode export + infer.py + --mode analyze first.")
        sys.exit(1)

    # Find regressions (samples where the crafted prompt made CER worse)
    failures: list[dict] = []
    with open(corrections_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gt = r.get("gt_text", "")
            ocr = r.get("ocr_text", "")
            corrected = r.get("corrected_text", "")

            if not gt or not ocr:
                continue

            cer_before = calculate_cer(gt, ocr, strip_diacritics=True)
            cer_after = calculate_cer(gt, corrected, strip_diacritics=True)

            if cer_after > cer_before:
                failures.append(
                    {
                        "sample_id": r.get("sample_id", ""),
                        "dataset": r.get("dataset", ""),
                        "ocr_text": ocr,
                        "corrected_text": corrected,
                        "gt_text": gt,
                        "cer_before": round(cer_before, 6),
                        "cer_after": round(cer_after, 6),
                        "cer_delta": round(cer_after - cer_before, 6),
                    }
                )

    # Sort by worst regression first
    failures.sort(key=lambda x: x["cer_delta"], reverse=True)
    logger.info(
        "Found %d regressions (samples where LLM made things worse)", len(failures)
    )

    if not failures:
        print("No regressions found -- the crafted prompt did not hurt any sample!")
        print("No refinement needed.")
        return

    # Save regressions for reference
    regressions_path = output_dir / "regressions.json"
    with open(regressions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_regressions": len(failures),
                "worst_10": failures[:10],
                "all": failures,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved %d regressions to %s", len(failures), regressions_path)

    # Build refine prompt and save as JSONL for infer.py
    refine_prompt_text = build_refine_prompt(crafted_prompt, failures)

    # Save readable copy
    refine_txt_path = output_dir / "refine_prompt.txt"
    with open(refine_txt_path, "w", encoding="utf-8") as f:
        f.write(refine_prompt_text)

    # Write JSONL for infer.py
    refine_jsonl_path = output_dir / "refine_input.jsonl"
    _write_meta_jsonl(
        refine_jsonl_path,
        refine_prompt_text,
        sample_id="refine_prompt_001",
        prompt_version="refine_v1",
    )
    logger.info("Saved refine JSONL to %s", refine_jsonl_path)

    refine_output = output_dir / "refine_output.jsonl"
    print("\n" + "=" * 60)
    print(f"Refine prompt ready!  ({len(failures)} regressions analyzed)")
    print(f"  JSONL : {refine_jsonl_path}")
    print(f"  Text  : {refine_txt_path}")
    print()
    print("Next step -- run inference on Kaggle/Colab:")
    print(f"  python scripts/infer.py \\")
    print(f"    --input  {refine_jsonl_path} \\")
    print(f"    --output {refine_output}")
    print()
    print("Then extract the revised prompt:")
    print(f"  python scripts/craft_prompt.py --mode extract \\")
    print(f"    --corrections {refine_output}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Mode: export
# ---------------------------------------------------------------------------


def run_export(args: argparse.Namespace, config: dict) -> None:
    """Export inference_input.jsonl using the crafted system prompt."""
    output_dir = _resolve_output_dir(args, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load crafted prompt
    crafted_path = _resolve_crafted_path(args, output_dir)
    if not crafted_path.exists():
        logger.error("Crafted prompt not found: %s", crafted_path)
        logger.error("Run --mode prepare + infer.py + --mode extract first.")
        sys.exit(1)

    crafted_prompt = crafted_path.read_text(encoding="utf-8").strip()
    logger.info("Loaded crafted prompt (%d chars)", len(crafted_prompt))

    loader = DataLoader(config)

    # Load sample list if provided (overrides --datasets)
    sample_ids: Optional[set[str]] = None
    if args.sample_list:
        with open(args.sample_list, encoding="utf-8") as f:
            sl_data = json.load(f)
        sample_ids = set(sl_data["sample_ids"])
        dataset_keys = list(sl_data.get("meta", {}).get("by_dataset", {}).keys())
        logger.info(
            "Sample list loaded: %d sample IDs from %s",
            len(sample_ids), args.sample_list,
        )
    elif args.datasets:
        dataset_keys = args.datasets
    else:
        dataset_keys = _get_val_datasets(config)
        if not dataset_keys:
            logger.error("No validation datasets found in config.")
            sys.exit(1)
    logger.info("Exporting %d datasets: %s", len(dataset_keys), dataset_keys)

    output_path = output_dir / "inference_input.jsonl"
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for ds_key in dataset_keys:
            try:
                samples = list(loader.iter_samples(
                    ds_key, limit=args.limit, sample_ids=sample_ids,
                ))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for s in samples:
                record = {
                    "sample_id": s.sample_id,
                    "dataset": ds_key,
                    "ocr_text": s.ocr_text,
                    "gt_text": s.gt_text,
                    "prompt_type": "crafted",
                    "prompt_version": "crafted_v1",
                    "system_prompt": crafted_prompt,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    logger.info("Exported %d samples to %s", total, output_path)

    corrections_path = output_dir / "corrections.jsonl"
    print("\n" + "=" * 60)
    print(f"Export complete: {total} samples")
    print(f"  File: {output_path}")
    print()
    print("Next step -- run inference on Kaggle/Colab:")
    print(f"  python scripts/infer.py \\")
    print(f"    --input  {output_path} \\")
    print(f"    --output {corrections_path}")
    print()
    print("Then: python scripts/craft_prompt.py --mode analyze")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Mode: analyze
# ---------------------------------------------------------------------------


def run_analyze(args: argparse.Namespace, config: dict) -> None:
    """Analyze crafted-prompt results vs Phase 1 and Phase 2 baselines."""
    output_dir = _resolve_output_dir(args, config)

    corrections_path = args.corrections or (output_dir / "corrections.jsonl")
    if not corrections_path.exists():
        logger.error("Corrections not found: %s", corrections_path)
        sys.exit(1)

    # Load corrections grouped by dataset
    by_dataset: dict[str, list[dict]] = {}
    with open(corrections_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ds = r.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(r)

    logger.info(
        "Loaded corrections: %d datasets, %d total samples",
        len(by_dataset),
        sum(len(v) for v in by_dataset.values()),
    )

    # Load Phase 1 baseline
    results_root = Path(config.get("output", {}).get("results_dir", "results"))

    phase1_metrics: dict = {}
    p1_path = results_root / "phase1" / "summary" / "aggregate_metrics.json"
    if p1_path.exists():
        with open(p1_path, encoding="utf-8") as f:
            phase1_metrics = json.load(f)

    # Load Phase 2 per-dataset baselines
    phase2_baselines: dict[str, dict] = {}
    for ds_key in by_dataset:
        p2_path = results_root / "phase2" / ds_key / "metrics.json"
        if p2_path.exists():
            with open(p2_path, encoding="utf-8") as f:
                phase2_baselines[ds_key] = json.load(f)

    # Calculate and display metrics
    print("\n" + "=" * 70)
    print("Crafted Prompt -- Evaluation Results")
    print("=" * 70)

    all_results: dict[str, dict] = {}
    # Per-sample detail for error analysis
    all_sample_details: list[dict] = []

    for ds_key, records in sorted(by_dataset.items()):
        cer_ocr_values: list[float] = []
        wer_ocr_values: list[float] = []
        cer_values: list[float] = []
        wer_values: list[float] = []
        n_total = len(records)

        # Error analysis counters
        n_improved = 0
        n_regressed = 0
        n_unchanged = 0

        # Severity buckets for improvements and regressions
        cer_delta_buckets = {"large_improve": 0, "small_improve": 0, "small_regress": 0, "large_regress": 0}

        sample_details: list[dict] = []

        for r in records:
            gt = r.get("gt_text", "")
            ocr = r.get("ocr_text", "")
            corrected = r.get("corrected_text", "")
            if not gt:
                continue

            cer_ocr = calculate_cer(gt, ocr, strip_diacritics=True)
            wer_ocr = calculate_wer(gt, ocr, strip_diacritics=True)
            cer_llm = calculate_cer(gt, corrected, strip_diacritics=True)
            wer_llm = calculate_wer(gt, corrected, strip_diacritics=True)

            cer_ocr_values.append(cer_ocr)
            wer_ocr_values.append(wer_ocr)
            cer_values.append(cer_llm)
            wer_values.append(wer_llm)

            cer_delta = cer_ocr - cer_llm  # positive = improvement

            if cer_delta > 1e-6:
                n_improved += 1
                if cer_delta >= 0.1:
                    cer_delta_buckets["large_improve"] += 1
                else:
                    cer_delta_buckets["small_improve"] += 1
            elif cer_delta < -1e-6:
                n_regressed += 1
                if abs(cer_delta) >= 0.1:
                    cer_delta_buckets["large_regress"] += 1
                else:
                    cer_delta_buckets["small_regress"] += 1
            else:
                n_unchanged += 1

            detail = {
                "sample_id": r.get("sample_id", ""),
                "dataset": ds_key,
                "cer_ocr": round(cer_ocr, 6),
                "cer_llm": round(cer_llm, 6),
                "cer_delta": round(cer_delta, 6),
                "wer_ocr": round(wer_ocr, 6),
                "wer_llm": round(wer_llm, 6),
                "outcome": (
                    "improved" if cer_delta > 1e-6
                    else "regressed" if cer_delta < -1e-6
                    else "unchanged"
                ),
            }
            sample_details.append(detail)
            all_sample_details.append(detail)

        if not cer_values:
            continue

        avg_cer_ocr = sum(cer_ocr_values) / len(cer_ocr_values)
        avg_wer_ocr = sum(wer_ocr_values) / len(wer_ocr_values)
        avg_cer = sum(cer_values) / len(cer_values)
        avg_wer = sum(wer_values) / len(wer_values)
        reg_rate = n_regressed / len(cer_values) * 100
        imp_rate = n_improved / len(cer_values) * 100
        cer_delta_overall = avg_cer_ocr - avg_cer  # positive = improvement

        result: dict = {
            "dataset": ds_key,
            "n_samples": n_total,
            # Before correction (OCR) metrics
            "ocr_cer": round(avg_cer_ocr, 6),
            "ocr_wer": round(avg_wer_ocr, 6),
            # After correction (LLM) metrics
            "cer": round(avg_cer, 6),
            "wer": round(avg_wer, 6),
            # Delta (positive = improvement)
            "cer_delta": round(cer_delta_overall, 6),
            "cer_rel_improvement_pct": (
                round(cer_delta_overall / avg_cer_ocr * 100, 2)
                if avg_cer_ocr > 0 else 0.0
            ),
            # Error analysis
            "n_improved": n_improved,
            "n_regressed": n_regressed,
            "n_unchanged": n_unchanged,
            "improvement_rate": round(imp_rate, 2),
            "regression_rate": round(reg_rate, 2),
            "cer_delta_buckets": cer_delta_buckets,
        }

        # Phase 1 comparison (aggregate baseline)
        p1_data = (
            phase1_metrics.get("results_normal_only_no_diacritics", {}).get(ds_key)
            or phase1_metrics.get("results_all_samples_no_diacritics", {}).get(ds_key)
            or phase1_metrics.get("results_normal_only", {}).get(ds_key)
            or phase1_metrics.get("results_all_samples", {}).get(ds_key)
        )
        if p1_data:
            p1_cer = p1_data.get("cer", 0.0)
            result["p1_cer"] = round(p1_cer, 6)
            result["vs_p1_cer_delta"] = round(p1_cer - avg_cer, 6)
            result["vs_p1_cer_rel"] = (
                round((p1_cer - avg_cer) / p1_cer * 100, 2) if p1_cer > 0 else 0.0
            )

        # Phase 2 comparison
        p2_full = phase2_baselines.get(ds_key, {})
        p2_data = (
            p2_full.get("corrected_all_no_diacritics")
            or p2_full.get("corrected_no_diacritics")
            or p2_full.get("corrected_all")
            or p2_full.get("corrected", {})
        )
        if p2_data:
            p2_cer = p2_data.get("cer", 0.0)
            result["p2_cer"] = round(p2_cer, 6)
            result["vs_p2_cer_delta"] = round(p2_cer - avg_cer, 6)
            result["vs_p2_cer_rel"] = (
                round((p2_cer - avg_cer) / p2_cer * 100, 2) if p2_cer > 0 else 0.0
            )

        all_results[ds_key] = result

        # Print per-dataset results
        print(f"\n--- {ds_key} ({n_total} samples) ---")
        print(f"  Before correction (OCR):  CER={avg_cer_ocr:.4f}  WER={avg_wer_ocr:.4f}")
        cer_dir = "better" if cer_delta_overall > 0 else "worse" if cer_delta_overall < 0 else "same"
        print(
            f"  After  correction (LLM):  CER={avg_cer:.4f}  WER={avg_wer:.4f}  "
            f"[{cer_dir}, {result['cer_rel_improvement_pct']:+.1f}%]"
        )
        print(
            f"  Sample outcomes: improved={n_improved} ({imp_rate:.1f}%)  "
            f"regressed={n_regressed} ({reg_rate:.1f}%)  unchanged={n_unchanged}"
        )
        print(
            f"  CER delta buckets: "
            f"large_improve={cer_delta_buckets['large_improve']}  "
            f"small_improve={cer_delta_buckets['small_improve']}  "
            f"small_regress={cer_delta_buckets['small_regress']}  "
            f"large_regress={cer_delta_buckets['large_regress']}"
        )
        if "p1_cer" in result:
            direction = "better" if result["vs_p1_cer_delta"] > 0 else "worse"
            print(
                f"  vs Phase 1 (OCR):         CER {result['p1_cer']:.4f} -> {avg_cer:.4f} "
                f"({direction}, {abs(result['vs_p1_cer_rel']):.1f}%)"
            )
        if "p2_cer" in result:
            direction = "better" if result["vs_p2_cer_delta"] > 0 else "worse"
            print(
                f"  vs Phase 2 (zero-shot):   CER {result['p2_cer']:.4f} -> {avg_cer:.4f} "
                f"({direction}, {abs(result['vs_p2_cer_rel']):.1f}%)"
            )

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # Save detailed error analysis
    if all_sample_details:
        # Worst regressions (for inspection)
        worst_regressions = sorted(
            [d for d in all_sample_details if d["outcome"] == "regressed"],
            key=lambda x: x["cer_delta"],  # most negative first
        )[:20]
        # Best improvements
        best_improvements = sorted(
            [d for d in all_sample_details if d["outcome"] == "improved"],
            key=lambda x: x["cer_delta"],
            reverse=True,
        )[:20]

        n_total_all = len(all_sample_details)
        n_imp_all = sum(1 for d in all_sample_details if d["outcome"] == "improved")
        n_reg_all = sum(1 for d in all_sample_details if d["outcome"] == "regressed")
        n_unch_all = sum(1 for d in all_sample_details if d["outcome"] == "unchanged")
        avg_cer_ocr_all = sum(d["cer_ocr"] for d in all_sample_details) / n_total_all
        avg_cer_llm_all = sum(d["cer_llm"] for d in all_sample_details) / n_total_all

        error_analysis = {
            "summary": {
                "n_total": n_total_all,
                "n_improved": n_imp_all,
                "n_regressed": n_reg_all,
                "n_unchanged": n_unch_all,
                "improvement_rate_pct": round(n_imp_all / n_total_all * 100, 2),
                "regression_rate_pct": round(n_reg_all / n_total_all * 100, 2),
                "avg_cer_ocr": round(avg_cer_ocr_all, 6),
                "avg_cer_llm": round(avg_cer_llm_all, 6),
                "avg_cer_delta": round(avg_cer_ocr_all - avg_cer_llm_all, 6),
            },
            "worst_regressions": worst_regressions,
            "best_improvements": best_improvements,
            "per_dataset": {
                ds_key: {
                    "n_improved": r["n_improved"],
                    "n_regressed": r["n_regressed"],
                    "n_unchanged": r["n_unchanged"],
                    "improvement_rate_pct": r["improvement_rate"],
                    "regression_rate_pct": r["regression_rate"],
                    "ocr_cer": r["ocr_cer"],
                    "llm_cer": r["cer"],
                    "cer_delta": r["cer_delta"],
                    "cer_rel_improvement_pct": r["cer_rel_improvement_pct"],
                    "cer_delta_buckets": r["cer_delta_buckets"],
                }
                for ds_key, r in all_results.items()
            },
        }
        error_analysis_path = output_dir / "error_analysis.json"
        with open(error_analysis_path, "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
        logger.info("Saved error analysis to %s", error_analysis_path)

    # Overall summary
    if all_results:
        avg_cer_all = sum(r["cer"] for r in all_results.values()) / len(all_results)
        avg_cer_ocr_all_ds = sum(r["ocr_cer"] for r in all_results.values()) / len(all_results)
        avg_reg_all = sum(r["regression_rate"] for r in all_results.values()) / len(
            all_results
        )
        avg_imp_all = sum(r["improvement_rate"] for r in all_results.values()) / len(
            all_results
        )
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print(f"  Before correction (OCR): avg CER = {avg_cer_ocr_all_ds:.4f}")
        overall_delta = avg_cer_ocr_all_ds - avg_cer_all
        overall_dir = "better" if overall_delta > 0 else "worse"
        overall_rel = (overall_delta / avg_cer_ocr_all_ds * 100) if avg_cer_ocr_all_ds > 0 else 0.0
        print(
            f"  After  correction (LLM): avg CER = {avg_cer_all:.4f} "
            f"({overall_dir}, {overall_rel:+.1f}%)"
        )
        print(
            f"  Sample outcomes: improved={avg_imp_all:.1f}%  regressed={avg_reg_all:.1f}%"
        )
        p2_deltas = [
            r["vs_p2_cer_delta"]
            for r in all_results.values()
            if "vs_p2_cer_delta" in r
        ]
        if p2_deltas:
            avg_p2_delta = sum(p2_deltas) / len(p2_deltas)
            direction = "better" if avg_p2_delta > 0 else "worse"
            print(f"  vs Phase 2 (zero-shot): avg CER change = {avg_p2_delta:+.4f} ({direction})")
        print("=" * 70)
        if all_sample_details:
            print(f"  Detailed error analysis saved to: {error_analysis_path}")

    # ------------------------------------------------------------------
    # Sample report with categorised OCR / LLM / GT blocks
    # ------------------------------------------------------------------
    if corrections_path.exists():
        write_corrections_report(
            corrections_path=corrections_path,
            output_path=output_dir / "sample_report.txt",
            title="Crafted Prompt Evaluation",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode == "prepare":
        run_prepare(args, config)
    elif args.mode == "extract":
        run_extract(args, config)
    elif args.mode == "refine":
        run_refine(args, config)
    elif args.mode == "export":
        run_export(args, config)
    elif args.mode == "analyze":
        run_analyze(args, config)


if __name__ == "__main__":
    main()
