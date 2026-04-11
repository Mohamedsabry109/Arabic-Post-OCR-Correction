#!/usr/bin/env python3
"""Generate train/validation splits for PATS-A01 datasets.

Two modes:

* **aligned** (default, ``--aligned``): Splits are based on numeric sample
  indices shared across all fonts.  Every font gets the same text content in
  train and validation.  This is the correct mode for controlled font
  comparisons.

* **independent** (``--independent``): Each font is shuffled and split
  independently.  Fonts may have different text content in their validation
  sets.  Useful only if font-level pairing is not needed.

Empty-file filtering (``--skip-empty``):
  When enabled, any numeric index that has an empty OCR prediction file in
  *any* font is excluded from the validation set (moved to train).  This
  ensures every validation sample has a non-empty prediction across all fonts.
  Can also be set via ``data.skip_empty_val: true`` in config.yaml.

Kitab-bench leakage prevention (``--kitab-bench-gt``):
  When a path to the kitab-bench patsocr GT directory is supplied, the script
  reads every .txt file there, matches each text against the PATS ground-truth
  file, and **pins those numeric indices to the validation set**.  This
  guarantees that any sample used in kitab-bench evaluation never appears in
  the training split, preventing data leakage.  Pinned indices are filled
  first; remaining validation slots (up to ``--val-ratio``) are drawn
  randomly from the non-pinned pool.  The default path
  ``data/kitab-bench/patsocr/gt`` is used automatically if it exists.

Run once and commit the output -- the split file is the source of truth.

Usage:
    python scripts/generate_pats_splits.py                    # aligned (default)
    python scripts/generate_pats_splits.py --independent      # per-font shuffle
    python scripts/generate_pats_splits.py --val-ratio 0.15 --seed 42
    python scripts/generate_pats_splits.py --skip-empty       # exclude empty OCR from val
    python scripts/generate_pats_splits.py --config configs/config.yaml  # read skip_empty_val
    python scripts/generate_pats_splits.py --kitab-bench-gt data/kitab-bench/patsocr/gt
    python scripts/generate_pats_splits.py --no-kitab-bench   # disable auto-detection
"""

import argparse
import codecs
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OCR_ROOT = _PROJECT_ROOT / "data" / "ocr-results" / "qaari-results"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "ocr-raw-data" / "PATS_A01_Dataset" / "pats_splits.json"
_DEFAULT_KB_GT = _PROJECT_ROOT / "data" / "kitab-bench" / "patsocr" / "gt"
# PATS GT file used for kitab-bench text matching (all fonts share the same text)
_PATS_GT_FILE = _PROJECT_ROOT / "data" / "ocr-raw-data" / "PATS_A01_Dataset" / "A01-AkhbarText.txt"
_PATS_GT_ENCODING = "cp1256"

FONTS = [
    "Akhbar", "Andalus", "Arial", "Naskh",
    "Simplified", "Tahoma", "Thuluth", "Traditional",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PATS-A01 train/val splits")
    parser.add_argument("--config", type=Path, default=None,
                        help="Config YAML to read data.skip_empty_val from")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of samples for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--ocr-root", type=Path, default=_DEFAULT_OCR_ROOT,
                        help="Root dir containing pats-a01-data/")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help="Output JSON file path")
    parser.add_argument(
        "--skip-empty", action="store_true", default=None,
        help=(
            "Exclude indices with empty OCR prediction files (any font) from "
            "validation. Also read from config data.skip_empty_val."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aligned", action="store_true", default=True,
                      help="Same samples across all fonts (default)")
    mode.add_argument("--independent", action="store_true",
                      help="Independent per-font shuffle (different samples per font)")

    # Kitab-bench leakage prevention
    kb_group = parser.add_mutually_exclusive_group()
    kb_group.add_argument(
        "--kitab-bench-gt", type=Path, default=None, metavar="DIR",
        help=(
            "Path to kitab-bench patsocr GT directory (one .txt per sample). "
            "Indices matching these texts are pinned to validation to prevent "
            "leakage. Auto-detected from the default location if it exists."
        ),
    )
    kb_group.add_argument(
        "--no-kitab-bench", action="store_true", default=False,
        help="Disable auto-detection of the default kitab-bench GT path.",
    )
    return parser.parse_args()


def _find_empty_ocr_indices(pats_root: Path, fonts: list[str]) -> dict[str, set[int]]:
    """Return {font: set of numeric indices} where the OCR file is empty."""
    empty: dict[str, set[int]] = {}
    for font in fonts:
        font_dir = pats_root / f"A01-{font}"
        if not font_dir.exists():
            continue
        font_empty: set[int] = set()
        for p in font_dir.glob("*.txt"):
            parts = p.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                if p.stat().st_size == 0:
                    font_empty.add(int(parts[1]))
        if font_empty:
            empty[font] = font_empty
    return empty


def _load_kitab_bench_pinned_indices(kb_gt_dir: Path, pats_gt_file: Path) -> set[int]:
    """Return the set of PATS numeric indices that appear in kitab-bench evaluation.

    Reads every .txt file in ``kb_gt_dir``, matches its text content against
    the PATS ground-truth file (one line per sample, 1-indexed), and returns
    the set of matched numeric indices.  Unmatched samples trigger a warning.
    """
    if not kb_gt_dir.exists():
        print(f"WARNING: kitab-bench GT dir not found: {kb_gt_dir}", file=sys.stderr)
        return set()
    if not pats_gt_file.exists():
        print(f"WARNING: PATS GT file not found: {pats_gt_file}", file=sys.stderr)
        return set()

    # Build text -> 1-indexed line number map from PATS GT
    with codecs.open(pats_gt_file, encoding=_PATS_GT_ENCODING) as f:
        pats_lines = [line.rstrip("\n").strip() for line in f.readlines()]
    text_to_num: dict[str, int] = {}
    for i, text in enumerate(pats_lines):
        if text:  # skip empty lines
            text_to_num[text] = i + 1  # 1-indexed sample id

    # Read kitab-bench GT files
    kb_files = sorted(kb_gt_dir.glob("*.txt"))
    pinned: set[int] = set()
    unmatched: list[str] = []
    for kb_file in kb_files:
        text = kb_file.read_text(encoding="utf-8").strip()
        if text in text_to_num:
            pinned.add(text_to_num[text])
        else:
            unmatched.append(kb_file.name)

    print(
        f"Kitab-bench: {len(kb_files)} samples -> "
        f"{len(pinned)} unique PATS indices pinned to validation"
    )
    if unmatched:
        print(
            f"  WARNING: {len(unmatched)} kitab-bench samples had no match in PATS GT "
            f"(first 5: {unmatched[:5]})",
            file=sys.stderr,
        )
    return pinned


def main() -> None:
    args = parse_args()
    pats_root = args.ocr_root / "pats-a01-data"

    if not pats_root.exists():
        print(f"ERROR: PATS OCR directory not found: {pats_root}", file=sys.stderr)
        sys.exit(1)

    # Resolve skip_empty: CLI flag takes priority, then config, then False
    skip_empty = args.skip_empty
    if skip_empty is None and args.config is not None:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        skip_empty = cfg.get("data", {}).get("skip_empty_val", False)
    skip_empty = bool(skip_empty)

    # Resolve kitab-bench GT path
    kb_gt_dir: Path | None = None
    if not args.no_kitab_bench:
        if args.kitab_bench_gt is not None:
            kb_gt_dir = args.kitab_bench_gt
        elif _DEFAULT_KB_GT.exists():
            kb_gt_dir = _DEFAULT_KB_GT
            print(f"Auto-detected kitab-bench GT: {kb_gt_dir}")

    # Step 1: Collect numeric sample indices from ALL fonts and intersect.
    # This ensures the split covers only samples present in every font.
    font_nums: dict[str, set[int]] = {}
    for font in FONTS:
        font_dir = pats_root / f"A01-{font}"
        if not font_dir.exists():
            print(f"WARNING: {font_dir} not found, skipping.", file=sys.stderr)
            continue
        nums = set()
        for p in font_dir.glob("*.txt"):
            # Sample IDs are "{Font}_{number}" -- extract the number.
            parts = p.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                nums.add(int(parts[1]))
        if not nums:
            print(f"WARNING: No .txt files in {font_dir}, skipping.", file=sys.stderr)
            continue
        font_nums[font] = nums
        print(f"{font}: {len(nums)} samples found")

    if not font_nums:
        print("ERROR: No fonts with samples found.", file=sys.stderr)
        sys.exit(1)

    aligned = not args.independent
    split_mode = "aligned" if aligned else "independent"
    print(f"\nSplit mode: {split_mode}, skip_empty_val: {skip_empty}")

    # Detect empty OCR prediction files across fonts
    empty_by_font = _find_empty_ocr_indices(pats_root, list(font_nums.keys()))
    # Union of all indices that have an empty file in any font
    empty_indices: set[int] = set()
    for font, nums in empty_by_font.items():
        empty_indices |= nums
        print(f"  {font}: {len(nums)} empty OCR files")
    if empty_indices:
        print(f"  Total unique indices with empty OCR in any font: {len(empty_indices)}")

    # Load kitab-bench pinned indices (must go to validation)
    kitab_pinned: set[int] = set()
    if kb_gt_dir is not None:
        kitab_pinned = _load_kitab_bench_pinned_indices(kb_gt_dir, _PATS_GT_FILE)

    splits: dict[str, dict[str, list[str]]] = {}

    if aligned:
        # ---- Aligned mode: same numeric indices across all fonts ----
        common_nums = sorted(set.intersection(*font_nums.values()))
        print(f"Common samples across {len(font_nums)} fonts: {len(common_nums)}")
        common_set = set(common_nums)

        # Kitab-bench pinned indices that are actually present in common pool
        pinned_val = kitab_pinned & common_set
        if kitab_pinned - common_set:
            print(
                f"  WARNING: {len(kitab_pinned - common_set)} kitab-bench indices not in "
                f"common pool (absent from some fonts) -- skipped",
                file=sys.stderr,
            )

        # Remove empty-OCR indices from pinned_val if skip_empty is set
        if skip_empty and empty_indices:
            demoted_pinned = pinned_val & empty_indices
            if demoted_pinned:
                print(
                    f"  skip_empty_val: {len(demoted_pinned)} kitab-bench indices have "
                    f"empty OCR -- they remain in validation anyway (kitab-bench priority)",
                    file=sys.stderr,
                )
                # Kitab-bench takes priority: keep them in val even if OCR is empty

        # Determine additional val slots beyond pinned set
        n_val_target = max(1, round(len(common_nums) * args.val_ratio))
        non_pinned = sorted(common_set - pinned_val)

        rng = random.Random(args.seed)
        shuffled_non_pinned = non_pinned[:]
        rng.shuffle(shuffled_non_pinned)

        n_val_extra = max(0, n_val_target - len(pinned_val))
        extra_val_candidates = shuffled_non_pinned[:n_val_extra]
        remaining_candidates = shuffled_non_pinned[n_val_extra:]

        # From the extra candidates, demote empty-OCR indices to train if skip_empty
        if skip_empty and empty_indices:
            demoted_extra = [n for n in extra_val_candidates if n in empty_indices]
            extra_val_candidates = [n for n in extra_val_candidates if n not in empty_indices]
            remaining_candidates = remaining_candidates + demoted_extra
            if demoted_extra:
                print(f"  skip_empty_val: moved {len(demoted_extra)} extra indices from val to train")

        val_nums = pinned_val | set(extra_val_candidates)
        train_nums = common_set - val_nums

        if kitab_pinned:
            print(
                f"Validation: {len(val_nums)} total "
                f"({len(pinned_val)} kitab-bench pinned + {len(extra_val_candidates)} random)"
            )
            print(f"Train: {len(train_nums)}")
        else:
            demoted_total = val_nums & empty_indices if (skip_empty and empty_indices) else set()
            if skip_empty and empty_indices:
                demoted = (set(shuffled_non_pinned[:n_val_extra + len(remaining_candidates)])
                           & empty_indices & val_nums)
            print(f"Validation: {len(val_nums)}, Train: {len(train_nums)}")

        for font in FONTS:
            if font not in font_nums:
                continue
            val_ids = sorted(f"{font}_{n}" for n in sorted(val_nums & font_nums[font]))
            train_ids = sorted(f"{font}_{n}" for n in sorted(train_nums & font_nums[font]))
            splits[font] = {"train": train_ids, "validation": val_ids}
            print(f"{font}: {len(train_ids)} train, {len(val_ids)} validation")

    else:
        # ---- Independent mode: each font shuffled separately ----
        # kitab_pinned indices are pinned to val for every font (shared text content)
        rng = random.Random(args.seed)
        for font in FONTS:
            if font not in font_nums:
                continue

            font_set = font_nums[font]
            pinned_val = kitab_pinned & font_set

            n_val_target = max(1, round(len(font_set) * args.val_ratio))
            non_pinned = sorted(font_set - pinned_val)
            shuffled_non_pinned = non_pinned[:]
            rng.shuffle(shuffled_non_pinned)

            n_val_extra = max(0, n_val_target - len(pinned_val))
            extra_val = shuffled_non_pinned[:n_val_extra]

            # Move empty-OCR samples from extra val to train
            if skip_empty and font in empty_by_font:
                font_empty = empty_by_font[font]
                demoted = [n for n in extra_val if n in font_empty]
                extra_val = [n for n in extra_val if n not in font_empty]
                if demoted:
                    print(f"  {font}: moved {len(demoted)} empty-OCR samples from val to train")

            val_nums = pinned_val | set(extra_val)
            train_nums = font_set - val_nums

            val_ids = sorted(f"{font}_{n}" for n in sorted(val_nums))
            train_ids = sorted(f"{font}_{n}" for n in sorted(train_nums))
            splits[font] = {"train": train_ids, "validation": val_ids}
            print(f"{font}: {len(train_ids)} train, {len(val_ids)} validation")

    output = {
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "split_mode": split_mode,
        "skip_empty_val": skip_empty,
        "kitab_bench_pinned_count": len(kitab_pinned) if kitab_pinned else None,
        "kitab_bench_pinned_indices": sorted(kitab_pinned) if kitab_pinned else None,
        "empty_indices_excluded": sorted(empty_indices) if (skip_empty and empty_indices) else None,
        "common_sample_count": len(common_nums) if aligned else None,
        "splits": splits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSplit file written to: {args.output}")


if __name__ == "__main__":
    main()
