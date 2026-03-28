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

Run once and commit the output -- the split file is the source of truth.

Usage:
    python scripts/generate_pats_splits.py                    # aligned (default)
    python scripts/generate_pats_splits.py --independent      # per-font shuffle
    python scripts/generate_pats_splits.py --val-ratio 0.15 --seed 42
    python scripts/generate_pats_splits.py --skip-empty       # exclude empty OCR from val
    python scripts/generate_pats_splits.py --config configs/config.yaml  # read skip_empty_val
"""

import argparse
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OCR_ROOT = _PROJECT_ROOT / "data" / "ocr-results" / "qaari-results"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "ocr-raw-data" / "PATS_A01_Dataset" / "pats_splits.json"

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
            # Sample IDs are "{Font}_{number}" — extract the number.
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

    splits: dict[str, dict[str, list[str]]] = {}

    if aligned:
        # ---- Aligned mode: same numeric indices across all fonts ----
        common_nums = sorted(set.intersection(*font_nums.values()))
        print(f"Common samples across {len(font_nums)} fonts: {len(common_nums)}")

        rng = random.Random(args.seed)
        shuffled = common_nums[:]
        rng.shuffle(shuffled)

        n_val = max(1, round(len(shuffled) * args.val_ratio))
        val_nums = set(shuffled[:n_val])
        train_nums = set(shuffled[n_val:])

        # Move empty-OCR indices from validation to train
        if skip_empty and empty_indices:
            demoted = val_nums & empty_indices
            if demoted:
                val_nums -= demoted
                train_nums |= demoted
                print(f"  skip_empty_val: moved {len(demoted)} indices from val to train")

        for font in FONTS:
            if font not in font_nums:
                continue
            val_ids = sorted(f"{font}_{n}" for n in sorted(val_nums & font_nums[font]))
            train_ids = sorted(f"{font}_{n}" for n in sorted(train_nums & font_nums[font]))
            splits[font] = {"train": train_ids, "validation": val_ids}
            print(f"{font}: {len(train_ids)} train, {len(val_ids)} validation")
    else:
        # ---- Independent mode: each font shuffled separately ----
        common_nums = []  # not applicable
        rng = random.Random(args.seed)
        for font in FONTS:
            if font not in font_nums:
                continue
            sample_ids = sorted(f"{font}_{n}" for n in sorted(font_nums[font]))
            shuffled = sample_ids[:]
            rng.shuffle(shuffled)

            n_val = max(1, round(len(shuffled) * args.val_ratio))
            val_ids = sorted(shuffled[:n_val])
            train_ids = sorted(shuffled[n_val:])

            # Move empty-OCR samples from validation to train
            if skip_empty and font in empty_by_font:
                font_empty = empty_by_font[font]
                demoted = [sid for sid in val_ids
                           if int(sid.split("_", 1)[1]) in font_empty]
                if demoted:
                    val_ids = [sid for sid in val_ids if sid not in set(demoted)]
                    train_ids = sorted(train_ids + demoted)
                    print(f"  {font}: moved {len(demoted)} empty-OCR samples from val to train")

            splits[font] = {"train": train_ids, "validation": val_ids}
            print(f"{font}: {len(train_ids)} train, {len(val_ids)} validation")

    output = {
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "split_mode": split_mode,
        "skip_empty_val": skip_empty,
        "empty_indices_excluded": sorted(empty_indices) if (skip_empty and empty_indices) else None,
        "common_sample_count": len(common_nums) if aligned else None,
        "splits": splits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSplit file written to: {args.output}")


if __name__ == "__main__":
    main()
