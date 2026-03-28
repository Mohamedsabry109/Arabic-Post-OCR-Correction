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

Run once and commit the output -- the split file is the source of truth.

Usage:
    python scripts/generate_pats_splits.py                    # aligned (default)
    python scripts/generate_pats_splits.py --independent      # per-font shuffle
    python scripts/generate_pats_splits.py --val-ratio 0.15 --seed 42
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
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of samples for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--ocr-root", type=Path, default=_DEFAULT_OCR_ROOT,
                        help="Root dir containing pats-a01-data/")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help="Output JSON file path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aligned", action="store_true", default=True,
                      help="Same samples across all fonts (default)")
    mode.add_argument("--independent", action="store_true",
                      help="Independent per-font shuffle (different samples per font)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pats_root = args.ocr_root / "pats-a01-data"

    if not pats_root.exists():
        print(f"ERROR: PATS OCR directory not found: {pats_root}", file=sys.stderr)
        sys.exit(1)

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
    print(f"\nSplit mode: {split_mode}")

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

            splits[font] = {"train": train_ids, "validation": val_ids}
            print(f"{font}: {len(train_ids)} train, {len(val_ids)} validation")

    output = {
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "split_mode": split_mode,
        "common_sample_count": len(common_nums) if aligned else None,
        "splits": splits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSplit file written to: {args.output}")


if __name__ == "__main__":
    main()
