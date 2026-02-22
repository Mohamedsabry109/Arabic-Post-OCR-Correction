#!/usr/bin/env python3
"""Generate train/validation splits for PATS-A01 datasets.

Scans the OCR directory for each font, collects all sample IDs,
shuffles with a fixed seed, and writes ./data/pats_splits.json.

Run once and commit the output — the split file is the source of truth.

Usage:
    python scripts/generate_pats_splits.py
    python scripts/generate_pats_splits.py --val-ratio 0.2 --seed 42
    python scripts/generate_pats_splits.py --ocr-root ./data/ocr-results/qaari-results
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
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of samples for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--ocr-root", type=Path, default=_DEFAULT_OCR_ROOT,
                        help="Root dir containing pats-a01-data/")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                        help="Output JSON file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pats_root = args.ocr_root / "pats-a01-data"

    if not pats_root.exists():
        print(f"ERROR: PATS OCR directory not found: {pats_root}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    splits: dict[str, dict[str, list[str]]] = {}

    for font in FONTS:
        font_dir = pats_root / f"A01-{font}"
        if not font_dir.exists():
            print(f"WARNING: {font_dir} not found, skipping.", file=sys.stderr)
            continue

        sample_ids = sorted(p.stem for p in font_dir.glob("*.txt"))
        if not sample_ids:
            print(f"WARNING: No .txt files in {font_dir}, skipping.", file=sys.stderr)
            continue

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
        "splits": splits,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSplit file written to: {args.output}")


if __name__ == "__main__":
    main()
