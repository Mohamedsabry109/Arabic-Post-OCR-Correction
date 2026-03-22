#!/usr/bin/env python3
"""Select test samples (mix of hard and easy) for research iteration.

Scans all PATS-A01 train fonts + KHATT-train, computes per-sample CER
(no-diacritics), and outputs a JSON file with a configurable mix of
hard (highest CER) and easy (lowest CER) samples.

Usage:
    python scripts/select_test_samples.py --n 250 --output data/test_samples.json
    python scripts/select_test_samples.py --n 250 --hard-ratio 0.8 --easy-ratio 0.2
    python scripts/select_test_samples.py --n 250 --include-khatt
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError
from src.analysis.metrics import calculate_cer
from src.data.text_utils import normalise_arabic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select test samples (hard + easy) by CER.")
    p.add_argument("--config", type=Path, default=_PROJECT_ROOT / "configs" / "config.yaml")
    p.add_argument("--n", type=int, default=250, help="Total number of test samples to select")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data" / "test_samples.json")
    p.add_argument(
        "--hard-ratio", type=float, default=0.75,
        help="Fraction of samples from the hard (high CER) end (default: 0.75)",
    )
    p.add_argument(
        "--easy-ratio", type=float, default=0.25,
        help="Fraction of samples from the easy (low CER) end (default: 0.25)",
    )
    p.add_argument(
        "--min-gt-chars", type=int, default=5,
        help="Minimum GT characters (skip near-empty samples)",
    )
    p.add_argument(
        "--max-runaway-ratio", type=float, default=5.0,
        help="Skip samples where OCR length > ratio * GT length (Qaari repetition bug)",
    )
    p.add_argument(
        "--min-cer", type=float, default=0.01,
        help="Minimum CER to be considered (skip perfect/near-perfect)",
    )
    p.add_argument(
        "--include-khatt", action="store_true",
        help="Include KHATT-train samples (handwritten)",
    )
    return p.parse_args()


def _print_selection_stats(label: str, samples: list[dict]) -> None:
    """Print summary stats for a selection of samples."""
    if not samples:
        print(f"  {label}: 0 samples")
        return
    avg_cer = sum(s["cer"] for s in samples) / len(samples)
    min_cer = min(s["cer"] for s in samples)
    max_cer = max(s["cer"] for s in samples)
    print(f"  {label}: {len(samples)} samples, "
          f"CER range: {min_cer:.4f} - {max_cer:.4f} (avg {avg_cer:.4f})")


def main() -> None:
    args = parse_args()

    # Validate ratios
    total_ratio = args.hard_ratio + args.easy_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: --hard-ratio ({args.hard_ratio}) + --easy-ratio ({args.easy_ratio}) "
              f"= {total_ratio}, must sum to 1.0")
        sys.exit(1)

    n_hard = round(args.n * args.hard_ratio)
    n_easy = args.n - n_hard  # Use remainder to avoid rounding issues

    print(f"Selection plan: {args.n} total = {n_hard} hard ({args.hard_ratio:.0%}) "
          f"+ {n_easy} easy ({args.easy_ratio:.0%})")

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(config)

    # Collect train datasets: all PATS fonts + optionally KHATT
    datasets_cfg = config.get("datasets", [])
    train_datasets = []
    for ds in datasets_cfg:
        name = ds["name"]
        ds_type = ds.get("type", "")
        if ds_type == "PATS-A01" and ds.get("pats_split") == "train":
            train_datasets.append(name)
        elif ds_type == "KHATT" and ds.get("split") == "train" and args.include_khatt:
            train_datasets.append(name)

    print(f"Scanning {len(train_datasets)} datasets: {train_datasets}")

    # Score every sample
    all_scored: list[dict] = []

    for ds_key in train_datasets:
        try:
            samples = list(loader.iter_samples(ds_key))
        except DataError as exc:
            print(f"  Skipping {ds_key}: {exc}")
            continue

        n_skipped = 0
        for s in samples:
            gt_norm = normalise_arabic(s.gt_text, remove_diacritics=True)
            ocr_norm = normalise_arabic(s.ocr_text, remove_diacritics=True)

            # Skip near-empty
            if len(gt_norm) < args.min_gt_chars:
                n_skipped += 1
                continue

            # Skip runaway (Qaari repetition bug)
            if len(ocr_norm) / max(len(gt_norm), 1) > args.max_runaway_ratio:
                n_skipped += 1
                continue

            cer = calculate_cer(s.gt_text, s.ocr_text, strip_diacritics=True)

            # Skip perfect/near-perfect
            if cer < args.min_cer:
                continue

            all_scored.append({
                "sample_id": s.sample_id,
                "dataset": ds_key,
                "font": s.font,
                "cer": round(cer, 6),
                "gt_chars": len(gt_norm),
                "ocr_chars": len(ocr_norm),
            })

        print(f"  {ds_key}: {len(samples)} total, {n_skipped} skipped, "
              f"{len([x for x in all_scored if x['dataset'] == ds_key])} scored")

    if not all_scored:
        print("No samples matched the criteria!")
        sys.exit(1)

    # Sort by CER descending (hardest first)
    all_scored.sort(key=lambda x: x["cer"], reverse=True)

    # Select hard samples (highest CER) from the top
    hard_samples = all_scored[:n_hard]

    # Select easy samples (lowest CER) from the bottom, avoiding overlap
    remaining = all_scored[n_hard:]
    easy_samples = remaining[-n_easy:] if n_easy > 0 else []

    # Combine: hard first, then easy
    selected = hard_samples + easy_samples

    # Tag each sample with its category
    hard_ids = {s["sample_id"] for s in hard_samples}
    for s in selected:
        s["category"] = "hard" if s["sample_id"] in hard_ids else "easy"

    # Summary stats
    datasets_in_selection: dict[str, int] = {}
    for s in selected:
        ds = s["dataset"]
        datasets_in_selection[ds] = datasets_in_selection.get(ds, 0) + 1

    avg_cer = sum(s["cer"] for s in selected) / len(selected)
    min_cer = min(s["cer"] for s in selected)
    max_cer = max(s["cer"] for s in selected)

    print(f"\nSelected {len(selected)} test samples:")
    print(f"  Overall CER range: {min_cer:.4f} - {max_cer:.4f} (avg {avg_cer:.4f})")
    _print_selection_stats("Hard", hard_samples)
    _print_selection_stats("Easy", easy_samples)
    print(f"  By dataset:")
    for ds, count in sorted(datasets_in_selection.items()):
        print(f"    {ds}: {count}")

    # Build output
    output = {
        "meta": {
            "description": (
                f"Test samples: {n_hard} hard ({args.hard_ratio:.0%}) + "
                f"{n_easy} easy ({args.easy_ratio:.0%}) by CER (no-diacritics)"
            ),
            "total_selected": len(selected),
            "n_hard": len(hard_samples),
            "n_easy": len(easy_samples),
            "hard_ratio": args.hard_ratio,
            "easy_ratio": args.easy_ratio,
            "selection_criteria": {
                "min_gt_chars": args.min_gt_chars,
                "max_runaway_ratio": args.max_runaway_ratio,
                "min_cer": args.min_cer,
                "include_khatt": args.include_khatt,
            },
            "cer_range": {"min": min_cer, "max": max_cer, "avg": round(avg_cer, 6)},
            "by_dataset": datasets_in_selection,
            "source_datasets": train_datasets,
        },
        "sample_ids": [s["sample_id"] for s in selected],
        "samples": selected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
