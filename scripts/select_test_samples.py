#!/usr/bin/env python3
"""Select stratified test samples for research evaluation.

Scans all PATS-A01 train fonts + KHATT-train, computes per-sample CER
(no-diacritics), buckets by difficulty (correct / easy / medium / hard),
and randomly selects from each bucket with font diversity.

CER boundaries (same as craft_prompt.py):
    correct:  CER = 0.0
    easy:     0 < CER <= 0.05
    medium:   0.05 < CER <= 0.25
    hard:     CER > 0.25

Usage:
    python scripts/select_test_samples.py --n 250 --include-khatt
    python scripts/select_test_samples.py --n-correct 20 --n-easy 30 --n-medium 100 --n-hard 100
    python scripts/select_test_samples.py --seed 123
"""

import argparse
import json
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError
from src.analysis.metrics import calculate_cer
from src.data.text_utils import normalise_arabic

# CER boundaries for stratified sampling (shared with craft_prompt.py)
_CER_EASY_MAX = 0.05
_CER_MEDIUM_MAX = 0.25

# Default bucket sizes (total = 250)
_N_CORRECT = 25
_N_EASY = 25
_N_MEDIUM = 100
_N_HARD = 100

_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select stratified test samples by CER difficulty."
    )
    p.add_argument("--config", type=Path, default=_PROJECT_ROOT / "configs" / "config.yaml")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data" / "test_samples.json")
    p.add_argument(
        "--n", type=int, default=None,
        help=(
            "Total number of test samples. When provided, distributes across "
            "buckets proportionally (10%% correct, 10%% easy, 40%% medium, "
            "40%% hard). Overrides individual --n-* flags."
        ),
    )
    p.add_argument("--n-correct", type=int, default=_N_CORRECT, dest="n_correct",
                    help=f"Correct samples (CER = 0.0, default: {_N_CORRECT})")
    p.add_argument("--n-easy", type=int, default=_N_EASY, dest="n_easy",
                    help=f"Easy samples (CER <= 0.05, default: {_N_EASY})")
    p.add_argument("--n-medium", type=int, default=_N_MEDIUM, dest="n_medium",
                    help=f"Medium samples (0.05 < CER <= 0.25, default: {_N_MEDIUM})")
    p.add_argument("--n-hard", type=int, default=_N_HARD, dest="n_hard",
                    help=f"Hard samples (CER > 0.25, default: {_N_HARD})")
    p.add_argument(
        "--min-gt-chars", type=int, default=5, dest="min_gt_chars",
        help="Minimum GT characters (skip near-empty samples)",
    )
    p.add_argument(
        "--max-runaway-ratio", type=float, default=5.0, dest="max_runaway_ratio",
        help="Skip samples where OCR length > ratio * GT length (Qaari repetition bug)",
    )
    p.add_argument(
        "--include-khatt", action="store_true", dest="include_khatt",
        help="Include KHATT-train samples (handwritten)",
    )
    p.add_argument("--seed", type=int, default=_SEED, help=f"Random seed (default: {_SEED})")
    return p.parse_args()


def _select_with_font_diversity(
    pool: list[dict],
    target_n: int,
    rng: random.Random,
) -> list[dict]:
    """Pick up to target_n samples from pool with font diversity.

    First pass: one random sample per font/dataset.
    Second pass: fill remaining slots from the rest.
    """
    if not pool:
        return []

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

    return picked


def _print_bucket_stats(label: str, samples: list[dict]) -> None:
    """Print summary stats for a bucket of samples."""
    if not samples:
        print(f"  {label}: 0 samples")
        return
    avg_cer = sum(s["cer"] for s in samples) / len(samples)
    min_cer = min(s["cer"] for s in samples)
    max_cer = max(s["cer"] for s in samples)
    fonts = len({s.get("font") or s.get("dataset") for s in samples})
    print(f"  {label}: {len(samples)} samples, "
          f"CER range: {min_cer:.4f} - {max_cer:.4f} (avg {avg_cer:.4f}), "
          f"{fonts} fonts")


def main() -> None:
    args = parse_args()

    # If --n is provided, distribute proportionally across buckets
    if args.n is not None:
        args.n_correct = round(args.n * 0.10)
        args.n_easy = round(args.n * 0.10)
        args.n_medium = round(args.n * 0.40)
        args.n_hard = args.n - args.n_correct - args.n_easy - args.n_medium

    total_target = args.n_correct + args.n_easy + args.n_medium + args.n_hard

    print(f"Selection plan: {total_target} total = "
          f"{args.n_correct} correct + {args.n_easy} easy + "
          f"{args.n_medium} medium + {args.n_hard} hard")
    print(f"Seed: {args.seed}")

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(config)
    rng = random.Random(args.seed)

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

    # Score every sample into buckets
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
            print(f"  Skipping {ds_key}: {exc}")
            continue

        n_skipped = 0
        ds_counts = {"correct": 0, "easy": 0, "medium": 0, "hard": 0}

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

            entry = {
                "sample_id": s.sample_id,
                "dataset": ds_key,
                "font": s.font,
                "cer": round(cer, 6),
                "gt_chars": len(gt_norm),
                "ocr_chars": len(ocr_norm),
            }

            if cer == 0.0:
                buckets["correct"].append(entry)
                ds_counts["correct"] += 1
            elif cer <= _CER_EASY_MAX:
                buckets["easy"].append(entry)
                ds_counts["easy"] += 1
            elif cer <= _CER_MEDIUM_MAX:
                buckets["medium"].append(entry)
                ds_counts["medium"] += 1
            else:
                buckets["hard"].append(entry)
                ds_counts["hard"] += 1

        scored = sum(ds_counts.values())
        print(f"  {ds_key}: {len(samples)} total, {n_skipped} skipped, "
              f"{scored} scored "
              f"(C:{ds_counts['correct']} E:{ds_counts['easy']} "
              f"M:{ds_counts['medium']} H:{ds_counts['hard']})")

    print(f"\nPool sizes: correct={len(buckets['correct'])}, "
          f"easy={len(buckets['easy'])}, medium={len(buckets['medium'])}, "
          f"hard={len(buckets['hard'])}")

    # Stratified selection with font diversity
    targets = {
        "correct": args.n_correct,
        "easy": args.n_easy,
        "medium": args.n_medium,
        "hard": args.n_hard,
    }

    selected_by_bucket: dict[str, list[dict]] = {}
    for bucket_name, target_n in targets.items():
        selected_by_bucket[bucket_name] = _select_with_font_diversity(
            buckets[bucket_name], target_n, rng,
        )

    # Combine all buckets, tag each sample with its category
    selected: list[dict] = []
    for bucket_name in ("correct", "easy", "medium", "hard"):
        for s in selected_by_bucket[bucket_name]:
            s["category"] = bucket_name
            selected.append(s)

    if not selected:
        print("No samples matched the criteria!")
        sys.exit(1)

    # Summary stats
    datasets_in_selection: dict[str, int] = {}
    for s in selected:
        ds = s["dataset"]
        datasets_in_selection[ds] = datasets_in_selection.get(ds, 0) + 1

    all_cers = [s["cer"] for s in selected]
    avg_cer = sum(all_cers) / len(all_cers)
    min_cer = min(all_cers)
    max_cer = max(all_cers)

    print(f"\nSelected {len(selected)} test samples:")
    print(f"  Overall CER range: {min_cer:.4f} - {max_cer:.4f} (avg {avg_cer:.4f})")
    for bucket_name in ("correct", "easy", "medium", "hard"):
        label = bucket_name.capitalize()
        _print_bucket_stats(label, selected_by_bucket[bucket_name])
    print(f"  By dataset:")
    for ds, count in sorted(datasets_in_selection.items()):
        print(f"    {ds}: {count}")

    # Build output
    bucket_counts = {
        name: len(samples) for name, samples in selected_by_bucket.items()
    }
    output = {
        "meta": {
            "description": (
                f"Stratified test samples: {bucket_counts['correct']} correct + "
                f"{bucket_counts['easy']} easy + {bucket_counts['medium']} medium + "
                f"{bucket_counts['hard']} hard (by CER, no-diacritics)"
            ),
            "total_selected": len(selected),
            "by_category": bucket_counts,
            "cer_boundaries": {
                "correct": 0.0,
                "easy_max": _CER_EASY_MAX,
                "medium_max": _CER_MEDIUM_MAX,
            },
            "selection_criteria": {
                "min_gt_chars": args.min_gt_chars,
                "max_runaway_ratio": args.max_runaway_ratio,
                "include_khatt": args.include_khatt,
                "seed": args.seed,
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
