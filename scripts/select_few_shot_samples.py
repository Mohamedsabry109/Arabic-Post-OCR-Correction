#!/usr/bin/env python3
"""Select random few-shot samples (OCR + GT pairs) from training data.

Scans PATS-A01 train fonts (and optionally KHATT-train), excludes any
sample IDs already reserved for testing (data/test_samples.json), filters
by quality criteria, then randomly selects n samples — optionally stratified
by CER difficulty.

Output JSON contains the full OCR + GT text so samples can be directly
injected into prompts.

CER boundaries (same as select_test_samples.py):
    correct:  CER = 0.0
    easy:     0 < CER <= 0.05
    medium:   0.05 < CER <= 0.25
    hard:     CER > 0.25

Usage:
    # Simple random selection (skip correct samples)
    python scripts/select_few_shot_samples.py --n 20

    # Stratified selection
    python scripts/select_few_shot_samples.py --n-easy 5 --n-medium 10 --n-hard 5

    # Include KHATT and correct samples
    python scripts/select_few_shot_samples.py --n 30 --include-khatt --include-correct

    # Custom output path and seed
    python scripts/select_few_shot_samples.py --n 20 --output data/my_few_shot.json --seed 7
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

# CER boundaries (shared with select_test_samples.py and craft_prompt.py)
_CER_EASY_MAX = 0.05
_CER_MEDIUM_MAX = 0.25

_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select random few-shot samples (OCR+GT pairs) from training data."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "config.yaml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "few_shot_samples.json",
        help="Output JSON file (default: data/few_shot_samples.json).",
    )
    p.add_argument(
        "--exclude",
        type=Path,
        default=_PROJECT_ROOT / "data" / "test_samples.json",
        dest="exclude",
        help=(
            "JSON file whose sample_ids are excluded (default: data/test_samples.json). "
            "Pass an empty string to disable exclusion."
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Total number of few-shot samples. When provided, distributes across "
            "buckets proportionally (0%% correct, 25%% easy, 37.5%% medium, "
            "37.5%% hard). Correct bucket is excluded unless --include-correct is set."
        ),
    )
    p.add_argument("--n-correct", type=int, default=0, dest="n_correct",
                   help="Correct samples (CER = 0.0, default: 0)")
    p.add_argument("--n-easy", type=int, default=5, dest="n_easy",
                   help="Easy samples (0 < CER <= 0.05, default: 5)")
    p.add_argument("--n-medium", type=int, default=10, dest="n_medium",
                   help="Medium samples (0.05 < CER <= 0.25, default: 10)")
    p.add_argument("--n-hard", type=int, default=5, dest="n_hard",
                   help="Hard samples (CER > 0.25, default: 5)")
    p.add_argument(
        "--include-correct",
        action="store_true",
        dest="include_correct",
        help="Include correct samples (CER = 0.0) in --n proportional distribution.",
    )
    p.add_argument(
        "--include-khatt",
        action="store_true",
        dest="include_khatt",
        help="Include KHATT-train samples (handwritten).",
    )
    p.add_argument(
        "--min-gt-chars",
        type=int,
        default=10,
        dest="min_gt_chars",
        help="Minimum GT characters after normalisation (default: 10).",
    )
    p.add_argument(
        "--max-gt-chars",
        type=int,
        default=300,
        dest="max_gt_chars",
        help="Maximum GT characters (keep examples concise, default: 300).",
    )
    p.add_argument(
        "--max-runaway-ratio",
        type=float,
        default=5.0,
        dest="max_runaway_ratio",
        help="Skip samples where OCR length > ratio * GT length (default: 5.0).",
    )
    p.add_argument("--seed", type=int, default=_SEED, help=f"Random seed (default: {_SEED})")
    return p.parse_args()


def _load_excluded_ids(exclude_path: Path) -> set[str]:
    """Load sample IDs to exclude from a test_samples.json file."""
    if not exclude_path or not exclude_path.exists():
        return set()
    with open(exclude_path, encoding="utf-8") as f:
        data = json.load(f)
    ids = set(data.get("sample_ids", []))
    print(f"Loaded {len(ids)} excluded sample IDs from: {exclude_path}")
    return ids


def main() -> None:
    args = parse_args()

    # Resolve bucket targets
    if args.n is not None:
        if args.include_correct:
            args.n_correct = round(args.n * 0.10)
            args.n_easy    = round(args.n * 0.25)
            args.n_medium  = round(args.n * 0.325)
            args.n_hard    = args.n - args.n_correct - args.n_easy - args.n_medium
        else:
            args.n_correct = 0
            args.n_easy    = round(args.n * 0.25)
            args.n_medium  = round(args.n * 0.375)
            args.n_hard    = args.n - args.n_easy - args.n_medium

    total_target = args.n_correct + args.n_easy + args.n_medium + args.n_hard

    print(f"Selection plan: {total_target} total = "
          f"{args.n_correct} correct + {args.n_easy} easy + "
          f"{args.n_medium} medium + {args.n_hard} hard")
    print(f"Seed: {args.seed}")

    # Load exclusion list
    exclude_path = args.exclude if args.exclude and str(args.exclude) else None
    excluded_ids = _load_excluded_ids(exclude_path) if exclude_path else set()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(config)
    rng = random.Random(args.seed)

    # Collect train datasets
    datasets_cfg = config.get("datasets", [])
    train_datasets: list[str] = []
    for ds in datasets_cfg:
        name = ds["name"]
        ds_type = ds.get("type", "")
        if ds_type == "PATS-A01" and ds.get("pats_split") == "train":
            train_datasets.append(name)
        elif ds_type == "KHATT" and ds.get("split") == "train" and args.include_khatt:
            train_datasets.append(name)

    print(f"Scanning {len(train_datasets)} datasets: {train_datasets}")

    # Collect candidates per bucket
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
            # Skip reserved test samples
            if s.sample_id in excluded_ids:
                continue

            gt_norm = normalise_arabic(s.gt_text, remove_diacritics=True)
            ocr_norm = normalise_arabic(s.ocr_text, remove_diacritics=True)

            # Skip near-empty
            if len(gt_norm) < args.min_gt_chars:
                n_skipped += 1
                continue

            # Skip very long (few-shot examples should be concise)
            if len(gt_norm) > args.max_gt_chars:
                n_skipped += 1
                continue

            # Skip runaway OCR (repetition bug)
            if len(ocr_norm) / max(len(gt_norm), 1) > args.max_runaway_ratio:
                n_skipped += 1
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
        print(f"  {ds_key}: {len(samples)} total, {n_skipped} filtered out, "
              f"{scored} candidates "
              f"(C:{ds_counts['correct']} E:{ds_counts['easy']} "
              f"M:{ds_counts['medium']} H:{ds_counts['hard']})")

    print(f"\nPool sizes: correct={len(buckets['correct'])}, "
          f"easy={len(buckets['easy'])}, medium={len(buckets['medium'])}, "
          f"hard={len(buckets['hard'])}")

    # Random selection per bucket (shuffle then take first n)
    targets = {
        "correct": args.n_correct,
        "easy": args.n_easy,
        "medium": args.n_medium,
        "hard": args.n_hard,
    }

    selected_by_bucket: dict[str, list[dict]] = {}
    for bucket_name, target_n in targets.items():
        pool = buckets[bucket_name]
        rng.shuffle(pool)
        selected_by_bucket[bucket_name] = pool[:target_n]
        actual = len(selected_by_bucket[bucket_name])
        if actual < target_n:
            print(f"  WARNING: {bucket_name} bucket: wanted {target_n}, only {actual} available")

    # Combine and tag with category
    selected: list[dict] = []
    for bucket_name in ("correct", "easy", "medium", "hard"):
        for s in selected_by_bucket[bucket_name]:
            s["category"] = bucket_name
            selected.append(s)

    if not selected:
        print("No samples matched the criteria!")
        sys.exit(1)

    # Summary
    bucket_counts = {name: len(samples) for name, samples in selected_by_bucket.items()}
    all_cers = [s["cer"] for s in selected]
    avg_cer = sum(all_cers) / len(all_cers)

    print(f"\nSelected {len(selected)} few-shot samples:")
    for bucket_name in ("correct", "easy", "medium", "hard"):
        bucket_samples = selected_by_bucket[bucket_name]
        if not bucket_samples:
            continue
        b_cers = [s["cer"] for s in bucket_samples]
        print(f"  {bucket_name.capitalize()}: {len(bucket_samples)} samples, "
              f"CER range: {min(b_cers):.4f} - {max(b_cers):.4f} "
              f"(avg {sum(b_cers)/len(b_cers):.4f})")
    print(f"  Overall CER avg: {avg_cer:.4f}")

    # Build output
    output = {
        "meta": {
            "description": (
                f"Few-shot samples: {bucket_counts['correct']} correct + "
                f"{bucket_counts['easy']} easy + {bucket_counts['medium']} medium + "
                f"{bucket_counts['hard']} hard (stratified by CER, no-diacritics)"
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
                "max_gt_chars": args.max_gt_chars,
                "max_runaway_ratio": args.max_runaway_ratio,
                "include_khatt": args.include_khatt,
                "include_correct": args.include_correct,
                "excluded_ids_file": str(exclude_path) if exclude_path else None,
                "n_excluded": len(excluded_ids),
                "seed": args.seed,
            },
            "cer_range": {
                "min": min(all_cers),
                "max": max(all_cers),
                "avg": round(avg_cer, 6),
            },
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
