#!/usr/bin/env python3
"""Remove new validation samples from results/phase2-training.

After regenerating pats_splits.json with kitab-bench indices pinned to
validation, this script removes those samples from the phase2-training
results so that training artifacts are not contaminated by evaluation data.

What it does:
  1. Reads the pinned validation indices from pats_splits.json.
  2. Filters corrections.jsonl (per-dataset) and the top-level
     corrections.jsonl, removing any PATS sample whose numeric index is
     in the new validation set.
  3. Filters inference_input.jsonl the same way.
  4. Re-runs scripts/analyze_training.py on the filtered top-level
     corrections.jsonl to regenerate analysis/ artifacts.

Note: per-dataset metrics.json files are NOT automatically recomputed here
because they require the full pipeline infrastructure.  Run
  python pipelines/run_phase2.py --mode analyze --results-dir results/phase2-training --force
after this script to refresh them.

Usage:
    python scripts/filter_phase2_training.py
    python scripts/filter_phase2_training.py --dry-run
    python scripts/filter_phase2_training.py --training-dir results/phase2-training
    python scripts/filter_phase2_training.py --splits-file data/ocr-raw-data/PATS_A01_Dataset/pats_splits.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAINING_DIR = _PROJECT_ROOT / "results" / "phase2-training"
_DEFAULT_SPLITS = (
    _PROJECT_ROOT / "data" / "ocr-raw-data" / "PATS_A01_Dataset" / "pats_splits.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove new validation samples from phase2-training results."
    )
    parser.add_argument(
        "--training-dir", type=Path, default=_DEFAULT_TRAINING_DIR,
        help="Phase 2 training results directory (default: results/phase2-training)",
    )
    parser.add_argument(
        "--splits-file", type=Path, default=_DEFAULT_SPLITS,
        help="pats_splits.json with kitab_bench_pinned_indices (default: auto)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be removed without writing any files.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip re-running analyze_training.py after filtering.",
    )
    return parser.parse_args()


def _load_pinned_val_indices(splits_file: Path) -> set[int]:
    """Load the kitab-bench pinned indices from pats_splits.json."""
    with open(splits_file, encoding="utf-8") as f:
        data = json.load(f)
    pinned = data.get("kitab_bench_pinned_indices")
    if not pinned:
        print("ERROR: pats_splits.json has no kitab_bench_pinned_indices.", file=sys.stderr)
        print("       Re-run scripts/generate_pats_splits.py first.", file=sys.stderr)
        sys.exit(1)
    return set(pinned)


_PATS_FONTS_LOWER = frozenset({
    "akhbar", "andalus", "arial", "naskh",
    "simplified", "tahoma", "thuluth", "traditional",
})


def _is_pats_val_sample(sample_id: str, pinned: set[int]) -> bool:
    """Return True if sample_id is a PATS sample whose index is now in val.

    PATS sample_ids follow the pattern "{Font}_{number}", e.g. "Akhbar_2176"
    or "andalus_2176" (Andalus uses lowercase in stored results).
    KHATT sample_ids look like "AHTD3A0165_Para3_2" -- they also end in a
    digit, so we must check the font prefix to avoid false positives.
    """
    parts = sample_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0].lower() in _PATS_FONTS_LOWER:
        return int(parts[1]) in pinned
    return False


def _filter_jsonl(
    path: Path,
    pinned: set[int],
    dry_run: bool,
    label: str,
) -> tuple[int, int]:
    """Filter a JSONL file in-place, returning (kept, removed) counts."""
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return 0, 0

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    kept: list[str] = []
    removed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue

        sample_id = record.get("sample_id", "")
        if _is_pats_val_sample(sample_id, pinned):
            removed += 1
        else:
            kept.append(line)

    print(f"  {label}: {len(lines)} -> {len(kept)} records (removed {removed})")
    if not dry_run and removed > 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept))
            if kept:
                f.write("\n")
    return len(kept), removed


def main() -> None:
    args = parse_args()

    if not args.training_dir.exists():
        print(f"ERROR: training dir not found: {args.training_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.splits_file.exists():
        print(f"ERROR: splits file not found: {args.splits_file}", file=sys.stderr)
        sys.exit(1)

    pinned = _load_pinned_val_indices(args.splits_file)
    print(f"Pinned validation indices: {len(pinned)}")
    if args.dry_run:
        print("DRY RUN -- no files will be modified.\n")

    total_removed = 0

    # --- Top-level files ---
    print("\n[Top-level]")
    _, r = _filter_jsonl(
        args.training_dir / "corrections.jsonl", pinned, args.dry_run, "corrections.jsonl"
    )
    total_removed += r
    _, r = _filter_jsonl(
        args.training_dir / "inference_input.jsonl", pinned, args.dry_run, "inference_input.jsonl"
    )
    total_removed += r

    # --- Per-dataset corrections.jsonl ---
    print("\n[Per-dataset corrections.jsonl]")
    pats_dataset_dirs = sorted(
        d for d in args.training_dir.iterdir()
        if d.is_dir() and d.name.startswith("PATS-A01-")
    )
    for ds_dir in pats_dataset_dirs:
        jsonl = ds_dir / "corrections.jsonl"
        _, r = _filter_jsonl(jsonl, pinned, args.dry_run, ds_dir.name)
        total_removed += r

    print(f"\nTotal records removed: {total_removed}")

    if total_removed == 0:
        print("Nothing to remove -- training data is already clean.")
        return

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to apply changes.")
        return

    # --- Re-run analyze_training.py ---
    if args.skip_analysis:
        print("\nSkipping analysis re-run (--skip-analysis).")
    else:
        print("\n[Re-generating analysis/ artifacts]")
        corrections = args.training_dir / "corrections.jsonl"
        analysis_dir = args.training_dir / "analysis"
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "analyze_training.py"),
            "--input", str(corrections),
            "--output-dir", str(analysis_dir),
            "--force",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(
                "  WARNING: analyze_training.py exited with non-zero status.",
                file=sys.stderr,
            )
        else:
            print("  Analysis artifacts regenerated successfully.")

    print(
        "\nNote: per-dataset metrics.json files still reflect the old sample counts.\n"
        "      To refresh them, run:\n"
        "        python pipelines/run_phase2.py --mode analyze "
        "--results-dir results/phase2-training --force"
    )


if __name__ == "__main__":
    main()
