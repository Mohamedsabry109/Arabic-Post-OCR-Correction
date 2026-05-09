#!/usr/bin/env python3
"""
Clean up training samples from uploaded image directories.
Keeps only test/validation splits for Yarmouk, KHATT-Paragraph, and Muharaf.

Defaults assume paths as they appear on the local machine under data/.
Override with --yarmouk-dir, --khatt-dir, --muharaf-dir, and --splits-dir
to match wherever the directories live on the target machine (e.g. thunder).

Usage:
  # Preview what will be deleted (safe, no changes)
  python scripts/cleanup_training_images.py --dry-run

  # Actually delete training images
  python scripts/cleanup_training_images.py --delete

  # Override paths for a remote machine
  python scripts/cleanup_training_images.py --delete \\
      --yarmouk-dir /data/Yarmouk/images \\
      --khatt-dir /data/KHATT_Paragraph/proc_images \\
      --muharaf-dir /data/Muharaf/images \\
      --splits-dir /data/splits
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Set

_HERE = Path(__file__).resolve().parent.parent
_DATA = _HERE / "data" / "ocr-raw-data"

DEFAULT_YARMOUK_IMAGES = _DATA / "Yarmouk" / "images"
DEFAULT_KHATT_IMAGES   = _DATA / "KHATT_Paragraph" / "khatt-paragraphs-images" / "proc_images"
DEFAULT_MUHARAF_IMAGES = _DATA / "Muharaf" / "images"
DEFAULT_SPLITS_DIR     = _DATA  # split JSONs sit alongside their dataset folders here


def load_train_stems(splits_json: Path, split_key: str) -> Set[str]:
    with open(splits_json, encoding="utf-8") as f:
        data = json.load(f)
    return set(data["splits"][split_key])


def cleanup_yarmouk(images_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Delete the training/ subfolder inside images_dir. Returns (deleted_files, deleted_dirs)."""
    training_dir = images_dir / "training"
    if not training_dir.exists():
        print(f"  [Yarmouk] training/ not found at {training_dir} — skipping")
        return 0, 0

    file_count = sum(1 for _ in training_dir.rglob("*") if _.is_file())
    if dry_run:
        print(f"  [Yarmouk] Would delete {file_count} files in {training_dir}/")
    else:
        shutil.rmtree(training_dir)
        print(f"  [Yarmouk] Deleted {file_count} files from {training_dir}/")
    return file_count, 1


def cleanup_flat_dir(label: str, images_dir: Path, train_stems: Set[str],
                     extensions: tuple, dry_run: bool) -> int:
    """Delete image files whose stem is in train_stems. Returns deleted count."""
    if not images_dir.exists():
        print(f"  [{label}] Directory not found: {images_dir} — skipping")
        return 0

    to_delete = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions and f.stem in train_stems
    ]

    if not to_delete:
        print(f"  [{label}] No training images found to delete in {images_dir}")
        return 0

    if dry_run:
        print(f"  [{label}] Would delete {len(to_delete)} training images from {images_dir}/")
        for f in to_delete[:5]:
            print(f"            e.g. {f.name}")
        if len(to_delete) > 5:
            print(f"            ... and {len(to_delete) - 5} more")
    else:
        for f in to_delete:
            f.unlink()
        print(f"  [{label}] Deleted {len(to_delete)} training images from {images_dir}/")

    return len(to_delete)


def count_remaining(images_dir: Path, extensions: tuple) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in extensions)


def main():
    parser = argparse.ArgumentParser(description="Remove training images, keep test/validation only")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Preview deletions without making changes")
    mode.add_argument("--delete",  action="store_true", help="Actually delete training images")

    parser.add_argument("--yarmouk-dir", type=Path, default=DEFAULT_YARMOUK_IMAGES,
                        help="Path to Yarmouk images/ directory")
    parser.add_argument("--khatt-dir",   type=Path, default=DEFAULT_KHATT_IMAGES,
                        help="Path to KHATT-Paragraph proc_images/ directory")
    parser.add_argument("--muharaf-dir", type=Path, default=DEFAULT_MUHARAF_IMAGES,
                        help="Path to Muharaf images/ directory")
    parser.add_argument("--splits-dir",  type=Path, default=DEFAULT_SPLITS_DIR,
                        help="Directory containing the *_splits.json files (or parent of dataset dirs)")

    args = parser.parse_args()
    dry_run = args.dry_run

    print("=" * 60)
    print("CLEANUP TRAINING IMAGES" + (" [DRY RUN]" if dry_run else " [DELETE]"))
    print("=" * 60)

    # --- Yarmouk ---
    # images/training/ and images/testing/ are pre-separated by folder
    print("\n[Yarmouk]")
    print(f"  Dir: {args.yarmouk_dir}")
    cleanup_yarmouk(args.yarmouk_dir, dry_run)
    if not dry_run:
        remaining = count_remaining(args.yarmouk_dir / "testing", (".png", ".jpg", ".jpeg"))
        print(f"  Remaining in testing/: {remaining} images")

    # --- KHATT Paragraph ---
    print("\n[KHATT-Paragraph]")
    print(f"  Dir: {args.khatt_dir}")
    khatt_splits_json = args.splits_dir / "KHATT_Paragraph" / "khatt_paragraph_splits.json"
    if not khatt_splits_json.exists():
        # Fallback: maybe splits-dir IS the KHATT_Paragraph dir
        khatt_splits_json = args.splits_dir / "khatt_paragraph_splits.json"
    if not khatt_splits_json.exists():
        print(f"  ERROR: Split file not found. Tried:\n"
              f"    {args.splits_dir / 'KHATT_Paragraph' / 'khatt_paragraph_splits.json'}\n"
              f"    {args.splits_dir / 'khatt_paragraph_splits.json'}")
    else:
        train_stems = load_train_stems(khatt_splits_json, "train")
        print(f"  Train stems: {len(train_stems)}")
        cleanup_flat_dir("KHATT-Paragraph", args.khatt_dir, train_stems, (".jpg", ".jpeg", ".png"), dry_run)
        if not dry_run:
            remaining = count_remaining(args.khatt_dir, (".jpg", ".jpeg", ".png"))
            print(f"  Remaining: {remaining} images")

    # --- Muharaf ---
    print("\n[Muharaf]")
    # The dataset ships as images/images/ — detect the actual flat dir automatically
    muharaf_images_dir = args.muharaf_dir
    nested = muharaf_images_dir / "images"
    if nested.is_dir() and any(nested.iterdir()):
        muharaf_images_dir = nested
        print(f"  Dir: {muharaf_images_dir} (auto-detected nested images/ subfolder)")
    else:
        print(f"  Dir: {muharaf_images_dir}")

    muharaf_splits_json = args.splits_dir / "Muharaf" / "muharaf_splits.json"
    if not muharaf_splits_json.exists():
        muharaf_splits_json = args.splits_dir / "muharaf_splits.json"
    if not muharaf_splits_json.exists():
        print(f"  ERROR: Split file not found. Tried:\n"
              f"    {args.splits_dir / 'Muharaf' / 'muharaf_splits.json'}\n"
              f"    {args.splits_dir / 'muharaf_splits.json'}")
    else:
        train_stems = load_train_stems(muharaf_splits_json, "train")
        print(f"  Train stems: {len(train_stems)}")
        cleanup_flat_dir("Muharaf", muharaf_images_dir, train_stems, (".jpg", ".jpeg", ".png"), dry_run)
        if not dry_run:
            remaining = count_remaining(muharaf_images_dir, (".jpg", ".jpeg", ".png"))
            print(f"  Remaining: {remaining} images")

    print("\n" + "=" * 60)
    print("Done." + (" (no changes made)" if dry_run else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
