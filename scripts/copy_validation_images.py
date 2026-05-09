#!/usr/bin/env python3
"""
Copy validation/testing images for all datasets into data/validation-input-data/.

Output structure:
  validation-input-data/
    khatt/                    # 233 .tif (pre-split by folder)
    khatt-paragraph/          # 449 .jpg (split JSON, validation key)
    pats/akhbar/              # 460 .tif per font x 8 fonts (split JSON, per-font validation)
         andalus/
         arial/
         naskh/
         simplified/
         tahoma/
         thuluth/
         traditional/
    yarmouk/                  # 855 .png (pre-split by folder)
    muharaf/                  # 116 .jpg (split JSON, validation key)
    historical/book1/         # all 40 .png (no split)
              book2/ ...

Usage:
  python scripts/copy_validation_images.py --dry-run   # preview
  python scripts/copy_validation_images.py --copy      # execute
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Set

_HERE = Path(__file__).resolve().parent.parent
_RAW  = _HERE / "data" / "ocr-raw-data"
_OUT  = _HERE / "data" / "validation-input-data"

# Font name -> (subfolder under PATS_A01_Dataset, image prefix)
PATS_FONTS = {
    "Akhbar":     ("A01-Akhbar",     "Akhbar"),
    "Andalus":    ("A01-Andalus",    "Andalus"),
    "Arial":      ("A01-Arial",      "Arial"),
    "Naskh":      ("A01-Naskh",      "Naskh"),
    "Simplified": ("A01-Simplified", "Simplified"),
    "Tahoma":     ("A01-Tahoma",     "Tahoma"),
    "Thuluth":    ("A01-Thuluth",    "Thuluth"),
    "Traditional":("A01-Traditional","Traditional"),
}


def load_split(json_path: Path, split_key: str) -> Set[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return set(data["splits"][split_key])


def copy_files(src_files: list[Path], dst_dir: Path, dry_run: bool) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        dst = dst_dir / src.name
        if not dry_run:
            shutil.copy2(src, dst)
    return len(src_files)


def report(label: str, count: int, dst: Path, dry_run: bool):
    verb = "Would copy" if dry_run else "Copied"
    print(f"  [{label}] {verb} {count} files -> {dst.relative_to(_HERE)}/")


# ---------------------------------------------------------------------------

def do_khatt(dst_root: Path, dry_run: bool):
    src_dir = _RAW / "KHATT" / "data" / "validation" / "Validation"
    files = sorted(src_dir.glob("*.tif"))
    dst = dst_root / "khatt"
    n = copy_files(files, dst, dry_run)
    report("KHATT", n, dst, dry_run)


def do_khatt_paragraph(dst_root: Path, dry_run: bool):
    src_dir = _RAW / "KHATT_Paragraph" / "khatt-paragraphs-images" / "proc_images"
    splits_json = _RAW / "KHATT_Paragraph" / "khatt_paragraph_splits.json"

    val_stems = load_split(splits_json, "validation")
    files = [f for f in src_dir.iterdir()
             if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
             and f.stem in val_stems]
    files.sort()

    dst = dst_root / "khatt-paragraph"
    n = copy_files(files, dst, dry_run)
    report("KHATT-Paragraph", n, dst, dry_run)


def do_pats(dst_root: Path, dry_run: bool):
    splits_json = _RAW / "PATS_A01_Dataset" / "pats_splits.json"
    with open(splits_json, encoding="utf-8") as f:
        data = json.load(f)
    per_font_splits = data["splits"]

    total = 0
    for font_key, (folder, _prefix) in PATS_FONTS.items():
        val_stems_lower = {s.lower() for s in per_font_splits[font_key]["validation"]}
        src_dir = _RAW / "PATS_A01_Dataset" / folder
        files = [f for f in src_dir.glob("*.tif") if f.stem.lower() in val_stems_lower]
        files.sort()

        dst = dst_root / "pats" / font_key.lower()
        n = copy_files(files, dst, dry_run)
        report(f"PATS/{font_key}", n, dst, dry_run)
        total += n

    print(f"  [PATS total] {total} files across 8 fonts")


def do_yarmouk(dst_root: Path, dry_run: bool):
    src_dir = _RAW / "Yarmouk" / "images" / "testing"
    files = sorted(f for f in src_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif"))
    dst = dst_root / "yarmouk"
    n = copy_files(files, dst, dry_run)
    report("Yarmouk", n, dst, dry_run)


def do_muharaf(dst_root: Path, dry_run: bool):
    src_dir = _RAW / "Muharaf" / "images"
    # Dataset ships as images/images/ — detect nesting
    if (src_dir / "images").is_dir():
        src_dir = src_dir / "images"

    splits_json = _RAW / "Muharaf" / "muharaf_splits.json"
    val_stems = load_split(splits_json, "validation")

    files = [f for f in src_dir.iterdir()
             if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
             and f.stem in val_stems]
    files.sort()

    dst = dst_root / "muharaf"
    n = copy_files(files, dst, dry_run)
    report("Muharaf", n, dst, dry_run)


def do_historical(dst_root: Path, dry_run: bool):
    src_root = _RAW / "Historical Arabic Handwritten Text Recognition Dataset" / "images"
    total = 0
    for book_dir in sorted(src_root.iterdir()):
        if not book_dir.is_dir():
            continue
        files = sorted(book_dir.glob("*.png"))
        dst = dst_root / "historical" / book_dir.name.lower()
        n = copy_files(files, dst, dry_run)
        report(f"Historical/{book_dir.name}", n, dst, dry_run)
        total += n
    print(f"  [Historical total] {total} files")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Copy validation/testing images to data/validation-input-data/"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Preview without copying")
    mode.add_argument("--copy",    action="store_true", help="Actually copy files")
    parser.add_argument("--out", type=Path, default=_OUT,
                        help=f"Destination directory (default: {_OUT})")
    args = parser.parse_args()

    dry_run = args.dry_run
    dst_root = args.out

    print("=" * 60)
    print("COPY VALIDATION IMAGES" + (" [DRY RUN]" if dry_run else ""))
    print(f"Destination: {dst_root}")
    print("=" * 60)

    do_khatt(dst_root, dry_run)
    do_khatt_paragraph(dst_root, dry_run)
    do_pats(dst_root, dry_run)
    do_yarmouk(dst_root, dry_run)
    do_muharaf(dst_root, dry_run)
    do_historical(dst_root, dry_run)

    print("\n" + "=" * 60)
    print("Done." + (" (no files copied)" if dry_run else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
