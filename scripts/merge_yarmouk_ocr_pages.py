"""Merge per-page Qaari OCR outputs into per-document files for Yarmouk.

Run this AFTER Qaari has processed all page images.

Expected Qaari output layout (one .txt per page image):
    data/ocr-results/qaari-results/yarmouk-data/{split}/pages/{doc_id}_p{N:02d}.txt

This script reads the page manifest written by convert_yarmouk_pdfs.py, then
concatenates pages in order to produce:
    data/ocr-results/qaari-results/yarmouk-data/{split}/{doc_id}.txt

Usage:
    python scripts/merge_yarmouk_ocr_pages.py [--split training|testing|both]
"""

import argparse
import json
import sys
from pathlib import Path


_MANIFEST_DIR = Path("data/ocr-raw-data/Yarmouk/images")
_OCR_BASE     = Path("data/ocr-results/qaari-results/yarmouk-data")


def merge_split(split: str) -> None:
    manifest_path = _MANIFEST_DIR / f"{split}_manifest.json"
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found: {manifest_path}\nRun convert_yarmouk_pdfs.py first.")

    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages_dir = _OCR_BASE / split / "pages"
    out_dir   = _OCR_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = missing_pages = skipped = 0

    for doc_id, page_fnames in sorted(manifest.items()):
        out_file = out_dir / f"{doc_id}.txt"
        if out_file.exists():
            skipped += 1
            continue

        page_texts: list[str] = []
        any_missing = False
        for fname in page_fnames:
            page_stem = Path(fname).stem          # e.g. 10012_p01
            page_txt  = pages_dir / f"{page_stem}.txt"
            if not page_txt.exists():
                print(f"  WARN: missing OCR page: {page_txt}", file=sys.stderr)
                any_missing = True
                continue
            text = page_txt.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                page_texts.append(text)

        if any_missing:
            missing_pages += 1

        if page_texts:
            out_file.write_text("\n".join(page_texts), encoding="utf-8")
            merged += 1

    print(f"{split}: merged={merged}, skipped(exist)={skipped}, had_missing_pages={missing_pages}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["training", "testing", "both"])
    args = parser.parse_args()

    splits = ["training", "testing"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"Merging {split}...")
        merge_split(split)


if __name__ == "__main__":
    main()
