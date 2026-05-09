"""Remove multi-page PNG images from the Yarmouk training images directory.

Reads the training manifest, deletes all PNG files for docs with more than
one page, and rewrites the manifest keeping only single-page entries.

Usage:
    python scripts/cleanup_yarmouk_multipage.py [--dry-run]
"""

import argparse
import json
from pathlib import Path

_IMAGES = Path("data/ocr-raw-data/Yarmouk/images")


def cleanup_split(split: str, dry_run: bool) -> None:
    out_dir = _IMAGES / split
    manifest_path = _IMAGES / f"{split}_manifest.json"

    if not manifest_path.exists():
        print(f"  {manifest_path} not found, skipping.")
        return

    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = len(manifest)
    single_page = {did: pages for did, pages in manifest.items() if len(pages) == 1}
    multi_page  = {did: pages for did, pages in manifest.items() if len(pages) != 1}

    deleted_files = 0
    for did, pages in multi_page.items():
        for fname in pages:
            fpath = out_dir / fname
            if fpath.exists():
                if not dry_run:
                    fpath.unlink()
                deleted_files += 1
                print(f"  {'[DRY]' if dry_run else 'DEL'} {fpath.name}")

    if not dry_run:
        manifest_path.write_text(json.dumps(single_page, indent=2), encoding="utf-8")

    print(
        f"\n{split}: total={total}  single-page kept={len(single_page)}  "
        f"multi-page removed={len(multi_page)}  files deleted={deleted_files}"
    )
    if dry_run:
        print("  (dry run — no files changed)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["training", "testing", "both"])
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    splits = ["training", "testing"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"\nCleaning {split}...")
        cleanup_split(split, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
