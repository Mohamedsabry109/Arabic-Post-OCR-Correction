"""Convert Yarmouk PDF scans to PNG images for Qaari OCR.

Each PDF (one Wikipedia article) is converted page-by-page at 300 DPI.
Output filenames: {doc_id}_p{N:02d}.png  (N=01, 02, …)

Raw PDFs are left untouched; PNGs go to a separate images/ directory.

A page manifest is written so the post-Qaari merge script knows which
page files belong to each document.

Usage:
    python scripts/convert_yarmouk_pdfs.py [--split training|testing|both]
                                            [--dpi 300]
                                            [--limit N]

Output layout:
    data/ocr-raw-data/Yarmouk/images/training/{doc_id}_p{N:02d}.png
    data/ocr-raw-data/Yarmouk/images/testing/{doc_id}_p{N:02d}.png
    data/ocr-raw-data/Yarmouk/images/training_manifest.json
    data/ocr-raw-data/Yarmouk/images/testing_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

from pdf2image import convert_from_path


_SCANNED = {
    "training": Path("data/ocr-raw-data/Yarmouk/Training/Training/Scanned/training/training"),
    "testing":  Path("data/ocr-raw-data/Yarmouk/Testing/Testing/Scanned/testing/testing"),
}
_OUT_BASE = Path("data/ocr-raw-data/Yarmouk/images")


def convert_split(split: str, dpi: int, limit: int | None) -> None:
    pdf_dir = _SCANNED[split]
    out_dir = _OUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _OUT_BASE / f"{split}_manifest.json"

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    # Load existing manifest so we can resume; also scan disk for existing PNGs
    manifest: dict[str, list[str]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Rebuild manifest from disk for any doc not already tracked
    for png in out_dir.glob("*.png"):
        # stem like "10012_p01" -> doc_id "10012"
        parts = png.stem.rsplit("_p", 1)
        if len(parts) == 2 and parts[1].isdigit():
            did = parts[0]
            if did not in manifest:
                manifest[did] = []
            fname = png.name
            if fname not in manifest[did]:
                manifest[did].append(fname)
    # Sort page lists
    for did in manifest:
        manifest[did] = sorted(manifest[did])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = len(pdfs)
    converted = skipped = 0

    for i, pdf_path in enumerate(pdfs, 1):
        doc_id = pdf_path.stem
        # Resume: skip if all pages already exist on disk
        if doc_id in manifest:
            existing = [out_dir / pg for pg in manifest[doc_id]]
            if all(p.exists() for p in existing):
                skipped += 1
                continue

        try:
            pages = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as exc:
            print(f"  WARN [{i}/{total}] {pdf_path.name}: {exc}", file=sys.stderr)
            continue

        page_names: list[str] = []
        for n, img in enumerate(pages, 1):
            fname = f"{doc_id}_p{n:02d}.png"
            img.save(out_dir / fname, "PNG")
            page_names.append(fname)

        manifest[doc_id] = page_names
        converted += 1

        if i % 100 == 0 or i == total:
            # Save manifest incrementally
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"  [{i}/{total}] converted={converted} skipped={skipped}")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{split}: {converted} converted, {skipped} skipped, manifest -> {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["training", "testing", "both"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="process at most N PDFs per split (for testing)")
    args = parser.parse_args()

    splits = ["training", "testing"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"\nConverting {split}...")
        convert_split(split, dpi=args.dpi, limit=args.limit)


if __name__ == "__main__":
    main()
