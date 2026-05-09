"""Convert only single-page Yarmouk PDFs to PNG images.

For each PDF, checks the page count first (fast, no rendering) and skips
any document with more than one page.  Resume-safe: already-converted PNGs
are detected from the manifest and skipped.

Usage:
    python scripts/convert_yarmouk_single_page.py [--split training|testing|both]
                                                   [--dpi 300]

Output:
    data/ocr-raw-data/Yarmouk/images/{split}/{doc_id}_p01.png
    data/ocr-raw-data/Yarmouk/images/{split}_manifest.json   (single-page docs only)
    data/ocr-raw-data/Yarmouk/images/{split}_multipage.json  (skipped multi-page doc_ids)
"""

import argparse
import json
import sys
from pathlib import Path

from pdf2image import convert_from_path, pdfinfo_from_path


_SCANNED = {
    "training": Path("data/ocr-raw-data/Yarmouk/Training/Training/Scanned/training/training"),
    "testing":  Path("data/ocr-raw-data/Yarmouk/Testing/Testing/Scanned/testing/testing"),
}
_OUT_BASE = Path("data/ocr-raw-data/Yarmouk/images")


def convert_split(split: str, dpi: int) -> None:
    pdf_dir = _SCANNED[split]
    out_dir = _OUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path  = _OUT_BASE / f"{split}_manifest.json"
    multipage_path = _OUT_BASE / f"{split}_multipage.json"

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    total = len(pdfs)

    # Load existing manifests
    manifest: dict[str, list[str]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    multipage: list[str] = []
    if multipage_path.exists():
        multipage = json.loads(multipage_path.read_text(encoding="utf-8"))
    multipage_set = set(multipage)

    # Rebuild manifest from disk (resume after crash)
    for png in out_dir.glob("*.png"):
        parts = png.stem.rsplit("_p", 1)
        if len(parts) == 2 and parts[1].isdigit():
            did = parts[0]
            if did not in manifest:
                manifest[did] = []
            if png.name not in manifest[did]:
                manifest[did].append(png.name)
    for did in manifest:
        manifest[did] = sorted(manifest[did])

    converted = skipped_done = skipped_multi = 0

    for i, pdf_path in enumerate(pdfs, 1):
        doc_id = pdf_path.stem

        # Already converted
        if doc_id in manifest:
            png_path = out_dir / f"{doc_id}_p01.png"
            if png_path.exists():
                skipped_done += 1
                continue

        # Already known multi-page
        if doc_id in multipage_set:
            skipped_multi += 1
            continue

        # Check page count (fast)
        try:
            info = pdfinfo_from_path(str(pdf_path))
            n_pages = info.get("Pages", 0)
        except Exception as exc:
            print(f"  WARN [{i}/{total}] {pdf_path.name}: pdfinfo failed: {exc}", file=sys.stderr)
            continue

        if n_pages != 1:
            multipage.append(doc_id)
            multipage_set.add(doc_id)
            skipped_multi += 1
            continue

        # Convert the single page
        try:
            pages = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as exc:
            print(f"  WARN [{i}/{total}] {pdf_path.name}: convert failed: {exc}", file=sys.stderr)
            continue

        fname = f"{doc_id}_p01.png"
        pages[0].save(out_dir / fname, "PNG")
        manifest[doc_id] = [fname]
        converted += 1

        if i % 100 == 0 or i == total:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            multipage_path.write_text(json.dumps(sorted(multipage), indent=2), encoding="utf-8")
            print(
                f"  [{i}/{total}] converted={converted} "
                f"skipped_done={skipped_done} skipped_multi={skipped_multi}"
            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    multipage_path.write_text(json.dumps(sorted(multipage), indent=2), encoding="utf-8")
    single_count = len(manifest)
    print(
        f"{split}: {converted} newly converted, {skipped_done} already done, "
        f"{skipped_multi} multi-page skipped -> {single_count} single-page docs total"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="both", choices=["training", "testing", "both"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    splits = ["training", "testing"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"\nProcessing {split}...")
        convert_split(split, dpi=args.dpi)


if __name__ == "__main__":
    main()
