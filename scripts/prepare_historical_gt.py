"""Prepare ground-truth files for the Historical Arabic Handwritten Text dataset.

The dataset ships as a zip containing 8 book directories, each with PNG page
images and DOCX ground-truth files.  The DOCX naming follows two patterns:

    Book1_00000022_B.docx  →  covers exactly  Book1_00000022_B.PNG  (1:1)
    Book1_00000126.docx    →  covers both      Book1_00000126_A.PNG
                                          and   Book1_00000126_B.PNG  (1:2)

This script:
  1. Extracts images and GT text from the zip.
  2. Converts each DOCX to a UTF-8 TXT file, one per PNG image.
  3. Writes to a flat gt/ directory and an images/ directory so the DataLoader
     can pair them by stem.

Usage:
    python scripts/prepare_historical_gt.py

Output layout:
    data/ocr-raw-data/Historical Arabic Handwritten Text Recognition Dataset/
        images/Book1/Book1_00000022_B.png   (lowercase PNG → renamed consistently)
        gt/Book1/Book1_00000022_B.txt
        gt/Book1/Book1_00000126_A.txt       (same text as Book1_00000126.docx)
        gt/Book1/Book1_00000126_B.txt
"""

import io
import re
import sys
import zipfile
from pathlib import Path


_DATASET_NAME = "Historical Arabic Handwritten Text Recognition Dataset"
_ZIP_PATH = (
    Path("data/ocr-raw-data")
    / _DATASET_NAME
    / _DATASET_NAME
    / (_DATASET_NAME + ".zip")
)
_OUT_ROOT = Path("data/ocr-raw-data") / _DATASET_NAME


def _extract_docx_text(docx_bytes: bytes) -> str:
    """Extract plain text from a DOCX byte blob without python-docx."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    # Collect all <w:t> text nodes
    fragments = re.findall(r"<w:t[^>]*>([^<]+)</w:t>", xml)
    text = " ".join(fragments)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _strip_doc_header(text: str) -> str:
    """Remove the DOCX identifier prefix (e.g. 'Book1_ 00000022 _ B ')."""
    # Pattern: starts with 'Book\d+[_ ]+\d+[_ ]+[A-B]? ' before Arabic content
    cleaned = re.sub(
        r"^Book\s*\d+[_ ]+\s*\d+[_ ]*\s*[ABab]?\s*",
        "",
        text,
    ).strip()
    return cleaned if cleaned else text


def _canonical_stem(raw_stem: str) -> str:
    """Normalise case: preserve book capitalisation, keep rest as-is."""
    # PNG stems come out lowercase; DOCX stems are mixed-case.
    # We normalise everything to the DOCX casing (which we can read from the archive).
    return raw_stem  # caller already has the docx stem as-is


def main() -> None:
    if not _ZIP_PATH.exists():
        sys.exit(f"ERROR: zip not found: {_ZIP_PATH}")

    img_root = _OUT_ROOT / "images"
    gt_root  = _OUT_ROOT / "gt"

    with zipfile.ZipFile(_ZIP_PATH) as z:
        names = z.namelist()

        # Collect PNG entries  → {lowercase_stem: entry_name}
        png_map: dict[str, str] = {}
        for n in names:
            if n.lower().endswith(".png"):
                stem = Path(n).stem.lower()
                png_map[stem] = n

        # Collect DOCX entries → {lowercase_stem: (entry_name, book)}
        docx_map: dict[str, tuple[str, str]] = {}
        for n in names:
            if n.endswith(".docx"):
                stem = Path(n).stem
                book = Path(n).parts[1]  # e.g. "Book1"
                docx_map[stem.lower()] = (n, book, stem)

        # Extract PNGs first
        for lower_stem, entry in sorted(png_map.items()):
            book = Path(entry).parts[1]
            # Use the DOCX-cased stem if we can find it, else keep original
            if lower_stem in docx_map:
                canonical = docx_map[lower_stem][2]
            else:
                # Could be an _A or _B variant whose parent docx has no suffix
                canonical = Path(entry).stem  # keep original casing from zip
            out_img = img_root / book / (canonical + ".png")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_img.write_bytes(z.read(entry))

        # Build GT: for each DOCX, write one TXT per corresponding PNG
        for lower_stem, (entry, book, orig_stem) in sorted(docx_map.items()):
            docx_bytes = z.read(entry)
            text = _strip_doc_header(_extract_docx_text(docx_bytes))
            if not text:
                print(f"  WARN: empty text in {entry}", file=sys.stderr)
                continue

            # Determine which PNGs this DOCX covers
            target_png_stems: list[str] = []

            if lower_stem in png_map:
                # Direct 1:1 match (e.g. Book1_00000022_B)
                target_png_stems.append(orig_stem)
            else:
                # Parent docx (no _A/_B suffix) → find _A and _B variants
                for variant in (lower_stem + "_a", lower_stem + "_b"):
                    if variant in png_map:
                        # Use canonical stem from the PNG entry
                        target_png_stems.append(Path(png_map[variant]).stem)

            if not target_png_stems:
                print(f"  WARN: no PNGs found for {entry}", file=sys.stderr)
                continue

            for png_stem in target_png_stems:
                gt_path = gt_root / book / (png_stem + ".txt")
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                gt_path.write_text(text, encoding="utf-8")

    # Summary
    img_count = sum(1 for _ in img_root.rglob("*.png"))
    gt_count  = sum(1 for _ in gt_root.rglob("*.txt"))
    print(f"Extracted {img_count} images -> {img_root}")
    print(f"Wrote {gt_count} GT files  -> {gt_root}")


if __name__ == "__main__":
    main()
