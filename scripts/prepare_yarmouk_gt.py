"""Extract ground-truth text from Yarmouk dataset HTML files.

Yarmouk scans Wikipedia articles as PDFs and pairs them with HTML source
files (the GT).  This script strips HTML tags and saves per-document UTF-8
text files that the DataLoader can read directly.

Usage:
    python scripts/prepare_yarmouk_gt.py

Output layout:
    data/ocr-raw-data/Yarmouk/gt/training/{id}.txt
    data/ocr-raw-data/Yarmouk/gt/testing/{id}.txt
"""

import re
import sys
import html as html_lib
from pathlib import Path


def _strip_html(raw: str) -> str:
    """Strip HTML tags and unescape entities, return plain text."""
    # Remove <script> and <style> blocks entirely
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    raw = re.sub(r"<[^>]+>", " ", raw)
    # Unescape HTML entities (&amp; &nbsp; etc.)
    raw = html_lib.unescape(raw)
    # Collapse whitespace
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _read_html(path: Path) -> str:
    """Read an HTML file — the files declare cp1256 but are actually UTF-8."""
    for enc in ("utf-8", "utf-8-sig", "cp1256"):
        try:
            content = path.read_text(encoding=enc, errors="strict")
            # Quick sanity: must contain some Arabic Unicode
            if any("؀" <= c <= "ۿ" for c in content[:500]):
                return content
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort
    return path.read_text(encoding="utf-8", errors="replace")


def process_split(html_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    htm_files = list(html_dir.glob("*.htm")) + list(html_dir.glob("*.html"))
    if not htm_files:
        print(f"  WARNING: no .htm files found in {html_dir}", file=sys.stderr)
        return 0

    count = 0
    for htm_path in sorted(htm_files):
        raw = _read_html(htm_path)
        text = _strip_html(raw)
        if not text:
            print(f"  WARN: empty after stripping: {htm_path.name}", file=sys.stderr)
            continue
        out_path = out_dir / (htm_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        count += 1

    return count


def main() -> None:
    root = Path("data/ocr-raw-data/Yarmouk")

    splits = [
        (root / "Training" / "Training" / "HTML" / "html" / "html", "training"),
        (root / "Testing"  / "Testing"  / "HTML" / "html" / "html", "testing"),
    ]

    for html_dir, split_name in splits:
        out_dir = root / "gt" / split_name
        print(f"Processing {split_name}: {html_dir}")
        n = process_split(html_dir, out_dir)
        print(f"  -> wrote {n} GT files to {out_dir}")


if __name__ == "__main__":
    main()
