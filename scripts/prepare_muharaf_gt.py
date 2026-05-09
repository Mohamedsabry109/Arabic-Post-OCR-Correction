"""Extract ground-truth text from Muharaf dataset _tagged.json files.

Each image in public_data/ has a corresponding *_tagged.json that stores
per-line transcriptions with polygon coordinates and annotation tags.
This script concatenates the line texts in reading order (key order)
into one UTF-8 plain-text GT file per image.

Lines tagged as Crossed_out are excluded (struck-through content).

Raw data is left untouched; output goes to a separate gt/ directory.

Usage:
    python scripts/prepare_muharaf_gt.py

Output:
    data/ocr-raw-data/Muharaf/gt/{stem}.txt   (one per image, UTF-8)
"""

import json
import sys
from pathlib import Path


def extract_gt(tagged: dict) -> str:
    """Return plain text from a _tagged.json dict (line_1, line_2, …)."""
    lines = []
    for key in sorted(
        (k for k in tagged if k.startswith("line_")),
        key=lambda k: int(k.split("_", 1)[1]),
    ):
        entry = tagged[key]
        tags = entry.get("tags", {})
        if tags.get("Crossed_out", 0):
            continue
        text = entry.get("text", "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def main() -> None:
    src_dir = Path("data/ocr-raw-data/Muharaf/public_data")
    out_dir = Path("data/ocr-raw-data/Muharaf/gt")
    out_dir.mkdir(parents=True, exist_ok=True)

    tagged_files = sorted(src_dir.glob("*_tagged.json"))
    if not tagged_files:
        sys.exit(f"ERROR: no *_tagged.json files found in {src_dir}")

    written = skipped = 0
    for tf in tagged_files:
        # stem of the image: strip _tagged suffix
        img_stem = tf.stem[: -len("_tagged")]
        data = json.loads(tf.read_text(encoding="utf-8"))
        text = extract_gt(data)
        if not text:
            print(f"  WARN: empty GT for {tf.name}", file=sys.stderr)
            skipped += 1
            continue
        (out_dir / (img_stem + ".txt")).write_text(text, encoding="utf-8")
        written += 1

    print(f"Written: {written}  Skipped (empty): {skipped}")
    print(f"Output:  {out_dir}")


if __name__ == "__main__":
    main()
