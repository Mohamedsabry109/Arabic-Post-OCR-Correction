"""Generate yarmouk_splits.json from the single-page manifests.

Reads training_manifest.json and testing_manifest.json (produced by
convert_yarmouk_single_page.py) and writes a splits file that maps
split -> list of doc_ids that have exactly one page converted.

Usage:
    python scripts/generate_yarmouk_splits.py
"""

import json
from pathlib import Path

_IMAGES = Path("data/ocr-raw-data/Yarmouk/images")
_OUT    = Path("data/ocr-raw-data/Yarmouk/yarmouk_splits.json")


def load_single_page_ids(split: str) -> list[str]:
    manifest_path = _IMAGES / f"{split}_manifest.json"
    if not manifest_path.exists():
        print(f"  WARNING: {manifest_path} not found, returning empty list")
        return []
    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Keep only docs where exactly 1 page was produced
    return sorted(did for did, pages in manifest.items() if len(pages) == 1)


def main() -> None:
    train_ids = load_single_page_ids("training")
    test_ids  = load_single_page_ids("testing")

    splits = {
        "splits": {
            "training": train_ids,
            "testing":  test_ids,
        }
    }
    _OUT.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {_OUT}")
    print(f"  training: {len(train_ids)} single-page docs")
    print(f"  testing:  {len(test_ids)} single-page docs")


if __name__ == "__main__":
    main()
