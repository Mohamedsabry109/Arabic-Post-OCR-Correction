"""Generate yarmouk_splits.json from the single-page manifests.

Reads training_manifest.json and testing_manifest.json (produced by
convert_yarmouk_single_page.py) and writes a splits file that maps
split -> list of doc_ids that have exactly one page converted.

The testing split is capped at --test-size (default 250) randomly sampled
doc_ids from the full single-page testing pool (seed=42 for reproducibility).

Usage:
    python scripts/generate_yarmouk_splits.py
    python scripts/generate_yarmouk_splits.py --test-size 250
    python scripts/generate_yarmouk_splits.py --test-size 855   # use all
"""

import argparse
import json
import random
from pathlib import Path

_IMAGES = Path("data/ocr-raw-data/Yarmouk/images")
_OUT    = Path("data/ocr-raw-data/Yarmouk/yarmouk_splits.json")
_SEED   = 42


def load_single_page_ids(split: str) -> list[str]:
    manifest_path = _IMAGES / f"{split}_manifest.json"
    if not manifest_path.exists():
        print(f"  WARNING: {manifest_path} not found, returning empty list")
        return []
    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted(did for did, pages in manifest.items() if len(pages) == 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=250,
                        help="Max testing doc IDs to keep (random sample, seed=42). "
                             "Use 0 or a value >= pool size to keep all.")
    args = parser.parse_args()

    train_ids     = load_single_page_ids("training")
    test_ids_pool = load_single_page_ids("testing")

    if args.test_size > 0 and args.test_size < len(test_ids_pool):
        rng = random.Random(_SEED)
        test_ids = sorted(rng.sample(test_ids_pool, args.test_size))
        print(f"  testing pool: {len(test_ids_pool)} -> sampled {len(test_ids)} (seed={_SEED})")
    else:
        test_ids = test_ids_pool

    splits = {
        "seed": _SEED,
        "test_size": len(test_ids),
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
