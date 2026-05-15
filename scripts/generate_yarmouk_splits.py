"""Generate yarmouk_splits.json from the single-page manifests.

Reads training_manifest.json and testing_manifest.json (produced by
convert_yarmouk_single_page.py) and writes a splits file that maps
split -> list of doc_ids that have exactly one page converted.

The testing split is filtered to documents whose GT contains ONLY Arabic
(no Latin-alphabet words, i.e. no sequences of 2+ ASCII letters), then
capped at --test-size (default 200) randomly sampled doc_ids (seed=42).

Usage:
    python scripts/generate_yarmouk_splits.py
    python scripts/generate_yarmouk_splits.py --test-size 200
    python scripts/generate_yarmouk_splits.py --test-size 0   # keep all
"""

import argparse
import json
import random
import re
from pathlib import Path

_IMAGES  = Path("data/ocr-raw-data/Yarmouk/images")
_GT_ROOT = Path("data/ocr-raw-data/Yarmouk/gt")
_OUT     = Path("data/ocr-raw-data/Yarmouk/yarmouk_splits.json")
_SEED    = 42

# Any run of 2+ ASCII letters counts as an English word in the GT
_ENGLISH_RE = re.compile(r'[a-zA-Z]{2,}')


def load_single_page_ids(split: str) -> list[str]:
    manifest_path = _IMAGES / f"{split}_manifest.json"
    if not manifest_path.exists():
        print(f"  WARNING: {manifest_path} not found, returning empty list")
        return []
    manifest: dict[str, list[str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted(did for did, pages in manifest.items() if len(pages) == 1)


def is_arabic_only(doc_id: str, split: str) -> bool:
    """Return True if the GT file for this doc has no Latin-alphabet words."""
    gt_path = _GT_ROOT / split / f"{doc_id}.txt"
    if not gt_path.exists():
        return False
    text = gt_path.read_text(encoding="utf-8")
    return not bool(_ENGLISH_RE.search(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=200,
                        help="Max testing doc IDs to keep (random sample, seed=42). "
                             "Use 0 to keep all Arabic-only docs.")
    args = parser.parse_args()

    train_ids     = load_single_page_ids("training")
    test_ids_pool = load_single_page_ids("testing")

    # Filter testing pool: keep only docs with Arabic-only GT
    arabic_pool = [did for did in test_ids_pool if is_arabic_only(did, "testing")]
    print(f"  testing pool: {len(test_ids_pool)} single-page docs")
    print(f"  Arabic-only GT: {len(arabic_pool)} "
          f"({len(test_ids_pool) - len(arabic_pool)} excluded for English words in GT)")

    if args.test_size > 0 and args.test_size < len(arabic_pool):
        rng = random.Random(_SEED)
        test_ids = sorted(rng.sample(arabic_pool, args.test_size))
        print(f"  sampled {len(test_ids)} from Arabic-only pool (seed={_SEED})")
    else:
        test_ids = arabic_pool
        print(f"  keeping all {len(test_ids)} Arabic-only docs")

    splits = {
        "seed": _SEED,
        "test_size": len(test_ids),
        "arabic_only_filter": True,
        "splits": {
            "training": train_ids,
            "testing":  test_ids,
        }
    }
    _OUT.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {_OUT}")
    print(f"  training: {len(train_ids)} single-page docs")
    print(f"  testing:  {len(test_ids)} docs (Arabic-only GT)")


if __name__ == "__main__":
    main()
