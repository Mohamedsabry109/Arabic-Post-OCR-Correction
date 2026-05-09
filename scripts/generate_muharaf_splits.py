"""Generate train / validation splits for the Muharaf dataset.

Uses the official set0 split from public_1100_untrained/trials/:
    train      = 1,100 samples  (pretrain_train_1150.json)
    validation = 116 samples   (pretrain_valid_1150.json + pretrain_test_1150.json)

The validation set combines the official val (50) and test (66) so there is
a meaningful held-out set for OCR correction evaluation.

Usage:
    python scripts/generate_muharaf_splits.py

Output:
    data/ocr-raw-data/Muharaf/muharaf_splits.json
"""

import json
from pathlib import Path


_SET0 = Path(
    "data/ocr-raw-data/Muharaf/public_1100_untrained"
    "/trials/public_1100_untrained/set0"
)


def _stems_from_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Each entry: [json_path, image_path]
    return [Path(img_path).stem for _, img_path in data]


def main() -> None:
    train_stems = _stems_from_json(_SET0 / "pretrain_train_1150.json")
    val_stems   = _stems_from_json(_SET0 / "pretrain_valid_1150.json")
    test_stems  = _stems_from_json(_SET0 / "pretrain_test_1150.json")

    # Merge val + test into a single validation split
    validation_stems = sorted(set(val_stems + test_stems))
    train_stems_sorted = sorted(train_stems)

    # Sanity: no overlap
    overlap = set(train_stems_sorted) & set(validation_stems)
    if overlap:
        raise ValueError(f"Train/val overlap: {overlap}")

    result = {
        "splits": {
            "train":      train_stems_sorted,
            "validation": validation_stems,
        },
        "total":      len(train_stems_sorted) + len(validation_stems),
        "val_source": "set0 official val(50) + test(66) = 116",
    }

    out_path = Path("data/ocr-raw-data/Muharaf/muharaf_splits.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train:      {len(train_stems_sorted)}")
    print(f"Validation: {len(validation_stems)}  (val={len(val_stems)} + test={len(test_stems)})")
    print(f"Saved:      {out_path}")


if __name__ == "__main__":
    main()
