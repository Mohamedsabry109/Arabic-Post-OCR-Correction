"""Generate train / validation splits for the KHATT_Paragraph dataset.

Validation = all 449 KHATT-Paragraph stems whose paragraph ID (stem without
the _lineN suffix) appears anywhere in the KHATT line-level dataset (train or
validation).  This ensures no document overlap between KP-train and KHATT-line.

Training = all remaining 3,547 samples.

Usage:
    python scripts/generate_khatt_paragraph_splits.py

Output:
    data/ocr-raw-data/KHATT_Paragraph/khatt_paragraph_splits.json
"""

import json
from pathlib import Path


def _para_id(stem: str) -> str:
    """AHTD3A0001_Para2_3 -> AHTD3A0001_Para2  (strip line number)."""
    return stem.rsplit("_", 1)[0]


def main() -> None:
    kp_img_dir    = Path("data/ocr-raw-data/KHATT_Paragraph/khatt-paragraphs-images/proc_images")
    khatt_train   = Path("data/ocr-raw-data/KHATT/data/train/Training")
    khatt_val     = Path("data/ocr-raw-data/KHATT/data/validation/Validation")
    out_path      = Path("data/ocr-raw-data/KHATT_Paragraph/khatt_paragraph_splits.json")

    if not kp_img_dir.exists():
        raise FileNotFoundError(f"KHATT-Paragraph image dir not found: {kp_img_dir}")

    # Build set of paragraph IDs that appear in KHATT line-level (train + val)
    khatt_para_ids: set[str] = set()
    for khatt_dir in (khatt_train, khatt_val):
        if not khatt_dir.exists():
            raise FileNotFoundError(f"KHATT directory not found: {khatt_dir}")
        for p in khatt_dir.glob("*.txt"):
            khatt_para_ids.add(_para_id(p.stem))

    print(f"KHATT line-level unique paragraph IDs: {len(khatt_para_ids)}")

    # All KHATT-Paragraph stems
    all_stems = sorted(p.stem for p in kp_img_dir.glob("*.jpg"))
    print(f"KHATT-Paragraph total samples: {len(all_stems)}")

    # Validation = KP stems whose paragraph ID appears in KHATT line-level
    val_stems   = sorted(s for s in all_stems if s in khatt_para_ids)
    train_stems = sorted(s for s in all_stems if s not in khatt_para_ids)

    print(f"Validation (KHATT-overlap): {len(val_stems)}")
    print(f"Train:                      {len(train_stems)}")

    result = {
        "splits": {
            "train":      train_stems,
            "validation": val_stems,
        },
        "total":      len(all_stems),
        "val_source": "KHATT line-level paragraph-ID overlap (train+val)",
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
