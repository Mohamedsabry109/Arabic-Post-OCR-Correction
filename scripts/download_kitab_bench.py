"""Download KITAB-Bench image-to-text datasets from HuggingFace.

Saves images (PNG) and ground-truth text files under:
    ./data/kitab-bench/{dataset_name}/images/{idx}.png
    ./data/kitab-bench/{dataset_name}/gt/{idx}.txt

Skips datasets that overlap with existing PATS-A01 / KHATT data.
Supports resume: existing files are not re-downloaded.

Usage:
    python scripts/download_kitab_bench.py
    python scripts/download_kitab_bench.py --datasets hindawi adab
    python scripts/download_kitab_bench.py --list
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    # GT field verified from actual HF dataset columns on 2026-03-31.
    # name -> {hf_id, gt_field, description, samples}
    "hindawi": {
        "hf_id": "ahmedheakl/arocrbench_hindawi",
        "gt_field": "text",
        "desc": "Hindawi literary/academic book pages (200)",
    },
    "adab": {
        "hf_id": "ahmedheakl/arocrbench_adab",
        "gt_field": "text",
        "desc": "Classical Arabic literature (200)",
    },
    "muharaf": {
        "hf_id": "ahmedheakl/arocrbench_muharaf",
        "gt_field": "text",
        "desc": "Calligraphic/variant Arabic text (200)",
    },
    "onlinekhatt": {
        "hf_id": "ahmedheakl/arocrbench_onlinekhatt",
        "gt_field": "text",
        "desc": "Online handwritten Arabic (200)",
    },
    "historicalbooks": {
        "hf_id": "ahmedheakl/arocrbench_historicalbooks",
        "gt_field": "answer",
        "desc": "Scanned historical Arabic documents (10)",
    },
    "isippt": {
        "hf_id": "ahmedheakl/arocrbench_isippt",
        "gt_field": "text",
        "desc": "Arabic presentation slides (500)",
    },
    "arabicocr": {
        "hf_id": "ahmedheakl/arocrbench_arabicocr",
        "gt_field": "text",
        "desc": "General Arabic OCR corpus (50)",
    },
    "khattparagraph": {
        "hf_id": "ahmedheakl/arocrbench_khattparagraph",
        "gt_field": "answer",
        "desc": "Handwritten Arabic paragraphs (200)",
    },
    "patsocr": {
        "hf_id": "ahmedheakl/arocrbench_patsocr",
        "gt_field": "answer",
        "desc": "Typewritten Arabic / PATS (500)",
    },
    "khatt": {
        "hf_id": "ahmedheakl/arocrbench_khatt",
        "gt_field": "text",
        "desc": "Handwritten Arabic / KHATT (200)",
    },
    "synthesizear": {
        "hf_id": "ahmedheakl/arocrbench_synthesizear",
        "gt_field": "text",
        "desc": "Synthetically generated Arabic text (500)",
    },
    "historyar": {
        "hf_id": "ahmedheakl/arocrbench_historyar",
        "gt_field": "text",
        "desc": "Historical Arabic documents (200)",
    },
    "evarest": {
        "hf_id": "ahmedheakl/arocrbench_evarest",
        "gt_field": "text",
        "desc": "Restaurant/receipt Arabic text (800)",
    },
}

OUTPUT_ROOT = Path("./data/kitab-bench")


def list_datasets() -> None:
    """Print available datasets and exit."""
    print(f"{'Name':<20} {'GT Field':<10} {'Description'}")
    print("-" * 70)
    for name, info in DATASETS.items():
        print(f"{name:<20} {info['gt_field']:<10} {info['desc']}")


def download_dataset(name: str, info: dict) -> None:
    """Download a single KITAB-Bench dataset."""
    import datasets as hf_datasets

    hf_id = info["hf_id"]
    gt_field = info["gt_field"]

    ds_dir = OUTPUT_ROOT / name
    img_dir = ds_dir / "images"
    gt_dir = ds_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Downloading {name} ({hf_id}) ---")
    ds = hf_datasets.load_dataset(hf_id, split="train")
    total = len(ds)
    print(f"  Total samples: {total}")

    saved = 0
    skipped = 0

    for idx, sample in enumerate(ds):
        img_path = img_dir / f"{idx}.png"
        gt_path = gt_dir / f"{idx}.txt"

        # Resume: skip if both files exist and GT is non-empty
        if img_path.exists() and gt_path.exists() and gt_path.stat().st_size > 0:
            skipped += 1
            continue

        # Save image
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(str(img_path))

        # Save ground truth
        gt_text = sample.get(gt_field, "")
        if gt_text is None:
            gt_text = ""
        gt_path.write_text(gt_text, encoding="utf-8")

        saved += 1
        if (saved + skipped) % 100 == 0:
            print(f"  Progress: {saved + skipped}/{total} ({skipped} resumed, {saved} new)")

    print(f"  Done: {saved} saved, {skipped} already existed. Total: {total}")

    # Write a small metadata file
    meta_path = ds_dir / "metadata.txt"
    meta_path.write_text(
        f"source: {hf_id}\n"
        f"gt_field: {gt_field}\n"
        f"total_samples: {total}\n"
        f"description: {info['desc']}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download KITAB-Bench image-to-text datasets from HuggingFace"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset names to download (default: all). Use --list to see options.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        sys.exit(0)

    targets = args.datasets if args.datasets else list(DATASETS.keys())

    # Validate names
    for name in targets:
        if name not in DATASETS:
            print(f"ERROR: Unknown dataset '{name}'. Use --list to see options.")
            sys.exit(1)

    print(f"Output directory: {OUTPUT_ROOT.resolve()}")
    print(f"Datasets to download: {', '.join(targets)}")

    for name in targets:
        download_dataset(name, DATASETS[name])

    print("\nAll done!")
    print(f"Data saved to: {OUTPUT_ROOT.resolve()}")
    print("\nDirectory structure:")
    print("  data/kitab-bench/{dataset}/images/{idx}.png")
    print("  data/kitab-bench/{dataset}/gt/{idx}.txt")


if __name__ == "__main__":
    main()
