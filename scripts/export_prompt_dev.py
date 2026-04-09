#!/usr/bin/env python3
"""Export a small inference JSONL for fast prompt iteration using dev samples.

Takes the dev sample list (from select_prompt_dev_samples.py) and the
full inference_input.jsonl, filters to only the selected samples.

Prompt selection is handled at inference time by infer.py's
``--crafted-prompt`` flag — this script does not touch prompts.

Usage:
    # Export dev subset
    python scripts/export_prompt_dev.py

    # Custom sample list or input
    python scripts/export_prompt_dev.py \\
        --samples data/prompt_dev_samples.json \\
        --input results/phase2/inference_input.jsonl

    # Then on Kaggle (default prompt):
    python scripts/infer.py \\
        --input results/prompt_dev/inference_input.jsonl \\
        --output results/prompt_dev/corrections.jsonl --force

    # Or with a different prompt:
    python scripts/infer.py \\
        --input results/prompt_dev/inference_input.jsonl \\
        --output results/prompt_dev/corrections.jsonl --force \\
        --system-prompt configs/crafted_system_prompt_v2.txt
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_DEFAULT_SAMPLES = _PROJECT_ROOT / "data" / "prompt_dev_samples.json"
_DEFAULT_INPUT = _PROJECT_ROOT / "results" / "phase2" / "inference_input.jsonl"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "prompt_dev"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export filtered inference JSONL for prompt dev iteration."
    )
    p.add_argument(
        "--samples", type=Path, default=_DEFAULT_SAMPLES,
        help="Path to dev samples JSON (from select_prompt_dev_samples.py).",
    )
    p.add_argument(
        "--input", type=Path, default=_DEFAULT_INPUT,
        help="Full inference_input.jsonl to filter from.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, dest="output_dir",
        help="Output directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load sample IDs
    if not args.samples.exists():
        print(f"ERROR: {args.samples} not found. Run select_prompt_dev_samples.py first.")
        sys.exit(1)

    with open(args.samples, encoding="utf-8") as f:
        dev_data = json.load(f)
    sample_ids = set(dev_data["sample_ids"])
    print(f"Dev samples: {len(sample_ids)}")

    # Filter inference input
    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "inference_input.jsonl"

    written = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("sample_id") not in sample_ids:
                continue
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {output_path}")

    if written < len(sample_ids):
        print(f"  WARNING: {len(sample_ids) - written} sample_ids not found in input")

    # Save metadata for reference
    meta_path = args.output_dir / "dev_samples_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_samples": str(args.samples),
            "source_input": str(args.input),
            "total_exported": written,
            "categories": dev_data["meta"]["by_category"],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nNext steps:")
    print(f"  # On Kaggle (default prompt):")
    print(f"  python scripts/infer.py \\")
    print(f"    --input {output_path} \\")
    print(f"    --output {args.output_dir}/corrections.jsonl --force")
    print(f"")
    print(f"  # Or with a specific prompt file:")
    print(f"  python scripts/infer.py \\")
    print(f"    --input {output_path} \\")
    print(f"    --output {args.output_dir}/corrections.jsonl --force \\")
    print(f"    --system-prompt configs/crafted_system_prompt_v2.txt")
    print(f"")
    print(f"  # Then locally:")
    print(f"  python scripts/analyze_training.py \\")
    print(f"    --input {args.output_dir}/corrections.jsonl \\")
    print(f"    --output-dir {args.output_dir}/analysis")


if __name__ == "__main__":
    main()
