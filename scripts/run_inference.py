#!/usr/bin/env python3
"""Standalone inference script for Kaggle / Google Colab.

Reads inference_input.jsonl, corrects each OCR text using a local
HuggingFace model (Qwen3-4B-Instruct-2507), and writes corrections.jsonl.

This script is designed to be uploaded to and run on Kaggle/Colab kernels.
It only depends on:
  - src/core/prompt_builder.py   (no external deps)
  - src/core/llm_corrector.py    (needs transformers + torch)
These two files are uploaded to the remote environment alongside this script.

RESUME SUPPORT: If the output file already exists, completed sample IDs are
read on startup and skipped. Restart the kernel after a timeout — progress
is preserved.

Usage (Kaggle/Colab cell):
    !python run_inference.py
    !python run_inference.py --input /kaggle/input/my-data/inference_input.jsonl \\
                             --output /kaggle/working/corrections.jsonl \\
                             --model Qwen/Qwen3-4B-Instruct-2507 \\
                             --limit 100 \\
                             --quantize-4bit
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project src/ is importable when run from any working directory.
# On Kaggle: upload the project repo as a dataset or add src/ to the notebook.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import TransformersCorrector, CorrectionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Kaggle/Colab inference script for Arabic OCR correction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("inference_input.jsonl"),
        help="Path to inference_input.jsonl (produced by run_phase2.py --mode export).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corrections.jsonl"),
        help="Path to write corrections.jsonl. Appended to if it already exists (resume).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N samples (for quick smoke tests).",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        default=False,
        help="Enable 4-bit quantization via bitsandbytes (for GPUs with <8GB VRAM).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count on empty or failed LLM generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0.1 for near-deterministic output).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate per sample.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_completed_ids(output_path: Path) -> set[str]:
    """Read already-completed sample IDs from an existing output file.

    Used on resume to skip samples that were already corrected.

    Args:
        output_path: Path to existing corrections.jsonl (may not exist).

    Returns:
        Set of sample_id strings that are already written to the file.
    """
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sid = record.get("sample_id")
                if sid:
                    completed.add(sid)
            except json.JSONDecodeError:
                pass  # corrupt line — skip silently

    logger.info("Resume: found %d already-completed samples in %s", len(completed), output_path)
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if not args.input.exists():
        logger.error(
            "Input file not found: %s\n"
            "Run 'python pipelines/run_phase2.py --mode export' locally first, "
            "then upload the resulting inference_input.jsonl to Kaggle/Colab.",
            args.input,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load input records
    # ------------------------------------------------------------------
    logger.info("Reading input: %s", args.input)
    records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    logger.info("Total samples to process: %d", len(records))

    # ------------------------------------------------------------------
    # Resume: skip already-completed samples
    # ------------------------------------------------------------------
    completed_ids = load_completed_ids(args.output)
    pending = [r for r in records if r["sample_id"] not in completed_ids]
    logger.info(
        "Pending: %d  |  Already done: %d",
        len(pending), len(completed_ids),
    )

    if not pending:
        logger.info("All samples already completed. Nothing to do.")
        _print_summary(args.output)
        return

    # ------------------------------------------------------------------
    # Load model (done once, before the loop)
    # ------------------------------------------------------------------
    minimal_config: dict = {
        "model": {
            "name": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "device": "auto",
            "quantize_4bit": args.quantize_4bit,
        }
    }

    logger.info("Loading model: %s", args.model)
    corrector = TransformersCorrector(minimal_config)
    builder = PromptBuilder()
    logger.info("Model ready. Starting inference...")

    # ------------------------------------------------------------------
    # Inference loop — write each result immediately for resume support
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_failed = 0
    total_latency = 0.0

    try:
        from tqdm import tqdm
        iterator = tqdm(pending, desc="Correcting", unit="sample")
    except ImportError:
        iterator = pending  # tqdm not available — plain loop

    with open(args.output, "a", encoding="utf-8") as out_f:
        for record in iterator:
            sample_id = record["sample_id"]
            ocr_text = record["ocr_text"]
            gt_text = record.get("gt_text", "")

            messages = builder.build_zero_shot(ocr_text)

            result: CorrectionResult = corrector.correct(
                sample_id=sample_id,
                ocr_text=ocr_text,
                messages=messages,
                max_retries=args.max_retries,
            )

            # Build the output record (the contract file schema)
            output_record: dict = {
                "sample_id":      sample_id,
                "dataset":        record.get("dataset", ""),
                "ocr_text":       ocr_text,
                "corrected_text": result.corrected_text,
                "gt_text":        gt_text,
                "model":          corrector.model_name,
                "prompt_version": builder.prompt_version,
                "prompt_tokens":  result.prompt_tokens,
                "output_tokens":  result.output_tokens,
                "latency_s":      result.latency_s,
                "success":        result.success,
                "error":          result.error,
            }

            # Write immediately — preserves progress on session timeout
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out_f.flush()

            total_latency += result.latency_s
            if result.success:
                n_success += 1
            else:
                n_failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_processed = n_success + n_failed
    avg_latency = total_latency / max(total_processed, 1)

    logger.info("=" * 60)
    logger.info("Inference complete.")
    logger.info("  Successful : %d", n_success)
    logger.info("  Failed     : %d", n_failed)
    logger.info("  Avg latency: %.2f s/sample", avg_latency)
    logger.info("  Output     : %s", args.output)
    logger.info("=" * 60)
    logger.info(
        "Download %s and place it at:\n"
        "  results/phase2/{dataset_key}/corrections.jsonl\n"
        "Then run locally:\n"
        "  python pipelines/run_phase2.py --mode analyze",
        args.output.name,
    )

    _print_summary(args.output)


def _print_summary(output_path: Path) -> None:
    """Print a quick count of records in the output file."""
    if not output_path.exists():
        return
    total = sum(1 for line in open(output_path, encoding="utf-8") if line.strip())
    success = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("success"):
                        success += 1
                except json.JSONDecodeError:
                    pass
    print(f"\nOutput file: {output_path}")
    print(f"  Total records : {total}")
    print(f"  Successful    : {success}")
    print(f"  Failed        : {total - success}")


if __name__ == "__main__":
    main()
