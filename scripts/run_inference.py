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

HF SYNC: Pass --hf-repo to sync corrections.jsonl to a HuggingFace dataset.
On startup the existing file is pulled from HF (enabling cross-session resume).
During inference the file is pushed every --sync-every samples and at the end.
The HF token is read from --hf-token or the HF_TOKEN environment variable.

Usage (Kaggle/Colab cell):
    !python run_inference.py
    !python run_inference.py --input /kaggle/input/my-data/inference_input.jsonl \\
                             --output /kaggle/working/corrections.jsonl \\
                             --model Qwen/Qwen3-4B-Instruct-2507 \\
                             --limit 100 \\
                             --quantize-4bit \\
                             --hf-repo username/my-corrections \\
                             --hf-token hf_xxx \\
                             --sync-every 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Flexible import: works in two layouts:
#   A) Project structure — script lives in scripts/, src/core/ is one level up
#   B) Flat Kaggle/Colab upload — all .py files in the same directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

for _p in [str(_PROJECT_ROOT), str(_SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.core.prompt_builder import PromptBuilder
    from src.core.llm_corrector import TransformersCorrector, CorrectionResult
except ModuleNotFoundError:
    # Flat layout: prompt_builder.py and llm_corrector.py sit next to this script
    from prompt_builder import PromptBuilder  # type: ignore
    from llm_corrector import TransformersCorrector, CorrectionResult  # type: ignore

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
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID to sync corrections.jsonl (e.g. username/my-corrections).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=100,
        help="Push to HF every N completed samples (default: 100).",
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
# HuggingFace sync helpers
# ---------------------------------------------------------------------------

_HF_FILENAME = "corrections.jsonl"


def _hf_token(args_token: str | None) -> str | None:
    """Return HF token from CLI arg or HF_TOKEN env var."""
    return args_token or os.environ.get("HF_TOKEN")


def hf_pull(output_path: Path, repo: str, token: str | None) -> None:
    """Download existing corrections.jsonl from HF dataset into output_path.

    Merges with any locally-existing records so neither source loses progress.
    Silently skips if the file doesn't exist in the repo yet.
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        logger.warning("huggingface_hub not installed — HF sync disabled.")
        return

    try:
        remote_path = hf_hub_download(
            repo_id=repo,
            filename=_HF_FILENAME,
            repo_type="dataset",
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        logger.info("HF repo has no existing %s — starting fresh.", _HF_FILENAME)
        return
    except Exception as exc:
        logger.warning("HF pull failed (will start from local file): %s", exc)
        return

    # Merge remote records with any local records (union by sample_id).
    remote_ids: set[str] = set()
    remote_lines: list[str] = []
    with open(remote_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sid = r.get("sample_id", "")
                if sid and sid not in remote_ids:
                    remote_ids.add(sid)
                    remote_lines.append(line)
            except json.JSONDecodeError:
                pass

    local_ids: set[str] = set()
    local_lines: list[str] = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    sid = r.get("sample_id", "")
                    if sid and sid not in local_ids:
                        local_ids.add(sid)
                        local_lines.append(line)
                except json.JSONDecodeError:
                    pass

    # Write merged result: remote first, then any local-only records.
    merged = remote_lines + [l for l in local_lines if json.loads(l).get("sample_id") not in remote_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in merged:
            f.write(line + "\n")

    logger.info(
        "HF pull complete: %d remote + %d local-only = %d total records.",
        len(remote_ids), len(local_ids) - len(local_ids & remote_ids), len(merged),
    )


def hf_push(output_path: Path, repo: str, token: str | None) -> None:
    """Upload corrections.jsonl to HF dataset repo.

    Creates the repo if it doesn't exist. Silently logs on failure so
    inference is never interrupted by a network hiccup.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return

    try:
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, token=token)
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=_HF_FILENAME,
            repo_id=repo,
            repo_type="dataset",
            token=token,
            commit_message=f"sync: {sum(1 for _ in open(output_path, encoding='utf-8') if _.strip())} records",
        )
        logger.info("HF push OK -> %s/%s", repo, _HF_FILENAME)
    except Exception as exc:
        logger.warning("HF push failed (progress saved locally): %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    hf_repo  = args.hf_repo
    hf_token = _hf_token(args.hf_token)

    # ------------------------------------------------------------------
    # HF pull — restore progress from previous session before anything else
    # ------------------------------------------------------------------
    if hf_repo:
        logger.info("HF sync enabled: %s (pulling existing progress...)", hf_repo)
        hf_pull(args.output, hf_repo, hf_token)

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
    n_since_sync = 0

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

            # Periodic HF sync
            n_since_sync += 1
            if hf_repo and n_since_sync >= args.sync_every:
                out_f.flush()
                hf_push(args.output, hf_repo, hf_token)
                n_since_sync = 0

    # ------------------------------------------------------------------
    # Final HF push
    # ------------------------------------------------------------------
    if hf_repo:
        hf_push(args.output, hf_repo, hf_token)

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
