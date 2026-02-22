#!/usr/bin/env python3
"""Unified inference script for Arabic OCR post-correction.

Works on local machine (GPU or API), Kaggle, and Google Colab.
Reads from inference_input.jsonl (produced by run_phase2.py --mode export).
Writes results to a single corrections.jsonl.

Workflow
--------
    1. LOCAL:  python pipelines/run_phase2.py --mode export
    2. ANY:    python scripts/infer.py [options]
    3. LOCAL:  python pipelines/run_phase2.py --mode analyze

Usage
-----
    # Local (default paths)
    python scripts/infer.py

    # Local — subset / smoke test
    python scripts/infer.py --datasets KHATT-train --limit 50

    # Kaggle (with HF sync for resume across sessions)
    python scripts/infer.py \\
        --input  /kaggle/input/your-dataset/inference_input.jsonl \\
        --output /kaggle/working/corrections.jsonl \\
        --model  Qwen/Qwen3-4B-Instruct-2507 \\
        --hf-repo  user/arabic-ocr-corrections \\
        --hf-token hf_xxx --sync-every 100

    # Colab (output to Drive)
    python scripts/infer.py \\
        --input  /content/drive/MyDrive/arabic-ocr/inference_input.jsonl \\
        --output /content/drive/MyDrive/arabic-ocr/corrections.jsonl \\
        --model  Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.llm_corrector import get_corrector
from src.core.prompt_builder import PromptBuilder

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
        description="Unified LLM inference for Arabic OCR correction (local/Kaggle/Colab)."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/phase2/inference_input.jsonl"),
        help="Path to inference_input.jsonl (from run_phase2.py --mode export).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/phase2/corrections.jsonl"),
        help="Path to write corrections.jsonl.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config YAML (for model backend settings).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name from config.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, metavar="DATASET",
        help="Process only these dataset keys (default: all in input file).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per dataset (for testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all samples, ignoring existing output.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="LLM retry count on empty or failed generation.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo (user/name) for cross-session sync.",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--sync-every", type=int, default=100,
        help="Push to HF every N samples (default: 100).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HF sync helpers
# ---------------------------------------------------------------------------


def hf_pull(hf_repo: str, token: str, output_path: Path) -> set[str]:
    """Pull existing corrections from HF and merge into output_path.

    Remote records win on sample_id conflicts.
    Returns the set of sample_ids now in the file.
    """
    try:
        from huggingface_hub import hf_hub_download
        logger.info("HF pull: downloading from %s ...", hf_repo)
        tmp = hf_hub_download(
            repo_id=hf_repo,
            filename="corrections.jsonl",
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        logger.warning("HF pull skipped (%s) — starting fresh or from local file.", exc)
        return _read_completed_ids(output_path)

    remote: dict[str, dict] = {}
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                remote[r["sample_id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass

    local: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    local[r["sample_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass

    merged = {**local, **remote}  # remote wins on conflict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("HF pull complete: %d records in %s", len(merged), output_path)
    return set(merged.keys())


def hf_push(hf_repo: str, token: str, output_path: Path) -> None:
    """Push corrections.jsonl to HF dataset repo."""
    if not output_path.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo="corrections.jsonl",
            repo_id=hf_repo,
            repo_type="dataset",
            token=token,
        )
        logger.info("HF push: uploaded %s to %s", output_path, hf_repo)
    except Exception as exc:
        logger.warning("HF push failed: %s", exc)


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------


def _read_completed_ids(output_path: Path) -> set[str]:
    """Return sample_ids already written to output_path."""
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def read_input(
    input_path: Path,
    datasets_filter: Optional[set[str]],
    limit: Optional[int],
) -> list[dict]:
    """Read inference_input.jsonl, applying dataset filter and per-dataset limit."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run: python pipelines/run_phase2.py --mode export"
        )

    counts: dict[str, int] = {}
    records: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            ds = r.get("dataset", "")
            if datasets_filter and ds not in datasets_filter:
                continue

            if limit is not None:
                if counts.get(ds, 0) >= limit:
                    continue
                counts[ds] = counts.get(ds, 0) + 1

            records.append(r)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load config (optional — fall back to defaults if missing)
    if args.config.exists():
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found at %s — using defaults.", args.config)
        config = {}

    # Apply --model override
    if args.model:
        config.setdefault("model", {})["name"] = args.model

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Inference started: %s", datetime.now(timezone.utc).isoformat())
    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output)
    logger.info("Backend: %s", config.get("model", {}).get("backend", "transformers"))
    logger.info("Model  : %s", config.get("model", {}).get("name", "?"))
    logger.info("=" * 60)

    # HF sync setup
    token: Optional[str] = args.hf_token or os.environ.get("HF_TOKEN")
    use_hf = bool(args.hf_repo and token)
    if args.hf_repo and not token:
        logger.warning("--hf-repo provided but no HF token found. HF sync disabled.")

    # Determine completed sample_ids
    if args.force:
        completed: set[str] = set()
        if args.output.exists():
            args.output.write_text("", encoding="utf-8")  # truncate
    elif use_hf:
        completed = hf_pull(args.hf_repo, token, args.output)  # type: ignore[arg-type]
    else:
        completed = _read_completed_ids(args.output)
        if completed:
            logger.info("%d samples already done — resuming.", len(completed))

    # Read input
    datasets_filter = set(args.datasets) if args.datasets else None
    try:
        all_records = read_input(args.input, datasets_filter, args.limit)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    pending = [r for r in all_records if r["sample_id"] not in completed]
    logger.info(
        "Total: %d | Done: %d | Pending: %d",
        len(all_records), len(all_records) - len(pending), len(pending),
    )

    if not pending:
        logger.info("Nothing to do.")
        return

    # Load model once
    logger.info("Loading model...")
    corrector = get_corrector(config)
    builder = PromptBuilder()
    logger.info("Model ready: %s", corrector.model_name)

    # Inference loop
    n_success = n_failed = 0
    pushed_at = 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, record in enumerate(tqdm(pending, desc="Inference", unit="sample")):
            # Dispatch to the correct prompt builder based on prompt_type.
            # Phase 2 records have no prompt_type field — default to "zero_shot".
            prompt_type = record.get("prompt_type", "zero_shot")

            if prompt_type == "ocr_aware":
                confusion_context = record.get("confusion_context", "")
                messages = builder.build_ocr_aware(record["ocr_text"], confusion_context)
                prompt_ver = builder.ocr_aware_prompt_version
                if not confusion_context.strip():
                    # build_ocr_aware fell back to zero_shot internally
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "rule_augmented":
                rules_context = record.get("rules_context", "")
                messages = builder.build_rule_augmented(record["ocr_text"], rules_context)
                prompt_ver = builder.rules_prompt_version
                if not rules_context.strip():
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "few_shot":
                examples_context = record.get("examples_context", "")
                messages = builder.build_few_shot(record["ocr_text"], examples_context)
                prompt_ver = builder.few_shot_prompt_version
                if not examples_context.strip():
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "zero_shot":
                messages = builder.build_zero_shot(record["ocr_text"])
                prompt_ver = builder.prompt_version
            else:
                logger.warning(
                    "Unknown prompt_type '%s' for %s -- falling back to zero_shot.",
                    prompt_type, record["sample_id"],
                )
                messages = builder.build_zero_shot(record["ocr_text"])
                prompt_ver = builder.prompt_version
                prompt_type = "zero_shot_fallback"

            result = corrector.correct(
                sample_id=record["sample_id"],
                ocr_text=record["ocr_text"],
                messages=messages,
                max_retries=args.max_retries,
            )

            out_record = {
                "sample_id":      record["sample_id"],
                "dataset":        record.get("dataset", ""),
                "ocr_text":       record["ocr_text"],
                "corrected_text": result.corrected_text,
                "gt_text":        record.get("gt_text", ""),
                "model":          corrector.model_name,
                "prompt_type":    prompt_type,
                "prompt_version": prompt_ver,
                "prompt_tokens":  result.prompt_tokens,
                "output_tokens":  result.output_tokens,
                "latency_s":      result.latency_s,
                "success":        result.success,
                "error":          result.error,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()

            if result.success:
                n_success += 1
            else:
                n_failed += 1

            # Periodic HF push
            if use_hf and (i + 1 - pushed_at) >= args.sync_every:
                hf_push(args.hf_repo, token, args.output)  # type: ignore[arg-type]
                pushed_at = i + 1

    # Final HF push
    if use_hf:
        hf_push(args.hf_repo, token, args.output)  # type: ignore[arg-type]

    logger.info("=" * 60)
    logger.info("Done: %d success, %d failed.", n_success, n_failed)
    logger.info("Output: %s", args.output)
    logger.info("")
    logger.info("Next: python pipelines/run_phase2.py --mode analyze")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
