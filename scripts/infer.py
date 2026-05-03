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

    # Phase 3 with HF sync (auto-derives HF path from --output)
    python scripts/infer.py \\
        --input  results/phase3/inference_input.jsonl \\
        --output results/phase3/corrections.jsonl \\
        --hf-repo user/arabic-ocr-corrections \\
        --hf-token hf_xxx --sync-every 100
    # -> syncs to HF as: phase3/corrections.jsonl

    # Phase 5 combo (each combo gets its own HF path)
    python scripts/infer.py \\
        --input  results/phase5/full_prompt/inference_input.jsonl \\
        --output results/phase5/full_prompt/corrections.jsonl \\
        --hf-repo user/arabic-ocr-corrections \\
        --hf-token hf_xxx
    # -> syncs to HF as: phase5/full_prompt/corrections.jsonl

    # Kaggle with explicit HF path override
    python scripts/infer.py \\
        --input  /kaggle/working/inference_input.jsonl \\
        --output /kaggle/working/corrections.jsonl \\
        --hf-repo user/arabic-ocr-corrections \\
        --hf-path phase2/corrections.jsonl \\
        --hf-token hf_xxx
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
        "--backend", type=str, default=None,
        choices=["mock", "transformers", "api"],
        help=(
            "Override model backend from config. "
            "Use 'mock' for local smoke tests (no GPU, no model download). "
            "Use 'transformers' for Kaggle/Colab GPU inference."
        ),
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
        "--max-retries", type=int, default=1,
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
    parser.add_argument(
        "--hf-path", type=str, default=None,
        help=(
            "Override path_in_repo for HF sync. "
            "Default: auto-derived from --output relative to results/ "
            "(e.g. results/phase3/corrections.jsonl -> phase3/corrections.jsonl)."
        ),
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None, dest="system_prompt",
        help=(
            "Path to the system prompt file to use as the base prompt. "
            "Overrides the default (configs/crafted_system_prompt.txt). "
            "Example: --system-prompt configs/crafted_system_prompt_v2.txt"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HF sync helpers
# ---------------------------------------------------------------------------


def _resolve_hf_path(output_path: Path, hf_path: Optional[str] = None) -> str:
    """Derive the path_in_repo for HF from the local output path.

    If --hf-path is explicitly provided, use that. Otherwise, derive it
    from the output path relative to a 'results/' parent directory.

    Examples:
        results/phase2/corrections.jsonl       -> phase2/corrections.jsonl
        results/phase3/corrections.jsonl       -> phase3/corrections.jsonl
        results/phase5/full_prompt/corrections.jsonl -> phase5/full_prompt/corrections.jsonl
        results/phase7/corrections.jsonl       -> phase7/corrections.jsonl

    This ensures each phase (and each Phase 5 combo) gets its own file
    in the HF repo, preventing cross-phase overwrites.
    """
    if hf_path:
        return hf_path

    parts = output_path.resolve().parts
    # Find the 'results' directory in the path and take everything after it
    for i, part in enumerate(parts):
        if part == "results" and i + 1 < len(parts):
            return "/".join(parts[i + 1:])

    # Fallback: use the filename only
    return output_path.name


def hf_pull(
    hf_repo: str, token: str, output_path: Path,
    hf_path: Optional[str] = None,
) -> set[str]:
    """Pull existing corrections from HF and merge into output_path.

    Remote records win on sample_id conflicts.
    Returns the set of sample_ids now in the file.
    """
    path_in_repo = _resolve_hf_path(output_path, hf_path)
    try:
        from huggingface_hub import hf_hub_download
        logger.info("HF pull: %s/%s ...", hf_repo, path_in_repo)
        tmp = hf_hub_download(
            repo_id=hf_repo,
            filename=path_in_repo,
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

    logger.info("HF pull complete: %d records (%s)", len(merged), path_in_repo)
    return set(merged.keys())


def hf_push(
    hf_repo: str, token: str, output_path: Path,
    hf_path: Optional[str] = None,
) -> None:
    """Push corrections JSONL to HF dataset repo at the correct path."""
    if not output_path.exists():
        return
    path_in_repo = _resolve_hf_path(output_path, hf_path)
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=path_in_repo,
            repo_id=hf_repo,
            repo_type="dataset",
            token=token,
        )
        logger.info("HF push: %s -> %s/%s", output_path.name, hf_repo, path_in_repo)
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

            # Skip samples with empty OCR text (e.g. Qaari produced no output).
            if not r.get("ocr_text", "").strip():
                logger.warning("Skipping %s with empty ocr_text.", r.get("sample_id", "?"))
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

    # Apply --model / --backend overrides
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.backend:
        config.setdefault("model", {})["backend"] = args.backend

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
    if use_hf:
        hf_remote_path = _resolve_hf_path(args.output, args.hf_path)
        logger.info("HF sync: %s -> %s/%s (every %d samples)",
                     args.output.name, args.hf_repo, hf_remote_path, args.sync_every)

    # Determine completed sample_ids
    if args.force:
        completed: set[str] = set()
        if args.output.exists():
            args.output.write_text("", encoding="utf-8")  # truncate
    elif use_hf:
        completed = hf_pull(args.hf_repo, token, args.output, args.hf_path)  # type: ignore[arg-type]
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
    crafted_prompt_path = args.system_prompt or config.get("prompt_craft", {}).get("crafted_prompt_path")
    builder = PromptBuilder(crafted_prompt_path=crafted_prompt_path)
    logger.info("Model ready: %s", corrector.model_name)

    # ------------------------------------------------------------------
    # Print the prompt that will be used for the first pending sample so
    # the user can verify the prompt before the full run begins.
    # ------------------------------------------------------------------
    def _build_messages_for_record(rec: dict) -> list[dict]:
        """Build messages for a single record (mirrors the dispatch below)."""
        pt = rec.get("prompt_type", "zero_shot")
        few_shot = rec.get("few_shot_examples") or ""
        if pt == "ocr_aware":
            return builder.build_ocr_aware(
                rec["ocr_text"],
                rec.get("confusion_context", ""),
                rec.get("word_examples", ""),
                few_shot_examples=few_shot,
            )
        if pt == "combined":
            return builder.build_combined(
                ocr_text=rec["ocr_text"],
                confusion_context=rec.get("confusion_context") or "",
                insights_context=rec.get("insights_context") or "",
                word_pairs_context=rec.get("word_pairs_context") or "",
                overcorrection_context=rec.get("overcorrection_context") or "",
                few_shot_examples=few_shot,
                word_examples=rec.get("word_examples") or "",
            )
        if pt == "self_reflective":
            return builder.build_self_reflective(
                rec["ocr_text"],
                rec.get("insights_context", ""),
                rec.get("word_pairs_context", ""),
                rec.get("overcorrection_context", ""),
                few_shot_examples=few_shot,
            )
        if pt == "rag":
            return builder.build_rag(
                rec["ocr_text"],
                retrieved_sentences=rec.get("retrieved_sentences", ""),
                retrieved_words=rec.get("retrieved_words", ""),
                few_shot_examples=few_shot,
            )
        if pt == "meta_prompt":
            return builder.build_meta_prompt(rec["ocr_text"])
        if pt == "crafted":
            return builder.build_crafted(rec["ocr_text"], rec.get("system_prompt", ""))
        # zero_shot and fallback
        return builder.build_zero_shot(
            rec["ocr_text"],
            version=rec.get("prompt_version", "crafted"),
            few_shot_examples=few_shot,
        )

    _preview = pending[0]
    _preview_messages = _build_messages_for_record(_preview)
    _sep = "=" * 70
    print(_sep, flush=True)
    print(
        f"PROMPT PREVIEW  (sample_id={_preview['sample_id']}  "
        f"prompt_type={_preview.get('prompt_type', 'zero_shot')})",
        flush=True,
    )
    print(_sep, flush=True)
    for _msg in _preview_messages:
        print(f"[{_msg['role'].upper()}]", flush=True)
        print(_msg["content"], flush=True)
        print(flush=True)
    print(_sep, flush=True)
    print(flush=True)

    # Inference loop
    n_success = n_failed = 0
    pushed_at = 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, record in enumerate(tqdm(pending, desc="Inference", unit="sample")):
            # Dispatch to the correct prompt builder based on prompt_type.
            # Phase 2 records have no prompt_type field — default to "zero_shot".
            prompt_type = record.get("prompt_type", "zero_shot")

            few_shot = record.get("few_shot_examples") or ""

            if prompt_type == "ocr_aware":
                confusion_context = record.get("confusion_context", "")
                word_examples = record.get("word_examples", "")
                messages = builder.build_ocr_aware(
                    record["ocr_text"], confusion_context, word_examples,
                    few_shot_examples=few_shot,
                )
                prompt_ver = builder.ocr_aware_prompt_version
                if not confusion_context.strip():
                    # build_ocr_aware fell back to zero_shot internally
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "combined":
                messages = builder.build_combined(
                    ocr_text=record["ocr_text"],
                    confusion_context=record.get("confusion_context") or "",
                    insights_context=record.get("insights_context") or "",
                    word_pairs_context=record.get("word_pairs_context") or "",
                    overcorrection_context=record.get("overcorrection_context") or "",
                    few_shot_examples=few_shot,
                    word_examples=record.get("word_examples") or "",
                )
                prompt_ver = builder.combined_prompt_version
                _any_ctx = any([
                    record.get("confusion_context"),
                    record.get("insights_context"),
                    record.get("word_pairs_context"),
                    record.get("overcorrection_context"),
                ])
                if not _any_ctx:
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "self_reflective":
                insights_context = record.get("insights_context", "")
                word_pairs_context = record.get("word_pairs_context", "")
                overcorrection_context = record.get("overcorrection_context", "")
                messages = builder.build_self_reflective(
                    record["ocr_text"], insights_context, word_pairs_context,
                    overcorrection_context, few_shot_examples=few_shot,
                )
                prompt_ver = builder.self_reflective_prompt_version
                if not insights_context.strip() and not word_pairs_context.strip() and not overcorrection_context.strip():
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "rag":
                retrieved_sentences = record.get("retrieved_sentences", "")
                retrieved_words = record.get("retrieved_words", "")
                messages = builder.build_rag(
                    record["ocr_text"],
                    retrieved_sentences=retrieved_sentences,
                    retrieved_words=retrieved_words,
                    few_shot_examples=few_shot,
                )
                prompt_ver = record.get("prompt_version", "p8v1")
                if not retrieved_sentences.strip() and not retrieved_words.strip():
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "meta_prompt":
                messages = builder.build_meta_prompt(record["ocr_text"])
                prompt_ver = record.get("prompt_version", builder.meta_prompt_version)
            elif prompt_type == "crafted":
                system_prompt = record.get("system_prompt", "")
                messages = builder.build_crafted(record["ocr_text"], system_prompt)
                prompt_ver = record.get("prompt_version", builder.crafted_prompt_version)
                if not system_prompt.strip():
                    prompt_type = "zero_shot_fallback"
            elif prompt_type == "zero_shot":
                pv = record.get("prompt_version", "v1")
                messages = builder.build_zero_shot(
                    record["ocr_text"], version=pv, few_shot_examples=few_shot,
                )
                prompt_ver = pv
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
                hf_push(args.hf_repo, token, args.output, args.hf_path)  # type: ignore[arg-type]
                pushed_at = i + 1

    # Final HF push (ensure last batch is synced)
    if use_hf:
        hf_push(args.hf_repo, token, args.output, args.hf_path)  # type: ignore[arg-type]

    logger.info("=" * 60)
    logger.info("Done: %d success, %d failed.", n_success, n_failed)
    logger.info("Output: %s", args.output)
    logger.info("")
    logger.info("Next: python pipelines/run_phase2.py --mode analyze")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
