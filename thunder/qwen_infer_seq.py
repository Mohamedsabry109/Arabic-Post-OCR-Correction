#!/usr/bin/env python3
"""Sequential (one-sample-at-a-time) Qwen3 OCR correction — reference implementation.

Processes one record at a time using HuggingFace transformers, exactly like
scripts/infer.py does.  No batching, no vLLM.  Use this script to produce a
corrections.jsonl you can compare directly against thunder/qwen_infer.py
(vLLM / batched paths) to verify numerical equivalence.

I/O format is 100% compatible with the existing pipeline:
    Input:   results/<phase>/inference_input.jsonl  (from run_phase*.py --mode export)
    Output:  results/<phase>/corrections_seq.jsonl  (read by run_phase*.py --mode analyze)

Usage:
    python thunder/qwen_infer_seq.py \\
        --input  results/phase2/inference_input.jsonl \\
        --output results/phase2/corrections_seq.jsonl

    # Smoke-test with 20 samples
    python thunder/qwen_infer_seq.py \\
        --input  results/phase2/inference_input.jsonl \\
        --output results/phase2/corrections_seq.jsonl \\
        --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm

# Project root on sys.path so src/ imports work when running from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import shared helpers from the optimised script to avoid drift
_THUNDER_DIR = Path(__file__).resolve().parent
if str(_THUNDER_DIR) not in sys.path:
    sys.path.insert(0, str(_THUNDER_DIR))

from src.core.prompt_builder import PromptBuilder
from qwen_infer import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_INPUT_TOKENS,
    _ARABIC_RE,
    read_input,
    _read_completed_ids,
    _make_out_record,
    _extract_text,
    build_messages,
    _build_token_ids,
)

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
    p = argparse.ArgumentParser(
        description="Sequential Qwen3 OCR correction (transformers, one sample at a time).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path,
        default=Path("results/phase2/inference_input.jsonl"),
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("results/phase2/corrections_seq.jsonl"),
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("configs/config.yaml"),
    )
    p.add_argument("--model",        type=str,   default=DEFAULT_MODEL)
    p.add_argument("--max-tokens",   type=int,   default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument(
        "--max-retries", type=int, default=2,
        help="Retry count on empty or failed generation (default: 2, matches scripts/infer.py).",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None, metavar="DATASET",
        help="Process only these dataset keys (default: all).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per dataset (for smoke-testing).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Reprocess all, ignoring existing output.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sequential inference loop
# ---------------------------------------------------------------------------


def run_sequential(
    pending: list[dict],
    builder: PromptBuilder,
    config: dict,
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict]:
    """Process one record at a time — mirrors scripts/infer.py exactly.

    Retry logic matches TransformersCorrector.correct(): on an empty generation
    (raw output strips to nothing) or any generation exception, the attempt is
    retried up to max_retries additional times before falling back to ocr_text.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name   = config.get("model", {}).get("name", args.model)
    temperature  = float(config.get("model", {}).get("temperature", args.temperature))
    max_tokens   = int(config.get("model", {}).get("max_tokens", args.max_tokens))
    max_retries  = args.max_retries

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logger.info("Loading model: %s (FP16, device_map=auto)", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    out_records: list[dict] = []
    n_success_run = n_failed_run = 0
    t_run_start = time.monotonic()

    pbar = tqdm(
        total=len(pending),
        desc="Qwen3 (sequential)",
        unit="sample",
        dynamic_ncols=True,
    )

    with open(output_path, "a", encoding="utf-8") as out_f:
        for record in pending:
            t0 = time.monotonic()
            sample_id = record.get("sample_id", "?")

            messages, pt, ver = build_messages(record, builder)
            ids = _build_token_ids(tokenizer, messages, record, builder, MAX_INPUT_TOKENS)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            prompt_len = len(ids)

            corrected  = record["ocr_text"]   # safe fallback
            out_tokens = 0
            success    = False
            last_error: Optional[str] = None

            # Retry loop — matches TransformersCorrector.correct() exactly:
            # retry on empty generation or any exception, up to max_retries times.
            for attempt in range(max_retries + 1):
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids,
                            max_new_tokens=max_tokens,
                            do_sample=(temperature > 0),
                            temperature=temperature if temperature > 0 else 1.0,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    new_ids  = output[:, prompt_len:]
                    raw_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
                    result   = _extract_text(raw_text, record["ocr_text"])

                    if result == record["ocr_text"] and not raw_text.strip():
                        # Empty generation — retry (matches infer.py retry condition)
                        last_error = f"Empty generation on attempt {attempt + 1}"
                        logger.warning("[%s] %s — retrying...", sample_id, last_error)
                        continue

                    corrected  = result
                    out_tokens = new_ids.shape[1]
                    success    = True
                    last_error = None
                    break

                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "[%s] Generation failed (attempt %d/%d): %s",
                        sample_id, attempt + 1, max_retries + 1, exc,
                    )

            if not success:
                logger.error(
                    "[%s] All %d attempts failed. Falling back to OCR text. Last error: %s",
                    sample_id, max_retries + 1, last_error,
                )

            latency = round(time.monotonic() - t0, 3)

            rec = _make_out_record(
                record=record,
                corrected_text=corrected,
                model_name=model_name,
                prompt_type=pt,
                prompt_ver=ver,
                prompt_tokens=prompt_len,
                output_tokens=out_tokens,
                latency_s=latency,
                success=success,
                error=last_error,
            )
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()  # per-record flush — matches scripts/infer.py behaviour
            out_records.append(rec)

            if success:
                n_success_run += 1
            else:
                n_failed_run += 1

            torch.cuda.empty_cache()
            pbar.update(1)
            elapsed = time.monotonic() - t_run_start
            tok_per_s = out_tokens / latency if latency > 0 else 0
            pbar.set_postfix({
                "ok":      n_success_run,
                "fail":    n_failed_run,
                "lat":     f"{latency:.1f}s",
                "tok/s":   f"{tok_per_s:.0f}",
                "samp/s":  f"{len(out_records) / elapsed:.2f}" if elapsed > 0 else "?",
            })

    pbar.close()
    return out_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load config
    config: dict = {}
    if args.config.exists():
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config not found at %s — using CLI defaults.", args.config)

    config.setdefault("model", {})["name"] = args.model

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sequential inference started: %s", datetime.now(timezone.utc).isoformat())
    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output)
    logger.info("Model  : %s", args.model)
    logger.info("=" * 60)

    # Read input and determine pending (resume by skipping already-written IDs)
    datasets_filter = set(args.datasets) if args.datasets else None
    all_records = read_input(args.input, datasets_filter, args.limit)
    logger.info("Input records loaded: %d", len(all_records))

    if not args.force:
        completed_ids = _read_completed_ids(args.output)
        pending = [r for r in all_records if r["sample_id"] not in completed_ids]
        if completed_ids:
            logger.info(
                "Resuming: %d already done, %d remaining.",
                len(completed_ids), len(pending),
            )
    else:
        pending = all_records

    if not pending:
        logger.info("Nothing to do.")
        return

    # Build PromptBuilder — uses crafted_prompt_path like scripts/infer.py
    crafted_prompt_path = config.get("prompt_craft", {}).get("crafted_prompt_path")
    builder = PromptBuilder(crafted_prompt_path=crafted_prompt_path)

    # ------------------------------------------------------------------
    # Print prompt preview for verification before inference starts
    # ------------------------------------------------------------------
    _preview_msgs, _preview_pt, _preview_ver = build_messages(pending[0], builder)
    sep = "=" * 70
    print(sep)
    print(
        f"  Qwen3 OCR Correction (sequential) — {datetime.now(timezone.utc).isoformat()}"
    )
    print(sep)
    print(f"  Model       : {args.model}")
    print(f"  Temperature : {config.get('model', {}).get('temperature', args.temperature)}")
    print(f"  Max tokens  : {config.get('model', {}).get('max_tokens', args.max_tokens)}")
    print(f"  Max retries : {args.max_retries}")
    print(f"  Pending     : {len(pending)}")
    print(sep)
    print(
        f"  PROMPT PREVIEW  "
        f"(sample_id={pending[0]['sample_id']}  "
        f"prompt_type={pending[0].get('prompt_type', 'zero_shot')}  "
        f"version={_preview_ver})"
    )
    print(sep)
    for _msg in _preview_msgs:
        print(f"  [{_msg['role'].upper()}]")
        # Truncate long context sections so the preview stays readable
        content = _msg["content"]
        if len(content) > 800:
            content = content[:800] + f"\n  ... [{len(content) - 800} chars truncated]"
        print(content)
        print()
    print(sep, flush=True)
    print()

    logger.info("Processing %d records...", len(pending))
    t_start = time.monotonic()
    out_records = run_sequential(pending, builder, config, args, args.output)
    elapsed = time.monotonic() - t_start

    n_success = sum(1 for r in out_records if r.get("success"))
    n_failed  = len(out_records) - n_success
    logger.info("=" * 60)
    logger.info(
        "Done: %d success | %d failed | %.1fs | %.2f samples/s",
        n_success, n_failed, elapsed,
        len(out_records) / elapsed if elapsed > 0 else 0,
    )
    logger.info("Output: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
