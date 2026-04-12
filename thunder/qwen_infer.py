#!/usr/bin/env python3
"""High-throughput Qwen3 OCR correction for Thunder Compute (A100 GPU).

Primary backend: vLLM offline inference
    - All prompts are submitted at once; vLLM's PagedAttention + continuous
      batching saturates the A100 at ~90%+ GPU utilisation.
    - Qwen3-4B weights ≈ 8 GB → 72 GB left for KV cache on A100 80 GB.
    - At max_model_len=32768, vLLM can concurrently process 100s of sequences.
    - Expected throughput: 5-15× faster than scripts/infer.py sample-by-sample.

Fallback backend: batched HuggingFace transformers
    - Used when vLLM is unavailable (--backend transformers).
    - Implements left-padded batched generate() for speedup over sequential.
    - Default batch_size=16; tune with --batch-size.

I/O is 100% compatible with the existing pipeline:
    Input:   results/<phase>/inference_input.jsonl  (from run_phase*.py --mode export)
    Output:  results/<phase>/corrections.jsonl      (read by run_phase*.py --mode analyze)

Usage:
    # Phase 2 — all records with vLLM (recommended)
    python thunder/qwen_infer.py \\
        --input  results/phase2/inference_input.jsonl \\
        --output results/phase2/corrections.jsonl

    # Phase 3 / 4 / 6 — same command, different input paths
    python thunder/qwen_infer.py \\
        --input  results/phase3/inference_input.jsonl \\
        --output results/phase3/corrections.jsonl

    # Transformers fallback (no vLLM)
    python thunder/qwen_infer.py \\
        --input  results/phase2/inference_input.jsonl \\
        --output results/phase2/corrections.jsonl \\
        --backend transformers --batch-size 16

    # Smoke-test with 20 samples
    python thunder/qwen_infer.py \\
        --input  results/phase2/inference_input.jsonl \\
        --output results/phase2/corrections.jsonl \\
        --limit 20

Performance notes (A100 80 GB, Qwen3-4B FP16):
    vLLM  : submits ALL pending prompts at once → continuous batching is fully
             utilised; throughput scales with prompt/output length mix.
    Batch : batch_size=16 with left-padding; diminishing returns beyond 32.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

# Project root on sys.path so src/ imports work when running from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.01
DEFAULT_GPU_MEM_UTIL = 0.92
# Qwen3-4B supports 32K context — set high to accommodate all phase prompts
DEFAULT_MAX_MODEL_LEN = 32768
# Max prompt length before truncation (matches TransformersCorrector.MAX_INPUT_TOKENS)
MAX_INPUT_TOKENS = 8192

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

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
        description="High-throughput Qwen3 OCR correction (vLLM) for Thunder Compute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path,
        default=Path("results/phase2/inference_input.jsonl"),
        help="Path to inference_input.jsonl (from run_phase*.py --mode export). "
             "Default: results/phase2/inference_input.jsonl",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("results/phase2/corrections.jsonl"),
        help="Path to write corrections.jsonl. "
             "Default: results/phase2/corrections.jsonl",
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("configs/config.yaml"),
        help="Config YAML (for prompt settings). Default: configs/config.yaml",
    )
    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model ID. Default: {DEFAULT_MODEL}",
    )
    p.add_argument(
        "--backend", type=str, default="vllm",
        choices=["vllm", "transformers"],
        help="Inference backend. 'vllm' (default, fastest) or 'transformers' (fallback).",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens per sample. Default: {DEFAULT_MAX_TOKENS}",
    )
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}",
    )
    p.add_argument(
        "--gpu-memory-util", type=float, default=DEFAULT_GPU_MEM_UTIL,
        help=f"vLLM GPU memory utilisation (0-1). Default: {DEFAULT_GPU_MEM_UTIL}",
    )
    p.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=f"vLLM max context length. Default: {DEFAULT_MAX_MODEL_LEN}",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for transformers backend. Default: 16",
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
    p.add_argument(
        "--system-prompt", type=str, default=None, dest="system_prompt",
        help="Path to system prompt file override.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers (matches scripts/infer.py format exactly)
# ---------------------------------------------------------------------------


def _read_completed_ids(output_path: Path) -> set[str]:
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
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run: python pipelines/run_phase2.py --mode export  (or the relevant phase)"
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
            if not r.get("ocr_text", "").strip():
                logger.warning("Skipping %s: empty ocr_text.", r.get("sample_id", "?"))
                continue
            if limit is not None:
                if counts.get(ds, 0) >= limit:
                    continue
                counts[ds] = counts.get(ds, 0) + 1
            records.append(r)
    return records


def _make_out_record(
    record: dict,
    corrected_text: str,
    model_name: str,
    prompt_type: str,
    prompt_ver: str,
    prompt_tokens: int,
    output_tokens: int,
    latency_s: float,
    success: bool,
    error: Optional[str] = None,
) -> dict:
    """Build an output record in the same format as scripts/infer.py."""
    return {
        "sample_id":      record["sample_id"],
        "dataset":        record.get("dataset", ""),
        "ocr_text":       record["ocr_text"],
        "corrected_text": corrected_text,
        "gt_text":        record.get("gt_text", ""),
        "model":          model_name,
        "prompt_type":    prompt_type,
        "prompt_version": prompt_ver,
        "prompt_tokens":  prompt_tokens,
        "output_tokens":  output_tokens,
        "latency_s":      latency_s,
        "success":        success,
        "error":          error,
    }


# ---------------------------------------------------------------------------
# Prompt dispatch (mirrors scripts/infer.py exactly)
# ---------------------------------------------------------------------------


def build_messages(record: dict, builder: PromptBuilder) -> tuple[list[dict], str, str]:
    """Dispatch to the correct PromptBuilder method based on prompt_type.

    Returns (messages, prompt_type_label, prompt_version).
    """
    pt = record.get("prompt_type", "zero_shot")
    few_shot = record.get("few_shot_examples") or ""

    if pt == "ocr_aware":
        msgs = builder.build_ocr_aware(
            record["ocr_text"],
            record.get("confusion_context", ""),
            record.get("word_examples", ""),
            few_shot_examples=few_shot,
        )
        ver = builder.ocr_aware_prompt_version
        if not record.get("confusion_context", "").strip():
            pt = "zero_shot_fallback"

    elif pt == "combined":
        msgs = builder.build_combined(
            ocr_text=record["ocr_text"],
            confusion_context=record.get("confusion_context") or "",
            insights_context=record.get("insights_context") or "",
            word_pairs_context=record.get("word_pairs_context") or "",
            overcorrection_context=record.get("overcorrection_context") or "",
            few_shot_examples=few_shot,
        )
        ver = builder.combined_prompt_version
        if not any([
            record.get("confusion_context"),
            record.get("insights_context"),
            record.get("word_pairs_context"),
            record.get("overcorrection_context"),
        ]):
            pt = "zero_shot_fallback"

    elif pt == "self_reflective":
        msgs = builder.build_self_reflective(
            record["ocr_text"],
            record.get("insights_context", ""),
            record.get("word_pairs_context", ""),
            record.get("overcorrection_context", ""),
            few_shot_examples=few_shot,
        )
        ver = builder.self_reflective_prompt_version
        if not any([
            record.get("insights_context", "").strip(),
            record.get("word_pairs_context", "").strip(),
            record.get("overcorrection_context", "").strip(),
        ]):
            pt = "zero_shot_fallback"

    elif pt == "meta_prompt":
        msgs = builder.build_meta_prompt(record["ocr_text"])
        ver = record.get("prompt_version", builder.meta_prompt_version)

    elif pt == "crafted":
        system_prompt = record.get("system_prompt", "")
        msgs = builder.build_crafted(record["ocr_text"], system_prompt)
        ver = record.get("prompt_version", builder.crafted_prompt_version)
        if not system_prompt.strip():
            pt = "zero_shot_fallback"

    elif pt == "zero_shot":
        pv = record.get("prompt_version", "v1")
        msgs = builder.build_zero_shot(record["ocr_text"], version=pv, few_shot_examples=few_shot)
        ver = pv

    else:
        logger.warning("Unknown prompt_type '%s' for %s — zero_shot fallback.", pt, record["sample_id"])
        msgs = builder.build_zero_shot(record["ocr_text"])
        ver = builder.prompt_version
        pt = "zero_shot_fallback"

    return msgs, pt, ver


# ---------------------------------------------------------------------------
# Text extraction (mirrors TransformersCorrector._extract_corrected_text)
# ---------------------------------------------------------------------------


def _extract_text(raw_output: str, ocr_text: str) -> str:
    """Clean model output — fall back to ocr_text if empty or non-Arabic."""
    cleaned = raw_output.strip()
    if not cleaned:
        return ocr_text
    if not _ARABIC_RE.search(cleaned):
        return ocr_text
    return cleaned


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------


def _build_token_ids(
    tokenizer,
    messages: list[dict],
    record: dict,
    builder: "PromptBuilder",
    max_input_tokens: int,
) -> list[int]:
    """Format messages and tokenize with context-dropping truncation.

    Truncation strategy (in order, preserving OCR text at all costs):
      1. Use full prompt if it fits within max_input_tokens.
      2. If over budget and the prompt has extra context (ocr_aware, combined,
         self_reflective, etc.), rebuild as zero_shot — drops confusion matrix,
         insights, word-pairs, and overcorrection warnings but always keeps the
         OCR text and the correction instruction.
      3. Last resort: keep the TAIL of the zero_shot token sequence.  The OCR
         text and generation prompt are at the end of the sequence, so tail
         truncation preserves them while discarding only leading system content.

    enable_thinking=False suppresses Qwen3's internal <think> scratchpad.
    """
    def _fmt_and_encode(msgs: list[dict]) -> list[int]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # add_special_tokens=False: template already includes BOS/special tokens
        return tokenizer.encode(text, add_special_tokens=False)

    ids = _fmt_and_encode(messages)
    if len(ids) <= max_input_tokens:
        return ids

    # Step 2: drop context — rebuild as zero_shot to keep OCR text + instruction
    pt = record.get("prompt_type", "zero_shot")
    sample_id = record.get("sample_id", "?")
    if pt not in ("zero_shot", "zero_shot_fallback"):
        pv = record.get("prompt_version", "v1")
        fallback_msgs = builder.build_zero_shot(record["ocr_text"], version=pv)
        fallback_ids = _fmt_and_encode(fallback_msgs)
        logger.warning(
            "[%s] Context dropped for truncation: %d -> %d tokens (prompt_type=%s)",
            sample_id, len(ids), len(fallback_ids), pt,
        )
        if len(fallback_ids) <= max_input_tokens:
            return fallback_ids
        ids = fallback_ids

    # Step 3: last resort — keep tail (OCR text + generation prompt are at end)
    logger.warning(
        "[%s] Prompt still over budget (%d tokens) — keeping last %d tokens.",
        sample_id, len(ids), max_input_tokens,
    )
    return ids[-max_input_tokens:]


def run_vllm(
    pending: list[dict],
    builder: PromptBuilder,
    config: dict,
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict]:
    """Run all pending records through vLLM offline inference.

    Submissions are chunked (default 2000 records per chunk) so that results
    are written to disk after each chunk — matching infer.py's per-record flush
    behaviour. If the Thunder instance crashes mid-run, at most one chunk of
    work is lost and the next run resumes from the last written sample_id.

    Within each chunk, ALL prompts are submitted to vLLM at once so continuous
    batching is fully utilised — there is no artificial rate-limiting inside a chunk.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM not installed. Run: pip install vllm  "
            "or use --backend transformers"
        )

    from transformers import AutoTokenizer

    model_name = config.get("model", {}).get("name", args.model)
    temperature = config.get("model", {}).get("temperature", args.temperature)
    max_tokens = config.get("model", {}).get("max_tokens", args.max_tokens)

    logger.info("Loading tokenizer for prompt formatting: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ---------- Build ALL prompts up front ----------
    logger.info("Building %d prompts...", len(pending))
    t_build = time.monotonic()

    prompt_token_ids: list[list[int]] = []
    meta: list[tuple[str, str]] = []  # (prompt_type, prompt_version) per record

    for record in pending:
        messages, pt, ver = build_messages(record, builder)
        ids = _build_token_ids(tokenizer, messages, record, builder, MAX_INPUT_TOKENS)
        prompt_token_ids.append(ids)
        meta.append((pt, ver))

    prompt_lens = [len(ids) for ids in prompt_token_ids]
    logger.info(
        "Prompts built in %.1fs | lengths: min=%d  median=%d  max=%d",
        time.monotonic() - t_build,
        min(prompt_lens),
        sorted(prompt_lens)[len(prompt_lens) // 2],
        max(prompt_lens),
    )

    # ---------- Initialise vLLM ----------
    logger.info(
        "Initialising vLLM: model=%s  gpu_util=%.2f  max_model_len=%d",
        model_name, args.gpu_memory_util, args.max_model_len,
    )
    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        # Prefix caching: reuse KV cache for shared prompt prefixes (helps phases 3/4/6
        # where the system prompt is identical across all records).
        enable_prefix_caching=True,
        # Swap space on CPU for sequences that overflow GPU KV cache
        swap_space=16,
    )

    sampling_params = SamplingParams(
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        skip_special_tokens=True,
    )

    # ---------- Chunked submission with progressive write ----------
    # Each chunk is submitted fully to vLLM (continuous batching), then results
    # are written to disk per-record so a crash loses at most one chunk's work.
    CHUNK_SIZE = 2000
    n_chunks = (len(pending) + CHUNK_SIZE - 1) // CHUNK_SIZE
    out_records: list[dict] = []
    t_start = time.monotonic()

    pbar = tqdm(
        total=len(pending),
        desc="Qwen3 (vLLM)",
        unit="sample",
        dynamic_ncols=True,
    )

    with open(output_path, "a", encoding="utf-8") as out_f:
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end   = min(chunk_start + CHUNK_SIZE, len(pending))

            chunk_records = pending[chunk_start:chunk_end]
            chunk_ids     = prompt_token_ids[chunk_start:chunk_end]
            chunk_meta    = meta[chunk_start:chunk_end]
            chunk_lens    = prompt_lens[chunk_start:chunk_end]

            logger.info(
                "vLLM chunk %d/%d — submitting %d prompts...",
                chunk_idx + 1, n_chunks, len(chunk_records),
            )
            t_chunk = time.monotonic()

            prompts = [{"prompt_token_ids": ids} for ids in chunk_ids]
            outputs = llm.generate(prompts, sampling_params)

            elapsed_chunk = time.monotonic() - t_chunk
            logger.info(
                "  chunk done in %.1fs  (%.1f samples/s)",
                elapsed_chunk, len(chunk_records) / elapsed_chunk if elapsed_chunk > 0 else 0,
            )

            # Write immediately — crash-safe per record
            n_out_tok_chunk = 0
            for record, vllm_out, (pt, ver), prompt_len in zip(
                chunk_records, outputs, chunk_meta, chunk_lens
            ):
                try:
                    raw_text   = vllm_out.outputs[0].text
                    out_tokens = len(vllm_out.outputs[0].token_ids)
                    corrected  = _extract_text(raw_text, record["ocr_text"])
                    success    = True
                    error      = None
                except Exception as exc:
                    corrected  = record["ocr_text"]
                    out_tokens = 0
                    success    = False
                    error      = str(exc)
                    logger.warning("Failed result for %s: %s", record["sample_id"], exc)

                n_out_tok_chunk += out_tokens
                rec = _make_out_record(
                    record=record,
                    corrected_text=corrected,
                    model_name=model_name,
                    prompt_type=pt,
                    prompt_ver=ver,
                    prompt_tokens=prompt_len,
                    output_tokens=out_tokens,
                    latency_s=round(elapsed_chunk / len(chunk_records), 3),
                    success=success,
                    error=error,
                )
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                out_records.append(rec)
                pbar.update(1)

            elapsed_total = time.monotonic() - t_start
            done = len(out_records)
            tok_per_s = n_out_tok_chunk / elapsed_chunk if elapsed_chunk > 0 else 0
            pbar.set_postfix({
                "chunk":   f"{chunk_idx + 1}/{n_chunks}",
                "tok/s":   f"{tok_per_s:.0f}",
                "samp/s":  f"{done / elapsed_total:.1f}" if elapsed_total > 0 else "?",
                "ok/fail": f"{sum(1 for r in out_records if r['success'])}/{sum(1 for r in out_records if not r['success'])}",
            })

    pbar.close()
    elapsed_total = time.monotonic() - t_start
    logger.info(
        "vLLM inference done: %.1fs total  (%.1f samples/s)",
        elapsed_total, len(pending) / elapsed_total if elapsed_total > 0 else 0,
    )
    return out_records


# ---------------------------------------------------------------------------
# Transformers batched backend (fallback)
# ---------------------------------------------------------------------------


def run_transformers_batched(
    pending: list[dict],
    builder: PromptBuilder,
    config: dict,
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict]:
    """Batched HuggingFace transformers inference (fallback when vLLM unavailable).

    Uses left-padding so all sequences in a batch are right-aligned, which is
    correct for causal LM generation. Generated tokens begin at the same
    position for every sample in the batch.

    Results are written to output_path per-record immediately after each batch
    so a crash only loses the current in-flight batch.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.get("model", {}).get("name", args.model)
    temperature = float(config.get("model", {}).get("temperature", args.temperature))
    max_tokens = int(config.get("model", {}).get("max_tokens", args.max_tokens))
    batch_size = args.batch_size

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Left-padding: aligns generated tokens to the same position across the batch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s (FP16, device_map=auto)", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    out_records: list[dict] = []
    n_batches = (len(pending) + batch_size - 1) // batch_size
    t_run_start = time.monotonic()

    pbar = tqdm(
        total=len(pending),
        desc="Qwen3 (transformers)",
        unit="sample",
        dynamic_ncols=True,
    )

    # Write per-record immediately — crash loses only the current in-flight batch
    with open(output_path, "a", encoding="utf-8") as out_f:
        for batch_idx in range(n_batches):
            batch = pending[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            t0 = time.monotonic()

            # Build messages + per-sample safe tokenization
            all_msgs: list[list[dict]] = []
            pts: list[str] = []
            vers: list[str] = []
            for record in batch:
                messages, pt, ver = build_messages(record, builder)
                all_msgs.append(messages)
                pts.append(pt)
                vers.append(ver)

            all_ids: list[list[int]] = []
            for record, msgs in zip(batch, all_msgs):
                ids = _build_token_ids(tokenizer, msgs, record, builder, MAX_INPUT_TOKENS)
                all_ids.append(ids)

            # Manual left-padding (never use tokenizer truncation=True — it cuts OCR text)
            pad_id = tokenizer.pad_token_id
            input_seq_len = max(len(ids) for ids in all_ids)
            input_ids_padded = [[pad_id] * (input_seq_len - len(ids)) + ids for ids in all_ids]
            attn_mask_padded = [[0] * (input_seq_len - len(ids)) + [1] * len(ids) for ids in all_ids]
            prompt_lens = [len(ids) for ids in all_ids]

            device = next(model.parameters()).device
            input_ids = torch.tensor(input_ids_padded, dtype=torch.long, device=device)
            attn_mask = torch.tensor(attn_mask_padded, dtype=torch.long, device=device)

            # Helper: build and flush one output record
            def _write_rec(record, corrected, out_tok, lat, success, err, pt, ver, plen):
                rec = _make_out_record(
                    record=record,
                    corrected_text=corrected,
                    model_name=model_name,
                    prompt_type=pt,
                    prompt_ver=ver,
                    prompt_tokens=plen,
                    output_tokens=out_tok,
                    latency_s=round(lat, 3),
                    success=success,
                    error=err,
                )
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                out_records.append(rec)
                return rec

            try:
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_tokens,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else 1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                new_token_ids = generated[:, input_seq_len:]
                decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
                lat_per_sample = (time.monotonic() - t0) / len(batch)

                for i, (record, raw_text, pt, ver, plen) in enumerate(
                    zip(batch, decoded, pts, vers, prompt_lens)
                ):
                    corrected  = _extract_text(raw_text, record["ocr_text"])
                    out_tokens = int((new_token_ids[i] != tokenizer.pad_token_id).sum().item())
                    _write_rec(record, corrected, out_tokens, lat_per_sample, True, None, pt, ver, plen)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("OOM on batch of %d — processing sample-by-sample.", len(batch))
                for record, ids, pt, ver, plen in zip(batch, all_ids, pts, vers, prompt_lens):
                    t1 = time.monotonic()
                    try:
                        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                        with torch.inference_mode():
                            out = model.generate(
                                ids_t,
                                max_new_tokens=max_tokens,
                                do_sample=(temperature > 0),
                                temperature=temperature if temperature > 0 else 1.0,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        new_ids   = out[:, len(ids):]
                        corrected = _extract_text(
                            tokenizer.decode(new_ids[0], skip_special_tokens=True),
                            record["ocr_text"],
                        )
                        out_tokens = new_ids.shape[1]
                        _write_rec(record, corrected, out_tokens, time.monotonic() - t1, True, None, pt, ver, plen)
                    except Exception as exc:
                        logger.error("Sample %s failed: %s", record["sample_id"], exc)
                        _write_rec(record, record["ocr_text"], 0, time.monotonic() - t1, False, str(exc), pt, ver, plen)

            torch.cuda.empty_cache()
            pbar.update(len(batch))
            elapsed = time.monotonic() - t_run_start
            n_ok   = sum(1 for r in out_records if r["success"])
            n_fail = len(out_records) - n_ok
            pbar.set_postfix({
                "lat/s":   f"{(time.monotonic() - t0) / len(batch):.1f}",
                "samp/s":  f"{len(out_records) / elapsed:.1f}" if elapsed > 0 else "?",
                "ok/fail": f"{n_ok}/{n_fail}",
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

    # Apply CLI overrides
    config.setdefault("model", {})["name"] = args.model

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Thunder inference started: %s", datetime.now(timezone.utc).isoformat())
    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output)
    logger.info("Backend: %s", args.backend)
    logger.info("Model  : %s", args.model)
    logger.info("=" * 60)

    # Determine already-completed sample_ids
    if args.force:
        completed: set[str] = set()
        if args.output.exists():
            args.output.write_text("", encoding="utf-8")
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

    # Initialise prompt builder
    crafted_path = args.system_prompt or config.get("prompt_craft", {}).get("crafted_prompt_path")
    builder = PromptBuilder(crafted_prompt_path=crafted_path)

    # Print prompt preview for verification
    _preview_messages, _, _ = build_messages(pending[0], builder)
    sep = "=" * 70
    print(sep)
    print(
        f"PROMPT PREVIEW  (sample_id={pending[0]['sample_id']}  "
        f"prompt_type={pending[0].get('prompt_type', 'zero_shot')})"
    )
    print(sep)
    for msg in _preview_messages:
        print(f"[{msg['role'].upper()}]")
        print(msg["content"])
        print()
    print(sep)
    print(flush=True)

    # Run inference
    t_start = time.monotonic()

    if args.backend == "vllm":
        out_records = run_vllm(pending, builder, config, args, args.output)
    else:
        # Transformers path writes per-record internally — output_path passed directly
        out_records = run_transformers_batched(pending, builder, config, args, args.output)

    n_success = sum(1 for r in out_records if r["success"])
    n_failed  = len(out_records) - n_success

    elapsed = time.monotonic() - t_start
    logger.info("=" * 60)
    logger.info(
        "Done: %d success | %d failed | %.1fs total | %.1f samples/s",
        n_success, n_failed, elapsed,
        len(out_records) / elapsed if elapsed > 0 else 0,
    )
    logger.info("Output: %s", args.output)
    logger.info("")
    logger.info("Next step (local machine):")
    phase_dir = args.output.parent.name
    logger.info("  python pipelines/run_%s.py --mode analyze", phase_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
