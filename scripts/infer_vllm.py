#!/usr/bin/env python3
"""Fast vLLM inference for Arabic OCR post-correction.

Drop-in replacement for scripts/infer.py with identical JSONL output format.
Uses vLLM's offline LLM engine (continuous batching + PagedAttention) for
significantly higher throughput than the HuggingFace transformers backend.

Typical speedup: ~10-20x on T4, higher on A100.

Install:
    pip install vllm

Usage (same flags as infer.py, plus vLLM-specific options):
    # Basic (mirrors infer.py defaults)
    python scripts/infer_vllm.py

    # Phase 3 with HF sync
    python scripts/infer_vllm.py \\
        --input  results/phase3/inference_input.jsonl \\
        --output results/phase3/corrections.jsonl \\
        --hf-repo user/arabic-ocr-corrections \\
        --hf-token hf_xxx

vLLM-specific flags:
    --tensor-parallel-size N    GPUs for tensor parallelism (default: 1)
    --gpu-memory-utilization F  KV-cache VRAM fraction (default: 0.90)
    --max-model-len N           Override context window (default: 16384)
    --enforce-eager             Disable CUDA graphs (lower VRAM, slightly slower)
    --quantization Q            awq | gptq | bitsandbytes | fp8 (default: none)
    --chunk-size N              Samples per llm.generate() call for HF sync (default: all)

Notes
-----
- latency_s in output is wall-clock time / chunk_size (averaged), not per-sample.
- Qwen3 <think> suppression is applied via apply_chat_template enable_thinking=False.
- Gemma and other models automatically skip the enable_thinking kwarg.
"""

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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Experiment 3 model registry
# ---------------------------------------------------------------------------
EXPERIMENT_MODELS: dict[str, dict[str, str]] = {
    "qwen3-4b":    {"backend": "transformers", "model_id": "Qwen/Qwen3-4B-Instruct-2507"},
    "qwen3-14b":   {"backend": "transformers", "model_id": "Qwen/Qwen3-14B"},
    "gemma-3-4b":  {"backend": "gemma",        "model_id": "google/gemma-3-4b-it"},
    "gemma-3-12b": {"backend": "gemma",        "model_id": "google/gemma-3-12b-it"},
}

from src.core.prompt_builder import PromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_MAX_INPUT_TOKENS: int = 16384  # mirrors TransformersCorrector.MAX_INPUT_TOKENS


# ---------------------------------------------------------------------------
# HF sync helpers — mirrors infer.py exactly
# ---------------------------------------------------------------------------


def _resolve_hf_path(output_path: Path, hf_path: Optional[str] = None) -> str:
    if hf_path:
        return hf_path
    parts = output_path.resolve().parts
    for i, part in enumerate(parts):
        if part == "results" and i + 1 < len(parts):
            return "/".join(parts[i + 1:])
    return output_path.name


def hf_pull(
    hf_repo: str, token: str, output_path: Path,
    hf_path: Optional[str] = None,
) -> set[str]:
    path_in_repo = _resolve_hf_path(output_path, hf_path)
    try:
        from huggingface_hub import hf_hub_download
        logger.info("HF pull: %s/%s ...", hf_repo, path_in_repo)
        tmp = hf_hub_download(
            repo_id=hf_repo, filename=path_in_repo,
            repo_type="dataset", token=token,
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

    merged = {**local, **remote}
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
# Prompt / tokenisation helpers
# ---------------------------------------------------------------------------


def _apply_chat_template(tokenizer, messages: list[dict], extra_kwargs: dict) -> str:
    """Format messages; silently drop kwargs the model doesn't accept (e.g. enable_thinking)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _fit_to_budget(
    tokenizer,
    messages: list[dict],
    budget: int,
    extra_kwargs: dict,
) -> str:
    """Format messages → prompt string, trimming injected context if over budget.

    Mirrors TransformersCorrector._fit_to_token_budget: trims the system
    message tail first (removing injected context blocks), then falls back
    to tail-truncating the full token sequence.  The OCR text (user message)
    is never modified.
    """
    def _fmt(msgs: list[dict]) -> tuple[str, int]:
        text = _apply_chat_template(tokenizer, msgs, extra_kwargs)
        return text, len(tokenizer.encode(text, add_special_tokens=False))

    full_text, full_len = _fmt(messages)
    if full_len <= budget:
        return full_text

    sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    if sys_idx is not None:
        shell_msgs = [{**m, "content": ""} if i == sys_idx else m for i, m in enumerate(messages)]
        _, shell_cost = _fmt(shell_msgs)
        sys_budget = budget - shell_cost
        if sys_budget > 0:
            sys_ids = tokenizer.encode(messages[sys_idx]["content"], add_special_tokens=False)
            if len(sys_ids) > sys_budget:
                trimmed_sys = tokenizer.decode(sys_ids[:sys_budget], skip_special_tokens=True)
                trimmed_msgs = list(messages)
                trimmed_msgs[sys_idx] = {**messages[sys_idx], "content": trimmed_sys}
                trimmed_text, trimmed_len = _fmt(trimmed_msgs)
                logger.warning("Prompt context truncated: %d -> %d tokens.", full_len, trimmed_len)
                if trimmed_len <= budget:
                    return trimmed_text
                full_text, full_len = trimmed_text, trimmed_len

    logger.warning("Tail-truncating prompt: %d -> %d tokens.", full_len, budget)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    return tokenizer.decode(full_ids[-budget:], skip_special_tokens=True)


def _extract_corrected_text(
    raw: str, ocr_text: str, sample_id: str,
) -> tuple[str, bool, Optional[str]]:
    """Clean model output; fall back to ocr_text on empty or non-Arabic output."""
    cleaned = raw.strip()
    if not cleaned:
        logger.warning("[%s] Empty model output — using OCR text as fallback.", sample_id)
        return ocr_text, False, "empty_output"
    if not _ARABIC_RE.search(cleaned):
        logger.warning("[%s] No Arabic in output (%r…) — fallback.", sample_id, cleaned[:50])
        return ocr_text, False, "no_arabic_output"
    return cleaned, True, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM inference for Arabic OCR correction (drop-in for infer.py)."
    )
    # Same flags as infer.py
    p.add_argument(
        "--input", type=Path, default=Path("results/phase2/inference_input.jsonl"),
        help="Path to inference_input.jsonl.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/phase2/corrections.jsonl"),
        help="Path to write corrections.jsonl.",
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/config.yaml"),
        help="Path to config YAML.",
    )
    p.add_argument("--model", type=str, default=None, help="Override model name from config.")
    p.add_argument(
        "--backend", type=str, default=None, choices=["transformers", "gemma", "mock", "api"],
        help=(
            "'gemma' disables enable_thinking kwarg. "
            "'mock' skips vLLM load and returns OCR text unchanged (smoke test)."
        ),
    )
    p.add_argument("--datasets", nargs="+", default=None, metavar="DATASET")
    p.add_argument("--limit", type=int, default=None, help="Max samples per dataset.")
    p.add_argument("--force", action="store_true", help="Re-run all samples, ignoring existing output.")
    p.add_argument("--max-retries", type=int, default=1, help="Retry count on empty output.")
    p.add_argument("--hf-repo", type=str, default=None)
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--sync-every", type=int, default=100)
    p.add_argument("--hf-path", type=str, default=None)
    p.add_argument(
        "--system-prompt", type=str, default=None, dest="system_prompt",
        help="Path to system prompt file (overrides configs/crafted_system_prompt.txt).",
    )
    # vLLM-specific flags
    p.add_argument(
        "--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size",
        help="Number of GPUs for tensor parallelism (default: 1).",
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90, dest="gpu_memory_utilization",
        help="Fraction of GPU VRAM reserved for KV cache (default: 0.90).",
    )
    p.add_argument(
        "--max-model-len", type=int, default=None, dest="max_model_len",
        help="Override model context window in tokens (default: 16384).",
    )
    p.add_argument(
        "--enforce-eager", action="store_true", dest="enforce_eager",
        help="Disable CUDA graphs — lower peak VRAM, slightly lower throughput.",
    )
    p.add_argument(
        "--quantization", type=str, default=None,
        choices=["awq", "gptq", "bitsandbytes", "fp8"],
        help="Quantization method (default: none). Use 'bitsandbytes' for 4-bit on low VRAM.",
    )
    p.add_argument(
        "--chunk-size", type=int, default=None, dest="chunk_size",
        help=(
            "Process N samples per llm.generate() call (enables periodic HF sync). "
            "Default: all pending at once (maximum throughput)."
        ),
    )
    p.add_argument(
        "--v1", action="store_true", dest="use_v1_engine",
        help=(
            "Enable vLLM V1 engine (default: off). V1 uses UvaBuffer pinned memory "
            "which fails on some cloud VMs (cudaHostGetDevicePointer invalid argument). "
            "V0 engine is used by default for compatibility."
        ),
    )
    p.add_argument(
        "--experiment-model", type=str, default=None, dest="experiment_model",
        choices=list(EXPERIMENT_MODELS.keys()),
        help=(
            "Experiment 3 model shorthand — sets --backend and --model automatically. "
            "Choices: " + ", ".join(EXPERIMENT_MODELS.keys())
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.config.exists():
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found at %s — using defaults.", args.config)
        config = {}

    # --experiment-model sets both backend and model_id; --model/--backend can override further
    if args.experiment_model:
        em = EXPERIMENT_MODELS[args.experiment_model]
        config.setdefault("model", {})["backend"] = em["backend"]
        config.setdefault("model", {})["name"]    = em["model_id"]
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.backend:
        config.setdefault("model", {})["backend"] = args.backend

    model_cfg = config.get("model", {})
    model_id: str = model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507")
    temperature: float = float(model_cfg.get("temperature", 0.1))
    max_new_tokens: int = int(model_cfg.get("max_tokens", 1024))
    backend: str = model_cfg.get("backend", "transformers")
    max_model_len: int = args.max_model_len or _MAX_INPUT_TOKENS

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("vLLM inference started: %s", datetime.now(timezone.utc).isoformat())
    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output)
    logger.info("Model  : %s", model_id)
    logger.info("Engine : vllm (tp=%d, gpu_mem=%.2f, max_len=%d)",
                args.tensor_parallel_size, args.gpu_memory_utilization, max_model_len)
    logger.info("=" * 60)

    # HF sync
    token: Optional[str] = args.hf_token or os.environ.get("HF_TOKEN")
    use_hf = bool(args.hf_repo and token)
    if args.hf_repo and not token:
        logger.warning("--hf-repo provided but no HF token found. HF sync disabled.")
    if use_hf:
        hf_remote_path = _resolve_hf_path(args.output, args.hf_path)
        logger.info("HF sync: %s -> %s/%s (every %d samples)",
                    args.output.name, args.hf_repo, hf_remote_path, args.sync_every)

    # Completed IDs
    if args.force:
        completed: set[str] = set()
        if args.output.exists():
            args.output.write_text("", encoding="utf-8")
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

    # Tokenizer — used only for prompt formatting; vLLM loads its own internally
    logger.info("Loading tokenizer: %s", model_id)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # enable_thinking=False suppresses Qwen3 <think> scratchpad at template level.
    # Gemma and other models don't accept this kwarg — _apply_chat_template handles it gracefully.
    use_thinking_kwarg = backend != "gemma" and "gemma" not in model_id.lower()
    extra_kwargs: dict = {"enable_thinking": False} if use_thinking_kwarg else {}

    crafted_prompt_path = args.system_prompt or config.get("prompt_craft", {}).get("crafted_prompt_path")
    builder = PromptBuilder(crafted_prompt_path=crafted_prompt_path)

    # Token budget: same formula as TransformersCorrector
    budget = max_model_len - max_new_tokens - 32

    # ------------------------------------------------------------------
    # Prompt preparation (CPU — fast)
    # Each entry: (prompt_string, effective_prompt_type, prompt_version)
    # ------------------------------------------------------------------

    def _prepare_record(rec: dict) -> tuple[str, str, str]:
        pt = rec.get("prompt_type", "zero_shot")
        few_shot = rec.get("few_shot_examples") or ""

        if pt == "ocr_aware":
            confusion_context = rec.get("confusion_context", "")
            word_examples = rec.get("word_examples", "")
            msgs = builder.build_ocr_aware(
                rec["ocr_text"], confusion_context, word_examples,
                few_shot_examples=few_shot,
            )
            pv = builder.ocr_aware_prompt_version
            if not confusion_context.strip():
                pt = "zero_shot_fallback"
        elif pt == "combined":
            msgs = builder.build_combined(
                ocr_text=rec["ocr_text"],
                confusion_context=rec.get("confusion_context") or "",
                insights_context=rec.get("insights_context") or "",
                word_pairs_context=rec.get("word_pairs_context") or "",
                overcorrection_context=rec.get("overcorrection_context") or "",
                few_shot_examples=few_shot,
                word_examples=rec.get("word_examples") or "",
            )
            pv = builder.combined_prompt_version
            if not any([
                rec.get("confusion_context"), rec.get("insights_context"),
                rec.get("word_pairs_context"), rec.get("overcorrection_context"),
            ]):
                pt = "zero_shot_fallback"
        elif pt == "self_reflective":
            insights_context = rec.get("insights_context", "")
            word_pairs_context = rec.get("word_pairs_context", "")
            overcorrection_context = rec.get("overcorrection_context", "")
            msgs = builder.build_self_reflective(
                rec["ocr_text"], insights_context, word_pairs_context,
                overcorrection_context, few_shot_examples=few_shot,
            )
            pv = builder.self_reflective_prompt_version
            if (not insights_context.strip()
                    and not word_pairs_context.strip()
                    and not overcorrection_context.strip()):
                pt = "zero_shot_fallback"
        elif pt == "rag":
            retrieved_sentences = rec.get("retrieved_sentences", "")
            retrieved_words = rec.get("retrieved_words", "")
            msgs = builder.build_rag(
                rec["ocr_text"],
                retrieved_sentences=retrieved_sentences,
                retrieved_words=retrieved_words,
                few_shot_examples=few_shot,
            )
            pv = rec.get("prompt_version", "p8v1")
            if not retrieved_sentences.strip() and not retrieved_words.strip():
                pt = "zero_shot_fallback"
        elif pt == "meta_prompt":
            msgs = builder.build_meta_prompt(rec["ocr_text"])
            pv = rec.get("prompt_version", builder.meta_prompt_version)
        elif pt == "crafted":
            system_prompt = rec.get("system_prompt", "")
            msgs = builder.build_crafted(rec["ocr_text"], system_prompt)
            pv = rec.get("prompt_version", builder.crafted_prompt_version)
            if not system_prompt.strip():
                pt = "zero_shot_fallback"
        elif pt == "zero_shot":
            pv = rec.get("prompt_version", "v1")
            msgs = builder.build_zero_shot(rec["ocr_text"], version=pv, few_shot_examples=few_shot)
        else:
            logger.warning(
                "Unknown prompt_type '%s' for %s — falling back to zero_shot.",
                pt, rec["sample_id"],
            )
            msgs = builder.build_zero_shot(rec["ocr_text"])
            pv = builder.prompt_version
            pt = "zero_shot_fallback"

        prompt_str = _fit_to_budget(tokenizer, msgs, budget, extra_kwargs)
        return prompt_str, pt, pv

    logger.info("Building %d prompts ...", len(pending))
    t_prep = time.monotonic()
    prepared: list[tuple[str, str, str]] = [
        _prepare_record(r)
        for r in tqdm(pending, desc="Preparing prompts", unit="sample")
    ]
    logger.info("Prompts built in %.1fs", time.monotonic() - t_prep)

    # Preview first prompt (same as infer.py)
    _sep = "=" * 70
    _first_prompt, _first_pt, _ = prepared[0]
    print(_sep, flush=True)
    print(
        f"PROMPT PREVIEW  (sample_id={pending[0]['sample_id']}  prompt_type={_first_pt})",
        flush=True,
    )
    print(_sep, flush=True)
    print(_first_prompt, flush=True)
    print(_sep, flush=True)
    print(flush=True)

    # ------------------------------------------------------------------
    # Load vLLM engine (after prompt prep, so tokenizer errors surface first)
    # ------------------------------------------------------------------

    if backend == "mock":
        llm = None
        sampling_params = None
        logger.info("Mock backend — vLLM not loaded.")
    else:
        if args.use_v1_engine:
            logger.info("Using vLLM V1 engine.")
        else:
            logger.info("Using vLLM default engine.")

        logger.info("Loading vLLM engine: %s ...", model_id)
        from vllm import LLM, SamplingParams  # type: ignore[import]

        llm = LLM(
            model=model_id,
            dtype="float16",
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=args.enforce_eager,
            quantization=args.quantization,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        logger.info("vLLM engine ready.")

    # ------------------------------------------------------------------
    # Inference loop (in chunks for HF sync support)
    # ------------------------------------------------------------------

    chunk_size = args.chunk_size or len(pending)
    n_chunks = (len(pending) + chunk_size - 1) // chunk_size
    n_success = n_failed = 0
    total_processed = 0
    pushed_at = 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        for chunk_start in tqdm(range(0, len(pending), chunk_size),
                                desc="Inference", unit="chunk", total=n_chunks):
            chunk_end = chunk_start + chunk_size
            chunk_records = pending[chunk_start:chunk_end]
            chunk_prepared = prepared[chunk_start:chunk_end]
            chunk_prompts = [p[0] for p in chunk_prepared]
            chunk_pt      = [p[1] for p in chunk_prepared]
            chunk_pv      = [p[2] for p in chunk_prepared]

            t0 = time.monotonic()

            if llm is None:
                # Mock: identity correction, no model
                raw_texts        = [r["ocr_text"] for r in chunk_records]
                prompt_toks_list = [max(1, len(p) // 4) for p in chunk_prompts]
                output_toks_list = [max(1, len(r["ocr_text"].split())) for r in chunk_records]
            else:
                vllm_outputs = llm.generate(chunk_prompts, sampling_params)
                raw_texts        = [o.outputs[0].text for o in vllm_outputs]
                prompt_toks_list = [len(o.prompt_token_ids) for o in vllm_outputs]
                output_toks_list = [len(o.outputs[0].token_ids) for o in vllm_outputs]

                # Retry empty outputs individually (rare; keeps success rate high)
                empty_idx = [i for i, t in enumerate(raw_texts) if not t.strip()]
                for attempt in range(args.max_retries):
                    if not empty_idx:
                        break
                    logger.warning(
                        "Retrying %d empty outputs (attempt %d/%d) ...",
                        len(empty_idx), attempt + 1, args.max_retries,
                    )
                    retry_out = llm.generate(
                        [chunk_prompts[i] for i in empty_idx], sampling_params,
                    )
                    still_empty = []
                    for j, i in enumerate(empty_idx):
                        retry_text = retry_out[j].outputs[0].text
                        if retry_text.strip():
                            raw_texts[i]        = retry_text
                            prompt_toks_list[i] = len(retry_out[j].prompt_token_ids)
                            output_toks_list[i] = len(retry_out[j].outputs[0].token_ids)
                        else:
                            still_empty.append(i)
                    empty_idx = still_empty

            elapsed = time.monotonic() - t0
            latency_per_sample = round(elapsed / len(chunk_records), 3)

            for record, raw, pt, pv, ptoks, otoks in zip(
                chunk_records, raw_texts,
                chunk_pt, chunk_pv,
                prompt_toks_list, output_toks_list,
            ):
                corrected, success, error = _extract_corrected_text(
                    raw, record["ocr_text"], record["sample_id"],
                )
                out_record = {
                    "sample_id":      record["sample_id"],
                    "dataset":        record.get("dataset", ""),
                    "ocr_source":     record.get("ocr_source", "qaari"),
                    "ocr_text":       record["ocr_text"],
                    "corrected_text": corrected,
                    "gt_text":        record.get("gt_text", ""),
                    "model":          model_id,
                    "prompt_type":    pt,
                    "prompt_version": pv,
                    "prompt_tokens":  ptoks,
                    "output_tokens":  otoks,
                    "latency_s":      latency_per_sample,
                    "success":        success,
                    "error":          error,
                }
                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                if success:
                    n_success += 1
                else:
                    n_failed += 1

            out_f.flush()
            total_processed += len(chunk_records)

            tps = len(chunk_records) / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Chunk done: %d samples in %.1fs (%.1f samples/s)",
                len(chunk_records), elapsed, tps,
            )

            if use_hf and (total_processed - pushed_at) >= args.sync_every:
                hf_push(args.hf_repo, token, args.output, args.hf_path)  # type: ignore[arg-type]
                pushed_at = total_processed

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
