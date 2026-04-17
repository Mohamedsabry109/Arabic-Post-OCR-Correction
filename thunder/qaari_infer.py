#!/usr/bin/env python3
"""Batched Qaari OCR inference for Thunder Compute (A100 GPU).

Primary backend: vLLM (--backend vllm, default)
    Uses vLLM's continuous batching + PagedAttention for Qwen2VL. All images
    in a chunk are submitted at once; vLLM's scheduler admits new images as
    soon as a KV-cache page is freed, with no padding waste between sequences.
    Expected throughput: 3-5x faster than the HF transformers batch path.

    Images are loaded in a background thread pool while the current chunk is
    running on the GPU. By the time vLLM finishes chunk i, chunk i+1 is
    already in RAM — zero GPU stall between chunks.

    Resume: existing non-empty .txt files are skipped automatically.

Fallback backend: HF transformers (--backend transformers)
    Area-sorted batches + PREFETCH_DEPTH=3 pipeline + optional torch.compile.
    Use when vLLM is unavailable or for debugging.

Output mirrors the input directory structure under --output-root:
    {image_root}/pats-a01-data/A01-Akhbar/Akhbar_1.png
        -> {output_root}/pats-a01-data/A01-Akhbar/Akhbar_1.txt

Usage:
    # Standard — vLLM (fastest)
    python thunder/qaari_infer.py \\
        --image-root  ./data/images \\
        --output-root ./data/ocr-results/qaari-results

    # Transformers fallback
    python thunder/qaari_infer.py \\
        --image-root  ./data/images \\
        --output-root ./data/ocr-results/qaari-results \\
        --backend transformers --batch-size 16

    # Resume after crash (default — already-done files are skipped)
    python thunder/qaari_infer.py \\
        --image-root  ./data/images \\
        --output-root ./data/ocr-results/qaari-results

    # Dry-run: count images without inference
    python thunder/qaari_infer.py \\
        --image-root ./data/images \\
        --output-root ./data/ocr-results/qaari-results \\
        --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QAARI_MODEL = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"

QAARI_PROMPT = (
    "Below is the image of one page of a document, as well as some raw textual "
    "content that was previously extracted for it. Just return the plain text "
    "representation of this document as if you were reading it naturally. "
    "Do not hallucinate."
)

IMAGE_EXTENSIONS: tuple[str, ...] = ("png", "jpg", "jpeg", "tiff", "tif", "bmp")

DEFAULT_BATCH_SIZE     = 16
DEFAULT_MAX_NEW_TOKENS = 2000   # Stop at EOS — vLLM wastes nothing unused
DEFAULT_CHUNK_SIZE     = 2000   # Images per vLLM submission (limits peak RAM)
DEFAULT_GPU_MEM_UTIL   = 0.92

# vLLM context: max visual tokens (~1280) + output (~1024) + prompt (~100) = 4096
VLLM_MAX_MODEL_LEN = 4096

# Image loading parallelism
NUM_LOAD_WORKERS = 32   # >= batch_size so all images in a batch load concurrently
PREFETCH_DEPTH   = 3    # HF path: batches pre-loaded ahead of GPU

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
        description="Qaari OCR inference on Thunder Compute A100 (vLLM primary).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image-root",  type=Path, required=True)
    p.add_argument(
        "--output-root", type=Path,
        default=Path("./data/ocr-results/qaari-results"),
    )
    p.add_argument(
        "--backend", type=str, default="vllm",
        choices=["vllm", "transformers"],
        help="Inference backend. 'vllm' (default, fastest) or 'transformers' (fallback).",
    )
    p.add_argument("--model", type=str, default=QAARI_MODEL)
    p.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max tokens to generate per image. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )
    p.add_argument(
        "--gpu-memory-util", type=float, default=DEFAULT_GPU_MEM_UTIL,
        help=f"vLLM GPU memory utilisation (0-1). Default: {DEFAULT_GPU_MEM_UTIL}",
    )
    p.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Images per vLLM submission. Default: {DEFAULT_CHUNK_SIZE}",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for transformers backend. Default: {DEFAULT_BATCH_SIZE}",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="Apply torch.compile to model (transformers backend, ~20%% speedup).",
    )
    p.add_argument(
        "--no-flash-attn", action="store_true",
        help="Disable Flash Attention 2 (transformers backend only).",
    )
    p.add_argument("--force",   action="store_true", help="Overwrite existing .txt files.")
    p.add_argument("--limit",   type=int, default=None, help="Max images to process.")
    p.add_argument("--dry-run", action="store_true",   help="Count images, no inference.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def discover_images(
    image_root: Path,
    output_root: Path,
    force: bool,
    limit: Optional[int],
) -> list[tuple[Path, Path]]:
    """Find (image_path, output_txt_path) pairs, sorted by image area ascending.

    Already-completed files (non-empty .txt) are skipped unless --force is set.
    Sorting by area groups images with similar visual-token counts into the same
    batch, cutting padding waste.
    """
    entries: list[tuple[int, Path, Path]] = []
    for ext in IMAGE_EXTENSIONS:
        for img_path in image_root.rglob(f"*.{ext}"):
            rel = img_path.relative_to(image_root)
            out_path = output_root / rel.with_suffix(".txt")
            if not force and out_path.exists():
                # Empty files count as done too — they represent images the
                # model produced nothing for, and we don't want to retry on
                # resume. Use --force to reprocess.
                continue
            entries.append((_estimate_area(img_path), img_path, out_path))

    entries.sort(key=lambda t: t[0])
    pairs = [(img, out) for _, img, out in entries]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def _estimate_area(img_path: Path) -> int:
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            return w * h
    except Exception:
        return 1_000_000


def _load_image(img_path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to load %s: %s", img_path.name, exc)
        return None


# ---------------------------------------------------------------------------
# vLLM backend  (PRIMARY)
# ---------------------------------------------------------------------------


def run_vllm(
    pending: list[tuple[Path, Path]],
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Run all images through vLLM's continuous batching engine.

    Processing flow per chunk:
        1. Background thread pool loads chunk_size images in parallel.
        2. While vLLM processes chunk i, the next chunk is loading.
        3. vLLM receives all images in the chunk at once.

    Returns (n_success, n_failed).
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM not installed. Run: pip install vllm>=0.6.3  "
            "or use --backend transformers"
        )

    from transformers import AutoProcessor

    logger.info(
        "Initialising vLLM: model=%s  gpu_util=%.2f  max_model_len=%d",
        args.model, args.gpu_memory_util, VLLM_MAX_MODEL_LEN,
    )

    # Build the exact prompt template that official QaariOCR produces.
    # Done once here; reused for every image.
    processor = AutoProcessor.from_pretrained(args.model)
    _template_messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": QAARI_PROMPT},
    ]}]
    _prompt_text = processor.apply_chat_template(
        _template_messages, tokenize=False, add_generation_prompt=True,
    )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",              # matches official QaariOCR dtype on A100
        gpu_memory_utilization=args.gpu_memory_util,
        max_model_len=VLLM_MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        enable_prefix_caching=True,    # reuse KV for fixed prompt prefix
    )

    sampling_params = SamplingParams(
        temperature=0.0,                # greedy: OCR is deterministic
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )

    n_success = n_failed = 0
    chunk_size = args.chunk_size
    n_chunks = (len(pending) + chunk_size - 1) // chunk_size
    t_start = time.monotonic()
    t_load_wait = 0.0

    pbar = tqdm(
        total=len(pending),
        desc="Qaari OCR (vLLM)",
        unit="img",
        dynamic_ncols=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS) as pool:
        def _submit_chunk_load(chunk):
            futures  = [pool.submit(_load_image, img_p) for img_p, _ in chunk]
            out_paths = [out_p for _, out_p in chunk]
            return futures, out_paths

        if not pending:
            pbar.close()
            return 0, 0

        chunks = [
            pending[i : i + chunk_size]
            for i in range(0, len(pending), chunk_size)
        ]

        next_load_idx = 0
        prefetch_q: deque[tuple[list, list[Path]]] = deque()

        futures, out_ps = _submit_chunk_load(chunks[next_load_idx])
        prefetch_q.append((futures, out_ps))
        next_load_idx += 1

        for chunk_idx, chunk in enumerate(chunks):
            load_futures, cur_out_paths = prefetch_q.popleft()

            if next_load_idx < len(chunks):
                nf, nop = _submit_chunk_load(chunks[next_load_idx])
                prefetch_q.append((nf, nop))
                next_load_idx += 1

            t_w0 = time.monotonic()
            images = [f.result() for f in load_futures]
            t_load_wait += time.monotonic() - t_w0

            valid_pairs = [
                (out_p, img) for out_p, img in zip(cur_out_paths, images)
                if img is not None
            ]
            n_failed += len(images) - len(valid_pairs)

            if not valid_pairs:
                pbar.update(len(images))
                continue

            valid_out_paths, valid_images = zip(*valid_pairs)

            vllm_prompts = [
                {"prompt": _prompt_text, "multi_modal_data": {"image": img}}
                for img in valid_images
            ]

            t_chunk = time.monotonic()
            chunk_outputs = llm.generate(vllm_prompts, sampling_params)
            elapsed_chunk = time.monotonic() - t_chunk

            # Save results immediately — crash-safe per image
            for out_path, vllm_out in zip(valid_out_paths, chunk_outputs):
                try:
                    text = vllm_out.outputs[0].text.strip()
                    if text:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(text, encoding="utf-8")
                        n_success += 1
                    else:
                        logger.warning("Empty output for %s", out_path.name)
                        n_failed += 1
                except Exception as exc:
                    logger.error("Save failed for %s: %s", out_path.name, exc)
                    n_failed += 1

            pbar.update(len(images))
            done = n_success + n_failed
            elapsed_total = time.monotonic() - t_start
            pbar.set_postfix({
                "chunk":      f"{chunk_idx + 1}/{n_chunks}",
                "img/s":      f"{done / elapsed_total:.1f}" if elapsed_total > 0 else "?",
                "lat/img":    f"{elapsed_chunk / len(valid_pairs):.2f}s",
                "load_wait":  f"{t_load_wait:.1f}s",
                "ok/fail":    f"{n_success}/{n_failed}",
            })

    pbar.close()
    elapsed = time.monotonic() - t_start
    logger.info(
        "vLLM done: %d success | %d failed | %.1fs | %.2f img/s",
        n_success, n_failed, elapsed,
        (n_success + n_failed) / elapsed if elapsed > 0 else 0,
    )
    return n_success, n_failed


# ---------------------------------------------------------------------------
# HF transformers backend  (FALLBACK)
# ---------------------------------------------------------------------------


def _load_hf_model(model_id: str, use_flash_attn: bool, compile_model: bool):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    logger.info("Loading Qaari processor: %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    # dtype="auto" matches official QaariOCR without FA2 (resolves to bfloat16 on A100).
    # With FA2 the official code uses bfloat16 explicitly. Never float16.
    kwargs: dict = dict(torch_dtype="auto", device_map="cuda")
    if use_flash_attn:
        logger.info("Flash Attention 2 requested — will try at model load time.")
        kwargs["attn_implementation"] = "flash_attention_2"
        kwargs["torch_dtype"] = torch.bfloat16

    logger.info("Loading Qaari model: %s", model_id)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        if use_flash_attn:
            logger.warning(
                "Flash Attention 2 load failed (%s) — retrying without FA2.", exc,
            )
            kwargs.pop("attn_implementation", None)
            kwargs["torch_dtype"] = "auto"
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        else:
            raise

    model.eval()

    if compile_model:
        logger.info("Applying torch.compile (dynamic=True, mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        logger.info("torch.compile done.")

    return model, processor


def _hf_run_single(model, processor, pil_img: Image.Image, max_new_tokens: int) -> str:
    """Run inference on a single image — exact match to official QaariOCR."""
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text",  "text":  QAARI_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


def _hf_run_batch(
    model,
    processor,
    pairs: list[tuple[Path, Optional[Image.Image]]],
    max_new_tokens: int,
    batch_size_ref: list[int],
) -> list[tuple[Path, str]]:
    """Run one fixed-size batch; fall back per-image on OOM or any error."""
    from qwen_vl_utils import process_vision_info

    valid = [(p, img) for p, img in pairs if img is not None]
    if not valid:
        return []

    try:
        paths, imgs = zip(*valid)
        batch_msgs = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  QAARI_PROMPT},
            ]}]
            for img in imgs
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_msgs
        ]
        flat_msgs = [msg for msgs in batch_msgs for msg in msgs]
        img_inputs, vid_inputs = process_vision_info(flat_msgs)

        inputs = processor(
            text=list(texts), images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens)

        prompt_len = inputs.input_ids.shape[1]
        decoded = processor.batch_decode(
            gen[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return list(zip(paths, decoded))

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        new_bs = max(1, batch_size_ref[0] // 2)
        logger.warning("OOM — halving batch_size to %d, falling back per-image.", new_bs)
        batch_size_ref[0] = new_bs
        return _fallback_per_image(model, processor, valid, max_new_tokens)

    except Exception as exc:
        torch.cuda.empty_cache()
        logger.warning("Batch inference failed (%s) — falling back per-image.", exc)
        return _fallback_per_image(model, processor, valid, max_new_tokens)


def _fallback_per_image(
    model, processor,
    valid: list[tuple[Path, Image.Image]],
    max_new_tokens: int,
) -> list[tuple[Path, str]]:
    """Process each image individually — used as OOM/error fallback."""
    results = []
    for out_path, img in valid:
        try:
            text = _hf_run_single(model, processor, img, max_new_tokens)
            results.append((out_path, text))
        except Exception as exc:
            logger.error("Failed %s: %s", out_path.name, exc)
    return results


def run_transformers(
    pending: list[tuple[Path, Path]],
    args: argparse.Namespace,
) -> tuple[int, int]:
    """HF transformers batched inference with deep prefetch pipeline."""
    model, processor = _load_hf_model(
        args.model,
        use_flash_attn=not args.no_flash_attn,
        compile_model=args.compile,
    )
    batch_size = [args.batch_size]

    # Build initial batch list — may be rebuilt if OOM halves batch_size
    def _make_batches(items, bs):
        return [items[i : i + bs] for i in range(0, len(items), bs)]

    batches = _make_batches(pending, batch_size[0])

    n_success = n_failed = 0
    t_start = time.monotonic()
    t_load_wait = 0.0

    pbar = tqdm(
        total=len(pending),
        desc="Qaari OCR (transformers)",
        unit="img",
        dynamic_ncols=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS) as pool:
        def _submit_batch_load(batch):
            futures   = [pool.submit(_load_image, img_p) for img_p, _ in batch]
            out_paths = [out_p for _, out_p in batch]
            return futures, out_paths

        if not batches:
            pbar.close()
            return 0, 0

        pipeline: deque[tuple[list, list[Path]]] = deque()
        next_submit = 0
        for _ in range(min(PREFETCH_DEPTH, len(batches))):
            futs, ops = _submit_batch_load(batches[next_submit])
            pipeline.append((futs, ops))
            next_submit += 1

        batch_idx = 0
        while batch_idx < len(batches):
            load_futs, cur_out_paths = pipeline.popleft()

            if next_submit < len(batches):
                futs, ops = _submit_batch_load(batches[next_submit])
                pipeline.append((futs, ops))
                next_submit += 1

            t_w0 = time.monotonic()
            images = [f.result() for f in load_futs]
            t_load_wait += time.monotonic() - t_w0

            pairs = list(zip(cur_out_paths, images))
            t_batch = time.monotonic()

            try:
                results = _hf_run_batch(
                    model, processor, pairs, args.max_new_tokens, batch_size,
                )
            except Exception as exc:
                logger.error("Batch %d crashed unexpectedly: %s — skipping.", batch_idx, exc)
                n_failed += len(pairs)
                pbar.update(len(pairs))
                batch_idx += 1
                continue

            elapsed_batch = time.monotonic() - t_batch

            # Rebuild batches if OOM shrank batch_size mid-run
            prev_bs = args.batch_size
            if batch_size[0] != prev_bs:
                remaining = [p for b in batches[batch_idx + 1:] for p in b]
                new_batches = _make_batches(remaining, batch_size[0])
                pipeline.clear()
                next_submit_new = 0
                for _ in range(min(PREFETCH_DEPTH, len(new_batches))):
                    futs, ops = _submit_batch_load(new_batches[next_submit_new])
                    pipeline.append((futs, ops))
                    next_submit_new += 1
                batches = batches[:batch_idx + 1] + new_batches
                next_submit = batch_idx + 1 + next_submit_new

            # Write results immediately — crash-safe
            for out_path, text in results:
                if text.strip():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(text, encoding="utf-8")
                    n_success += 1
                else:
                    logger.warning("Empty output for %s", out_path.name)
                    n_failed += 1

            n_failed += len(pairs) - len(results)   # images that errored in fallback
            pbar.update(len(pairs))

            done = n_success + n_failed
            elapsed_total = time.monotonic() - t_start
            pbar.set_postfix({
                "img/s":      f"{done / elapsed_total:.1f}" if elapsed_total > 0 else "?",
                "lat/batch":  f"{elapsed_batch:.1f}s",
                "load_wait":  f"{t_load_wait:.1f}s",
                "ok/fail":    f"{n_success}/{n_failed}",
            })

            torch.cuda.empty_cache()
            batch_idx += 1

    pbar.close()
    elapsed = time.monotonic() - t_start
    logger.info(
        "Transformers done: %d success | %d failed | %.1fs | %.2f img/s",
        n_success, n_failed, elapsed,
        (n_success + n_failed) / elapsed if elapsed > 0 else 0,
    )
    return n_success, n_failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not args.image_root.exists():
        logger.error("--image-root does not exist: %s", args.image_root)
        sys.exit(1)

    logger.info("Scanning images under: %s", args.image_root)
    pending = discover_images(args.image_root, args.output_root, args.force, args.limit)

    total_images = sum(
        1 for ext in IMAGE_EXTENSIONS
        for _ in args.image_root.rglob(f"*.{ext}")
    )
    n_done = total_images - len(pending)
    logger.info(
        "Images: %d total | %d already done | %d pending",
        total_images, n_done, len(pending),
    )

    if not pending:
        logger.info("Nothing to do. (Use --force to reprocess.)")
        return

    if args.dry_run:
        logger.info("Dry-run — first 20 of %d pending images:", len(pending))
        for img_path, out_path in pending[:20]:
            print(f"  {img_path} -> {out_path}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    # ------------------------------------------------------------------
    # Print configuration + prompt before inference starts
    # ------------------------------------------------------------------
    sep = "=" * 70
    print(sep)
    print(f"  Qaari OCR Inference — {datetime.now(timezone.utc).isoformat()}")
    print(sep)
    print(f"  Backend        : {args.backend}")
    print(f"  Model          : {args.model}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    if args.backend == "vllm":
        print(f"  GPU mem util   : {args.gpu_memory_util}")
        print(f"  Chunk size     : {args.chunk_size}")
    else:
        print(f"  Batch size     : {args.batch_size}")
        print(f"  torch.compile  : {args.compile}")
        print(f"  Flash Attn 2   : {not args.no_flash_attn}")
    print(f"  Pending images : {len(pending)}")
    print(sep)
    print("  PROMPT USED:")
    print(f"  {QAARI_PROMPT}")
    print(sep, flush=True)
    print()

    t_start = time.monotonic()

    if args.backend == "vllm":
        n_success, n_failed = run_vllm(pending, args)
    else:
        n_success, n_failed = run_transformers(pending, args)

    elapsed = time.monotonic() - t_start
    total_done = n_success + n_failed

    print()
    print(sep)
    print(f"  Done   : {n_success} success | {n_failed} failed")
    print(f"  Time   : {elapsed:.1f}s  ({total_done / elapsed:.2f} img/s)" if elapsed > 0 else "")
    print(f"  Output : {args.output_root}")
    print(sep, flush=True)


if __name__ == "__main__":
    main()
