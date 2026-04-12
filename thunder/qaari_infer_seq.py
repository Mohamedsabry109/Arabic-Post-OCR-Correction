#!/usr/bin/env python3
"""Sequential (single-image) Qaari OCR inference — reference implementation.

Processes one image at a time using HuggingFace transformers.  No batching,
no vLLM.  Use this script to produce a result set you can compare directly
against thunder/qaari_infer.py (vLLM / batched paths) to verify numerical
equivalence.

Resume: existing non-empty .txt files are skipped automatically on restart.

Output format is identical to qaari_infer.py: one .txt file per image,
mirroring the input directory structure under --output-root.

Usage:
    python thunder/qaari_infer_seq.py \\
        --image-root  ./data/images \\
        --output-root ./data/ocr-results/qaari-seq-results

    # Resume after crash (default)
    python thunder/qaari_infer_seq.py \\
        --image-root  ./data/images \\
        --output-root ./data/ocr-results/qaari-seq-results

    # Smoke test
    python thunder/qaari_infer_seq.py \\
        --image-root  ./data/images \\
        --output-root ./data/ocr-results/qaari-seq-results \\
        --limit 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import shared helpers from the optimised script to avoid drift
# ---------------------------------------------------------------------------

_THUNDER_DIR = Path(__file__).resolve().parent
if str(_THUNDER_DIR) not in sys.path:
    sys.path.insert(0, str(_THUNDER_DIR))

from qaari_infer import (  # noqa: E402
    QAARI_MODEL,
    QAARI_PROMPT,
    IMAGE_EXTENSIONS,
    DEFAULT_MAX_NEW_TOKENS,
    discover_images,
    _estimate_area,    # noqa: F401 (used inside discover_images)
    _load_image,
    _load_hf_model,
    _hf_run_single,
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
        description="Sequential Qaari OCR inference (transformers, one image at a time).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image-root",  type=Path, required=True)
    p.add_argument(
        "--output-root", type=Path,
        default=Path("./data/ocr-results/qaari-seq-results"),
    )
    p.add_argument("--model", type=str, default=QAARI_MODEL)
    p.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max tokens per image. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )
    p.add_argument(
        "--max-retries", type=int, default=1,
        help="Retries on empty output (default: 1). Note: greedy decoding — "
             "retrying only helps for transient GPU errors.",
    )
    p.add_argument(
        "--no-flash-attn", action="store_true",
        help="Disable Flash Attention 2.",
    )
    p.add_argument("--force",   action="store_true", help="Overwrite existing .txt files.")
    p.add_argument("--limit",   type=int, default=None, help="Max images to process.")
    p.add_argument("--dry-run", action="store_true",   help="Count images, no inference.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sequential inference loop
# ---------------------------------------------------------------------------


def run_sequential(
    pending: list[tuple[Path, Path]],
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Process one image at a time.  No batching, no prefetch, no vLLM."""
    model, processor = _load_hf_model(
        args.model,
        use_flash_attn=not args.no_flash_attn,
        compile_model=False,
    )
    max_retries = args.max_retries

    n_success = n_failed = 0
    t_start = time.monotonic()

    pbar = tqdm(
        total=len(pending),
        desc="Qaari OCR (sequential)",
        unit="img",
        dynamic_ncols=True,
    )

    for img_path, out_path in pending:
        pil_img = _load_image(img_path)
        if pil_img is None:
            n_failed += 1
            pbar.update(1)
            pbar.set_postfix({"ok": n_success, "fail": n_failed})
            continue

        t0 = time.monotonic()
        text = ""
        last_exc: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                text = _hf_run_single(model, processor, pil_img, args.max_new_tokens)
                if text.strip():
                    break
                last_exc = f"Empty output on attempt {attempt + 1}"
                logger.warning("[%s] %s", img_path.name, last_exc)
            except Exception as exc:
                last_exc = str(exc)
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s",
                    img_path.name, attempt + 1, max_retries + 1, exc,
                )
                torch.cuda.empty_cache()

        latency = time.monotonic() - t0

        if text.strip():
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text.strip(), encoding="utf-8")
                n_success += 1
            except Exception as exc:
                logger.error("Save failed for %s: %s", img_path.name, exc)
                n_failed += 1
        else:
            logger.error(
                "[%s] All %d attempts failed. Last: %s",
                img_path.name, max_retries + 1, last_exc,
            )
            n_failed += 1

        torch.cuda.empty_cache()
        pbar.update(1)
        done = n_success + n_failed
        elapsed = time.monotonic() - t_start
        pbar.set_postfix({
            "ok":      n_success,
            "fail":    n_failed,
            "lat":     f"{latency:.1f}s",
            "img/s":   f"{done / elapsed:.2f}" if elapsed > 0 else "?",
        })

    pbar.close()
    elapsed = time.monotonic() - t_start
    logger.info(
        "Sequential done: %d success | %d failed | %.1fs | %.2f img/s",
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
    print(f"  Qaari OCR Inference (sequential) — {datetime.now(timezone.utc).isoformat()}")
    print(sep)
    print(f"  Model          : {args.model}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Max retries    : {args.max_retries}")
    print(f"  Flash Attn 2   : {not args.no_flash_attn}")
    print(f"  Pending images : {len(pending)}")
    print(sep)
    print("  PROMPT USED:")
    print(f"  {QAARI_PROMPT}")
    print(sep, flush=True)
    print()

    t_start = time.monotonic()
    n_success, n_failed = run_sequential(pending, args)
    elapsed = time.monotonic() - t_start

    print()
    print(sep)
    print(f"  Done   : {n_success} success | {n_failed} failed")
    print(f"  Time   : {elapsed:.1f}s  ({(n_success + n_failed) / elapsed:.2f} img/s)" if elapsed > 0 else "")
    print(f"  Output : {args.output_root}")
    print(sep, flush=True)


if __name__ == "__main__":
    main()
