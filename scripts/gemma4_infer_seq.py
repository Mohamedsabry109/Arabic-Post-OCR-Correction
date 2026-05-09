#!/usr/bin/env python3
"""
Sequential Gemma 4 OCR inference for Thunder (or any single-GPU machine).

Usage:
  python scripts/gemma4_infer_seq.py \
      --image-root  data/validation-input-data \
      --output-root data/ocr-results/gemma4-results \
      --no-flash-attn

  # Specific sub-directories only
  python scripts/gemma4_infer_seq.py \
      --image-root  data/validation-input-data/khatt \
      --output-root data/ocr-results/gemma4-results/khatt \
      --no-flash-attn

  # Quick smoke test (10 images)
  python scripts/gemma4_infer_seq.py \
      --image-root  data/validation-input-data \
      --output-root data/ocr-results/gemma4-results \
      --no-flash-attn --limit 10

Output mirrors the input tree:
  {output-root}/{relative/path/to/image}.txt
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

OCR_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Do not hallucinate."
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LOG_INTERVAL = 10
CLEANUP_INTERVAL = 50


class Gemma4OCR:
    def __init__(self, model_path: str, max_new_tokens: int, device: str, flash_attn: bool):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        attn_impl = "flash_attention_2" if flash_attn else "eager"
        print(f"Loading model: {model_path}")
        print(f"  device={device}  attn={attn_impl}  max_new_tokens={max_new_tokens}")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=device,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = device

    def __call__(self, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": OCR_PROMPT},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        output_ids = generated_ids[0][input_len:]
        text = self.processor.decode(output_ids, skip_special_tokens=True)

        del inputs, generated_ids, output_ids
        return text


def scan_images(image_root: Path, output_root: Path, limit: int) -> List[Tuple[Path, Path]]:
    """Return (src_image, dst_txt) pairs for images not yet processed."""
    tasks = []
    for src in sorted(image_root.rglob("*")):
        if src.is_file() and src.suffix.lower() in IMAGE_EXTENSIONS:
            rel = src.relative_to(image_root)
            dst = output_root / rel.with_suffix(".txt")
            tasks.append((src, dst))

    total = len(tasks)
    pending = [(src, dst) for src, dst in tasks if not dst.exists()]
    skipped = total - len(pending)

    print(f"Images found : {total}")
    print(f"Already done : {skipped}")
    print(f"To process   : {len(pending)}")

    if limit > 0:
        pending = pending[:limit]
        print(f"Limit applied: running {len(pending)}")

    return pending


def run(args: argparse.Namespace):
    image_root  = Path(args.image_root)
    output_root = Path(args.output_root)

    if not image_root.exists():
        raise SystemExit(f"ERROR: --image-root not found: {image_root}")

    print("=" * 60)
    print("GEMMA 4 OCR  —  Sequential Inference")
    print("=" * 60)
    print(f"Image root  : {image_root}")
    print(f"Output root : {output_root}")
    print(f"Model       : {args.model}")
    print(f"Flash attn  : {not args.no_flash_attn}")
    print(f"Device      : {args.device}")
    print(f"Max tokens  : {args.max_new_tokens}")
    print("=" * 60)

    tasks = scan_images(image_root, output_root, args.limit)
    if not tasks:
        print("Nothing to do.")
        return

    # Pre-fetch weights to CPU cache before moving to GPU
    print("\nPre-fetching model weights...")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        AutoProcessor.from_pretrained(args.model)
        tmp = AutoModelForImageTextToText.from_pretrained(
            args.model, device_map="cpu", low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        del tmp
        gc.collect()
        print("Weights cached.")
    except Exception as e:
        print(f"Pre-fetch failed (continuing anyway): {e}")

    model = Gemma4OCR(
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        flash_attn=not args.no_flash_attn,
    )

    total      = len(tasks)
    processed  = 0
    failed     = 0
    since_gc   = 0
    start_time = time.time()

    print(f"\nStarting inference on {total} images...\n")

    for idx, (src, dst) in enumerate(tasks, 1):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            image = Image.open(src).convert("RGB")
            text = model(image)
            dst.write_text(text, encoding="utf-8")
            processed += 1
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM: {src.name} — skipping", flush=True)
            failed += 1
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR [{src.name}]: {type(e).__name__}: {e}", flush=True)
            failed += 1

        since_gc += 1
        if since_gc >= CLEANUP_INTERVAL:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            since_gc = 0

        if idx % LOG_INTERVAL == 0 or idx == total:
            elapsed = time.time() - start_time
            speed   = processed / elapsed if elapsed > 0 else 0
            eta     = (total - idx) / speed if speed > 0 else 0
            mem     = ""
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                mem = f"  mem={alloc:.1f}GB"
            print(
                f"[{idx}/{total}] done={processed} fail={failed} "
                f"speed={speed:.2f}img/s  ETA={eta/60:.1f}m{mem}",
                flush=True,
            )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"DONE  processed={processed}  failed={failed}  "
          f"time={elapsed/60:.1f}m  avg={elapsed/max(processed,1):.1f}s/img")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Gemma 4 OCR inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-root",    required=True,
                        help="Root directory to scan recursively for images")
    parser.add_argument("--output-root",   required=True,
                        help="Root directory for .txt output (mirrors input tree)")
    parser.add_argument("--model",         default="google/gemma-3-4b-it",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable flash attention (use eager — safer on older GPUs)")
    parser.add_argument("--device",        default="cuda:0",
                        help="Torch device string")
    parser.add_argument("--max-new-tokens", type=int, default=2000,
                        help="Maximum tokens to generate per image")
    parser.add_argument("--limit",         type=int, default=0,
                        help="Process at most N images (0 = no limit, useful for testing)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
