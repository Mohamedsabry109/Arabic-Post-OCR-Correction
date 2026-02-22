#!/usr/bin/env python3
"""Self-contained inference script for Kaggle kernels.

Upload ONLY this file to Kaggle — no other project files needed.

Features:
  - Zero-shot Arabic OCR correction via Qwen3 (or any HuggingFace model)
  - HuggingFace dataset sync: pulls existing progress on startup, pushes
    every --sync-every samples — survives Kaggle session timeouts
  - Resume-safe: already-completed sample_ids are skipped automatically

Kaggle setup (notebook cells):
    # Cell 1 — install deps
    !pip install transformers accelerate huggingface_hub -q

    # Cell 2 — run (replace YOUR_HF_REPO and YOUR_HF_TOKEN)
    !python /kaggle/input/your-dataset/kaggle_inference.py \\
        --input  /kaggle/input/your-dataset/inference_input.jsonl \\
        --output /kaggle/working/corrections.jsonl \\
        --model  Qwen/Qwen3-4B-Instruct-2507 \\
        --hf-repo  YourUsername/arabic-ocr-corrections \\
        --hf-token hf_xxxxxxxxxxxx

    # Or store the token as a Kaggle secret (safer):
    # Settings > Secrets > add HF_TOKEN
    # Then omit --hf-token; the script reads HF_TOKEN automatically.
"""

# ============================================================
# Stdlib imports only — no project dependencies
# ============================================================
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Prompt — versioned so output metadata stays traceable
# ============================================================

PROMPT_VERSION = "v1"

ZERO_SHOT_SYSTEM = (
    "أنت مصحح نصوص عربية متخصص. "
    "مهمتك تصحيح أخطاء التعرف الضوئي (OCR) في النص العربي. "
    "أعد النص المصحح فقط بدون أي شرح أو تعليق إضافي."
)


def build_messages(ocr_text: str) -> list[dict]:
    return [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {"role": "user",   "content": ocr_text},
    ]


# ============================================================
# CorrectionResult
# ============================================================

@dataclass
class CorrectionResult:
    sample_id:      str
    ocr_text:       str
    corrected_text: str
    prompt_tokens:  int
    output_tokens:  int
    latency_s:      float
    success:        bool
    error:          Optional[str] = None


# ============================================================
# Model (TransformersCorrector inlined)
# ============================================================

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_MAX_INPUT_TOKENS = 1024


class Corrector:
    """Thin wrapper around a HuggingFace causal LM."""

    def __init__(self, model_id: str, quantize_4bit: bool, temperature: float, max_tokens: int) -> None:
        self._model_id    = model_id
        self._temperature = temperature
        self._max_tokens  = max_tokens

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Run: pip install transformers torch accelerate")

        logger.info("Loading tokenizer: %s", model_id)
        self._tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if quantize_4bit:
            logger.info("Loading model (4-bit quantization)...")
            try:
                import torch as _t
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=_t.float16,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=bnb,
                    device_map="auto", trust_remote_code=True,
                )
            except ImportError:
                raise ImportError("Run: pip install bitsandbytes")
        else:
            import torch as _t
            logger.info("Loading model: %s", model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=_t.float16,
                device_map="auto", trust_remote_code=True,
            )

        self._model.eval()
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e9
        logger.info("Model ready (%.2fB params).", n_params)

    @property
    def model_name(self) -> str:
        return self._model_id

    def correct(self, sample_id: str, ocr_text: str, messages: list[dict], max_retries: int = 2) -> CorrectionResult:
        t0 = time.monotonic()
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                raw, prompt_tokens, output_tokens = self._generate(messages)
                corrected = self._clean(raw, ocr_text)

                if corrected == ocr_text and not raw.strip():
                    last_error = f"empty generation attempt {attempt + 1}"
                    logger.warning("[%s] %s", sample_id, last_error)
                    continue

                return CorrectionResult(
                    sample_id=sample_id, ocr_text=ocr_text, corrected_text=corrected,
                    prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                    latency_s=round(time.monotonic() - t0, 3), success=True,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning("[%s] attempt %d failed: %s", sample_id, attempt + 1, exc)

        logger.error("[%s] all attempts failed — falling back to OCR text.", sample_id)
        return CorrectionResult(
            sample_id=sample_id, ocr_text=ocr_text, corrected_text=ocr_text,
            prompt_tokens=0, output_tokens=0,
            latency_s=round(time.monotonic() - t0, 3), success=False, error=last_error,
        )

    def _generate(self, messages: list[dict]) -> tuple[str, int, int]:
        import torch
        formatted = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self._tok(formatted, return_tensors="pt")
        device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        prompt_len = input_ids.shape[1]
        if prompt_len > _MAX_INPUT_TOKENS:
            logger.warning("Input %d tokens > %d — truncating.", prompt_len, _MAX_INPUT_TOKENS)
            input_ids = input_ids[:, -_MAX_INPUT_TOKENS:]
            if attn_mask is not None:
                attn_mask = attn_mask[:, -_MAX_INPUT_TOKENS:]
            prompt_len = input_ids.shape[1]

        gen_kwargs: dict = dict(
            input_ids=input_ids, max_new_tokens=self._max_tokens,
            do_sample=True, temperature=self._temperature,
            pad_token_id=self._tok.eos_token_id,
        )
        if attn_mask is not None:
            gen_kwargs["attention_mask"] = attn_mask

        with torch.no_grad():
            out_ids = self._model.generate(**gen_kwargs)

        new_ids = out_ids[:, prompt_len:]
        decoded = self._tok.decode(new_ids[0], skip_special_tokens=True)
        return decoded, prompt_len, new_ids.shape[1]

    def _clean(self, raw: str, fallback: str) -> str:
        cleaned = raw.strip()
        if not cleaned:
            logger.warning("Empty model output — using OCR text.")
            return fallback
        if not _ARABIC_RE.search(cleaned):
            logger.warning("No Arabic in output (%r…) — using OCR text.", cleaned[:40])
            return fallback
        return cleaned


# ============================================================
# HuggingFace sync
# ============================================================

_HF_FILENAME = "corrections.jsonl"


def _hf_pull(output_path: Path, repo: str, token: Optional[str]) -> None:
    """Download existing corrections from HF and merge with local file."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        logger.warning("huggingface_hub not installed — skipping pull.")
        return

    try:
        remote_path = hf_hub_download(
            repo_id=repo, filename=_HF_FILENAME,
            repo_type="dataset", token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        logger.info("No existing %s on HF — starting fresh.", _HF_FILENAME)
        return
    except Exception as exc:
        logger.warning("HF pull failed: %s", exc)
        return

    def _read_jsonl(path: str) -> dict[str, str]:
        """Read JSONL → {sample_id: raw_line}."""
        records: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    sid = r.get("sample_id", "")
                    if sid:
                        records[sid] = line
                except json.JSONDecodeError:
                    pass
        return records

    remote = _read_jsonl(remote_path)
    local: dict[str, str] = {}
    if output_path.exists():
        local = _read_jsonl(str(output_path))

    merged = {**remote, **local}   # local wins on conflict (more recent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in merged.values():
            f.write(line + "\n")

    logger.info(
        "HF pull: %d remote + %d new-local = %d total.",
        len(remote), len(local) - len(set(local) & set(remote)), len(merged),
    )


def _hf_push(output_path: Path, repo: str, token: Optional[str]) -> None:
    """Upload corrections.jsonl to HF dataset. Never raises."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return
    try:
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, token=token)
        n = sum(1 for _ in open(output_path, encoding="utf-8") if _.strip())
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=_HF_FILENAME,
            repo_id=repo, repo_type="dataset", token=token,
            commit_message=f"sync: {n} records",
        )
        logger.info("HF push OK -> %s (%d records)", repo, n)
    except Exception as exc:
        logger.warning("HF push failed (progress safe locally): %s", exc)


# ============================================================
# Resume helpers
# ============================================================

def _load_completed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    logger.info("Resume: %d samples already done.", len(ids))
    return ids


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle inference — Arabic OCR correction")
    p.add_argument("--input",  type=Path, default=Path("inference_input.jsonl"))
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/corrections.jsonl"))
    p.add_argument("--model",  type=str,  default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--limit",  type=int,  default=None, help="Stop after N samples.")
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--max-retries",   type=int, default=2)
    p.add_argument("--temperature",   type=float, default=0.1)
    p.add_argument("--max-tokens",    type=int,   default=1024)
    p.add_argument("--hf-repo",    type=str, default=None,
                   help="HF dataset repo for sync (e.g. user/arabic-ocr-corrections).")
    p.add_argument("--hf-token",   type=str, default=None,
                   help="HF token. Falls back to HF_TOKEN env var.")
    p.add_argument("--sync-every", type=int, default=100,
                   help="Push to HF every N samples (default: 100).")
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    hf_repo  = args.hf_repo
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if not hf_repo:
        logger.warning(
            "No --hf-repo provided. Output saved to %s only.\n"
            "If the Kaggle session times out, progress may be lost.\n"
            "Provide --hf-repo to enable automatic sync.", args.output,
        )

    # Pull existing progress from HF before anything else
    if hf_repo:
        logger.info("Pulling existing progress from HF: %s", hf_repo)
        _hf_pull(args.output, hf_repo, hf_token)

    # Load input
    if not args.input.exists():
        logger.error(
            "Input not found: %s\n"
            "Upload inference_input.jsonl (from: python pipelines/run_phase2.py --mode export).",
            args.input,
        )
        sys.exit(1)

    records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]
    logger.info("Total samples in input: %d", len(records))

    # Skip already-completed
    completed = _load_completed(args.output)
    pending   = [r for r in records if r["sample_id"] not in completed]
    logger.info("Pending: %d | Done: %d", len(pending), len(completed))

    if not pending:
        logger.info("All samples already complete.")
        _print_summary(args.output)
        return

    # Load model
    corrector = Corrector(
        model_id=args.model,
        quantize_4bit=args.quantize_4bit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Inference loop
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_success = n_failed = n_since_sync = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(pending, desc="Correcting", unit="sample")
    except ImportError:
        iterator = pending

    with open(args.output, "a", encoding="utf-8") as out_f:
        for record in iterator:
            messages = build_messages(record["ocr_text"])
            result   = corrector.correct(
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
                "prompt_version": PROMPT_VERSION,
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

            n_since_sync += 1
            if hf_repo and n_since_sync >= args.sync_every:
                out_f.flush()
                _hf_push(args.output, hf_repo, hf_token)
                n_since_sync = 0

    # Final push
    if hf_repo:
        _hf_push(args.output, hf_repo, hf_token)

    logger.info("=" * 55)
    logger.info("Done. Success: %d | Failed: %d", n_success, n_failed)
    logger.info("Output: %s", args.output)
    if hf_repo:
        logger.info("Synced to HF: %s/%s", hf_repo, _HF_FILENAME)
    logger.info("=" * 55)
    logger.info(
        "Next: download corrections.jsonl and run:\n"
        "  python pipelines/run_phase2.py --mode analyze"
    )

    _print_summary(args.output)


def _print_summary(path: Path) -> None:
    if not path.exists():
        return
    total = success = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                if json.loads(line).get("success"):
                    success += 1
            except json.JSONDecodeError:
                pass
    print(f"\nOutput: {path}")
    print(f"  Total   : {total}")
    print(f"  Success : {success}")
    print(f"  Failed  : {total - success}")


if __name__ == "__main__":
    main()
