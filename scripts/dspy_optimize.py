#!/usr/bin/env python3
"""Phase 7: DSPy prompt optimization + inference (run on Kaggle).

Loads the LLM, runs DSPy BootstrapFewShot to discover optimal few-shot
demonstrations for Arabic OCR correction, then runs inference on the full
validation set with the optimized prompt.

Prerequisites
-------------
    pip install dspy-ai>=2.4
    # Model must be available (downloaded or in /kaggle/input/)

Usage (Kaggle)
--------------
    python scripts/dspy_optimize.py \\
        --trainset  results/phase7/dspy_trainset.jsonl \\
        --devset    results/phase7/dspy_devset.jsonl \\
        --input     results/phase7/inference_input.jsonl \\
        --output    results/phase7/corrections.jsonl

    # With custom model path (e.g., from Kaggle input)
    python scripts/dspy_optimize.py \\
        --model /kaggle/input/qwen3-4b/Qwen3-4B-Instruct-2507 \\
        --trainset  results/phase7/dspy_trainset.jsonl \\
        --devset    results/phase7/dspy_devset.jsonl \\
        --input     results/phase7/inference_input.jsonl \\
        --output    results/phase7/corrections.jsonl

    # Quick test (5 optimization samples, 10 inference samples)
    python scripts/dspy_optimize.py \\
        --trainset  results/phase7/dspy_trainset.jsonl \\
        --devset    results/phase7/dspy_devset.jsonl \\
        --input     results/phase7/inference_input.jsonl \\
        --output    results/phase7/corrections.jsonl \\
        --limit 10

Output
------
    results/phase7/corrections.jsonl     — standard corrections file
    results/phase7/dspy_compiled.json    — saved DSPy compiled program
    results/phase7/optimized_prompt.txt  — extracted optimized prompt (for reference)
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Arabic regex for validating model output
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: DSPy prompt optimization + inference (Kaggle)"
    )
    parser.add_argument(
        "--trainset", type=Path, required=True,
        help="Path to dspy_trainset.jsonl (OCR + GT pairs for optimization)",
    )
    parser.add_argument(
        "--devset", type=Path, required=True,
        help="Path to dspy_devset.jsonl (OCR + GT pairs for metric evaluation)",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to inference_input.jsonl (full val set for post-optimization inference)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write corrections.jsonl",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name or local path (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/config.yaml"),
        help="Config YAML (for phase7 settings).",
    )
    parser.add_argument(
        "--max-bootstrapped-demos", type=int, default=None,
        help="Override config max_bootstrapped_demos.",
    )
    parser.add_argument(
        "--max-labeled-demos", type=int, default=None,
        help="Override config max_labeled_demos.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max inference samples (for quick testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if output exists.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature (default: 0.1).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Max new tokens per generation (default: 1024).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DSPy LM Adapter
# ---------------------------------------------------------------------------


class LocalTransformersLM:
    """DSPy-compatible LM adapter for a locally loaded transformers model.

    Wraps a HuggingFace model+tokenizer pair so DSPy's Predict module can
    use it for optimization and inference. The adapter handles:
    - OpenAI-format messages (from DSPy's chat templates)
    - Raw prompt strings (from DSPy's text templates)
    - Qwen3's enable_thinking=False to suppress scratchpad tokens

    Returns list[str] completions, which is the most portable DSPy format.
    """

    def __init__(self, model, tokenizer, temperature: float = 0.1,
                 max_tokens: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        # DSPy internals may read these attributes
        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": 1,
        }
        self.history: list[dict] = []
        self.model_type = "chat"

    def __call__(self, prompt=None, messages=None, n=1, **kwargs):
        """Generate completions. Called by DSPy's Predict module.

        Args:
            prompt: Raw prompt string (DSPy text template mode).
            messages: OpenAI-format messages list (DSPy chat template mode).
            n: Number of completions to return.
            **kwargs: Generation overrides (temperature, max_tokens).

        Returns:
            List of completion strings.
        """
        import torch

        if messages:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                # Fallback: concatenate message contents
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(messages)}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
        elif prompt:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            return [""]

        inputs = self.tokenizer(formatted, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_len = input_ids.shape[1]

        # Guard against OOM
        if prompt_len > 8192:
            logger.warning("Input %d tokens -- truncating to 8192.", prompt_len)
            input_ids = input_ids[:, -8192:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -8192:]
            prompt_len = input_ids.shape[1]

        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_tokens", self.max_tokens)

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tok,
            temperature=max(temp, 0.01),  # avoid division by zero
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Log for debugging
        self.history.append({
            "prompt_tokens": prompt_len,
            "output_tokens": len(new_tokens),
            "output_preview": text[:100],
        })

        return [text] * n

    def inspect_history(self, n: int = 1):
        """Return recent generation history (DSPy debugging helper)."""
        return self.history[-n:] if self.history else []


# ---------------------------------------------------------------------------
# Model loading (mirrors TransformersCorrector._load_model)
# ---------------------------------------------------------------------------


def load_model(model_name: str, temperature: float, max_tokens: int):
    """Load model + tokenizer and return (model, tokenizer, lm_adapter).

    Returns:
        Tuple of (model, tokenizer, LocalTransformersLM adapter).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded. Parameters: %.2fB", num_params)

    lm = LocalTransformersLM(model, tokenizer, temperature, max_tokens)
    return model, tokenizer, lm


# ---------------------------------------------------------------------------
# DSPy setup
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dspy_datasets(trainset_path: Path, devset_path: Path):
    """Load train/dev JSONL files and convert to DSPy Example lists.

    Returns:
        Tuple of (trainset, devset) as lists of dspy.Example.
    """
    import dspy

    train_records = load_jsonl(trainset_path)
    dev_records = load_jsonl(devset_path)

    trainset = []
    for rec in train_records:
        ex = dspy.Example(
            ocr_text=rec["ocr_text"],
            corrected_text=rec["gt_text"],
        ).with_inputs("ocr_text")
        trainset.append(ex)

    devset = []
    for rec in dev_records:
        ex = dspy.Example(
            ocr_text=rec["ocr_text"],
            corrected_text=rec["gt_text"],
        ).with_inputs("ocr_text")
        devset.append(ex)

    logger.info("DSPy datasets: %d train, %d dev", len(trainset), len(devset))
    return trainset, devset


def cer_metric(example, pred, trace=None):
    """CER-based metric for DSPy optimization. Returns float in [0, 1].

    DSPy maximizes metrics, so we return 1 - CER (higher is better).
    Uses diacritics-stripped CER for consistency with evaluation.
    """
    from src.analysis.metrics import calculate_cer

    pred_text = getattr(pred, "corrected_text", str(pred))
    gt_text = example.corrected_text

    if not gt_text or not pred_text:
        return 0.0

    cer = calculate_cer(gt_text, pred_text, strip_diacritics=True)
    return max(0.0, 1.0 - min(cer, 1.0))


def run_optimization(lm, trainset, devset, config: dict):
    """Run DSPy BootstrapFewShot optimization.

    Args:
        lm: DSPy-compatible LM adapter.
        trainset: List of dspy.Example for training.
        devset: List of dspy.Example for evaluation.
        config: Phase 7 config dict.

    Returns:
        Compiled dspy.Module with optimized demonstrations.
    """
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    dspy.configure(lm=lm)

    # Define the OCR correction signature and module
    class ArabicOCRCorrection(dspy.Signature):
        """Fix OCR errors in Arabic text. Output only the corrected Arabic text with no other content."""

        ocr_text: str = dspy.InputField(
            desc="Noisy Arabic text produced by an OCR system"
        )
        corrected_text: str = dspy.OutputField(
            desc="Corrected Arabic text with OCR errors fixed"
        )

    class OCRCorrector(dspy.Module):
        def __init__(self):
            self.correct = dspy.Predict(ArabicOCRCorrection)

        def forward(self, ocr_text: str):
            return self.correct(ocr_text=ocr_text)

    # Create the student module
    student = OCRCorrector()

    # Configure optimizer from config
    max_bootstrapped = config.get("max_bootstrapped_demos", 3)
    max_labeled = config.get("max_labeled_demos", 4)
    max_rounds = config.get("max_rounds", 1)

    logger.info("=" * 60)
    logger.info("DSPy BootstrapFewShot Optimization")
    logger.info("  max_bootstrapped_demos: %d", max_bootstrapped)
    logger.info("  max_labeled_demos: %d", max_labeled)
    logger.info("  max_rounds: %d", max_rounds)
    logger.info("  trainset: %d examples", len(trainset))
    logger.info("  devset: %d examples", len(devset))
    logger.info("=" * 60)

    optimizer = BootstrapFewShot(
        metric=cer_metric,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=max_rounds,
    )

    t0 = time.monotonic()
    compiled = optimizer.compile(student, trainset=trainset)
    elapsed = time.monotonic() - t0

    logger.info("Optimization complete in %.1f seconds.", elapsed)

    # Evaluate on devset
    logger.info("Evaluating compiled program on devset...")
    n_correct = 0
    total_score = 0.0
    for ex in devset:
        try:
            pred = compiled(ocr_text=ex.ocr_text)
            score = cer_metric(ex, pred)
            total_score += score
            if score > 0.9:
                n_correct += 1
        except Exception as e:
            logger.warning("Dev eval failed for one sample: %s", e)

    avg_score = total_score / len(devset) if devset else 0.0
    logger.info(
        "Dev evaluation: avg_score=%.4f  high_quality=%d/%d (>0.9)",
        avg_score, n_correct, len(devset),
    )

    return compiled


def extract_optimized_prompt(compiled_module, output_dir: Path) -> str:
    """Extract the optimized prompt from the compiled DSPy program.

    Saves both the full compiled state (JSON) and a human-readable
    prompt text file for inspection.

    Returns:
        String representation of the optimized prompt.
    """
    # Save compiled program
    compiled_path = output_dir / "dspy_compiled.json"
    try:
        compiled_module.save(str(compiled_path))
        logger.info("Compiled program saved -> %s", compiled_path)
    except Exception as e:
        logger.warning("Could not save compiled program: %s", e)

    # Extract demonstrations and instruction for reference
    prompt_parts = []
    try:
        predictor = compiled_module.correct
        # Get instruction
        if hasattr(predictor, "extended_signature"):
            instructions = getattr(predictor.extended_signature, "instructions", "")
            if instructions:
                prompt_parts.append(f"INSTRUCTION:\n{instructions}")

        # Get demonstrations
        demos = getattr(predictor, "demos", [])
        if demos:
            prompt_parts.append(f"\nDEMONSTRATIONS ({len(demos)} examples):")
            for i, demo in enumerate(demos):
                ocr = getattr(demo, "ocr_text", "?")
                corrected = getattr(demo, "corrected_text", "?")
                prompt_parts.append(f"\n  Example {i + 1}:")
                prompt_parts.append(f"    OCR:       {ocr[:200]}")
                prompt_parts.append(f"    Corrected: {corrected[:200]}")
    except Exception as e:
        prompt_parts.append(f"(Could not extract details: {e})")

    prompt_text = "\n".join(prompt_parts) if prompt_parts else "(No prompt extracted)"

    # Save human-readable version
    prompt_path = output_dir / "optimized_prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    logger.info("Optimized prompt reference -> %s", prompt_path)

    return prompt_text


# ---------------------------------------------------------------------------
# Inference with compiled program
# ---------------------------------------------------------------------------


def run_inference(compiled_module, input_path: Path, output_path: Path,
                  limit: int = None, force: bool = False) -> None:
    """Run inference on full validation set using the compiled DSPy program.

    Produces corrections.jsonl in the same format as scripts/infer.py.
    """
    from tqdm import tqdm

    # Read completed IDs for resume
    completed: set[str] = set()
    if output_path.exists() and not force:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed.add(rec.get("sample_id", ""))
        if completed:
            logger.info("Resuming: %d samples already done.", len(completed))
    elif force and output_path.exists():
        output_path.write_text("", encoding="utf-8")

    # Load input records
    records = load_jsonl(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records if r["sample_id"] not in completed]
    logger.info(
        "Inference: total=%d  done=%d  pending=%d",
        len(records), len(records) - len(pending), len(pending),
    )

    if not pending:
        logger.info("Nothing to do.")
        return

    n_success = n_failed = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for record in tqdm(pending, desc="DSPy Inference", unit="sample"):
            t0 = time.monotonic()
            try:
                pred = compiled_module(ocr_text=record["ocr_text"])
                corrected = getattr(pred, "corrected_text", "")

                # Validate output has Arabic text
                if not corrected or not _ARABIC_RE.search(corrected):
                    corrected = record["ocr_text"]
                    success = False
                    error = "No Arabic text in DSPy output"
                else:
                    corrected = corrected.strip()
                    success = True
                    error = None

            except Exception as e:
                corrected = record["ocr_text"]
                success = False
                error = str(e)

            latency = round(time.monotonic() - t0, 3)

            out_rec = {
                "sample_id": record["sample_id"],
                "dataset": record.get("dataset", "unknown"),
                "ocr_text": record["ocr_text"],
                "gt_text": record.get("gt_text", ""),
                "corrected_text": corrected,
                "prompt_type": "dspy_optimized",
                "prompt_version": "p7v1",
                "success": success,
                "latency_s": latency,
            }
            if error:
                out_rec["error"] = error

            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if success:
                n_success += 1
            else:
                n_failed += 1

    logger.info(
        "Inference complete: %d success, %d failed out of %d.",
        n_success, n_failed, len(pending),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load config
    config = {}
    if args.config.exists():
        import yaml
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    phase7_cfg = config.get("phase7", {})

    # Apply CLI overrides
    if args.max_bootstrapped_demos is not None:
        phase7_cfg["max_bootstrapped_demos"] = args.max_bootstrapped_demos
    if args.max_labeled_demos is not None:
        phase7_cfg["max_labeled_demos"] = args.max_labeled_demos

    # Validate inputs exist
    for path_name, path in [
        ("trainset", args.trainset),
        ("devset", args.devset),
        ("input", args.input),
    ]:
        if not path.exists():
            logger.error("%s not found: %s", path_name, path)
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 7: DSPy Prompt Optimization + Inference")
    logger.info("  Model:    %s", args.model)
    logger.info("  Trainset: %s", args.trainset)
    logger.info("  Devset:   %s", args.devset)
    logger.info("  Input:    %s", args.input)
    logger.info("  Output:   %s", args.output)
    logger.info("=" * 60)

    # Step 1: Load model
    logger.info("Step 1/4: Loading model...")
    model, tokenizer, lm = load_model(
        args.model, args.temperature, args.max_tokens,
    )

    # Step 2: Build DSPy datasets
    logger.info("Step 2/4: Building DSPy datasets...")
    try:
        import dspy
    except ImportError:
        logger.error(
            "dspy is not installed. Install with: pip install dspy-ai>=2.4"
        )
        sys.exit(1)

    trainset, devset = build_dspy_datasets(args.trainset, args.devset)

    # Step 3: Run optimization
    logger.info("Step 3/4: Running DSPy optimization...")
    compiled = run_optimization(lm, trainset, devset, phase7_cfg)

    # Extract and save the optimized prompt for reference
    output_dir = args.output.parent
    extract_optimized_prompt(compiled, output_dir)

    # Step 4: Run inference with compiled program
    logger.info("Step 4/4: Running inference with optimized prompt...")
    run_inference(compiled, args.input, args.output, args.limit, args.force)

    logger.info("=" * 60)
    logger.info("Phase 7 complete.")
    logger.info("  Corrections: %s", args.output)
    logger.info("  Compiled program: %s", output_dir / "dspy_compiled.json")
    logger.info("")
    logger.info("Next step: pull corrections.jsonl and run:")
    logger.info("  python pipelines/run_phase7.py --mode analyze")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
