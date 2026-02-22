#!/usr/bin/env python3
"""Local inference script for Arabic OCR post-correction.

Runs LLM correction directly on the local machine using either:
  - A local GPU via HuggingFace transformers  (config: model.backend = "transformers")
  - A remote API (OpenAI-compatible)           (config: model.backend = "api")

Reads samples via DataLoader, writes per-dataset corrections.jsonl files
to the results directory. Run --mode analyze afterwards to compute metrics.

No HuggingFace dataset sync — output lives locally in results/.

Usage:
    python scripts/run_local.py
    python scripts/run_local.py --datasets KHATT-train KHATT-validation
    python scripts/run_local.py --limit 50
    python scripts/run_local.py --force
    python scripts/run_local.py --config configs/config.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError
from src.core.llm_corrector import get_corrector
from src.core.prompt_builder import PromptBuilder
from pipelines._utils import resolve_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local LLM inference for Arabic OCR correction (GPU or API)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--datasets", nargs="+", default=None, metavar="DATASET",
        help="Dataset keys to process. Defaults to all in config.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per dataset (for testing).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/phase2"),
        help="Output directory. Per-dataset sub-folders are created automatically.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run datasets that already have a corrections.jsonl.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="LLM retry count on empty or failed generation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_completed_ids(corrections_path: Path) -> set[str]:
    """Return sample_ids already written to corrections_path."""
    if not corrections_path.exists():
        return set()
    completed: set[str] = set()
    with open(corrections_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


# ---------------------------------------------------------------------------
# Per-dataset inference
# ---------------------------------------------------------------------------


def run_dataset(
    dataset_key: str,
    loader: DataLoader,
    corrector,
    builder: PromptBuilder,
    results_dir: Path,
    limit: int | None,
    max_retries: int,
    force: bool,
) -> dict:
    """Run inference for one dataset. Returns summary dict."""
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = out_dir / "corrections.jsonl"

    # Resume check
    if corrections_path.exists() and not force:
        completed_ids = load_completed_ids(corrections_path)
        if completed_ids:
            logger.info(
                "[%s] %d samples already done — resuming (use --force to re-run).",
                dataset_key, len(completed_ids),
            )
        else:
            completed_ids = set()
    else:
        completed_ids = set()

    try:
        samples = list(loader.iter_samples(dataset_key, limit=limit))
    except DataError as exc:
        logger.error("[%s] Could not load data: %s", dataset_key, exc)
        return {"dataset": dataset_key, "status": "error", "error": str(exc)}

    pending = [s for s in samples if s.sample_id not in completed_ids]
    logger.info(
        "[%s] %d total | %d already done | %d pending",
        dataset_key, len(samples), len(completed_ids), len(pending),
    )

    if not pending:
        logger.info("[%s] Nothing to do.", dataset_key)
        return {"dataset": dataset_key, "status": "skipped", "n": len(samples)}

    n_success = n_failed = 0

    with open(corrections_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(pending, desc=dataset_key, unit="sample"):
            messages = builder.build_zero_shot(sample.ocr_text)
            result = corrector.correct(
                sample_id=sample.sample_id,
                ocr_text=sample.ocr_text,
                messages=messages,
                max_retries=max_retries,
            )

            record = {
                "sample_id":      sample.sample_id,
                "dataset":        dataset_key,
                "ocr_text":       sample.ocr_text,
                "corrected_text": result.corrected_text,
                "gt_text":        sample.gt_text,
                "model":          corrector.model_name,
                "prompt_version": builder.prompt_version,
                "prompt_tokens":  result.prompt_tokens,
                "output_tokens":  result.output_tokens,
                "latency_s":      result.latency_s,
                "success":        result.success,
                "error":          result.error,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if result.success:
                n_success += 1
            else:
                n_failed += 1

    logger.info(
        "[%s] Done — %d success, %d failed. Output: %s",
        dataset_key, n_success, n_failed, corrections_path,
    )
    return {
        "dataset":    dataset_key,
        "status":     "done",
        "n_success":  n_success,
        "n_failed":   n_failed,
        "output":     str(corrections_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Logging to file as well
    log_handler = logging.FileHandler(args.results_dir / "local_inference.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logging.getLogger().addHandler(log_handler)

    logger.info("=" * 60)
    logger.info("Local inference started: %s", datetime.now(timezone.utc).isoformat())
    logger.info("Backend: %s", config.get("model", {}).get("backend", "transformers"))
    logger.info("Model  : %s", config.get("model", {}).get("name", "?"))
    logger.info("=" * 60)

    active_datasets = resolve_datasets(config, args.datasets)
    loader = DataLoader(config)

    # Load model once (expensive) before the dataset loop
    logger.info("Loading model...")
    corrector = get_corrector(config)
    builder = PromptBuilder()
    logger.info("Model ready.")

    summaries = []
    for ds_key in active_datasets:
        summary = run_dataset(
            dataset_key=ds_key,
            loader=loader,
            corrector=corrector,
            builder=builder,
            results_dir=args.results_dir,
            limit=args.limit,
            max_retries=args.max_retries,
            force=args.force,
        )
        summaries.append(summary)

    # Print final table
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for s in summaries:
        status = s["status"]
        if status == "done":
            logger.info(
                "  %-40s  success=%-5d  failed=%d",
                s["dataset"], s.get("n_success", 0), s.get("n_failed", 0),
            )
        else:
            logger.info("  %-40s  %s", s["dataset"], status.upper())

    logger.info("")
    logger.info("Next step: python pipelines/run_phase2.py --mode analyze")


if __name__ == "__main__":
    main()
