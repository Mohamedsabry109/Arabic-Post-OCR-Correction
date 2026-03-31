#!/usr/bin/env python3
"""Phase 7: DSPy Automated Prompt Optimization.

Uses DSPy BootstrapFewShot to automatically discover optimal few-shot
demonstrations and instructions for Arabic OCR post-correction.

Three-stage pipeline:

  --mode export   -> Sample train/dev sets for DSPy optimization + full val set
                     -> results/phase7/dspy_trainset.jsonl
                        results/phase7/dspy_devset.jsonl
                        results/phase7/inference_input.jsonl

  (REMOTE)        -> python scripts/dspy_optimize.py (on Kaggle)
                     -> results/phase7/corrections.jsonl
                        results/phase7/dspy_compiled.json

  --mode analyze  -> Load corrections.jsonl, compute metrics and comparison
                     -> results/phase7/{dataset}/metrics.json
                        results/phase7/metrics.json
                        results/phase7/comparison.json
                        results/phase7/paper_tables.md

Typical workflow
----------------
1. Prerequisite: Phase 2 corrections must exist (used as baseline for comparison).

2. LOCAL:  python pipelines/run_phase7.py --mode export
   (samples training data for DSPy, exports full val inference set)

3. REMOTE: python scripts/dspy_optimize.py \\
               --trainset  results/phase7/dspy_trainset.jsonl \\
               --devset    results/phase7/dspy_devset.jsonl \\
               --input     results/phase7/inference_input.jsonl \\
               --output    results/phase7/corrections.jsonl
   (see notebooks/kaggle_setup.ipynb)

4. LOCAL:  python pipelines/run_phase7.py --mode analyze

Usage
-----
    python pipelines/run_phase7.py --mode export
    python pipelines/run_phase7.py --mode export --limit 50
    python pipelines/run_phase7.py --mode analyze
    python pipelines/run_phase7.py --mode analyze --datasets KHATT-validation
"""

import argparse
import json
import logging
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError
from src.analysis.metrics import MetricResult, calculate_metrics, calculate_metrics_dual
from src.analysis.report_formatter import write_corrections_report
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from pipelines._utils import (
    resolve_datasets, load_sample_list, compute_group_aggregates,
    split_runaway_samples, DEFAULT_RUNAWAY_RATIO_THRESHOLD,
    load_phase2_full_metrics, pick_phase2_variant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-type helpers (shared with run_phase4d.py)
# ---------------------------------------------------------------------------


def _is_train_split(dataset_key: str) -> bool:
    key_lower = dataset_key.lower()
    return key_lower.endswith("-train") or key_lower.endswith("train")


def _is_val_split(dataset_key: str) -> bool:
    key_lower = dataset_key.lower()
    return (
        key_lower.endswith("-val")
        or key_lower.endswith("-validation")
        or key_lower.endswith("validation")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: DSPy Automated Prompt Optimization"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["export", "analyze"],
        help=(
            "export  -> sample train/dev sets + export full val inference_input.jsonl; "
            "analyze -> load corrections.jsonl and compute metrics"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Subset of dataset keys to process. Defaults to all from config.",
    )
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=None,
        dest="sample_list",
        help="Path to test_samples.json to filter samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset (for quick testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process even if output already exists.",
    )
    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        default=False,
        help="Skip error_changes.json computation in analyze mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config YAML file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(
                open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
            )
        ],
    )


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _make_meta(config: dict, mode: str, dataset: str = "") -> dict:
    return {
        "phase": "phase7",
        "mode": mode,
        "dataset": dataset,
        "model": config.get("model", {}).get("name", ""),
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------


def run_export(args: argparse.Namespace, config: dict) -> None:
    """Export DSPy train/dev sets and full validation inference_input.jsonl."""
    results_dir = Path(config.get("output", {}).get("results_dir", "results")) / "phase7"
    results_dir.mkdir(parents=True, exist_ok=True)

    phase7_cfg = config.get("phase7", {})
    n_train = phase7_cfg.get("n_train", 50)
    n_dev = phase7_cfg.get("n_dev", 20)
    seed = phase7_cfg.get("seed", 42)
    source_datasets = phase7_cfg.get("source_datasets", None)

    all_datasets = resolve_datasets(config, args.datasets)
    loader = DataLoader(config)

    # ---- Collect training samples for DSPy optimization ----
    train_datasets = [k for k in all_datasets if _is_train_split(k)]
    val_datasets = [k for k in all_datasets if _is_val_split(k)]

    # Override source if config specifies particular datasets
    if source_datasets:
        train_datasets = [k for k in source_datasets if k in all_datasets or True]

    logger.info("=" * 60)
    logger.info("Phase 7 EXPORT")
    logger.info("  Train datasets for DSPy: %s", train_datasets)
    logger.info("  Val datasets for inference: %s", val_datasets)
    logger.info("  n_train=%d  n_dev=%d  seed=%d", n_train, n_dev, seed)
    logger.info("=" * 60)

    # Load all training samples
    all_train_samples = []
    for ds_key in train_datasets:
        try:
            samples = list(loader.iter_samples(ds_key))
            for s in samples:
                all_train_samples.append({
                    "sample_id": s.sample_id,
                    "dataset": ds_key,
                    "ocr_text": s.ocr_text,
                    "gt_text": s.gt_text,
                })
        except DataError as e:
            logger.warning("Skipping %s: %s", ds_key, e)

    logger.info("Total training samples available: %d", len(all_train_samples))

    if not all_train_samples:
        logger.error("No training samples found. Cannot create DSPy sets.")
        sys.exit(1)

    # Shuffle and split into train / dev
    rng = random.Random(seed)
    rng.shuffle(all_train_samples)

    total_needed = n_train + n_dev
    if total_needed > len(all_train_samples):
        logger.warning(
            "Requested %d samples but only %d available. Using all.",
            total_needed, len(all_train_samples),
        )
        n_train = min(n_train, len(all_train_samples))
        n_dev = min(n_dev, len(all_train_samples) - n_train)

    dspy_train = all_train_samples[:n_train]
    dspy_dev = all_train_samples[n_train:n_train + n_dev]

    # Write trainset
    trainset_path = results_dir / "dspy_trainset.jsonl"
    with open(trainset_path, "w", encoding="utf-8") as f:
        for rec in dspy_train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d DSPy training samples -> %s", len(dspy_train), trainset_path)

    # Write devset
    devset_path = results_dir / "dspy_devset.jsonl"
    with open(devset_path, "w", encoding="utf-8") as f:
        for rec in dspy_dev:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d DSPy dev samples -> %s", len(dspy_dev), devset_path)

    # ---- Export full validation set for post-optimization inference ----
    inference_path = results_dir / "inference_input.jsonl"
    if inference_path.exists() and not args.force:
        logger.info("inference_input.jsonl already exists (use --force to overwrite)")
    else:
        sample_list_ids = None
        if args.sample_list:
            sample_list_ids, _ = load_sample_list(args.sample_list)

        n_val_total = 0
        with open(inference_path, "w", encoding="utf-8") as f:
            for ds_key in val_datasets:
                try:
                    samples = list(loader.iter_samples(ds_key))
                except DataError as e:
                    logger.warning("Skipping %s: %s", ds_key, e)
                    continue

                if sample_list_ids:
                    samples = [s for s in samples if s.sample_id in sample_list_ids]

                if args.limit:
                    samples = samples[:args.limit]

                for s in samples:
                    record = {
                        "sample_id": s.sample_id,
                        "dataset": ds_key,
                        "ocr_text": s.ocr_text,
                        "gt_text": s.gt_text,
                        "prompt_type": "dspy_optimized",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_val_total += 1

                logger.info("  %s: %d val samples", ds_key, len(samples))

        logger.info("Wrote %d total val samples -> %s", n_val_total, inference_path)

    # Write metadata
    meta = _make_meta(config, "export")
    meta["n_train"] = len(dspy_train)
    meta["n_dev"] = len(dspy_dev)
    meta["train_datasets"] = train_datasets
    meta["val_datasets"] = val_datasets
    with open(results_dir / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("Next step: run DSPy optimization on Kaggle:")
    logger.info("  python scripts/dspy_optimize.py \\")
    logger.info("      --trainset  results/phase7/dspy_trainset.jsonl \\")
    logger.info("      --devset    results/phase7/dspy_devset.jsonl \\")
    logger.info("      --input     results/phase7/inference_input.jsonl \\")
    logger.info("      --output    results/phase7/corrections.jsonl")


# ---------------------------------------------------------------------------
# ANALYZE MODE
# ---------------------------------------------------------------------------


def _load_corrections(corrections_path: Path) -> dict[str, list[dict]]:
    """Load corrections.jsonl grouped by dataset key."""
    by_dataset: dict[str, list[dict]] = {}
    with open(corrections_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ds = rec.get("dataset", "unknown")
            by_dataset.setdefault(ds, []).append(rec)
    return by_dataset


def _load_phase2_metrics(config: dict, dataset_key: str) -> Optional[dict]:
    """Load Phase 2 metrics (full dict, new + old format compat)."""
    phase2_dir = Path(config.get("output", {}).get("results_dir", "results")) / "phase2"
    return load_phase2_full_metrics(phase2_dir, dataset_key)


def run_analyze(args: argparse.Namespace, config: dict) -> None:
    """Analyze corrections from DSPy-optimized inference."""
    results_dir = Path(config.get("output", {}).get("results_dir", "results")) / "phase7"
    corrections_path = results_dir / "corrections.jsonl"

    if not corrections_path.exists():
        logger.error("corrections.jsonl not found at %s", corrections_path)
        logger.error("Run DSPy optimization + inference on Kaggle first.")
        sys.exit(1)

    all_datasets = resolve_datasets(config, args.datasets)
    val_datasets = [k for k in all_datasets if _is_val_split(k)]

    by_dataset = _load_corrections(corrections_path)
    strip_diac = config.get("evaluation", {}).get("strip_diacritics", True)
    report_both = config.get("evaluation", {}).get("report_both", True)
    do_error_analysis = (
        config.get("phase7", {}).get("analyze_errors", True)
        and not args.no_error_analysis
    )

    all_metrics: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}

    logger.info("=" * 60)
    logger.info("Phase 7 ANALYZE")
    logger.info("=" * 60)

    for ds_key in val_datasets:
        records = by_dataset.get(ds_key, [])
        if not records:
            logger.warning("No corrections for %s -- skipping", ds_key)
            continue

        logger.info("Processing %s (%d samples)", ds_key, len(records))

        ds_dir = results_dir / ds_key
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Build sample-like dicts for metrics calculation
        samples = []
        for rec in records:
            samples.append({
                "sample_id": rec["sample_id"],
                "gt_text": rec.get("gt_text", ""),
                "corrected_text": rec.get("corrected_text", ""),
                "ocr_text": rec.get("ocr_text", ""),
            })

        # Split into normal / runaway
        eval_cfg = config.get("evaluation", {})
        exclude_runaway = eval_cfg.get("exclude_runaway", False)
        threshold = eval_cfg.get("runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD)
        normal_samples, runaway_samples, data_quality = split_runaway_samples(
            samples, threshold=threshold,
        )
        if runaway_samples:
            logger.info("  %s", data_quality["description"])

        # OCR baseline
        ocr_all, ocr_all_nd = calculate_metrics_dual(
            samples, ds_key, text_field="ocr_text",
        )
        ocr_normal, ocr_normal_nd = calculate_metrics_dual(
            normal_samples, ds_key, text_field="ocr_text",
        )

        # Corrected
        all_result, all_result_nd = calculate_metrics_dual(
            samples, ds_key, text_field="corrected_text",
        )
        normal_result, normal_result_nd = calculate_metrics_dual(
            normal_samples, ds_key, text_field="corrected_text",
        )

        if exclude_runaway:
            primary_m = normal_result
            primary_nd = normal_result_nd
            primary_source = "normal_only"
        else:
            primary_m = all_result
            primary_nd = all_result_nd
            primary_source = "all"

        metrics_data = {
            "meta": {"primary_variant": primary_source},
            # OCR baseline
            "ocr_all": ocr_all.to_dict() if ocr_all else {},
            "ocr_all_no_diacritics": ocr_all_nd.to_dict() if ocr_all_nd else {},
            "ocr_normal_only": ocr_normal.to_dict() if ocr_normal else {},
            "ocr_normal_only_no_diacritics": ocr_normal_nd.to_dict() if ocr_normal_nd else {},
            # Corrected
            "corrected_all": all_result.to_dict() if all_result else {},
            "corrected_all_no_diacritics": all_result_nd.to_dict() if all_result_nd else {},
            "corrected_normal_only": normal_result.to_dict() if normal_result else {},
            "corrected_normal_only_no_diacritics": normal_result_nd.to_dict() if normal_result_nd else {},
            "data_quality": data_quality,
        }

        # Save per-dataset metrics
        with open(ds_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        all_metrics[ds_key] = metrics_data

        # Compare vs Phase 2
        p2_full = _load_phase2_metrics(config, ds_key)
        if p2_full and primary_m:
            p2_corr, p2_nd_dict, p2_src = pick_phase2_variant(p2_full, exclude_runaway)
            p2_cer = p2_corr.get("cer", None)
            p7_cer = primary_m.cer
            if p2_cer is not None:
                delta = p7_cer - p2_cer
                comparisons[ds_key] = {
                    "phase2_cer": p2_cer,
                    "phase7_cer": p7_cer,
                    "delta_cer": delta,
                    "improved": delta < 0,
                    "variant": p2_src,
                }
                direction = "v" if delta < 0 else "^"
                logger.info(
                    "  %s CER: Phase2=%.4f  Phase7=%.4f  D=%.4f %s",
                    ds_key, p2_cer, p7_cer, abs(delta), direction,
                )

        # Error analysis (optional)
        if do_error_analysis:
            try:
                analyzer = ErrorAnalyzer()
                error_data = []
                for s in samples:
                    if s["gt_text"] and s["corrected_text"]:
                        errors = analyzer.analyze(s["gt_text"], s["corrected_text"])
                        error_data.append({
                            "sample_id": s["sample_id"],
                            "errors": [e.to_dict() for e in errors] if hasattr(errors, '__iter__') else [],
                        })
                if error_data:
                    with open(ds_dir / "error_analysis.json", "w", encoding="utf-8") as f:
                        json.dump(error_data[:100], f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning("Error analysis failed for %s: %s", ds_key, e)

    # Save aggregate metrics
    # Macro-averaged aggregates by dataset group (PATS-A01 / KHATT).
    _corr = {ds: v.get("corrected_all", v.get("corrected", {})) for ds, v in all_metrics.items()
             if v.get("corrected_all") or v.get("corrected")}
    _corr_nd = {ds: v.get("corrected_all_no_diacritics", v.get("corrected_no_diacritics", {}))
                for ds, v in all_metrics.items()
                if v.get("corrected_all_no_diacritics") or v.get("corrected_no_diacritics")}
    agg_output = {
        "datasets": all_metrics,
        "group_aggregates": compute_group_aggregates(_corr) if _corr else {},
    }
    if _corr_nd:
        agg_output["group_aggregates_no_diacritics"] = compute_group_aggregates(_corr_nd)
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg_output, f, indent=2, ensure_ascii=False)

    # Save comparison
    if comparisons:
        with open(results_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2, ensure_ascii=False)

    # Generate paper tables
    _write_paper_tables(results_dir, all_metrics, comparisons, strip_diac)

    # Save metadata
    meta = _make_meta(config, "analyze")
    meta["datasets_analyzed"] = list(all_metrics.keys())
    with open(results_dir / "analyze_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Phase 7 analyze complete.")
    logger.info("=" * 60)


def _write_paper_tables(
    results_dir: Path,
    all_metrics: dict[str, dict],
    comparisons: dict[str, dict],
    strip_diac: bool,
) -> None:
    """Generate paper_tables.md with Phase 7 results (both metric variants)."""
    lines = [
        "# Phase 7: DSPy Automated Prompt Optimization -- Results",
        "",
        "## CER/WER Comparison vs Phase 2 (With Diacritics)",
        "",
        "| Dataset | Phase 2 CER | Phase 7 CER | Delta | Improved? |",
        "|---------|-------------|-------------|-------|-----------|",
    ]

    for ds_key, comp in sorted(comparisons.items()):
        improved = "Yes" if comp["improved"] else "No"
        lines.append(
            f"| {ds_key} | {comp['phase2_cer']:.4f} | {comp['phase7_cer']:.4f} "
            f"| {comp['delta_cer']:+.4f} | {improved} |"
        )

    # Full metrics — WITH DIACRITICS
    lines.extend(["", "## Full Metrics (With Diacritics)", ""])
    lines.append("| Dataset | CER | WER | CER_std | N |")
    lines.append("|---------|-----|-----|---------|---|")

    for ds_key, m in sorted(all_metrics.items()):
        md = m.get("corrected_all", m.get("corrected", {}))
        lines.append(
            f"| {ds_key} | {md.get('cer', 0):.4f} | {md.get('wer', 0):.4f} "
            f"| {md.get('cer_std', 0):.4f} | {md.get('num_samples', 0)} |"
        )

    # Full metrics — NO DIACRITICS
    has_nd = any(m.get("corrected_all_no_diacritics") or m.get("corrected_no_diacritics") for m in all_metrics.values())
    if has_nd:
        lines.extend(["", "## Full Metrics (No Diacritics)", ""])
        lines.append("| Dataset | CER | WER | CER_std | N |")
        lines.append("|---------|-----|-----|---------|---|")

        for ds_key, m in sorted(all_metrics.items()):
            md = m.get("corrected_all_no_diacritics", m.get("corrected_no_diacritics", {}))
            lines.append(
                f"| {ds_key} | {md.get('cer', 0):.4f} | {md.get('wer', 0):.4f} "
                f"| {md.get('cer_std', 0):.4f} | {md.get('num_samples', 0)} |"
            )

    with open(results_dir / "paper_tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Paper tables -> %s", results_dir / "paper_tables.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _setup_logging()
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle sample-list override
    if args.sample_list:
        _, ds_keys = load_sample_list(args.sample_list)
        if ds_keys and not args.datasets:
            args.datasets = ds_keys

    if args.mode == "export":
        run_export(args, config)
    elif args.mode == "analyze":
        run_analyze(args, config)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
