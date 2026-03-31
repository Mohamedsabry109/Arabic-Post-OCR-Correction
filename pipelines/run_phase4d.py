#!/usr/bin/env python3
"""Phase 4D: Self-Reflective Prompting.

Analyses the LLM's own predictions on training splits (where GT is available)
to extract systematic failure patterns, then injects those patterns as a
self-awareness context during validation inference.

Three-stage pipeline:

  --mode analyze-train -> Analyse LLM training-split predictions vs GT
                          -> results/phase4d/insights/PATS-A01_insights.json
                             results/phase4d/insights/KHATT_insights.json

  --mode export        -> Build self-reflective inference_input.jsonl (val splits only)
                          -> results/phase4d/inference_input.jsonl

  --mode analyze       -> Load corrections.jsonl, compute metrics and reports
                          -> results/phase4d/{dataset}/metrics.json
                             results/phase4d/metrics.json
                             results/phase4d/comparison.json
                             results/phase4d/paper_tables.md

Typical workflow
----------------
1. Prerequisite: Phase 2 train-split corrections must exist at
   results/phase2/{dataset}/corrections.jsonl

2. LOCAL:  python pipelines/run_phase4d.py --mode analyze-train
   (reads Phase 2 train predictions, outputs insights JSON files)

3. LOCAL:  python pipelines/run_phase4d.py --mode export
   (builds val-split inference JSONL with self-reflective prompts)

4. REMOTE: git clone <repo> && python scripts/infer.py \\
               --input  results/phase4d/inference_input.jsonl \\
               --output results/phase4d/corrections.jsonl \\
               --model  Qwen/Qwen3-4B-Instruct-2507
   (see notebooks/kaggle_setup.ipynb or notebooks/colab_setup.ipynb)

5. LOCAL:  python pipelines/run_phase4d.py --mode analyze

Usage
-----
    python pipelines/run_phase4d.py --mode analyze-train
    python pipelines/run_phase4d.py --mode analyze-train --source-phase phase2
    python pipelines/run_phase4d.py --mode export
    python pipelines/run_phase4d.py --mode export   --limit 50
    python pipelines/run_phase4d.py --mode export   --force
    python pipelines/run_phase4d.py --mode analyze
    python pipelines/run_phase4d.py --mode analyze  --datasets KHATT-validation
    python pipelines/run_phase4d.py --mode analyze  --no-error-analysis
"""

import argparse
import json
import logging
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
from src.data.knowledge_base import LLMInsightsLoader, WordErrorPairsLoader
from src.analysis.metrics import MetricResult, calculate_metrics, calculate_metrics_dual
from src.analysis.report_formatter import write_corrections_report
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.analysis.llm_error_analyzer import LLMErrorAnalyzer
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from pipelines._utils import (
    resolve_datasets, load_sample_list, compute_group_aggregates,
    split_runaway_samples, DEFAULT_RUNAWAY_RATIO_THRESHOLD,
    load_phase2_full_metrics, pick_phase2_variant, _pick_corrected_key,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-type classification helpers
# ---------------------------------------------------------------------------

def get_dataset_type(dataset_key: str) -> str:
    """Return 'PATS-A01' or 'KHATT' based on dataset key prefix."""
    if dataset_key.startswith("PATS-A01-"):
        return "PATS-A01"
    if dataset_key.startswith("KHATT-"):
        return "KHATT"
    return "unknown"


def is_train_split(dataset_key: str) -> bool:
    """Return True if the dataset key refers to a training split."""
    key_lower = dataset_key.lower()
    return key_lower.endswith("-train") or key_lower.endswith("train")


def is_val_split(dataset_key: str) -> bool:
    """Return True if the dataset key refers to a validation split."""
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
        description="Phase 4D: Self-Reflective Prompting for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analyze-train", "export", "analyze"],
        help=(
            "analyze-train -> analyse LLM train-split predictions to extract insights; "
            "export        -> produce inference_input.jsonl with self-reflective prompts; "
            "analyze       -> load corrections.jsonl and compute metrics"
        ),
    )
    parser.add_argument(
        "--source-phase",
        type=str,
        default=None,
        dest="source_phase",
        help=(
            "Phase whose train-split corrections to analyse (default: from config phase4d.source_phase, "
            "fallback: 'phase2')."
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
        help="Path to test_samples.json to filter samples (overrides --datasets with relevant datasets).",
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
        help="Skip error_changes.json computation in analyze mode (faster).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase4d"),
        dest="results_dir",
        help="Phase 4D results root directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase4d.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    console_stream = open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        errors="replace",
        closefd=False,
        buffering=1,
    )
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(logging.Formatter(fmt))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logger.info("Logging to %s", log_path)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def make_meta(
    dataset: str,
    num_samples: int,
    config: dict,
    limit: Optional[int],
    extra: Optional[dict] = None,
) -> dict:
    model_cfg = config.get("model", {})
    meta = {
        "phase":       "phase4d",
        "dataset":     dataset,
        "model":       model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend":     model_cfg.get("backend", "transformers"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit":  get_git_commit(),
        "num_samples": num_samples,
        "limit_applied": limit,
    }
    if extra:
        meta.update(extra)
    return meta


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_exported_datasets(output_path: Path) -> set[str]:
    """Return dataset keys already written to an inference_input.jsonl."""
    if not output_path.exists():
        return set()
    found: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ds = json.loads(line).get("dataset")
                if ds:
                    found.add(ds)
            except json.JSONDecodeError:
                pass
    if found:
        logger.info(
            "Export resume: found existing datasets in %s: %s",
            output_path, sorted(found),
        )
    return found


def _maybe_split_combined_corrections(results_dir: Path, force: bool = False) -> None:
    """Split combined corrections.jsonl into per-dataset files if needed.

    When *force* is True, existing per-dataset files are overwritten.
    """
    combined = results_dir / "corrections.jsonl"
    if not combined.exists():
        return

    logger.info("Found combined corrections.jsonl -- splitting by dataset ...")
    records_by_dataset: dict[str, list[dict]] = {}
    with open(combined, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                ds = r.get("dataset", "")
                if ds:
                    records_by_dataset.setdefault(ds, []).append(r)
            except json.JSONDecodeError:
                pass

    for ds_key, records in records_by_dataset.items():
        out_path = results_dir / ds_key / "corrections.jsonl"
        if out_path.exists() and not force:
            logger.info("  [%s] Already split -- skipping (use --force to re-split).", ds_key)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("  Split: %d records -> %s", len(records), out_path)


def load_phase2_dataset_metrics(phase2_dir: Path, dataset_key: str) -> Optional[dict]:
    """Load Phase 2 per-dataset metrics (full dict, new + old format compat)."""
    return load_phase2_full_metrics(phase2_dir, dataset_key)


def _nd_comparison_block(p2_nd: dict, corrected_nd: "MetricResult", phase_label: str) -> dict:
    """Build no-diacritics comparison sub-dict for JSON output."""
    if not p2_nd:
        return {}
    p2_cer_nd = float(p2_nd.get("cer", 0.0))
    p2_wer_nd = float(p2_nd.get("wer", 0.0))
    cer_d = p2_cer_nd - corrected_nd.cer
    wer_d = p2_wer_nd - corrected_nd.wer
    cer_r = (cer_d / p2_cer_nd * 100) if p2_cer_nd > 0 else 0.0
    wer_r = (wer_d / p2_wer_nd * 100) if p2_wer_nd > 0 else 0.0
    return {
        "phase2_baseline_no_diacritics": {"cer": round(p2_cer_nd, 6), "wer": round(p2_wer_nd, 6)},
        f"{phase_label}_corrected_no_diacritics": {"cer": round(corrected_nd.cer, 6), "wer": round(corrected_nd.wer, 6)},
        "delta_no_diacritics": {
            "cer_absolute": round(cer_d, 6), "wer_absolute": round(wer_d, 6),
            "cer_relative_pct": round(cer_r, 2), "wer_relative_pct": round(wer_r, 2),
        },
    }


def load_corrections_jsonl(corrections_path: Path) -> list[dict]:
    """Load raw correction records from a corrections.jsonl file."""
    if not corrections_path.exists():
        raise FileNotFoundError(
            f"corrections.jsonl not found: {corrections_path}\n"
            "Did you download it from Kaggle/Colab?"
        )

    records: list[dict] = []
    skipped = 0
    with open(corrections_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", lineno, corrections_path, exc
                )
                skipped += 1

    if skipped:
        logger.warning("Skipped %d malformed lines in %s", skipped, corrections_path)
    logger.info("Loaded %d records from %s", len(records), corrections_path)
    return records


# ---------------------------------------------------------------------------
# Error change analysis (same pattern as Phase 5)
# ---------------------------------------------------------------------------


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
) -> dict:
    """Compare per-type error counts before (OCR) and after (corrected)."""
    type_keys = [et.value for et in ErrorType]
    ocr_counts:   dict[str, int] = {k: 0 for k in type_keys}
    corr_counts:  dict[str, int] = {k: 0 for k in type_keys}
    fixed_counts: dict[str, int] = {k: 0 for k in type_keys}
    intro_counts: dict[str, int] = {k: 0 for k in type_keys}

    total_ocr_errors = 0
    total_corrected_errors = 0
    analyzer = ErrorAnalyzer()

    for cs in tqdm(corrected_samples, desc="  Error analysis", unit="sample"):
        try:
            class _Stub:
                pass
            s1 = _Stub()
            s1.sample_id = cs.sample.sample_id   # type: ignore[attr-defined]
            s1.dataset   = cs.sample.dataset      # type: ignore[attr-defined]
            s1.gt_text   = cs.sample.gt_text      # type: ignore[attr-defined]
            s1.ocr_text  = cs.sample.ocr_text     # type: ignore[attr-defined]

            s2 = _Stub()
            s2.sample_id = cs.sample.sample_id   # type: ignore[attr-defined]
            s2.dataset   = cs.sample.dataset      # type: ignore[attr-defined]
            s2.gt_text   = cs.sample.gt_text      # type: ignore[attr-defined]
            s2.ocr_text  = cs.corrected_text      # type: ignore[attr-defined]

            err1 = analyzer.analyse_sample(s1)  # type: ignore[arg-type]
            err2 = analyzer.analyse_sample(s2)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error analysis failed for %s: %s", cs.sample.sample_id, exc
            )
            continue

        for ce in err1.char_errors:
            k = ce.error_type.value
            ocr_counts[k] += 1
            total_ocr_errors += 1

        for ce in err2.char_errors:
            k = ce.error_type.value
            corr_counts[k] += 1
            total_corrected_errors += 1

        for k in type_keys:
            delta = ocr_counts[k] - corr_counts[k]
            if delta > 0:
                fixed_counts[k] += delta
            else:
                intro_counts[k] += abs(delta)

    total_fixed = sum(fixed_counts.values())
    total_intro = sum(intro_counts.values())

    def _pct(n: int, total: int) -> float:
        return round(n / total * 100, 4) if total > 0 else 0.0

    by_type: dict = {}
    for k in type_keys:
        if ocr_counts[k] == 0 and corr_counts[k] == 0:
            continue
        by_type[k] = {
            "ocr_count":       ocr_counts[k],
            "corrected_count": corr_counts[k],
            "fixed":           fixed_counts[k],
            "introduced":      intro_counts[k],
            "fix_rate":        round(fixed_counts[k] / max(ocr_counts[k], 1), 4),
        }

    return {
        "summary": {
            "total_ocr_char_errors":       total_ocr_errors,
            "total_corrected_char_errors": total_corrected_errors,
            "net_reduction":               total_ocr_errors - total_corrected_errors,
            "errors_fixed":                total_fixed,
            "errors_introduced":           total_intro,
            "fix_rate":          _pct(total_fixed, max(total_ocr_errors, 1)),
            "introduction_rate": _pct(total_intro, max(total_ocr_errors, 1)),
        },
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# ANALYZE-TRAIN MODE
# ---------------------------------------------------------------------------


def run_analyze_train(
    config: dict,
    all_dataset_names: list[str],
    results_dir: Path,
    source_phase: str,
    force: bool,
) -> None:
    """Analyse LLM training-split predictions to extract self-reflective insights.

    Reads corrections.jsonl for each train-split dataset from the source phase,
    computes per-ErrorType fix/introduction rates using LLMErrorAnalyzer, and
    saves aggregated insights JSON for PATS-A01 and KHATT separately.

    Args:
        config:             Parsed configuration dict.
        all_dataset_names:  All dataset keys from config.
        results_dir:        Phase 4D results root (e.g. results/phase4d/).
        source_phase:       Phase whose train-split corrections to analyse.
        force:              Re-run even if insights files already exist.
    """
    insights_dir = results_dir / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)

    pats_path = insights_dir / "PATS-A01_insights.json"
    khatt_path = insights_dir / "KHATT_insights.json"

    if not force and pats_path.exists() and khatt_path.exists():
        logger.info(
            "Insights already exist (use --force to regenerate):\n  %s\n  %s",
            pats_path, khatt_path,
        )
        return

    source_dir = _PROJECT_ROOT / "results" / source_phase
    if not source_dir.exists():
        logger.error(
            "Source phase directory not found: %s\n"
            "Run inference for %s train splits first.",
            source_dir, source_phase,
        )
        sys.exit(1)

    # Identify train-split datasets
    train_datasets = [ds for ds in all_dataset_names if is_train_split(ds)]
    if not train_datasets:
        logger.error("No train-split datasets found in config.")
        sys.exit(1)

    logger.info(
        "analyze-train: source=%s, train datasets: %s",
        source_phase, train_datasets,
    )

    llm_analyzer = LLMErrorAnalyzer()

    # Collect results by dataset type
    pats_results: list[dict] = []
    khatt_results: list[dict] = []

    for ds_key in train_datasets:
        dataset_type = get_dataset_type(ds_key)
        if dataset_type == "unknown":
            logger.warning("Unknown dataset type for %s -- skipping.", ds_key)
            continue

        # Look for corrections in per-dataset subdir (split) or combined file
        per_ds_path = source_dir / ds_key / "corrections.jsonl"
        combined_path = source_dir / "corrections.jsonl"

        if per_ds_path.exists():
            src_path = per_ds_path
        elif combined_path.exists():
            src_path = combined_path
        else:
            logger.warning(
                "[%s] No corrections.jsonl found at:\n  %s\n  %s -- skipping.",
                ds_key, per_ds_path, combined_path,
            )
            continue

        try:
            records = load_corrections_jsonl(src_path)
        except FileNotFoundError as exc:
            logger.warning("[%s] %s", ds_key, exc)
            continue

        # If reading from combined file, filter to this dataset
        if src_path == combined_path:
            records = [r for r in records if r.get("dataset") == ds_key]

        if not records:
            logger.warning("[%s] No records found -- skipping.", ds_key)
            continue

        logger.info(
            "[%s] Analysing %d records (dataset_type=%s) ...",
            ds_key, len(records), dataset_type,
        )

        sample_results: list[dict] = []
        for r in tqdm(records, desc=f"  Analyse {ds_key}", unit="sample"):
            ocr_text = r.get("ocr_text", "")
            gt_text  = r.get("gt_text", "")
            llm_text = r.get("corrected_text", ocr_text)

            if not gt_text:
                continue  # GT required for analysis

            sample_result = llm_analyzer.analyse_sample(
                ocr_text=ocr_text,
                gt_text=gt_text,
                llm_text=llm_text,
                sample_id=r.get("sample_id", ""),
                dataset=ds_key,
            )
            sample_results.append(sample_result)

        logger.info(
            "[%s] Analysed %d samples with GT.", ds_key, len(sample_results)
        )

        if dataset_type == "PATS-A01":
            pats_results.extend(sample_results)
        else:
            khatt_results.extend(sample_results)

    # Aggregate and save
    if pats_results:
        insights = llm_analyzer.aggregate(pats_results, dataset_type="PATS-A01")
        save_json(insights, pats_path)
        overall = insights["overall"]
        logger.info(
            "PATS-A01 insights: %d samples, fix_rate=%.1f%%, intro_rate=%.1f%%",
            insights["meta"]["total_samples"],
            (overall.get("fix_rate") or 0) * 100,
            (overall.get("introduction_rate") or 0) * 100,
        )
    else:
        logger.warning(
            "No PATS-A01 training results collected. "
            "Ensure %s/PATS-A01-*/corrections.jsonl exist.", source_dir,
        )

    if khatt_results:
        insights = llm_analyzer.aggregate(khatt_results, dataset_type="KHATT")
        save_json(insights, khatt_path)
        overall = insights["overall"]
        logger.info(
            "KHATT insights: %d samples, fix_rate=%.1f%%, intro_rate=%.1f%%",
            insights["meta"]["total_samples"],
            (overall.get("fix_rate") or 0) * 100,
            (overall.get("introduction_rate") or 0) * 100,
        )
    else:
        logger.warning(
            "No KHATT training results collected. "
            "Ensure %s/KHATT-train/corrections.jsonl exists.", source_dir,
        )

    logger.info("=" * 60)
    logger.info("analyze-train complete. Insights saved to %s", insights_dir)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info(
        "  python pipelines/run_phase4d.py --mode export"
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------


def run_export(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    limit: Optional[int],
    force: bool,
    explicit_datasets: bool = False,
    sample_ids: Optional[set[str]] = None,
) -> None:
    """Build inference_input.jsonl for Phase 4D (val splits only by default).

    Loads PATS-A01 and KHATT insights, formats them as Arabic text, and builds
    a self-reflective prompt per OCR sample.

    When explicit_datasets=True (--datasets passed on CLI), the val-only filter
    is skipped so any dataset (including train) can be exported for smoke tests.
    """
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    insights_dir = results_dir / "insights"
    pats_insight_path = insights_dir / "PATS-A01_insights.json"
    khatt_insight_path = insights_dir / "KHATT_insights.json"

    # Load insights
    insights_loader = LLMInsightsLoader()
    phase4d_cfg = config.get("phase4d", {})
    insight_cfg = phase4d_cfg.get("insights", {})

    format_kwargs = {
        "min_fix_rate_strength":  float(insight_cfg.get("min_fix_rate_strength",  0.6)),
        "max_fix_rate_weakness":  float(insight_cfg.get("max_fix_rate_weakness",  0.4)),
        "min_intro_rate":         float(insight_cfg.get("min_intro_rate",         0.05)),
        "min_sample_size":        int(insight_cfg.get("min_sample_size",           10)),
        "top_n_weaknesses":       int(insight_cfg.get("top_n_weaknesses",           3)),
        "top_n_overcorrections":  int(insight_cfg.get("top_n_overcorrections",      2)),
    }

    pats_insights: Optional[dict] = None
    khatt_insights: Optional[dict] = None

    if pats_insight_path.exists():
        pats_insights = insights_loader.load(pats_insight_path)
        logger.info("Loaded PATS-A01 insights from %s", pats_insight_path)
    else:
        logger.warning(
            "PATS-A01 insights not found: %s\n"
            "Run: python pipelines/run_phase4d.py --mode analyze-train",
            pats_insight_path,
        )

    if khatt_insight_path.exists():
        khatt_insights = insights_loader.load(khatt_insight_path)
        logger.info("Loaded KHATT insights from %s", khatt_insight_path)
    else:
        logger.warning(
            "KHATT insights not found: %s\n"
            "Run: python pipelines/run_phase4d.py --mode analyze-train",
            khatt_insight_path,
        )

    if pats_insights is None and khatt_insights is None:
        logger.error(
            "No insights files found. Cannot export.\n"
            "Run: python pipelines/run_phase4d.py --mode analyze-train"
        )
        sys.exit(1)

    # Pre-format insights contexts
    pats_context = (
        insights_loader.format_for_prompt(pats_insights, **format_kwargs)
        if pats_insights is not None else ""
    )
    khatt_context = (
        insights_loader.format_for_prompt(khatt_insights, **format_kwargs)
        if khatt_insights is not None else ""
    )

    logger.info(
        "Insights context lengths: PATS-A01=%d chars, KHATT=%d chars",
        len(pats_context), len(khatt_context),
    )

    # Load word-error pairs (auto-generated by analyze-train mode)
    word_pairs_context = ""
    phase4d_cfg = config.get("phase4d", {})
    pairs_path = results_dir / "word_error_pairs.txt"
    if pairs_path.exists():
        pairs_loader = WordErrorPairsLoader()
        try:
            all_pairs = pairs_loader.load(pairs_path)
            selected = pairs_loader.select(
                all_pairs,
                n=int(phase4d_cfg.get("word_pairs_n", 15)),
                strategy=str(phase4d_cfg.get("word_pairs_strategy", "random")),
                seed=int(phase4d_cfg.get("word_pairs_seed", 42)),
            )
            word_pairs_context = pairs_loader.format_for_prompt(selected)
            logger.info(
                "Loaded %d word-error pairs from %s (%d chars context).",
                len(selected), pairs_path, len(word_pairs_context),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load word-error pairs from %s: %s", pairs_path, exc)
    else:
        logger.info(
            "Word-error pairs file not found: %s\n"
            "  (Run --mode analyze-train to auto-generate it.)",
            pairs_path,
        )

    # By default only process val-split datasets; if --datasets was explicitly
    # passed on the CLI, respect that and export whatever was requested.
    if explicit_datasets:
        export_datasets = active_datasets
    else:
        export_datasets = [ds for ds in active_datasets if is_val_split(ds)]
        if not export_datasets:
            logger.warning(
                "No validation-split datasets in active set. "
                "Phase 4D only exports val splits by default. Active: %s", active_datasets,
            )
            return

    already_exported = _load_exported_datasets(output_path) if not force else set()
    loader_data = DataLoader(config)
    builder = PromptBuilder(crafted_prompt_path=config.get("prompt_craft", {}).get("crafted_prompt_path"))
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in export_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported -- skipping (use --force to re-export).",
                    ds_key,
                )
                continue

            dataset_type = get_dataset_type(ds_key)
            insights_context = pats_context if dataset_type == "PATS-A01" else khatt_context

            if not insights_context.strip() and not word_pairs_context.strip():
                logger.warning(
                    "[%s] No insights or word pairs context -- prompt will fall back to zero-shot.",
                    ds_key,
                )

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit, sample_ids=sample_ids))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in tqdm(samples, desc=f"  Export {ds_key}", unit="sample"):
                record: dict = {
                    "sample_id":        sample.sample_id,
                    "dataset":          ds_key,
                    "ocr_text":         sample.ocr_text,
                    "gt_text":          sample.gt_text,
                    "prompt_type":      "self_reflective",
                    "prompt_version":   builder.self_reflective_prompt_version,
                    "insights_context": insights_context or None,
                    "word_pairs_context": word_pairs_context or None,
                    "dataset_type":     dataset_type,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info(
                "Exported %d samples for %s (dataset_type=%s, word_pairs=%s).",
                len(samples), ds_key, dataset_type,
                "yes" if word_pairs_context else "no",
            )

    logger.info("=" * 60)
    logger.info(
        "Phase 4D export complete: %d new samples -> %s", total_written, output_path
    )
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       python scripts/infer.py \\")
    logger.info(
        "           --input  %s \\", results_dir / "inference_input.jsonl"
    )
    logger.info(
        "           --output %s", results_dir / "corrections.jsonl"
    )
    logger.info("  3. Run analysis locally:")
    logger.info("       python pipelines/run_phase4d.py --mode analyze")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# ANALYZE MODE
# ---------------------------------------------------------------------------


def process_dataset_analyze(
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    analyze_errors: bool,
) -> MetricResult:
    """Run CER/WER, comparison vs Phase 2, and optional error analysis for one dataset."""
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(corrected_samples)
    n_failed = sum(1 for cs in corrected_samples if not cs.success)
    total_prompt_tokens = sum(cs.prompt_tokens for cs in corrected_samples)
    total_output_tokens = sum(cs.output_tokens for cs in corrected_samples)
    total_latency = sum(cs.latency_s for cs in corrected_samples)

    eval_cfg = config.get("evaluation", {})
    exclude_runaway = eval_cfg.get("exclude_runaway", False)
    threshold = eval_cfg.get("runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD)

    logger.info("=" * 60)
    logger.info(
        "Analyzing [phase4d / %s]: %d samples, %d failed",
        dataset_key, n, n_failed,
    )

    # Step 1: Split into normal / runaway, compute metrics for both
    normal_samples, runaway_samples, data_quality = split_runaway_samples(
        corrected_samples, threshold=threshold,
    )
    if runaway_samples:
        logger.info("[%s] %s", dataset_key, data_quality["description"])

    logger.info("[%s] Calculating OCR baseline + corrected CER/WER ...", dataset_key)

    # OCR baseline
    ocr_all, ocr_all_nd = calculate_metrics_dual(
        corrected_samples, dataset_name=dataset_key, text_field="ocr_text",
    )
    ocr_normal, ocr_normal_nd = calculate_metrics_dual(
        normal_samples, dataset_name=dataset_key, text_field="ocr_text",
    )

    # Corrected
    all_result, all_result_nd = calculate_metrics_dual(
        corrected_samples, dataset_name=dataset_key, text_field="corrected_text",
    )
    normal_result, normal_result_nd = calculate_metrics_dual(
        normal_samples, dataset_name=dataset_key, text_field="corrected_text",
    )

    if exclude_runaway:
        primary, primary_nd = normal_result, normal_result_nd
        primary_source = "normal_only"
    else:
        primary, primary_nd = all_result, all_result_nd
        primary_source = "all"

    builder = PromptBuilder(crafted_prompt_path=config.get("prompt_craft", {}).get("crafted_prompt_path"))
    metrics_json = {
        "meta": make_meta(
            dataset_key, n, config, limit,
            extra={
                "prompt_type":    "self_reflective",
                "prompt_version": builder.self_reflective_prompt_version,
                "total_prompt_tokens":      total_prompt_tokens,
                "total_output_tokens":      total_output_tokens,
                "total_latency_s":          round(total_latency, 2),
                "avg_latency_per_sample_s": round(total_latency / max(n, 1), 3),
                "failed_samples":           n_failed,
                "primary_variant":          primary_source,
            },
        ),
        # OCR baseline
        "ocr_all": ocr_all.to_dict(),
        "ocr_all_no_diacritics": ocr_all_nd.to_dict(),
        "ocr_normal_only": ocr_normal.to_dict(),
        "ocr_normal_only_no_diacritics": ocr_normal_nd.to_dict(),
        # Corrected
        "corrected_all": all_result.to_dict(),
        "corrected_all_no_diacritics": all_result_nd.to_dict(),
        "corrected_normal_only": normal_result.to_dict(),
        "corrected_normal_only_no_diacritics": normal_result_nd.to_dict(),
        "data_quality": data_quality,
    }
    save_json(metrics_json, out_dir / "metrics.json")
    logger.info(
        "[%s] Phase4D Primary (%s): OCR CER=%.2f%% -> LLM CER=%.2f%%  |  no-diac: %.2f%% -> %.2f%%",
        dataset_key, primary_source,
        (ocr_all if not exclude_runaway else ocr_normal).cer * 100,
        primary.cer * 100,
        (ocr_all_nd if not exclude_runaway else ocr_normal_nd).cer * 100,
        primary_nd.cer * 100,
    )

    # Step 2: Comparison vs Phase 2
    p2_full = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_full is not None:
        p2_corr, p2_nd, p2_src = pick_phase2_variant(p2_full, exclude_runaway)
        p2_cer = float(p2_corr.get("cer", 0.0))
        p2_wer = float(p2_corr.get("wer", 0.0))
        cer_delta = p2_cer - primary.cer
        wer_delta = p2_wer - primary.wer
        cer_rel = (cer_delta / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(
                dataset_key, n, config, limit,
                extra={"comparison": "phase4d_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
                "variant": p2_src,
            },
            "phase4d_corrected": {
                "cer": round(primary.cer, 6),
                "wer": round(primary.wer, 6),
                "variant": primary_source,
            },
            "delta": {
                "cer_absolute":     round(cer_delta, 6),
                "wer_absolute":     round(wer_delta, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            **(_nd_comparison_block(p2_nd, primary_nd, "phase4d")),
            "interpretation": (
                f"CER {'reduced' if cer_delta >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {primary.cer*100:.2f}%). "
                f"WER {'reduced' if wer_delta >= 0 else 'increased'} by "
                f"{abs(wer_rel):.1f}% "
                f"({p2_wer*100:.2f}% -> {primary.wer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s] Phase2->Phase4D CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key,
            p2_cer * 100, primary.cer * 100, cer_rel,
            p2_wer * 100, primary.wer * 100, wer_rel,
        )
        if p2_nd:
            p2_cer_nd_v = float(p2_nd.get("cer", 0.0))
            p2_wer_nd_v = float(p2_nd.get("wer", 0.0))
            cer_d_nd_v = p2_cer_nd_v - primary_nd.cer
            wer_d_nd_v = p2_wer_nd_v - primary_nd.wer
            cer_r_nd_v = (cer_d_nd_v / p2_cer_nd_v * 100) if p2_cer_nd_v > 0 else 0.0
            wer_r_nd_v = (wer_d_nd_v / p2_wer_nd_v * 100) if p2_wer_nd_v > 0 else 0.0
            logger.info(
                "[%s] ND  CER: %.2f%% -> %.2f%% (%+.1f%%)  |  ND  WER: %.2f%% -> %.2f%% (%+.1f%%)",
                dataset_key,
                p2_cer_nd_v * 100, primary_nd.cer * 100, cer_r_nd_v,
                p2_wer_nd_v * 100, primary_nd.wer * 100, wer_r_nd_v,
            )
    else:
        logger.warning(
            "[%s] Phase 2 metrics not found -- skipping comparison.", dataset_key
        )

    # Step 3: Error change analysis (optional)
    if analyze_errors:
        logger.info("[%s] Running error change analysis ...", dataset_key)
        error_changes = run_error_change_analysis(corrected_samples, dataset_key)
        error_changes["meta"] = make_meta(
            dataset_key, n, config, limit,
            extra={"prompt_type": "self_reflective"},
        )
        save_json(error_changes, out_dir / "error_changes.json")

    return primary


def run_analyze(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    force: bool,
    analyze_errors: bool,
) -> None:
    """Compute metrics across all active val-split datasets from corrections.jsonl."""
    from src.data.data_loader import OCRSample
    from src.core.llm_corrector import CorrectedSample as CS

    _maybe_split_combined_corrections(results_dir, force=force)

    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    # Only analyze val-split datasets (train splits were used for analyze-train)
    val_datasets = [ds for ds in active_datasets if is_val_split(ds)]
    if not val_datasets:
        logger.warning("No val-split datasets in active set: %s", active_datasets)
        val_datasets = active_datasets  # fallback: try all

    for ds_key in val_datasets:
        try:
            metrics_path = results_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s] Already analyzed -- skipping (use --force).", ds_key
                )
                with open(metrics_path, encoding="utf-8") as f:
                    mdata = json.load(f)
                exclude_r = config.get("evaluation", {}).get("exclude_runaway", False)
                cd = _pick_corrected_key(mdata, exclude_r)
                all_corrected[ds_key] = MetricResult(
                    dataset=mdata.get("meta", {}).get("dataset", ds_key),
                    num_samples=cd.get("num_samples", 0),
                    num_chars_ref=cd.get("num_chars_ref", 0),
                    num_words_ref=cd.get("num_words_ref", 0),
                    cer=cd.get("cer", 0.0),
                    wer=cd.get("wer", 0.0),
                    cer_std=cd.get("cer_std", 0.0),
                    wer_std=cd.get("wer_std", 0.0),
                    cer_median=cd.get("cer_median", 0.0),
                    wer_median=cd.get("wer_median", 0.0),
                    cer_p95=cd.get("cer_p95", 0.0),
                    wer_p95=cd.get("wer_p95", 0.0),
                )
                cmp_path = results_dir / ds_key / "comparison_vs_phase2.json"
                if cmp_path.exists():
                    with open(cmp_path, encoding="utf-8") as f:
                        all_comparisons[ds_key] = json.load(f)
                continue

            corrections_path = results_dir / ds_key / "corrections.jsonl"
            records = load_corrections_jsonl(corrections_path)
            if limit:
                records = records[:limit]

            # Build CorrectedSample objects
            corrected_samples: list[CS] = []
            for r in records:
                ocr_sample = OCRSample(
                    sample_id=r["sample_id"],
                    dataset=r.get("dataset", ds_key),
                    font=None,
                    split=None,
                    ocr_text=r.get("ocr_text", ""),
                    gt_text=r.get("gt_text", ""),
                    ocr_path=Path(""),
                    gt_path=None,
                )
                corrected_samples.append(CS(
                    sample=ocr_sample,
                    corrected_text=r.get("corrected_text", r.get("ocr_text", "")),
                    prompt_tokens=r.get("prompt_tokens", 0),
                    output_tokens=r.get("output_tokens", 0),
                    latency_s=r.get("latency_s", 0.0),
                    success=r.get("success", True),
                    error=r.get("error"),
                ))

            metric_result = process_dataset_analyze(
                dataset_key=ds_key,
                corrected_samples=corrected_samples,
                config=config,
                results_dir=results_dir,
                phase2_dir=phase2_dir,
                limit=limit,
                analyze_errors=analyze_errors,
            )
            all_corrected[ds_key] = metric_result

            cmp_path = results_dir / ds_key / "comparison_vs_phase2.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("[phase4d] Dataset %s: %s", ds_key, exc)
        except DataError as exc:
            logger.warning("[phase4d] Skipping %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[phase4d] Unexpected error on %s: %s", ds_key, exc, exc_info=True
            )

    if not all_corrected:
        logger.warning("No datasets analyzed.")
        return

    # Aggregate across all datasets
    total_chars  = sum(r.num_chars_ref  for r in all_corrected.values())
    total_words  = sum(r.num_words_ref  for r in all_corrected.values())
    total_samples = sum(r.num_samples   for r in all_corrected.values())

    if total_chars > 0:
        agg_cer = sum(
            r.cer * r.num_chars_ref for r in all_corrected.values()
        ) / total_chars
    else:
        agg_cer = 0.0

    if total_words > 0:
        agg_wer = sum(
            r.wer * r.num_words_ref for r in all_corrected.values()
        ) / total_words
    else:
        agg_wer = 0.0

    # Aggregate no-diacritics metrics
    nd_data = _load_nd_results(all_corrected, results_dir)
    nd_chars  = sum(v.get("num_chars_ref", 0) for v in nd_data.values())
    nd_words  = sum(v.get("num_words_ref", 0) for v in nd_data.values())
    agg_cer_nd = (
        sum(v.get("cer", 0) * v.get("num_chars_ref", 0) for v in nd_data.values()) / nd_chars
        if nd_chars > 0 else 0.0
    )
    agg_wer_nd = (
        sum(v.get("wer", 0) * v.get("num_words_ref", 0) for v in nd_data.values()) / nd_words
        if nd_words > 0 else 0.0
    )

    aggregated = {
        "meta": make_meta(
            "all_datasets", total_samples, config, limit,
            extra={"prompt_type": "self_reflective", "datasets": sorted(all_corrected.keys())},
        ),
        "aggregated_corrected": {
            "cer": round(agg_cer, 6),
            "wer": round(agg_wer, 6),
            "total_samples": total_samples,
            "total_chars_ref": total_chars,
            "total_words_ref": total_words,
        },
        "aggregated_corrected_no_diacritics": {
            "cer": round(agg_cer_nd, 6),
            "wer": round(agg_wer_nd, 6),
        },
        "by_dataset": {
            ds: {"cer": round(r.cer, 6), "wer": round(r.wer, 6), "num_samples": r.num_samples}
            for ds, r in all_corrected.items()
        },
        "by_dataset_no_diacritics": {
            ds: {"cer": round(v.get("cer", 0), 6), "wer": round(v.get("wer", 0), 6)}
            for ds, v in nd_data.items()
        },
    }
    # Macro-averaged aggregates by dataset group (PATS-A01 / KHATT).
    aggregated["group_aggregates"] = compute_group_aggregates(aggregated["by_dataset"])
    if nd_data:
        aggregated["group_aggregates_no_diacritics"] = compute_group_aggregates(
            {ds: {"cer": v.get("cer", 0), "wer": v.get("wer", 0)} for ds, v in nd_data.items()}
        )
    save_json(aggregated, results_dir / "metrics.json")

    # Paper tables markdown
    _write_paper_tables(all_corrected, all_comparisons, results_dir)

    logger.info("=" * 60)
    logger.info("Phase 4D analysis complete.")
    logger.info(
        "Aggregated CER=%.2f%%  WER=%.2f%%  |  no-diac CER=%.2f%%  WER=%.2f%%  across %d datasets.",
        agg_cer * 100, agg_wer * 100,
        agg_cer_nd * 100, agg_wer_nd * 100,
        len(all_corrected),
    )
    logger.info("=" * 60)

    write_corrections_report(
        corrections_path=results_dir,
        output_path=results_dir / "sample_report.txt",
        title="Phase 4D -- Self-Reflective Prompting",
    )


def _load_nd_results(all_corrected: dict, results_dir: Path, exclude_runaway: bool = False) -> dict:
    """Load corrected no-diacritics from per-dataset metrics.json files."""
    nd: dict = {}
    for ds_key in all_corrected:
        path = results_dir / ds_key / "metrics.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if exclude_runaway:
                    nd_val = (
                        data.get("corrected_normal_only_no_diacritics")
                        or data.get("corrected_all_no_diacritics")
                        or data.get("corrected_no_diacritics", {})
                    )
                else:
                    nd_val = (
                        data.get("corrected_all_no_diacritics")
                        or data.get("corrected_no_diacritics", {})
                    )
                if nd_val:
                    nd[ds_key] = nd_val
            except (json.JSONDecodeError, OSError):
                pass
    return nd


def _write_paper_tables(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    results_dir: Path,
) -> None:
    """Write a markdown file with ready-to-paste paper tables."""
    nd_data = _load_nd_results(all_corrected, results_dir)

    lines: list[str] = [
        "# Phase 4D Results -- Self-Reflective Prompting",
        "",
        "## Per-Dataset Metrics (With Diacritics)",
        "",
        "| Dataset | CER (%) | WER (%) | vs Phase 2 CER | vs Phase 2 WER |",
        "|---------|---------|---------|----------------|----------------|",
    ]

    for ds_key in sorted(all_corrected.keys()):
        r = all_corrected[ds_key]
        cmp = all_comparisons.get(ds_key, {})
        delta = cmp.get("delta", {})
        cer_rel = delta.get("cer_relative_pct", "N/A")
        wer_rel = delta.get("wer_relative_pct", "N/A")
        cer_rel_str = f"{cer_rel:+.1f}%" if isinstance(cer_rel, (int, float)) else "N/A"
        wer_rel_str = f"{wer_rel:+.1f}%" if isinstance(wer_rel, (int, float)) else "N/A"
        lines.append(
            f"| {ds_key} | {r.cer*100:.2f} | {r.wer*100:.2f} "
            f"| {cer_rel_str} | {wer_rel_str} |"
        )

    lines.extend([
        "",
        "## Per-Dataset Metrics (No Diacritics)",
        "",
        "| Dataset | ND CER (%) | ND WER (%) | vs Phase 2 ND CER | vs Phase 2 ND WER |",
        "|---------|------------|------------|-------------------|-------------------|",
    ])

    for ds_key in sorted(all_corrected.keys()):
        nd = nd_data.get(ds_key, {})
        cmp = all_comparisons.get(ds_key, {})
        delta_nd = cmp.get("delta_no_diacritics", {})
        nd_cer_str = f"{nd.get('cer', 0)*100:.2f}" if nd else "N/A"
        nd_wer_str = f"{nd.get('wer', 0)*100:.2f}" if nd else "N/A"
        nd_cer_rel = delta_nd.get("cer_relative_pct", "N/A")
        nd_wer_rel = delta_nd.get("wer_relative_pct", "N/A")
        nd_cer_rel_str = f"{nd_cer_rel:+.1f}%" if isinstance(nd_cer_rel, (int, float)) else "N/A"
        nd_wer_rel_str = f"{nd_wer_rel:+.1f}%" if isinstance(nd_wer_rel, (int, float)) else "N/A"
        lines.append(
            f"| {ds_key} | {nd_cer_str} | {nd_wer_str} "
            f"| {nd_cer_rel_str} | {nd_wer_rel_str} |"
        )

    out_path = results_dir / "paper_tables.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Paper tables saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # setup_logging is called after results_dir exists
    # (done inside each mode function or here)
    args.results_dir = _PROJECT_ROOT / args.results_dir
    args.results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.results_dir)

    config = load_config(_PROJECT_ROOT / args.config)

    phase4d_cfg = config.get("phase4d", {})
    source_phase = (
        args.source_phase
        or phase4d_cfg.get("source_phase", "phase2")
    )

    all_dataset_names = resolve_datasets(config, None)  # all datasets from config
    active_datasets   = resolve_datasets(config, args.datasets)

    sample_ids: Optional[set[str]] = None
    if args.sample_list:
        sample_ids, sl_datasets = load_sample_list(args.sample_list)
        if not args.datasets:
            active_datasets = sl_datasets
        logger.info("Sample list loaded: %d sample IDs from %s", len(sample_ids), args.sample_list)

    logger.info("=" * 60)
    logger.info("Phase 4D: Self-Reflective Prompting")
    logger.info("Mode: %s", args.mode)
    if args.datasets:
        logger.info("Dataset filter: %s", args.datasets)
    if args.limit:
        logger.info("Limit: %d samples per dataset", args.limit)
    logger.info("Results dir: %s", args.results_dir)
    logger.info("=" * 60)

    phase2_dir = _PROJECT_ROOT / "results" / "phase2"

    if args.mode == "analyze-train":
        run_analyze_train(
            config=config,
            all_dataset_names=all_dataset_names,
            results_dir=args.results_dir,
            source_phase=source_phase,
            force=args.force,
        )

    elif args.mode == "export":
        run_export(
            config=config,
            active_datasets=active_datasets,
            results_dir=args.results_dir,
            limit=args.limit,
            force=args.force,
            explicit_datasets=args.datasets is not None,
            sample_ids=sample_ids,
        )

    elif args.mode == "analyze":
        run_analyze(
            config=config,
            active_datasets=active_datasets,
            results_dir=args.results_dir,
            phase2_dir=phase2_dir,
            limit=args.limit,
            force=args.force,
            analyze_errors=not args.no_error_analysis,
        )

    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
