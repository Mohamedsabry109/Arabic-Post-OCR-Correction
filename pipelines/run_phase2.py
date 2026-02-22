#!/usr/bin/env python3
"""Phase 2: Zero-Shot LLM Correction.

Three-stage pipeline (no local GPU required):

  --mode export   → Export OCR texts to inference_input.jsonl for Kaggle/Colab
  --mode analyze  → Load corrections.jsonl, compute metrics and reports
  --mode full     → End-to-end with API backend (requires model.backend='api')

Typical workflow
----------------
1. LOCAL:  python pipelines/run_phase2.py --mode export
2. REMOTE: git clone <repo> && python scripts/infer.py --input ... --output ...
           (see notebooks/kaggle_setup.ipynb or notebooks/colab_setup.ipynb)
3. LOCAL:  copy corrections.jsonl to results/phase2/
           then run: python pipelines/run_phase2.py --mode analyze
           (analyze auto-splits a combined corrections.jsonl by dataset)

Usage
-----
    python pipelines/run_phase2.py --mode export
    python pipelines/run_phase2.py --mode export   --limit 50
    python pipelines/run_phase2.py --mode export   --datasets KHATT-train KHATT-validation
    python pipelines/run_phase2.py --mode analyze  --datasets KHATT-train
    python pipelines/run_phase2.py --mode analyze  --no-error-analysis
    python pipelines/run_phase2.py --mode export   --force
    python pipelines/run_phase2.py --mode full     --datasets KHATT-train
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

from src.data.data_loader import DataLoader, DataError, OCRSample
from src.analysis.metrics import (
    MetricResult,
    calculate_metrics,
    compare_metrics,
)
from src.analysis.error_analyzer import ErrorAnalyzer, SampleError
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import (
    BaseLLMCorrector,
    CorrectionResult,
    CorrectedSample,
    get_corrector,
)
from pipelines._utils import resolve_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Zero-Shot LLM Correction for Arabic Post-OCR"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["export", "analyze", "full"],
        help=(
            "export  -> produce inference_input.jsonl for Kaggle/Colab upload; "
            "analyze -> load corrections.jsonl and compute metrics; "
            "full    -> end-to-end with API backend (requires model.backend=api)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick testing).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            "One or more dataset keys to process "
            "(e.g. KHATT-train PATS-A01-Akhbar). "
            "Defaults to all datasets from config."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process datasets that already have results (overrides resume).",
    )
    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        default=False,
        help="Skip error_changes.json computation (faster analyze runs).",
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
        default=Path("results/phase2"),
        help="Output directory for Phase 2 results.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("results/phase1"),
        help="Phase 1 results directory (used for CER/WER comparison).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers (identical pattern to run_phase1.py)
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase2.log"

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
    """Load and return config YAML as a dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    """Return short git commit hash, or 'unknown' on failure."""
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
    """Build the standard metadata block for all Phase 2 output JSON files."""
    model_cfg = config.get("model", {})
    phase2_cfg = config.get("phase2", {})
    meta = {
        "phase": "phase2",
        "dataset": dataset,
        "model": model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend": model_cfg.get("backend", "transformers"),
        "prompt_type": "zero_shot",
        "prompt_version": phase2_cfg.get("prompt_version", "v1"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "num_samples": num_samples,
        "limit_applied": limit,
    }
    if extra:
        meta.update(extra)
    return meta


def save_json(data: dict, path: Path) -> None:
    """Write data as indented JSON to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------


def _load_exported_datasets(output_path: Path) -> set[str]:
    """Return dataset keys already present in an existing inference_input.jsonl.

    Used for export resume: if a dataset key appears in the file it was already
    exported and can be skipped (unless --force).

    Args:
        output_path: Path to existing inference_input.jsonl (may not exist).

    Returns:
        Set of dataset key strings found in the file.
    """
    if not output_path.exists():
        return set()

    found: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ds = record.get("dataset")
                if ds:
                    found.add(ds)
            except json.JSONDecodeError:
                pass

    if found:
        logger.info("Export resume: found existing datasets in %s: %s", output_path, sorted(found))
    return found


def run_export(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    limit: Optional[int],
    force: bool,
) -> None:
    """Export OCR texts to inference_input.jsonl for upload to Kaggle/Colab.

    Each line of the output JSONL contains:
      sample_id, dataset, ocr_text, gt_text

    gt_text is included so the Kaggle kernel is self-contained and the
    downloaded corrections.jsonl carries gt_text for the analyze stage.

    Resume support: datasets already present in the output file are skipped
    unless force=True.
    """
    loader = DataLoader(config)
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume: discover which datasets are already in the file
    already_exported = _load_exported_datasets(output_path) if not force else set()

    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already_exported:
                logger.info("[%s] Already exported — skipping (use --force to re-export).", ds_key)
                continue

            try:
                samples = list(loader.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in samples:
                record = {
                    "sample_id": sample.sample_id,
                    "dataset":   ds_key,
                    "ocr_text":  sample.ocr_text,
                    "gt_text":   sample.gt_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info("Exported %d samples for %s", len(samples), ds_key)

    logger.info("=" * 60)
    logger.info("Export complete: %d new samples -> %s", total_written, output_path)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       git clone <repo_url> project")
    logger.info("       pip install transformers accelerate huggingface_hub pyyaml tqdm -q")
    logger.info("       python project/scripts/infer.py \\")
    logger.info("           --input  <path>/inference_input.jsonl \\")
    logger.info("           --output <path>/corrections.jsonl \\")
    logger.info("           --model  Qwen/Qwen3-4B-Instruct-2507")
    logger.info("     See notebooks/kaggle_setup.ipynb or notebooks/colab_setup.ipynb")
    logger.info("  3. Copy corrections.jsonl to results/phase2/corrections.jsonl")
    logger.info("  4. Run analysis locally:")
    logger.info("       python pipelines/run_phase2.py --mode analyze")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers for corrections.jsonl loading
# ---------------------------------------------------------------------------


def load_corrections(corrections_path: Path) -> list[CorrectedSample]:
    """Load corrections.jsonl into CorrectedSample objects.

    Args:
        corrections_path: Path to corrections.jsonl downloaded from Kaggle/Colab.

    Returns:
        List of CorrectedSample objects with nested OCRSample stubs.

    Raises:
        FileNotFoundError: If the file does not exist (with a clear message).
    """
    from src.data.data_loader import OCRSample

    if not corrections_path.exists():
        raise FileNotFoundError(
            f"corrections.jsonl not found: {corrections_path}\n"
            f"Did you download it from Kaggle/Colab?\n"
            f"Expected path: results/phase2/{{dataset_key}}/corrections.jsonl\n"
            f"Run the export + inference steps first — see docs/Kaggle_Colab_Guide.md"
        )

    corrected: list[CorrectedSample] = []
    skipped = 0

    with open(corrections_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, corrections_path, exc)
                skipped += 1
                continue

            # Reconstruct a minimal OCRSample for the nested .sample field
            ocr_sample = OCRSample(
                sample_id=r["sample_id"],
                dataset=r.get("dataset", ""),
                font=None,
                split=None,
                ocr_text=r["ocr_text"],
                gt_text=r.get("gt_text", ""),
                ocr_path=Path(""),
                gt_path=None,
            )

            corrected.append(CorrectedSample(
                sample=ocr_sample,
                corrected_text=r.get("corrected_text", r["ocr_text"]),
                prompt_tokens=r.get("prompt_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                latency_s=r.get("latency_s", 0.0),
                success=r.get("success", True),
                error=r.get("error"),
            ))

    if skipped:
        logger.warning("Skipped %d malformed lines in %s", skipped, corrections_path)

    logger.info("Loaded %d corrections from %s", len(corrected), corrections_path)
    return corrected


def _maybe_split_combined_corrections(results_dir: Path) -> None:
    """Split a combined corrections.jsonl into per-dataset files if needed.

    scripts/infer.py writes a single corrections.jsonl containing all datasets.
    This function detects that file and splits it into the per-dataset paths
    that the analyze mode expects (results_dir/{dataset_key}/corrections.jsonl).

    Already-split datasets are left untouched (their per-dataset file exists).
    """
    combined = results_dir / "corrections.jsonl"
    if not combined.exists():
        return

    logger.info("Found combined corrections.jsonl — splitting by dataset ...")
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
        if out_path.exists():
            logger.info("  [%s] Already split — skipping.", ds_key)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("  Split: %d records -> %s", len(records), out_path)


# ---------------------------------------------------------------------------
# Error change analysis
# ---------------------------------------------------------------------------


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
) -> dict:
    """Compare per-type error counts before (OCR) and after (LLM corrected).

    For each sample, runs ErrorAnalyzer on (gt, ocr) and (gt, corrected),
    then aggregates per-ErrorType fixed/introduced counts across all samples.

    Args:
        corrected_samples: List of CorrectedSample objects.
        dataset_name: Dataset label for metadata.

    Returns:
        Dict matching the error_changes.json schema.
    """
    from src.analysis.error_analyzer import ErrorType

    # Initialise per-type counters
    type_keys = [et.value for et in ErrorType]
    phase1_counts: dict[str, int] = {k: 0 for k in type_keys}
    phase2_counts: dict[str, int] = {k: 0 for k in type_keys}
    fixed_counts:  dict[str, int] = {k: 0 for k in type_keys}
    intro_counts:  dict[str, int] = {k: 0 for k in type_keys}

    total_ocr_errors = 0
    total_corrected_errors = 0

    analyzer = ErrorAnalyzer()

    for cs in tqdm(corrected_samples, desc="  Error analysis", unit="sample"):
        gt = cs.sample.gt_text
        ocr = cs.sample.ocr_text
        corrected = cs.corrected_text

        try:
            # Build minimal OCRSample-like objects for analyze_sample
            class _Stub:
                pass

            s1 = _Stub()
            s1.sample_id = cs.sample.sample_id
            s1.gt_text = gt
            s1.ocr_text = ocr

            s2 = _Stub()
            s2.sample_id = cs.sample.sample_id
            s2.gt_text = gt
            s2.ocr_text = corrected

            err1 = analyzer.analyse_sample(s1)  # type: ignore[arg-type]
            err2 = analyzer.analyse_sample(s2)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error analysis failed for %s: %s", cs.sample.sample_id, exc)
            continue

        # Count errors per type for this sample
        for ce in err1.char_errors:
            k = ce.error_type.value
            phase1_counts[k] += 1
            total_ocr_errors += 1

        for ce in err2.char_errors:
            k = ce.error_type.value
            phase2_counts[k] += 1
            total_corrected_errors += 1

        # Fixed = errors in phase1 that disappeared  (simple count difference, per type)
        for k in type_keys:
            delta = phase1_counts[k] - phase2_counts[k]
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
        if phase1_counts[k] == 0 and phase2_counts[k] == 0:
            continue
        by_type[k] = {
            "phase1_count":  phase1_counts[k],
            "phase2_count":  phase2_counts[k],
            "fixed":         fixed_counts[k],
            "introduced":    intro_counts[k],
            "fix_rate":      round(fixed_counts[k] / max(phase1_counts[k], 1), 4),
        }

    return {
        "summary": {
            "total_ocr_char_errors":        total_ocr_errors,
            "total_corrected_char_errors":  total_corrected_errors,
            "net_reduction":                total_ocr_errors - total_corrected_errors,
            "errors_fixed":                 total_fixed,
            "errors_introduced":            total_intro,
            "fix_rate":        _pct(total_fixed, max(total_ocr_errors, 1)),
            "introduction_rate": _pct(total_intro, max(total_ocr_errors, 1)),
        },
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# ANALYZE MODE — per-dataset processing
# ---------------------------------------------------------------------------


def process_dataset_analyze(
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
    limit: Optional[int],
    phase1_metrics: Optional[dict],
    analyze_errors: bool,
) -> MetricResult:
    """Run all analysis steps for one dataset given its CorrectedSample list.

    Args:
        dataset_key: E.g. "KHATT-train".
        corrected_samples: Loaded from corrections.jsonl.
        config: Parsed config dict.
        results_dir: Phase 2 results root directory.
        limit: Sample limit applied (for metadata only).
        phase1_metrics: Parsed baseline_metrics.json from Phase 1 (or None).
        analyze_errors: If True, run error_changes.json computation.

    Returns:
        MetricResult for corrected text on this dataset.
    """
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(corrected_samples)
    n_failed = sum(1 for cs in corrected_samples if not cs.success)
    total_prompt_tokens = sum(cs.prompt_tokens for cs in corrected_samples)
    total_output_tokens = sum(cs.output_tokens for cs in corrected_samples)
    total_latency = sum(cs.latency_s for cs in corrected_samples)
    avg_latency = total_latency / max(n, 1)

    logger.info("=" * 60)
    logger.info("Analyzing dataset: %s  (%d samples, %d failed)", dataset_key, n, n_failed)

    # ------------------------------------------------------------------
    # Step 1: CER/WER on corrected texts
    # ------------------------------------------------------------------
    logger.info("[%s] Calculating corrected CER/WER ...", dataset_key)
    corrected_result = calculate_metrics(
        corrected_samples,
        dataset_name=dataset_key,
        text_field="corrected_text",
    )

    metrics_json = {
        "meta": make_meta(
            dataset_key, n, config, limit,
            extra={
                "total_prompt_tokens":    total_prompt_tokens,
                "total_output_tokens":    total_output_tokens,
                "total_latency_s":        round(total_latency, 2),
                "avg_latency_per_sample_s": round(avg_latency, 3),
                "failed_samples":         n_failed,
            },
        ),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] Corrected CER=%.2f%%  WER=%.2f%%",
        dataset_key,
        corrected_result.cer * 100,
        corrected_result.wer * 100,
    )

    # ------------------------------------------------------------------
    # Step 2: Comparison vs Phase 1
    # ------------------------------------------------------------------
    if phase1_metrics is not None:
        # Pull per-dataset Phase 1 numbers from baseline_metrics.json
        p1_normal = phase1_metrics.get("results_normal_only", {}).get(dataset_key)
        p1_all    = phase1_metrics.get("results_all_samples", {}).get(dataset_key)
        p1_data   = p1_normal or p1_all  # prefer normal-only (excludes runaway)

        if p1_data:
            p1_cer = p1_data.get("cer", 0.0)
            p1_wer = p1_data.get("wer", 0.0)
            cer_delta_abs = p1_cer - corrected_result.cer
            wer_delta_abs = p1_wer - corrected_result.wer
            cer_rel = (cer_delta_abs / p1_cer * 100) if p1_cer > 0 else 0.0
            wer_rel = (wer_delta_abs / p1_wer * 100) if p1_wer > 0 else 0.0

            comparison = {
                "meta": make_meta(dataset_key, n, config, limit),
                "phase1_ocr": {
                    "cer": round(p1_cer, 6),
                    "wer": round(p1_wer, 6),
                    "source": "results_normal_only" if p1_normal else "results_all_samples",
                },
                "phase2_corrected": {
                    "cer": round(corrected_result.cer, 6),
                    "wer": round(corrected_result.wer, 6),
                },
                "delta": {
                    "cer_absolute":       round(cer_delta_abs, 6),
                    "wer_absolute":       round(wer_delta_abs, 6),
                    "cer_relative_pct":   round(cer_rel, 2),
                    "wer_relative_pct":   round(wer_rel, 2),
                },
                "interpretation": (
                    f"CER {'reduced' if cer_delta_abs >= 0 else 'increased'} by "
                    f"{abs(cer_rel):.1f}% "
                    f"({p1_cer*100:.2f}% → {corrected_result.cer*100:.2f}%). "
                    f"WER {'reduced' if wer_delta_abs >= 0 else 'increased'} by "
                    f"{abs(wer_rel):.1f}% "
                    f"({p1_wer*100:.2f}% → {corrected_result.wer*100:.2f}%)."
                ),
            }
            save_json(comparison, out_dir / "comparison_vs_phase1.json")
            logger.info(
                "[%s] CER: %.2f%% → %.2f%% (%+.1f%%)  |  WER: %.2f%% → %.2f%% (%+.1f%%)",
                dataset_key,
                p1_cer * 100, corrected_result.cer * 100, cer_rel,
                p1_wer * 100, corrected_result.wer * 100, wer_rel,
            )
        else:
            logger.warning(
                "[%s] Phase 1 metrics not found in baseline_metrics.json — skipping comparison.",
                dataset_key,
            )
    else:
        logger.warning(
            "[%s] Phase 1 baseline not available — skipping comparison_vs_phase1.json.",
            dataset_key,
        )

    # ------------------------------------------------------------------
    # Step 3: Error change analysis (optional)
    # ------------------------------------------------------------------
    if analyze_errors:
        logger.info("[%s] Running error change analysis ...", dataset_key)
        error_changes = run_error_change_analysis(corrected_samples, dataset_key)
        error_changes["meta"] = make_meta(dataset_key, n, config, limit)
        save_json(error_changes, out_dir / "error_changes.json")

    return corrected_result


# ---------------------------------------------------------------------------
# FULL MODE — inline inference + analyze
# ---------------------------------------------------------------------------


def process_dataset_full(
    dataset_key: str,
    samples: list[OCRSample],
    corrector: BaseLLMCorrector,
    builder: PromptBuilder,
    config: dict,
    results_dir: Path,
    limit: Optional[int],
    phase1_metrics: Optional[dict],
    analyze_errors: bool,
) -> MetricResult:
    """Run inline inference + analysis for one dataset (full mode).

    Writes corrections.jsonl line-by-line for resume support, then delegates
    to process_dataset_analyze for the analysis steps.
    """
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = out_dir / "corrections.jsonl"

    # Resume: load already-completed sample IDs
    completed_ids: set[str] = set()
    if corrections_path.exists():
        with open(corrections_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        completed_ids.add(json.loads(line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if completed_ids:
            logger.info(
                "[%s] Resuming: %d samples already corrected.",
                dataset_key, len(completed_ids),
            )

    pending = [s for s in samples if s.sample_id not in completed_ids]
    phase2_cfg = config.get("phase2", {})
    max_retries = phase2_cfg.get("max_retries", 2)

    with open(corrections_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(pending, desc=f"  Correcting {dataset_key}", unit="sample"):
            messages = builder.build_zero_shot(sample.ocr_text)
            result: CorrectionResult = corrector.correct(
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

    # Load all (including previously completed) for analysis
    corrected_samples = load_corrections(corrections_path)
    return process_dataset_analyze(
        dataset_key, corrected_samples, config, results_dir,
        limit, phase1_metrics, analyze_errors,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    all_corrected: dict[str, MetricResult],
    config: dict,
    results_dir: Path,
    limit: Optional[int],
) -> None:
    """Write combined metrics.json across all datasets."""
    output = {
        "meta": {
            "phase": "phase2",
            "model": config.get("model", {}).get("name", ""),
            "prompt_type": "zero_shot",
            "prompt_version": config.get("phase2", {}).get("prompt_version", "v1"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
            "limit_applied": limit,
        },
        "results": {k: v.to_dict() for k, v in all_corrected.items()},
    }
    save_json(output, results_dir / "metrics.json")


def aggregate_comparisons(
    all_comparisons: dict[str, dict],
    config: dict,
    results_dir: Path,
    limit: Optional[int],
) -> None:
    """Write combined comparison.json across all datasets."""
    if not all_comparisons:
        return

    output = {
        "meta": {
            "phase": "phase2",
            "model": config.get("model", {}).get("name", ""),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
        },
        "datasets": all_comparisons,
        "note": (
            "Phase 2 CER/WER are the reference baseline for all subsequent phases "
            "(3, 4A, 4B, 4C, 5). Negative delta = improvement."
        ),
    }
    save_json(output, results_dir / "comparison.json")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    model_name: str,
    results_dir: Path,
) -> None:
    """Write human-readable Markdown report to results_dir/report.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Phase 2 Report: Zero-Shot LLM Correction")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"Model: {model_name}")
    lines.append("Prompt: Zero-Shot (v1)")
    lines.append("")

    # Summary table
    lines.append("## Results vs Phase 1 (OCR Baseline)\n")
    if all_comparisons:
        lines.append(
            "| Dataset | Phase 1 CER | Phase 2 CER | Δ CER | "
            "Phase 1 WER | Phase 2 WER | Δ WER |"
        )
        lines.append("|---------|-------------|-------------|-------|-------------|-------------|-------|")
        for ds, cmp in all_comparisons.items():
            p1 = cmp.get("phase1_ocr", {})
            p2 = cmp.get("phase2_corrected", {})
            d  = cmp.get("delta", {})
            lines.append(
                f"| {ds} "
                f"| {p1.get('cer', 0)*100:.2f}% "
                f"| {p2.get('cer', 0)*100:.2f}% "
                f"| {d.get('cer_relative_pct', 0):+.1f}% "
                f"| {p1.get('wer', 0)*100:.2f}% "
                f"| {p2.get('wer', 0)*100:.2f}% "
                f"| {d.get('wer_relative_pct', 0):+.1f}% |"
            )
    else:
        lines.append("*Phase 1 baseline not available — comparison skipped.*")
    lines.append("")

    # Corrected metrics only (when no comparison available)
    lines.append("## Post-Correction Metrics\n")
    lines.append(
        "| Dataset | CER | WER | CER Median | WER Median | CER p95 | Samples |"
    )
    lines.append("|---------|-----|-----|------------|------------|---------|---------|")
    for ds, r in all_corrected.items():
        lines.append(
            f"| {ds} "
            f"| {r.cer*100:.2f}% "
            f"| {r.wer*100:.2f}% "
            f"| {r.cer_median*100:.2f}% "
            f"| {r.wer_median*100:.2f}% "
            f"| {r.cer_p95*100:.2f}% "
            f"| {r.num_samples:,} |"
        )
    lines.append("")

    # Error analysis tables (loaded from saved files)
    lines.append("## Error Change Analysis\n")
    for ds in all_corrected:
        ec_path = results_dir / ds / "error_changes.json"
        if ec_path.exists():
            with open(ec_path, encoding="utf-8") as f:
                ec = json.load(f)
            s = ec.get("summary", {})
            lines.append(f"### {ds}\n")
            lines.append(
                f"- Total OCR errors: {s.get('total_ocr_char_errors', 0):,}\n"
                f"- After correction: {s.get('total_corrected_char_errors', 0):,}\n"
                f"- Fixed: {s.get('errors_fixed', 0):,} "
                f"({s.get('fix_rate', 0):.1f}%)\n"
                f"- Introduced: {s.get('errors_introduced', 0):,} "
                f"({s.get('introduction_rate', 0):.1f}%)"
            )
            by_type = ec.get("by_type", {})
            if by_type:
                lines.append("")
                lines.append("| Error Type | Phase 1 | Phase 2 | Fixed | Introduced | Fix Rate |")
                lines.append("|-----------|---------|---------|-------|------------|----------|")
                for etype, data in sorted(
                    by_type.items(), key=lambda x: -x[1]["phase1_count"]
                ):
                    lines.append(
                        f"| {etype} "
                        f"| {data['phase1_count']:,} "
                        f"| {data['phase2_count']:,} "
                        f"| {data['fixed']:,} "
                        f"| {data['introduced']:,} "
                        f"| {data['fix_rate']*100:.1f}% |"
                    )
            lines.append("")

    # Key findings
    lines.append("## Key Findings\n")
    if all_corrected:
        best = min(all_corrected.items(), key=lambda x: x[1].cer)
        worst = max(all_corrected.items(), key=lambda x: x[1].cer)
        lines.append(f"- Best corrected CER: **{best[0]}** at {best[1].cer*100:.2f}%")
        lines.append(f"- Worst corrected CER: **{worst[0]}** at {worst[1].cer*100:.2f}%")
    lines.append("")
    lines.append(
        "> Phase 2 establishes the zero-shot LLM baseline.\n"
        "> Phases 3–5 each compare their results against Phase 2 CER/WER.\n"
        "> Phase 6 uses Phase 2 as the lower bound in the ablation study."
    )

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------


def print_summary(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
) -> None:
    """Print a final summary table to stdout."""
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY — Zero-Shot LLM Correction")
    print("=" * 80)
    print(f"{'Dataset':<28} {'P1 CER':>8} {'P2 CER':>8} {'Δ CER':>8} {'P2 WER':>8} {'N':>6}")
    print("-" * 80)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p1_cer = cmp.get("phase1_ocr", {}).get("cer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        p1_str = f"{p1_cer*100:.2f}%" if cmp else "N/A"
        delta_str = f"{cer_rel:+.1f}%" if cmp else "N/A"
        print(
            f"{ds:<28} {p1_str:>8} {r.cer*100:>7.2f}% {delta_str:>8} "
            f"{r.wer*100:>7.2f}% {r.num_samples:>6}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    setup_logging(results_dir)

    logger.info("Phase 2: Zero-Shot LLM Correction  (mode=%s)", args.mode)
    logger.info("Config: %s", args.config)
    logger.info("Results dir: %s", results_dir)

    config = load_config(args.config)
    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")
    analyze_errors = not args.no_error_analysis

    # Determine which datasets to run (CLI --datasets overrides config list)
    active_datasets = resolve_datasets(config, args.datasets)
    logger.info("Datasets to process: %s", active_datasets)

    # ------------------------------------------------------------------
    # EXPORT mode
    # ------------------------------------------------------------------
    if args.mode == "export":
        run_export(config, active_datasets, results_dir, limit, force=args.force)
        return

    # ------------------------------------------------------------------
    # ANALYZE / FULL modes — load Phase 1 baseline for comparison
    # ------------------------------------------------------------------
    phase1_metrics: Optional[dict] = None
    p1_baseline_path = args.phase1_dir / "baseline_metrics.json"
    if p1_baseline_path.exists():
        with open(p1_baseline_path, encoding="utf-8") as f:
            phase1_metrics = json.load(f)
        logger.info("Loaded Phase 1 baseline from %s", p1_baseline_path)
    else:
        logger.warning(
            "Phase 1 baseline not found at %s — comparison will be skipped. "
            "Run run_phase1.py first.",
            p1_baseline_path,
        )

    # ------------------------------------------------------------------
    # FULL mode — load model (expensive; done once before the loop)
    # ------------------------------------------------------------------
    corrector: Optional[BaseLLMCorrector] = None
    builder: Optional[PromptBuilder] = None
    if args.mode == "full":
        backend = config.get("model", {}).get("backend", "transformers")
        if backend == "transformers":
            logger.error(
                "--mode full requires an API backend (model.backend='api'). "
                "For GPU inference use: export -> scripts/infer.py (Kaggle/Colab) -> analyze."
            )
            sys.exit(1)
        corrector = get_corrector(config)
        builder = PromptBuilder()
        logger.info("Corrector ready: %s (backend=%s)", corrector.model_name, backend)

    # ------------------------------------------------------------------
    # ANALYZE mode: auto-split combined corrections.jsonl if present
    # ------------------------------------------------------------------
    if args.mode == "analyze":
        _maybe_split_combined_corrections(results_dir)

    # ------------------------------------------------------------------
    # Per-dataset processing
    # ------------------------------------------------------------------
    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    for ds_key in active_datasets:
        try:
            if args.mode == "analyze":
                # Resume: skip if already analyzed (unless --force)
                metrics_path = results_dir / ds_key / "metrics.json"
                if metrics_path.exists() and not args.force:
                    logger.info(
                        "[%s] Already analyzed — skipping (use --force to re-analyze).", ds_key
                    )
                    with open(metrics_path, encoding="utf-8") as f:
                        mdata = json.load(f)
                    corrected_data = mdata.get("corrected", {})
                    ds_name = mdata.get("meta", {}).get("dataset", ds_key)
                    all_corrected[ds_key] = MetricResult(
                        dataset=ds_name,
                        num_samples=corrected_data.get("num_samples", 0),
                        num_chars_ref=corrected_data.get("num_chars_ref", 0),
                        num_words_ref=corrected_data.get("num_words_ref", 0),
                        cer=corrected_data.get("cer", 0.0),
                        wer=corrected_data.get("wer", 0.0),
                        cer_std=corrected_data.get("cer_std", 0.0),
                        wer_std=corrected_data.get("wer_std", 0.0),
                        cer_median=corrected_data.get("cer_median", 0.0),
                        wer_median=corrected_data.get("wer_median", 0.0),
                        cer_p95=corrected_data.get("cer_p95", 0.0),
                        wer_p95=corrected_data.get("wer_p95", 0.0),
                    )
                    cmp_path = results_dir / ds_key / "comparison_vs_phase1.json"
                    if cmp_path.exists():
                        with open(cmp_path, encoding="utf-8") as f:
                            all_comparisons[ds_key] = json.load(f)
                    continue

                corrections_path = results_dir / ds_key / "corrections.jsonl"
                corrected_samples = load_corrections(corrections_path)
                if limit:
                    corrected_samples = corrected_samples[:limit]

                metric_result = process_dataset_analyze(
                    dataset_key=ds_key,
                    corrected_samples=corrected_samples,
                    config=config,
                    results_dir=results_dir,
                    limit=limit,
                    phase1_metrics=phase1_metrics,
                    analyze_errors=analyze_errors,
                )

            else:  # full
                loader = DataLoader(config)
                samples = list(loader.iter_samples(ds_key, limit=limit))
                if not samples:
                    logger.warning("No samples loaded for %s — skipping.", ds_key)
                    continue

                metric_result = process_dataset_full(
                    dataset_key=ds_key,
                    samples=samples,
                    corrector=corrector,
                    builder=builder,
                    config=config,
                    results_dir=results_dir,
                    limit=limit,
                    phase1_metrics=phase1_metrics,
                    analyze_errors=analyze_errors,
                )

            all_corrected[ds_key] = metric_result

            # Load saved comparison for the summary
            cmp_path = results_dir / ds_key / "comparison_vs_phase1.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("Dataset %s: %s", ds_key, exc)
        except DataError as exc:
            logger.warning("Skipping %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error on %s: %s", ds_key, exc, exc_info=True)

    if not all_corrected:
        logger.error("No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate + report
    # ------------------------------------------------------------------
    aggregate_results(all_corrected, config, results_dir, limit)
    aggregate_comparisons(all_comparisons, config, results_dir, limit)

    model_name = config.get("model", {}).get("name", "unknown")
    generate_report(all_corrected, all_comparisons, model_name, results_dir)
    print_summary(all_corrected, all_comparisons)

    logger.info("Phase 2 complete. Results in: %s", results_dir)


if __name__ == "__main__":
    main()
