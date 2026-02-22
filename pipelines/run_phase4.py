#!/usr/bin/env python3
"""Phase 4: Linguistic Knowledge Enhancement.

Three isolated sub-phases, each compared against Phase 2 (zero-shot) baseline:

  4A -- Rule-Augmented Prompting
        Injects Arabic orthographic rules into the system prompt.
        Pipeline: export -> scripts/infer.py -> analyze

  4B -- Few-Shot Prompting (QALB)
        Injects QALB error-correction examples into the system prompt.
        Pipeline: export -> scripts/infer.py -> analyze

  4C -- CAMeL Tools Validation (local post-processing)
        Applies morphological revert strategy to Phase 2 corrections.
        Pipeline: validate  (no Kaggle/Colab step needed)

Typical workflow
----------------
Phase 4A:
  1. LOCAL:  python pipelines/run_phase4.py --sub-phase 4a --mode export
  2. REMOTE: python scripts/infer.py \\
                 --input  results/phase4a/inference_input.jsonl \\
                 --output results/phase4a/corrections.jsonl
  3. LOCAL:  python pipelines/run_phase4.py --sub-phase 4a --mode analyze

Phase 4B:
  1. LOCAL:  python pipelines/run_phase4.py --sub-phase 4b --mode export
  2. REMOTE: python scripts/infer.py \\
                 --input  results/phase4b/inference_input.jsonl \\
                 --output results/phase4b/corrections.jsonl
  3. LOCAL:  python pipelines/run_phase4.py --sub-phase 4b --mode analyze

Phase 4C:
  1. LOCAL:  python pipelines/run_phase4.py --sub-phase 4c --mode validate

Usage
-----
    python pipelines/run_phase4.py --sub-phase 4a --mode export
    python pipelines/run_phase4.py --sub-phase 4a --mode export   --limit 50
    python pipelines/run_phase4.py --sub-phase 4a --mode export   --datasets KHATT-train
    python pipelines/run_phase4.py --sub-phase 4a --mode analyze
    python pipelines/run_phase4.py --sub-phase 4a --mode analyze  --no-error-analysis
    python pipelines/run_phase4.py --sub-phase 4b --mode export   --num-examples 10
    python pipelines/run_phase4.py --sub-phase 4b --mode analyze
    python pipelines/run_phase4.py --sub-phase 4c --mode validate
    python pipelines/run_phase4.py --sub-phase 4c --mode validate --datasets KHATT-train
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
from src.data.knowledge_base import RulesLoader, ArabicRule, QALBLoader, QALBPair
from src.analysis.metrics import MetricResult, calculate_metrics
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import (
    BaseLLMCorrector,
    CorrectionResult,
    CorrectedSample,
    get_corrector,
)
from src.linguistic.morphology import MorphAnalyzer
from src.linguistic.validator import WordValidator, TextCorrectionResult
from pipelines._utils import resolve_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Linguistic Knowledge Enhancement (4A/4B/4C)"
    )
    parser.add_argument(
        "--sub-phase",
        type=str,
        required=True,
        choices=["4a", "4b", "4c"],
        dest="sub_phase",
        help=(
            "4a = Rule-Augmented Prompting; "
            "4b = Few-Shot (QALB); "
            "4c = CAMeL Tools Validation"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["export", "analyze", "validate"],
        help=(
            "export   -> write inference_input.jsonl (4A/4B only); "
            "analyze  -> compute metrics from corrections.jsonl (4A/4B only); "
            "validate -> apply local post-processing (4C only)"
        ),
    )
    # Phase 4A options
    parser.add_argument(
        "--n-rules",
        type=int,
        default=None,
        dest="n_rules",
        help="Max number of rules to inject (Phase 4A). Default: all rules.",
    )
    parser.add_argument(
        "--rule-categories",
        nargs="+",
        default=None,
        dest="rule_categories",
        help=(
            "Rule categories to include (Phase 4A). "
            "Default: all. Options: taa_marbuta hamza alef_maksura "
            "alef_forms dots similar_shapes"
        ),
    )
    # Phase 4B options
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        dest="num_examples",
        help="Number of QALB few-shot examples (Phase 4B, default: 5).",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="diverse",
        choices=["diverse", "random", "most_common"],
        help="Example selection strategy (Phase 4B, default: diverse).",
    )
    parser.add_argument(
        "--qalb-years",
        nargs="+",
        default=["2014"],
        dest="qalb_years",
        metavar="YEAR",
        help="QALB corpus years to use (Phase 4B, default: 2014).",
    )
    # Phase 4C options
    parser.add_argument(
        "--strategy",
        type=str,
        default="revert",
        choices=["revert"],
        help="Validation strategy (Phase 4C, default: revert).",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("results/phase2"),
        dest="phase2_dir",
        help="Phase 2 results directory (baseline for comparison / 4C source).",
    )
    # Common options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset (for quick testing).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            "One or more dataset keys to process "
            "(e.g. KHATT-train PATS-A01-Akhbar-train). "
            "Defaults to all datasets from config."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process datasets that already have results.",
    )
    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        default=False,
        help="Skip error_changes.json computation (4A/4B analyze, faster).",
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


def setup_logging(results_dir: Path, phase_label: str) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{phase_label}.log"

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
    phase: str,
    dataset: str,
    num_samples: int,
    config: dict,
    limit: Optional[int],
    extra: Optional[dict] = None,
) -> dict:
    """Build the standard metadata block for all Phase 4 output JSON files."""
    model_cfg = config.get("model", {})
    meta = {
        "phase": phase,
        "dataset": dataset,
        "model": model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend": model_cfg.get("backend", "transformers"),
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
# corrections.jsonl loading (shared by 4A/4B analyze)
# ---------------------------------------------------------------------------


def load_corrections(corrections_path: Path) -> list[CorrectedSample]:
    """Load corrections.jsonl into CorrectedSample objects."""
    if not corrections_path.exists():
        raise FileNotFoundError(
            f"corrections.jsonl not found: {corrections_path}\n"
            "Did you download it from Kaggle/Colab?\n"
            "Run the export + inference steps first."
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
                logger.warning(
                    "Skipping malformed line %d in %s: %s", lineno, corrections_path, exc
                )
                skipped += 1
                continue

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


def _load_exported_datasets(output_path: Path) -> set[str]:
    """Return dataset keys already present in an existing inference_input.jsonl."""
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


def _maybe_split_combined_corrections(results_dir: Path) -> None:
    """Split a combined corrections.jsonl into per-dataset files if needed."""
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
        if out_path.exists():
            logger.info("  [%s] Already split -- skipping.", ds_key)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("  Split: %d records -> %s", len(records), out_path)


# ---------------------------------------------------------------------------
# Error change analysis (shared by 4A, 4B)
# ---------------------------------------------------------------------------


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
    phase_label: str,
) -> dict:
    """Compare per-type error counts before (OCR) and after (corrected).

    Args:
        corrected_samples: List of CorrectedSample objects.
        dataset_name: Dataset key label (for logging).
        phase_label: Phase label for result keys (e.g. "phase4a").

    Returns:
        error_changes dict with summary and by_type breakdown.
    """
    type_keys = [et.value for et in ErrorType]
    ocr_counts:     dict[str, int] = {k: 0 for k in type_keys}
    corr_counts:    dict[str, int] = {k: 0 for k in type_keys}
    fixed_counts:   dict[str, int] = {k: 0 for k in type_keys}
    intro_counts:   dict[str, int] = {k: 0 for k in type_keys}

    total_ocr_errors = 0
    total_corrected_errors = 0

    analyzer = ErrorAnalyzer()

    for cs in tqdm(corrected_samples, desc="  Error analysis", unit="sample"):
        gt  = cs.sample.gt_text
        ocr = cs.sample.ocr_text
        corrected = cs.corrected_text

        try:
            class _Stub:
                pass

            s1 = _Stub()
            s1.sample_id = cs.sample.sample_id
            s1.gt_text   = gt
            s1.ocr_text  = ocr

            s2 = _Stub()
            s2.sample_id = cs.sample.sample_id
            s2.gt_text   = gt
            s2.ocr_text  = corrected

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
# Phase 2 baseline loading
# ---------------------------------------------------------------------------


def load_phase2_dataset_metrics(phase2_dir: Path, dataset_key: str) -> Optional[dict]:
    """Load Phase 2 per-dataset corrected metrics for comparison."""
    path = phase2_dir / dataset_key / "metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("corrected", {})
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load Phase 2 metrics for %s: %s", dataset_key, exc)
        return None


# ---------------------------------------------------------------------------
# Generic analyze step (shared by 4A and 4B)
# ---------------------------------------------------------------------------


def process_dataset_analyze(
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    phase_label: str,
    prompt_type: str,
    prompt_version: str,
    limit: Optional[int],
    analyze_errors: bool,
    extra_meta: Optional[dict] = None,
) -> MetricResult:
    """Run all analysis steps for one dataset (4A or 4B).

    Args:
        dataset_key: e.g. "KHATT-train".
        corrected_samples: Loaded from corrections.jsonl.
        config: Parsed config dict.
        results_dir: Phase 4A or 4B results root.
        phase2_dir: Phase 2 results root (baseline metrics).
        phase_label: "phase4a" or "phase4b" (for metadata and file labels).
        prompt_type: "rule_augmented" or "few_shot".
        prompt_version: e.g. "p4av1" or "p4bv1".
        limit: Sample limit applied (for metadata only).
        analyze_errors: If True, compute error_changes.json.
        extra_meta: Additional metadata to merge into meta blocks.

    Returns:
        MetricResult for corrected text.
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
    logger.info(
        "Analyzing [%s] dataset: %s  (%d samples, %d failed)",
        phase_label, dataset_key, n, n_failed,
    )

    # ------------------------------------------------------------------
    # Step 1: CER/WER on corrected texts
    # ------------------------------------------------------------------
    logger.info("[%s] Calculating corrected CER/WER ...", dataset_key)
    corrected_result = calculate_metrics(
        corrected_samples,
        dataset_name=dataset_key,
        text_field="corrected_text",
    )

    metrics_extra = {
        "prompt_type":              prompt_type,
        "prompt_version":           prompt_version,
        "total_prompt_tokens":      total_prompt_tokens,
        "total_output_tokens":      total_output_tokens,
        "total_latency_s":          round(total_latency, 2),
        "avg_latency_per_sample_s": round(avg_latency, 3),
        "failed_samples":           n_failed,
    }
    if extra_meta:
        metrics_extra.update(extra_meta)

    metrics_json = {
        "meta": make_meta(phase_label, dataset_key, n, config, limit, extra=metrics_extra),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] %s CER=%.2f%%  WER=%.2f%%",
        dataset_key, phase_label.upper(),
        corrected_result.cer * 100, corrected_result.wer * 100,
    )

    # ------------------------------------------------------------------
    # Step 2: Comparison vs Phase 2 baseline (ISOLATED)
    # ------------------------------------------------------------------
    p2_metrics = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    comparison: dict = {}
    if p2_metrics is not None:
        p2_cer = float(p2_metrics.get("cer", 0.0))
        p2_wer = float(p2_metrics.get("wer", 0.0))
        cer_delta_abs = p2_cer - corrected_result.cer    # positive = current phase better
        wer_delta_abs = p2_wer - corrected_result.wer
        cer_rel = (cer_delta_abs / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta_abs / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(
                phase_label, dataset_key, n, config, limit,
                extra={"comparison": f"{phase_label}_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
            },
            f"{phase_label}_corrected": {
                "cer": round(corrected_result.cer, 6),
                "wer": round(corrected_result.wer, 6),
            },
            "delta": {
                "cer_absolute":     round(cer_delta_abs, 6),
                "wer_absolute":     round(wer_delta_abs, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            "interpretation": (
                f"CER {'reduced' if cer_delta_abs >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {corrected_result.cer*100:.2f}%). "
                f"WER {'reduced' if wer_delta_abs >= 0 else 'increased'} by "
                f"{abs(wer_rel):.1f}% "
                f"({p2_wer*100:.2f}% -> {corrected_result.wer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s] Phase2->%s CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key, phase_label.upper(),
            p2_cer * 100, corrected_result.cer * 100, cer_rel,
            p2_wer * 100, corrected_result.wer * 100, wer_rel,
        )
    else:
        logger.warning(
            "[%s] Phase 2 metrics not found at %s -- skipping comparison_vs_phase2.json.",
            dataset_key, phase2_dir / dataset_key / "metrics.json",
        )

    # ------------------------------------------------------------------
    # Step 3: Error change analysis (optional)
    # ------------------------------------------------------------------
    if analyze_errors:
        logger.info("[%s] Running error change analysis ...", dataset_key)
        error_changes = run_error_change_analysis(
            corrected_samples, dataset_key, phase_label
        )
        error_changes["meta"] = make_meta(
            phase_label, dataset_key, n, config, limit,
            extra={"prompt_type": prompt_type},
        )
        save_json(error_changes, out_dir / "error_changes.json")

    return corrected_result


# ---------------------------------------------------------------------------
# PHASE 4A -- Rule-Augmented Prompting
# ---------------------------------------------------------------------------


def build_rules_context(
    config: dict,
    categories: Optional[list[str]],
    n_rules: Optional[int],
) -> str:
    """Build the rules context string to inject into prompts (Phase 4A).

    Args:
        config: Parsed config dict.
        categories: Filter to these rule categories (None = all).
        n_rules: Cap the number of rules injected (None = all).

    Returns:
        Pre-formatted Arabic rules string for prompt injection.
    """
    phase4_cfg = config.get("phase4", {}).get("rules", {})
    cats = categories or phase4_cfg.get("categories")
    n = n_rules or phase4_cfg.get("n_rules")
    style = phase4_cfg.get("format_style", "compact_arabic")

    loader = RulesLoader()
    rules = loader.load(categories=cats)

    if not rules:
        logger.warning("No rules loaded. Falling back to zero-shot for all samples.")
        return ""

    context = loader.format_for_prompt(rules, n=n, style=style)
    logger.info(
        "Rules context: %d rules loaded, %d chars injected (style=%s).",
        len(rules), len(context), style,
    )
    return context


def run_export_4a(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    categories: Optional[list[str]],
    n_rules: Optional[int],
    limit: Optional[int],
    force: bool,
) -> None:
    """Export OCR texts with rules context to inference_input.jsonl (Phase 4A).

    Each line contains: sample_id, dataset, ocr_text, gt_text,
    prompt_type ("rule_augmented"), rules_context.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        results_dir: Phase 4A results root.
        categories: Rule categories to include (None = all).
        n_rules: Max rules to inject (None = all).
        limit: Max samples per dataset.
        force: If True, ignore existing exported datasets.
    """
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    loader_data = DataLoader(config)

    # Build rules context once — same for all datasets
    rules_context = build_rules_context(config, categories, n_rules)
    prompt_type = "rule_augmented" if rules_context else "zero_shot"

    already_exported = _load_exported_datasets(output_path) if not force else set()
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported -- skipping (use --force to re-export).", ds_key
                )
                continue

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in samples:
                record = {
                    "sample_id":    sample.sample_id,
                    "dataset":      ds_key,
                    "ocr_text":     sample.ocr_text,
                    "gt_text":      sample.gt_text,
                    "prompt_type":  prompt_type,
                    "rules_context": rules_context,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info(
                "Exported %d samples for %s (prompt_type=%s, context_len=%d chars).",
                len(samples), ds_key, prompt_type, len(rules_context),
            )

    logger.info("=" * 60)
    logger.info("Phase 4A export complete: %d new samples -> %s", total_written, output_path)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       python scripts/infer.py \\")
    logger.info("           --input  results/phase4a/inference_input.jsonl \\")
    logger.info("           --output results/phase4a/corrections.jsonl")
    logger.info("  3. Run analysis locally:")
    logger.info("       python pipelines/run_phase4.py --sub-phase 4a --mode analyze")
    logger.info("=" * 60)


def run_analyze_4a(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    force: bool,
    analyze_errors: bool,
    categories: Optional[list[str]],
    n_rules: Optional[int],
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Analyze Phase 4A corrections: compute metrics and comparisons.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        results_dir: Phase 4A results root.
        phase2_dir: Phase 2 results root (baseline).
        limit: Sample limit (for metadata).
        force: If True, re-analyze already-done datasets.
        analyze_errors: If True, compute error_changes.json.
        categories: Rule categories used in export (for metadata).
        n_rules: N rules cap used in export (for metadata).

    Returns:
        Tuple of (all_corrected, all_comparisons) dicts.
    """
    _maybe_split_combined_corrections(results_dir)

    builder = PromptBuilder()
    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    for ds_key in active_datasets:
        try:
            metrics_path = results_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s] Already analyzed -- skipping (use --force to re-analyze).", ds_key
                )
                with open(metrics_path, encoding="utf-8") as f:
                    mdata = json.load(f)
                cd = mdata.get("corrected", {})
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
            corrected_samples = load_corrections(corrections_path)
            if limit:
                corrected_samples = corrected_samples[:limit]

            extra_meta = {
                "rule_categories": categories,
                "n_rules": n_rules,
            }
            metric_result = process_dataset_analyze(
                dataset_key=ds_key,
                corrected_samples=corrected_samples,
                config=config,
                results_dir=results_dir,
                phase2_dir=phase2_dir,
                phase_label="phase4a",
                prompt_type="rule_augmented",
                prompt_version=builder.rules_prompt_version,
                limit=limit,
                analyze_errors=analyze_errors,
                extra_meta=extra_meta,
            )
            all_corrected[ds_key] = metric_result

            cmp_path = results_dir / ds_key / "comparison_vs_phase2.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("Dataset %s: %s", ds_key, exc)
        except DataError as exc:
            logger.warning("Skipping %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error on %s: %s", ds_key, exc, exc_info=True)

    return all_corrected, all_comparisons


# ---------------------------------------------------------------------------
# PHASE 4B -- Few-Shot Prompting (QALB)
# ---------------------------------------------------------------------------


def build_examples_context(
    config: dict,
    num_examples: int,
    selection: str,
    years: list[str],
) -> str:
    """Build the few-shot examples context string (Phase 4B).

    Args:
        config: Parsed config dict.
        num_examples: Number of examples to select.
        selection: Selection strategy ("diverse", "random", "most_common").
        years: QALB corpus years to load.

    Returns:
        Pre-formatted Arabic few-shot examples string for prompt injection.
    """
    phase4b_cfg = config.get("phase4", {}).get("few_shot", {})
    seed = phase4b_cfg.get("seed", 42)
    max_length = phase4b_cfg.get("max_length", 100)
    min_length = phase4b_cfg.get("min_length", 10)
    max_words_changed = phase4b_cfg.get("max_words_changed", 4)
    format_style = phase4b_cfg.get("format_style", "inline_arabic")

    loader = QALBLoader(config)
    try:
        pairs = loader.load(splits=["train"], years=years)
    except Exception as exc:  # noqa: BLE001
        logger.warning("QALB load failed: %s -- falling back to zero-shot.", exc)
        return ""

    if not pairs:
        logger.warning(
            "No QALB pairs loaded for years=%s. Falling back to zero-shot.", years
        )
        return ""

    filtered = loader.filter_ocr_relevant(
        pairs,
        max_length=max_length,
        min_length=min_length,
        max_words_changed=max_words_changed,
    )
    logger.info(
        "QALB: %d total pairs, %d after OCR filtering.", len(pairs), len(filtered)
    )

    if not filtered:
        logger.warning(
            "No OCR-relevant QALB pairs found. Falling back to zero-shot."
        )
        return ""

    selected = loader.select(
        filtered, n=num_examples, strategy=selection, seed=seed
    )
    context = loader.format_for_prompt(selected, style=format_style)
    logger.info(
        "Few-shot context: %d examples selected (%s), %d chars.",
        len(selected), selection, len(context),
    )
    return context


def run_export_4b(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    num_examples: int,
    selection: str,
    years: list[str],
    limit: Optional[int],
    force: bool,
) -> None:
    """Export OCR texts with few-shot examples context (Phase 4B).

    Each line contains: sample_id, dataset, ocr_text, gt_text,
    prompt_type ("few_shot"), examples_context.
    """
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    loader_data = DataLoader(config)

    # Build examples context once -- same for all datasets
    examples_context = build_examples_context(config, num_examples, selection, years)
    prompt_type = "few_shot" if examples_context else "zero_shot"

    already_exported = _load_exported_datasets(output_path) if not force else set()
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported -- skipping (use --force to re-export).", ds_key
                )
                continue

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in samples:
                record = {
                    "sample_id":       sample.sample_id,
                    "dataset":         ds_key,
                    "ocr_text":        sample.ocr_text,
                    "gt_text":         sample.gt_text,
                    "prompt_type":     prompt_type,
                    "examples_context": examples_context,
                    "num_examples":    num_examples,
                    "selection":       selection,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info(
                "Exported %d samples for %s (prompt_type=%s, context_len=%d chars).",
                len(samples), ds_key, prompt_type, len(examples_context),
            )

    logger.info("=" * 60)
    logger.info("Phase 4B export complete: %d new samples -> %s", total_written, output_path)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       python scripts/infer.py \\")
    logger.info("           --input  results/phase4b/inference_input.jsonl \\")
    logger.info("           --output results/phase4b/corrections.jsonl")
    logger.info("  3. Run analysis locally:")
    logger.info("       python pipelines/run_phase4.py --sub-phase 4b --mode analyze")
    logger.info("=" * 60)


def run_analyze_4b(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    force: bool,
    analyze_errors: bool,
    num_examples: int,
    selection: str,
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Analyze Phase 4B corrections: compute metrics and comparisons."""
    _maybe_split_combined_corrections(results_dir)

    builder = PromptBuilder()
    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    for ds_key in active_datasets:
        try:
            metrics_path = results_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s] Already analyzed -- skipping (use --force to re-analyze).", ds_key
                )
                with open(metrics_path, encoding="utf-8") as f:
                    mdata = json.load(f)
                cd = mdata.get("corrected", {})
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
            corrected_samples = load_corrections(corrections_path)
            if limit:
                corrected_samples = corrected_samples[:limit]

            extra_meta = {
                "num_examples": num_examples,
                "selection_strategy": selection,
            }
            metric_result = process_dataset_analyze(
                dataset_key=ds_key,
                corrected_samples=corrected_samples,
                config=config,
                results_dir=results_dir,
                phase2_dir=phase2_dir,
                phase_label="phase4b",
                prompt_type="few_shot",
                prompt_version=builder.few_shot_prompt_version,
                limit=limit,
                analyze_errors=analyze_errors,
                extra_meta=extra_meta,
            )
            all_corrected[ds_key] = metric_result

            cmp_path = results_dir / ds_key / "comparison_vs_phase2.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("Dataset %s: %s", ds_key, exc)
        except DataError as exc:
            logger.warning("Skipping %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error on %s: %s", ds_key, exc, exc_info=True)

    return all_corrected, all_comparisons


# ---------------------------------------------------------------------------
# PHASE 4C -- CAMeL Tools Validation
# ---------------------------------------------------------------------------


def _load_phase2_corrections_for_dataset(
    phase2_dir: Path,
    dataset_key: str,
) -> list[dict]:
    """Load Phase 2 raw correction records for one dataset.

    Tries per-dataset path first, then falls back to combined corrections.jsonl.

    Returns:
        List of raw dicts from corrections.jsonl for this dataset.
    """
    per_ds_path = phase2_dir / dataset_key / "corrections.jsonl"
    combined_path = phase2_dir / "corrections.jsonl"

    records: list[dict] = []

    if per_ds_path.exists():
        src = per_ds_path
    elif combined_path.exists():
        src = combined_path
    else:
        raise FileNotFoundError(
            f"Phase 2 corrections not found for {dataset_key}.\n"
            f"Looked at: {per_ds_path} and {combined_path}\n"
            "Run Phase 2 first."
        )

    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("dataset") == dataset_key or src == per_ds_path:
                    records.append(r)
            except json.JSONDecodeError:
                pass

    logger.info(
        "Loaded %d Phase 2 records for %s from %s", len(records), dataset_key, src
    )
    return records


def process_dataset_validate(
    dataset_key: str,
    phase2_records: list[dict],
    validator: WordValidator,
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    strategy: str,
    limit: Optional[int],
) -> MetricResult:
    """Apply CAMeL revert strategy to Phase 2 corrections for one dataset.

    Reads Phase 2 {llm_text, ocr_text}, applies revert strategy, writes
    corrected output, computes metrics, and saves all result files.

    Args:
        dataset_key: e.g. "KHATT-train".
        phase2_records: Raw dicts from Phase 2 corrections.jsonl.
        validator: Initialised WordValidator instance.
        config: Parsed config dict.
        results_dir: Phase 4C results root.
        phase2_dir: Phase 2 results root (for baseline comparison).
        strategy: Revert strategy (currently only "revert").
        limit: Sample limit applied (for metadata).

    Returns:
        MetricResult for the post-validation corrected text.
    """
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        phase2_records = phase2_records[:limit]

    n = len(phase2_records)
    logger.info("=" * 60)
    logger.info("Phase 4C validate: %s  (%d samples, strategy=%s)", dataset_key, n, strategy)

    corrected_samples: list[CorrectedSample] = []
    revert_results: list[TextCorrectionResult] = []

    out_corrections_path = out_dir / "corrections.jsonl"
    with open(out_corrections_path, "w", encoding="utf-8") as out_f:
        for r in tqdm(phase2_records, desc=f"  Validating {dataset_key}", unit="sample"):
            llm_text = r.get("corrected_text", r["ocr_text"])
            ocr_text = r.get("ocr_text", "")
            gt_text  = r.get("gt_text", "")

            revert_result = validator.validate_correction(llm_text, ocr_text, strategy=strategy)
            revert_results.append(revert_result)

            out_record = {
                "sample_id":      r["sample_id"],
                "dataset":        dataset_key,
                "ocr_text":       ocr_text,
                "corrected_text": revert_result.final_text,
                "gt_text":        gt_text,
                "phase2_text":    llm_text,
                "reverted_count": revert_result.reverted_count,
                "kept_count":     revert_result.kept_count,
                "unchanged_count": revert_result.unchanged_count,
                "revert_rate":    round(revert_result.revert_rate, 4),
                "strategy":       strategy,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            ocr_sample = OCRSample(
                sample_id=r["sample_id"],
                dataset=dataset_key,
                font=None,
                split=None,
                ocr_text=ocr_text,
                gt_text=gt_text,
                ocr_path=Path(""),
                gt_path=None,
            )
            corrected_samples.append(CorrectedSample(
                sample=ocr_sample,
                corrected_text=revert_result.final_text,
                prompt_tokens=0,
                output_tokens=0,
                latency_s=0.0,
                success=True,
                error=None,
            ))

    logger.info("Phase 4C corrections written to %s", out_corrections_path)

    # ------------------------------------------------------------------
    # Aggregate revert statistics
    # ------------------------------------------------------------------
    total_words    = sum(rr.total_words for rr in revert_results)
    total_reverted = sum(rr.reverted_count for rr in revert_results)
    total_kept     = sum(rr.kept_count for rr in revert_results)
    total_unchanged = sum(rr.unchanged_count for rr in revert_results)
    avg_revert_rate = total_reverted / max(total_words, 1)

    samples_with_reverts = sum(1 for rr in revert_results if rr.reverted_count > 0)

    validation_stats = {
        "meta": make_meta(
            "phase4c", dataset_key, n, config, limit,
            extra={
                "strategy":         strategy,
                "camel_enabled":    validator._analyzer.enabled,  # noqa: SLF001
            },
        ),
        "summary": {
            "total_samples":         n,
            "samples_with_reverts":  samples_with_reverts,
            "total_arabic_words":    total_words,
            "words_reverted":        total_reverted,
            "words_kept":            total_kept,
            "words_unchanged":       total_unchanged,
            "avg_revert_rate":       round(avg_revert_rate, 4),
            "revert_rate_pct":       round(avg_revert_rate * 100, 2),
        },
        "note": (
            "Words reverted = LLM word was morphologically invalid AND "
            "OCR word was morphologically valid -> revert to OCR. "
            "Words kept = LLM correction kept as-is. "
            "Words unchanged = identical in LLM and OCR output."
        ),
    }
    save_json(validation_stats, out_dir / "validation_stats.json")

    logger.info(
        "[%s] Revert stats: %d total words | %d reverted (%.1f%%) | %d kept | %d unchanged",
        dataset_key, total_words, total_reverted,
        avg_revert_rate * 100, total_kept, total_unchanged,
    )

    # ------------------------------------------------------------------
    # CER/WER on post-validation corrected texts
    # ------------------------------------------------------------------
    logger.info("[%s] Calculating Phase 4C CER/WER ...", dataset_key)
    corrected_result = calculate_metrics(
        corrected_samples, dataset_name=dataset_key, text_field="corrected_text"
    )

    metrics_json = {
        "meta": make_meta(
            "phase4c", dataset_key, n, config, limit,
            extra={
                "strategy":         strategy,
                "prompt_type":      "camel_validation",
                "total_reverted":   total_reverted,
                "avg_revert_rate":  round(avg_revert_rate, 4),
            },
        ),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] Phase 4C CER=%.2f%%  WER=%.2f%%",
        dataset_key, corrected_result.cer * 100, corrected_result.wer * 100,
    )

    # ------------------------------------------------------------------
    # Comparison vs Phase 2
    # ------------------------------------------------------------------
    p2_metrics = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_metrics is not None:
        p2_cer = float(p2_metrics.get("cer", 0.0))
        p2_wer = float(p2_metrics.get("wer", 0.0))
        cer_delta_abs = p2_cer - corrected_result.cer
        wer_delta_abs = p2_wer - corrected_result.wer
        cer_rel = (cer_delta_abs / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta_abs / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(
                "phase4c", dataset_key, n, config, limit,
                extra={"comparison": "phase4c_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
            },
            "phase4c_corrected": {
                "cer": round(corrected_result.cer, 6),
                "wer": round(corrected_result.wer, 6),
            },
            "delta": {
                "cer_absolute":     round(cer_delta_abs, 6),
                "wer_absolute":     round(wer_delta_abs, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            "interpretation": (
                f"CER {'reduced' if cer_delta_abs >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {corrected_result.cer*100:.2f}%). "
                f"WER {'reduced' if wer_delta_abs >= 0 else 'increased'} by "
                f"{abs(wer_rel):.1f}% "
                f"({p2_wer*100:.2f}% -> {corrected_result.wer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s] Phase2->Phase4C CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key,
            p2_cer * 100, corrected_result.cer * 100, cer_rel,
            p2_wer * 100, corrected_result.wer * 100, wer_rel,
        )
    else:
        logger.warning(
            "[%s] Phase 2 metrics not found -- skipping comparison_vs_phase2.json.", dataset_key
        )

    return corrected_result


def run_validate_4c(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    strategy: str,
    limit: Optional[int],
    force: bool,
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Run Phase 4C validation across all active datasets.

    Loads Phase 2 corrections, applies the revert strategy using CAMeL Tools
    morphological validation, and writes results to results/phase4c/.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        results_dir: Phase 4C results root.
        phase2_dir: Phase 2 results root (source of corrections).
        strategy: Revert strategy ("revert").
        limit: Max samples per dataset.
        force: If True, re-validate already-done datasets.

    Returns:
        Tuple of (all_corrected, all_comparisons) dicts.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialise CAMeL validator once
    try:
        analyzer = MorphAnalyzer(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MorphAnalyzer init failed: %s -- validation will pass through unchanged.", exc
        )
        analyzer = MorphAnalyzer({})

    validator = WordValidator(analyzer)
    if not analyzer.enabled:
        logger.warning(
            "CAMeL Tools not available or disabled. "
            "validate_correction() will pass LLM text through unchanged."
        )

    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    for ds_key in active_datasets:
        try:
            metrics_path = results_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s] Already validated -- skipping (use --force to re-validate).", ds_key
                )
                with open(metrics_path, encoding="utf-8") as f:
                    mdata = json.load(f)
                cd = mdata.get("corrected", {})
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

            phase2_records = _load_phase2_corrections_for_dataset(phase2_dir, ds_key)
            if not phase2_records:
                logger.warning("No Phase 2 records for %s -- skipping.", ds_key)
                continue

            metric_result = process_dataset_validate(
                dataset_key=ds_key,
                phase2_records=phase2_records,
                validator=validator,
                config=config,
                results_dir=results_dir,
                phase2_dir=phase2_dir,
                strategy=strategy,
                limit=limit,
            )
            all_corrected[ds_key] = metric_result

            cmp_path = results_dir / ds_key / "comparison_vs_phase2.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("Dataset %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error on %s: %s", ds_key, exc, exc_info=True)

    return all_corrected, all_comparisons


# ---------------------------------------------------------------------------
# Aggregation + Report (shared)
# ---------------------------------------------------------------------------


def aggregate_results(
    all_corrected: dict[str, MetricResult],
    config: dict,
    results_dir: Path,
    phase_label: str,
    prompt_type: str,
    prompt_version: str,
    limit: Optional[int],
    extra_meta: Optional[dict] = None,
) -> None:
    """Write combined metrics.json across all datasets."""
    meta: dict = {
        "phase": phase_label,
        "model": config.get("model", {}).get("name", ""),
        "prompt_type": prompt_type,
        "prompt_version": prompt_version,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "limit_applied": limit,
    }
    if extra_meta:
        meta.update(extra_meta)

    output = {
        "meta": meta,
        "results": {k: v.to_dict() for k, v in all_corrected.items()},
    }
    save_json(output, results_dir / "metrics.json")


def aggregate_comparisons(
    all_comparisons: dict[str, dict],
    config: dict,
    results_dir: Path,
    phase_label: str,
    limit: Optional[int],
) -> None:
    """Write combined comparison.json across all datasets."""
    if not all_comparisons:
        return
    output = {
        "meta": {
            "phase": phase_label,
            "comparison": f"{phase_label}_vs_phase2",
            "model": config.get("model", {}).get("name", ""),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
        },
        "datasets": all_comparisons,
        "note": (
            f"{phase_label.upper()} is an isolated experiment. "
            "Delta measures the contribution of linguistic knowledge over zero-shot (Phase 2). "
            "Negative delta = improvement."
        ),
    }
    save_json(output, results_dir / "comparison.json")


def generate_report(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    phase_label: str,
    phase_title: str,
    results_dir: Path,
    config: dict,
) -> None:
    """Write human-readable Markdown report to results_dir/report.md."""
    model_name = config.get("model", {}).get("name", "unknown")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append(f"# {phase_label.upper()} Report: {phase_title}")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"Model: {model_name}")
    lines.append("")

    # Comparison table vs Phase 2
    lines.append("## Results vs Phase 2 (Zero-Shot Baseline)\n")
    lines.append("> Isolated comparison. This phase vs Phase 2 only.\n")
    if all_comparisons:
        lines.append(
            "| Dataset | Phase 2 CER | This CER | Delta CER | "
            "Phase 2 WER | This WER | Delta WER |"
        )
        lines.append(
            "|---------|-------------|----------|-----------|"
            "-------------|----------|-----------|"
        )
        for ds, cmp in all_comparisons.items():
            p2 = cmp.get("phase2_baseline", {})
            p4 = cmp.get(f"{phase_label}_corrected", {})
            d  = cmp.get("delta", {})
            lines.append(
                f"| {ds} "
                f"| {p2.get('cer', 0)*100:.2f}% "
                f"| {p4.get('cer', 0)*100:.2f}% "
                f"| {d.get('cer_relative_pct', 0):+.1f}% "
                f"| {p2.get('wer', 0)*100:.2f}% "
                f"| {p4.get('wer', 0)*100:.2f}% "
                f"| {d.get('wer_relative_pct', 0):+.1f}% |"
            )
    else:
        lines.append("*Phase 2 baseline not available -- run Phase 2 first.*")
    lines.append("")

    # Absolute metrics
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

    # Error change summary (4A / 4B only — 4C does not have error_changes.json)
    any_ec = False
    for ds in all_corrected:
        ec_path = results_dir / ds / "error_changes.json"
        if ec_path.exists():
            try:
                with open(ec_path, encoding="utf-8") as f:
                    ec = json.load(f)
                s = ec.get("summary", {})
                if not any_ec:
                    lines.append("## Error Change Analysis\n")
                    any_ec = True
                lines.append(f"### {ds}\n")
                lines.append(
                    f"- Total OCR errors: {s.get('total_ocr_char_errors', 0):,}\n"
                    f"- After correction: {s.get('total_corrected_char_errors', 0):,}\n"
                    f"- Fixed: {s.get('errors_fixed', 0):,} "
                    f"({s.get('fix_rate', 0):.1f}%)\n"
                    f"- Introduced: {s.get('errors_introduced', 0):,} "
                    f"({s.get('introduction_rate', 0):.1f}%)"
                )
                lines.append("")
            except (json.JSONDecodeError, OSError):
                pass

    # Validation stats summary (4C only)
    any_vs = False
    for ds in all_corrected:
        vs_path = results_dir / ds / "validation_stats.json"
        if vs_path.exists():
            try:
                with open(vs_path, encoding="utf-8") as f:
                    vs = json.load(f)
                s = vs.get("summary", {})
                if not any_vs:
                    lines.append("## CAMeL Validation Statistics\n")
                    any_vs = True
                lines.append(f"### {ds}\n")
                lines.append(
                    f"- Total Arabic words compared: {s.get('total_arabic_words', 0):,}\n"
                    f"- Words reverted to OCR: {s.get('words_reverted', 0):,} "
                    f"({s.get('revert_rate_pct', 0):.2f}%)\n"
                    f"- Words kept (LLM): {s.get('words_kept', 0):,}\n"
                    f"- Words unchanged: {s.get('words_unchanged', 0):,}\n"
                    f"- Samples with any revert: {s.get('samples_with_reverts', 0):,}"
                )
                lines.append("")
            except (json.JSONDecodeError, OSError):
                pass

    # Key findings
    lines.append("## Key Findings\n")
    if all_corrected:
        best  = min(all_corrected.items(), key=lambda x: x[1].cer)
        worst = max(all_corrected.items(), key=lambda x: x[1].cer)
        lines.append(f"- Best CER: **{best[0]}** at {best[1].cer*100:.2f}%")
        lines.append(f"- Worst CER: **{worst[0]}** at {worst[1].cer*100:.2f}%")
    if all_comparisons:
        improving = sum(
            1 for cmp in all_comparisons.values()
            if cmp.get("delta", {}).get("cer_absolute", 0) > 0
        )
        lines.append(
            f"- {improving}/{len(all_comparisons)} datasets improved CER vs Phase 2."
        )
    lines.append("")
    lines.append(
        f"> {phase_label.upper()} is an isolated experiment comparing {phase_title} "
        f"vs zero-shot.\n"
        f"> Phase 6 will combine all linguistic sources for the full system."
    )

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


def print_summary(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    phase_label: str,
) -> None:
    """Print a final summary table to stdout."""
    print("\n" + "=" * 80)
    print(f"{phase_label.upper()} SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'This CER':>9} {'D CER':>8} {'This WER':>9} {'N':>6}")
    print("-" * 80)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_cer = cmp.get("phase2_baseline", {}).get("cer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        p2_str = f"{p2_cer*100:.2f}%" if cmp else "N/A"
        delta_str = f"{cer_rel:+.1f}%" if cmp else "N/A"
        print(
            f"{ds:<28} {p2_str:>8} {r.cer*100:>8.2f}% {delta_str:>8} "
            f"{r.wer*100:>8.2f}% {r.num_samples:>6}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Validate mode/sub-phase compatibility
    if args.sub_phase == "4c" and args.mode in ("export", "analyze"):
        print(
            f"ERROR: --sub-phase 4c does not support --mode {args.mode}. "
            "Use --mode validate for Phase 4C.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.sub_phase in ("4a", "4b") and args.mode == "validate":
        print(
            f"ERROR: --sub-phase {args.sub_phase} does not support --mode validate. "
            "Use --mode export or --mode analyze.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine results directory
    phase_dir_map = {
        "4a": Path("results/phase4a"),
        "4b": Path("results/phase4b"),
        "4c": Path("results/phase4c"),
    }
    results_dir = phase_dir_map[args.sub_phase]

    phase_label_map = {"4a": "phase4a", "4b": "phase4b", "4c": "phase4c"}
    phase_label = phase_label_map[args.sub_phase]

    phase_title_map = {
        "4a": "Rule-Augmented Prompting",
        "4b": "Few-Shot Prompting (QALB)",
        "4c": "CAMeL Tools Validation",
    }
    phase_title = phase_title_map[args.sub_phase]

    setup_logging(results_dir, phase_label)
    logger.info("%s  (mode=%s)", phase_title, args.mode)
    logger.info("Results dir: %s", results_dir)
    logger.info("Phase 2 dir: %s", args.phase2_dir)

    config = load_config(args.config)
    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")
    active_datasets = resolve_datasets(config, args.datasets)
    logger.info("Datasets to process: %s", active_datasets)

    # ------------------------------------------------------------------
    # PHASE 4A
    # ------------------------------------------------------------------
    if args.sub_phase == "4a":
        phase4a_cfg = config.get("phase4", {}).get("rules", {})
        categories = args.rule_categories or phase4a_cfg.get("categories")
        n_rules    = args.n_rules or phase4a_cfg.get("n_rules")

        if args.mode == "export":
            run_export_4a(
                config=config,
                active_datasets=active_datasets,
                results_dir=results_dir,
                categories=categories,
                n_rules=n_rules,
                limit=limit,
                force=args.force,
            )
            return

        # analyze
        analyze_errors = not args.no_error_analysis
        all_corrected, all_comparisons = run_analyze_4a(
            config=config,
            active_datasets=active_datasets,
            results_dir=results_dir,
            phase2_dir=args.phase2_dir,
            limit=limit,
            force=args.force,
            analyze_errors=analyze_errors,
            categories=categories,
            n_rules=n_rules,
        )

        if not all_corrected:
            logger.error("No datasets successfully processed.")
            sys.exit(1)

        builder = PromptBuilder()
        aggregate_results(
            all_corrected, config, results_dir,
            phase_label, "rule_augmented", builder.rules_prompt_version,
            limit, extra_meta={"rule_categories": categories, "n_rules": n_rules},
        )
        aggregate_comparisons(all_comparisons, config, results_dir, phase_label, limit)
        generate_report(
            all_corrected, all_comparisons, phase_label, phase_title, results_dir, config
        )
        print_summary(all_corrected, all_comparisons, phase_label)
        logger.info("Phase 4A complete. Results in: %s", results_dir)

    # ------------------------------------------------------------------
    # PHASE 4B
    # ------------------------------------------------------------------
    elif args.sub_phase == "4b":
        phase4b_cfg = config.get("phase4", {}).get("few_shot", {})
        num_examples = args.num_examples or phase4b_cfg.get("num_examples", 5)
        selection    = args.selection or phase4b_cfg.get("selection", "diverse")
        years        = args.qalb_years or phase4b_cfg.get("years", ["2014"])

        if args.mode == "export":
            run_export_4b(
                config=config,
                active_datasets=active_datasets,
                results_dir=results_dir,
                num_examples=num_examples,
                selection=selection,
                years=years,
                limit=limit,
                force=args.force,
            )
            return

        # analyze
        analyze_errors = not args.no_error_analysis
        all_corrected, all_comparisons = run_analyze_4b(
            config=config,
            active_datasets=active_datasets,
            results_dir=results_dir,
            phase2_dir=args.phase2_dir,
            limit=limit,
            force=args.force,
            analyze_errors=analyze_errors,
            num_examples=num_examples,
            selection=selection,
        )

        if not all_corrected:
            logger.error("No datasets successfully processed.")
            sys.exit(1)

        builder = PromptBuilder()
        aggregate_results(
            all_corrected, config, results_dir,
            phase_label, "few_shot", builder.few_shot_prompt_version,
            limit, extra_meta={"num_examples": num_examples, "selection": selection},
        )
        aggregate_comparisons(all_comparisons, config, results_dir, phase_label, limit)
        generate_report(
            all_corrected, all_comparisons, phase_label, phase_title, results_dir, config
        )
        print_summary(all_corrected, all_comparisons, phase_label)
        logger.info("Phase 4B complete. Results in: %s", results_dir)

    # ------------------------------------------------------------------
    # PHASE 4C
    # ------------------------------------------------------------------
    elif args.sub_phase == "4c":
        phase4c_cfg = config.get("phase4", {}).get("camel_validation", {})
        strategy = args.strategy or phase4c_cfg.get("strategy", "revert")

        all_corrected, all_comparisons = run_validate_4c(
            config=config,
            active_datasets=active_datasets,
            results_dir=results_dir,
            phase2_dir=args.phase2_dir,
            strategy=strategy,
            limit=limit,
            force=args.force,
        )

        if not all_corrected:
            logger.error("No datasets successfully processed.")
            sys.exit(1)

        aggregate_results(
            all_corrected, config, results_dir,
            phase_label, "camel_validation", "p4cv1",
            limit, extra_meta={"strategy": strategy},
        )
        aggregate_comparisons(all_comparisons, config, results_dir, phase_label, limit)
        generate_report(
            all_corrected, all_comparisons, phase_label, phase_title, results_dir, config
        )
        print_summary(all_corrected, all_comparisons, phase_label)
        logger.info("Phase 4C complete. Results in: %s", results_dir)


if __name__ == "__main__":
    main()
