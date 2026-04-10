#!/usr/bin/env python3
"""Phase 5: CAMeL Morphological Validation.

Applies morphological revert strategy to Phase 2 LLM corrections using
CAMeL Tools. Also loads known-overcorrection pairs from training artifacts
and reverts those BEFORE morphological checks.

Pipeline (entirely local, no remote inference needed):
  1. LOCAL:  python pipelines/run_phase5.py --mode validate
     -> reads Phase 2 corrections.jsonl, applies revert strategy,
        writes results/phase5/{dataset}/corrections.jsonl + metrics.json

Typical workflow
----------------
Prerequisite: Phase 2 corrections must exist at results/phase2/{dataset}/corrections.jsonl
Prerequisite (optional): Training artifacts for known overcorrections at
  results/phase2-training/analysis/word_pairs_llm_failures.txt

1. LOCAL:  python pipelines/run_phase5.py --mode validate
2. LOCAL:  python pipelines/run_phase5.py --mode validate --datasets KHATT-train
3. LOCAL:  python pipelines/run_phase5.py --mode validate --force

Usage
-----
    python pipelines/run_phase5.py --mode validate
    python pipelines/run_phase5.py --mode validate --datasets KHATT-train
    python pipelines/run_phase5.py --mode validate --limit 100
    python pipelines/run_phase5.py --mode validate --force
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
# NOTE: RulesLoader/QALBLoader removed in phase refactoring.

# once Phase 5 (CAMeL) is extracted to its own pipeline.
from src.analysis.metrics import MetricResult, calculate_metrics, calculate_metrics_dual
from src.analysis.report_formatter import write_corrections_report
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
from pipelines._utils import (
    resolve_datasets, load_sample_list, compute_group_aggregates,
    split_runaway_samples, DEFAULT_RUNAWAY_RATIO_THRESHOLD,
    load_phase2_full_metrics, pick_phase2_variant, _pick_corrected_key,
)

logger = logging.getLogger(__name__)

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
# PHASE 5 -- CAMeL Tools Validation
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
    known_overcorrections: set[tuple[str, str]] | None = None,
) -> MetricResult:
    """Apply CAMeL revert strategy to Phase 2 corrections for one dataset.

    Reads Phase 2 {llm_text, ocr_text}, applies revert strategy, writes
    corrected output, computes metrics, and saves all result files.

    Args:
        dataset_key: e.g. "KHATT-train".
        phase2_records: Raw dicts from Phase 2 corrections.jsonl.
        validator: Initialised WordValidator instance.
        config: Parsed config dict.
        results_dir: Phase 5 results root.
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
    logger.info("Phase 5 validate: %s  (%d samples, strategy=%s)", dataset_key, n, strategy)

    corrected_samples: list[CorrectedSample] = []
    revert_results: list[TextCorrectionResult] = []

    out_corrections_path = out_dir / "corrections.jsonl"
    with open(out_corrections_path, "w", encoding="utf-8") as out_f:
        for r in tqdm(phase2_records, desc=f"  Validating {dataset_key}", unit="sample"):
            llm_text = r.get("corrected_text", r["ocr_text"])
            ocr_text = r.get("ocr_text", "")
            gt_text  = r.get("gt_text", "")

            revert_result = validator.validate_correction(
                llm_text, ocr_text, strategy=strategy,
                known_overcorrections=known_overcorrections,
            )
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

    logger.info("Phase 5 corrections written to %s", out_corrections_path)

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
            "phase5", dataset_key, n, config, limit,
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
    # CER/WER on post-validation corrected texts (with runaway splitting)
    # ------------------------------------------------------------------
    eval_cfg = config.get("evaluation", {})
    exclude_runaway = eval_cfg.get("exclude_runaway", False)
    threshold = eval_cfg.get("runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD)

    normal_samples, runaway_samples, data_quality = split_runaway_samples(
        corrected_samples, threshold=threshold,
    )
    if runaway_samples:
        logger.info("[%s] %s", dataset_key, data_quality["description"])

    logger.info("[%s] Calculating Phase 5 OCR baseline + corrected CER/WER ...", dataset_key)

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

    metrics_json = {
        "meta": make_meta(
            "phase5", dataset_key, n, config, limit,
            extra={
                "strategy":         strategy,
                "prompt_type":      "camel_validation",
                "total_reverted":   total_reverted,
                "avg_revert_rate":  round(avg_revert_rate, 4),
                "primary_variant":  primary_source,
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
        "[%s] Phase 5 Primary (%s): OCR CER=%.2f%% -> LLM CER=%.2f%%  |  no-diac: %.2f%% -> %.2f%%",
        dataset_key, primary_source,
        (ocr_all if not exclude_runaway else ocr_normal).cer * 100,
        primary.cer * 100,
        (ocr_all_nd if not exclude_runaway else ocr_normal_nd).cer * 100,
        primary_nd.cer * 100,
    )

    # ------------------------------------------------------------------
    # Comparison vs Phase 2
    # ------------------------------------------------------------------
    p2_full = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_full is not None:
        p2_corr, p2_nd, p2_src = pick_phase2_variant(p2_full, exclude_runaway)
        p2_cer = float(p2_corr.get("cer", 0.0))
        p2_wer = float(p2_corr.get("wer", 0.0))
        cer_delta_abs = p2_cer - primary.cer
        wer_delta_abs = p2_wer - primary.wer
        cer_rel = (cer_delta_abs / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta_abs / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(
                "phase5", dataset_key, n, config, limit,
                extra={"comparison": "phase5_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
                "variant": p2_src,
            },
            "phase5_corrected": {
                "cer": round(primary.cer, 6),
                "wer": round(primary.wer, 6),
                "variant": primary_source,
            },
            "delta": {
                "cer_absolute":     round(cer_delta_abs, 6),
                "wer_absolute":     round(wer_delta_abs, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            **(_nd_comparison_block(p2_nd, primary_nd, "phase5")),
            "interpretation": (
                f"CER {'reduced' if cer_delta_abs >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {primary.cer*100:.2f}%). "
                f"WER {'reduced' if wer_delta_abs >= 0 else 'increased'} by "
                f"{abs(wer_rel):.1f}% "
                f"({p2_wer*100:.2f}% -> {primary.wer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s] Phase2->Phase5 CER: %.2f%% -> %.2f%% (%+.1f%%) | "
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
            "[%s] Phase 2 metrics not found -- skipping comparison_vs_phase2.json.", dataset_key
        )

    return primary


def run_validate(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    strategy: str,
    limit: Optional[int],
    force: bool,
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Run Phase 5 validation across all active datasets.

    Loads Phase 2 corrections, applies the revert strategy using CAMeL Tools
    morphological validation, and writes results to results/phase5/.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        results_dir: Phase 5 results root.
        phase2_dir: Phase 2 results root (source of corrections).
        strategy: Revert strategy ("revert").
        limit: Max samples per dataset.
        force: If True, re-validate already-done datasets.

    Returns:
        Tuple of (all_corrected, all_comparisons) dicts.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialise CAMeL validator once (match Phase 1 pattern)
    camel_cfg = config.get("camel", {})
    morph_cfg = camel_cfg.get("morphology", {})
    analyzer = MorphAnalyzer(
        db=morph_cfg.get("db", "calima-msa-r13"),
        cache_size=morph_cfg.get("cache_size", 10_000),
        enabled=camel_cfg.get("enabled", True),
    )

    if not analyzer.enabled:
        logger.error(
            "Phase 5 requires camel_tools but it is not available.\n"
            "Install: pip install camel-tools && camel_data -i morphology-db-msa-r13\n"
            "Or disable with: camel.enabled: false in config.yaml"
        )
        sys.exit(1)

    validator = WordValidator(analyzer)

    # Load known overcorrections (INTRODUCED pairs from training analysis)
    known_overcorrections: set[tuple[str, str]] | None = None
    phase5_cfg = config.get("phase5", {})
    if phase5_cfg.get("use_known_overcorrections", True):
        overcorrections_path_str = phase5_cfg.get(
            "overcorrections_path",
            "results/phase2-training/analysis/word_pairs_llm_failures.txt",
        )
        overcorrections_path = _PROJECT_ROOT / overcorrections_path_str
        if overcorrections_path.exists():
            from src.data.knowledge_base import load_introduced_word_pairs
            introduced = load_introduced_word_pairs(overcorrections_path)
            if introduced:
                known_overcorrections = set(introduced)
                logger.info(
                    "Loaded %d known overcorrection pairs from %s",
                    len(known_overcorrections), overcorrections_path,
                )
        else:
            logger.warning(
                "Known overcorrections file not found: %s — "
                "proceeding with morphological validation only.",
                overcorrections_path,
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
                known_overcorrections=known_overcorrections,
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
        "results_no_diacritics": _load_nd_results(all_corrected, results_dir),
    }
    # Macro-averaged aggregates by dataset group (PATS-A01 / KHATT).
    output["group_aggregates"] = compute_group_aggregates(output["results"])
    nd = output.get("results_no_diacritics", {})
    if nd:
        output["group_aggregates_no_diacritics"] = compute_group_aggregates(nd)
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

    nd_results = _load_nd_results(all_corrected, results_dir) if results_dir else {}

    # Comparison table vs Phase 2 — WITH DIACRITICS
    lines.append("## Results vs Phase 2 (Zero-Shot Baseline)\n")
    lines.append("> Isolated comparison. This phase vs Phase 2 only.\n")
    if all_comparisons:
        lines.append("### With Diacritics\n")
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
        lines.append("")

        # Comparison table — NO DIACRITICS
        lines.append("### No Diacritics\n")
        lines.append(
            "| Dataset | Phase 2 CER | This CER | Delta CER | "
            "Phase 2 WER | This WER | Delta WER |"
        )
        lines.append(
            "|---------|-------------|----------|-----------|"
            "-------------|----------|-----------|"
        )
        for ds, cmp in all_comparisons.items():
            p2_nd = cmp.get("phase2_baseline_no_diacritics", {})
            p4_nd = cmp.get(f"{phase_label}_corrected_no_diacritics", {})
            d_nd  = cmp.get("delta_no_diacritics", {})
            lines.append(
                f"| {ds} "
                f"| {p2_nd.get('cer', 0)*100:.2f}% "
                f"| {p4_nd.get('cer', 0)*100:.2f}% "
                f"| {d_nd.get('cer_relative_pct', 0):+.1f}% "
                f"| {p2_nd.get('wer', 0)*100:.2f}% "
                f"| {p4_nd.get('wer', 0)*100:.2f}% "
                f"| {d_nd.get('wer_relative_pct', 0):+.1f}% |"
            )
    else:
        lines.append("*Phase 2 baseline not available -- run Phase 2 first.*")
    lines.append("")

    # Absolute metrics — WITH DIACRITICS
    lines.append("## Post-Correction Metrics\n")
    lines.append("### With Diacritics\n")
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

    # Absolute metrics — NO DIACRITICS
    if nd_results:
        lines.append("### No Diacritics\n")
        lines.append("| Dataset | CER | WER |")
        lines.append("|---------|-----|-----|")
        for ds in all_corrected:
            nd = nd_results.get(ds, {})
            nd_cer = f"{nd.get('cer', 0)*100:.2f}%" if nd else "N/A"
            nd_wer = f"{nd.get('wer', 0)*100:.2f}%" if nd else "N/A"
            lines.append(f"| {ds} | {nd_cer} | {nd_wer} |")
        lines.append("")

    # Error change summary (not used in Phase 5 -- error_changes.json)
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

    # Validation stats summary
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
        f"> Phase 6 will combine all phases for the final system."
    )

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


def print_summary(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    phase_label: str,
    results_dir: Optional[Path] = None,
) -> None:
    """Print a final summary table to stdout (with-diacritics and no-diacritics side-by-side)."""
    nd_results = _load_nd_results(all_corrected, results_dir) if results_dir else {}

    sep = "=" * 90
    row_sep = "  " + "-" * 82

    # Table 1: WITH DIACRITICS
    print("\n" + sep)
    print(f"{phase_label.upper()} SUMMARY  [WITH DIACRITICS]")
    print(sep)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'Px CER':>8} {'D(CER)':>8} {'P2 WER':>8} {'Px WER':>8} {'D(WER)':>8} {'N':>6}")
    print(row_sep)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_cer = cmp.get("phase2_baseline", {}).get("cer", 0.0)
        p2_wer = cmp.get("phase2_baseline", {}).get("wer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        wer_rel = cmp.get("delta", {}).get("wer_relative_pct", 0.0)
        p2_cer_str = f"{p2_cer*100:.2f}%" if cmp else "N/A"
        p2_wer_str = f"{p2_wer*100:.2f}%" if cmp else "N/A"
        d_cer_str  = f"{cer_rel:+.1f}%"   if cmp else "N/A"
        d_wer_str  = f"{wer_rel:+.1f}%"   if cmp else "N/A"
        print(
            f"{ds:<28} {p2_cer_str:>8} {r.cer*100:>7.2f}% {d_cer_str:>8} "
            f"{p2_wer_str:>8} {r.wer*100:>7.2f}% {d_wer_str:>8} {r.num_samples:>6}"
        )
    print(sep)

    # Table 2: NO DIACRITICS
    print(f"\n{phase_label.upper()} SUMMARY  [NO DIACRITICS]")
    print(sep)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'Px CER':>8} {'D(CER)':>8} {'P2 WER':>8} {'Px WER':>8} {'D(WER)':>8} {'N':>6}")
    print(row_sep)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        nd = nd_results.get(ds, {})
        p2_nd_cer = cmp.get("phase2_baseline_no_diacritics", {}).get("cer", 0.0)
        p2_nd_wer = cmp.get("phase2_baseline_no_diacritics", {}).get("wer", 0.0)
        nd_cer_rel = cmp.get("delta_no_diacritics", {}).get("cer_relative_pct", 0.0)
        nd_wer_rel = cmp.get("delta_no_diacritics", {}).get("wer_relative_pct", 0.0)
        p2_nd_cer_str = f"{p2_nd_cer*100:.2f}%" if cmp else "N/A"
        p2_nd_wer_str = f"{p2_nd_wer*100:.2f}%" if cmp else "N/A"
        nd_d_cer_str  = f"{nd_cer_rel:+.1f}%"   if cmp else "N/A"
        nd_d_wer_str  = f"{nd_wer_rel:+.1f}%"   if cmp else "N/A"
        nd_cer_val = nd.get("cer", 0.0) * 100 if nd else 0.0
        nd_wer_val = nd.get("wer", 0.0) * 100 if nd else 0.0
        nd_cer_cur = f"{nd_cer_val:.2f}%" if nd else "N/A"
        nd_wer_cur = f"{nd_wer_val:.2f}%" if nd else "N/A"
        print(
            f"{ds:<28} {p2_nd_cer_str:>8} {nd_cer_cur:>8} {nd_d_cer_str:>8} "
            f"{p2_nd_wer_str:>8} {nd_wer_cur:>8} {nd_d_wer_str:>8} {r.num_samples:>6}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: CAMeL Morphological Validation for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["validate"],
        help="validate -> apply CAMeL revert strategy to Phase 2 corrections (entirely local)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="revert",
        choices=["revert"],
        help="Validation strategy (default: revert).",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("results/phase2"),
        dest="phase2_dir",
        help="Phase 2 results directory (source of corrections).",
    )
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
        "--sample-list",
        type=Path,
        default=None,
        dest="sample_list",
        help="Path to test_samples.json to filter samples.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-validate datasets that already have results.",
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
        default=Path("results/phase5"),
        dest="results_dir",
        help="Phase 5 results root directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    setup_logging(results_dir, "phase5")
    logger.info("Phase 5: CAMeL Morphological Validation  (mode=%s)", args.mode)
    logger.info("Results dir: %s", results_dir)
    logger.info("Phase 2 dir: %s", args.phase2_dir)

    config = load_config(args.config)
    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")
    active_datasets = resolve_datasets(config, args.datasets)

    sample_ids = None
    if args.sample_list:
        sample_ids, sl_datasets = load_sample_list(args.sample_list)
        if not args.datasets:
            active_datasets = sl_datasets
        logger.info(
            "Sample list loaded: %d sample IDs from %s", len(sample_ids), args.sample_list
        )
    _ = sample_ids  # Phase 5 operates on Phase 2 records directly

    logger.info("Datasets to process: %s", active_datasets)

    phase5_cfg = config.get("phase5", {})
    strategy = args.strategy or phase5_cfg.get("strategy", "revert")

    all_corrected, all_comparisons = run_validate(
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
        "phase5", "camel_validation", "p5v1",
        limit, extra_meta={"strategy": strategy},
    )
    aggregate_comparisons(all_comparisons, config, results_dir, "phase5", limit)
    generate_report(
        all_corrected, all_comparisons, "phase5",
        "CAMeL Morphological Validation", results_dir, config,
    )
    print_summary(all_corrected, all_comparisons, "phase5", results_dir)
    write_corrections_report(
        corrections_path=results_dir,
        output_path=results_dir / "sample_report.txt",
        title="Phase 5 -- CAMeL Morphological Validation",
    )
    logger.info("Phase 5 complete. Results in: %s", results_dir)


if __name__ == "__main__":
    main()
