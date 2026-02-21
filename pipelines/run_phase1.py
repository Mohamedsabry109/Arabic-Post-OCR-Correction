#!/usr/bin/env python3
"""Phase 1: Baseline & Error Taxonomy.

Quantifies Qaari OCR error rates and builds character confusion matrices
and error taxonomies for PATS-A01 and KHATT datasets. No LLM calls.

Usage:
    python pipelines/run_phase1.py
    python pipelines/run_phase1.py --limit 50
    python pipelines/run_phase1.py --dataset KHATT-train
    python pipelines/run_phase1.py --no-camel
    python pipelines/run_phase1.py --config configs/config.yaml
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

# Ensure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError, OCRSample
from src.analysis.metrics import calculate_metrics, calculate_metrics_split, MetricResult
from src.analysis.error_analyzer import ErrorAnalyzer, SampleError
from src.linguistic.morphology import MorphAnalyzer
from src.linguistic.validator import WordValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Baseline & Error Taxonomy for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick testing).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["PATS-A01-Akhbar", "PATS-A01-Andalus", "KHATT-train", "KHATT-validation"],
        help="Run only one specific dataset.",
    )
    parser.add_argument(
        "--no-camel",
        action="store_true",
        default=False,
        help="Skip morphological analysis (useful if CAMeL Tools is not installed).",
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
        default=Path("results/phase1"),
        help="Output directory for Phase 1 results.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to both console and a log file.

    On Windows the console may use a narrow codepage (cp1252) that cannot
    encode Arabic.  We wrap stdout with UTF-8 so Arabic characters in log
    messages are replaced with '?' rather than crashing.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase1.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Console handler: encode Arabic as '?' on narrow Windows codepages
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
    """Return current git commit hash, or 'unknown' on failure."""
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
    limit: int | None,
) -> dict:
    """Build the standard metadata block for all output JSON files."""
    return {
        "phase": "phase1",
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "num_samples": num_samples,
        "limit_applied": limit,
        "model": config.get("model", {}).get("name", "N/A (Phase 1 has no LLM)"),
    }


def save_json(data: dict, path: Path) -> None:
    """Write *data* as indented JSON to *path*, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def process_dataset(
    dataset_key: str,
    samples: list[OCRSample],
    config: dict,
    results_dir: Path,
    limit: int | None,
    validator: WordValidator | None,
) -> tuple[MetricResult, MetricResult, dict]:
    """Run all Phase 1 analyses for one dataset.

    Args:
        dataset_key: E.g. "KHATT-train".
        samples: Loaded and aligned OCRSample objects.
        config: Parsed config dict.
        results_dir: Phase 1 results root directory.
        limit: Sample limit applied (for metadata).
        validator: WordValidator (or None if CAMeL disabled).

    Returns:
        MetricResult for this dataset.
    """
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    phase1_cfg = config.get("phase1", {})

    logger.info("=" * 60)
    logger.info("Processing dataset: %s  (%d samples)", dataset_key, len(samples))

    # ------------------------------------------------------------------
    # Step 1: CER / WER metrics (all samples + normal-only split)
    # ------------------------------------------------------------------
    logger.info("[%s] Calculating CER / WER ...", dataset_key)
    metric_result, normal_result, data_quality = calculate_metrics_split(
        samples, dataset_name=dataset_key
    )

    metrics_json = {
        "meta": make_meta(dataset_key, len(samples), config, limit),
        "all_samples": metric_result.to_dict(),
        "normal_samples_only": normal_result.to_dict(),
        "data_quality": data_quality,
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] CER=%.2f%% (all) / %.2f%% (normal-only)  |  "
        "WER=%.2f%% (all) / %.2f%% (normal-only)  |  "
        "Runaway: %d/%d samples (%.1f%%)",
        dataset_key,
        metric_result.cer * 100, normal_result.cer * 100,
        metric_result.wer * 100, normal_result.wer * 100,
        data_quality["runaway_samples"], data_quality["total_samples"],
        data_quality["runaway_percentage"],
    )

    # ------------------------------------------------------------------
    # Step 2: Error analysis (confusion matrix + taxonomy)
    # ------------------------------------------------------------------
    logger.info("[%s] Running error analysis …", dataset_key)
    analyzer = ErrorAnalyzer()
    all_errors: list[SampleError] = []

    for sample in tqdm(samples, desc=f"  Analysing {dataset_key}", unit="sample"):
        try:
            all_errors.append(analyzer.analyse_sample(sample))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error analysing sample %s: %s", sample.sample_id, exc)

    # Confusion matrix
    min_cnt = phase1_cfg.get("min_confusion_count", 2)
    confusion = analyzer.build_confusion_matrix(all_errors, dataset=dataset_key, min_count=min_cnt)
    confusion["meta"].update(make_meta(dataset_key, len(samples), config, limit))
    save_json(confusion, out_dir / "confusion_matrix.json")

    # Error taxonomy
    taxonomy = analyzer.build_taxonomy(all_errors, dataset=dataset_key)
    taxonomy["meta"].update(make_meta(dataset_key, len(samples), config, limit))
    save_json(taxonomy, out_dir / "error_taxonomy.json")

    top_n = phase1_cfg.get("top_confusions_n", 20)
    top_conf = analyzer.get_top_confusions(confusion, n=top_n)
    logger.info("[%s] Top 3 confusions: %s", dataset_key, top_conf[:3])

    # Error examples
    save_json(
        {
            "meta": make_meta(dataset_key, len(samples), config, limit),
            "examples": _collect_error_examples(all_errors, max_per_type=5),
        },
        out_dir / "error_examples.json",
    )

    # ------------------------------------------------------------------
    # Step 3: Morphological analysis (optional)
    # ------------------------------------------------------------------
    if validator is not None:
        logger.info("[%s] Running morphological analysis …", dataset_key)
        morph_data = run_morphological_analysis(samples, validator, dataset_key)
        morph_data["meta"] = make_meta(dataset_key, len(samples), config, limit)
        morph_data["meta"]["camel_available"] = True
        morph_data["meta"]["camel_db"] = config.get("camel", {}).get("morphology", {}).get("db", "?")
        save_json(morph_data, out_dir / "morphological_analysis.json")
    else:
        save_json(
            {
                "meta": {
                    **make_meta(dataset_key, len(samples), config, limit),
                    "camel_available": False,
                    "note": "CAMeL Tools not installed or --no-camel flag used.",
                }
            },
            out_dir / "morphological_analysis.json",
        )

    return metric_result, normal_result, data_quality


def run_morphological_analysis(
    samples: list[OCRSample],
    validator: WordValidator,
    dataset_name: str,
) -> dict:
    """Compute morphological validity statistics for OCR outputs vs GT.

    For each sample, tokenises both gt_text and ocr_text, validates each
    Arabic token, and classifies error words into:
    - non_word_error: GT valid but OCR invalid (obvious error)
    - valid_but_wrong: Both valid but different (subtle error)
    - both_invalid: Both morphologically unknown (possibly noisy GT)

    Args:
        samples: List of OCRSample objects.
        validator: Initialised WordValidator.
        dataset_name: Label for output metadata.

    Returns:
        Dict with gt_validity, ocr_validity, and error_breakdown sections.
    """
    gt_total = gt_valid = 0
    ocr_total = ocr_valid = 0
    non_word_errors = valid_but_wrong = both_invalid = 0

    for sample in tqdm(samples, desc="  Morphological analysis", unit="sample"):
        gt_results = validator.validate_text(sample.gt_text)
        ocr_results = validator.validate_text(sample.ocr_text)

        gt_total += len(gt_results)
        gt_valid += sum(1 for r in gt_results if r.is_valid)
        ocr_total += len(ocr_results)
        ocr_valid += sum(1 for r in ocr_results if r.is_valid)

        # Word-level comparison: zip GT and OCR tokens that differ
        from src.data.text_utils import tokenise_arabic
        gt_words = tokenise_arabic(sample.gt_text)
        ocr_words = tokenise_arabic(sample.ocr_text)

        # Only compare differing word pairs (simple positional approach)
        for i, (gw, ow) in enumerate(zip(gt_words, ocr_words)):
            if gw == ow:
                continue
            from src.data.text_utils import is_arabic_word
            if not is_arabic_word(gw) or not is_arabic_word(ow):
                continue
            gw_valid = validator.validate_word(gw).is_valid
            ow_valid = validator.validate_word(ow).is_valid

            if gw_valid and not ow_valid:
                non_word_errors += 1
            elif gw_valid and ow_valid:
                valid_but_wrong += 1
            elif not gw_valid and not ow_valid:
                both_invalid += 1

    total_word_errors = non_word_errors + valid_but_wrong + both_invalid

    def _pct(n: int, total: int) -> float:
        return round(n / total * 100, 2) if total > 0 else 0.0

    return {
        "gt_validity": {
            "total_words": gt_total,
            "valid_words": gt_valid,
            "valid_rate": round(gt_valid / gt_total, 4) if gt_total else 0.0,
        },
        "ocr_validity": {
            "total_words": ocr_total,
            "valid_words": ocr_valid,
            "valid_rate": round(ocr_valid / ocr_total, 4) if ocr_total else 0.0,
        },
        "error_breakdown": {
            "non_word_errors": {
                "count": non_word_errors,
                "percentage_of_total_errors": _pct(non_word_errors, total_word_errors),
                "description": "GT word valid → OCR word invalid (obvious/detectable error)",
            },
            "valid_but_wrong": {
                "count": valid_but_wrong,
                "percentage_of_total_errors": _pct(valid_but_wrong, total_word_errors),
                "description": "GT word valid → OCR word valid but different (subtle error)",
            },
            "both_invalid": {
                "count": both_invalid,
                "percentage_of_total_errors": _pct(both_invalid, total_word_errors),
                "description": "Both GT and OCR word are morphologically unknown",
            },
        },
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(
    all_results: dict[str, MetricResult],
    normal_results: dict[str, MetricResult],
    quality_stats: dict[str, dict],
    config: dict,
    results_dir: Path,
    limit: int | None,
) -> None:
    """Build the combined baseline_metrics.json from all per-dataset results."""
    output = {
        "meta": {
            "phase": "phase1",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
            "limit_applied": limit,
            "description": (
                "Qaari OCR baseline CER/WER. "
                "'all' includes runaway samples (Qaari repetition bug). "
                "'normal_only' excludes them (OCR/GT ratio <= 5)."
            ),
        },
        "results_all_samples": {k: v.to_dict() for k, v in all_results.items()},
        "results_normal_only": {k: v.to_dict() for k, v in normal_results.items()},
        "data_quality": quality_stats,
    }

    save_json(output, results_dir / "baseline_metrics.json")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY -- Qaari OCR Baseline")
    print("=" * 80)
    print(f"{'Dataset':<28} {'CER(all)':>10} {'CER(norm)':>10} {'WER(norm)':>10} {'Runaway%':>9} {'N':>6}")
    print("-" * 80)
    for ds in all_results:
        r_all = all_results[ds]
        r_norm = normal_results.get(ds)
        dq = quality_stats.get(ds, {})
        norm_cer = f"{r_norm.cer*100:.2f}%" if r_norm else "N/A"
        norm_wer = f"{r_norm.wer*100:.2f}%" if r_norm else "N/A"
        runaway_pct = f"{dq.get('runaway_percentage', 0):.1f}%" if dq else "N/A"
        print(
            f"{ds:<28} {r_all.cer*100:>9.2f}% {norm_cer:>10} {norm_wer:>10} "
            f"{runaway_pct:>9} {r_all.num_samples:>6}"
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    all_results: dict[str, MetricResult],
    results_dir: Path,
) -> None:
    """Write a human-readable Markdown report to results_dir/report.md."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Phase 1 Report: Baseline & Error Taxonomy")
    lines.append(f"\nGenerated: {now}")
    lines.append("\n## Summary\n")
    lines.append(
        "> **Note**: 'Normal CER/WER' excludes samples where Qaari's OCR output is "
        ">5× longer than GT (runaway repetition bug). Both numbers are reported.\n"
    )
    lines.append("| Dataset | CER (all) | CER (normal) | WER (normal) | Runaway% | Samples |")
    lines.append("|---------|-----------|--------------|--------------|----------|---------|")

    # Load per-dataset quality stats from saved metrics.json files
    for ds, r in all_results.items():
        metrics_path = results_dir / ds / "metrics.json"
        norm_cer = norm_wer = runaway_pct = "N/A"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                mj = json.load(f)
            nr = mj.get("normal_samples_only", {})
            dq = mj.get("data_quality", {})
            norm_cer = f"{nr.get('cer', 0)*100:.2f}%" if nr else "N/A"
            norm_wer = f"{nr.get('wer', 0)*100:.2f}%" if nr else "N/A"
            runaway_pct = f"{dq.get('runaway_percentage', 0):.1f}%" if dq else "N/A"
        lines.append(
            f"| {ds} | {r.cer*100:.2f}% | {norm_cer} | {norm_wer} | {runaway_pct} | {r.num_samples:,} |"
        )

    lines.append("\n## Metric Details\n")
    for ds, r in all_results.items():
        lines.append(f"### {ds}\n")
        lines.append(f"- **CER**: {r.cer*100:.2f}% (std: {r.cer_std*100:.2f}%, "
                     f"median: {r.cer_median*100:.2f}%, p95: {r.cer_p95*100:.2f}%)")
        lines.append(f"- **WER**: {r.wer*100:.2f}% (std: {r.wer_std*100:.2f}%, "
                     f"median: {r.wer_median*100:.2f}%, p95: {r.wer_p95*100:.2f}%)")
        lines.append(f"- Total GT characters: {r.num_chars_ref:,}")
        lines.append(f"- Total GT words: {r.num_words_ref:,}\n")

    # Load confusion matrix summaries for each dataset
    lines.append("\n## Top Character Confusions\n")
    for ds in all_results:
        cm_path = results_dir / ds / "confusion_matrix.json"
        if cm_path.exists():
            with open(cm_path, encoding="utf-8") as f:
                cm = json.load(f)
            top = cm.get("top_20", [])[:10]
            if top:
                lines.append(f"### {ds}\n")
                lines.append("| GT Char | OCR Char | Count | Probability |")
                lines.append("|---------|----------|-------|-------------|")
                for entry in top:
                    lines.append(
                        f"| {entry['gt']} | {entry['ocr']} "
                        f"| {entry['count']} | {entry['probability']*100:.1f}% |"
                    )
                lines.append("")

    # Error type distribution
    lines.append("\n## Error Type Distribution\n")
    for ds in all_results:
        tax_path = results_dir / ds / "error_taxonomy.json"
        if tax_path.exists():
            with open(tax_path, encoding="utf-8") as f:
                tax = json.load(f)
            by_type = tax.get("by_type", {})
            if by_type:
                lines.append(f"### {ds}\n")
                lines.append("| Error Type | Count | Percentage |")
                lines.append("|-----------|-------|------------|")
                for etype, data in sorted(
                    by_type.items(), key=lambda x: -x[1]["count"]
                ):
                    if data["count"] > 0:
                        lines.append(
                            f"| {etype} | {data['count']:,} | {data['percentage']:.1f}% |"
                        )
                lines.append("")

    # Morphological analysis
    lines.append("\n## Morphological Validity\n")
    for ds in all_results:
        morph_path = results_dir / ds / "morphological_analysis.json"
        if morph_path.exists():
            with open(morph_path, encoding="utf-8") as f:
                morph = json.load(f)
            meta = morph.get("meta", {})
            if not meta.get("camel_available", False):
                lines.append(f"- **{ds}**: CAMeL Tools not available.\n")
                continue
            gt_v = morph.get("gt_validity", {})
            ocr_v = morph.get("ocr_validity", {})
            eb = morph.get("error_breakdown", {})
            lines.append(f"### {ds}\n")
            lines.append(
                f"- GT word validity: {gt_v.get('valid_rate', 0)*100:.1f}% "
                f"({gt_v.get('valid_words', 0):,} / {gt_v.get('total_words', 0):,})"
            )
            lines.append(
                f"- OCR word validity: {ocr_v.get('valid_rate', 0)*100:.1f}% "
                f"({ocr_v.get('valid_words', 0):,} / {ocr_v.get('total_words', 0):,})"
            )
            for category, data in eb.items():
                lines.append(
                    f"- {data.get('description', category)}: "
                    f"{data.get('count', 0):,} ({data.get('percentage_of_total_errors', 0):.1f}%)"
                )
            lines.append("")

    lines.append("\n## Key Findings\n")
    if all_results:
        worst_cer = max(all_results.items(), key=lambda x: x[1].cer)
        best_cer = min(all_results.items(), key=lambda x: x[1].cer)
        lines.append(
            f"- Highest CER: **{worst_cer[0]}** at {worst_cer[1].cer*100:.2f}%"
        )
        lines.append(
            f"- Lowest CER: **{best_cer[0]}** at {best_cer[1].cer*100:.2f}%"
        )

    lines.append(
        "\n> Phase 1 establishes the Qaari OCR baseline. "
        "All subsequent phases compare against these numbers.\n"
    )

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_error_examples(
    all_errors: list[SampleError],
    max_per_type: int = 5,
) -> dict[str, list[dict]]:
    """Collect representative error examples grouped by error type."""
    from src.analysis.error_analyzer import ErrorType
    examples: dict[str, list] = {et.value: [] for et in ErrorType}

    for sample_err in all_errors:
        for ce in sample_err.char_errors:
            key = ce.error_type.value
            if len(examples[key]) < max_per_type:
                examples[key].append({
                    "sample_id": sample_err.sample_id,
                    "gt": ce.gt_char,
                    "ocr": ce.ocr_char,
                    "context": ce.gt_context,
                    "position": ce.position.value,
                })

    # Remove empty entries
    return {k: v for k, v in examples.items() if v}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    setup_logging(results_dir)

    logger.info("Phase 1: Baseline & Error Taxonomy")
    logger.info("Config: %s", args.config)
    logger.info("Results dir: %s", results_dir)

    config = load_config(args.config)
    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")

    # Initialise CAMeL Tools (optional)
    validator: WordValidator | None = None
    if not args.no_camel:
        camel_cfg = config.get("camel", {})
        if camel_cfg.get("enabled", True):
            morph_cfg = camel_cfg.get("morphology", {})
            analyzer = MorphAnalyzer(
                db=morph_cfg.get("db", "calima-msa-r13"),
                cache_size=morph_cfg.get("cache_size", 10_000),
                enabled=True,
            )
            if analyzer.enabled:
                validator = WordValidator(analyzer)
                logger.info("CAMeL Tools morphological analyser ready.")
            else:
                logger.info("CAMeL Tools not available — skipping morphological analysis.")
        else:
            logger.info("CAMeL Tools disabled in config.")
    else:
        logger.info("--no-camel flag set — skipping morphological analysis.")

    # Load data
    loader = DataLoader(config)

    # Determine which datasets to run
    all_dataset_keys = [
        "PATS-A01-Akhbar",
        "PATS-A01-Andalus",
        "KHATT-train",
        "KHATT-validation",
    ]
    if args.dataset:
        dataset_keys = [args.dataset]
    else:
        dataset_keys = all_dataset_keys

    all_metric_results: dict[str, MetricResult] = {}
    normal_metric_results: dict[str, MetricResult] = {}
    all_quality_stats: dict[str, dict] = {}

    for ds_key in dataset_keys:
        try:
            logger.info("Loading dataset: %s ...", ds_key)
            samples = list(loader.iter_samples(ds_key, limit=limit))

            if not samples:
                logger.warning("No samples loaded for %s -- skipping.", ds_key)
                continue

            metric_result, normal_result, dq = process_dataset(
                dataset_key=ds_key,
                samples=samples,
                config=config,
                results_dir=results_dir,
                limit=limit,
                validator=validator,
            )
            all_metric_results[ds_key] = metric_result
            normal_metric_results[ds_key] = normal_result
            all_quality_stats[ds_key] = dq

        except DataError as exc:
            logger.warning("Skipping %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error processing %s: %s", ds_key, exc, exc_info=True)

    if not all_metric_results:
        logger.error("No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # Aggregate and write combined output
    aggregate_metrics(all_metric_results, normal_metric_results, all_quality_stats,
                      config, results_dir, limit)
    generate_report(all_metric_results, results_dir)

    logger.info("Phase 1 complete. Results in: %s", results_dir)


if __name__ == "__main__":
    main()
