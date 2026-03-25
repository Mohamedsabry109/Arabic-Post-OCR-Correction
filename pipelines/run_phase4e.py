#!/usr/bin/env python3
"""Phase 4E: Word-Error Pairs Prompting.

Reads concrete ``ocr_word > gt_word`` pairs from a pre-generated text file
and injects them into the system prompt so the LLM can recognise Qaari's
known word-level mistakes.

This is an isolated experiment that compares against Phase 2 (zero-shot
baseline).  The word-error pairs file is assumed to exist; use the word-level
analysis script (see docs/Word-Level-Analysis.txt) to generate it.

Two-stage pipeline:

  --mode export  -> Build inference_input.jsonl with word-error-pair prompts
                    -> results/phase4e/inference_input.jsonl

  --mode analyze -> Load corrections.jsonl, compute metrics and reports
                    -> results/phase4e/{dataset}/metrics.json
                       results/phase4e/metrics.json
                       results/phase4e/comparison.json
                       results/phase4e/paper_tables.md

Typical workflow
----------------
1. Ensure word-error pairs file exists at data/word_error_pairs.txt
   (or set phase4e.word_pairs_path in configs/config.yaml).

2. LOCAL:  python pipelines/run_phase4e.py --mode export

3. REMOTE: git clone <repo> && python scripts/infer.py \\
               --input  results/phase4e/inference_input.jsonl \\
               --output results/phase4e/corrections.jsonl \\
               --model  Qwen/Qwen3-4B-Instruct-2507

4. LOCAL:  python pipelines/run_phase4e.py --mode analyze

Usage
-----
    python pipelines/run_phase4e.py --mode export
    python pipelines/run_phase4e.py --mode export  --limit 50
    python pipelines/run_phase4e.py --mode export  --force
    python pipelines/run_phase4e.py --mode analyze
    python pipelines/run_phase4e.py --mode analyze --datasets KHATT-validation
    python pipelines/run_phase4e.py --mode analyze --no-error-analysis
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
from src.data.knowledge_base import WordErrorPairsLoader
from src.analysis.metrics import MetricResult, calculate_metrics_dual
from src.analysis.report_formatter import write_corrections_report
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from pipelines._utils import resolve_datasets, load_sample_list

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4E: Word-Error Pairs Prompting for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["export", "analyze"],
        help=(
            "export  -> produce inference_input.jsonl with word-error-pair prompts; "
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
        default=Path("results/phase4e"),
        dest="results_dir",
        help="Phase 4E results root directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase4e.log"

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
        "phase":        "phase4e",
        "dataset":      dataset,
        "model":        model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend":      model_cfg.get("backend", "transformers"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit":   get_git_commit(),
        "num_samples":  num_samples,
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


def _maybe_split_combined_corrections(results_dir: Path) -> None:
    """Split combined corrections.jsonl into per-dataset files if needed."""
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
# Error change analysis
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
# EXPORT MODE
# ---------------------------------------------------------------------------


def run_export(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    limit: Optional[int],
    force: bool,
    sample_ids: Optional[set[str]] = None,
) -> None:
    """Build inference_input.jsonl for Phase 4E (validation splits only).

    Loads all word-error pairs, formats them once, and embeds the formatted
    context into every JSONL record so the remote inference machine needs no
    extra files.
    """
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    phase4e_cfg = config.get("phase4e", {})
    pairs_path_str = phase4e_cfg.get(
        "word_pairs_path", WordErrorPairsLoader.DEFAULT_PAIRS_PATH
    )
    pairs_path = _PROJECT_ROOT / pairs_path_str

    select_n        = int(phase4e_cfg.get("select_n", 15))
    select_strategy = str(phase4e_cfg.get("select_strategy", "random"))
    select_seed     = int(phase4e_cfg.get("select_seed", 42))

    # Load and format word-error pairs (same context for every sample).
    pairs_loader = WordErrorPairsLoader()
    try:
        all_pairs = pairs_loader.load(pairs_path)
    except FileNotFoundError:
        logger.error(
            "Word-error pairs file not found: %s\n"
            "Generate it first (see docs/Word-Level-Analysis.txt).",
            pairs_path,
        )
        sys.exit(1)

    selected_pairs = pairs_loader.select(
        all_pairs, n=select_n, strategy=select_strategy, seed=select_seed
    )
    word_pairs_context = pairs_loader.format_for_prompt(selected_pairs)

    logger.info(
        "Loaded %d pairs from %s; selected %d (strategy=%s).",
        len(all_pairs), pairs_path, len(selected_pairs), select_strategy,
    )
    logger.info("Word-pairs context: %d chars", len(word_pairs_context))

    if not word_pairs_context.strip():
        logger.error("Word-pairs context is empty. Cannot export.")
        sys.exit(1)

    # Only export validation splits.
    export_datasets = [
        ds for ds in active_datasets
        if ds.lower().endswith(("-val", "-validation", "validation"))
    ]
    if not export_datasets:
        logger.warning(
            "No validation-split datasets in active set. "
            "Phase 4E only exports val splits. Active: %s", active_datasets,
        )
        return

    already_exported = _load_exported_datasets(output_path) if not force else set()
    loader_data = DataLoader(config)
    builder = PromptBuilder(
        crafted_prompt_path=config.get("prompt_craft", {}).get("crafted_prompt_path")
    )
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in export_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported -- skipping (use --force to re-export).",
                    ds_key,
                )
                continue

            try:
                samples = list(
                    loader_data.iter_samples(ds_key, limit=limit, sample_ids=sample_ids)
                )
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in tqdm(samples, desc=f"  Export {ds_key}", unit="sample"):
                record: dict = {
                    "sample_id":          sample.sample_id,
                    "dataset":            ds_key,
                    "ocr_text":           sample.ocr_text,
                    "gt_text":            sample.gt_text,
                    "prompt_type":        "word_error_pairs",
                    "prompt_version":     builder.word_error_pairs_prompt_version,
                    "word_pairs_context": word_pairs_context,
                    "num_pairs_injected": len(selected_pairs),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info("Exported %d samples for %s.", len(samples), ds_key)

    logger.info("=" * 60)
    logger.info(
        "Phase 4E export complete: %d new samples -> %s", total_written, output_path
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
    logger.info("       python pipelines/run_phase4e.py --mode analyze")
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

    logger.info("=" * 60)
    logger.info(
        "Analyzing [phase4e / %s]: %d samples, %d failed",
        dataset_key, n, n_failed,
    )

    # Step 1: CER/WER
    corrected_result, corrected_result_nd = calculate_metrics_dual(
        corrected_samples, dataset_name=dataset_key, text_field="corrected_text"
    )

    builder = PromptBuilder(
        crafted_prompt_path=config.get("prompt_craft", {}).get("crafted_prompt_path")
    )
    metrics_json = {
        "meta": make_meta(
            dataset_key, n, config, limit,
            extra={
                "prompt_type":              "word_error_pairs",
                "prompt_version":           builder.word_error_pairs_prompt_version,
                "total_prompt_tokens":      total_prompt_tokens,
                "total_output_tokens":      total_output_tokens,
                "total_latency_s":          round(total_latency, 2),
                "avg_latency_per_sample_s": round(total_latency / max(n, 1), 3),
                "failed_samples":           n_failed,
            },
        ),
        "corrected":                corrected_result.to_dict(),
        "corrected_no_diacritics":  corrected_result_nd.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")
    logger.info(
        "[%s] CER=%.2f%%  WER=%.2f%%",
        dataset_key, corrected_result.cer * 100, corrected_result.wer * 100,
    )

    # Step 2: Comparison vs Phase 2
    p2_metrics = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_metrics is not None:
        p2_cer = float(p2_metrics.get("cer", 0.0))
        p2_wer = float(p2_metrics.get("wer", 0.0))

        p2_path_full = phase2_dir / dataset_key / "metrics.json"
        p2_nd: dict = {}
        if p2_path_full.exists():
            with open(p2_path_full, encoding="utf-8") as _f:
                p2_nd = json.load(_f).get("corrected_no_diacritics", {})

        cer_delta = p2_cer - corrected_result.cer
        wer_delta = p2_wer - corrected_result.wer
        cer_rel = (cer_delta / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta / p2_wer * 100) if p2_wer > 0 else 0.0

        p2_cer_nd = float(p2_nd.get("cer", 0.0))
        p2_wer_nd = float(p2_nd.get("wer", 0.0))
        cer_d_nd = p2_cer_nd - corrected_result_nd.cer
        wer_d_nd = p2_wer_nd - corrected_result_nd.wer
        cer_r_nd = (cer_d_nd / p2_cer_nd * 100) if p2_cer_nd > 0 else 0.0
        wer_r_nd = (wer_d_nd / p2_wer_nd * 100) if p2_wer_nd > 0 else 0.0

        comparison = {
            "meta": make_meta(
                dataset_key, n, config, limit,
                extra={"comparison": "phase4e_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
            },
            "phase4e_corrected": {
                "cer": round(corrected_result.cer, 6),
                "wer": round(corrected_result.wer, 6),
            },
            "delta": {
                "cer_absolute":     round(cer_delta, 6),
                "wer_absolute":     round(wer_delta, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            "phase2_baseline_no_diacritics":   {"cer": round(p2_cer_nd, 6), "wer": round(p2_wer_nd, 6)},
            "phase4e_corrected_no_diacritics": {"cer": round(corrected_result_nd.cer, 6), "wer": round(corrected_result_nd.wer, 6)},
            "delta_no_diacritics": {
                "cer_absolute":     round(cer_d_nd, 6),
                "wer_absolute":     round(wer_d_nd, 6),
                "cer_relative_pct": round(cer_r_nd, 2),
                "wer_relative_pct": round(wer_r_nd, 2),
            },
            "interpretation": (
                f"CER {'reduced' if cer_delta >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {corrected_result.cer*100:.2f}%). "
                f"WER {'reduced' if wer_delta >= 0 else 'increased'} by "
                f"{abs(wer_rel):.1f}% "
                f"({p2_wer*100:.2f}% -> {corrected_result.wer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s] Phase2->Phase4E CER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key, p2_cer * 100, corrected_result.cer * 100, cer_rel,
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
            extra={"prompt_type": "word_error_pairs"},
        )
        save_json(error_changes, out_dir / "error_changes.json")

    return corrected_result


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

    _maybe_split_combined_corrections(results_dir)

    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    val_datasets = [
        ds for ds in active_datasets
        if ds.lower().endswith(("-val", "-validation", "validation"))
    ]
    if not val_datasets:
        logger.warning("No val-split datasets in active set: %s", active_datasets)
        val_datasets = active_datasets  # fallback

    for ds_key in val_datasets:
        try:
            metrics_path = results_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s] Already analyzed -- skipping (use --force).", ds_key
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
            records = load_corrections_jsonl(corrections_path)
            if limit:
                records = records[:limit]

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
            logger.error("[phase4e] Dataset %s: %s", ds_key, exc)
        except DataError as exc:
            logger.warning("[phase4e] Skipping %s: %s", ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[phase4e] Unexpected error on %s: %s", ds_key, exc, exc_info=True
            )

    if not all_corrected:
        logger.warning("No datasets analyzed.")
        return

    # Aggregate
    total_chars   = sum(r.num_chars_ref for r in all_corrected.values())
    total_words   = sum(r.num_words_ref for r in all_corrected.values())
    total_samples = sum(r.num_samples   for r in all_corrected.values())

    agg_cer = (
        sum(r.cer * r.num_chars_ref for r in all_corrected.values()) / total_chars
        if total_chars > 0 else 0.0
    )
    agg_wer = (
        sum(r.wer * r.num_words_ref for r in all_corrected.values()) / total_words
        if total_words > 0 else 0.0
    )

    aggregated = {
        "meta": make_meta(
            "all_datasets", total_samples, config, None,
            extra={"prompt_type": "word_error_pairs", "datasets": sorted(all_corrected.keys())},
        ),
        "aggregated_corrected": {
            "cer": round(agg_cer, 6),
            "wer": round(agg_wer, 6),
            "total_samples":    total_samples,
            "total_chars_ref":  total_chars,
            "total_words_ref":  total_words,
        },
        "by_dataset": {
            ds: {"cer": round(r.cer, 6), "wer": round(r.wer, 6), "num_samples": r.num_samples}
            for ds, r in all_corrected.items()
        },
    }
    save_json(aggregated, results_dir / "metrics.json")

    _write_paper_tables(all_corrected, all_comparisons, results_dir)

    logger.info("=" * 60)
    logger.info("Phase 4E analysis complete.")
    logger.info(
        "Aggregated CER=%.2f%%  WER=%.2f%% across %d datasets.",
        agg_cer * 100, agg_wer * 100, len(all_corrected),
    )
    logger.info("=" * 60)

    write_corrections_report(
        corrections_path=results_dir,
        output_path=results_dir / "sample_report.txt",
        title="Phase 4E -- Word-Error Pairs Prompting",
    )


def _write_paper_tables(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    results_dir: Path,
) -> None:
    """Write a markdown file with ready-to-paste paper tables."""
    lines: list[str] = [
        "# Phase 4E Results -- Word-Error Pairs Prompting",
        "",
        "## Per-Dataset Metrics",
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

    out_path = results_dir / "paper_tables.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Paper tables saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    args.results_dir = _PROJECT_ROOT / args.results_dir
    args.results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.results_dir)

    config = load_config(_PROJECT_ROOT / args.config)

    active_datasets = resolve_datasets(config, args.datasets)

    sample_ids: Optional[set[str]] = None
    if args.sample_list:
        sample_ids, sl_datasets = load_sample_list(args.sample_list)
        if not args.datasets:
            active_datasets = sl_datasets
        logger.info(
            "Sample list loaded: %d sample IDs from %s",
            len(sample_ids), args.sample_list,
        )

    logger.info("=" * 60)
    logger.info("Phase 4E: Word-Error Pairs Prompting")
    logger.info("Mode: %s", args.mode)
    if args.datasets:
        logger.info("Dataset filter: %s", args.datasets)
    if args.limit:
        logger.info("Limit: %d samples per dataset", args.limit)
    logger.info("Results dir: %s", args.results_dir)
    logger.info("=" * 60)

    phase2_dir = _PROJECT_ROOT / "results" / "phase2"

    if args.mode == "export":
        run_export(
            config=config,
            active_datasets=active_datasets,
            results_dir=args.results_dir,
            limit=args.limit,
            force=args.force,
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


if __name__ == "__main__":
    main()
