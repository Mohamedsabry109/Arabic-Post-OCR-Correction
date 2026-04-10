#!/usr/bin/env python3
"""Phase 6: Combinations.

Tests combinations of Phase 3 (OCR-Aware) and Phase 4 (Self-Reflective)
prompt components, plus CAMeL post-processing on the best combo.

Experiment set
--------------
Inference-based combos (need export -> Kaggle -> analyze):
  conf_only    Phase 3 alone (OCR-Aware confusion matrix)
  self_only    Phase 4 alone (Self-Reflective insights + word pairs)
  conf_self    Phase 3 + 4 combined (full prompt)

CAMeL post-processing combo (local only, no Kaggle step):
  best_camel   Best inference combo + Phase 5 CAMeL validation

Pipeline per inference combo
-----------------------------
  1. LOCAL:  python pipelines/run_phase6.py --combo conf_only --mode export
  2. REMOTE: python scripts/infer.py \\
                 --input  results/phase6/conf_only/inference_input.jsonl \\
                 --output results/phase6/conf_only/corrections.jsonl
  3. LOCAL:  python pipelines/run_phase6.py --combo conf_only --mode analyze

CAMeL combo (no Kaggle step):
  python pipelines/run_phase6.py --combo best_camel --mode validate

Cross-combo summary (after all combos analyzed/validated):
  python pipelines/run_phase6.py --mode summarize

Usage
-----
    # Smoke test: export + infer + analyze one combo
    python pipelines/run_phase6.py --combo conf_only --mode export \\
        --datasets KHATT-train --limit 50
    python scripts/infer.py \\
        --input  results/phase6/conf_only/inference_input.jsonl \\
        --output results/phase6/conf_only/corrections.jsonl \\
        --datasets KHATT-train --limit 50
    python pipelines/run_phase6.py --combo conf_only --mode analyze \\
        --datasets KHATT-train

    # Run all inference combos in sequence:
    python pipelines/run_phase6.py --combo all --mode export
    # (then run Kaggle for each combo)
    python pipelines/run_phase6.py --combo all --mode analyze

    # After setting phase6.best_combo in config.yaml:
    python pipelines/run_phase6.py --combo best_camel --mode validate

    # Final cross-combo analysis:
    python pipelines/run_phase6.py --mode summarize
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
from src.data.knowledge_base import (
    ConfusionMatrixLoader, ConfusionPair,
    LLMInsightsLoader,
    WordErrorPairsLoader,
)
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from src.analysis.metrics import MetricResult, calculate_metrics, calculate_metrics_dual, calculate_cer
from src.analysis.report_formatter import write_corrections_report
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.analysis.stats_tester import StatsTester
from src.linguistic.morphology import MorphAnalyzer
from src.linguistic.validator import WordValidator, TextCorrectionResult
from pipelines._utils import (
    resolve_datasets, load_sample_list, compute_group_aggregates,
    split_runaway_samples, DEFAULT_RUNAWAY_RATIO_THRESHOLD,
    load_phase2_full_metrics, pick_phase2_variant, _pick_corrected_key,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combo definitions
# ---------------------------------------------------------------------------

# (use_confusion, use_self)
# use_confusion = True injects Phase 3 confusion matrix (filtered by LLM failures)
# use_self = True injects Phase 4 self-reflective signals (insights + word pairs + overcorrections)
COMBO_COMPONENTS: dict[str, tuple[bool, bool]] = {
    "conf_only":   (True,  False),   # Phase 3 alone
    "self_only":   (False, True),    # Phase 4 alone
    "conf_self":   (True,  True),    # Phase 3 + 4 (full prompt)
}

CAMEL_COMBOS = {"best_camel"}   # Best inference combo + Phase 5 CAMeL post-processing

INFERENCE_COMBOS = list(COMBO_COMPONENTS.keys())
ALL_COMBOS = INFERENCE_COMBOS + sorted(CAMEL_COMBOS)

COMBO_DESCRIPTIONS: dict[str, str] = {
    "conf_only":    "Phase 3: OCR-Aware Confusion Matrix",
    "self_only":    "Phase 4: Self-Reflective (insights + word pairs + overcorrections)",
    "conf_self":    "Phase 3 + 4: Confusion + Self-Reflective",
    "best_camel":   "Best Inference Combo + Phase 5 CAMeL Validation",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: Combinations & Ablation Study"
    )
    parser.add_argument(
        "--combo",
        type=str,
        default=None,
        metavar="COMBO_ID",
        help=(
            "Combo to process: "
            + ", ".join(ALL_COMBOS)
            + ". Use 'all' for all inference combos (export/analyze modes only)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["export", "analyze", "validate", "summarize"],
        help=(
            "export    -> write inference_input.jsonl (inference combos); "
            "analyze   -> compute metrics from corrections.jsonl; "
            "validate  -> apply CAMeL post-processing (best_camel); "
            "summarize -> cross-combo analysis (no --combo needed)"
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
        help="Skip error_changes.json computation (faster).",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("results/phase2"),
        dest="phase2_dir",
        help="Phase 2 results directory (baseline for comparison).",
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("results/phase1"),
        dest="phase1_dir",
        help="Phase 1 results directory (source of confusion matrices).",
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
        default=Path("results/phase6"),
        dest="results_dir",
        help="Phase 6 results root directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path, label: str = "phase6") -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{label}.log"

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
    combo_id: str,
    dataset: str,
    num_samples: int,
    config: dict,
    limit: Optional[int],
    extra: Optional[dict] = None,
) -> dict:
    model_cfg = config.get("model", {})
    meta = {
        "phase":       "phase6",
        "combo_id":    combo_id,
        "description": COMBO_DESCRIPTIONS.get(combo_id, combo_id),
        "components":  list(_active_component_names(combo_id)),
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


def _active_component_names(combo_id: str) -> list[str]:
    """Return human-readable component list for a combo."""
    if combo_id in CAMEL_COMBOS:
        return ["best_combo", "camel"]
    flags = COMBO_COMPONENTS.get(combo_id, (False, False))
    names = []
    labels = ["confusion", "self"]
    for flag, label in zip(flags, labels):
        if flag:
            names.append(label)
    return names


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Context builders (adapted from Phase 3/4/5)
# ---------------------------------------------------------------------------


def build_confusion_context_for_dataset(
    dataset_key: str,
    phase1_dir: Path,
    pooled: dict[str, list],
    loader_conf: "ConfusionMatrixLoader",
    top_n: int,
    format_style: str,
    min_substitutions: int,
) -> tuple[str, str]:
    """Return (confusion_context, source_label) for one dataset.

    Resolution: dataset-specific -> pooled (PATS-A01 or KHATT) -> empty.
    """
    matrix_path = phase1_dir / dataset_key / "confusion_matrix.json"

    if loader_conf.has_enough_data(matrix_path, min_substitutions):
        try:
            pairs = loader_conf.load(matrix_path)
            context = loader_conf.format_for_prompt(pairs, n=top_n, style=format_style)
            return context, "dataset_specific"
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[%s] Could not load confusion matrix: %s", dataset_key, exc)

    if dataset_key.startswith("PATS-A01-"):
        pool_key = "PATS-A01"
    elif dataset_key.startswith("KHATT-"):
        pool_key = "KHATT"
    else:
        pool_key = None

    if pool_key and pooled.get(pool_key):
        pairs = pooled[pool_key]
        context = loader_conf.format_for_prompt(pairs, n=top_n, style=format_style)
        return context, f"pooled_{pool_key}"

    logger.warning(
        "[%s] No confusion data available -- confusion_context empty.", dataset_key
    )
    return "", "none"


def build_pooled_matrices(
    phase1_dir: Path,
    loader_conf: "ConfusionMatrixLoader",
    all_dataset_names: list[str],
) -> dict[str, list]:
    """Pre-build pooled confusion matrices for PATS-A01 and KHATT types."""
    pats_paths = [
        phase1_dir / name / "confusion_matrix.json"
        for name in all_dataset_names
        if name.startswith("PATS-A01-")
    ]
    khatt_paths = [
        phase1_dir / name / "confusion_matrix.json"
        for name in all_dataset_names
        if name.startswith("KHATT-")
    ]

    pats_pooled = loader_conf.load_pooled(pats_paths) if pats_paths else []
    khatt_pooled = loader_conf.load_pooled(khatt_paths) if khatt_paths else []

    logger.info(
        "Pooled confusion matrices: PATS-A01=%d pairs, KHATT=%d pairs.",
        len(pats_pooled), len(khatt_pooled),
    )
    return {"PATS-A01": pats_pooled, "KHATT": khatt_pooled}


    # build_rules_context and build_examples_context removed in phase refactoring


# ---------------------------------------------------------------------------
# Shared corrections loading helpers
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
                logger.warning("Skipping malformed line %d in %s: %s", lineno, corrections_path, exc)
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
            "Export resume: found existing datasets in %s: %s", output_path, sorted(found)
        )
    return found


def _maybe_split_combined_corrections(combo_dir: Path, force: bool = False) -> None:
    """Split combined corrections.jsonl into per-dataset files if needed.

    When *force* is True, existing per-dataset files are overwritten.
    """
    combined = combo_dir / "corrections.jsonl"
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
        out_path = combo_dir / ds_key / "corrections.jsonl"
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


# ---------------------------------------------------------------------------
# Error change analysis
# ---------------------------------------------------------------------------


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
    combo_id: str,
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
            s1.sample_id = cs.sample.sample_id
            s1.dataset   = cs.sample.dataset
            s1.gt_text   = cs.sample.gt_text
            s1.ocr_text  = cs.sample.ocr_text

            s2 = _Stub()
            s2.sample_id = cs.sample.sample_id
            s2.dataset   = cs.sample.dataset
            s2.gt_text   = cs.sample.gt_text
            s2.ocr_text  = cs.corrected_text

            err1 = analyzer.analyse_sample(s1)  # type: ignore[arg-type]
            err2 = analyzer.analyse_sample(s2)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error analysis failed for %s: %s", cs.sample.sample_id, exc)
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
# Per-dataset analyze step
# ---------------------------------------------------------------------------


def process_dataset_analyze(
    combo_id: str,
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    combo_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    analyze_errors: bool,
) -> MetricResult:
    """Run CER/WER, comparison vs Phase 2, and optional error analysis for one dataset."""
    out_dir = combo_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(corrected_samples)
    n_failed = sum(1 for cs in corrected_samples if not cs.success)
    total_prompt_tokens = sum(cs.prompt_tokens for cs in corrected_samples)
    total_output_tokens = sum(cs.output_tokens for cs in corrected_samples)
    total_latency = sum(cs.latency_s for cs in corrected_samples)
    avg_latency = total_latency / max(n, 1)

    eval_cfg = config.get("evaluation", {})
    exclude_runaway = eval_cfg.get("exclude_runaway", False)
    threshold = eval_cfg.get("runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD)

    logger.info("=" * 60)
    logger.info(
        "Analyzing [%s / %s]: %d samples, %d failed",
        combo_id, dataset_key, n, n_failed,
    )

    # Step 1: Split into normal / runaway, compute metrics for both
    normal_samples, runaway_samples, data_quality = split_runaway_samples(
        corrected_samples, threshold=threshold,
    )
    if runaway_samples:
        logger.info("[%s / %s] %s", combo_id, dataset_key, data_quality["description"])

    logger.info("[%s / %s] Calculating OCR baseline + corrected CER/WER ...", combo_id, dataset_key)

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
    metrics_extra = {
        "prompt_type":              "combined",
        "prompt_version":           builder.combined_prompt_version,
        "total_prompt_tokens":      total_prompt_tokens,
        "total_output_tokens":      total_output_tokens,
        "total_latency_s":          round(total_latency, 2),
        "avg_latency_per_sample_s": round(avg_latency, 3),
        "failed_samples":           n_failed,
        "primary_variant":          primary_source,
    }
    metrics_json = {
        "meta": make_meta(combo_id, dataset_key, n, config, limit, extra=metrics_extra),
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
        "[%s / %s] Primary (%s): OCR CER=%.2f%% -> LLM CER=%.2f%%  |  no-diac: %.2f%% -> %.2f%%",
        combo_id, dataset_key, primary_source,
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
                combo_id, dataset_key, n, config, limit,
                extra={"comparison": f"phase5_{combo_id}_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
                "variant": p2_src,
            },
            f"phase5_{combo_id}_corrected": {
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
            **(_nd_comparison_block(p2_nd, primary_nd, f"phase5_{combo_id}")),
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
            "[%s / %s] Phase2->Phase5 CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            combo_id, dataset_key,
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
                "[%s / %s] ND  CER: %.2f%% -> %.2f%% (%+.1f%%)  |  ND  WER: %.2f%% -> %.2f%% (%+.1f%%)",
                combo_id, dataset_key,
                p2_cer_nd_v * 100, primary_nd.cer * 100, cer_r_nd_v,
                p2_wer_nd_v * 100, primary_nd.wer * 100, wer_r_nd_v,
            )
    else:
        logger.warning(
            "[%s / %s] Phase 2 metrics not found -- skipping comparison.", combo_id, dataset_key
        )

    # Step 3: Error change analysis (optional)
    if analyze_errors:
        logger.info("[%s / %s] Running error change analysis ...", combo_id, dataset_key)
        error_changes = run_error_change_analysis(corrected_samples, dataset_key, combo_id)
        error_changes["meta"] = make_meta(
            combo_id, dataset_key, n, config, limit, extra={"prompt_type": "combined"}
        )
        save_json(error_changes, out_dir / "error_changes.json")

    return primary


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------


def run_export(
    combo_id: str,
    config: dict,
    active_datasets: list[str],
    all_dataset_names: list[str],
    combo_dir: Path,
    phase1_dir: Path,
    limit: Optional[int],
    force: bool,
    sample_ids: Optional[set[str]] = None,
) -> None:
    """Build inference_input.jsonl for one inference combo.

    Loads contexts once, then iterates over all active datasets.
    Confusion context is per-dataset; rules/examples are global; retrieval is per-sample.
    """
    use_conf, use_self = COMBO_COMPONENTS[combo_id]
    use_rules = False  # Removed in phase refactoring
    use_fewshot = False  # Removed in phase refactoring

    output_path = combo_dir / "inference_input.jsonl"
    combo_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 6 EXPORT: combo=%s  (%s)", combo_id, COMBO_DESCRIPTIONS[combo_id])
    logger.info(
        "  Components: confusion=%s  rules=%s  fewshot=%s  self=%s",
        use_conf, use_rules, use_fewshot, use_self,
    )

    # --- Build global contexts once ---
    phase3_cfg = config.get("phase3", {})
    top_n = phase3_cfg.get("top_n", 10)
    format_style_conf = phase3_cfg.get("format_style", "flat_arabic")
    min_subs = phase3_cfg.get(
        "min_substitutions_for_dataset_matrix", ConfusionMatrixLoader.MIN_SUBSTITUTIONS
    )

    loader_conf = ConfusionMatrixLoader() if use_conf else None
    pooled: dict[str, list] = {}
    if use_conf and loader_conf is not None:
        pooled = build_pooled_matrices(phase1_dir, loader_conf, all_dataset_names)

    # rules_context and examples_context removed in phase refactoring

    pats_insights_context = ""
    khatt_insights_context = ""
    word_pairs_context = ""
    overcorrection_context = ""
    if use_self:
        from src.data.knowledge_base import (
            load_unfixed_word_pairs, load_introduced_word_pairs,
            format_word_examples_for_prompt, format_overcorrection_warnings,
        )

        phase4_cfg = config.get("phase4", {})
        training_dir = _PROJECT_ROOT / phase4_cfg.get(
            "training_artifacts_dir", "results/phase2-training/analysis"
        )
        failures_path = training_dir / "word_pairs_llm_failures.txt"

        # Load insights from Phase 4 results
        phase4_dir = _PROJECT_ROOT / "results" / "phase4"
        insights_dir = phase4_dir / "insights"
        pats_path = insights_dir / "PATS-A01_insights.json"
        khatt_path = insights_dir / "KHATT_insights.json"

        insights_loader = LLMInsightsLoader()
        insight_cfg = phase4_cfg.get("insights", {})
        format_kwargs = {
            "min_fix_rate_strength":  float(insight_cfg.get("min_fix_rate_strength",  0.6)),
            "max_fix_rate_weakness":  float(insight_cfg.get("max_fix_rate_weakness",  0.4)),
            "min_intro_rate":         float(insight_cfg.get("min_intro_rate",         0.05)),
            "min_sample_size":        int(insight_cfg.get("min_sample_size",           10)),
            "top_n_weaknesses":       int(insight_cfg.get("top_n_weaknesses",           3)),
            "top_n_overcorrections":  int(insight_cfg.get("top_n_overcorrections",      2)),
        }

        if pats_path.exists():
            pats_insights_context = insights_loader.format_for_prompt(
                insights_loader.load(pats_path), **format_kwargs
            )
            logger.info("Loaded PATS-A01 insights (%d chars).", len(pats_insights_context))
        else:
            logger.warning("Phase 4 PATS-A01 insights not found: %s", pats_path)

        if khatt_path.exists():
            khatt_insights_context = insights_loader.format_for_prompt(
                insights_loader.load(khatt_path), **format_kwargs
            )
            logger.info("Loaded KHATT insights (%d chars).", len(khatt_insights_context))
        else:
            logger.warning("Phase 4 KHATT insights not found: %s", khatt_path)

        # Load word pairs + overcorrections from training artifacts
        if failures_path.exists():
            unfixed = load_unfixed_word_pairs(failures_path)
            if unfixed:
                n_pairs = int(phase4_cfg.get("word_pairs_n", 15))
                word_pairs_context = format_word_examples_for_prompt(unfixed, n=n_pairs)
                logger.info("Loaded %d unfixed word pairs -> %d chars.", len(unfixed), len(word_pairs_context))

            introduced = load_introduced_word_pairs(failures_path)
            if introduced:
                n_over = int(phase4_cfg.get("overcorrection_n", 10))
                overcorrection_context = format_overcorrection_warnings(introduced, n=n_over)
                logger.info("Loaded %d introduced pairs -> %d chars.", len(introduced), len(overcorrection_context))
        else:
            # Fallback: legacy word_error_pairs.txt
            pairs_path = phase4_dir / "word_error_pairs.txt"
            if pairs_path.exists():
                pairs_loader = WordErrorPairsLoader()
                try:
                    all_pairs = pairs_loader.load(pairs_path)
                    selected = pairs_loader.select(
                        all_pairs,
                        n=int(phase4_cfg.get("word_pairs_n", 15)),
                        strategy=str(phase4_cfg.get("word_pairs_strategy", "random")),
                        seed=int(phase4_cfg.get("word_pairs_seed", 42)),
                    )
                    word_pairs_context = pairs_loader.format_for_prompt(selected)
                    logger.info("Loaded %d word-error pairs (legacy) (%d chars).", len(selected), len(word_pairs_context))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load word-error pairs: %s", exc)

        if not pats_insights_context and not khatt_insights_context and not word_pairs_context:
            logger.error(
                "use_self=True but no Phase 4 outputs found. "
                "Cannot export combo '%s'. "
                "Run: python pipelines/run_phase4.py --mode analyze-train first.",
                combo_id,
            )
            return

    loader_data = DataLoader(config)
    already_exported = _load_exported_datasets(output_path) if not force else set()
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported -- skipping (use --force to re-export).", ds_key
                )
                continue

            # Resolve confusion context for this dataset
            if use_conf and loader_conf is not None:
                confusion_context, conf_source = build_confusion_context_for_dataset(
                    dataset_key=ds_key,
                    phase1_dir=phase1_dir,
                    pooled=pooled,
                    loader_conf=loader_conf,
                    top_n=top_n,
                    format_style=format_style_conf,
                    min_substitutions=min_subs,
                )
            else:
                confusion_context, conf_source = "", "disabled"

            # Resolve self-reflective context for this dataset
            if use_self:
                insights_context = (
                    pats_insights_context if ds_key.startswith("PATS-A01-") else khatt_insights_context
                )
            else:
                insights_context = ""

            # Determine prompt_type
            any_context = any([confusion_context, insights_context, word_pairs_context])
            prompt_type = "combined" if any_context else "zero_shot"

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit, sample_ids=sample_ids))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in tqdm(samples, desc=f"  Export {ds_key}", unit="sample"):
                record: dict = {
                    "sample_id":          sample.sample_id,
                    "dataset":            ds_key,
                    "ocr_text":           sample.ocr_text,
                    "gt_text":            sample.gt_text,
                    "prompt_type":        prompt_type,
                    "combo_id":           combo_id,
                    "prompt_version":     PromptBuilder.COMBINED_PROMPT_VERSION,
                    "confusion_context":        confusion_context or None,
                    "insights_context":         insights_context or None,
                    "word_pairs_context":       word_pairs_context or None,
                    "overcorrection_context":   overcorrection_context or None,
                }
                if use_conf:
                    record["confusion_source"] = conf_source

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info(
                "Exported %d samples for %s (conf_src=%s).",
                len(samples), ds_key, conf_source,
            )

    logger.info("=" * 60)
    logger.info(
        "Phase 6 export complete: %d new samples -> %s", total_written, output_path
    )
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       python scripts/infer.py \\")
    logger.info("           --input  %s/inference_input.jsonl \\", combo_dir)
    logger.info("           --output %s/corrections.jsonl", combo_dir)
    logger.info("  3. Run analysis locally:")
    logger.info("       python pipelines/run_phase6.py --combo %s --mode analyze", combo_id)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# ANALYZE MODE
# ---------------------------------------------------------------------------


def run_analyze(
    combo_id: str,
    config: dict,
    active_datasets: list[str],
    combo_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    force: bool,
    analyze_errors: bool,
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Compute metrics and comparisons for one combo across all active datasets."""
    _maybe_split_combined_corrections(combo_dir, force=force)

    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    for ds_key in active_datasets:
        try:
            metrics_path = combo_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s / %s] Already analyzed -- skipping (use --force).", combo_id, ds_key
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
                cmp_path = combo_dir / ds_key / "comparison_vs_phase2.json"
                if cmp_path.exists():
                    with open(cmp_path, encoding="utf-8") as f:
                        all_comparisons[ds_key] = json.load(f)
                continue

            corrections_path = combo_dir / ds_key / "corrections.jsonl"
            corrected_samples = load_corrections(corrections_path)
            if limit:
                corrected_samples = corrected_samples[:limit]

            metric_result = process_dataset_analyze(
                combo_id=combo_id,
                dataset_key=ds_key,
                corrected_samples=corrected_samples,
                config=config,
                combo_dir=combo_dir,
                phase2_dir=phase2_dir,
                limit=limit,
                analyze_errors=analyze_errors,
            )
            all_corrected[ds_key] = metric_result

            cmp_path = combo_dir / ds_key / "comparison_vs_phase2.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("[%s] Dataset %s: %s", combo_id, ds_key, exc)
        except DataError as exc:
            logger.warning("[%s] Skipping %s: %s", combo_id, ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%s] Unexpected error on %s: %s", combo_id, ds_key, exc, exc_info=True
            )

    return all_corrected, all_comparisons


# ---------------------------------------------------------------------------
# VALIDATE MODE (CAMeL post-processing combos)
# ---------------------------------------------------------------------------


def _load_base_corrections_for_dataset(
    base_combo_dir: Path,
    dataset_key: str,
) -> list[dict]:
    """Load raw correction records for dataset_key from base combo."""
    per_ds_path = base_combo_dir / dataset_key / "corrections.jsonl"
    combined_path = base_combo_dir / "corrections.jsonl"

    records: list[dict] = []
    if per_ds_path.exists():
        src = per_ds_path
    elif combined_path.exists():
        src = combined_path
    else:
        raise FileNotFoundError(
            f"Base combo corrections not found for {dataset_key}.\n"
            f"Tried: {per_ds_path} and {combined_path}"
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

    logger.info("Loaded %d base records for %s from %s", len(records), dataset_key, src)
    return records


def process_dataset_validate(
    combo_id: str,
    dataset_key: str,
    base_records: list[dict],
    validator: WordValidator,
    config: dict,
    combo_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
) -> MetricResult:
    """Apply CAMeL revert strategy to base combo corrections for one dataset."""
    out_dir = combo_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        base_records = base_records[:limit]

    n = len(base_records)
    logger.info("=" * 60)
    logger.info("Phase 6 VALIDATE: %s / %s  (%d samples)", combo_id, dataset_key, n)

    corrected_samples: list[CorrectedSample] = []
    revert_results: list[TextCorrectionResult] = []

    out_corrections_path = out_dir / "corrections.jsonl"
    with open(out_corrections_path, "w", encoding="utf-8") as out_f:
        for r in tqdm(base_records, desc=f"  Validating {dataset_key}", unit="sample"):
            llm_text = r.get("corrected_text", r["ocr_text"])
            ocr_text = r.get("ocr_text", "")
            gt_text  = r.get("gt_text", "")

            revert_result = validator.validate_correction(llm_text, ocr_text, strategy="revert")
            revert_results.append(revert_result)

            out_record = {
                "sample_id":       r["sample_id"],
                "dataset":         dataset_key,
                "ocr_text":        ocr_text,
                "corrected_text":  revert_result.final_text,
                "gt_text":         gt_text,
                "base_text":       llm_text,
                "reverted_count":  revert_result.reverted_count,
                "kept_count":      revert_result.kept_count,
                "unchanged_count": revert_result.unchanged_count,
                "revert_rate":     round(revert_result.revert_rate, 4),
                "combo_id":        combo_id,
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

    logger.info("Validated corrections written to %s", out_corrections_path)

    # Revert statistics
    total_words     = sum(rr.total_words for rr in revert_results)
    total_reverted  = sum(rr.reverted_count for rr in revert_results)
    total_kept      = sum(rr.kept_count for rr in revert_results)
    total_unchanged = sum(rr.unchanged_count for rr in revert_results)
    avg_revert_rate = total_reverted / max(total_words, 1)

    validation_stats = {
        "meta": make_meta(combo_id, dataset_key, n, config, limit),
        "summary": {
            "total_samples":        n,
            "samples_with_reverts": sum(1 for rr in revert_results if rr.reverted_count > 0),
            "total_arabic_words":   total_words,
            "words_reverted":       total_reverted,
            "words_kept":           total_kept,
            "words_unchanged":      total_unchanged,
            "avg_revert_rate":      round(avg_revert_rate, 4),
            "revert_rate_pct":      round(avg_revert_rate * 100, 2),
        },
    }
    save_json(validation_stats, out_dir / "validation_stats.json")
    logger.info(
        "[%s / %s] Reverted %d / %d words (%.1f%%)",
        combo_id, dataset_key, total_reverted, total_words, avg_revert_rate * 100,
    )

    # CER/WER with runaway splitting
    eval_cfg = config.get("evaluation", {})
    exclude_runaway = eval_cfg.get("exclude_runaway", False)
    threshold = eval_cfg.get("runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD)

    normal_samples_v, runaway_samples_v, data_quality = split_runaway_samples(
        corrected_samples, threshold=threshold,
    )
    if runaway_samples_v:
        logger.info("[%s / %s] %s", combo_id, dataset_key, data_quality["description"])

    # OCR baseline
    ocr_all, ocr_all_nd = calculate_metrics_dual(
        corrected_samples, dataset_name=dataset_key, text_field="ocr_text",
    )
    ocr_normal, ocr_normal_nd = calculate_metrics_dual(
        normal_samples_v, dataset_name=dataset_key, text_field="ocr_text",
    )

    # Corrected
    all_result, all_result_nd = calculate_metrics_dual(
        corrected_samples, dataset_name=dataset_key, text_field="corrected_text",
    )
    normal_result, normal_result_nd = calculate_metrics_dual(
        normal_samples_v, dataset_name=dataset_key, text_field="corrected_text",
    )

    if exclude_runaway:
        primary, primary_nd = normal_result, normal_result_nd
        primary_source = "normal_only"
    else:
        primary, primary_nd = all_result, all_result_nd
        primary_source = "all"

    metrics_json = {
        "meta": make_meta(
            combo_id, dataset_key, n, config, limit,
            extra={
                "prompt_type": "camel_validation",
                "total_reverted": total_reverted,
                "primary_variant": primary_source,
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

    # Comparison vs Phase 2
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
                combo_id, dataset_key, n, config, limit,
                extra={"comparison": f"phase5_{combo_id}_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "variant": p2_src,
            },
            f"phase5_{combo_id}_corrected": {
                "cer": round(primary.cer, 6),
                "wer": round(primary.wer, 6),
                "variant": primary_source,
            },
            "delta": {
                "cer_absolute": round(cer_delta, 6),
                "wer_absolute": round(wer_delta, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            **(_nd_comparison_block(p2_nd, primary_nd, f"phase5_{combo_id}")),
            "interpretation": (
                f"CER {'reduced' if cer_delta >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {primary.cer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s / %s] CER: %.2f%% -> %.2f%% (%+.1f%%)",
            combo_id, dataset_key, p2_cer * 100, primary.cer * 100, cer_rel,
        )

    return primary


def run_validate(
    combo_id: str,
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    force: bool,
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Apply CAMeL post-processing to a base combo for all active datasets."""
    # Determine base combo directory
    if combo_id == "best_camel":
        base_combo_id = config.get("phase6", {}).get("best_combo")
        if not base_combo_id:
            logger.error(
                "phase6.best_combo not set in config.yaml. "
                "Review inference combo results and set it before running best_camel."
            )
            sys.exit(1)
    else:
        logger.error("Unknown CAMeL combo: %s", combo_id)
        sys.exit(1)

    base_combo_dir = results_dir / base_combo_id
    if not base_combo_dir.exists():
        logger.error(
            "Base combo directory not found: %s\n"
            "Run: python pipelines/run_phase6.py --combo %s --mode analyze first.",
            base_combo_dir, base_combo_id,
        )
        sys.exit(1)

    combo_dir = results_dir / combo_id
    combo_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 6 VALIDATE: combo=%s (base=%s)", combo_id, base_combo_id)

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
            "Phase 5 validate requires camel_tools but it is not available.\n"
            "Install: pip install camel-tools && camel_data -i morphology-db-msa-r13\n"
            "Or disable with: camel.enabled: false in config.yaml"
        )
        sys.exit(1)

    validator = WordValidator(analyzer)
    logger.info(
        "CAMeL Tools initialised (db=%s, enabled=%s).", analyzer.db, analyzer.enabled
        )

    # Split base corrections if needed
    _maybe_split_combined_corrections(base_combo_dir, force=force)

    all_corrected: dict[str, MetricResult] = {}
    all_comparisons: dict[str, dict] = {}

    for ds_key in active_datasets:
        try:
            metrics_path = combo_dir / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info(
                    "[%s / %s] Already validated -- skipping (use --force).", combo_id, ds_key
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
                cmp_path = combo_dir / ds_key / "comparison_vs_phase2.json"
                if cmp_path.exists():
                    with open(cmp_path, encoding="utf-8") as f:
                        all_comparisons[ds_key] = json.load(f)
                continue

            base_records = _load_base_corrections_for_dataset(base_combo_dir, ds_key)
            if not base_records:
                logger.warning("No base records for %s -- skipping.", ds_key)
                continue

            metric_result = process_dataset_validate(
                combo_id=combo_id,
                dataset_key=ds_key,
                base_records=base_records,
                validator=validator,
                config=config,
                combo_dir=combo_dir,
                phase2_dir=phase2_dir,
                limit=limit,
            )
            all_corrected[ds_key] = metric_result

            cmp_path = combo_dir / ds_key / "comparison_vs_phase2.json"
            if cmp_path.exists():
                with open(cmp_path, encoding="utf-8") as f:
                    all_comparisons[ds_key] = json.load(f)

        except FileNotFoundError as exc:
            logger.error("[%s] Dataset %s: %s", combo_id, ds_key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Unexpected error on %s: %s", combo_id, ds_key, exc, exc_info=True)

    return all_corrected, all_comparisons


# ---------------------------------------------------------------------------
# Aggregation helpers
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


def aggregate_combo_results(
    combo_id: str,
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    config: dict,
    combo_dir: Path,
    limit: Optional[int],
) -> None:
    """Write aggregated metrics.json and comparison.json for one combo."""
    builder = PromptBuilder(crafted_prompt_path=config.get("prompt_craft", {}).get("crafted_prompt_path"))

    # Aggregate no-diacritics metrics
    nd_data = _load_nd_results(all_corrected, combo_dir)
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

    output = {
        "meta": {
            "phase":       "phase6",
            "combo_id":    combo_id,
            "description": COMBO_DESCRIPTIONS.get(combo_id, combo_id),
            "components":  _active_component_names(combo_id),
            "model":       config.get("model", {}).get("name", ""),
            "prompt_type": "combined",
            "prompt_version": builder.combined_prompt_version,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit":  get_git_commit(),
            "limit_applied": limit,
        },
        "results": {k: v.to_dict() for k, v in all_corrected.items()},
        "aggregated_corrected_no_diacritics": {
            "cer": round(agg_cer_nd, 6),
            "wer": round(agg_wer_nd, 6),
        },
        "results_no_diacritics": {
            ds: {"cer": round(v.get("cer", 0), 6), "wer": round(v.get("wer", 0), 6)}
            for ds, v in nd_data.items()
        },
    }
    # Macro-averaged aggregates by dataset group (PATS-A01 / KHATT).
    output["group_aggregates"] = compute_group_aggregates(output["results"])
    if nd_data:
        output["group_aggregates_no_diacritics"] = compute_group_aggregates(
            {ds: {"cer": v.get("cer", 0), "wer": v.get("wer", 0)} for ds, v in nd_data.items()}
        )
    save_json(output, combo_dir / "metrics.json")

    if all_comparisons:
        cmp_output = {
            "meta": {
                "phase":   "phase6",
                "combo_id": combo_id,
                "comparison": f"phase5_{combo_id}_vs_phase2",
                "model":   config.get("model", {}).get("name", ""),
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "datasets": all_comparisons,
        }
        save_json(cmp_output, combo_dir / "comparison.json")


def print_combo_summary(
    combo_id: str,
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    combo_dir: Optional[Path] = None,
) -> None:
    nd_data = _load_nd_results(all_corrected, combo_dir) if combo_dir is not None else {}

    sep = "=" * 90
    row_sep = "  " + "-" * 82
    title = f"PHASE5 [{combo_id}] -- {COMBO_DESCRIPTIONS.get(combo_id, combo_id)}"

    # Table 1: WITH DIACRITICS
    print("\n" + sep)
    print(f"{title}  [WITH DIACRITICS]")
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
    print(f"\n{title}  [NO DIACRITICS]")
    print(sep)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'Px CER':>8} {'D(CER)':>8} {'P2 WER':>8} {'Px WER':>8} {'D(WER)':>8} {'N':>6}")
    print(row_sep)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        nd = nd_data.get(ds, {})
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
# SUMMARIZE MODE
# ---------------------------------------------------------------------------


def _load_combo_avg_metrics(
    results_dir: Path,
    combo_id: str,
) -> Optional[dict]:
    """Load aggregate metrics for one combo. Returns dict with avg_cer, avg_wer, n_datasets."""
    metrics_path = results_dir / combo_id / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    results = data.get("results", {})
    if not results:
        return None

    cers = [v.get("cer", 0.0) for v in results.values()]
    wers = [v.get("wer", 0.0) for v in results.values()]

    # No-diacritics variant
    nd_results = data.get("results_no_diacritics", {})
    nd_cers = [v.get("cer", 0.0) for v in nd_results.values()] if nd_results else []
    nd_wers = [v.get("wer", 0.0) for v in nd_results.values()] if nd_results else []

    out: dict = {
        "avg_cer":     round(sum(cers) / len(cers), 6),
        "avg_wer":     round(sum(wers) / len(wers), 6),
        "n_datasets":  len(results),
        "by_dataset":  {k: {"cer": v.get("cer", 0.0), "wer": v.get("wer", 0.0)}
                        for k, v in results.items()},
    }
    if nd_cers:
        out["avg_cer_nd"] = round(sum(nd_cers) / len(nd_cers), 6)
        out["avg_wer_nd"] = round(sum(nd_wers) / len(nd_wers), 6)
    return out


def _load_isolated_phase_avg_cer(
    phase_dir: Path,
    corrected_key: str = "corrected",
) -> Optional[float]:
    """Load average CER from an isolated phase's metrics.json."""
    metrics_path = phase_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", {})
        if not results:
            return None
        cers = [v.get("cer", 0.0) for v in results.values()]
        return round(sum(cers) / len(cers), 6)
    except (json.JSONDecodeError, OSError):
        return None


def _compute_per_sample_cers_from_corrections(corrections_path: Path) -> dict[str, float]:
    """Read corrections.jsonl and compute per-sample CER. Returns {sample_id: cer}."""
    result: dict[str, float] = {}
    if not corrections_path.exists():
        return result
    with open(corrections_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                gt = r.get("gt_text", "")
                corrected = r.get("corrected_text", r.get("ocr_text", ""))
                sid = r.get("sample_id", "")
                if gt and sid:
                    result[sid] = calculate_cer(gt, corrected)
            except (json.JSONDecodeError, Exception):  # noqa: BLE001
                pass
    return result


def run_summarize(
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
) -> None:
    """Cross-combo analysis: synergy, ablation, statistical tests, final comparison."""
    logger.info("=" * 60)
    logger.info("Phase 6 SUMMARIZE")
    logger.info("=" * 60)

    p6_stats_cfg = config.get("phase6", {}).get("stats", {})
    alpha = p6_stats_cfg.get("alpha", 0.05)
    n_bootstrap = p6_stats_cfg.get("n_bootstrap", 1000)

    # --- Load Phase 2 baseline ---
    p2_avg = _load_isolated_phase_avg_cer(phase2_dir)
    if p2_avg is None:
        logger.warning("Phase 2 metrics not found -- some analyses will be skipped.")

    # --- Load all combo avg metrics ---
    combo_metrics: dict[str, dict] = {}
    for combo_id in ALL_COMBOS:
        m = _load_combo_avg_metrics(results_dir, combo_id)
        if m is not None:
            combo_metrics[combo_id] = m
            logger.info(
                "  [%s] avg_cer=%.4f  avg_wer=%.4f  (%d datasets)",
                combo_id, m["avg_cer"], m["avg_wer"], m["n_datasets"],
            )
        else:
            logger.info("  [%s] not available (skipped).", combo_id)

    if not combo_metrics:
        logger.warning("No combo results found. Run export -> inference -> analyze first.")
        return

    # --- Load isolated phase avg CERs ---
    isolated_cers: dict[str, float] = {}
    for phase_key, phase_dir in [
        ("phase3", Path("results/phase3")),
        ("phase4", Path("results/phase4")),
        ("phase6_camel", Path("results/phase6")),
    ]:
        v = _load_isolated_phase_avg_cer(phase_dir)
        if v is not None:
            isolated_cers[phase_key] = v

    # Component -> isolated phase mapping for synergy
    _component_to_phase = {
        "confusion": "phase3",
        "self":      "phase4",
    }

    # --- Combinations summary ---
    combinations: dict[str, dict] = {}
    for cid in ALL_COMBOS:
        if cid not in combo_metrics:
            continue
        m = combo_metrics[cid]
        entry: dict = {
            "components": _active_component_names(cid),
            "avg_cer": m["avg_cer"],
            "avg_wer": m["avg_wer"],
        }
        if p2_avg is not None:
            delta_cer = p2_avg - m["avg_cer"]
            entry["delta_cer"] = round(delta_cer, 6)
            entry["cer_relative_pct"] = round((delta_cer / p2_avg * 100) if p2_avg > 0 else 0.0, 2)
        combinations[cid] = entry

    best_combo = (
        min(combo_metrics, key=lambda k: combo_metrics[k]["avg_cer"])
        if combo_metrics else None
    )
    combinations_summary = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_combos": len(combinations),
        },
        "phase2_baseline": {"avg_cer": p2_avg},
        "combinations": combinations,
        "best_combo": best_combo,
    }
    save_json(combinations_summary, results_dir / "combinations_summary.json")

    # --- Ablation summary ---
    # With 2 components (confusion + self), ablation is implicit:
    #   conf_only = conf_self minus self   (ablate self)
    #   self_only = conf_self minus conf   (ablate confusion)
    conf_self_cer = combo_metrics.get("conf_self", {}).get("avg_cer")
    best_camel_cer = combo_metrics.get("best_camel", {}).get("avg_cer")
    ablations: dict[str, dict] = {}

    # Ablate self-reflective (keep confusion only)
    if "conf_only" in combo_metrics and conf_self_cer is not None:
        m = combo_metrics["conf_only"]
        delta = m["avg_cer"] - conf_self_cer
        ablations["ablate_self"] = {
            "combo": "conf_only",
            "component_removed": "self",
            "components_remaining": ["confusion"],
            "avg_cer": m["avg_cer"],
            "delta_from_conf_self": round(delta, 6),
            "interpretation": (
                "Removing self-reflective hurts" if delta > 0
                else "Self-reflective was interfering" if delta < -0.001
                else "Self-reflective has minimal effect"
            ),
        }

    # Ablate confusion (keep self only)
    if "self_only" in combo_metrics and conf_self_cer is not None:
        m = combo_metrics["self_only"]
        delta = m["avg_cer"] - conf_self_cer
        ablations["ablate_confusion"] = {
            "combo": "self_only",
            "component_removed": "confusion",
            "components_remaining": ["self"],
            "avg_cer": m["avg_cer"],
            "delta_from_conf_self": round(delta, 6),
            "interpretation": (
                "Removing confusion matrix hurts" if delta > 0
                else "Confusion matrix was interfering" if delta < -0.001
                else "Confusion matrix has minimal effect"
            ),
        }

    # Ablate CAMeL (best_camel vs its base combo)
    if best_camel_cer is not None:
        base_combo_id = config.get("phase6", {}).get("best_combo")
        base_cer = combo_metrics.get(base_combo_id, {}).get("avg_cer") if base_combo_id else None
        if base_cer is not None:
            delta = base_cer - best_camel_cer
            ablations["ablate_camel"] = {
                "combo": "best_camel",
                "component_removed": "camel",
                "base_combo": base_combo_id,
                "avg_cer_with_camel": best_camel_cer,
                "avg_cer_without_camel": base_cer,
                "delta": round(delta, 6),
                "interpretation": (
                    "CAMeL post-processing helps" if delta > 0
                    else "CAMeL post-processing hurts" if delta < -0.001
                    else "CAMeL has minimal effect"
                ),
            }

    ablation_summary = {
        "meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "conf_self": {"avg_cer": conf_self_cer},
        "best_camel": {"avg_cer": best_camel_cer},
        "ablations": ablations,
        "interpretation": (
            "delta_from_conf_self > 0: removing this component hurts (CER rises). "
            "delta_from_conf_self < 0: component was interfering (CER drops without it)."
        ),
    }
    save_json(ablation_summary, results_dir / "ablation_summary.json")

    # --- Synergy analysis ---
    # With 2 components: does conf_self beat the sum of conf_only + self_only improvements?
    synergy: dict[str, dict] = {}
    if "conf_self" in combo_metrics and p2_avg is not None:
        delta_combined = p2_avg - combo_metrics["conf_self"]["avg_cer"]

        # Use Phase 6 combo results (conf_only, self_only) rather than isolated phases
        # because these are the same prompt framework (combined prompt) with one component
        delta_conf = (p2_avg - combo_metrics["conf_only"]["avg_cer"]) if "conf_only" in combo_metrics else None
        delta_self = (p2_avg - combo_metrics["self_only"]["avg_cer"]) if "self_only" in combo_metrics else None

        entry: dict = {
            "delta_combined": round(delta_combined, 6),
            "delta_conf_only": round(delta_conf, 6) if delta_conf is not None else None,
            "delta_self_only": round(delta_self, 6) if delta_self is not None else None,
        }
        if delta_conf is not None and delta_self is not None:
            sum_individual = delta_conf + delta_self
            synergy_val = delta_combined - sum_individual
            entry["sum_individual"] = round(sum_individual, 6)
            entry["synergy"] = round(synergy_val, 6)
            if synergy_val > 0.001:
                entry["interpretation"] = "Super-additive: components amplify each other."
            elif synergy_val < -0.001:
                entry["interpretation"] = "Sub-additive: components partially overlap or interfere."
            else:
                entry["interpretation"] = "Near-additive: components are approximately independent."

        synergy["conf_self"] = entry

    save_json(
        {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
         "methodology": "synergy = delta_combined - (delta_conf_only + delta_self_only); positive = super-additive.",
         "pairs": synergy},
        results_dir / "synergy_analysis.json",
    )

    # --- Statistical tests ---
    logger.info("Computing statistical tests ...")

    # Load per-sample CERs for Phase 2 baseline
    p2_cers_by_id: dict[str, float] = {}
    for p2_corrections_path in [
        phase2_dir / "corrections.jsonl",
        *(phase2_dir / ds["name"] / "corrections.jsonl"
          for ds in (config.get("datasets", [{}]) or [{}])
          if isinstance(ds, dict) and ds.get("name")),
    ]:
        p2_cers_by_id.update(_compute_per_sample_cers_from_corrections(p2_corrections_path))

    statistical_tests: dict[str, dict] = {}
    if p2_cers_by_id and combo_metrics:
        tester = StatsTester()
        systems_cers: dict[str, dict[str, float]] = {}

        for cid in combo_metrics:
            combo_dir_path = results_dir / cid
            # Try per-dataset corrections
            per_sample: dict[str, float] = {}
            for ds_dir in combo_dir_path.iterdir() if combo_dir_path.exists() else []:
                if ds_dir.is_dir():
                    per_sample.update(
                        _compute_per_sample_cers_from_corrections(ds_dir / "corrections.jsonl")
                    )
            if not per_sample:
                per_sample.update(
                    _compute_per_sample_cers_from_corrections(combo_dir_path / "corrections.jsonl")
                )
            if per_sample:
                systems_cers[cid] = per_sample

        if systems_cers:
            # Build paired lists (common sample_ids)
            all_sids = set(p2_cers_by_id.keys())
            baseline_list: list[float] = []
            systems_lists: dict[str, list[float]] = {cid: [] for cid in systems_cers}

            for sid in sorted(all_sids):
                if all(sid in systems_cers.get(cid, {}) for cid in systems_cers):
                    baseline_list.append(p2_cers_by_id.get(sid, 0.0))
                    for cid in systems_cers:
                        systems_lists[cid].append(systems_cers[cid][sid])

            if baseline_list and len(baseline_list) >= 10:
                stat_results = tester.compare_all(
                    baseline=baseline_list,
                    systems={cid: systems_lists[cid] for cid in systems_cers},
                    alpha=alpha,
                )
                statistical_tests = {
                    "meta": {
                        "test": "paired_ttest",
                        "baseline": "phase2",
                        "correction": "bonferroni",
                        "alpha_family": alpha,
                        "n_tests": len(stat_results),
                        "n_samples_paired": len(baseline_list),
                        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    "results": stat_results,
                }
                logger.info(
                    "Statistical tests: %d systems, %d paired samples.",
                    len(stat_results), len(baseline_list),
                )
            else:
                logger.warning(
                    "Not enough paired samples for statistical tests (%d). Skipping.",
                    len(baseline_list),
                )
        else:
            logger.warning("No per-sample CERs found for combos. Skipping statistical tests.")
    else:
        logger.warning(
            "Phase 2 per-sample CERs not found. Skipping statistical tests. "
            "Ensure results/phase2/corrections.jsonl (or per-dataset) exists."
        )

    if statistical_tests:
        save_json(statistical_tests, results_dir / "statistical_tests.json")

    # --- Final comparison table ---
    final_systems: dict[str, dict] = {}

    # Add isolated phases
    for ph_key, ph_dir in [
        ("phase1_ocr", Path("results/phase1")),
        ("phase2_zero_shot", phase2_dir),
        ("phase3_ocr_aware", Path("results/phase3")),
        ("phase4_self_reflective", Path("results/phase4")),
        ("phase6_camel", Path("results/phase6")),
    ]:
        # Phase 1 stores OCR metrics differently
        if ph_key == "phase1_ocr":
            m_path = ph_dir / "metrics.json"
            if m_path.exists():
                try:
                    with open(m_path, encoding="utf-8") as f:
                        d = json.load(f)
                    results_block = d.get("results", {})
                    if results_block:
                        ocr_cers = [v.get("cer", 0.0) for v in results_block.values()]
                        ocr_wers = [v.get("wer", 0.0) for v in results_block.values()]
                        final_systems[ph_key] = {
                            "avg_cer": round(sum(ocr_cers) / len(ocr_cers), 6),
                            "avg_wer": round(sum(ocr_wers) / len(ocr_wers), 6),
                        }
                except (json.JSONDecodeError, OSError):
                    pass
        else:
            cer = _load_isolated_phase_avg_cer(ph_dir)
            # Also get WER and ND variants
            m_path = ph_dir / "metrics.json"
            wer_v = None
            cer_nd = None
            wer_nd = None
            if m_path.exists():
                try:
                    with open(m_path, encoding="utf-8") as f:
                        d = json.load(f)
                    wers = [v.get("wer", 0.0) for v in d.get("results", {}).values()]
                    wer_v = round(sum(wers) / len(wers), 6) if wers else None
                    # No-diacritics from results_no_diacritics or per-dataset corrected_no_diacritics
                    nd_block = d.get("results_no_diacritics", {})
                    if not nd_block:
                        # Fallback: aggregate corrected_no_diacritics from nested results
                        nd_block = {}
                        for ds_key, ds_dir in [(k, ph_dir / k) for k in d.get("results", {})]:
                            ds_metrics = ds_dir / "metrics.json"
                            if ds_metrics.exists():
                                try:
                                    with open(ds_metrics, encoding="utf-8") as ndf:
                                        ds_data = json.load(ndf)
                                        nd_d = (
                                            ds_data.get("corrected_all_no_diacritics")
                                            or ds_data.get("corrected_no_diacritics", {})
                                        )
                                    if nd_d:
                                        nd_block[ds_key] = nd_d
                                except (json.JSONDecodeError, OSError):
                                    pass
                    if nd_block:
                        nd_cers = [v.get("cer", 0.0) for v in nd_block.values()]
                        nd_wers = [v.get("wer", 0.0) for v in nd_block.values()]
                        cer_nd = round(sum(nd_cers) / len(nd_cers), 6) if nd_cers else None
                        wer_nd = round(sum(nd_wers) / len(nd_wers), 6) if nd_wers else None
                except (json.JSONDecodeError, OSError):
                    pass
            if cer is not None:
                entry = {"avg_cer": cer, "avg_wer": wer_v}
                if cer_nd is not None:
                    entry["avg_cer_nd"] = cer_nd
                    entry["avg_wer_nd"] = wer_nd
                final_systems[ph_key] = entry

    # Add Phase 6 combos
    for cid, m in combo_metrics.items():
        entry: dict = {"avg_cer": m["avg_cer"], "avg_wer": m["avg_wer"]}
        if "avg_cer_nd" in m:
            entry["avg_cer_nd"] = m["avg_cer_nd"]
            entry["avg_wer_nd"] = m["avg_wer_nd"]
        final_systems[f"phase6_{cid}"] = entry

    save_json(
        {
            "meta": {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "note": "avg_cer / avg_wer averaged across all completed datasets.",
            },
            "systems": final_systems,
        },
        results_dir / "final_comparison.json",
    )

    # --- Paper tables ---
    _generate_paper_tables(final_systems, results_dir)

    # --- Report ---
    _generate_summary_report(
        combo_metrics=combo_metrics,
        combinations_summary=combinations_summary,
        ablation_summary=ablation_summary,
        synergy=synergy,
        statistical_tests=statistical_tests,
        results_dir=results_dir,
        config=config,
    )

    logger.info("=" * 60)
    logger.info("Phase 6 SUMMARIZE complete. Results in: %s", results_dir)
    logger.info("  combinations_summary.json")
    logger.info("  ablation_summary.json")
    logger.info("  synergy_analysis.json")
    logger.info("  statistical_tests.json")
    logger.info("  final_comparison.json")
    logger.info("  paper_tables.md")
    logger.info("  report.md")
    logger.info("=" * 60)


def _generate_paper_tables(
    final_systems: dict[str, dict],
    results_dir: Path,
) -> None:
    """Write LaTeX-ready markdown table to paper_tables.md."""
    lines: list[str] = []
    lines.append("# Phase 6: Paper Tables")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Order: phase1 -> phase2 -> phases 3-5 -> phase6 combos
    system_order = [
        ("phase1_ocr",              "Phase 1 (OCR only)"),
        ("phase2_zero_shot",        "Phase 2 (Zero-shot)"),
        ("phase3_ocr_aware",        "Phase 3 (OCR-Aware)"),
        ("phase4_self_reflective",  "Phase 4 (Self-Reflective)"),
        ("phase6_camel",            "Phase 6 Best CAMeL"),
    ]
    for cid in INFERENCE_COMBOS + sorted(CAMEL_COMBOS):
        system_order.append((f"phase6_{cid}", f"Phase 6: {COMBO_DESCRIPTIONS.get(cid, cid)}"))

    # Table 1: With Diacritics
    lines.append("## Main Results Table (With Diacritics)")
    lines.append("")
    lines.append("| System | Avg CER | Avg WER |")
    lines.append("|--------|---------|---------|")

    for key, label in system_order:
        if key not in final_systems:
            continue
        s = final_systems[key]
        cer_s = f"{s['avg_cer']*100:.2f}%" if s.get("avg_cer") is not None else "--"
        wer_s = f"{s['avg_wer']*100:.2f}%" if s.get("avg_wer") is not None else "--"
        lines.append(f"| {label} | {cer_s} | {wer_s} |")

    lines.append("")

    # Table 2: No Diacritics
    has_nd = any(s.get("avg_cer_nd") is not None for s in final_systems.values())
    if has_nd:
        lines.append("## Main Results Table (No Diacritics)")
        lines.append("")
        lines.append("| System | Avg CER (ND) | Avg WER (ND) |")
        lines.append("|--------|--------------|--------------|")

        for key, label in system_order:
            if key not in final_systems:
                continue
            s = final_systems[key]
            cer_nd = f"{s['avg_cer_nd']*100:.2f}%" if s.get("avg_cer_nd") is not None else "--"
            wer_nd = f"{s['avg_wer_nd']*100:.2f}%" if s.get("avg_wer_nd") is not None else "--"
            lines.append(f"| {label} | {cer_nd} | {wer_nd} |")

        lines.append("")

    lines.append("> avg_cer / avg_wer = macro-average across all evaluated datasets.")
    lines.append("")

    report_path = results_dir / "paper_tables.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Paper tables written to %s", report_path)


def _generate_summary_report(
    combo_metrics: dict[str, dict],
    combinations_summary: dict,
    ablation_summary: dict,
    synergy: dict,
    statistical_tests: dict,
    results_dir: Path,
    config: dict,
) -> None:
    """Write human-readable Phase 6 summary report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_name = config.get("model", {}).get("name", "unknown")
    lines: list[str] = []

    lines.append("# Phase 6 Report: Combinations & Ablation Study")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"Model: {model_name}")
    lines.append("")

    lines.append("## Combination Results vs Phase 2\n")
    combs = combinations_summary.get("combinations", {})
    p2_avg = (combinations_summary.get("phase2_baseline") or {}).get("avg_cer")
    if combs:
        lines.append("| Combo | Components | Avg CER | Delta CER |")
        lines.append("|-------|------------|---------|-----------|")
        for cid, c in combs.items():
            delta_s = (
                f"{c.get('cer_relative_pct', 0.0):+.1f}%"
                if "cer_relative_pct" in c else "—"
            )
            lines.append(
                f"| {cid} | {', '.join(c.get('components', []))} "
                f"| {c.get('avg_cer', 0)*100:.2f}% | {delta_s} |"
            )
    lines.append("")

    best = combinations_summary.get("best_combo")
    if best:
        lines.append(f"**Best combo**: `{best}` "
                     f"(avg CER {combo_metrics.get(best, {}).get('avg_cer', 0)*100:.2f}%)\n")

    lines.append("## Ablation Results\n")
    abls = ablation_summary.get("ablations", {})
    if abls:
        lines.append("| Ablation | Removed | Avg CER | Delta | Interpretation |")
        lines.append("|----------|---------|---------|-------|----------------|")
        for abl_id, a in abls.items():
            delta_key = "delta_from_conf_self" if "delta_from_conf_self" in a else "delta"
            delta_val = a.get(delta_key)
            delta_s = f"{delta_val*100:+.2f}%" if delta_val is not None else "---"
            removed = a.get("component_removed", "?")
            interp = a.get("interpretation", "---")
            cer_val = a.get("avg_cer") or a.get("avg_cer_with_camel", 0)
            lines.append(
                f"| {abl_id} | {removed} | {cer_val*100:.2f}% | {delta_s} | {interp} |"
            )
    lines.append("")

    lines.append("## Synergy Analysis\n")
    if synergy:
        lines.append("| Combo | Delta Combined | Sum Individual | Synergy | Interpretation |")
        lines.append("|-------|---------------|----------------|---------|----------------|")
        for cid, s in synergy.items():
            interp = s.get("interpretation", "---")
            lines.append(
                f"| {cid} "
                f"| {s.get('delta_combined', 0)*100:+.3f}% "
                f"| {s.get('sum_individual', 0)*100:+.3f}% "
                f"| {s.get('synergy', 0)*100:+.3f}% "
                f"| {interp} |"
            )
    lines.append("")

    if statistical_tests:
        stats_results = statistical_tests.get("results", {})
        lines.append("## Statistical Significance (Bonferroni-corrected)\n")
        lines.append("| System | t-stat | p-value | p-corrected | Significant | Cohen's d |")
        lines.append("|--------|--------|---------|-------------|-------------|-----------|")
        for sname, sr in stats_results.items():
            sig = "Yes *" if sr.get("significant") else "No"
            lines.append(
                f"| {sname} "
                f"| {sr.get('t_stat', 0):.3f} "
                f"| {sr.get('p_value', 1):.4f} "
                f"| {sr.get('p_corrected', 1):.4f} "
                f"| {sig} "
                f"| {sr.get('cohens_d', 0):.3f} |"
            )
        lines.append("")

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    setup_logging(results_dir)
    logger.info("Phase 6: Combinations & Ablation Study  (mode=%s)", args.mode)
    logger.info("Results dir: %s", results_dir)

    config = load_config(args.config)
    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")
    active_datasets = resolve_datasets(config, args.datasets)
    all_dataset_names = [entry["name"] for entry in config.get("datasets", [])]

    sample_ids: Optional[set[str]] = None
    if args.sample_list:
        sample_ids, sl_datasets = load_sample_list(args.sample_list)
        if not args.datasets:
            active_datasets = sl_datasets
        logger.info("Sample list loaded: %d sample IDs from %s", len(sample_ids), args.sample_list)

    # ------------------------------------------------------------------
    # SUMMARIZE (no --combo needed)
    # ------------------------------------------------------------------
    if args.mode == "summarize":
        run_summarize(config=config, results_dir=results_dir, phase2_dir=args.phase2_dir)
        return

    # ------------------------------------------------------------------
    # Validate --combo for other modes
    # ------------------------------------------------------------------
    if args.combo is None:
        print(
            f"ERROR: --combo is required for --mode {args.mode}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve combo list
    if args.combo == "all":
        if args.mode in ("validate",):
            print(
                "ERROR: --combo all is not supported for --mode validate. "
                "Specify combo explicitly: best_camel.",
                file=sys.stderr,
            )
            sys.exit(1)
        combo_ids = INFERENCE_COMBOS
    else:
        if args.combo not in ALL_COMBOS:
            print(
                f"ERROR: Unknown combo '{args.combo}'. "
                f"Choose from: {', '.join(ALL_COMBOS)} or 'all'.",
                file=sys.stderr,
            )
            sys.exit(1)
        combo_ids = [args.combo]

    for combo_id in combo_ids:
        logger.info(
            "Processing combo: %s — %s", combo_id, COMBO_DESCRIPTIONS.get(combo_id, combo_id)
        )
        combo_dir = results_dir / combo_id

        # ------------------------------------------------------------------
        # EXPORT
        # ------------------------------------------------------------------
        if args.mode == "export":
            if combo_id in CAMEL_COMBOS:
                logger.warning(
                    "[%s] is a CAMeL combo -- no export needed. Use --mode validate.", combo_id
                )
                continue

            run_export(
                combo_id=combo_id,
                config=config,
                active_datasets=active_datasets,
                all_dataset_names=all_dataset_names,
                combo_dir=combo_dir,
                phase1_dir=args.phase1_dir,
                limit=limit,
                force=args.force,
                sample_ids=sample_ids,
            )

        # ------------------------------------------------------------------
        # ANALYZE
        # ------------------------------------------------------------------
        elif args.mode == "analyze":
            if combo_id in CAMEL_COMBOS:
                logger.warning(
                    "[%s] is a CAMeL combo -- use --mode validate instead.", combo_id
                )
                continue

            analyze_errors = not args.no_error_analysis
            all_corrected, all_comparisons = run_analyze(
                combo_id=combo_id,
                config=config,
                active_datasets=active_datasets,
                combo_dir=combo_dir,
                phase2_dir=args.phase2_dir,
                limit=limit,
                force=args.force,
                analyze_errors=analyze_errors,
            )

            if not all_corrected:
                logger.error("[%s] No datasets successfully processed.", combo_id)
                continue

            aggregate_combo_results(
                combo_id, all_corrected, all_comparisons, config, combo_dir, limit
            )
            print_combo_summary(combo_id, all_corrected, all_comparisons, combo_dir)
            write_corrections_report(
                corrections_path=combo_dir,
                output_path=combo_dir / "sample_report.txt",
                title=f"Phase 6 -- {combo_id}",
            )
            logger.info("[%s] complete. Results in: %s", combo_id, combo_dir)

        # ------------------------------------------------------------------
        # VALIDATE (CAMeL combos)
        # ------------------------------------------------------------------
        elif args.mode == "validate":
            if combo_id not in CAMEL_COMBOS:
                logger.warning(
                    "[%s] is not a CAMeL combo. Use --mode export and --mode analyze instead.",
                    combo_id,
                )
                continue

            all_corrected, all_comparisons = run_validate(
                combo_id=combo_id,
                config=config,
                active_datasets=active_datasets,
                results_dir=results_dir,
                phase2_dir=args.phase2_dir,
                limit=limit,
                force=args.force,
            )

            if not all_corrected:
                logger.error("[%s] No datasets successfully processed.", combo_id)
                continue

            aggregate_combo_results(
                combo_id, all_corrected, all_comparisons, config, combo_dir, limit
            )
            print_combo_summary(combo_id, all_corrected, all_comparisons, combo_dir)
            logger.info("[%s] validate complete. Results in: %s", combo_id, combo_dir)

    logger.info("Phase 6 done.")


if __name__ == "__main__":
    main()
