#!/usr/bin/env python3
"""Phase 6: Combinations & Ablation Study.

Tests combinations of knowledge sources from Phases 3-5 and measures each
component's contribution via ablation.

Experiment set
--------------
Inference-based combos (need export -> Kaggle -> analyze):
  pair_conf_rules    Confusion + Rules
  pair_conf_fewshot  Confusion + Few-Shot
  pair_conf_rag      Confusion + RAG
  pair_rules_fewshot Rules + Few-Shot
  full_prompt        Confusion + Rules + Few-Shot + RAG
  abl_no_confusion   Rules + Few-Shot + RAG
  abl_no_rules       Confusion + Few-Shot + RAG
  abl_no_fewshot     Confusion + Rules + RAG
  abl_no_rag         Confusion + Rules + Few-Shot

CAMeL post-processing combos (local only, no Kaggle step):
  pair_best_camel    Best pair (config.phase6.pair_best) + CAMeL revert
  full_system        full_prompt + CAMeL revert

Note: abl_no_camel == full_prompt (no separate run needed).

Pipeline per inference combo
-----------------------------
  1. LOCAL:  python pipelines/run_phase6.py --combo pair_conf_rules --mode export
  2. REMOTE: python scripts/infer.py \\
                 --input  results/phase6/pair_conf_rules/inference_input.jsonl \\
                 --output results/phase6/pair_conf_rules/corrections.jsonl
  3. LOCAL:  python pipelines/run_phase6.py --combo pair_conf_rules --mode analyze

CAMeL combos (no Kaggle step):
  python pipelines/run_phase6.py --combo full_system  --mode validate
  python pipelines/run_phase6.py --combo pair_best_camel --mode validate

Cross-combo summary (after all combos analyzed/validated):
  python pipelines/run_phase6.py --mode summarize

Usage
-----
    # Smoke test: export + infer + analyze one pair
    python pipelines/run_phase6.py --combo pair_conf_rules --mode export \\
        --datasets KHATT-train --limit 50
    python scripts/infer.py \\
        --input  results/phase6/pair_conf_rules/inference_input.jsonl \\
        --output results/phase6/pair_conf_rules/corrections.jsonl \\
        --datasets KHATT-train --limit 50
    python pipelines/run_phase6.py --combo pair_conf_rules --mode analyze \\
        --datasets KHATT-train

    # Run all inference combos in sequence:
    python pipelines/run_phase6.py --combo all --mode export
    # (then run Kaggle for each combo)
    python pipelines/run_phase6.py --combo all --mode analyze

    # After setting pair_best in config.yaml:
    python pipelines/run_phase6.py --combo full_system     --mode validate
    python pipelines/run_phase6.py --combo pair_best_camel --mode validate

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
    RulesLoader,
    QALBLoader,
)
from src.core.rag_retriever import RAGRetriever
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from src.analysis.metrics import MetricResult, calculate_metrics, calculate_cer
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.analysis.stats_tester import StatsTester
from src.linguistic.morphology import MorphAnalyzer
from src.linguistic.validator import WordValidator, TextCorrectionResult
from pipelines._utils import resolve_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combo definitions
# ---------------------------------------------------------------------------

# (use_confusion, use_rules, use_fewshot, use_rag)
COMBO_COMPONENTS: dict[str, tuple[bool, bool, bool, bool]] = {
    "pair_conf_rules":    (True,  True,  False, False),
    "pair_conf_fewshot":  (True,  False, True,  False),
    "pair_conf_rag":      (True,  False, False, True),
    "pair_rules_fewshot": (False, True,  True,  False),
    "full_prompt":        (True,  True,  True,  True),
    "abl_no_confusion":   (False, True,  True,  True),
    "abl_no_rules":       (True,  False, True,  True),
    "abl_no_fewshot":     (True,  True,  False, True),
    "abl_no_rag":         (True,  True,  True,  False),
}

CAMEL_COMBOS = {"pair_best_camel", "full_system"}

INFERENCE_COMBOS = list(COMBO_COMPONENTS.keys())
ALL_COMBOS = INFERENCE_COMBOS + sorted(CAMEL_COMBOS)

COMBO_DESCRIPTIONS: dict[str, str] = {
    "pair_conf_rules":    "Confusion + Rules",
    "pair_conf_fewshot":  "Confusion + Few-Shot",
    "pair_conf_rag":      "Confusion + RAG",
    "pair_rules_fewshot": "Rules + Few-Shot",
    "full_prompt":        "All Prompt Components (Confusion + Rules + Few-Shot + RAG)",
    "abl_no_confusion":   "Full Prompt minus Confusion",
    "abl_no_rules":       "Full Prompt minus Rules",
    "abl_no_fewshot":     "Full Prompt minus Few-Shot",
    "abl_no_rag":         "Full Prompt minus RAG",
    "pair_best_camel":    "Best Pair + CAMeL Validation",
    "full_system":        "Full System (All + CAMeL)",
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
            "validate  -> apply CAMeL post-processing (pair_best_camel / full_system); "
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
        base = "full_prompt" if combo_id == "full_system" else "pair_best"
        return [base, "camel"]
    flags = COMBO_COMPONENTS.get(combo_id, (False, False, False, False))
    names = []
    labels = ["confusion", "rules", "fewshot", "rag"]
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


def build_rules_context(config: dict) -> str:
    """Build rules context string (Phase 4A style)."""
    phase4_cfg = config.get("phase4", {}).get("rules", {})
    style = phase4_cfg.get("format_style", "compact_arabic")
    cats = phase4_cfg.get("categories")
    n = phase4_cfg.get("n_rules")

    loader = RulesLoader()
    rules = loader.load(categories=cats)

    if not rules:
        logger.warning("No rules loaded -- rules_context empty.")
        return ""

    context = loader.format_for_prompt(rules, n=n, style=style)
    logger.info("Rules context: %d rules, %d chars (style=%s).", len(rules), len(context), style)
    return context


def build_examples_context(config: dict) -> str:
    """Build few-shot examples context string (Phase 4B style)."""
    phase4b_cfg = config.get("phase4", {}).get("few_shot", {})
    seed = phase4b_cfg.get("seed", 42)
    max_length = phase4b_cfg.get("max_length", 100)
    min_length = phase4b_cfg.get("min_length", 10)
    max_words_changed = phase4b_cfg.get("max_words_changed", 4)
    num_examples = phase4b_cfg.get("num_examples", 5)
    selection = phase4b_cfg.get("selection", "diverse")
    format_style = phase4b_cfg.get("format_style", "inline_arabic")
    years = phase4b_cfg.get("years", ["2014"])

    loader = QALBLoader(config)
    try:
        pairs = loader.load(splits=["train"], years=years)
    except Exception as exc:  # noqa: BLE001
        logger.warning("QALB load failed: %s -- examples_context empty.", exc)
        return ""

    if not pairs:
        logger.warning("No QALB pairs loaded -- examples_context empty.")
        return ""

    filtered = loader.filter_ocr_relevant(
        pairs,
        max_length=max_length,
        min_length=min_length,
        max_words_changed=max_words_changed,
    )
    if not filtered:
        logger.warning("No OCR-relevant QALB pairs -- examples_context empty.")
        return ""

    selected = loader.select(filtered, n=num_examples, strategy=selection, seed=seed)
    context = loader.format_for_prompt(selected, style=format_style)
    logger.info(
        "Examples context: %d examples (%s), %d chars.",
        len(selected), selection, len(context),
    )
    return context


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


def _maybe_split_combined_corrections(combo_dir: Path) -> None:
    """Split combined corrections.jsonl into per-dataset files if needed."""
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
            s1.gt_text   = cs.sample.gt_text
            s1.ocr_text  = cs.sample.ocr_text

            s2 = _Stub()
            s2.sample_id = cs.sample.sample_id
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

    logger.info("=" * 60)
    logger.info(
        "Analyzing [%s / %s]: %d samples, %d failed",
        combo_id, dataset_key, n, n_failed,
    )

    # Step 1: CER/WER
    logger.info("[%s / %s] Calculating CER/WER ...", combo_id, dataset_key)
    corrected_result = calculate_metrics(
        corrected_samples, dataset_name=dataset_key, text_field="corrected_text"
    )

    builder = PromptBuilder()
    metrics_extra = {
        "prompt_type":              "combined",
        "prompt_version":           builder.combined_prompt_version,
        "total_prompt_tokens":      total_prompt_tokens,
        "total_output_tokens":      total_output_tokens,
        "total_latency_s":          round(total_latency, 2),
        "avg_latency_per_sample_s": round(avg_latency, 3),
        "failed_samples":           n_failed,
    }
    metrics_json = {
        "meta":      make_meta(combo_id, dataset_key, n, config, limit, extra=metrics_extra),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")
    logger.info(
        "[%s / %s] CER=%.2f%%  WER=%.2f%%",
        combo_id, dataset_key, corrected_result.cer * 100, corrected_result.wer * 100,
    )

    # Step 2: Comparison vs Phase 2
    p2_metrics = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_metrics is not None:
        p2_cer = float(p2_metrics.get("cer", 0.0))
        p2_wer = float(p2_metrics.get("wer", 0.0))
        cer_delta = p2_cer - corrected_result.cer
        wer_delta = p2_wer - corrected_result.wer
        cer_rel = (cer_delta / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(
                combo_id, dataset_key, n, config, limit,
                extra={"comparison": f"phase6_{combo_id}_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
            },
            f"phase6_{combo_id}_corrected": {
                "cer": round(corrected_result.cer, 6),
                "wer": round(corrected_result.wer, 6),
            },
            "delta": {
                "cer_absolute":     round(cer_delta, 6),
                "wer_absolute":     round(wer_delta, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
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
            "[%s / %s] Phase2->Phase6 CER: %.2f%% -> %.2f%% (%+.1f%%)",
            combo_id, dataset_key, p2_cer * 100, corrected_result.cer * 100, cer_rel,
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

    return corrected_result


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
) -> None:
    """Build inference_input.jsonl for one inference combo.

    Loads contexts once, then iterates over all active datasets.
    Confusion context is per-dataset; rules/examples are global; retrieval is per-sample.
    """
    use_conf, use_rules, use_fewshot, use_rag = COMBO_COMPONENTS[combo_id]

    output_path = combo_dir / "inference_input.jsonl"
    combo_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 6 EXPORT: combo=%s  (%s)", combo_id, COMBO_DESCRIPTIONS[combo_id])
    logger.info(
        "  Components: confusion=%s  rules=%s  fewshot=%s  rag=%s",
        use_conf, use_rules, use_fewshot, use_rag,
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

    rules_context = build_rules_context(config) if use_rules else ""
    examples_context = build_examples_context(config) if use_fewshot else ""

    # --- RAG retriever ---
    retriever: Optional[RAGRetriever] = None
    if use_rag:
        p5_ret_cfg = config.get("phase5", {}).get("retrieval", {})
        top_k = p5_ret_cfg.get("top_k") or config.get("rag", {}).get("top_k", 3)
        min_score = p5_ret_cfg.get("min_score", 0.0)
        format_style_rag = p5_ret_cfg.get("format_style", "numbered_arabic")

        index_path = Path("results/phase5/faiss.index")
        if not index_path.exists():
            logger.error(
                "FAISS index not found: %s\n"
                "Run: python pipelines/run_phase5.py --mode build",
                index_path,
            )
            sys.exit(1)

        retriever = RAGRetriever(config)
        try:
            retriever.load_index(index_path)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("Cannot load RAG retriever: %s", exc)
            sys.exit(1)

        if not retriever.enabled:
            logger.error("RAGRetriever disabled -- install sentence-transformers and faiss-cpu.")
            sys.exit(1)

        logger.info("RAG retriever loaded: %d sentences in index.", retriever.corpus_size)
    else:
        top_k = 0
        min_score = 0.0
        format_style_rag = "numbered_arabic"

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

            # Determine prompt_type
            any_context = any([confusion_context, rules_context, examples_context, use_rag])
            prompt_type = "combined" if any_context else "zero_shot"

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            n_with_rag = 0
            for sample in tqdm(samples, desc=f"  Export {ds_key}", unit="sample"):
                retrieval_context = ""
                retrieval_scores: list[float] = []
                retrieved_k = 0

                if retriever is not None and retriever.enabled:
                    chunks = retriever.retrieve(sample.ocr_text, k=top_k, min_score=min_score)
                    retrieval_context = retriever.format_for_prompt(
                        chunks, style=format_style_rag
                    )
                    retrieval_scores = [c.score for c in chunks]
                    retrieved_k = len(chunks)
                    if chunks:
                        n_with_rag += 1

                record: dict = {
                    "sample_id":         sample.sample_id,
                    "dataset":           ds_key,
                    "ocr_text":          sample.ocr_text,
                    "gt_text":           sample.gt_text,
                    "prompt_type":       prompt_type,
                    "combo_id":          combo_id,
                    "prompt_version":    PromptBuilder.COMBINED_PROMPT_VERSION,
                    "confusion_context": confusion_context or None,
                    "rules_context":     rules_context or None,
                    "examples_context":  examples_context or None,
                    "retrieval_context": retrieval_context or None,
                }
                if use_rag:
                    record["retrieved_k"] = retrieved_k
                    record["retrieval_scores"] = retrieval_scores
                if use_conf:
                    record["confusion_source"] = conf_source

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info(
                "Exported %d samples for %s (conf_src=%s, rag_hits=%d/%d).",
                len(samples), ds_key, conf_source,
                n_with_rag if use_rag else 0, len(samples),
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
    _maybe_split_combined_corrections(combo_dir)

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

    # CER/WER
    corrected_result = calculate_metrics(
        corrected_samples, dataset_name=dataset_key, text_field="corrected_text"
    )
    metrics_json = {
        "meta": make_meta(
            combo_id, dataset_key, n, config, limit,
            extra={"prompt_type": "camel_validation", "total_reverted": total_reverted},
        ),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")

    # Comparison vs Phase 2
    p2_metrics = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_metrics is not None:
        p2_cer = float(p2_metrics.get("cer", 0.0))
        p2_wer = float(p2_metrics.get("wer", 0.0))
        cer_delta = p2_cer - corrected_result.cer
        wer_delta = p2_wer - corrected_result.wer
        cer_rel = (cer_delta / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta / p2_wer * 100) if p2_wer > 0 else 0.0
        comparison = {
            "meta": make_meta(
                combo_id, dataset_key, n, config, limit,
                extra={"comparison": f"phase6_{combo_id}_vs_phase2"},
            ),
            "phase2_baseline": {"cer": round(p2_cer, 6), "wer": round(p2_wer, 6)},
            f"phase6_{combo_id}_corrected": {
                "cer": round(corrected_result.cer, 6),
                "wer": round(corrected_result.wer, 6),
            },
            "delta": {
                "cer_absolute": round(cer_delta, 6),
                "wer_absolute": round(wer_delta, 6),
                "cer_relative_pct": round(cer_rel, 2),
                "wer_relative_pct": round(wer_rel, 2),
            },
            "interpretation": (
                f"CER {'reduced' if cer_delta >= 0 else 'increased'} by "
                f"{abs(cer_rel):.1f}% vs Phase 2 "
                f"({p2_cer*100:.2f}% -> {corrected_result.cer*100:.2f}%)."
            ),
        }
        save_json(comparison, out_dir / "comparison_vs_phase2.json")
        logger.info(
            "[%s / %s] CER: %.2f%% -> %.2f%% (%+.1f%%)",
            combo_id, dataset_key, p2_cer * 100, corrected_result.cer * 100, cer_rel,
        )

    return corrected_result


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
    if combo_id == "full_system":
        base_combo_id = "full_prompt"
    elif combo_id == "pair_best_camel":
        base_combo_id = config.get("phase6", {}).get("pair_best")
        if not base_combo_id:
            logger.error(
                "phase6.pair_best not set in config.yaml. "
                "Review pair results and set it before running pair_best_camel."
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

    # Initialise CAMeL validator once
    try:
        analyzer = MorphAnalyzer(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MorphAnalyzer init failed: %s -- validation passthrough.", exc)
        analyzer = MorphAnalyzer({})

    validator = WordValidator(analyzer)
    if not analyzer.enabled:
        logger.warning(
            "CAMeL Tools not available -- validate_correction() will pass text through unchanged."
        )

    # Split base corrections if needed
    _maybe_split_combined_corrections(base_combo_dir)

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


def aggregate_combo_results(
    combo_id: str,
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    config: dict,
    combo_dir: Path,
    limit: Optional[int],
) -> None:
    """Write aggregated metrics.json and comparison.json for one combo."""
    builder = PromptBuilder()
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
    }
    save_json(output, combo_dir / "metrics.json")

    if all_comparisons:
        cmp_output = {
            "meta": {
                "phase":   "phase6",
                "combo_id": combo_id,
                "comparison": f"phase6_{combo_id}_vs_phase2",
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
) -> None:
    print("\n" + "=" * 80)
    print(f"PHASE6 [{combo_id}] SUMMARY — {COMBO_DESCRIPTIONS.get(combo_id, combo_id)}")
    print("=" * 80)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'This CER':>9} {'D CER':>8} {'This WER':>9} {'N':>6}")
    print("-" * 80)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_cer = cmp.get("phase2_baseline", {}).get("cer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        p2_str    = f"{p2_cer*100:.2f}%" if cmp else "N/A"
        delta_str = f"{cer_rel:+.1f}%"   if cmp else "N/A"
        print(
            f"{ds:<28} {p2_str:>8} {r.cer*100:>8.2f}% {delta_str:>8} "
            f"{r.wer*100:>8.2f}% {r.num_samples:>6}"
        )
    print("=" * 80)


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

    return {
        "avg_cer":     round(sum(cers) / len(cers), 6),
        "avg_wer":     round(sum(wers) / len(wers), 6),
        "n_datasets":  len(results),
        "by_dataset":  {k: {"cer": v.get("cer", 0.0), "wer": v.get("wer", 0.0)}
                        for k, v in results.items()},
    }


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
        ("phase4a", Path("results/phase4a")),
        ("phase4b", Path("results/phase4b")),
        ("phase4c", Path("results/phase4c")),
        ("phase5", Path("results/phase5")),
    ]:
        v = _load_isolated_phase_avg_cer(phase_dir)
        if v is not None:
            isolated_cers[phase_key] = v

    # Component -> isolated phase mapping for synergy
    _component_to_phase = {
        "confusion": "phase3",
        "rules":     "phase4a",
        "fewshot":   "phase4b",
        "rag":       "phase5",
    }

    # --- Combinations summary ---
    pair_combos = ["pair_conf_rules", "pair_conf_fewshot", "pair_conf_rag", "pair_rules_fewshot"]
    combinations: dict[str, dict] = {}
    for cid in pair_combos + ["pair_best_camel", "full_prompt", "full_system"]:
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
    full_prompt_cer = combo_metrics.get("full_prompt", {}).get("avg_cer")
    full_system_cer = combo_metrics.get("full_system", {}).get("avg_cer")
    abl_combos = ["abl_no_confusion", "abl_no_rules", "abl_no_fewshot", "abl_no_rag"]
    ablations: dict[str, dict] = {}
    for abl_id in abl_combos:
        if abl_id not in combo_metrics:
            continue
        m = combo_metrics[abl_id]
        entry = {
            "components_remaining": _active_component_names(abl_id),
            "avg_cer": m["avg_cer"],
        }
        if full_system_cer is not None:
            delta_from_full = m["avg_cer"] - full_system_cer
            entry["delta_from_full_system"] = round(delta_from_full, 6)
        if full_prompt_cer is not None:
            delta_from_full_prompt = m["avg_cer"] - full_prompt_cer
            entry["delta_from_full_prompt"] = round(delta_from_full_prompt, 6)
        ablations[abl_id] = entry

    # abl_no_camel = full_prompt (no separate run)
    if full_prompt_cer is not None:
        ablations["abl_no_camel"] = {
            "components_remaining": _active_component_names("full_prompt"),
            "avg_cer": full_prompt_cer,
            "note": "Same as full_prompt (no CAMeL applied). No separate inference needed.",
            "delta_from_full_system": (
                round(full_prompt_cer - full_system_cer, 6)
                if full_system_cer is not None else None
            ),
        }

    ablation_summary = {
        "meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "full_prompt": {"avg_cer": full_prompt_cer},
        "full_system": {"avg_cer": full_system_cer},
        "ablations": ablations,
        "interpretation": (
            "delta_from_full_system > 0: removing this component hurts (CER rises). "
            "delta_from_full_system < 0: component was interfering (CER drops without it)."
        ),
    }
    save_json(ablation_summary, results_dir / "ablation_summary.json")

    # --- Synergy analysis (pairs only) ---
    synergy: dict[str, dict] = {}
    pair_component_map = {
        "pair_conf_rules":    ("confusion", "rules"),
        "pair_conf_fewshot":  ("confusion", "fewshot"),
        "pair_conf_rag":      ("confusion", "rag"),
        "pair_rules_fewshot": ("rules", "fewshot"),
    }
    for cid, (comp_a, comp_b) in pair_component_map.items():
        if cid not in combo_metrics or p2_avg is None:
            continue
        delta_pair = p2_avg - combo_metrics[cid]["avg_cer"]
        ph_a = _component_to_phase.get(comp_a)
        ph_b = _component_to_phase.get(comp_b)
        delta_a = (p2_avg - isolated_cers[ph_a]) if ph_a and ph_a in isolated_cers else None
        delta_b = (p2_avg - isolated_cers[ph_b]) if ph_b and ph_b in isolated_cers else None

        entry: dict = {
            "delta_pair": round(delta_pair, 6),
            "delta_a": round(delta_a, 6) if delta_a is not None else None,
            "delta_b": round(delta_b, 6) if delta_b is not None else None,
        }
        if delta_a is not None and delta_b is not None:
            sum_individual = delta_a + delta_b
            synergy_val = delta_pair - sum_individual
            entry["sum_individual"] = round(sum_individual, 6)
            entry["synergy"] = round(synergy_val, 6)
            if synergy_val > 0.001:
                entry["interpretation"] = "Super-additive: components amplify each other."
            elif synergy_val < -0.001:
                entry["interpretation"] = "Sub-additive: components partially overlap or interfere."
            else:
                entry["interpretation"] = "Near-additive: components are approximately independent."

        synergy[cid] = entry

    save_json(
        {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
         "methodology": "synergy = delta_pair - (delta_a + delta_b); positive = super-additive.",
         "pairs": synergy},
        results_dir / "synergy_analysis.json",
    )

    # --- Statistical tests ---
    logger.info("Computing statistical tests ...")

    # Load per-sample CERs for Phase 2 baseline
    p2_cers_by_id: dict[str, float] = {}
    for p2_corrections_path in [
        phase2_dir / "corrections.jsonl",
        *(phase2_dir / ds / "corrections.jsonl"
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
        ("phase3_confusion", Path("results/phase3")),
        ("phase4a_rules", Path("results/phase4a")),
        ("phase4b_fewshot", Path("results/phase4b")),
        ("phase4c_camel", Path("results/phase4c")),
        ("phase5_rag", Path("results/phase5")),
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
            # Also get WER
            m_path = ph_dir / "metrics.json"
            wer_v = None
            if m_path.exists():
                try:
                    with open(m_path, encoding="utf-8") as f:
                        d = json.load(f)
                    wers = [v.get("wer", 0.0) for v in d.get("results", {}).values()]
                    wer_v = round(sum(wers) / len(wers), 6) if wers else None
                except (json.JSONDecodeError, OSError):
                    pass
            if cer is not None:
                final_systems[ph_key] = {"avg_cer": cer, "avg_wer": wer_v}

    # Add Phase 6 combos
    for cid, m in combo_metrics.items():
        final_systems[f"phase6_{cid}"] = {"avg_cer": m["avg_cer"], "avg_wer": m["avg_wer"]}

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
    lines.append("## Main Results Table")
    lines.append("")
    lines.append("| System | Avg CER | Avg WER |")
    lines.append("|--------|---------|---------|")

    # Order: phase1 -> phase2 -> phases 3-5 -> phase6 combos
    system_order = [
        ("phase1_ocr",           "Phase 1 (OCR only)"),
        ("phase2_zero_shot",     "Phase 2 (Zero-shot)"),
        ("phase3_confusion",     "Phase 3 (+Confusion)"),
        ("phase4a_rules",        "Phase 4A (+Rules)"),
        ("phase4b_fewshot",      "Phase 4B (+Few-Shot)"),
        ("phase4c_camel",        "Phase 4C (+CAMeL)"),
        ("phase5_rag",           "Phase 5 (+RAG)"),
    ]
    for cid in INFERENCE_COMBOS + sorted(CAMEL_COMBOS):
        system_order.append((f"phase6_{cid}", f"Phase 6: {COMBO_DESCRIPTIONS.get(cid, cid)}"))

    for key, label in system_order:
        if key not in final_systems:
            continue
        s = final_systems[key]
        cer_s = f"{s['avg_cer']*100:.2f}%" if s.get("avg_cer") is not None else "—"
        wer_s = f"{s['avg_wer']*100:.2f}%" if s.get("avg_wer") is not None else "—"
        lines.append(f"| {label} | {cer_s} | {wer_s} |")

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
    full_s = (ablation_summary.get("full_system") or {}).get("avg_cer")
    if abls:
        lines.append("| Ablation | Avg CER | Delta from Full System |")
        lines.append("|----------|---------|------------------------|")
        for abl_id, a in abls.items():
            delta_s = (
                f"{a.get('delta_from_full_system', 0)*100:+.2f}%"
                if "delta_from_full_system" in a else "—"
            )
            lines.append(
                f"| {abl_id} | {a.get('avg_cer', 0)*100:.2f}% | {delta_s} |"
            )
    lines.append("")

    lines.append("## Synergy Analysis\n")
    if synergy:
        lines.append("| Pair | Delta Pair | Sum Individual | Synergy | Interpretation |")
        lines.append("|------|-----------|----------------|---------|----------------|")
        for cid, s in synergy.items():
            interp = s.get("interpretation", "—")
            lines.append(
                f"| {cid} "
                f"| {s.get('delta_pair', 0)*100:+.3f}% "
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
                "Specify combo explicitly: pair_best_camel or full_system.",
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
            print_combo_summary(combo_id, all_corrected, all_comparisons)
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
            print_combo_summary(combo_id, all_corrected, all_comparisons)
            logger.info("[%s] validate complete. Results in: %s", combo_id, combo_dir)

    logger.info("Phase 6 done.")


if __name__ == "__main__":
    main()
