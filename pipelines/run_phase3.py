#!/usr/bin/env python3
"""Phase 3: OCR-Aware Prompting (Confusion Matrix Injection).

Isolated experiment: compares against Phase 2 (zero-shot) baseline only.
Each dataset uses its Phase 1 confusion matrix; sparse datasets fall back
to a pooled matrix for the same dataset type (PATS-A01 or KHATT).

Three-stage pipeline (no local GPU required):

  --mode export   -> Build confusion-injected inference_input.jsonl
  --mode analyze  -> Load corrections.jsonl, compute metrics and reports
  --mode full     -> End-to-end with API backend (requires model.backend='api')

Typical workflow
----------------
1. LOCAL:  python pipelines/run_phase3.py --mode export
2. REMOTE: git clone <repo> && python scripts/infer.py \\
               --input  <path>/inference_input.jsonl \\
               --output <path>/corrections.jsonl \\
               --model  Qwen/Qwen3-4B-Instruct-2507
           (see notebooks/kaggle_setup.ipynb or notebooks/colab_setup.ipynb)
3. LOCAL:  copy corrections.jsonl to results/phase3/
           then run: python pipelines/run_phase3.py --mode analyze

Usage
-----
    python pipelines/run_phase3.py --mode export
    python pipelines/run_phase3.py --mode export   --limit 50
    python pipelines/run_phase3.py --mode export   --top-n 5
    python pipelines/run_phase3.py --mode export   --top-n 20 --format grouped_arabic
    python pipelines/run_phase3.py --mode export   --datasets KHATT-train
    python pipelines/run_phase3.py --mode analyze
    python pipelines/run_phase3.py --mode analyze  --datasets KHATT-train
    python pipelines/run_phase3.py --mode analyze  --no-error-analysis
    python pipelines/run_phase3.py --mode analyze  --no-confusion-impact
    python pipelines/run_phase3.py --mode export   --force
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
from src.data.knowledge_base import ConfusionMatrixLoader, ConfusionPair
from src.analysis.metrics import MetricResult, calculate_metrics, compare_metrics
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
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
        description="Phase 3: OCR-Aware Prompting for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["export", "analyze", "full"],
        help=(
            "export  -> produce inference_input.jsonl with confusion context; "
            "analyze -> load corrections.jsonl and compute metrics; "
            "full    -> end-to-end with API backend (requires model.backend=api)"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="Number of top confusion pairs to inject into the prompt (default: 10).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="flat_arabic",
        choices=["flat_arabic", "grouped_arabic"],
        dest="format_style",
        help="Confusion formatting style (default: flat_arabic).",
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
            "(e.g. KHATT-train PATS-A01-Akhbar-train). "
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
        "--no-confusion-impact",
        action="store_true",
        default=False,
        help="Skip confusion_impact.json computation.",
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
        default=Path("results/phase3"),
        help="Output directory for Phase 3 results.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("results/phase1"),
        help="Phase 1 results directory (source of confusion matrices).",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("results/phase2"),
        help="Phase 2 results directory (baseline for comparison).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase3.log"

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
    top_n: int,
    format_style: str,
    extra: Optional[dict] = None,
) -> dict:
    """Build the standard metadata block for all Phase 3 output JSON files."""
    model_cfg = config.get("model", {})
    phase3_cfg = config.get("phase3", {})
    meta = {
        "phase": "phase3",
        "dataset": dataset,
        "model": model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend": model_cfg.get("backend", "transformers"),
        "prompt_type": "ocr_aware",
        "prompt_version": "p3v1",
        "top_n": top_n,
        "format_style": format_style,
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
# Confusion matrix helpers
# ---------------------------------------------------------------------------


def build_pooled_matrices(
    phase1_dir: Path,
    loader: ConfusionMatrixLoader,
    all_dataset_names: list[str],
) -> dict[str, list[ConfusionPair]]:
    """Pre-build pooled confusion matrices for PATS-A01 and KHATT dataset types.

    Args:
        phase1_dir: Root of Phase 1 results (contains per-dataset subdirs).
        loader: ConfusionMatrixLoader instance.
        all_dataset_names: All dataset keys configured (used to find matrix paths).

    Returns:
        Dict with keys "PATS-A01" and "KHATT", each mapping to a sorted list
        of pooled ConfusionPair objects.
    """
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

    pats_pooled = loader.load_pooled(pats_paths) if pats_paths else []
    khatt_pooled = loader.load_pooled(khatt_paths) if khatt_paths else []

    logger.info(
        "Pooled matrices: PATS-A01 has %d pairs, KHATT has %d pairs.",
        len(pats_pooled), len(khatt_pooled),
    )
    return {"PATS-A01": pats_pooled, "KHATT": khatt_pooled}


def resolve_confusion_matrix(
    dataset_key: str,
    phase1_dir: Path,
    pooled: dict[str, list[ConfusionPair]],
    loader: ConfusionMatrixLoader,
    min_substitutions: int = ConfusionMatrixLoader.MIN_SUBSTITUTIONS,
) -> tuple[list[ConfusionPair], str]:
    """Return (confusion_pairs, source_label) for a dataset.

    Resolution order:
    1. Dataset-specific Phase 1 confusion matrix (if present and data-rich).
    2. Pooled matrix for the same dataset type (PATS-A01 or KHATT).
    3. Empty list with source="none" (triggers zero-shot fallback in prompt builder).

    Args:
        dataset_key: e.g. "KHATT-train" or "PATS-A01-Akhbar-train".
        phase1_dir: Root of Phase 1 results.
        pooled: Pre-built pooled matrices from build_pooled_matrices().
        loader: ConfusionMatrixLoader instance.
        min_substitutions: Sparsity threshold.

    Returns:
        Tuple of (pairs_list, source_label_string).
    """
    matrix_path = phase1_dir / dataset_key / "confusion_matrix.json"

    if loader.has_enough_data(matrix_path, min_substitutions):
        try:
            pairs = loader.load(matrix_path)
            logger.info("[%s] Using dataset-specific confusion matrix (%d pairs).",
                        dataset_key, len(pairs))
            return pairs, "dataset_specific"
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[%s] Could not load matrix: %s — falling back to pooled.", dataset_key, exc)

    # Determine dataset type for pooled fallback
    if dataset_key.startswith("PATS-A01-"):
        pool_key = "PATS-A01"
    elif dataset_key.startswith("KHATT-"):
        pool_key = "KHATT"
    else:
        pool_key = None

    if pool_key and pooled.get(pool_key):
        logger.warning(
            "[%s] Dataset matrix sparse/missing — using pooled %s matrix (%d pairs).",
            dataset_key, pool_key, len(pooled[pool_key]),
        )
        return pooled[pool_key], f"pooled_{pool_key}"

    logger.warning(
        "[%s] No confusion data available — confusion_context will be empty. "
        "Samples will use zero_shot fallback.",
        dataset_key,
    )
    return [], "none"


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------


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
        logger.info("Export resume: found existing datasets in %s: %s",
                    output_path, sorted(found))
    return found


def run_export(
    config: dict,
    active_datasets: list[str],
    all_dataset_names: list[str],
    results_dir: Path,
    phase1_dir: Path,
    top_n: int,
    format_style: str,
    limit: Optional[int],
    force: bool,
) -> None:
    """Export OCR texts with confusion context to inference_input.jsonl.

    Each line contains: sample_id, dataset, ocr_text, gt_text,
    prompt_type, confusion_context, top_n, format_style, confusion_source.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        all_dataset_names: All configured dataset keys (for building pooled matrices).
        results_dir: Phase 3 results root directory.
        phase1_dir: Phase 1 results root (source of confusion matrices).
        top_n: Number of confusion pairs to inject.
        format_style: "flat_arabic" or "grouped_arabic".
        limit: Max samples per dataset (None = no limit).
        force: If True, ignore existing exported datasets.
    """
    loader_data = DataLoader(config)
    loader_conf = ConfusionMatrixLoader()
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    phase3_cfg = config.get("phase3", {})
    min_subs = phase3_cfg.get("min_substitutions_for_dataset_matrix",
                               ConfusionMatrixLoader.MIN_SUBSTITUTIONS)

    # Pre-build pooled matrices once (used as fallback for sparse datasets)
    pooled = build_pooled_matrices(phase1_dir, loader_conf, all_dataset_names)

    already_exported = _load_exported_datasets(output_path) if not force else set()
    total_written = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported — skipping (use --force to re-export).", ds_key
                )
                continue

            # Resolve confusion pairs for this dataset
            pairs, confusion_source = resolve_confusion_matrix(
                ds_key, phase1_dir, pooled, loader_conf, min_subs,
            )

            # Format once — all samples in this dataset share the same confusion context
            confusion_context = loader_conf.format_for_prompt(pairs, n=top_n, style=format_style)

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            for sample in samples:
                record = {
                    "sample_id":         sample.sample_id,
                    "dataset":           ds_key,
                    "ocr_text":          sample.ocr_text,
                    "gt_text":           sample.gt_text,
                    "prompt_type":       "ocr_aware" if confusion_context else "zero_shot",
                    "confusion_context": confusion_context,
                    "top_n":             top_n,
                    "format_style":      format_style,
                    "confusion_source":  confusion_source,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            logger.info(
                "Exported %d samples for %s (confusion_source=%s, context_len=%d chars).",
                len(samples), ds_key, confusion_source, len(confusion_context),
            )

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
    logger.info("     See notebooks/kaggle_setup.ipynb or docs/Kaggle_Colab_Guide.md")
    logger.info("  3. Copy corrections.jsonl to results/phase3/corrections.jsonl")
    logger.info("  4. Run analysis locally:")
    logger.info("       python pipelines/run_phase3.py --mode analyze")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# corrections.jsonl loading
# ---------------------------------------------------------------------------


def load_corrections(corrections_path: Path) -> list[CorrectedSample]:
    """Load corrections.jsonl into CorrectedSample objects.

    Args:
        corrections_path: Path to corrections.jsonl downloaded from Kaggle/Colab.

    Returns:
        List of CorrectedSample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not corrections_path.exists():
        raise FileNotFoundError(
            f"corrections.jsonl not found: {corrections_path}\n"
            f"Did you download it from Kaggle/Colab?\n"
            f"Expected path: results/phase3/{{dataset_key}}/corrections.jsonl\n"
            f"Run the export + inference steps first — see docs/HOW_TO_RUN.md"
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


def _maybe_split_combined_corrections(results_dir: Path) -> None:
    """Split a combined corrections.jsonl into per-dataset files if needed."""
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
# Error change analysis (same logic as Phase 2)
# ---------------------------------------------------------------------------


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
) -> dict:
    """Compare per-type error counts before (OCR) and after (Phase 3 corrected).

    Identical logic to Phase 2's run_error_change_analysis — reused here so
    Phase 3 error_changes.json has the same schema as Phase 2's.
    """
    type_keys = [et.value for et in ErrorType]
    phase1_counts: dict[str, int] = {k: 0 for k in type_keys}
    phase3_counts: dict[str, int] = {k: 0 for k in type_keys}
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

        for ce in err1.char_errors:
            k = ce.error_type.value
            phase1_counts[k] += 1
            total_ocr_errors += 1

        for ce in err2.char_errors:
            k = ce.error_type.value
            phase3_counts[k] += 1
            total_corrected_errors += 1

        for k in type_keys:
            delta = phase1_counts[k] - phase3_counts[k]
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
        if phase1_counts[k] == 0 and phase3_counts[k] == 0:
            continue
        by_type[k] = {
            "phase1_count":  phase1_counts[k],
            "phase3_count":  phase3_counts[k],
            "fixed":         fixed_counts[k],
            "introduced":    intro_counts[k],
            "fix_rate":      round(fixed_counts[k] / max(phase1_counts[k], 1), 4),
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
# Confusion impact analysis
# ---------------------------------------------------------------------------

# Character groups for mapping confusion pairs to error types.
# Mirrors ErrorAnalyzer definitions.
_HAMZA_GROUP     = frozenset("أاإآء")
_TAA_GROUP       = frozenset("ةه")
_ALEF_MAX_GROUP  = frozenset("ىي")
_DOT_GROUPS      = [
    frozenset("بتثن"),
    frozenset("جحخ"),
    frozenset("دذ"),
    frozenset("رز"),
    frozenset("سش"),
    frozenset("صض"),
    frozenset("طظ"),
    frozenset("فق"),
]


def _classify_confusion_pair(gt_char: str, ocr_char: str) -> str:
    """Map a (gt_char, ocr_char) confusion pair to an ErrorType value string.

    Returns the ErrorType.value string most likely associated with this pair.
    Falls back to "other_substitution" if no group matches.
    """
    if gt_char in _TAA_GROUP and ocr_char in _TAA_GROUP:
        return ErrorType.TAA_MARBUTA.value
    if gt_char in _HAMZA_GROUP and ocr_char in _HAMZA_GROUP:
        return ErrorType.HAMZA.value
    if gt_char in _ALEF_MAX_GROUP and ocr_char in _ALEF_MAX_GROUP:
        return ErrorType.ALEF_MAKSURA.value
    for grp in _DOT_GROUPS:
        if gt_char in grp and ocr_char in grp:
            return ErrorType.DOT_CONFUSION.value
    return ErrorType.OTHER_SUB.value


def run_confusion_impact_analysis(
    dataset_name: str,
    phase2_error_changes_path: Path,
    phase3_error_changes: dict,
    confusion_pairs: list[ConfusionPair],
    top_n: int,
) -> dict:
    """Measure the impact of confusion injection per injected confusion pair.

    Compares Phase 2 and Phase 3 error_changes for each confusion pair that
    was injected into the prompt. Shows which pairs the LLM learned to fix
    better after being told about them.

    Args:
        dataset_name: Dataset key label.
        phase2_error_changes_path: Path to Phase 2 error_changes.json.
        phase3_error_changes: Already-computed Phase 3 error_changes dict.
        confusion_pairs: All confusion pairs from the matrix (sorted by count).
        top_n: Number of top pairs that were injected.

    Returns:
        confusion_impact dict.
    """
    injected_pairs = confusion_pairs[:top_n]

    # Load Phase 2 error_changes
    phase2_by_type: dict = {}
    if phase2_error_changes_path.exists():
        try:
            with open(phase2_error_changes_path, encoding="utf-8") as f:
                p2_data = json.load(f)
            phase2_by_type = p2_data.get("by_type", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not load Phase 2 error_changes for %s: %s — impact analysis skipped.",
                dataset_name, exc,
            )
            return _empty_impact(dataset_name, injected_pairs, top_n, str(phase2_error_changes_path))
    else:
        logger.warning(
            "Phase 2 error_changes.json not found at %s — impact analysis skipped.",
            phase2_error_changes_path,
        )
        return _empty_impact(dataset_name, injected_pairs, top_n, str(phase2_error_changes_path))

    phase3_by_type = phase3_error_changes.get("by_type", {})

    injected_list = []
    impact_by_pair: dict = {}
    improved = unchanged = worsened = 0
    marginal_improvements: list[float] = []
    best_pair = worst_pair = ""
    best_delta = float("-inf")
    worst_delta = float("inf")

    for i, pair in enumerate(injected_pairs):
        error_type = _classify_confusion_pair(pair.gt_char, pair.ocr_char)
        pair_key = f"{pair.gt_char}\u2192{pair.ocr_char}"  # e.g. "ة→ه"

        injected_list.append({
            "gt": pair.gt_char,
            "ocr": pair.ocr_char,
            "count": pair.count,
            "probability": pair.probability,
            "rank": i + 1,
            "error_type": error_type,
        })

        p2_type = phase2_by_type.get(error_type, {})
        p3_type = phase3_by_type.get(error_type, {})

        p2_fix_rate = float(p2_type.get("fix_rate", 0.0))
        p3_fix_rate = float(p3_type.get("fix_rate", 0.0))
        delta = p3_fix_rate - p2_fix_rate

        if delta > 0.01:
            direction = "improved"
            improved += 1
        elif delta < -0.01:
            direction = "worsened"
            worsened += 1
        else:
            direction = "unchanged"
            unchanged += 1

        marginal_improvements.append(delta)

        if delta > best_delta:
            best_delta = delta
            best_pair = pair_key
        if delta < worst_delta:
            worst_delta = delta
            worst_pair = pair_key

        impact_by_pair[pair_key] = {
            "error_type":           error_type,
            "phase1_error_count":   p2_type.get("phase1_count", 0),
            "phase2_remaining":     p2_type.get("phase2_count", 0),
            "phase3_remaining":     p3_type.get("phase3_count", p2_type.get("phase2_count", 0)),
            "phase2_fix_rate":      round(p2_fix_rate, 4),
            "phase3_fix_rate":      round(p3_fix_rate, 4),
            "marginal_improvement": round(delta, 4),
            "direction":            direction,
        }

    avg_improvement = (
        round(sum(marginal_improvements) / len(marginal_improvements), 4)
        if marginal_improvements else 0.0
    )

    return {
        "meta": {
            "dataset": dataset_name,
            "top_n_injected": top_n,
            "phase2_error_changes_source": str(phase2_error_changes_path),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": (
                "marginal_improvement = phase3_fix_rate - phase2_fix_rate per error_type. "
                "Positive = LLM fixed more of this error type after being told about it."
            ),
        },
        "injected_pairs": injected_list,
        "impact_by_pair": impact_by_pair,
        "summary": {
            "total_pairs_injected":    top_n,
            "pairs_improved":          improved,
            "pairs_unchanged":         unchanged,
            "pairs_worsened":          worsened,
            "avg_marginal_improvement": avg_improvement,
            "most_improved_pair":      best_pair,
            "least_improved_pair":     worst_pair,
        },
    }


def _empty_impact(
    dataset_name: str,
    injected_pairs: list[ConfusionPair],
    top_n: int,
    source: str,
) -> dict:
    """Return a stub confusion_impact dict when Phase 2 data is unavailable."""
    return {
        "meta": {
            "dataset": dataset_name,
            "top_n_injected": top_n,
            "phase2_error_changes_source": source,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "Phase 2 error_changes.json not available — impact analysis skipped.",
        },
        "injected_pairs": [
            {"gt": p.gt_char, "ocr": p.ocr_char, "count": p.count,
             "probability": p.probability, "rank": i + 1,
             "error_type": _classify_confusion_pair(p.gt_char, p.ocr_char)}
            for i, p in enumerate(injected_pairs)
        ],
        "impact_by_pair": {},
        "summary": {"note": "Skipped — Phase 2 error_changes.json not found."},
    }


# ---------------------------------------------------------------------------
# ANALYZE MODE — per-dataset processing
# ---------------------------------------------------------------------------


def load_phase2_dataset_metrics(phase2_dir: Path, dataset_key: str) -> Optional[dict]:
    """Load Phase 2 per-dataset corrected metrics for comparison.

    Args:
        phase2_dir: Phase 2 results root directory.
        dataset_key: Dataset key, e.g. "KHATT-train".

    Returns:
        The "corrected" sub-dict from metrics.json, or None if not found.
    """
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


def process_dataset_analyze(
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    phase1_dir: Path,
    limit: Optional[int],
    top_n: int,
    format_style: str,
    pooled: dict[str, list[ConfusionPair]],
    analyze_errors: bool,
    analyze_impact: bool,
) -> MetricResult:
    """Run all analysis steps for one dataset.

    Args:
        dataset_key: e.g. "KHATT-train".
        corrected_samples: Loaded from corrections.jsonl.
        config: Parsed config dict.
        results_dir: Phase 3 results root.
        phase2_dir: Phase 2 results root (baseline metrics + error_changes).
        phase1_dir: Phase 1 results root (confusion matrices for impact analysis).
        limit: Sample limit applied (for metadata only).
        top_n: Number of confusion pairs used in export (for impact analysis).
        format_style: Format style used in export.
        pooled: Pre-built pooled matrices (for re-resolving pairs in impact analysis).
        analyze_errors: If True, compute error_changes.json.
        analyze_impact: If True, compute confusion_impact.json.

    Returns:
        MetricResult for corrected text on this dataset.
    """
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(corrected_samples)
    n_failed = sum(1 for cs in corrected_samples if not cs.success)
    n_fallback = sum(
        1 for cs in corrected_samples
        if getattr(cs, "error", None) is None
        # We detect fallbacks via corrected_text == ocr_text only when success=True
    )
    total_prompt_tokens = sum(cs.prompt_tokens for cs in corrected_samples)
    total_output_tokens = sum(cs.output_tokens for cs in corrected_samples)
    total_latency = sum(cs.latency_s for cs in corrected_samples)
    avg_latency = total_latency / max(n, 1)

    logger.info("=" * 60)
    logger.info("Analyzing dataset: %s  (%d samples, %d failed)", dataset_key, n, n_failed)

    # ------------------------------------------------------------------
    # Step 1: CER/WER on Phase 3 corrected texts
    # ------------------------------------------------------------------
    logger.info("[%s] Calculating corrected CER/WER ...", dataset_key)
    corrected_result = calculate_metrics(
        corrected_samples,
        dataset_name=dataset_key,
        text_field="corrected_text",
    )

    metrics_json = {
        "meta": make_meta(
            dataset_key, n, config, limit, top_n, format_style,
            extra={
                "total_prompt_tokens":      total_prompt_tokens,
                "total_output_tokens":      total_output_tokens,
                "total_latency_s":          round(total_latency, 2),
                "avg_latency_per_sample_s": round(avg_latency, 3),
                "failed_samples":           n_failed,
            },
        ),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] Phase 3 CER=%.2f%%  WER=%.2f%%",
        dataset_key, corrected_result.cer * 100, corrected_result.wer * 100,
    )

    # ------------------------------------------------------------------
    # Step 2: Comparison vs Phase 2 baseline (ISOLATED comparison)
    # ------------------------------------------------------------------
    p2_metrics = load_phase2_dataset_metrics(phase2_dir, dataset_key)
    if p2_metrics is not None:
        p2_cer = float(p2_metrics.get("cer", 0.0))
        p2_wer = float(p2_metrics.get("wer", 0.0))
        cer_delta_abs = p2_cer - corrected_result.cer    # positive = Phase 3 better
        wer_delta_abs = p2_wer - corrected_result.wer
        cer_rel = (cer_delta_abs / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta_abs / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(dataset_key, n, config, limit, top_n, format_style,
                              extra={"comparison": "phase3_vs_phase2"}),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
            },
            "phase3_corrected": {
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
            "[%s] Phase2->Phase3 CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key,
            p2_cer * 100, corrected_result.cer * 100, cer_rel,
            p2_wer * 100, corrected_result.wer * 100, wer_rel,
        )
    else:
        logger.warning(
            "[%s] Phase 2 metrics not found at %s — skipping comparison_vs_phase2.json.",
            dataset_key, phase2_dir / dataset_key / "metrics.json",
        )

    # ------------------------------------------------------------------
    # Step 3: Error change analysis (optional)
    # ------------------------------------------------------------------
    phase3_error_changes: dict = {}
    if analyze_errors:
        logger.info("[%s] Running error change analysis ...", dataset_key)
        phase3_error_changes = run_error_change_analysis(corrected_samples, dataset_key)
        phase3_error_changes["meta"] = make_meta(
            dataset_key, n, config, limit, top_n, format_style
        )
        save_json(phase3_error_changes, out_dir / "error_changes.json")

    # ------------------------------------------------------------------
    # Step 4: Confusion impact analysis (optional — requires error_changes)
    # ------------------------------------------------------------------
    if analyze_impact and analyze_errors:
        logger.info("[%s] Running confusion impact analysis ...", dataset_key)

        # Re-resolve confusion pairs (same logic as export step)
        conf_loader = ConfusionMatrixLoader()
        phase3_cfg = config.get("phase3", {})
        min_subs = phase3_cfg.get("min_substitutions_for_dataset_matrix",
                                   ConfusionMatrixLoader.MIN_SUBSTITUTIONS)
        pairs, _ = resolve_confusion_matrix(
            dataset_key, phase1_dir, pooled, conf_loader, min_subs,
        )

        if pairs:
            p2_error_changes_path = phase2_dir / dataset_key / "error_changes.json"
            impact = run_confusion_impact_analysis(
                dataset_name=dataset_key,
                phase2_error_changes_path=p2_error_changes_path,
                phase3_error_changes=phase3_error_changes,
                confusion_pairs=pairs,
                top_n=top_n,
            )
            save_json(impact, out_dir / "confusion_impact.json")
        else:
            logger.warning(
                "[%s] No confusion pairs available — skipping confusion_impact.json.", dataset_key
            )
    elif analyze_impact and not analyze_errors:
        logger.warning(
            "[%s] Skipping confusion_impact.json because --no-error-analysis was set "
            "(impact analysis requires error_changes).", dataset_key
        )

    return corrected_result


# ---------------------------------------------------------------------------
# FULL MODE — inline inference + analyze
# ---------------------------------------------------------------------------


def process_dataset_full(
    dataset_key: str,
    samples: list[OCRSample],
    corrector: BaseLLMCorrector,
    builder: PromptBuilder,
    confusion_context: str,
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    phase1_dir: Path,
    limit: Optional[int],
    top_n: int,
    format_style: str,
    pooled: dict[str, list[ConfusionPair]],
    analyze_errors: bool,
    analyze_impact: bool,
) -> MetricResult:
    """Run inline inference + analysis for one dataset (full/API mode)."""
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
            logger.info("[%s] Resuming: %d samples already corrected.",
                        dataset_key, len(completed_ids))

    pending = [s for s in samples if s.sample_id not in completed_ids]
    max_retries = config.get("phase3", {}).get("max_retries", 2)

    with open(corrections_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(pending, desc=f"  Correcting {dataset_key}", unit="sample"):
            messages = builder.build_ocr_aware(sample.ocr_text, confusion_context)
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
                "prompt_type":    "ocr_aware",
                "prompt_version": builder.ocr_aware_prompt_version,
                "prompt_tokens":  result.prompt_tokens,
                "output_tokens":  result.output_tokens,
                "latency_s":      result.latency_s,
                "success":        result.success,
                "error":          result.error,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    corrected_samples = load_corrections(corrections_path)
    return process_dataset_analyze(
        dataset_key, corrected_samples, config, results_dir,
        phase2_dir, phase1_dir, limit, top_n, format_style, pooled,
        analyze_errors, analyze_impact,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(
    all_corrected: dict[str, MetricResult],
    config: dict,
    results_dir: Path,
    limit: Optional[int],
    top_n: int,
    format_style: str,
) -> None:
    """Write combined metrics.json across all datasets."""
    output = {
        "meta": {
            "phase": "phase3",
            "model": config.get("model", {}).get("name", ""),
            "prompt_type": "ocr_aware",
            "prompt_version": "p3v1",
            "top_n": top_n,
            "format_style": format_style,
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
            "phase": "phase3",
            "comparison": "phase3_vs_phase2",
            "model": config.get("model", {}).get("name", ""),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
        },
        "datasets": all_comparisons,
        "note": (
            "Phase 3 is an isolated experiment. Δ measures the contribution of "
            "confusion matrix injection over zero-shot (Phase 2). Negative delta = improvement."
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
    top_n: int,
    format_style: str,
    results_dir: Path,
) -> None:
    """Write human-readable Markdown report to results_dir/report.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Phase 3 Report: OCR-Aware Prompting")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"Model: {model_name}")
    lines.append(f"Prompt: OCR-Aware (p3v1) — Top-{top_n} confusions, {format_style} format")
    lines.append("")

    # Comparison table vs Phase 2
    lines.append("## Results vs Phase 2 (Zero-Shot Baseline)\n")
    lines.append("> Isolated comparison. Phase 3 vs Phase 2 only.\n")
    if all_comparisons:
        lines.append(
            "| Dataset | Phase 2 CER | Phase 3 CER | Delta CER | "
            "Phase 2 WER | Phase 3 WER | Delta WER |"
        )
        lines.append(
            "|---------|-------------|-------------|-----------|"
            "-------------|-------------|-----------|"
        )
        for ds, cmp in all_comparisons.items():
            p2 = cmp.get("phase2_baseline", {})
            p3 = cmp.get("phase3_corrected", {})
            d  = cmp.get("delta", {})
            lines.append(
                f"| {ds} "
                f"| {p2.get('cer', 0)*100:.2f}% "
                f"| {p3.get('cer', 0)*100:.2f}% "
                f"| {d.get('cer_relative_pct', 0):+.1f}% "
                f"| {p2.get('wer', 0)*100:.2f}% "
                f"| {p3.get('wer', 0)*100:.2f}% "
                f"| {d.get('wer_relative_pct', 0):+.1f}% |"
            )
    else:
        lines.append("*Phase 2 baseline not available — run Phase 2 first.*")
    lines.append("")

    # Phase 3 absolute metrics
    lines.append("## Phase 3 Post-Correction Metrics\n")
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

    # Confusion impact summary (load from saved files)
    lines.append("## Confusion Impact per Injected Pair\n")
    any_impact = False
    for ds in all_corrected:
        impact_path = results_dir / ds / "confusion_impact.json"
        if impact_path.exists():
            try:
                with open(impact_path, encoding="utf-8") as f:
                    impact = json.load(f)
                s = impact.get("summary", {})
                if "note" in s:
                    continue
                any_impact = True
                lines.append(f"### {ds}\n")
                lines.append(
                    f"- Pairs injected: {s.get('total_pairs_injected', 0)}\n"
                    f"- Improved: {s.get('pairs_improved', 0)}  "
                    f"Unchanged: {s.get('pairs_unchanged', 0)}  "
                    f"Worsened: {s.get('pairs_worsened', 0)}\n"
                    f"- Avg marginal improvement: "
                    f"{s.get('avg_marginal_improvement', 0)*100:+.2f}%\n"
                    f"- Most improved pair: {s.get('most_improved_pair', 'N/A')}\n"
                    f"- Least improved pair: {s.get('least_improved_pair', 'N/A')}"
                )
                lines.append("")
            except (json.JSONDecodeError, OSError):
                pass
    if not any_impact:
        lines.append("*No confusion impact data available.*")
    lines.append("")

    # Error change analysis summary
    lines.append("## Error Change Analysis\n")
    for ds in all_corrected:
        ec_path = results_dir / ds / "error_changes.json"
        if ec_path.exists():
            try:
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
                    lines.append("| Error Type | Phase 1 | Phase 3 | Fixed | Introduced | Fix Rate |")
                    lines.append("|-----------|---------|---------|-------|------------|----------|")
                    for etype, data in sorted(
                        by_type.items(), key=lambda x: -x[1].get("phase1_count", 0)
                    ):
                        lines.append(
                            f"| {etype} "
                            f"| {data.get('phase1_count', 0):,} "
                            f"| {data.get('phase3_count', 0):,} "
                            f"| {data.get('fixed', 0):,} "
                            f"| {data.get('introduced', 0):,} "
                            f"| {data.get('fix_rate', 0)*100:.1f}% |"
                        )
                lines.append("")
            except (json.JSONDecodeError, OSError):
                pass

    # Key findings
    lines.append("## Key Findings\n")
    if all_corrected:
        best = min(all_corrected.items(), key=lambda x: x[1].cer)
        worst = max(all_corrected.items(), key=lambda x: x[1].cer)
        lines.append(f"- Best Phase 3 CER: **{best[0]}** at {best[1].cer*100:.2f}%")
        lines.append(f"- Worst Phase 3 CER: **{worst[0]}** at {worst[1].cer*100:.2f}%")
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
        "> Phase 3 is an isolated experiment comparing OCR-aware prompting vs zero-shot.\n"
        "> Phase 6 will combine Phase 3 knowledge with other sources for the full system."
    )

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


def print_summary(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    top_n: int,
) -> None:
    """Print a final summary table to stdout."""
    print("\n" + "=" * 80)
    print(f"PHASE 3 SUMMARY — OCR-Aware Prompting (Top-{top_n} confusions)")
    print("=" * 80)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'P3 CER':>8} {'D CER':>8} {'P3 WER':>8} {'N':>6}")
    print("-" * 80)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_cer = cmp.get("phase2_baseline", {}).get("cer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        p2_str = f"{p2_cer*100:.2f}%" if cmp else "N/A"
        delta_str = f"{cer_rel:+.1f}%" if cmp else "N/A"
        print(
            f"{ds:<28} {p2_str:>8} {r.cer*100:>7.2f}% {delta_str:>8} "
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

    logger.info("Phase 3: OCR-Aware Prompting  (mode=%s)", args.mode)
    logger.info("Config: %s", args.config)
    logger.info("Results dir: %s", results_dir)
    logger.info("Phase 1 dir: %s", args.phase1_dir)
    logger.info("Phase 2 dir: %s", args.phase2_dir)
    logger.info("Top-N confusions: %d  Format: %s", args.top_n, args.format_style)

    config = load_config(args.config)
    phase3_cfg = config.get("phase3", {})

    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")
    top_n = args.top_n or phase3_cfg.get("top_n", 10)
    format_style = args.format_style or phase3_cfg.get("format_style", "flat_arabic")
    analyze_errors = not args.no_error_analysis
    analyze_impact = not args.no_confusion_impact

    active_datasets = resolve_datasets(config, args.datasets)
    all_dataset_names = [entry["name"] for entry in config.get("datasets", [])]
    logger.info("Datasets to process: %s", active_datasets)

    # ------------------------------------------------------------------
    # EXPORT mode
    # ------------------------------------------------------------------
    if args.mode == "export":
        run_export(
            config=config,
            active_datasets=active_datasets,
            all_dataset_names=all_dataset_names,
            results_dir=results_dir,
            phase1_dir=args.phase1_dir,
            top_n=top_n,
            format_style=format_style,
            limit=limit,
            force=args.force,
        )
        return

    # ------------------------------------------------------------------
    # FULL mode — load model
    # ------------------------------------------------------------------
    corrector: Optional[BaseLLMCorrector] = None
    builder: Optional[PromptBuilder] = None
    if args.mode == "full":
        backend = config.get("model", {}).get("backend", "transformers")
        if backend == "transformers":
            logger.error(
                "--mode full requires an API backend (model.backend='api'). "
                "For GPU inference: export -> scripts/infer.py (Kaggle/Colab) -> analyze."
            )
            sys.exit(1)
        corrector = get_corrector(config)
        builder = PromptBuilder()
        logger.info("Corrector ready: %s (backend=%s)", corrector.model_name, backend)

    # ------------------------------------------------------------------
    # ANALYZE / FULL — pre-build pooled matrices for impact analysis
    # ------------------------------------------------------------------
    conf_loader = ConfusionMatrixLoader()
    pooled = build_pooled_matrices(args.phase1_dir, conf_loader, all_dataset_names)

    # ------------------------------------------------------------------
    # ANALYZE mode — auto-split combined corrections.jsonl if present
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
                # Resume: skip if already analyzed
                metrics_path = results_dir / ds_key / "metrics.json"
                if metrics_path.exists() and not args.force:
                    logger.info(
                        "[%s] Already analyzed — skipping (use --force to re-analyze).", ds_key
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

                metric_result = process_dataset_analyze(
                    dataset_key=ds_key,
                    corrected_samples=corrected_samples,
                    config=config,
                    results_dir=results_dir,
                    phase2_dir=args.phase2_dir,
                    phase1_dir=args.phase1_dir,
                    limit=limit,
                    top_n=top_n,
                    format_style=format_style,
                    pooled=pooled,
                    analyze_errors=analyze_errors,
                    analyze_impact=analyze_impact,
                )

            else:  # full
                # Resolve confusion context for this dataset
                min_subs = phase3_cfg.get(
                    "min_substitutions_for_dataset_matrix",
                    ConfusionMatrixLoader.MIN_SUBSTITUTIONS,
                )
                pairs, _ = resolve_confusion_matrix(
                    ds_key, args.phase1_dir, pooled, conf_loader, min_subs,
                )
                confusion_context = conf_loader.format_for_prompt(
                    pairs, n=top_n, style=format_style,
                )

                loader_data = DataLoader(config)
                samples = list(loader_data.iter_samples(ds_key, limit=limit))
                if not samples:
                    logger.warning("No samples loaded for %s — skipping.", ds_key)
                    continue

                metric_result = process_dataset_full(
                    dataset_key=ds_key,
                    samples=samples,
                    corrector=corrector,
                    builder=builder,
                    confusion_context=confusion_context,
                    config=config,
                    results_dir=results_dir,
                    phase2_dir=args.phase2_dir,
                    phase1_dir=args.phase1_dir,
                    limit=limit,
                    top_n=top_n,
                    format_style=format_style,
                    pooled=pooled,
                    analyze_errors=analyze_errors,
                    analyze_impact=analyze_impact,
                )

            all_corrected[ds_key] = metric_result

            # Load saved comparison for the summary
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

    if not all_corrected:
        logger.error("No datasets were successfully processed. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate + report
    # ------------------------------------------------------------------
    aggregate_results(all_corrected, config, results_dir, limit, top_n, format_style)
    aggregate_comparisons(all_comparisons, config, results_dir, limit)

    model_name = config.get("model", {}).get("name", "unknown")
    generate_report(all_corrected, all_comparisons, model_name, top_n, format_style, results_dir)
    print_summary(all_corrected, all_comparisons, top_n)

    logger.info("Phase 3 complete. Results in: %s", results_dir)


if __name__ == "__main__":
    main()
