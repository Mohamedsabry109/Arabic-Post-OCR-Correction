#!/usr/bin/env python3
"""Phase 8: Retrieval-Augmented Generation (RAG) for OCR Correction.

Isolated experiment: compares against Phase 2 (zero-shot) baseline only.
For each validation sample, retrieves the most similar training corrections
and injects them as per-sample adaptive context into the LLM prompt.

Three-stage pipeline (no local GPU required for BM25 mode):

  --mode build-index -> Build RAG index from Phase 2 training corrections
                        -> results/phase8/index/

  --mode export      -> Retrieve + build inference_input.jsonl (val splits only)
                        -> results/phase8/inference_input.jsonl

  --mode analyze     -> Load corrections.jsonl, compute metrics and reports
                        -> results/phase8/{dataset}/metrics.json
                           results/phase8/metrics.json
                           results/phase8/comparison.json
                           results/phase8/report.md

Typical workflow
----------------
1. Prerequisite: Phase 2 training-split corrections must exist at
   results/phase2-training/corrections.jsonl (or per-dataset files).

2. LOCAL:  python pipelines/run_phase8.py --mode build-index
   (builds BM25 / dense indices from training corrections)

3. LOCAL:  python pipelines/run_phase8.py --mode export
   (retrieves per-sample context, builds inference JSONL)

4. REMOTE: git clone <repo> && python scripts/infer.py \\
               --input  results/phase8/inference_input.jsonl \\
               --output results/phase8/corrections.jsonl \\
               --model  Qwen/Qwen3-4B-Instruct-2507
   (see notebooks/kaggle_setup.ipynb or notebooks/colab_setup.ipynb)

5. LOCAL:  python pipelines/run_phase8.py --mode analyze

Usage
-----
    python pipelines/run_phase8.py --mode build-index
    python pipelines/run_phase8.py --mode build-index --retrieval-mode bm25
    python pipelines/run_phase8.py --mode build-index --force
    python pipelines/run_phase8.py --mode export
    python pipelines/run_phase8.py --mode export --limit 50
    python pipelines/run_phase8.py --mode export --datasets KHATT-validation
    python pipelines/run_phase8.py --mode export --retrieval-mode bm25
    python pipelines/run_phase8.py --mode export --force
    python pipelines/run_phase8.py --mode analyze
    python pipelines/run_phase8.py --mode analyze --datasets PATS-A01-Akhbar-val
    python pipelines/run_phase8.py --mode analyze --no-error-analysis
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
from src.data.rag_index import RAGIndexBuilder, RAGRetriever
from src.analysis.metrics import MetricResult, calculate_metrics_dual
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from pipelines._utils import (
    resolve_datasets, load_sample_list, compute_group_aggregates,
    split_runaway_samples, DEFAULT_RUNAWAY_RATIO_THRESHOLD,
    load_phase2_full_metrics, pick_phase2_variant, _pick_corrected_key,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8: RAG for Arabic Post-OCR Correction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["build-index", "export", "analyze"],
        help=(
            "build-index -> build RAG index from Phase 2 training corrections; "
            "export      -> retrieve + build inference_input.jsonl; "
            "analyze     -> load corrections.jsonl and compute metrics"
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default=None,
        dest="retrieval_mode",
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval mode (overrides config). Default: from config or bm25.",
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
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/phase8"),
        dest="results_dir",
        help="Phase 8 results root directory.",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("results/phase2"),
        dest="phase2_dir",
        help="Phase 2 results directory (baseline for comparison).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase8.log"

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
    phase8_cfg = config.get("phase8", {})
    meta = {
        "phase":           "phase8",
        "dataset":         dataset,
        "model":           model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend":         model_cfg.get("backend", "transformers"),
        "prompt_type":     "rag",
        "prompt_version":  "p8v1",
        "retrieval_mode":  phase8_cfg.get("retrieval_mode", "bm25"),
        "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit":      get_git_commit(),
        "num_samples":     num_samples,
        "limit_applied":   limit,
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
        if out_path.exists() and not force:
            logger.info("  [%s] Already split -- skipping (use --force to re-split).", ds_key)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("  Split: %d records -> %s", len(records), out_path)


# ---------------------------------------------------------------------------
# BUILD-INDEX MODE
# ---------------------------------------------------------------------------


def _resolve_training_corrections(config: dict) -> list[Path]:
    """Find all training-split corrections.jsonl files."""
    phase8_cfg = config.get("phase8", {})
    source = phase8_cfg.get("source_corrections", "phase2-training")
    results_root = Path(config.get("output", {}).get("results_dir", "results"))

    paths: list[Path] = []

    # Try combined file first
    combined = results_root / source / "corrections.jsonl"
    if combined.exists():
        paths.append(combined)
        logger.info("Found combined training corrections: %s", combined)
        return paths

    # Try per-dataset files
    source_dir = results_root / source
    if source_dir.exists():
        for p in sorted(source_dir.rglob("corrections.jsonl")):
            paths.append(p)

    if not paths:
        # Fallback: look for training-split datasets in phase2 results
        phase2_dir = results_root / "phase2"
        if phase2_dir.exists():
            for entry in config.get("datasets", []):
                name = entry["name"]
                if name.endswith("-train"):
                    p = phase2_dir / name / "corrections.jsonl"
                    if p.exists():
                        paths.append(p)

    if paths:
        logger.info("Found %d training corrections file(s).", len(paths))
    else:
        logger.warning(
            "No training corrections found. "
            "Run Phase 2 on training splits first."
        )

    return paths


def run_build_index(
    config: dict,
    results_dir: Path,
    retrieval_mode: str,
    force: bool,
) -> None:
    """Build RAG index from Phase 2 training corrections."""
    phase8_cfg = config.get("phase8", {})
    index_dir = results_dir / "index"

    # Check if index already exists
    meta_path = index_dir / "index_meta.json"
    if meta_path.exists() and not force:
        logger.info(
            "RAG index already exists at %s -- skipping (use --force to rebuild).",
            index_dir,
        )
        return

    # Find training corrections
    corrections_paths = _resolve_training_corrections(config)
    if not corrections_paths:
        logger.error(
            "No training corrections found. Cannot build RAG index.\n"
            "Run Phase 2 on training splits first:\n"
            "  python pipelines/run_phase2.py --mode export --datasets *-train\n"
            "  python scripts/infer.py --input ... --output ...\n"
            "  python pipelines/run_phase2.py --mode analyze"
        )
        sys.exit(1)

    # Build index
    build_dense = retrieval_mode in ("dense", "hybrid")
    builder = RAGIndexBuilder(config)
    builder.build_from_corrections(
        corrections_paths,
        min_cer=phase8_cfg.get("min_cer", 0.001),
        max_text_len=phase8_cfg.get("max_text_len", 300),
        runaway_ratio=config.get("evaluation", {}).get(
            "runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD
        ),
        build_dense=build_dense,
    )

    # Save
    builder.save(index_dir)

    logger.info("=" * 60)
    logger.info("RAG index built successfully:")
    logger.info("  Sentence entries: %d", len(builder.sentence_store))
    logger.info("  Word-error pairs: %d", len(builder.word_store))
    logger.info("  Dense index:      %s", builder.has_dense)
    logger.info("  Saved to:         %s", index_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------


def run_export(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    retrieval_mode: str,
    limit: Optional[int],
    force: bool,
    sample_ids: Optional[set[str]] = None,
) -> None:
    """Export OCR texts with retrieved RAG context to inference_input.jsonl."""
    phase8_cfg = config.get("phase8", {})
    index_dir = results_dir / "index"
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load RAG index
    meta_path = index_dir / "index_meta.json"
    if not meta_path.exists():
        logger.error(
            "RAG index not found at %s.\n"
            "Run build-index first:\n"
            "  python pipelines/run_phase8.py --mode build-index",
            index_dir,
        )
        sys.exit(1)

    index_builder = RAGIndexBuilder(config)
    index_builder.load(index_dir)

    # Create retriever
    alpha = phase8_cfg.get("alpha", 0.6)
    top_k_candidates = phase8_cfg.get("top_k_candidates", 50)
    retriever = RAGRetriever(
        index=index_builder,
        mode=retrieval_mode,
        alpha=alpha,
        top_k_candidates=top_k_candidates,
    )

    top_k_sentences = phase8_cfg.get("top_k_sentences", 5)
    top_k_words = phase8_cfg.get("top_k_words", 15)
    top_k_per_word = phase8_cfg.get("word_top_k_per_input_word", 3)

    # Resume logic
    already_exported = _load_exported_datasets(output_path) if not force else set()

    # Load data
    loader_data = DataLoader(config)
    total_written = 0

    # Retrieval diagnostics
    diag_scores: list[float] = []
    diag_per_dataset: dict[str, list[float]] = {}

    with open(output_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already_exported:
                logger.info(
                    "[%s] Already exported -- skipping (use --force to re-export).",
                    ds_key,
                )
                continue

            try:
                samples = list(loader_data.iter_samples(
                    ds_key, limit=limit, sample_ids=sample_ids,
                ))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            ds_scores: list[float] = []

            for sample in tqdm(samples, desc=f"  Retrieving {ds_key}", unit="sample"):
                # Retrieve similar corrections for this specific sample
                sent_results = retriever.retrieve_sentences(
                    sample.ocr_text, top_k=top_k_sentences,
                )
                word_results = retriever.retrieve_words(
                    sample.ocr_text,
                    top_k=top_k_words,
                    top_k_per_word=top_k_per_word,
                )

                # Format for prompt
                retrieved_sentences = retriever.format_sentences_for_prompt(sent_results)
                retrieved_words = retriever.format_words_for_prompt(word_results)

                # Track diagnostics
                if sent_results:
                    top1_score = sent_results[0].score
                    ds_scores.append(top1_score)
                    diag_scores.append(top1_score)

                # Determine prompt_type
                has_context = bool(retrieved_sentences.strip()) or bool(retrieved_words.strip())
                prompt_type = "rag" if has_context else "zero_shot"

                record = {
                    "sample_id":            sample.sample_id,
                    "dataset":              ds_key,
                    "ocr_text":             sample.ocr_text,
                    "gt_text":              sample.gt_text,
                    "prompt_type":          prompt_type,
                    "prompt_version":       "p8v1",
                    "retrieved_sentences":  retrieved_sentences,
                    "retrieved_words":      retrieved_words,
                    "retrieval_mode":       retrieval_mode,
                    "retrieval_alpha":      alpha,
                    "top_k_sentences":      top_k_sentences,
                    "top_k_words":          top_k_words,
                    "index_size":           len(index_builder.sentence_store),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

            diag_per_dataset[ds_key] = ds_scores
            avg_score = sum(ds_scores) / len(ds_scores) if ds_scores else 0.0
            logger.info(
                "Exported %d samples for %s (avg top-1 score=%.3f, "
                "sentences_ctx=%d chars, words_ctx=%d chars).",
                len(samples), ds_key, avg_score,
                len(retrieved_sentences), len(retrieved_words),
            )

    # Save retrieval diagnostics
    if diag_scores:
        diagnostics = {
            "index_size": len(index_builder.sentence_store),
            "retrieval_mode": retrieval_mode,
            "alpha": alpha,
            "top_k_sentences": top_k_sentences,
            "top_k_words": top_k_words,
            "avg_top1_score": round(sum(diag_scores) / len(diag_scores), 4),
            "min_top1_score": round(min(diag_scores), 4),
            "max_top1_score": round(max(diag_scores), 4),
            "per_dataset": {
                ds: {
                    "num_queries": len(scores),
                    "avg_top1_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                }
                for ds, scores in diag_per_dataset.items()
            },
        }
        save_json(diagnostics, results_dir / "retrieval_diagnostics.json")

    logger.info("=" * 60)
    logger.info("Export complete: %d new samples -> %s", total_written, output_path)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       python scripts/infer.py \\")
    logger.info("           --input  results/phase8/inference_input.jsonl \\")
    logger.info("           --output results/phase8/corrections.jsonl")
    logger.info("  3. Run analysis locally:")
    logger.info("       python pipelines/run_phase8.py --mode analyze")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# ANALYZE MODE
# ---------------------------------------------------------------------------


def load_corrections(corrections_path: Path) -> list[CorrectedSample]:
    """Load corrections.jsonl into CorrectedSample objects."""
    if not corrections_path.exists():
        raise FileNotFoundError(
            f"corrections.jsonl not found: {corrections_path}\n"
            "Did you download it from Kaggle/Colab?"
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
                logger.warning("Skipping malformed line %d: %s", lineno, exc)
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


def run_error_change_analysis(
    corrected_samples: list[CorrectedSample],
    dataset_name: str,
) -> dict:
    """Compare per-type error counts before (OCR) and after (Phase 8 corrected)."""
    type_keys = [et.value for et in ErrorType]
    phase1_counts: dict[str, int] = {k: 0 for k in type_keys}
    phase8_counts: dict[str, int] = {k: 0 for k in type_keys}
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
            s1.dataset = cs.sample.dataset
            s1.gt_text = gt
            s1.ocr_text = ocr

            s2 = _Stub()
            s2.sample_id = cs.sample.sample_id
            s2.dataset = cs.sample.dataset
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
            phase8_counts[k] += 1
            total_corrected_errors += 1

        for k in type_keys:
            delta = phase1_counts[k] - phase8_counts[k]
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
        if phase1_counts[k] == 0 and phase8_counts[k] == 0:
            continue
        by_type[k] = {
            "phase1_count":  phase1_counts[k],
            "phase8_count":  phase8_counts[k],
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


def _nd_comparison_block(p2_nd: dict, corrected_nd: MetricResult) -> dict:
    """Build no-diacritics comparison sub-dict."""
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
        "phase8_corrected_no_diacritics": {"cer": round(corrected_nd.cer, 6), "wer": round(corrected_nd.wer, 6)},
        "delta_no_diacritics": {
            "cer_absolute": round(cer_d, 6), "wer_absolute": round(wer_d, 6),
            "cer_relative_pct": round(cer_r, 2), "wer_relative_pct": round(wer_r, 2),
        },
    }


def process_dataset_analyze(
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    limit: Optional[int],
    analyze_errors: bool,
) -> MetricResult:
    """Run all analysis steps for one dataset."""
    out_dir = results_dir / dataset_key
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
    logger.info("Analyzing dataset: %s  (%d samples, %d failed)", dataset_key, n, n_failed)

    # Split normal / runaway
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

    metrics_json = {
        "meta": make_meta(
            dataset_key, n, config, limit,
            extra={
                "total_prompt_tokens":      total_prompt_tokens,
                "total_output_tokens":      total_output_tokens,
                "total_latency_s":          round(total_latency, 2),
                "avg_latency_per_sample_s": round(avg_latency, 3),
                "failed_samples":           n_failed,
                "primary_variant":          primary_source,
            },
        ),
        "ocr_all": ocr_all.to_dict(),
        "ocr_all_no_diacritics": ocr_all_nd.to_dict(),
        "ocr_normal_only": ocr_normal.to_dict(),
        "ocr_normal_only_no_diacritics": ocr_normal_nd.to_dict(),
        "corrected_all": all_result.to_dict(),
        "corrected_all_no_diacritics": all_result_nd.to_dict(),
        "corrected_normal_only": normal_result.to_dict(),
        "corrected_normal_only_no_diacritics": normal_result_nd.to_dict(),
        "data_quality": data_quality,
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] Phase 8 Primary (%s): OCR CER=%.2f%% -> LLM CER=%.2f%%  |  "
        "no-diac: %.2f%% -> %.2f%%",
        dataset_key, primary_source,
        (ocr_all if not exclude_runaway else ocr_normal).cer * 100,
        primary.cer * 100,
        (ocr_all_nd if not exclude_runaway else ocr_normal_nd).cer * 100,
        primary_nd.cer * 100,
    )

    # Comparison vs Phase 2
    p2_full = load_phase2_full_metrics(phase2_dir, dataset_key)
    if p2_full is not None:
        p2_corr, p2_nd, p2_src = pick_phase2_variant(p2_full, exclude_runaway)
        p2_cer = float(p2_corr.get("cer", 0.0))
        p2_wer = float(p2_corr.get("wer", 0.0))
        cer_delta_abs = p2_cer - primary.cer
        wer_delta_abs = p2_wer - primary.wer
        cer_rel = (cer_delta_abs / p2_cer * 100) if p2_cer > 0 else 0.0
        wer_rel = (wer_delta_abs / p2_wer * 100) if p2_wer > 0 else 0.0

        comparison = {
            "meta": make_meta(dataset_key, n, config, limit,
                              extra={"comparison": "phase8_vs_phase2"}),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
                "variant": p2_src,
            },
            "phase8_corrected": {
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
            **(_nd_comparison_block(p2_nd, primary_nd)),
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
            "[%s] Phase2->Phase8 CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key,
            p2_cer * 100, primary.cer * 100, cer_rel,
            p2_wer * 100, primary.wer * 100, wer_rel,
        )
        if p2_nd:
            p2_cer_nd = float(p2_nd.get("cer", 0.0))
            p2_wer_nd = float(p2_nd.get("wer", 0.0))
            cer_d_nd = p2_cer_nd - primary_nd.cer
            wer_d_nd = p2_wer_nd - primary_nd.wer
            cer_r_nd = (cer_d_nd / p2_cer_nd * 100) if p2_cer_nd > 0 else 0.0
            wer_r_nd = (wer_d_nd / p2_wer_nd * 100) if p2_wer_nd > 0 else 0.0
            logger.info(
                "[%s] ND  CER: %.2f%% -> %.2f%% (%+.1f%%)  |  "
                "ND  WER: %.2f%% -> %.2f%% (%+.1f%%)",
                dataset_key,
                p2_cer_nd * 100, primary_nd.cer * 100, cer_r_nd,
                p2_wer_nd * 100, primary_nd.wer * 100, wer_r_nd,
            )
    else:
        logger.warning(
            "[%s] Phase 2 metrics not found at %s -- skipping comparison.",
            dataset_key, phase2_dir / dataset_key / "metrics.json",
        )

    # Error change analysis (optional)
    if analyze_errors:
        logger.info("[%s] Running error change analysis ...", dataset_key)
        error_changes = run_error_change_analysis(corrected_samples, dataset_key)
        error_changes["meta"] = make_meta(dataset_key, n, config, limit)
        save_json(error_changes, out_dir / "error_changes.json")

    return primary


# ---------------------------------------------------------------------------
# Aggregation
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
    limit: Optional[int],
) -> None:
    """Write combined metrics.json across all datasets."""
    phase8_cfg = config.get("phase8", {})
    output = {
        "meta": {
            "phase": "phase8",
            "model": config.get("model", {}).get("name", ""),
            "prompt_type": "rag",
            "prompt_version": "p8v1",
            "retrieval_mode": phase8_cfg.get("retrieval_mode", "bm25"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
            "limit_applied": limit,
        },
        "results": {k: v.to_dict() for k, v in all_corrected.items()},
        "results_no_diacritics": _load_nd_results(all_corrected, results_dir),
    }
    output["group_aggregates"] = compute_group_aggregates(output["results"])
    nd = output.get("results_no_diacritics", {})
    if nd:
        output["group_aggregates_no_diacritics"] = compute_group_aggregates(nd)
    save_json(output, results_dir / "metrics.json")


def aggregate_comparisons(
    all_comparisons: dict[str, dict],
    config: dict,
    results_dir: Path,
) -> None:
    """Write combined comparison.json across all datasets."""
    if not all_comparisons:
        return
    output = {
        "meta": {
            "phase": "phase8",
            "comparison": "phase8_vs_phase2",
            "model": config.get("model", {}).get("name", ""),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
        },
        "datasets": all_comparisons,
        "note": (
            "Phase 8 is an isolated experiment. Delta measures the contribution of "
            "RAG retrieval-augmented context over zero-shot (Phase 2). "
            "Positive delta = improvement (lower error rate)."
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
    """Write human-readable Markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Phase 8 Report: RAG-Augmented OCR Correction")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"Model: {model_name}")
    lines.append("Prompt: RAG (p8v1) -- per-sample adaptive retrieval")
    lines.append("")

    nd_results = _load_nd_results(all_corrected, results_dir) if results_dir else {}

    # Comparison table vs Phase 2
    lines.append("## Results vs Phase 2 (Zero-Shot Baseline)\n")
    lines.append("> Isolated comparison. Phase 8 vs Phase 2 only.\n")
    if all_comparisons:
        lines.append("### With Diacritics\n")
        lines.append(
            "| Dataset | Phase 2 CER | Phase 8 CER | Delta CER | "
            "Phase 2 WER | Phase 8 WER | Delta WER |"
        )
        lines.append(
            "|---------|-------------|-------------|-----------|"
            "-------------|-------------|-----------|"
        )
        for ds, cmp in all_comparisons.items():
            p2 = cmp.get("phase2_baseline", {})
            p8 = cmp.get("phase8_corrected", {})
            d = cmp.get("delta", {})
            lines.append(
                f"| {ds} "
                f"| {p2.get('cer', 0)*100:.2f}% "
                f"| {p8.get('cer', 0)*100:.2f}% "
                f"| {d.get('cer_relative_pct', 0):+.1f}% "
                f"| {p2.get('wer', 0)*100:.2f}% "
                f"| {p8.get('wer', 0)*100:.2f}% "
                f"| {d.get('wer_relative_pct', 0):+.1f}% |"
            )
        lines.append("")

        lines.append("### No Diacritics\n")
        lines.append(
            "| Dataset | Phase 2 CER | Phase 8 CER | Delta CER | "
            "Phase 2 WER | Phase 8 WER | Delta WER |"
        )
        lines.append(
            "|---------|-------------|-------------|-----------|"
            "-------------|-------------|-----------|"
        )
        for ds, cmp in all_comparisons.items():
            p2_nd = cmp.get("phase2_baseline_no_diacritics", {})
            p8_nd = cmp.get("phase8_corrected_no_diacritics", {})
            d_nd = cmp.get("delta_no_diacritics", {})
            lines.append(
                f"| {ds} "
                f"| {p2_nd.get('cer', 0)*100:.2f}% "
                f"| {p8_nd.get('cer', 0)*100:.2f}% "
                f"| {d_nd.get('cer_relative_pct', 0):+.1f}% "
                f"| {p2_nd.get('wer', 0)*100:.2f}% "
                f"| {p8_nd.get('wer', 0)*100:.2f}% "
                f"| {d_nd.get('wer_relative_pct', 0):+.1f}% |"
            )
    else:
        lines.append("*Phase 2 baseline not available -- run Phase 2 first.*")
    lines.append("")

    # Absolute metrics
    lines.append("## Phase 8 Post-Correction Metrics\n")
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

    # Key findings
    lines.append("## Key Findings\n")
    if all_corrected:
        best = min(all_corrected.items(), key=lambda x: x[1].cer)
        worst = max(all_corrected.items(), key=lambda x: x[1].cer)
        lines.append(f"- Best Phase 8 CER: **{best[0]}** at {best[1].cer*100:.2f}%")
        lines.append(f"- Worst Phase 8 CER: **{worst[0]}** at {worst[1].cer*100:.2f}%")
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
        "> Phase 8 is an isolated experiment comparing RAG-augmented prompting "
        "vs zero-shot.\n"
        "> Each sample receives per-input adaptive context retrieved from "
        "similar training corrections."
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
    results_dir: Optional[Path] = None,
) -> None:
    """Print a final summary table to stdout."""
    nd_results = _load_nd_results(all_corrected, results_dir) if results_dir else {}

    sep = "=" * 90
    print("\n" + sep)
    print("PHASE 8 SUMMARY -- RAG-Augmented OCR Correction")

    print(sep)
    print("  [WITH DIACRITICS]")
    print(f"  {'Dataset':<30} {'P2 CER':>8} {'P8 CER':>8} {'D(CER)':>8} {'P2 WER':>8} {'P8 WER':>8} {'D(WER)':>8} {'N':>6}")
    print("  " + "-" * 82)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_cer = cmp.get("phase2_baseline", {}).get("cer", 0.0)
        p2_wer = cmp.get("phase2_baseline", {}).get("wer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        wer_rel = cmp.get("delta", {}).get("wer_relative_pct", 0.0)
        p2_cer_str = f"{p2_cer*100:.2f}%" if cmp else "N/A"
        p2_wer_str = f"{p2_wer*100:.2f}%" if cmp else "N/A"
        d_cer_str = f"{cer_rel:+.1f}%" if cmp else "N/A"
        d_wer_str = f"{wer_rel:+.1f}%" if cmp else "N/A"
        print(
            f"  {ds:<30} {p2_cer_str:>8} {r.cer*100:>7.2f}% {d_cer_str:>8} "
            f"{p2_wer_str:>8} {r.wer*100:>7.2f}% {d_wer_str:>8} {r.num_samples:>6}"
        )

    print()
    print("  [NO DIACRITICS]")
    print(f"  {'Dataset':<30} {'P2 CER':>8} {'P8 CER':>8} {'D(CER)':>8} {'P2 WER':>8} {'P8 WER':>8} {'D(WER)':>8} {'N':>6}")
    print("  " + "-" * 82)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_nd = cmp.get("phase2_baseline_no_diacritics", {})
        delta_nd = cmp.get("delta_no_diacritics", {})
        nd_curr = nd_results.get(ds, {})
        p2_cer_nd = p2_nd.get("cer", None)
        p2_wer_nd = p2_nd.get("wer", None)
        p8_cer_nd = nd_curr.get("cer", None)
        p8_wer_nd = nd_curr.get("wer", None)
        d_cer_nd = delta_nd.get("cer_relative_pct", None)
        d_wer_nd = delta_nd.get("wer_relative_pct", None)
        p2_cer_str = f"{p2_cer_nd*100:.2f}%" if p2_cer_nd is not None else "N/A"
        p2_wer_str = f"{p2_wer_nd*100:.2f}%" if p2_wer_nd is not None else "N/A"
        p8_cer_str = f"{p8_cer_nd*100:.2f}%" if p8_cer_nd is not None else "N/A"
        p8_wer_str = f"{p8_wer_nd*100:.2f}%" if p8_wer_nd is not None else "N/A"
        d_cer_str = f"{d_cer_nd:+.1f}%" if d_cer_nd is not None else "N/A"
        d_wer_str = f"{d_wer_nd:+.1f}%" if d_wer_nd is not None else "N/A"
        print(
            f"  {ds:<30} {p2_cer_str:>8} {p8_cer_str:>8} {d_cer_str:>8} "
            f"{p2_wer_str:>8} {p8_wer_str:>8} {d_wer_str:>8} {r.num_samples:>6}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    setup_logging(results_dir)

    logger.info("Phase 8: RAG-Augmented OCR Correction  (mode=%s)", args.mode)
    logger.info("Config: %s", args.config)
    logger.info("Results dir: %s", results_dir)

    config = load_config(args.config)
    phase8_cfg = config.get("phase8", {})

    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")
    analyze_errors = not args.no_error_analysis

    # Retrieval mode: CLI > config > default
    retrieval_mode = (
        args.retrieval_mode
        or phase8_cfg.get("retrieval_mode", "bm25")
    )
    logger.info("Retrieval mode: %s", retrieval_mode)

    # ------------------------------------------------------------------
    # BUILD-INDEX mode
    # ------------------------------------------------------------------
    if args.mode == "build-index":
        run_build_index(config, results_dir, retrieval_mode, args.force)
        return

    # ------------------------------------------------------------------
    # EXPORT mode
    # ------------------------------------------------------------------
    if args.mode == "export":
        active_datasets = resolve_datasets(config, args.datasets)

        sample_ids: Optional[set[str]] = None
        if args.sample_list:
            sample_ids, sl_datasets = load_sample_list(args.sample_list)
            if not args.datasets:
                active_datasets = sl_datasets
            logger.info("Sample list: %d IDs from %s", len(sample_ids), args.sample_list)

        logger.info("Datasets to process: %s", active_datasets)

        run_export(
            config=config,
            active_datasets=active_datasets,
            results_dir=results_dir,
            retrieval_mode=retrieval_mode,
            limit=limit,
            force=args.force,
            sample_ids=sample_ids,
        )
        return

    # ------------------------------------------------------------------
    # ANALYZE mode
    # ------------------------------------------------------------------
    if args.mode == "analyze":
        active_datasets = resolve_datasets(config, args.datasets)

        sample_ids_analyze: Optional[set[str]] = None
        if args.sample_list:
            sample_ids_analyze, sl_datasets = load_sample_list(args.sample_list)
            if not args.datasets:
                active_datasets = sl_datasets

        logger.info("Datasets to process: %s", active_datasets)

        _maybe_split_combined_corrections(results_dir, force=args.force)

        all_corrected: dict[str, MetricResult] = {}
        all_comparisons: dict[str, dict] = {}

        for ds_key in active_datasets:
            try:
                # Resume: skip if already analyzed
                metrics_path = results_dir / ds_key / "metrics.json"
                if metrics_path.exists() and not args.force:
                    logger.info(
                        "[%s] Already analyzed -- skipping (use --force to re-analyze).",
                        ds_key,
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
                corrected_samples = load_corrections(corrections_path)
                if limit:
                    corrected_samples = corrected_samples[:limit]

                result = process_dataset_analyze(
                    dataset_key=ds_key,
                    corrected_samples=corrected_samples,
                    config=config,
                    results_dir=results_dir,
                    phase2_dir=args.phase2_dir,
                    limit=limit,
                    analyze_errors=analyze_errors,
                )
                all_corrected[ds_key] = result

                cmp_path = results_dir / ds_key / "comparison_vs_phase2.json"
                if cmp_path.exists():
                    with open(cmp_path, encoding="utf-8") as f:
                        all_comparisons[ds_key] = json.load(f)

            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing %s: %s", ds_key, exc, exc_info=True)

        if not all_corrected:
            logger.error("No datasets were successfully analyzed.")
            sys.exit(1)

        # Aggregate
        aggregate_results(all_corrected, config, results_dir, limit)
        aggregate_comparisons(all_comparisons, config, results_dir)

        # Report
        model_name = config.get("model", {}).get("name", "")
        generate_report(all_corrected, all_comparisons, model_name, results_dir)

        # Print summary
        print_summary(all_corrected, all_comparisons, results_dir)

        logger.info("Phase 8 analysis complete. Results in %s", results_dir)


if __name__ == "__main__":
    main()
