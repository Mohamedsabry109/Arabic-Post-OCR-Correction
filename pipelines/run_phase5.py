#!/usr/bin/env python3
"""Phase 5: Retrieval-Augmented Generation (RAG) with OpenITI corpus.

Isolated experiment comparing RAG-augmented correction against Phase 2
(zero-shot) baseline. For each OCR sample the top-K most similar sentences
from the OpenITI corpus are retrieved and injected into the system prompt.

Pipeline
--------
  Stage 0 (one-time local):
    python pipelines/run_phase5.py --mode build

  Stage 1 (local):
    python pipelines/run_phase5.py --mode export

  Stage 2 (Kaggle/Colab):
    python scripts/infer.py \\
        --input  results/phase5/inference_input.jsonl \\
        --output results/phase5/corrections.jsonl

  Stage 3 (local):
    python pipelines/run_phase5.py --mode analyze

Usage
-----
    python pipelines/run_phase5.py --mode build
    python pipelines/run_phase5.py --mode build  --max-sentences 1000  # smoke test
    python pipelines/run_phase5.py --mode export
    python pipelines/run_phase5.py --mode export --limit 50 --datasets KHATT-train
    python pipelines/run_phase5.py --mode analyze
    python pipelines/run_phase5.py --mode analyze --datasets KHATT-train
    python pipelines/run_phase5.py --mode analyze --no-error-analysis
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
from src.data.knowledge_base import OpenITILoader, CorpusSentence
from src.core.rag_retriever import RAGRetriever
from src.core.prompt_builder import PromptBuilder
from src.core.llm_corrector import CorrectedSample
from src.analysis.metrics import MetricResult, calculate_metrics
from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType
from pipelines._utils import resolve_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: RAG with OpenITI corpus (build / export / analyze)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["build", "export", "analyze"],
        help=(
            "build   -> build OpenITI corpus and FAISS index (one-time); "
            "export  -> retrieve for each OCR sample; write inference_input.jsonl; "
            "analyze -> compute metrics from corrections.jsonl"
        ),
    )
    # Build-mode options
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        dest="max_sentences",
        help="Max corpus sentences to extract from OpenITI (default: from config).",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        dest="corpus_path",
        help="Override path to corpus.jsonl (default: results/phase5/corpus.jsonl).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        dest="index_path",
        help="Override path to FAISS index (default: results/phase5/faiss.index).",
    )
    # Export-mode options
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Number of sentences to retrieve per OCR sample (default: from config).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        dest="min_score",
        help="Minimum cosine similarity to include (default: from config).",
    )
    # Analyze-mode options
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("results/phase2"),
        dest="phase2_dir",
        help="Phase 2 results directory (baseline for comparison).",
    )
    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        default=False,
        help="Skip error_changes.json computation (analyze mode, faster).",
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
        help="Re-process datasets/steps that already have results.",
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
        help="Override phase 5 results directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers (same pattern as Phase 4)
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    """Configure logging to console (UTF-8) and log file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "phase5.log"

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
    """Build the standard metadata block for all Phase 5 output JSON files."""
    model_cfg = config.get("model", {})
    rag_cfg = config.get("rag", {})
    meta = {
        "phase": "phase5",
        "dataset": dataset,
        "model": model_cfg.get("name", "Qwen/Qwen3-4B-Instruct-2507"),
        "backend": model_cfg.get("backend", "transformers"),
        "embedding_model": rag_cfg.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
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
# Shared corrections loading helpers (same as Phase 4)
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
        gt = cs.sample.gt_text
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
# Retrieval analysis
# ---------------------------------------------------------------------------


def compute_retrieval_analysis(
    dataset_key: str,
    inference_input_path: Path,
    corrected_samples: list[CorrectedSample],
    config: dict,
    top_k: int,
) -> dict:
    """Compute retrieval quality statistics for one dataset.

    Loads retrieval_scores from the inference_input.jsonl records,
    then correlates high/low retrieval score with CER improvement.

    Args:
        dataset_key: Dataset key (for filtering).
        inference_input_path: Path to the inference_input.jsonl.
        corrected_samples: List of CorrectedSample with gt/ocr/corrected texts.
        config: Parsed config dict.
        top_k: Number of retrieved sentences used.

    Returns:
        retrieval_analysis dict ready to save as JSON.
    """
    rag_cfg = config.get("rag", {})
    embedding_model = rag_cfg.get(
        "embedding_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Load retrieval scores from inference_input.jsonl
    retrieval_by_id: dict[str, list[float]] = {}
    if inference_input_path.exists():
        with open(inference_input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("dataset") == dataset_key:
                        scores = r.get("retrieval_scores", [])
                        retrieval_by_id[r["sample_id"]] = scores
                except json.JSONDecodeError:
                    pass

    if not retrieval_by_id:
        logger.warning(
            "[%s] No retrieval scores found in %s -- skipping retrieval analysis.",
            dataset_key, inference_input_path,
        )
        return {}

    # Aggregate retrieval statistics
    top1_scores: list[float] = []
    topk_avg_scores: list[float] = []
    zero_result_count = 0

    score_buckets = {"0.8+": 0, "0.7-0.8": 0, "0.6-0.7": 0, "<0.6": 0}

    for scores in retrieval_by_id.values():
        if not scores:
            zero_result_count += 1
            continue
        top1 = scores[0]
        top1_scores.append(top1)
        topk_avg_scores.append(sum(scores) / len(scores))

        if top1 >= 0.8:
            score_buckets["0.8+"] += 1
        elif top1 >= 0.7:
            score_buckets["0.7-0.8"] += 1
        elif top1 >= 0.6:
            score_buckets["0.6-0.7"] += 1
        else:
            score_buckets["<0.6"] += 1

    avg_top1 = sum(top1_scores) / max(len(top1_scores), 1)
    avg_topk = sum(topk_avg_scores) / max(len(topk_avg_scores), 1)

    # CER improvement correlation: high vs low score samples
    from src.analysis.metrics import calculate_cer

    # Compute per-sample CER delta (ocr_cer - corrected_cer vs gt)
    per_sample: list[tuple[float, float]] = []  # (top1_score, cer_improvement)
    for cs in corrected_samples:
        sid = cs.sample.sample_id
        if sid not in retrieval_by_id:
            continue
        scores = retrieval_by_id[sid]
        if not scores:
            continue
        gt = cs.sample.gt_text
        ocr = cs.sample.ocr_text
        corrected = cs.corrected_text
        if not gt:
            continue
        try:
            cer_ocr = calculate_cer(gt, ocr)
            cer_cor = calculate_cer(gt, corrected)
            improvement = cer_ocr - cer_cor  # positive = improvement
            per_sample.append((scores[0], improvement))
        except Exception:  # noqa: BLE001
            pass

    # Split into high/low halves by top1 score
    high_low: dict = {}
    if per_sample:
        per_sample.sort(key=lambda x: x[0], reverse=True)
        half = max(1, len(per_sample) // 2)
        high_group = per_sample[:half]
        low_group  = per_sample[half:]
        high_low = {
            "high_score_samples": {
                "n_samples":       len(high_group),
                "avg_top1_score":  round(sum(s for s, _ in high_group) / len(high_group), 4),
                "avg_cer_improvement": round(
                    sum(d for _, d in high_group) / len(high_group), 6
                ),
            },
            "low_score_samples": {
                "n_samples":       len(low_group),
                "avg_top1_score":  round(sum(s for s, _ in low_group) / max(len(low_group), 1), 4),
                "avg_cer_improvement": round(
                    sum(d for _, d in low_group) / max(len(low_group), 1), 6
                ),
            },
            "note": (
                "avg_cer_improvement > 0 means that group had better correction "
                "than OCR baseline. Does higher retrieval score correlate with "
                "more improvement?"
            ),
        }

    return {
        "meta": {
            "dataset":         dataset_key,
            "top_k":           top_k,
            "embedding_model": embedding_model,
            "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "retrieval_stats": {
            "samples_with_any_retrieval": len(top1_scores),
            "samples_zero_results":       zero_result_count,
            "avg_top1_score":             round(avg_top1, 4),
            "avg_topk_score":             round(avg_topk, 4),
            "score_distribution":         score_buckets,
        },
        "retrieval_by_error_type": high_low,
    }


# ---------------------------------------------------------------------------
# Stage 0: Build corpus and FAISS index
# ---------------------------------------------------------------------------


def run_build(
    config: dict,
    results_dir: Path,
    corpus_path: Path,
    index_path: Path,
    max_sentences: int,
    force: bool,
) -> None:
    """Build OpenITI corpus and FAISS index (one-time step).

    Args:
        config: Parsed config dict.
        results_dir: Phase 5 results root.
        corpus_path: Path to save/load corpus.jsonl.
        index_path: Path to save FAISS index.
        max_sentences: Maximum sentences to extract.
        force: If True, rebuild even if files exist.
    """
    p5_cfg = config.get("phase5", {})
    corpus_cfg = p5_cfg.get("corpus", {})
    index_cfg = p5_cfg.get("index", {})

    min_char_len = corpus_cfg.get("min_char_len", 30)
    max_char_len = corpus_cfg.get("max_char_len", 300)
    status_filter = corpus_cfg.get("status_filter", "pri")
    seed = corpus_cfg.get("seed", 42)
    batch_size = index_cfg.get("batch_size", 256)

    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build or load corpus
    # ------------------------------------------------------------------
    if corpus_path.exists() and not force:
        logger.info("Corpus already exists at %s (use --force to rebuild).", corpus_path)
        logger.info("Loading existing corpus ...")
        loader = OpenITILoader(config)
        sentences = loader.load_corpus(corpus_path)
    else:
        logger.info("=" * 60)
        logger.info("Phase 5 BUILD: extracting corpus from OpenITI ...")
        logger.info("  Target sentences : %d", max_sentences)
        logger.info("  min_char_len     : %d", min_char_len)
        logger.info("  max_char_len     : %d", max_char_len)
        logger.info("  status_filter    : %s", status_filter)
        logger.info("  seed             : %d", seed)
        logger.info("=" * 60)

        loader = OpenITILoader(config)
        try:
            sentences = loader.load(
                max_sentences=max_sentences,
                min_char_len=min_char_len,
                max_char_len=max_char_len,
                status_filter=status_filter,
                seed=seed,
                show_progress=True,
            )
        except FileNotFoundError as exc:
            logger.error(str(exc))
            sys.exit(1)

        if not sentences:
            logger.error("No sentences extracted from OpenITI. Check data/OpenITI/ path.")
            sys.exit(1)

        loader.save_corpus(sentences, corpus_path)

        # Log era distribution
        era_counts: dict[str, int] = {}
        for s in sentences:
            d = s.date
            if d < 300:
                era = "0-300"
            elif d < 600:
                era = "300-600"
            elif d < 900:
                era = "600-900"
            else:
                era = "900+"
            era_counts[era] = era_counts.get(era, 0) + 1
        logger.info("Corpus era distribution: %s", era_counts)

    logger.info("Corpus ready: %d sentences.", len(sentences))

    # ------------------------------------------------------------------
    # Step 2: Build FAISS index
    # ------------------------------------------------------------------
    sentences_list_path = Path(str(index_path) + ".sentences.jsonl")
    if index_path.exists() and sentences_list_path.exists() and not force:
        logger.info(
            "FAISS index already exists at %s (use --force to rebuild).", index_path
        )
        logger.info("Phase 5 build complete. Ready for export.")
        return

    logger.info("=" * 60)
    logger.info("Phase 5 BUILD: building FAISS index ...")
    logger.info("  Corpus size  : %d sentences", len(sentences))
    logger.info("  Batch size   : %d", batch_size)
    logger.info("=" * 60)

    retriever = RAGRetriever(config)
    try:
        retriever.build_index(
            corpus_path=corpus_path,
            index_path=index_path,
            batch_size=batch_size,
            show_progress=True,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Phase 5 build complete.")
    logger.info("  Corpus     : %s", corpus_path)
    logger.info("  FAISS index: %s", index_path)
    logger.info("")
    logger.info("NEXT STEP:")
    logger.info("  python pipelines/run_phase5.py --mode export")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Stage 1: Export (retrieve + write inference_input.jsonl)
# ---------------------------------------------------------------------------


def run_export(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    corpus_path: Path,
    index_path: Path,
    top_k: int,
    min_score: float,
    limit: Optional[int],
    force: bool,
) -> None:
    """Retrieve for each OCR sample and write inference_input.jsonl.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        results_dir: Phase 5 results root.
        corpus_path: Path to corpus.jsonl.
        index_path: Path to FAISS index.
        top_k: Number of sentences to retrieve per sample.
        min_score: Minimum cosine similarity threshold.
        limit: Max samples per dataset.
        force: If True, ignore existing exported datasets.
    """
    output_path = results_dir / "inference_input.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check index exists
    if not index_path.exists():
        logger.error(
            "FAISS index not found: %s\n"
            "Run: python pipelines/run_phase5.py --mode build",
            index_path,
        )
        sys.exit(1)

    # Load retriever
    retriever = RAGRetriever(config)
    try:
        retriever.load_index(index_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Cannot load RAG retriever: %s", exc)
        sys.exit(1)

    if not retriever.enabled:
        logger.error(
            "RAGRetriever is disabled (faiss or sentence-transformers not installed). "
            "Install them and re-run."
        )
        sys.exit(1)

    p5_ret_cfg = config.get("phase5", {}).get("retrieval", {})
    format_style = p5_ret_cfg.get("format_style", "numbered_arabic")

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

            try:
                samples = list(loader_data.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("Skipping %s: %s", ds_key, exc)
                continue

            n_with_retrieval = 0
            for sample in tqdm(samples, desc=f"  Retrieve {ds_key}", unit="sample"):
                chunks = retriever.retrieve(sample.ocr_text, k=top_k, min_score=min_score)
                retrieval_context = retriever.format_for_prompt(chunks, style=format_style)
                prompt_type = "rag" if chunks else "zero_shot"

                record = {
                    "sample_id":         sample.sample_id,
                    "dataset":           ds_key,
                    "ocr_text":          sample.ocr_text,
                    "gt_text":           sample.gt_text,
                    "prompt_type":       prompt_type,
                    "retrieval_context": retrieval_context,
                    "retrieved_k":       len(chunks),
                    "retrieval_scores":  [c.score for c in chunks],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                if chunks:
                    n_with_retrieval += 1

            logger.info(
                "Exported %d samples for %s "
                "(%d with retrieval, %d zero-shot fallback).",
                len(samples), ds_key,
                n_with_retrieval, len(samples) - n_with_retrieval,
            )

    logger.info("=" * 60)
    logger.info("Phase 5 export complete: %d new samples -> %s", total_written, output_path)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push latest code:  git push")
    logger.info("  2. On Kaggle/Colab:")
    logger.info("       python scripts/infer.py \\")
    logger.info("           --input  results/phase5/inference_input.jsonl \\")
    logger.info("           --output results/phase5/corrections.jsonl")
    logger.info("  3. Run analysis locally:")
    logger.info("       python pipelines/run_phase5.py --mode analyze")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Stage 3: Analyze
# ---------------------------------------------------------------------------


def process_dataset_analyze(
    dataset_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
    phase2_dir: Path,
    top_k: int,
    limit: Optional[int],
    analyze_errors: bool,
) -> MetricResult:
    """Run all analysis steps for one dataset (Phase 5).

    Steps:
    1. CER/WER on corrected texts
    2. Comparison vs Phase 2 (isolated)
    3. Error change analysis (optional)
    4. Retrieval quality analysis

    Args:
        dataset_key: e.g. "KHATT-train".
        corrected_samples: Loaded from corrections.jsonl for this dataset.
        config: Parsed config dict.
        results_dir: Phase 5 results root.
        phase2_dir: Phase 2 results root (baseline metrics).
        top_k: Number of retrieved sentences used (for metadata).
        limit: Sample limit applied (for metadata).
        analyze_errors: If True, compute error_changes.json.

    Returns:
        MetricResult for corrected text.
    """
    out_dir = results_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = PromptBuilder()
    n = len(corrected_samples)
    n_failed = sum(1 for cs in corrected_samples if not cs.success)
    total_prompt_tokens = sum(cs.prompt_tokens for cs in corrected_samples)
    total_output_tokens = sum(cs.output_tokens for cs in corrected_samples)
    total_latency = sum(cs.latency_s for cs in corrected_samples)
    avg_latency = total_latency / max(n, 1)

    logger.info("=" * 60)
    logger.info(
        "Analyzing [phase5] dataset: %s  (%d samples, %d failed)",
        dataset_key, n, n_failed,
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

    p5_cfg = config.get("phase5", {})
    corpus_cfg = p5_cfg.get("corpus", {})
    metrics_extra = {
        "prompt_type":              "rag",
        "prompt_version":           builder.rag_prompt_version,
        "top_k":                    top_k,
        "corpus_size":              corpus_cfg.get("max_sentences", 200000),
        "total_prompt_tokens":      total_prompt_tokens,
        "total_output_tokens":      total_output_tokens,
        "total_latency_s":          round(total_latency, 2),
        "avg_latency_per_sample_s": round(avg_latency, 3),
        "failed_samples":           n_failed,
    }

    metrics_json = {
        "meta": make_meta(dataset_key, n, config, limit, extra=metrics_extra),
        "corrected": corrected_result.to_dict(),
    }
    save_json(metrics_json, out_dir / "metrics.json")

    logger.info(
        "[%s] phase5 CER=%.2f%%  WER=%.2f%%",
        dataset_key, corrected_result.cer * 100, corrected_result.wer * 100,
    )

    # ------------------------------------------------------------------
    # Step 2: Comparison vs Phase 2 (isolated)
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
                dataset_key, n, config, limit,
                extra={"comparison": "phase5_vs_phase2"},
            ),
            "phase2_baseline": {
                "cer": round(p2_cer, 6),
                "wer": round(p2_wer, 6),
                "source": str(phase2_dir / dataset_key / "metrics.json"),
            },
            "phase5_corrected": {
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
            "[%s] Phase2->Phase5 CER: %.2f%% -> %.2f%% (%+.1f%%) | "
            "WER: %.2f%% -> %.2f%% (%+.1f%%)",
            dataset_key,
            p2_cer * 100, corrected_result.cer * 100, cer_rel,
            p2_wer * 100, corrected_result.wer * 100, wer_rel,
        )
    else:
        logger.warning(
            "[%s] Phase 2 metrics not found at %s -- skipping comparison.",
            dataset_key, phase2_dir / dataset_key / "metrics.json",
        )

    # ------------------------------------------------------------------
    # Step 3: Error change analysis (optional)
    # ------------------------------------------------------------------
    if analyze_errors:
        logger.info("[%s] Running error change analysis ...", dataset_key)
        error_changes = run_error_change_analysis(corrected_samples, dataset_key)
        error_changes["meta"] = make_meta(
            dataset_key, n, config, limit,
            extra={"prompt_type": "rag"},
        )
        save_json(error_changes, out_dir / "error_changes.json")

    # ------------------------------------------------------------------
    # Step 4: Retrieval quality analysis
    # ------------------------------------------------------------------
    logger.info("[%s] Computing retrieval quality analysis ...", dataset_key)
    try:
        retrieval_analysis = compute_retrieval_analysis(
            dataset_key=dataset_key,
            inference_input_path=results_dir / "inference_input.jsonl",
            corrected_samples=corrected_samples,
            config=config,
            top_k=top_k,
        )
        if retrieval_analysis:
            save_json(retrieval_analysis, out_dir / "retrieval_analysis.json")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] Retrieval analysis failed (non-fatal): %s", dataset_key, exc
        )

    return corrected_result


def run_analyze(
    config: dict,
    active_datasets: list[str],
    results_dir: Path,
    phase2_dir: Path,
    top_k: int,
    limit: Optional[int],
    force: bool,
    analyze_errors: bool,
) -> tuple[dict[str, MetricResult], dict[str, dict]]:
    """Analyze Phase 5 corrections across all active datasets.

    Args:
        config: Parsed config dict.
        active_datasets: Dataset keys to process.
        results_dir: Phase 5 results root.
        phase2_dir: Phase 2 results root (baseline metrics).
        top_k: Number of retrieved sentences used (for metadata).
        limit: Sample limit applied.
        force: If True, re-analyze already-done datasets.
        analyze_errors: If True, compute error_changes.json.

    Returns:
        Tuple of (all_corrected, all_comparisons) dicts.
    """
    _maybe_split_combined_corrections(results_dir)

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

            metric_result = process_dataset_analyze(
                dataset_key=ds_key,
                corrected_samples=corrected_samples,
                config=config,
                results_dir=results_dir,
                phase2_dir=phase2_dir,
                top_k=top_k,
                limit=limit,
                analyze_errors=analyze_errors,
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
# Aggregation + Report
# ---------------------------------------------------------------------------


def aggregate_results(
    all_corrected: dict[str, MetricResult],
    config: dict,
    results_dir: Path,
    top_k: int,
    limit: Optional[int],
) -> None:
    """Write combined metrics.json across all datasets."""
    builder = PromptBuilder()
    p5_cfg = config.get("phase5", {})
    rag_cfg = config.get("rag", {})

    output = {
        "meta": {
            "phase": "phase5",
            "model": config.get("model", {}).get("name", ""),
            "prompt_type": "rag",
            "prompt_version": builder.rag_prompt_version,
            "embedding_model": rag_cfg.get("embedding_model", ""),
            "top_k": top_k,
            "corpus_size": p5_cfg.get("corpus", {}).get("max_sentences", 200000),
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
            "phase": "phase5",
            "comparison": "phase5_vs_phase2",
            "model": config.get("model", {}).get("name", ""),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": get_git_commit(),
        },
        "datasets": all_comparisons,
        "note": (
            "PHASE5 is an isolated experiment. "
            "Delta measures the contribution of RAG (OpenITI corpus) over "
            "zero-shot (Phase 2). Negative CER delta = improvement."
        ),
    }
    save_json(output, results_dir / "comparison.json")


def generate_report(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
    results_dir: Path,
    config: dict,
    top_k: int,
) -> None:
    """Write human-readable Markdown report to results_dir/report.md."""
    model_name = config.get("model", {}).get("name", "unknown")
    rag_cfg = config.get("rag", {})
    embedding_model = rag_cfg.get("embedding_model", "unknown")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Phase 5 Report: RAG with OpenITI Corpus")
    lines.append(f"\nGenerated: {now}")
    lines.append(f"Model: {model_name}")
    lines.append(f"Embedding: {embedding_model} (top-{top_k})")
    lines.append("")

    lines.append("## Results vs Phase 2 (Zero-Shot Baseline)\n")
    lines.append("> Isolated comparison. Phase 5 vs Phase 2 only.\n")
    if all_comparisons:
        lines.append(
            "| Dataset | Phase 2 CER | Phase 5 CER | Delta CER | "
            "Phase 2 WER | Phase 5 WER | Delta WER |"
        )
        lines.append(
            "|---------|-------------|-------------|-----------|"
            "-------------|-------------|-----------|"
        )
        for ds, cmp in all_comparisons.items():
            p2 = cmp.get("phase2_baseline", {})
            p5 = cmp.get("phase5_corrected", {})
            d  = cmp.get("delta", {})
            lines.append(
                f"| {ds} "
                f"| {p2.get('cer', 0)*100:.2f}% "
                f"| {p5.get('cer', 0)*100:.2f}% "
                f"| {d.get('cer_relative_pct', 0):+.1f}% "
                f"| {p2.get('wer', 0)*100:.2f}% "
                f"| {p5.get('wer', 0)*100:.2f}% "
                f"| {d.get('wer_relative_pct', 0):+.1f}% |"
            )
    else:
        lines.append("*Phase 2 baseline not available -- run Phase 2 first.*")
    lines.append("")

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

    # Retrieval quality summary
    any_ra = False
    for ds in all_corrected:
        ra_path = results_dir / ds / "retrieval_analysis.json"
        if ra_path.exists():
            try:
                with open(ra_path, encoding="utf-8") as f:
                    ra = json.load(f)
                rs = ra.get("retrieval_stats", {})
                if not any_ra:
                    lines.append("## Retrieval Quality\n")
                    any_ra = True
                lines.append(f"### {ds}\n")
                lines.append(
                    f"- Samples with retrieval: {rs.get('samples_with_any_retrieval', 0):,}\n"
                    f"- Zero-result samples: {rs.get('samples_zero_results', 0):,}\n"
                    f"- Avg top-1 cosine similarity: {rs.get('avg_top1_score', 0):.4f}\n"
                    f"- Score distribution: {rs.get('score_distribution', {})}"
                )
                lines.append("")
            except (json.JSONDecodeError, OSError):
                pass

    # Error change summary
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

    lines.append("## Key Findings\n")
    if all_corrected:
        best  = min(all_corrected.items(), key=lambda x: x[1].cer)
        worst = max(all_corrected.items(), key=lambda x: x[1].cer)
        lines.append(f"- Best CER:  **{best[0]}** at {best[1].cer*100:.2f}%")
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
        "> Phase 5 is an isolated experiment comparing RAG (OpenITI) vs zero-shot.\n"
        "> Phase 6 will combine all knowledge sources for the full system."
    )

    report_path = results_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


def print_summary(
    all_corrected: dict[str, MetricResult],
    all_comparisons: dict[str, dict],
) -> None:
    """Print a final summary table to stdout."""
    print("\n" + "=" * 80)
    print("PHASE5 (RAG) SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<28} {'P2 CER':>8} {'P5 CER':>8} {'D CER':>8} {'P5 WER':>8} {'N':>6}")
    print("-" * 80)
    for ds, r in all_corrected.items():
        cmp = all_comparisons.get(ds, {})
        p2_cer = cmp.get("phase2_baseline", {}).get("cer", 0.0)
        cer_rel = cmp.get("delta", {}).get("cer_relative_pct", 0.0)
        p2_str    = f"{p2_cer*100:.2f}%" if cmp else "N/A"
        delta_str = f"{cer_rel:+.1f}%"   if cmp else "N/A"
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

    logger.info("Phase 5: RAG with OpenITI  (mode=%s)", args.mode)
    logger.info("Results dir: %s", results_dir)

    config = load_config(args.config)
    limit = args.limit or config.get("processing", {}).get("limit_per_dataset")

    p5_cfg = config.get("phase5", {})
    corpus_cfg = p5_cfg.get("corpus", {})
    ret_cfg = p5_cfg.get("retrieval", {})

    # Resolve paths
    corpus_path = args.corpus_path or (results_dir / "corpus.jsonl")
    index_path  = args.index_path  or (results_dir / "faiss.index")

    # Resolve retrieval params
    top_k = (
        args.top_k
        or ret_cfg.get("top_k")
        or config.get("rag", {}).get("top_k", 3)
    )
    min_score = (
        args.min_score
        if args.min_score is not None
        else ret_cfg.get("min_score", 0.0)
    )

    # Resolve corpus size
    max_sentences = (
        args.max_sentences
        or corpus_cfg.get("max_sentences", 200_000)
    )

    active_datasets = resolve_datasets(config, args.datasets)
    logger.info("Active datasets: %s", active_datasets)

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------
    if args.mode == "build":
        run_build(
            config=config,
            results_dir=results_dir,
            corpus_path=corpus_path,
            index_path=index_path,
            max_sentences=max_sentences,
            force=args.force,
        )
        return

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------
    if args.mode == "export":
        run_export(
            config=config,
            active_datasets=active_datasets,
            results_dir=results_dir,
            corpus_path=corpus_path,
            index_path=index_path,
            top_k=top_k,
            min_score=min_score,
            limit=limit,
            force=args.force,
        )
        return

    # ------------------------------------------------------------------
    # ANALYZE
    # ------------------------------------------------------------------
    analyze_errors = not args.no_error_analysis
    all_corrected, all_comparisons = run_analyze(
        config=config,
        active_datasets=active_datasets,
        results_dir=results_dir,
        phase2_dir=args.phase2_dir,
        top_k=top_k,
        limit=limit,
        force=args.force,
        analyze_errors=analyze_errors,
    )

    if not all_corrected:
        logger.error("No datasets successfully processed.")
        sys.exit(1)

    aggregate_results(all_corrected, config, results_dir, top_k, limit)
    aggregate_comparisons(all_comparisons, config, results_dir, limit)
    generate_report(all_corrected, all_comparisons, results_dir, config, top_k)
    print_summary(all_corrected, all_comparisons)
    logger.info("Phase 5 complete. Results in: %s", results_dir)


if __name__ == "__main__":
    main()
