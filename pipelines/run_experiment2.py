#!/usr/bin/env python3
"""Experiment 2: Prompt x Strategy ablation on full-page Arabic datasets.

Tests 8 combinations of:
  - Prompt style  : base vs. conservative (v2)
  - Examples set  : generic vs. hand-picked (hp)
  - Strategy      : zero-shot vs. BM25-RAG

Datasets evaluated (validation / testing splits only):
  PATS-A01-*-val (8 fonts), KHATT-validation,
  KHATT-Paragraph-validation, Yarmouk-testing,
  Muharaf-validation, Historical

Trial table
-----------
  T1  base_zs       configs/crafted_system_prompt.txt               zero-shot
  T2  base_rag      configs/crafted_system_prompt.txt               BM25-RAG
  T3  cons_zs       configs/crafted_system_prompt_v2.txt            zero-shot
  T4  cons_rag      configs/crafted_system_prompt_v2.txt            BM25-RAG
  T5  hp_zs         configs/crafted_system_prompt_hand_picked.txt   zero-shot
  T6  hp_rag        configs/crafted_system_prompt_hand_picked.txt   BM25-RAG
  T7  hp_cons_zs    configs/crafted_system_prompt_hand_picked_v2.txt zero-shot
  T8  hp_cons_rag   configs/crafted_system_prompt_hand_picked_v2.txt BM25-RAG

JSONL record format
-------------------
  Zero-shot records (prompt_type="zero_shot"):
    {sample_id, dataset, ocr_text, gt_text, prompt_type, prompt_version}

  RAG records (prompt_type="rag"):
    {sample_id, dataset, ocr_text, gt_text, prompt_type, prompt_version,
     retrieved_sentences, retrieved_words, retrieval_mode}

  The system prompt is NOT embedded — it is passed via --system-prompt to
  infer.py at inference time.  prompt_version="crafted" tells infer.py to
  use whatever file was given via --system-prompt.

  All four zero-shot trial JSOLs contain identical records (same data, separate
  files).  Same for the four RAG trial JSOLs.  Export generates the data once
  and copies to avoid redundant BM25 retrieval.

Workflow
--------
  LOCAL:  python pipelines/run_experiment2.py --mode export
          (produces results/experiment2/inputs/trial{N}_{name}.jsonl x8)

  REMOTE (Kaggle / Thunder): run 8 separate infer.py commands printed after
          export — each trial gets its own --system-prompt and --output.

  LOCAL:  python pipelines/run_experiment2.py --mode analyze
          python pipelines/run_experiment2.py --mode summarize

Usage
-----
    python pipelines/run_experiment2.py --mode export
    python pipelines/run_experiment2.py --mode export --limit 20
    python pipelines/run_experiment2.py --mode export --trials 1 2
    python pipelines/run_experiment2.py --mode analyze
    python pipelines/run_experiment2.py --mode analyze --trials 1 2
    python pipelines/run_experiment2.py --mode summarize
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.data_loader import DataLoader, DataError, OCRSample
from src.core.llm_corrector import CorrectedSample
from src.analysis.metrics import MetricResult, calculate_metrics_dual
from src.data.rag_index import RAGIndexBuilder, RAGRetriever
from pipelines._utils import split_runaway_samples, DEFAULT_RUNAWAY_RATIO_THRESHOLD

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("results/experiment2")
_DEFAULT_INDEX_DIR = Path("results/phase8/index")

DEFAULT_DATASETS = [
    # Original PATS-A01 validation splits (8 fonts)
    "PATS-A01-Akhbar-val",
    "PATS-A01-Andalus-val",
    "PATS-A01-Arial-val",
    "PATS-A01-Naskh-val",
    "PATS-A01-Simplified-val",
    "PATS-A01-Tahoma-val",
    "PATS-A01-Thuluth-val",
    "PATS-A01-Traditional-val",
    # Original KHATT validation
    "KHATT-validation",
    # New full-page datasets (validation / testing only)
    "KHATT-Paragraph-validation",
    "Yarmouk-testing",
    "Muharaf-validation",
    "Historical",
]


# ---------------------------------------------------------------------------
# Trial definitions
# ---------------------------------------------------------------------------


@dataclass
class TrialConfig:
    id: int
    name: str
    system_prompt_file: str
    use_rag: bool

    @property
    def label(self) -> str:
        return f"trial{self.id}_{self.name}"

    def input_path(self, results_dir: Path) -> Path:
        return results_dir / "inputs" / f"{self.label}.jsonl"

    def corrections_path(self, results_dir: Path) -> Path:
        return results_dir / "corrections" / f"{self.label}.jsonl"

    def output_dir(self, results_dir: Path) -> Path:
        return results_dir / self.label


TRIALS: list[TrialConfig] = [
    TrialConfig(1, "base_zs",      "configs/crafted_system_prompt.txt",                use_rag=False),
    TrialConfig(2, "base_rag",     "configs/crafted_system_prompt.txt",                use_rag=True),
    TrialConfig(3, "cons_zs",      "configs/crafted_system_prompt_v2.txt",             use_rag=False),
    TrialConfig(4, "cons_rag",     "configs/crafted_system_prompt_v2.txt",             use_rag=True),
    TrialConfig(5, "hp_zs",        "configs/crafted_system_prompt_hand_picked.txt",    use_rag=False),
    TrialConfig(6, "hp_rag",       "configs/crafted_system_prompt_hand_picked.txt",    use_rag=True),
    TrialConfig(7, "hp_cons_zs",   "configs/crafted_system_prompt_hand_picked_v2.txt", use_rag=False),
    TrialConfig(8, "hp_cons_rag",  "configs/crafted_system_prompt_hand_picked_v2.txt", use_rag=True),
]

_TRIAL_BY_ID: dict[int, TrialConfig] = {t.id: t for t in TRIALS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Prompt x Strategy ablation on full-page datasets."
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["export", "analyze", "summarize", "commands"],
        help=(
            "export   -> build per-trial input JSOLs; "
            "commands -> print the 8 infer.py commands (no data written); "
            "analyze  -> read corrections, compute metrics; "
            "summarize -> print cross-trial table from existing metrics"
        ),
    )
    parser.add_argument(
        "--trials", type=int, nargs="+", default=None, metavar="N",
        help="Trial IDs to process (1-8). Default: all 8.",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None, metavar="DATASET",
        help="Dataset keys to process. Default: all 13.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per dataset (smoke testing).",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-export / re-analyze even if output already exists.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/config.yaml"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR, dest="results_dir",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=_DEFAULT_INDEX_DIR, dest="index_dir",
        help=f"RAG index directory. Default: {_DEFAULT_INDEX_DIR}",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    console_stream = open(
        sys.stdout.fileno(), mode="w", encoding="utf-8",
        errors="replace", closefd=False, buffering=1,
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(logging.StreamHandler(console_stream))
    fh = logging.FileHandler(results_dir / "experiment2.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Config not found at %s — using empty dict.", config_path)
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=_PROJECT_ROOT,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", path)


def _load_exported_datasets(path: Path) -> set[str]:
    if not path.exists():
        return set()
    found: set[str] = set()
    with open(path, encoding="utf-8") as f:
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
    return found


def _load_corrections_by_dataset(path: Path) -> dict[str, list[CorrectedSample]]:
    if not path.exists():
        return {}
    by_ds: dict[str, list[CorrectedSample]] = {}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d malformed: %s", lineno, exc)
                skipped += 1
                continue
            ds = r.get("dataset", "")
            if not ds:
                skipped += 1
                continue
            sample = OCRSample(
                sample_id=r["sample_id"],
                dataset=ds,
                font=None, split=None,
                ocr_text=r.get("ocr_text", ""),
                gt_text=r.get("gt_text", ""),
                ocr_path=Path(""),
                gt_path=None,
            )
            cs = CorrectedSample(
                sample=sample,
                corrected_text=r.get("corrected_text", r.get("ocr_text", "")),
                prompt_tokens=r.get("prompt_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                latency_s=r.get("latency_s", 0.0),
                success=r.get("success", True),
                error=r.get("error"),
            )
            by_ds.setdefault(ds, []).append(cs)
    if skipped:
        logger.warning("Skipped %d malformed lines in %s", skipped, path)
    return by_ds


# ---------------------------------------------------------------------------
# EXPORT MODE
# ---------------------------------------------------------------------------
#
# Strategy: generate the data ONCE for the first zero-shot trial and the first
# RAG trial, then copy the file for the remaining same-strategy trials.
# This avoids running BM25 retrieval four times for identical data.
#


def _write_zero_shot_jsonl(
    out_path: Path,
    active_datasets: list[str],
    config: dict,
    limit: Optional[int],
    force: bool,
) -> int:
    """Write zero-shot inference records to out_path. Returns total samples written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = _load_exported_datasets(out_path) if not force else set()
    if already:
        logger.info("  Resume: datasets already in file: %s", sorted(already))

    loader = DataLoader(config)
    total = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already:
                logger.info("  [%s] Already exported — skipping.", ds_key)
                continue
            try:
                samples = list(loader.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("  [%s] Skipping: %s", ds_key, exc)
                continue
            for sample in samples:
                record = {
                    "sample_id":      sample.sample_id,
                    "dataset":        ds_key,
                    "ocr_text":       sample.ocr_text,
                    "gt_text":        sample.gt_text,
                    "prompt_type":    "zero_shot",
                    "prompt_version": "crafted",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += len(samples)
            logger.info("  [%s] %d samples exported.", ds_key, len(samples))
    return total


def _write_rag_jsonl(
    out_path: Path,
    active_datasets: list[str],
    config: dict,
    index_dir: Path,
    limit: Optional[int],
    force: bool,
) -> int:
    """Write RAG inference records to out_path. Returns total samples written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = index_dir / "index_meta.json"
    if not meta_path.exists():
        logger.error(
            "RAG index not found at %s.\n"
            "  Build it first: python pipelines/run_phase8.py --mode build-index\n"
            "  Or pass --index-dir pointing to an existing index.",
            index_dir,
        )
        return 0

    phase8_cfg = config.get("phase8", {})
    index_builder = RAGIndexBuilder(config)
    index_builder.load(index_dir)
    logger.info("  RAG index loaded: %d sentence entries from %s.", len(index_builder.sentence_store), index_dir)

    retriever = RAGRetriever(
        index=index_builder,
        mode=phase8_cfg.get("retrieval_mode", "bm25"),
        alpha=phase8_cfg.get("alpha", 0.6),
        top_k_candidates=phase8_cfg.get("top_k_candidates", 50),
    )
    top_k_s = phase8_cfg.get("top_k_sentences", 5)
    top_k_w = phase8_cfg.get("top_k_words", 15)
    top_k_pw = phase8_cfg.get("word_top_k_per_input_word", 3)

    already = _load_exported_datasets(out_path) if not force else set()
    if already:
        logger.info("  Resume: datasets already in file: %s", sorted(already))

    loader = DataLoader(config)
    total = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for ds_key in active_datasets:
            if ds_key in already:
                logger.info("  [%s] Already exported — skipping.", ds_key)
                continue
            try:
                samples = list(loader.iter_samples(ds_key, limit=limit))
            except DataError as exc:
                logger.warning("  [%s] Skipping: %s", ds_key, exc)
                continue
            for sample in tqdm(samples, desc=f"  RAG {ds_key}", unit="sample"):
                sent = retriever.retrieve_sentences(sample.ocr_text, top_k=top_k_s)
                words = retriever.retrieve_words(sample.ocr_text, top_k=top_k_w, top_k_per_word=top_k_pw)
                ret_sents = retriever.format_sentences_for_prompt(sent)
                ret_words = retriever.format_words_for_prompt(words)
                has_ctx = bool(ret_sents.strip()) or bool(ret_words.strip())
                record = {
                    "sample_id":           sample.sample_id,
                    "dataset":             ds_key,
                    "ocr_text":            sample.ocr_text,
                    "gt_text":             sample.gt_text,
                    "prompt_type":         "rag" if has_ctx else "zero_shot",
                    "prompt_version":      "crafted",
                    "retrieved_sentences": ret_sents,
                    "retrieved_words":     ret_words,
                    "retrieval_mode":      phase8_cfg.get("retrieval_mode", "bm25"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += len(samples)
            logger.info("  [%s] %d samples exported (RAG).", ds_key, len(samples))
    return total


def run_export(
    active_trials: list[TrialConfig],
    active_datasets: list[str],
    config: dict,
    results_dir: Path,
    index_dir: Path,
    limit: Optional[int],
    force: bool,
) -> None:
    zs_trials  = [t for t in active_trials if not t.use_rag]
    rag_trials = [t for t in active_trials if t.use_rag]

    # --- Zero-shot trials ---
    if zs_trials:
        logger.info("=" * 60)
        logger.info("Zero-shot export: generating for %s", zs_trials[0].label)
        primary_zs = zs_trials[0]
        primary_path = primary_zs.input_path(results_dir)
        total = _write_zero_shot_jsonl(primary_path, active_datasets, config, limit, force)
        logger.info("  Total: %d samples -> %s", total, primary_path)

        # Copy to remaining zero-shot trials
        for trial in zs_trials[1:]:
            dst = trial.input_path(results_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not force:
                logger.info("  %s already exists — skipping copy.", dst.name)
            else:
                shutil.copy2(primary_path, dst)
                logger.info("  Copied -> %s", dst.name)

    # --- RAG trials ---
    if rag_trials:
        logger.info("=" * 60)
        logger.info("RAG export: generating for %s (retrieval runs once)", rag_trials[0].label)
        primary_rag = rag_trials[0]
        primary_path = primary_rag.input_path(results_dir)
        total = _write_rag_jsonl(primary_path, active_datasets, config, index_dir, limit, force)
        logger.info("  Total: %d samples -> %s", total, primary_path)

        # Copy to remaining RAG trials
        for trial in rag_trials[1:]:
            dst = trial.input_path(results_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not force:
                logger.info("  %s already exists — skipping copy.", dst.name)
            else:
                shutil.copy2(primary_path, dst)
                logger.info("  Copied -> %s", dst.name)

    _print_inference_commands(active_trials, results_dir)


def _print_inference_commands(active_trials: list[TrialConfig], results_dir: Path) -> None:
    sep = "=" * 70
    out_lines = [
        sep,
        "EXPORT COMPLETE -- run these commands on Kaggle / Thunder:",
        sep,
        "",
        "NOTE: prompt_version='crafted' in the JSONL means infer.py uses",
        "      whatever file you pass via --system-prompt. Each trial below",
        "      uses the same input JSONL as other same-strategy trials but",
        "      a different --system-prompt and a different --output path.",
        "",
        "-- ZERO-SHOT TRIALS (prompt_type=zero_shot) " + "-" * 26,
        "",
    ]
    for trial in active_trials:
        if trial.use_rag:
            continue
        inp = trial.input_path(results_dir)
        out = trial.corrections_path(results_dir)
        out_lines += [
            f"# T{trial.id}: {trial.name}",
            "python scripts/infer.py \\",
            f"    --input  {inp} \\",
            f"    --output {out} \\",
            f"    --system-prompt {trial.system_prompt_file} \\",
            "    --hf-repo YOUR_HF_REPO --hf-token $HF_TOKEN",
            "",
        ]
    out_lines += [
        "-- RAG TRIALS (prompt_type=rag, retrieved_sentences/words pre-filled) " + "-" * 0,
        "",
    ]
    for trial in active_trials:
        if not trial.use_rag:
            continue
        inp = trial.input_path(results_dir)
        out = trial.corrections_path(results_dir)
        out_lines += [
            f"# T{trial.id}: {trial.name}",
            "python scripts/infer.py \\",
            f"    --input  {inp} \\",
            f"    --output {out} \\",
            f"    --system-prompt {trial.system_prompt_file} \\",
            "    --hf-repo YOUR_HF_REPO --hf-token $HF_TOKEN",
            "",
        ]
    out_lines.append(sep)
    # Write to UTF-8 stdout
    stdout_utf8 = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
    stdout_utf8.write("\n".join(out_lines) + "\n")
    stdout_utf8.flush()


# ---------------------------------------------------------------------------
# ANALYZE MODE
# ---------------------------------------------------------------------------


def _analyze_trial_dataset(
    trial: TrialConfig,
    ds_key: str,
    corrected_samples: list[CorrectedSample],
    config: dict,
    results_dir: Path,
) -> dict:
    out_dir = trial.output_dir(results_dir) / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get("evaluation", {})
    threshold = eval_cfg.get("runaway_ratio_threshold", DEFAULT_RUNAWAY_RATIO_THRESHOLD)
    exclude_runaway = eval_cfg.get("exclude_runaway", False)

    n = len(corrected_samples)
    n_failed = sum(1 for cs in corrected_samples if not cs.success)

    normal_samples, runaway_samples, _ = split_runaway_samples(corrected_samples, threshold=threshold)

    ocr_all, ocr_all_nd = calculate_metrics_dual(corrected_samples, dataset_name=ds_key, text_field="ocr_text")
    cor_all, cor_all_nd = calculate_metrics_dual(corrected_samples, dataset_name=ds_key, text_field="corrected_text")
    cor_norm, cor_norm_nd = calculate_metrics_dual(normal_samples, dataset_name=ds_key, text_field="corrected_text")

    primary = cor_norm if exclude_runaway else cor_all
    primary_nd = cor_norm_nd if exclude_runaway else cor_all_nd

    metrics = {
        "meta": {
            "trial":           trial.label,
            "system_prompt":   trial.system_prompt_file,
            "use_rag":         trial.use_rag,
            "dataset":         ds_key,
            "num_samples":     n,
            "failed_samples":  n_failed,
            "runaway_count":   len(runaway_samples),
            "exclude_runaway": exclude_runaway,
            "git_commit":      get_git_commit(),
            "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "ocr_baseline":         ocr_all.to_dict(),
        "ocr_baseline_nd":      ocr_all_nd.to_dict(),
        "corrected":            cor_all.to_dict(),
        "corrected_nd":         cor_all_nd.to_dict(),
        "corrected_normal":     cor_norm.to_dict(),
        "corrected_normal_nd":  cor_norm_nd.to_dict(),
        "primary_cer":          round(primary.cer, 6),
        "primary_wer":          round(primary.wer, 6),
        "primary_cer_nd":       round(primary_nd.cer, 6),
        "primary_wer_nd":       round(primary_nd.wer, 6),
        "delta_cer":            round(ocr_all.cer - primary.cer, 6),
        "delta_wer":            round(ocr_all.wer - primary.wer, 6),
    }

    save_json(metrics, out_dir / "metrics.json")
    logger.info(
        "[%s][%s] CER: %.4f -> %.4f (%.4f)  WER: %.4f -> %.4f (%.4f)",
        trial.label, ds_key,
        ocr_all.cer, primary.cer, ocr_all.cer - primary.cer,
        ocr_all.wer, primary.wer, ocr_all.wer - primary.wer,
    )
    return metrics


def run_analyze(
    active_trials: list[TrialConfig],
    active_datasets: list[str],
    config: dict,
    results_dir: Path,
    force: bool,
) -> None:
    for trial in active_trials:
        logger.info("=" * 60)
        logger.info("Analyzing [%s]", trial.label)

        corr_path = trial.corrections_path(results_dir)
        if not corr_path.exists():
            logger.warning(
                "[%s] corrections not found at %s -- skipping.\n"
                "  Download from Kaggle/Colab first.",
                trial.label, corr_path,
            )
            continue

        by_dataset = _load_corrections_by_dataset(corr_path)
        if not by_dataset:
            logger.warning("[%s] No records found — skipping.", trial.label)
            continue

        for ds_key in active_datasets:
            if ds_key not in by_dataset:
                logger.warning("[%s] Dataset '%s' not in corrections.", trial.label, ds_key)
                continue
            metrics_path = trial.output_dir(results_dir) / ds_key / "metrics.json"
            if metrics_path.exists() and not force:
                logger.info("[%s][%s] Already analyzed — skipping (use --force).", trial.label, ds_key)
                continue
            _analyze_trial_dataset(trial, ds_key, by_dataset[ds_key], config, results_dir)

    _write_cross_trial_summary(active_trials, active_datasets, results_dir)


# ---------------------------------------------------------------------------
# SUMMARIZE MODE
# ---------------------------------------------------------------------------


def _write_cross_trial_summary(
    active_trials: list[TrialConfig],
    active_datasets: list[str],
    results_dir: Path,
) -> None:
    rows: list[dict] = []
    for trial in active_trials:
        for ds_key in active_datasets:
            p = trial.output_dir(results_dir) / ds_key / "metrics.json"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "trial":         trial.label,
                "dataset":       ds_key,
                "use_rag":       trial.use_rag,
                "system_prompt": trial.system_prompt_file,
                "n":             m.get("meta", {}).get("num_samples", 0),
                "ocr_cer":       m.get("ocr_baseline", {}).get("cer"),
                "ocr_wer":       m.get("ocr_baseline", {}).get("wer"),
                "corr_cer":      m.get("primary_cer"),
                "corr_wer":      m.get("primary_wer"),
                "corr_cer_nd":   m.get("primary_cer_nd"),
                "corr_wer_nd":   m.get("primary_wer_nd"),
                "delta_cer":     m.get("delta_cer"),
                "delta_wer":     m.get("delta_wer"),
            })

    if not rows:
        logger.warning("No metrics found yet — run --mode analyze first.")
        return

    save_json({"rows": rows}, results_dir / "summary.json")

    def _pct(v: Optional[float]) -> str:
        return f"{v*100:.2f}%" if v is not None else "-"

    def _sign(v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"+{v*100:.2f}%" if v > 0 else f"{v*100:.2f}%"

    header = "| Trial | Dataset | N | OCR CER | Corr CER | dCER | OCR WER | Corr WER | dWER |"
    sep    = "|-------|---------|---|---------|----------|------|---------|----------|------|"
    table_lines = [header, sep]
    for r in rows:
        table_lines.append(
            f"| {r['trial']} | {r['dataset']} | {r['n']} "
            f"| {_pct(r['ocr_cer'])} | {_pct(r['corr_cer'])} | {_sign(r['delta_cer'])} "
            f"| {_pct(r['ocr_wer'])} | {_pct(r['corr_wer'])} | {_sign(r['delta_wer'])} |"
        )

    md = "# Experiment 2 -- Cross-Trial Summary\n\n" + "\n".join(table_lines) + "\n"
    md_path = results_dir / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Summary saved: %s  |  table: %s", results_dir / "summary.json", md_path)

    stdout_utf8 = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
    stdout_utf8.write("\n" + md + "\n")
    stdout_utf8.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    setup_logging(args.results_dir)

    config = load_config(args.config)

    if args.trials:
        invalid = [i for i in args.trials if i not in _TRIAL_BY_ID]
        if invalid:
            logger.error("Unknown trial IDs: %s  (valid: 1-8)", invalid)
            sys.exit(1)
        active_trials = [_TRIAL_BY_ID[i] for i in sorted(set(args.trials))]
    else:
        active_trials = TRIALS

    active_datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    logger.info("=" * 60)
    logger.info("Experiment 2  mode=%s", args.mode)
    logger.info("Trials  : %s", [t.label for t in active_trials])
    logger.info("Datasets: %s", active_datasets)
    logger.info("=" * 60)

    if args.mode == "export":
        run_export(active_trials, active_datasets, config, args.results_dir, args.index_dir, args.limit, args.force)
    elif args.mode == "commands":
        _print_inference_commands(active_trials, args.results_dir)
    elif args.mode == "analyze":
        run_analyze(active_trials, active_datasets, config, args.results_dir, args.force)
    elif args.mode == "summarize":
        _write_cross_trial_summary(active_trials, active_datasets, args.results_dir)


if __name__ == "__main__":
    main()
