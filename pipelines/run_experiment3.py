#!/usr/bin/env python3
"""Experiment 3 — Final Experiment: 4 models × 2 trials × 3 dataset groups × 2 OCR sources.

Models    : qwen3-4b, qwen3-14b, gemma-3-4b, gemma-3-12b
Trials    : T2 (base_rag), T3 (cons_zs)
Datasets  : Validation (13 datasets), BM (Line_Recognition only), Kitab (13 categories)
OCR input : Qaari, Gemma

Generates 12 input JSONL files → 48 inference runs (12 × 4 models).

Usage
-----
    python pipelines/run_experiment3.py --mode export
    python pipelines/run_experiment3.py --mode export --limit 10   # smoke test
    python pipelines/run_experiment3.py --mode commands            # print 48 commands only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.rag_index import RAGIndexBuilder, RAGRetriever
from scripts.infer import EXPERIMENT_MODELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_RESULTS         = _PROJECT_ROOT / "results" / "experiment3"
_INPUTS_DIR      = _RESULTS / "inputs"
_CORRECTIONS_DIR = _RESULTS / "corrections"
_RAG_INDEX_DIR   = _PROJECT_ROOT / "results" / "phase8" / "index"
_CONFIG_PATH     = _PROJECT_ROOT / "configs" / "config.yaml"

# OCR result roots
_QAARI_ROOT = _PROJECT_ROOT / "data" / "ocr-results" / "qaari-results"
_GEMMA_ROOT = _PROJECT_ROOT / "data" / "ocr-results" / "gemma-results"

# GT roots
_GT_ROOT = _PROJECT_ROOT / "data" / "ocr-raw-data"
_KITAB_GT_ROOT = _PROJECT_ROOT / "data" / "kitab-bench"

# System prompt files
_PROMPT_T2 = _PROJECT_ROOT / "configs" / "crafted_system_prompt.txt"       # base RAG
_PROMPT_T3 = _PROJECT_ROOT / "configs" / "crafted_system_prompt_v2.txt"    # conservative zs

# PATS split file
_PATS_SPLITS = _GT_ROOT / "PATS_A01_Dataset" / "pats_splits.json"

# ---------------------------------------------------------------------------
# Trial / model definitions
# ---------------------------------------------------------------------------

TRIALS = {
    "t2": {"name": "t2", "use_rag": True,  "system_prompt": str(_PROMPT_T2)},
    "t3": {"name": "t3", "use_rag": False, "system_prompt": str(_PROMPT_T3)},
}

MODELS = list(EXPERIMENT_MODELS.keys())  # ["qwen3-4b", "qwen3-14b", "gemma-3-4b", "gemma-3-12b"]

# ---------------------------------------------------------------------------
# Validation dataset definitions
# ---------------------------------------------------------------------------

PATS_FONTS = ["Akhbar", "Andalus", "Arial", "Naskh", "Simplified", "Tahoma", "Thuluth", "Traditional"]

# Maps dataset key → (qaari_ocr_dir, gt_dir_or_None)
# For Gemma, a separate mapping is used (gemma_val_path below)
VAL_DATASETS = [
    # PATS — handled via font loop below
    *[f"PATS-A01-{f}-val" for f in PATS_FONTS],
    "KHATT-validation",
    "KHATT-Paragraph-validation",
    "Yarmouk-testing",
    "Muharaf-validation",
    "Historical",
]

# BM Line_Recognition categories
BM_LR_CATS = ["Handwritten", "Manuscripts", "Typewritten"]

# Kitab categories (same order as filesystem)
KITAB_CATS = [
    "adab", "arabicocr", "evarest", "hindawi", "historicalbooks",
    "historyar", "isippt", "khatt", "khattparagraph", "muharaf",
    "onlinekhatt", "patsocr", "synthesizear",
]

# ---------------------------------------------------------------------------
# Helpers — file loading
# ---------------------------------------------------------------------------


_ARABIC_RE = re.compile(r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]')


def _clean_gemma_ocr(text: str) -> str:
    """Strip Gemma VLM preamble and return only the Arabic OCR content.

    Gemma outputs preambles like:
        "Here's the plain text representation of the document:\n\nARABIC..."
    We extract everything from the first Arabic character onward.
    """
    m = _ARABIC_RE.search(text)
    if m:
        arabic_part = text[m.start():]
        # Strip surrounding quotes that Gemma sometimes adds
        arabic_part = arabic_part.strip('"').strip()
        return arabic_part
    # No Arabic found — return original (may be empty/error output)
    return text.strip()


def _read_text(path: Path, clean_gemma: bool = False) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        return _clean_gemma_ocr(raw) if clean_gemma else raw
    except Exception:
        return ""


def _pats_val_stems(font: str) -> set[str]:
    """Return the set of validation-split stems for a PATS font."""
    with open(_PATS_SPLITS, encoding="utf-8") as f:
        splits = json.load(f)["splits"]
    return set(splits[font]["validation"])


# ---------------------------------------------------------------------------
# Helpers — Gemma OCR path resolution
# ---------------------------------------------------------------------------

_GEMMA_VAL_ROOT = _GEMMA_ROOT / "validation-input-data"


def _gemma_val_path(dataset_key: str, sample_id: str) -> Optional[Path]:
    """Return the Gemma OCR file path for a validation sample, or None."""
    if dataset_key.startswith("PATS-A01-"):
        # e.g. "PATS-A01-Akhbar-val" -> pats/akhbar/{sample_id}.txt
        font = dataset_key.replace("PATS-A01-", "").replace("-val", "").lower()
        return _GEMMA_VAL_ROOT / "pats" / font / f"{sample_id}.txt"
    if dataset_key == "KHATT-validation":
        return _GEMMA_VAL_ROOT / "khatt" / f"{sample_id}.txt"
    if dataset_key == "KHATT-Paragraph-validation":
        return _GEMMA_VAL_ROOT / "khatt-paragraph" / f"{sample_id}.txt"
    if dataset_key == "Muharaf-validation":
        return _GEMMA_VAL_ROOT / "muharaf" / f"{sample_id}.txt"
    if dataset_key == "Historical":
        # sample_id like "Book1_00000126_A" → book subfolder "book1"
        book_folder = sample_id[:5].lower()  # "book1"
        return _GEMMA_VAL_ROOT / "historical" / book_folder / f"{sample_id}.txt"
    if dataset_key == "Yarmouk-testing":
        # Gemma uses {sample_id}_p01.txt
        return _GEMMA_VAL_ROOT / "yarmouk-test-200" / "yarmouk" / f"{sample_id}_p01.txt"
    return None


# ---------------------------------------------------------------------------
# Helpers — record builders
# ---------------------------------------------------------------------------


def _zs_record(
    sample_id: str,
    dataset: str,
    ocr_source: str,
    ocr_text: str,
    gt_text: str,
) -> dict:
    return {
        "sample_id":      sample_id,
        "dataset":        dataset,
        "ocr_source":     ocr_source,
        "ocr_text":       ocr_text,
        "gt_text":        gt_text,
        "prompt_type":    "zero_shot",
        "prompt_version": "crafted",
    }


_RAG_QUERY_MAX_CHARS = 400  # cap BM25 query length to keep retrieval fast on long texts


def _rag_record(
    sample_id: str,
    dataset: str,
    ocr_source: str,
    ocr_text: str,
    gt_text: str,
    retriever: RAGRetriever,
    top_k_s: int,
    top_k_w: int,
    top_k_pw: int,
) -> dict:
    query  = ocr_text[:_RAG_QUERY_MAX_CHARS]  # truncate only for BM25 query
    sent   = retriever.retrieve_sentences(query, top_k=top_k_s)
    words  = retriever.retrieve_words(query, top_k=top_k_w, top_k_per_word=top_k_pw)
    r_sent = retriever.format_sentences_for_prompt(sent)
    r_word = retriever.format_words_for_prompt(words)
    has_ctx = bool(r_sent.strip()) or bool(r_word.strip())
    return {
        "sample_id":           sample_id,
        "dataset":             dataset,
        "ocr_source":          ocr_source,
        "ocr_text":            ocr_text,
        "gt_text":             gt_text,
        "prompt_type":         "rag" if has_ctx else "zero_shot",
        "prompt_version":      "crafted",
        "retrieved_sentences": r_sent,
        "retrieved_words":     r_word,
        "retrieval_mode":      "bm25",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


# ---------------------------------------------------------------------------
# RAG loader
# ---------------------------------------------------------------------------


def _load_retriever(config: dict) -> RAGRetriever:
    if not (_RAG_INDEX_DIR / "index_meta.json").exists():
        raise FileNotFoundError(
            f"RAG index not found at {_RAG_INDEX_DIR}.\n"
            "Build it first: python pipelines/run_phase8.py --mode build-index"
        )
    ph8_cfg = config.get("phase8", {})
    builder = RAGIndexBuilder(config)
    builder.load(_RAG_INDEX_DIR)
    logger.info("RAG index loaded: %d sentence entries.", len(builder.sentence_store))
    return RAGRetriever(
        index=builder,
        mode=ph8_cfg.get("retrieval_mode", "bm25"),
        alpha=ph8_cfg.get("alpha", 0.6),
        top_k_candidates=ph8_cfg.get("top_k_candidates", 50),
    )


# ---------------------------------------------------------------------------
# EXPORT: Validation set (Qaari)
# ---------------------------------------------------------------------------


def _export_val_qaari(
    limit: Optional[int],
    retriever: Optional[RAGRetriever],
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """Return (t3_records, t2_records) for validation / Qaari OCR."""
    # Reuse experiment2 inputs as the authoritative source for Qaari val.
    # They already have correct splits, sample_ids, gt_text, and RAG context.
    exp2_inputs = _PROJECT_ROOT / "results" / "experiment2" / "inputs"
    zs_src = exp2_inputs / "trial3_cons_zs.jsonl"
    rag_src = exp2_inputs / "trial2_base_rag.jsonl"

    t3_records: list[dict] = []
    t2_records: list[dict] = []

    counts: dict[str, int] = {}

    with open(zs_src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            ds = r["dataset"]
            if limit and counts.get(ds, 0) >= limit:
                continue
            counts[ds] = counts.get(ds, 0) + 1
            rec = _zs_record(r["sample_id"], ds, "qaari", r["ocr_text"], r.get("gt_text", ""))
            t3_records.append(rec)

    # T2: rebuild from zs records so RAG context uses correct OCR text,
    # but if experiment2 RAG source exists, just port it (faster).
    counts2: dict[str, int] = {}
    if rag_src.exists():
        with open(rag_src, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                ds = r["dataset"]
                if limit and counts2.get(ds, 0) >= limit:
                    continue
                counts2[ds] = counts2.get(ds, 0) + 1
                rec = {
                    "sample_id":           r["sample_id"],
                    "dataset":             ds,
                    "ocr_source":          "qaari",
                    "ocr_text":            r["ocr_text"],
                    "gt_text":             r.get("gt_text", ""),
                    "prompt_type":         r.get("prompt_type", "rag"),
                    "prompt_version":      "crafted",
                    "retrieved_sentences": r.get("retrieved_sentences", ""),
                    "retrieved_words":     r.get("retrieved_words", ""),
                    "retrieval_mode":      r.get("retrieval_mode", "bm25"),
                }
                t2_records.append(rec)
    else:
        logger.warning("Experiment2 RAG source not found; rebuilding RAG for val/qaari.")
        ph8 = config.get("phase8", {})
        for rec_zs in tqdm(t3_records, desc="  RAG val/qaari"):
            t2_records.append(_rag_record(
                rec_zs["sample_id"], rec_zs["dataset"], "qaari",
                rec_zs["ocr_text"], rec_zs["gt_text"], retriever,
                ph8.get("top_k_sentences", 5), ph8.get("top_k_words", 15),
                ph8.get("word_top_k_per_input_word", 3),
            ))

    return t3_records, t2_records


# ---------------------------------------------------------------------------
# EXPORT: Validation set (Gemma)
# ---------------------------------------------------------------------------


def _export_val_gemma(
    t3_qaari: list[dict],
    retriever: RAGRetriever,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """Build val/gemma records by swapping OCR text from Gemma files."""
    ph8 = config.get("phase8", {})
    t3_records: list[dict] = []
    t2_records: list[dict] = []
    skipped = 0

    for rec in tqdm(t3_qaari, desc="  Gemma val OCR"):
        gpath = _gemma_val_path(rec["dataset"], rec["sample_id"])
        if gpath is None or not gpath.exists():
            skipped += 1
            continue
        ocr_gemma = _read_text(gpath, clean_gemma=True)
        if not ocr_gemma:
            skipped += 1
            continue

        t3_records.append(_zs_record(
            rec["sample_id"], rec["dataset"], "gemma",
            ocr_gemma, rec["gt_text"],
        ))
        t2_records.append(_rag_record(
            rec["sample_id"], rec["dataset"], "gemma",
            ocr_gemma, rec["gt_text"], retriever,
            ph8.get("top_k_sentences", 5), ph8.get("top_k_words", 15),
            ph8.get("word_top_k_per_input_word", 3),
        ))

    if skipped:
        logger.warning("val/gemma: skipped %d samples with missing/empty Gemma OCR.", skipped)
    return t3_records, t2_records


# ---------------------------------------------------------------------------
# EXPORT: BM dataset (Line_Recognition only)
# ---------------------------------------------------------------------------


def _iter_bm_samples(
    ocr_root: Path,
    gt_root: Path,
    ocr_source: str,
    limit: Optional[int],
) -> Iterator[tuple[str, str, str, str]]:
    """Yield (sample_id, dataset, ocr_text, gt_text) for BM Line_Recognition."""
    is_gemma = ocr_source == "gemma"
    for cat in BM_LR_CATS:
        dataset  = f"BM-{cat}"
        ocr_dir  = ocr_root / "BM" / "raw" / "Line_Recognition" / cat
        gt_dir   = gt_root  / "BM" / "raw" / "Line_Recognition" / cat
        if not ocr_dir.exists():
            logger.warning("BM OCR dir not found: %s", ocr_dir)
            continue
        stems = sorted(f.stem for f in ocr_dir.glob("*.txt"))
        count = 0
        for stem in stems:
            if limit and count >= limit:
                break
            ocr_path = ocr_dir / f"{stem}.txt"
            gt_path  = gt_dir  / f"{stem}.txt"
            if not gt_path.exists():
                continue
            ocr_text = _read_text(ocr_path, clean_gemma=is_gemma)
            gt_text  = _read_text(gt_path)
            if not ocr_text or not gt_text:
                continue
            sample_id = f"BM-{cat}_{stem}"
            yield sample_id, dataset, ocr_text, gt_text
            count += 1


def _export_bm(
    ocr_source: str,
    retriever: RAGRetriever,
    config: dict,
    limit: Optional[int],
) -> tuple[list[dict], list[dict]]:
    ph8 = config.get("phase8", {})
    ocr_root = _QAARI_ROOT if ocr_source == "qaari" else _GEMMA_ROOT
    gt_root  = _GT_ROOT

    t3_records: list[dict] = []
    t2_records: list[dict] = []

    for sid, ds, ocr, gt in tqdm(
        list(_iter_bm_samples(ocr_root, gt_root, ocr_source, limit)),
        desc=f"  RAG bm/{ocr_source}",
    ):
        t3_records.append(_zs_record(sid, ds, ocr_source, ocr, gt))
        t2_records.append(_rag_record(
            sid, ds, ocr_source, ocr, gt, retriever,
            ph8.get("top_k_sentences", 5), ph8.get("top_k_words", 15),
            ph8.get("word_top_k_per_input_word", 3),
        ))

    return t3_records, t2_records


# ---------------------------------------------------------------------------
# EXPORT: Kitab dataset
# ---------------------------------------------------------------------------


def _iter_kitab_samples(
    ocr_root: Path,
    gt_root: Path,
    ocr_source: str,
    limit: Optional[int],
) -> Iterator[tuple[str, str, str, str]]:
    """Yield (sample_id, dataset, ocr_text, gt_text) for all Kitab categories."""
    is_gemma = ocr_source == "gemma"
    for cat in KITAB_CATS:
        dataset  = f"kitab-{cat}"
        if ocr_source == "qaari":
            ocr_dir = ocr_root / "kitab-results" / cat / "images"
        else:
            ocr_dir = ocr_root / "kitab-bench-images" / cat / "images"
        gt_dir = gt_root / cat / "gt"

        if not ocr_dir.exists():
            logger.warning("Kitab OCR dir not found: %s", ocr_dir)
            continue
        if not gt_dir.exists():
            logger.warning("Kitab GT dir not found: %s", gt_dir)
            continue

        # Only process stems that exist in BOTH ocr and gt
        ocr_stems = {f.stem for f in ocr_dir.glob("*.txt")}
        gt_stems  = {f.stem for f in gt_dir.glob("*.txt")}
        common    = sorted(ocr_stems & gt_stems, key=lambda s: int(s) if s.isdigit() else s)

        count = 0
        for stem in common:
            if limit and count >= limit:
                break
            ocr_text = _read_text(ocr_dir / f"{stem}.txt", clean_gemma=is_gemma)
            gt_text  = _read_text(gt_dir  / f"{stem}.txt")
            if not ocr_text or not gt_text:
                continue
            sample_id = f"kitab-{cat}_{stem}"
            yield sample_id, dataset, ocr_text, gt_text
            count += 1


def _export_kitab(
    ocr_source: str,
    retriever: RAGRetriever,
    config: dict,
    limit: Optional[int],
) -> tuple[list[dict], list[dict]]:
    ph8 = config.get("phase8", {})
    ocr_root = _QAARI_ROOT if ocr_source == "qaari" else _GEMMA_ROOT
    gt_root  = _KITAB_GT_ROOT

    t3_records: list[dict] = []
    t2_records: list[dict] = []

    for sid, ds, ocr, gt in tqdm(
        list(_iter_kitab_samples(ocr_root, gt_root, ocr_source, limit)),
        desc=f"  RAG kitab/{ocr_source}",
    ):
        t3_records.append(_zs_record(sid, ds, ocr_source, ocr, gt))
        t2_records.append(_rag_record(
            sid, ds, ocr_source, ocr, gt, retriever,
            ph8.get("top_k_sentences", 5), ph8.get("top_k_words", 15),
            ph8.get("word_top_k_per_input_word", 3),
        ))

    return t3_records, t2_records


# ---------------------------------------------------------------------------
# Print 48 inference commands
# ---------------------------------------------------------------------------


def _print_commands(results_dir: Path) -> None:
    import io as _io
    _out = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    inp = results_dir / "inputs"
    out = results_dir / "corrections"

    groups = [
        ("val",   "qaari", _PROMPT_T2, _PROMPT_T3),
        ("val",   "gemma", _PROMPT_T2, _PROMPT_T3),
        ("bm",    "qaari", _PROMPT_T2, _PROMPT_T3),
        ("bm",    "gemma", _PROMPT_T2, _PROMPT_T3),
        ("kitab", "qaari", _PROMPT_T2, _PROMPT_T3),
        ("kitab", "gemma", _PROMPT_T2, _PROMPT_T3),
    ]

    def _p(*args): _out.write(" ".join(str(a) for a in args) + "\n"); _out.flush()

    _p()
    _p("=" * 80)
    _p("EXPERIMENT 3 - 48 INFERENCE COMMANDS")
    _p("Run each on the remote machine (Kaggle A100 / Thunder).")
    _p("=" * 80)

    run_no = 1
    for group, ocr_src, prompt_t2, prompt_t3 in groups:
        _p(f"\n# -- {group.upper()} / {ocr_src.upper()} " + "-" * 40)
        for trial, prompt in [("t2", prompt_t2), ("t3", prompt_t3)]:
            input_file  = inp / f"{group}_{ocr_src}_{trial}.jsonl"
            for model in MODELS:
                output_file = out / f"{model}_{group}_{ocr_src}_{trial}.jsonl"
                _p(f"\n# Run {run_no:02d}: {model} | {group}/{ocr_src} | {trial}")
                _p(
                    f"python scripts/infer.py \\\n"
                    f"    --input  {input_file} \\\n"
                    f"    --output {output_file} \\\n"
                    f"    --system-prompt {prompt} \\\n"
                    f"    --experiment-model {model} \\\n"
                    f"    --batch-size 32 \\\n"
                    f"    --hf-repo YOUR_HF_REPO --hf-token $HF_TOKEN"
                )
                run_no += 1

    _p()
    _p("=" * 80)
    _p(f"Total runs: {run_no - 1}")
    _p("=" * 80)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def run_export(config: dict, limit: Optional[int], force: bool) -> None:
    _INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load RAG index once (used for all T2 files)
    logger.info("Loading RAG index...")
    retriever = _load_retriever(config)

    ph8 = config.get("phase8", {})

    # ------------------------------------------------------------------ #
    #  1. Validation / Qaari
    # ------------------------------------------------------------------ #
    logger.info("=== [1/6] Validation / Qaari ===")
    t3_path = _INPUTS_DIR / "val_qaari_t3.jsonl"
    t2_path = _INPUTS_DIR / "val_qaari_t2.jsonl"
    if not force and t3_path.exists() and t2_path.exists():
        logger.info("  Already exists — skipping (use --force to regenerate).")
        with open(t3_path, encoding="utf-8") as f:
            val_qaari_t3 = [json.loads(l) for l in f if l.strip()]
    else:
        val_qaari_t3, val_qaari_t2 = _export_val_qaari(limit, retriever, config)
        _write_jsonl(t3_path, val_qaari_t3)
        _write_jsonl(t2_path, val_qaari_t2)

    # ------------------------------------------------------------------ #
    #  2. Validation / Gemma
    # ------------------------------------------------------------------ #
    logger.info("=== [2/6] Validation / Gemma ===")
    t3_path = _INPUTS_DIR / "val_gemma_t3.jsonl"
    t2_path = _INPUTS_DIR / "val_gemma_t2.jsonl"
    if not force and t3_path.exists() and t2_path.exists():
        logger.info("  Already exists — skipping.")
    else:
        t3, t2 = _export_val_gemma(val_qaari_t3, retriever, config)
        _write_jsonl(t3_path, t3)
        _write_jsonl(t2_path, t2)

    # ------------------------------------------------------------------ #
    #  3. BM / Qaari
    # ------------------------------------------------------------------ #
    logger.info("=== [3/6] BM / Qaari ===")
    t3_path = _INPUTS_DIR / "bm_qaari_t3.jsonl"
    t2_path = _INPUTS_DIR / "bm_qaari_t2.jsonl"
    if not force and t3_path.exists() and t2_path.exists():
        logger.info("  Already exists — skipping.")
    else:
        t3, t2 = _export_bm("qaari", retriever, config, limit)
        _write_jsonl(t3_path, t3)
        _write_jsonl(t2_path, t2)

    # ------------------------------------------------------------------ #
    #  4. BM / Gemma
    # ------------------------------------------------------------------ #
    logger.info("=== [4/6] BM / Gemma ===")
    t3_path = _INPUTS_DIR / "bm_gemma_t3.jsonl"
    t2_path = _INPUTS_DIR / "bm_gemma_t2.jsonl"
    if not force and t3_path.exists() and t2_path.exists():
        logger.info("  Already exists — skipping.")
    else:
        t3, t2 = _export_bm("gemma", retriever, config, limit)
        _write_jsonl(t3_path, t3)
        _write_jsonl(t2_path, t2)

    # ------------------------------------------------------------------ #
    #  5. Kitab / Qaari
    # ------------------------------------------------------------------ #
    logger.info("=== [5/6] Kitab / Qaari ===")
    t3_path = _INPUTS_DIR / "kitab_qaari_t3.jsonl"
    t2_path = _INPUTS_DIR / "kitab_qaari_t2.jsonl"
    if not force and t3_path.exists() and t2_path.exists():
        logger.info("  Already exists — skipping.")
    else:
        t3, t2 = _export_kitab("qaari", retriever, config, limit)
        _write_jsonl(t3_path, t3)
        _write_jsonl(t2_path, t2)

    # ------------------------------------------------------------------ #
    #  6. Kitab / Gemma
    # ------------------------------------------------------------------ #
    logger.info("=== [6/6] Kitab / Gemma ===")
    t3_path = _INPUTS_DIR / "kitab_gemma_t3.jsonl"
    t2_path = _INPUTS_DIR / "kitab_gemma_t2.jsonl"
    if not force and t3_path.exists() and t2_path.exists():
        logger.info("  Already exists — skipping.")
    else:
        t3, t2 = _export_kitab("gemma", retriever, config, limit)
        _write_jsonl(t3_path, t3)
        _write_jsonl(t2_path, t2)

    logger.info("=== Export complete ===")
    _print_commands(_RESULTS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3 export: generates 12 input JSONL files + prints 48 inference commands."
    )
    parser.add_argument(
        "--mode", required=True, choices=["export", "commands"],
        help="export: build all JSONL files; commands: print 48 inference commands only.",
    )
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per dataset/category (smoke testing only).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if output files already exist.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(
                            open(sys.stdout.fileno(), "w", encoding="utf-8",
                                 errors="replace", closefd=False, buffering=1)
                        )])


def main() -> None:
    args = parse_args()
    setup_logging()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.mode == "commands":
        _print_commands(_RESULTS)
    elif args.mode == "export":
        run_export(config, args.limit, args.force)


if __name__ == "__main__":
    main()
