#!/usr/bin/env python3
"""Analyze Experiment 3 corrections — model comparison on a fixed (group, ocr, trial).

Computes, per model and per dataset:
  - OCR baseline CER/WER (from ocr_text; identical across models)
  - Corrected CER/WER (from corrected_text)
  - Corrected CER/WER after runaway correction (corrected_text_fixed)
All metrics: normal samples only (OCR/GT length ratio <= threshold), diacritics stripped.

Usage:
    python scripts/analyze_experiment3.py
    python scripts/analyze_experiment3.py --group val --ocr qaari --trial t2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.analysis.metrics import calculate_cer, calculate_wer
from src.data.text_utils import normalise_arabic
from src.analysis.runaway_corrector import fix_text

_CORR_DIR = _ROOT / "results" / "experiment3" / "corrections"
_RUNAWAY_THRESHOLD = 5.0   # OCR/GT length ratio for "normal" classification
_RUNAWAY_FIX_RATIO = 3.0   # corrected/GT ratio above which runaway-fix kicks in

MODELS = ["qwen3-4b", "qwen3-14b", "gemma-3-4b", "gemma-3-12b"]

# Dataset groupings for aggregation
PATS = [
    "PATS-A01-Akhbar-val", "PATS-A01-Andalus-val", "PATS-A01-Arial-val",
    "PATS-A01-Naskh-val", "PATS-A01-Simplified-val", "PATS-A01-Tahoma-val",
    "PATS-A01-Thuluth-val", "PATS-A01-Traditional-val",
]
OTHER = ["KHATT-validation", "KHATT-Paragraph-validation",
         "Yarmouk-testing", "Muharaf-validation", "Historical"]
ALL_DATASETS = PATS + OTHER


def _is_normal(ocr: str, gt: str) -> bool:
    gt_n = normalise_arabic(gt)
    if len(gt_n) == 0:
        return False
    return len(normalise_arabic(ocr)) / len(gt_n) <= _RUNAWAY_THRESHOLD


def _load(path: Path) -> dict[str, list[dict]]:
    by_ds: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_ds.setdefault(r["dataset"], []).append(r)
    return by_ds


def _mean_cer_wer(records: list[dict], field: str, runaway_fix: bool):
    """Mean CER/WER (no diacritics) over normal samples using `field` as hypothesis."""
    cers, wers = [], []
    n_normal = 0
    n_fixed = 0
    for r in records:
        ocr, gt = r["ocr_text"], r["gt_text"]
        if not _is_normal(ocr, gt):
            continue
        n_normal += 1
        hyp = r.get(field, r["ocr_text"])
        if runaway_fix:
            new = fix_text(hyp, gt_text=gt, ocr_text=ocr, ratio_threshold=_RUNAWAY_FIX_RATIO)
            if new != hyp:
                n_fixed += 1
            hyp = new
        cers.append(calculate_cer(gt, hyp, strip_diacritics=True))
        wers.append(calculate_wer(gt, hyp, strip_diacritics=True))
    cer = statistics.mean(cers) if cers else 0.0
    wer = statistics.mean(wers) if wers else 0.0
    return cer, wer, n_normal, n_fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="val")
    ap.add_argument("--ocr", default="qaari")
    ap.add_argument("--trial", default="t2")
    args = ap.parse_args()

    suffix = f"{args.group}_{args.ocr}_{args.trial}"

    # Load each model's corrections
    model_data: dict[str, dict[str, list[dict]]] = {}
    for m in MODELS:
        p = _CORR_DIR / f"{m}_{suffix}.jsonl"
        if p.exists():
            model_data[m] = _load(p)
        else:
            print(f"[skip] {p.name} not found")
    if not model_data:
        print("No correction files found.")
        return

    present = list(model_data.keys())
    ref = model_data[present[0]]
    datasets = [d for d in ALL_DATASETS if d in ref]

    # ---- OCR baseline (model-independent) ----
    ocr_base = {}
    for d in datasets:
        cer, wer, n, _ = _mean_cer_wer(ref[d], "ocr_text", runaway_fix=False)
        ocr_base[d] = (cer, wer, n)

    print("=" * 100)
    print(f"EXPERIMENT 3 — Model comparison on {suffix}")
    print(f"Metric: mean CER / WER, normal samples only (ratio<= {_RUNAWAY_THRESHOLD}), diacritics stripped")
    print("=" * 100)

    # ---- Per-dataset table (corrected, no runaway fix) ----
    header = f"{'Dataset':<28} {'N':>4} {'OCR_CER':>8}"
    for m in present:
        header += f" {m[:11]:>12}"
    print("\n--- Corrected CER per dataset (raw corrected_text) ---")
    print(header)
    print("-" * len(header))
    for d in datasets:
        cer_o, _, n = ocr_base[d]
        row = f"{d:<28} {n:>4} {cer_o*100:>7.2f}%"
        for m in present:
            cer, _, _, _ = _mean_cer_wer(model_data[m][d], "corrected_text", runaway_fix=False)
            row += f" {cer*100:>11.2f}%"
        print(row)

    # ---- Per-dataset table (corrected + runaway fix) ----
    print("\n--- Corrected CER per dataset (after runaway correction) ---")
    print(header)
    print("-" * len(header))
    for d in datasets:
        cer_o, _, n = ocr_base[d]
        row = f"{d:<28} {n:>4} {cer_o*100:>7.2f}%"
        for m in present:
            cer, _, _, _ = _mean_cer_wer(model_data[m][d], "corrected_text", runaway_fix=True)
            row += f" {cer*100:>11.2f}%"
        print(row)

    # ---- Aggregates ----
    def agg(model: str, ds_list: list[str], field: str, fix: bool):
        cs, ws = [], []
        for d in ds_list:
            if d not in model_data[model]:
                continue
            c, w, _, _ = _mean_cer_wer(model_data[model][d], field, runaway_fix=fix)
            cs.append(c); ws.append(w)
        return (statistics.mean(cs) if cs else 0.0,
                statistics.mean(ws) if ws else 0.0)

    def ocr_agg(ds_list):
        cs, ws = [], []
        for d in ds_list:
            c, w, _ = ocr_base[d]
            cs.append(c); ws.append(w)
        return statistics.mean(cs), statistics.mean(ws)

    print("\n" + "=" * 100)
    print("AGGREGATES (mean over datasets in group)")
    print("=" * 100)
    for label, ds_list in [("PATS-A01 (8 fonts)", [d for d in PATS if d in datasets]),
                           ("KHATT-validation", ["KHATT-validation"]),
                           ("New full-page (KP/Yarmouk/Muharaf/Hist)",
                            [d for d in ["KHATT-Paragraph-validation","Yarmouk-testing","Muharaf-validation","Historical"] if d in datasets])]:
        if not ds_list:
            continue
        oc, ow = ocr_agg(ds_list)
        print(f"\n### {label}   (OCR baseline: CER={oc*100:.2f}%  WER={ow*100:.2f}%)")
        print(f"  {'Model':<14} {'CER':>8} {'WER':>8}   |  {'CER(fix)':>9} {'WER(fix)':>9}")
        print("  " + "-" * 58)
        for m in present:
            c, w = agg(m, ds_list, "corrected_text", False)
            cf, wf = agg(m, ds_list, "corrected_text", True)
            print(f"  {m:<14} {c*100:>7.2f}% {w*100:>7.2f}%   |  {cf*100:>8.2f}% {wf*100:>8.2f}%")

    print()


if __name__ == "__main__":
    main()
