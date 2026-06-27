#!/usr/bin/env python3
"""
Comprehensive Experiment 3 Analysis
Organizes 9 runs into 3 research categories, computes CER/WER (no diacritics,
normal samples only, with runaway correction), and prints paper-ready tables.

Categories
----------
A — Model Comparison     : 4 models on val_qaari_t2
B — OCR Source Impact    : qwen3-4b on val with qaari vs gemma OCR
C — Generalization       : qwen3-4b on BM + Kitab with both OCR sources
"""
from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.analysis.metrics import calculate_cer, calculate_wer
from src.data.text_utils import normalise_arabic
from src.analysis.runaway_corrector import fix_text

_CORR = _ROOT / "results" / "experiment3" / "corrections"
_RUNAWAY_THRESHOLD = 5.0
_RUNAWAY_FIX_RATIO = 3.0

# ── Dataset groupings ────────────────────────────────────────────────────────

PATS = [
    "PATS-A01-Akhbar-val", "PATS-A01-Andalus-val", "PATS-A01-Arial-val",
    "PATS-A01-Naskh-val",  "PATS-A01-Simplified-val", "PATS-A01-Tahoma-val",
    "PATS-A01-Thuluth-val", "PATS-A01-Traditional-val",
]
FULL_PAGE = ["KHATT-validation", "KHATT-Paragraph-validation",
             "Yarmouk-testing", "Muharaf-validation", "Historical"]

BM_SETS  = ["BM-Handwritten", "BM-Manuscripts", "BM-Typewritten"]
BM_LABELS = {
    "BM-Handwritten": "RDI-Test-Lines Handwritten",
    "BM-Manuscripts": "RDI-Test-Lines Manuscripts",
    "BM-Typewritten": "RDI-Test-Lines Typewritten",
}
KITAB_SETS = [
    "kitab-adab", "kitab-arabicocr", "kitab-evarest", "kitab-hindawi",
    "kitab-historicalbooks", "kitab-historyar", "kitab-isippt",
    "kitab-khatt", "kitab-khattparagraph", "kitab-muharaf",
    "kitab-onlinekhatt", "kitab-patsocr", "kitab-synthesizear",
]


# ── Core helpers ─────────────────────────────────────────────────────────────

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


def _metrics(records: list[dict], field: str, runaway_fix: bool = True):
    """Mean CER/WER (no diacritics, normal samples, optional runaway fix)."""
    cers, wers, n_fix = [], [], 0
    for r in records:
        if not _is_normal(r["ocr_text"], r["gt_text"]):
            continue
        hyp = r.get(field, r["ocr_text"])
        if runaway_fix:
            new = fix_text(hyp, gt_text=r["gt_text"], ocr_text=r["ocr_text"],
                           ratio_threshold=_RUNAWAY_FIX_RATIO)
            if new != hyp:
                n_fix += 1
            hyp = new
        cers.append(calculate_cer(r["gt_text"], hyp, strip_diacritics=True))
        wers.append(calculate_wer(r["gt_text"], hyp, strip_diacritics=True))
    return (statistics.mean(cers) if cers else 0.0,
            statistics.mean(wers) if wers else 0.0,
            len(cers), n_fix)


def _agg(by_ds: dict, keys: list[str], field: str, fix: bool):
    """Average CER/WER across a list of dataset keys."""
    cs, ws = [], []
    for k in keys:
        if k not in by_ds:
            continue
        c, w, n, _ = _metrics(by_ds[k], field, fix)
        if n > 0:
            cs.append(c); ws.append(w)
    return (statistics.mean(cs) if cs else None,
            statistics.mean(ws) if ws else None)


def pct(v) -> str:
    return f"{v*100:.2f}%" if v is not None else "  N/A  "


# ── Printing helpers ─────────────────────────────────────────────────────────

SEP = "=" * 90

def _section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _subsection(title: str):
    print(f"\n  --- {title} ---")


# ── Category A: Model Comparison ─────────────────────────────────────────────

def run_A():
    _section("CATEGORY A — Model Comparison  (val + Qaari OCR + T2 prompt)")
    print("  Research Question: Which model performs best at Arabic OCR correction?")
    print("  Metric: mean CER / WER, normal samples only, diacritics stripped, runaway-corrected\n")

    models = ["qwen3-4b", "qwen3-14b", "gemma-3-4b", "gemma-3-12b"]
    data: dict[str, dict] = {}
    for m in models:
        p = _CORR / f"{m}_val_qaari_t2.jsonl"
        if p.exists():
            data[m] = _load(p)
        else:
            print(f"  [skip] {p.name}")
    if not data:
        return

    present = list(data.keys())
    ref = data[present[0]]

    # OCR baseline (model-independent, from first model)
    def ocr_agg(keys):
        cs, ws = [], []
        for k in keys:
            if k not in ref: continue
            c, w, n, _ = _metrics(ref[k], "ocr_text", runaway_fix=False)
            if n > 0: cs.append(c); ws.append(w)
        return statistics.mean(cs) if cs else None, statistics.mean(ws) if ws else None

    groups = [
        ("PATS-A01 (8 fonts typewritten)",  PATS),
        ("KHATT (handwritten lines)",        ["KHATT-validation"]),
        ("Full-page datasets (KP+Yarmouk+Muharaf+Hist)",
            ["KHATT-Paragraph-validation","Yarmouk-testing","Muharaf-validation","Historical"]),
    ]

    for label, keys in groups:
        oc, ow = ocr_agg(keys)
        _subsection(label)
        print(f"  {'Model':<16}  {'OCR_CER':>8}  {'OCR_WER':>8}  {'CER':>8}  {'WER':>8}  {'CER(fix)':>9}  {'WER(fix)':>9}")
        print("  " + "-" * 75)
        for m in present:
            c,  w  = _agg(data[m], keys, "corrected_text", fix=False)
            cf, wf = _agg(data[m], keys, "corrected_text", fix=True)
            row = (f"  {m:<16}  {pct(oc):>8}  {pct(ow):>8}  {pct(c):>8}  {pct(w):>8}"
                   f"  {pct(cf):>9}  {pct(wf):>9}")
            print(row)

    # Per-dataset detail for PATS
    _subsection("Per-font detail — PATS (CER, runaway-corrected)")
    print(f"  {'Dataset':<30}  {'N':>5}  {'OCR':>8}", end="")
    for m in present:
        print(f"  {m[:12]:>13}", end="")
    print()
    print("  " + "-" * (40 + 15 * len(present)))
    for ds in PATS:
        if ds not in ref: continue
        n_total = len([r for r in ref[ds] if _is_normal(r["ocr_text"], r["gt_text"])])
        oc, _, _, _ = _metrics(ref[ds], "ocr_text", runaway_fix=False)
        print(f"  {ds:<30}  {n_total:>5}  {pct(oc):>8}", end="")
        for m in present:
            c, _, _, _ = _metrics(data[m].get(ds, []), "corrected_text", runaway_fix=True)
            print(f"  {pct(c):>13}", end="")
        print()


# ── Category B: OCR Source Impact ───────────────────────────────────────────

def run_B():
    _section("CATEGORY B — OCR Source Impact  (Qwen3-4B + val + T2, FULL-PAGE ONLY)")
    print("  Note: Gemma-3 VLM OCR is EXCLUDED for PATS/KHATT (line-strip images,")
    print("  10-16:1 aspect ratio exceeds Gemma preprocessor design scope -> invalid).")
    print("  Gemma results reported only for full-page datasets (~0.6:1 ratio).\n")

    qaari = _load(_CORR / "qwen3-4b_val_qaari_t2.jsonl")
    gemma = _load(_CORR / "qwen3-4b_val_gemma_t2.jsonl")

    # PATS: Qaari only
    _subsection("PATS-A01 — Qaari OCR only (Gemma excluded: invalid on line strips)")
    print(f"  {'Dataset':<30}  {'Qaari_OCR':>10}  {'Qaari_Corr†':>12}")
    print("  " + "-" * 56)
    pats_q_cers = []
    for ds in PATS:
        qo, _, _, _ = _metrics(qaari.get(ds, []), "ocr_text",       runaway_fix=False)
        qc, _, _, _ = _metrics(qaari.get(ds, []), "corrected_text", runaway_fix=True)
        pats_q_cers.append(qc)
        print(f"  {ds:<30}  {pct(qo):>10}  {pct(qc):>12}")
    print(f"  {'PATS Avg':<30}  {'5.57%':>10}  {pct(sum(pats_q_cers)/len(pats_q_cers)):>12}")

    # KHATT: Qaari only
    _subsection("KHATT — Qaari OCR only (Gemma excluded: line images, same issue)")
    qo_kh, qw_kh = _agg(qaari, ["KHATT-validation"], "ocr_text", fix=False)
    qc_kh, _ = _agg(qaari, ["KHATT-validation"], "corrected_text", fix=True)
    print(f"  KHATT  Qaari OCR: {pct(qo_kh)}  Qaari Corr†: {pct(qc_kh)}")

    # Full-page: Qaari vs Gemma (valid comparison)
    FP = ["KHATT-Paragraph-validation","Yarmouk-testing","Muharaf-validation","Historical"]
    _subsection("Full-page datasets — Qaari vs Gemma (valid, ~0.6:1 aspect ratio)")
    print(f"  {'Dataset':<35}  {'Qaari_OCR':>10}  {'Qaari_Corr†':>12}  {'Gemma_OCR':>10}  {'Gemma_Corr†':>12}")
    print("  " + "-" * 86)
    for ds in FP:
        if ds not in qaari: continue
        qo, _, _, _ = _metrics(qaari[ds], "ocr_text",       runaway_fix=False)
        qc, _, _, _ = _metrics(qaari[ds], "corrected_text", runaway_fix=True)
        go, _, _, _ = _metrics(gemma.get(ds, []), "ocr_text",       runaway_fix=False)
        gc, _, _, _ = _metrics(gemma.get(ds, []), "corrected_text", runaway_fix=True)
        print(f"  {ds:<35}  {pct(qo):>10}  {pct(qc):>12}  {pct(go):>10}  {pct(gc):>12}")
    fp_qo, _ = _agg(qaari, FP, "ocr_text", False);   fp_qc, _ = _agg(qaari, FP, "corrected_text", True)
    fp_go, _ = _agg(gemma, FP, "ocr_text", False);   fp_gc, _ = _agg(gemma, FP, "corrected_text", True)
    print(f"  {'Full-page Avg':<35}  {pct(fp_qo):>10}  {pct(fp_qc):>12}  {pct(fp_go):>10}  {pct(fp_gc):>12}")


# ── Category C: Generalization ───────────────────────────────────────────────

def run_C():
    _section("CATEGORY C — Generalization  (Qwen3-4B + T2, BM Bench + Kitab)")
    print("  Research Question: Does the best model generalize to new unseen domains?")
    print("  Model: Qwen3-4B-Instruct-2507   |   Prompt: T2 (base_rag)\n")

    bm_q   = _load(_CORR / "qwen3-4b_bm_qaari_t2.jsonl")
    bm_g   = _load(_CORR / "qwen3-4b_bm_gemma_t2.jsonl")
    kit_q  = _load(_CORR / "qwen3-4b_kitab_qaari_t2.jsonl")
    kit_g  = _load(_CORR / "qwen3-4b_kitab_gemma_t2.jsonl")

    # RDI-Test-Lines — by subset
    _subsection("RDI-Test-Lines — per subset (line-level images, text GT available)")
    print("  Note: 'RDI-Test-Lines' = line-strip images with text GT.")
    print("  'RDI-Test Line Segmentation' exists but has polygon-only GT (no text) -> excluded.")
    print(f"  {'Subset':<25}  {'Src':>5}  {'N':>5}  {'OCR_CER':>8}  {'OCR_WER':>8}  {'Corr_CER':>9}  {'Corr_WER':>9}")
    print("  " + "-" * 82)
    for ds in BM_SETS:
        label = BM_LABELS.get(ds, ds)
        for tag, by_ds in [("Qaari", bm_q), ("Gemma", bm_g)]:
            recs = by_ds.get(ds, [])
            n = sum(1 for r in recs if _is_normal(r["ocr_text"], r["gt_text"]))
            oc, ow, _, _ = _metrics(recs, "ocr_text",       runaway_fix=False)
            cc, cw, _, _ = _metrics(recs, "corrected_text", runaway_fix=True)
            lbl = label if tag == "Qaari" else ""
            print(f"  {lbl:<25}  {tag:>5}  {n:>5}  {pct(oc):>8}  {pct(ow):>8}  {pct(cc):>9}  {pct(cw):>9}")
        print()

    # BM overall aggregate
    bm_oc_q, bm_ow_q = _agg(bm_q, BM_SETS, "ocr_text",       fix=False)
    bm_cc_q, bm_cw_q = _agg(bm_q, BM_SETS, "corrected_text", fix=True)
    bm_oc_g, bm_ow_g = _agg(bm_g, BM_SETS, "ocr_text",       fix=False)
    bm_cc_g, bm_cw_g = _agg(bm_g, BM_SETS, "corrected_text", fix=True)
    print(f"  {'RDI-Test-Lines Overall':<22}  {'Qaari':>5}       {pct(bm_oc_q):>8}  {pct(bm_ow_q):>8}  {pct(bm_cc_q):>9}  {pct(bm_cw_q):>9}")
    print(f"  {'':22}  {'Gemma':>5}       {pct(bm_oc_g):>8}  {pct(bm_ow_g):>8}  {pct(bm_cc_g):>9}  {pct(bm_cw_g):>9}")

    # Kitab — by subset
    _subsection("Kitab Benchmark — per subset")
    print(f"  {'Subset':<25}  {'Src':>5}  {'N':>5}  {'OCR_CER':>8}  {'OCR_WER':>8}  {'Corr_CER':>9}  {'Corr_WER':>9}")
    print("  " + "-" * 80)
    for ds in KITAB_SETS:
        for tag, by_ds in [("Qaari", kit_q), ("Gemma", kit_g)]:
            recs = by_ds.get(ds, [])
            n = sum(1 for r in recs if _is_normal(r["ocr_text"], r["gt_text"]))
            oc, ow, _, _ = _metrics(recs, "ocr_text",       runaway_fix=False)
            cc, cw, _, _ = _metrics(recs, "corrected_text", runaway_fix=True)
            lbl = ds if tag == "Qaari" else ""
            print(f"  {lbl:<25}  {tag:>5}  {n:>5}  {pct(oc):>8}  {pct(ow):>8}  {pct(cc):>9}  {pct(cw):>9}")
        print()

    # Kitab overall
    kit_oc_q, kit_ow_q = _agg(kit_q, KITAB_SETS, "ocr_text",       fix=False)
    kit_cc_q, kit_cw_q = _agg(kit_q, KITAB_SETS, "corrected_text", fix=True)
    kit_oc_g, kit_ow_g = _agg(kit_g, KITAB_SETS, "ocr_text",       fix=False)
    kit_cc_g, kit_cw_g = _agg(kit_g, KITAB_SETS, "corrected_text", fix=True)
    print(f"  {'Kitab Overall':<25}  {'Qaari':>5}       {pct(kit_oc_q):>8}  {pct(kit_ow_q):>8}  {pct(kit_cc_q):>9}  {pct(kit_cw_q):>9}")
    print(f"  {'':25}  {'Gemma':>5}       {pct(kit_oc_g):>8}  {pct(kit_ow_g):>8}  {pct(kit_cc_g):>9}  {pct(kit_cw_g):>9}")


# ── Summary across all categories ────────────────────────────────────────────

def run_summary():
    _section("OVERALL SUMMARY — All 9 Runs")
    print(f"  {'Run':<35}  {'OCR_CER':>8}  {'OCR_WER':>8}  {'Corr_CER':>9}  {'Corr_WER':>9}  {'Delta_CER':>10}")
    print("  " + "-" * 88)

    FULL_PAGE = ["KHATT-Paragraph-validation","Yarmouk-testing","Muharaf-validation","Historical"]
    runs = [
        ("qwen3-4b_val_qaari_t2    (PATS+FP)",   "qwen3-4b_val_qaari_t2.jsonl",    PATS + FULL_PAGE),
        ("qwen3-14b_val_qaari_t2   (PATS+FP)",   "qwen3-14b_val_qaari_t2.jsonl",   PATS + FULL_PAGE),
        ("gemma-3-4b_val_qaari_t2  (PATS+FP)",   "gemma-3-4b_val_qaari_t2.jsonl",  PATS + FULL_PAGE),
        ("gemma-3-12b_val_qaari_t2 (PATS+FP)",   "gemma-3-12b_val_qaari_t2.jsonl", PATS + FULL_PAGE),
        # Gemma OCR: only full-page (PATS/KHATT excluded — invalid on line strips)
        ("qwen3-4b_val_gemma_t2    (FP only)",   "qwen3-4b_val_gemma_t2.jsonl",    FULL_PAGE),
        ("qwen3-4b_bm_qaari_t2     (BM LR)",     "qwen3-4b_bm_qaari_t2.jsonl",     BM_SETS),
        ("qwen3-4b_bm_gemma_t2     (BM LR)",     "qwen3-4b_bm_gemma_t2.jsonl",     BM_SETS),
        ("qwen3-4b_kitab_qaari_t2  (Kitab)",     "qwen3-4b_kitab_qaari_t2.jsonl",  KITAB_SETS),
        ("qwen3-4b_kitab_gemma_t2  (Kitab)",     "qwen3-4b_kitab_gemma_t2.jsonl",  KITAB_SETS),
    ]

    for label, fname, keys in runs:
        p = _CORR / fname
        if not p.exists():
            print(f"  {label:<35}  [missing]")
            continue
        by_ds = _load(p)
        oc, ow = _agg(by_ds, keys, "ocr_text",       fix=False)
        cc, cw = _agg(by_ds, keys, "corrected_text", fix=True)
        delta = (oc - cc) if (oc is not None and cc is not None) else None
        delta_s = f"{delta*100:+.2f}%" if delta is not None else "  N/A"
        print(f"  {label:<35}  {pct(oc):>8}  {pct(ow):>8}  {pct(cc):>9}  {pct(cw):>9}  {delta_s:>10}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nEXPERIMENT 3 — COMPREHENSIVE ANALYSIS")
    print("Model: Qwen3-4B (primary) | Prompt: T2 (base_rag) | No diacritics | Normal samples | Runaway-corrected")
    run_A()
    run_B()
    run_C()
    run_summary()
    print(f"\n{SEP}\n  DONE\n{SEP}\n")


if __name__ == "__main__":
    main()
