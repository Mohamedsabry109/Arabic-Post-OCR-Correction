"""Human-readable sample report generation for OCR correction results.

Generates categorised sample reports with OCR / LLM / GT displayed on
separate lines, one under the other, for easy visual inspection and
error analysis.

Usage::

    from src.analysis.report_formatter import write_corrections_report

    write_corrections_report(
        corrections_path=Path("results/phase2/corrections.jsonl"),
        output_path=Path("results/phase2/sample_report.txt"),
        title="Phase 2 -- Zero-Shot Correction",
    )

The function also prints a short ASCII-safe category summary to stdout.
All Arabic text is written only to the UTF-8 report file.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Arabic diacritics stripping (used for DIACRITIC_ONLY categorisation)
# ---------------------------------------------------------------------------

_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)


def _strip_diacritics(text: str) -> str:
    return _DIACRITICS_RE.sub("", text)


def _word_count(text: str) -> int:
    return len(text.split())


def _flatten(text: str) -> str:
    """Replace newlines/tabs with a single space for single-line display."""
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_len: int = 220) -> str:
    text = _flatten(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + " [...]"


# ---------------------------------------------------------------------------
# Error / outcome categories
# ---------------------------------------------------------------------------

# Display order for the category table
CATEGORY_ORDER = [
    "NO_CHANGE",
    "PERFECT",
    "REPETITION_REMOVED",
    "PARTIAL_FIX",
    "FALSE_POSITIVE",
    "DIACRITIC_ONLY",
    "HALLUCINATION",
    "TRUNCATION",
    "WORD_SUBSTITUTION",
]

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "NO_CHANGE":           "LLM returned the OCR text unchanged",
    "PERFECT":             "LLM output exactly matches ground truth",
    "REPETITION_REMOVED":  "LLM removed hallucinated OCR repetition (large improvement)",
    "PARTIAL_FIX":         "LLM improved CER but did not reach ground truth",
    "FALSE_POSITIVE":      "OCR was near-correct (CER<5%) but LLM introduced errors",
    "DIACRITIC_ONLY":      "Changes limited to Arabic diacritics (tashkeel)",
    "HALLUCINATION":       "LLM output significantly longer than OCR input",
    "TRUNCATION":          "LLM output significantly shorter than OCR input",
    "WORD_SUBSTITUTION":   "LLM changed/substituted words, making output worse",
}


def categorize_sample(
    ocr_text: str,
    corrected_text: str,
    gt_text: str,
    cer_ocr: float,
    cer_llm: float,
) -> str:
    """Return a human-readable category string for this correction sample."""
    ocr = ocr_text.strip()
    corr = corrected_text.strip()
    gt = gt_text.strip()

    if corr == ocr:
        return "NO_CHANGE"
    if corr == gt:
        return "PERFECT"

    improved = cer_llm < cer_ocr - 1e-6

    if improved:
        if _word_count(corr) < _word_count(ocr) * 0.65:
            return "REPETITION_REMOVED"
        return "PARTIAL_FIX"

    # Regressed -- sub-categorise
    if cer_ocr < 0.05:
        return "FALSE_POSITIVE"
    if _strip_diacritics(corr) == _strip_diacritics(ocr):
        return "DIACRITIC_ONLY"
    if len(corr) > len(ocr) * 1.4:
        return "HALLUCINATION"
    if len(corr) < len(ocr) * 0.6:
        return "TRUNCATION"
    return "WORD_SUBSTITUTION"


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------

@dataclass
class _SampleResult:
    sample_id: str
    dataset: str
    ocr_text: str
    corrected_text: str
    gt_text: str
    # Primary metrics (diacritics stripped — used for ranking & categorisation)
    cer_ocr: float
    wer_ocr: float
    cer_llm: float
    wer_llm: float
    cer_delta: float   # positive = improvement
    outcome: str       # "improved" | "regressed" | "unchanged"
    category: str
    # Raw metrics (with diacritics — for dual reporting)
    cer_ocr_raw: float = 0.0
    wer_ocr_raw: float = 0.0
    cer_llm_raw: float = 0.0
    wer_llm_raw: float = 0.0
    is_runaway: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_corrections(results_dir: Path) -> list[dict]:
    """Load corrections from combined or per-dataset corrections.jsonl files."""
    combined = results_dir / "corrections.jsonl"
    if combined.exists():
        return _read_jsonl(combined)
    # Fall back: aggregate all per-dataset files
    records: list[dict] = []
    for path in sorted(results_dir.glob("*/corrections.jsonl")):
        records.extend(_read_jsonl(path))
    return records


_RUNAWAY_RATIO_THRESHOLD = 5.0
"""OCR/GT length ratio above which a sample is classified as runaway.

Must match the threshold in ``pipelines/run_phase2.py`` and
``src/analysis/metrics.calculate_metrics_split``.
"""


def _compute_results(records: list[dict]) -> list[_SampleResult]:
    # Import here to avoid circular imports
    from src.analysis.metrics import calculate_cer, calculate_wer
    from src.data.text_utils import normalise_arabic

    results: list[_SampleResult] = []
    for rec in records:
        if not rec.get("success", True):
            continue
        ocr = rec.get("ocr_text", "")
        corrected = rec.get("corrected_text", ocr)
        gt = rec.get("gt_text", "")
        if not gt:
            continue

        # Runaway detection (same criterion as Phase 1 / Phase 2 pipeline)
        gt_len = max(len(normalise_arabic(gt)), 1)
        ocr_len = len(normalise_arabic(ocr))
        is_runaway = ocr_len / gt_len > _RUNAWAY_RATIO_THRESHOLD

        # Primary metrics: diacritics stripped (used for ranking/categorisation)
        cer_ocr = calculate_cer(gt, ocr, strip_diacritics=True)
        wer_ocr = calculate_wer(gt, ocr, strip_diacritics=True)
        cer_llm = calculate_cer(gt, corrected, strip_diacritics=True)
        wer_llm = calculate_wer(gt, corrected, strip_diacritics=True)
        cer_delta = cer_ocr - cer_llm

        # Raw metrics: with diacritics (for dual reporting)
        cer_ocr_raw = calculate_cer(gt, ocr, strip_diacritics=False)
        wer_ocr_raw = calculate_wer(gt, ocr, strip_diacritics=False)
        cer_llm_raw = calculate_cer(gt, corrected, strip_diacritics=False)
        wer_llm_raw = calculate_wer(gt, corrected, strip_diacritics=False)

        if cer_delta > 1e-6:
            outcome = "improved"
        elif cer_delta < -1e-6:
            outcome = "regressed"
        else:
            outcome = "unchanged"

        category = categorize_sample(ocr, corrected, gt, cer_ocr, cer_llm)

        results.append(_SampleResult(
            sample_id=rec.get("sample_id", ""),
            dataset=rec.get("dataset", ""),
            ocr_text=ocr,
            corrected_text=corrected,
            gt_text=gt,
            cer_ocr=cer_ocr,
            wer_ocr=wer_ocr,
            cer_llm=cer_llm,
            wer_llm=wer_llm,
            cer_delta=cer_delta,
            outcome=outcome,
            category=category,
            cer_ocr_raw=cer_ocr_raw,
            wer_ocr_raw=wer_ocr_raw,
            cer_llm_raw=cer_llm_raw,
            wer_llm_raw=wer_llm_raw,
            is_runaway=is_runaway,
        ))
    return results


def _fmt_sample(sr: _SampleResult, index: int) -> str:
    """Format one sample as a multi-line block with dual metrics."""
    if sr.outcome == "improved":
        delta_str = f"+{sr.cer_delta * 100:.1f}pp better"
        tag = "IMPROVED"
    elif sr.outcome == "regressed":
        delta_str = f"{sr.cer_delta * 100:.1f}pp worse"
        tag = "REGRESSED"
    else:
        delta_str = "no change"
        tag = "UNCHANGED"

    header = (
        f"[{index:03d}] {sr.sample_id}  "
        f"dataset={sr.dataset}  "
        f"{tag}  category={sr.category}"
    )
    metrics_stripped = (
        f"      CER (no diac): {sr.cer_ocr * 100:5.1f}%  ->  {sr.cer_llm * 100:5.1f}%  "
        f"({delta_str})   "
        f"WER: {sr.wer_ocr * 100:5.1f}%  ->  {sr.wer_llm * 100:5.1f}%"
    )
    metrics_raw = (
        f"      CER (raw)    : {sr.cer_ocr_raw * 100:5.1f}%  ->  {sr.cer_llm_raw * 100:5.1f}%  "
        f"               "
        f"WER: {sr.wer_ocr_raw * 100:5.1f}%  ->  {sr.wer_llm_raw * 100:5.1f}%"
    )
    ocr_line  = f"      OCR: {_truncate(sr.ocr_text)}"
    llm_line  = f"      LLM: {_truncate(sr.corrected_text)}"
    gt_line   = f"       GT: {_truncate(sr.gt_text)}"

    return "\n".join([header, metrics_stripped, metrics_raw, ocr_line, llm_line, gt_line])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(
    corrections_path: Path,
    title: str,
    top_regressed: int = 40,
    top_improved: int = 20,
    exclude_runaway: bool = False,
) -> str:
    """Build and return the full sample report as a UTF-8 string.

    Args:
        corrections_path: Path to ``corrections.jsonl`` **or** the parent
            directory (the function will locate the file automatically).
        title: One-line title printed in the report header.
        top_regressed: How many regressed samples to include (worst first).
        top_improved: How many improved samples to include (best first).

    Returns:
        The complete report as a string, ready to be written to a .txt file.
    """
    # Accept either a file or a directory
    if corrections_path.is_dir():
        records = _load_corrections(corrections_path)
        src_display = str(corrections_path)
    else:
        records = _read_jsonl(corrections_path)
        src_display = str(corrections_path)

    all_results = _compute_results(records)
    if not all_results:
        return f"[report_formatter] No valid samples found in {src_display}\n"

    # Split into normal / runaway (same threshold as Phase 1 & Phase 2 pipeline)
    normal_results = [r for r in all_results if not r.is_runaway]
    n_runaway = len(all_results) - len(normal_results)

    # Primary sample set based on exclude_runaway config
    results = normal_results if exclude_runaway else all_results
    n = len(results)

    improved  = sorted([r for r in results if r.outcome == "improved"],
                       key=lambda r: r.cer_delta, reverse=True)
    regressed = sorted([r for r in results if r.outcome == "regressed"],
                       key=lambda r: r.cer_delta)
    unchanged = [r for r in results if r.outcome == "unchanged"]

    cat_counts: Counter[str] = Counter(r.category for r in results)

    # Per-dataset aggregation helper
    def _aggregate(sample_list: list[_SampleResult]) -> dict[str, dict]:
        ds: dict[str, dict] = {}
        for r in sample_list:
            d = ds.setdefault(r.dataset, {
                "improved": 0, "regressed": 0, "unchanged": 0,
                "cer_ocr_sum": 0.0, "cer_llm_sum": 0.0,
                "wer_ocr_sum": 0.0, "wer_llm_sum": 0.0,
                "cer_ocr_raw_sum": 0.0, "cer_llm_raw_sum": 0.0,
                "wer_ocr_raw_sum": 0.0, "wer_llm_raw_sum": 0.0,
                "n": 0,
            })
            d[r.outcome] += 1
            d["cer_ocr_sum"] += r.cer_ocr
            d["cer_llm_sum"] += r.cer_llm
            d["wer_ocr_sum"] += r.wer_ocr
            d["wer_llm_sum"] += r.wer_llm
            d["cer_ocr_raw_sum"] += r.cer_ocr_raw
            d["cer_llm_raw_sum"] += r.cer_llm_raw
            d["wer_ocr_raw_sum"] += r.wer_ocr_raw
            d["wer_llm_raw_sum"] += r.wer_llm_raw
            d["n"] += 1
        return ds

    ds_all = _aggregate(all_results)
    ds_normal = _aggregate(normal_results)

    W = 88
    SEP  = "=" * W
    THIN = "-" * W
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines += [
        SEP,
        f"  SAMPLE REPORT: {title}",
        f"  Source: {src_display}",
        SEP, "",
    ]

    # ------------------------------------------------------------------
    # Overall summary helper
    # ------------------------------------------------------------------
    def _overall_block(label: str, sample_list: list[_SampleResult]) -> None:
        sn = len(sample_list)
        if sn == 0:
            return
        s_imp = sum(1 for r in sample_list if r.outcome == "improved")
        s_reg = sum(1 for r in sample_list if r.outcome == "regressed")
        s_unc = sn - s_imp - s_reg

        a_cer_o = sum(r.cer_ocr for r in sample_list) / sn
        a_cer_l = sum(r.cer_llm for r in sample_list) / sn
        a_wer_o = sum(r.wer_ocr for r in sample_list) / sn
        a_wer_l = sum(r.wer_llm for r in sample_list) / sn
        net_s = a_cer_o - a_cer_l
        dir_s = "better" if net_s > 0 else "worse"

        a_cer_o_r = sum(r.cer_ocr_raw for r in sample_list) / sn
        a_cer_l_r = sum(r.cer_llm_raw for r in sample_list) / sn
        a_wer_o_r = sum(r.wer_ocr_raw for r in sample_list) / sn
        a_wer_l_r = sum(r.wer_llm_raw for r in sample_list) / sn
        net_r = a_cer_o_r - a_cer_l_r
        dir_r = "better" if net_r > 0 else "worse"

        lines.extend([
            f"{label}  ({sn} samples)",
            THIN,
            f"  -- Diacritics stripped --",
            f"  Avg CER       : {a_cer_o * 100:.2f}%  ->  {a_cer_l * 100:.2f}%  "
            f"({dir_s} by {abs(net_s) * 100:.2f}pp)",
            f"  Avg WER       : {a_wer_o * 100:.2f}%  ->  {a_wer_l * 100:.2f}%",
            f"  -- With diacritics --",
            f"  Avg CER       : {a_cer_o_r * 100:.2f}%  ->  {a_cer_l_r * 100:.2f}%  "
            f"({dir_r} by {abs(net_r) * 100:.2f}pp)",
            f"  Avg WER       : {a_wer_o_r * 100:.2f}%  ->  {a_wer_l_r * 100:.2f}%",
            f"  Improved={s_imp} ({s_imp/sn*100:.1f}%)  "
            f"Regressed={s_reg} ({s_reg/sn*100:.1f}%)  "
            f"Unchanged={s_unc} ({s_unc/sn*100:.1f}%)",
            "",
        ])

    # ------------------------------------------------------------------
    # Data quality line
    # ------------------------------------------------------------------
    lines += [
        f"  Total samples : {len(all_results)}",
        f"  Runaway       : {n_runaway}  (OCR>{_RUNAWAY_RATIO_THRESHOLD}x GT length)",
        f"  Normal        : {len(normal_results)}",
        "",
    ]

    _overall_block("ALL SAMPLES", all_results)
    _overall_block("NORMAL-ONLY (excludes runaway)", normal_results)

    # ------------------------------------------------------------------
    # Dataset summary table helper
    # ------------------------------------------------------------------
    def _ds_table(label: str, ds_data: dict[str, dict], show_outcomes: bool = True) -> None:
        if show_outcomes:
            lines.extend([
                label,
                THIN,
                f"  {'Dataset':<34} {'OCR CER':>8} {'LLM CER':>8} {'OCR WER':>8} {'LLM WER':>8} "
                f"{'Improved':>9} {'Regressed':>10} {'Unchanged':>9}",
                THIN,
            ])
            for ds, d in sorted(ds_data.items()):
                dn = d["n"]
                lines.append(
                    f"  {ds:<34} {d['cer_ocr_sum']/dn*100:>7.1f}%  {d['cer_llm_sum']/dn*100:>7.1f}%"
                    f"  {d['wer_ocr_sum']/dn*100:>7.1f}%  {d['wer_llm_sum']/dn*100:>7.1f}%"
                    f"  {d['improved']:>4} ({d['improved']/dn*100:>3.0f}%)"
                    f"  {d['regressed']:>4} ({d['regressed']/dn*100:>3.0f}%)"
                    f"  {d['unchanged']:>4}"
                )
        else:
            lines.extend([
                label,
                THIN,
                f"  {'Dataset':<34} {'OCR CER':>8} {'LLM CER':>8} {'OCR WER':>8} {'LLM WER':>8}  {'N':>5}",
                THIN,
            ])
            for ds, d in sorted(ds_data.items()):
                dn = d["n"]
                lines.append(
                    f"  {ds:<34} {d['cer_ocr_sum']/dn*100:>7.1f}%  {d['cer_llm_sum']/dn*100:>7.1f}%"
                    f"  {d['wer_ocr_sum']/dn*100:>7.1f}%  {d['wer_llm_sum']/dn*100:>7.1f}%  {dn:>5}"
                )
        lines.extend([THIN, ""])

    # All samples tables
    _ds_table("DATASET SUMMARY -- ALL SAMPLES (diacritics stripped)", ds_all, show_outcomes=True)
    # For raw/with-diacritics, swap in the raw sums
    ds_all_raw = {}
    for ds, d in ds_all.items():
        ds_all_raw[ds] = {**d, "cer_ocr_sum": d["cer_ocr_raw_sum"], "cer_llm_sum": d["cer_llm_raw_sum"],
                          "wer_ocr_sum": d["wer_ocr_raw_sum"], "wer_llm_sum": d["wer_llm_raw_sum"]}
    _ds_table("DATASET SUMMARY -- ALL SAMPLES (with diacritics)", ds_all_raw, show_outcomes=False)

    # Normal-only tables
    _ds_table("DATASET SUMMARY -- NORMAL-ONLY (diacritics stripped)", ds_normal, show_outcomes=True)
    ds_norm_raw = {}
    for ds, d in ds_normal.items():
        ds_norm_raw[ds] = {**d, "cer_ocr_sum": d["cer_ocr_raw_sum"], "cer_llm_sum": d["cer_llm_raw_sum"],
                           "wer_ocr_sum": d["wer_ocr_raw_sum"], "wer_llm_sum": d["wer_llm_raw_sum"]}
    _ds_table("DATASET SUMMARY -- NORMAL-ONLY (with diacritics)", ds_norm_raw, show_outcomes=False)

    # ------------------------------------------------------------------
    # Category breakdown
    # ------------------------------------------------------------------
    lines += [
        "CORRECTION CATEGORIES",
        THIN,
    ]
    for cat in CATEGORY_ORDER:
        count = cat_counts.get(cat, 0)
        desc = CATEGORY_DESCRIPTIONS.get(cat, "")
        lines.append(
            f"  {cat:<22}  {count:>4}  ({count / n * 100:>4.0f}%)   {desc}"
        )
    # Any categories not in the canonical order
    for cat, count in sorted(cat_counts.items()):
        if cat not in CATEGORY_ORDER:
            lines.append(
                f"  {cat:<22}  {count:>4}  ({count / n * 100:>4.0f}%)"
            )
    lines += ["", ""]

    # ------------------------------------------------------------------
    # Regressed samples
    # ------------------------------------------------------------------
    lines += [
        SEP,
        f"  REGRESSED SAMPLES  ({len(regressed)} total, sorted worst-first)",
        SEP, "",
    ]
    for i, sr in enumerate(regressed[:top_regressed], 1):
        lines.append(_fmt_sample(sr, i))
        lines.append("")
    if len(regressed) > top_regressed:
        lines.append(
            f"  ... {len(regressed) - top_regressed} more regressed samples "
            f"omitted (see corrections.jsonl)"
        )
    lines += ["", ""]

    # ------------------------------------------------------------------
    # Improved samples
    # ------------------------------------------------------------------
    lines += [
        SEP,
        f"  IMPROVED SAMPLES  ({len(improved)} total, sorted best-first)",
        SEP, "",
    ]
    for i, sr in enumerate(improved[:top_improved], 1):
        lines.append(_fmt_sample(sr, i))
        lines.append("")
    if len(improved) > top_improved:
        lines.append(
            f"  ... {len(improved) - top_improved} more improved samples "
            f"omitted (see corrections.jsonl)"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_corrections_report(
    corrections_path: Path,
    output_path: Path,
    title: str,
    top_regressed: int = 40,
    top_improved: int = 20,
    exclude_runaway: bool = False,
) -> None:
    """Build the report, save it to *output_path*, and print a category
    summary to stdout (ASCII-safe — no Arabic characters on the console).

    Args:
        corrections_path: Path to ``corrections.jsonl`` or its parent dir.
        output_path: Where to write the ``.txt`` report.
        title: One-line report title.
        top_regressed: Regressed samples to include in the report.
        top_improved: Improved samples to include in the report.
        exclude_runaway: If True, exclude runaway samples from aggregates.
    """
    report = build_report(
        corrections_path=corrections_path,
        title=title,
        top_regressed=top_regressed,
        top_improved=top_improved,
        exclude_runaway=exclude_runaway,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ------------------------------------------------------------------
    # ASCII-safe console summary (no Arabic, normal-only)
    # ------------------------------------------------------------------
    if corrections_path.is_dir():
        records = _load_corrections(corrections_path)
    else:
        records = _read_jsonl(corrections_path)
    all_results = _compute_results(records)
    if not all_results:
        print(f"[sample report] No valid samples -- skipping.")
        return

    normal_results = [r for r in all_results if not r.is_runaway]
    n_runaway = len(all_results) - len(normal_results)
    results = normal_results if exclude_runaway else all_results
    n = len(results)
    n_imp = sum(1 for r in results if r.outcome == "improved")
    n_reg = sum(1 for r in results if r.outcome == "regressed")
    n_unc = n - n_imp - n_reg
    cat_counts: Counter[str] = Counter(r.category for r in results)

    avg_cer_s = sum(r.cer_llm for r in results) / n
    avg_cer_r = sum(r.cer_llm_raw for r in results) / n
    avg_wer_s = sum(r.wer_llm for r in results) / n
    avg_wer_r = sum(r.wer_llm_raw for r in results) / n

    variant_label = "normal-only" if exclude_runaway else "all samples"
    print()
    print("  Sample report saved:", output_path)
    if n_runaway:
        print(f"  Total: {len(all_results)}  Normal: {len(normal_results)}  Runaway: {n_runaway}  Using: {variant_label}")
    print(f"  Improved={n_imp} ({n_imp/n*100:.1f}%)  "
          f"Regressed={n_reg} ({n_reg/n*100:.1f}%)  "
          f"Unchanged={n_unc} ({n_unc/n*100:.1f}%)")
    print(f"  LLM CER (no diac): {avg_cer_s*100:.2f}%  "
          f"WER: {avg_wer_s*100:.2f}%")
    print(f"  LLM CER (raw)    : {avg_cer_r*100:.2f}%  "
          f"WER: {avg_wer_r*100:.2f}%")
    print("  Categories:")
    for cat in CATEGORY_ORDER:
        count = cat_counts.get(cat, 0)
        if count:
            desc = CATEGORY_DESCRIPTIONS.get(cat, "")
            print(f"    {cat:<22}  {count:>4}  ({count/n*100:>4.0f}%)  {desc}")
