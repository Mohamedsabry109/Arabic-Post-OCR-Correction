#!/usr/bin/env python3
"""Analyze Phase 2 training corrections to extract reusable artifacts.

Reads corrections.jsonl (OCR text + LLM correction + ground truth) and produces:

  analysis/word_pairs_ocr.txt           - OCR->GT word substitution pairs by frequency
  analysis/word_pairs_llm_failures.txt  - Words the LLM failed to correct or introduced
  analysis/sentence_failures.jsonl      - Sentences where LLM degraded or left unchanged
  analysis/error_stats.json             - Fix/introduce rates by ErrorType (PATS + KHATT)
  analysis/sample_classification.json   - Bucket counts and per-dataset breakdown
  analysis/summary.md                   - Human-readable overview

The word_pairs_ocr.txt file uses the same format as WordErrorPairsLoader expects
(``ocr_word > gt_word``), so it can be directly consumed by Phase 4D and Phase 5.

Usage:
    python scripts/analyze_training.py
    python scripts/analyze_training.py --limit 500
    python scripts/analyze_training.py --input results/phase2/corrections.jsonl
    python scripts/analyze_training.py --skip-error-types   # fast mode, no char alignment
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.metrics import calculate_cer
from src.data.text_utils import normalise_arabic, tokenise_arabic, is_arabic_word

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INPUT = _PROJECT_ROOT / "results" / "phase2" / "corrections.jsonl"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase2" / "analysis"

# CER thresholds for classifying LLM improvement
_DEGRADED_THRESHOLD = 0.005   # CER increase > this = degraded
_UNCHANGED_THRESHOLD = 0.005  # |CER delta| < this = unchanged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze training corrections to extract error patterns."
    )
    p.add_argument(
        "--input", type=Path, default=_DEFAULT_INPUT,
        help="Path to corrections.jsonl (default: results/phase2/corrections.jsonl).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, dest="output_dir",
        help="Output directory for analysis artifacts.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records (for testing).",
    )
    p.add_argument(
        "--skip-error-types", action="store_true", dest="skip_error_types",
        help="Skip LLMErrorAnalyzer (fast mode -- no char-level error type breakdown).",
    )
    p.add_argument(
        "--top-word-pairs", type=int, default=200, dest="top_word_pairs",
        help="Number of top word pairs to output (default: 200).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_corrections(path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load corrections.jsonl records."""
    if not path.exists():
        logger.error("Input file not found: %s", path)
        sys.exit(1)

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Skip failed samples
            if not r.get("success", True):
                continue
            # Need all three texts
            if not r.get("gt_text") or not r.get("ocr_text"):
                continue
            records.append(r)
            if limit and len(records) >= limit:
                break

    logger.info("Loaded %d valid records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Word-level alignment (lightweight, no ErrorAnalyzer dependency)
# ---------------------------------------------------------------------------


def align_words(ref_words: list[str], hyp_words: list[str]) -> list[tuple[Optional[str], Optional[str]]]:
    """DP word alignment. Returns (ref_word, hyp_word) pairs; None = ins/del."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    alignment: list[tuple[Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            alignment.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref_words[i - 1], None))
            i -= 1
        else:
            alignment.append((None, hyp_words[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


def extract_word_pairs(
    gt_text: str,
    hyp_text: str,
    strip_diacritics: bool = True,
) -> list[tuple[str, str]]:
    """Extract (hyp_word, gt_word) substitution pairs from aligned words.

    Only returns pairs where both words are Arabic and different.
    """
    gt_norm = normalise_arabic(gt_text, remove_diacritics=strip_diacritics)
    hyp_norm = normalise_arabic(hyp_text, remove_diacritics=strip_diacritics)

    gt_words = tokenise_arabic(gt_norm)
    hyp_words = tokenise_arabic(hyp_norm)

    pairs: list[tuple[str, str]] = []
    for gt_w, hyp_w in align_words(gt_words, hyp_words):
        if gt_w and hyp_w and gt_w != hyp_w:
            if is_arabic_word(gt_w) and is_arabic_word(hyp_w):
                pairs.append((hyp_w, gt_w))
    return pairs


# ---------------------------------------------------------------------------
# Sample classification
# ---------------------------------------------------------------------------


def classify_sample(cer_ocr: float, cer_llm: float) -> str:
    """Classify a sample by LLM correction outcome."""
    if cer_llm == 0.0:
        return "perfect"
    delta = cer_llm - cer_ocr
    if delta > _DEGRADED_THRESHOLD:
        return "degraded"
    if abs(delta) <= _UNCHANGED_THRESHOLD:
        return "unchanged"
    return "improved"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(args: argparse.Namespace) -> None:
    """Run the full training analysis pipeline."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_corrections(args.input, args.limit)
    if not records:
        logger.error("No records to analyze.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Compute per-sample CER and classify
    # ------------------------------------------------------------------
    logger.info("Computing per-sample CER ...")

    ocr_word_pairs: Counter = Counter()       # (ocr_word, gt_word) -> count
    llm_unfixed_pairs: Counter = Counter()    # (ocr_word, gt_word) LLM didn't fix
    llm_introduced_pairs: Counter = Counter() # (llm_word, gt_word) LLM introduced

    classifications: list[dict] = []
    bucket_counts: Counter = Counter()
    per_dataset_counts: dict[str, Counter] = defaultdict(Counter)

    degraded_samples: list[dict] = []
    unchanged_hard_samples: list[dict] = []

    for r in tqdm(records, desc="Classifying samples", unit="sample"):
        ocr_text = r["ocr_text"]
        gt_text = r["gt_text"]
        llm_text = r.get("corrected_text", ocr_text)
        dataset = r.get("dataset", "unknown")
        sample_id = r.get("sample_id", "")

        cer_ocr = calculate_cer(gt_text, ocr_text, strip_diacritics=True)
        cer_llm = calculate_cer(gt_text, llm_text, strip_diacritics=True)
        bucket = classify_sample(cer_ocr, cer_llm)

        bucket_counts[bucket] += 1
        per_dataset_counts[dataset][bucket] += 1

        info = {
            "sample_id": sample_id,
            "dataset": dataset,
            "cer_ocr": round(cer_ocr, 6),
            "cer_llm": round(cer_llm, 6),
            "cer_delta": round(cer_llm - cer_ocr, 6),
            "bucket": bucket,
        }
        classifications.append(info)

        # Collect sentence-level failures
        if bucket == "degraded":
            degraded_samples.append({
                **info,
                "ocr_text": ocr_text,
                "corrected_text": llm_text,
                "gt_text": gt_text,
            })
        elif bucket == "unchanged" and cer_ocr > 0.05:
            unchanged_hard_samples.append({
                **info,
                "ocr_text": ocr_text,
                "corrected_text": llm_text,
                "gt_text": gt_text,
            })

        # Extract word-level pairs
        # OCR vs GT
        for ocr_w, gt_w in extract_word_pairs(gt_text, ocr_text):
            ocr_word_pairs[(ocr_w, gt_w)] += 1

        # LLM vs GT -- find what LLM still gets wrong
        llm_vs_gt = extract_word_pairs(gt_text, llm_text)
        llm_error_set = set((hw, gw) for hw, gw in llm_vs_gt)
        ocr_vs_gt = extract_word_pairs(gt_text, ocr_text)
        ocr_error_set = set((ow, gw) for ow, gw in ocr_vs_gt)

        for llm_w, gt_w in llm_vs_gt:
            if (llm_w, gt_w) in ocr_error_set:
                # Same error existed in OCR -- LLM didn't fix it
                llm_unfixed_pairs[(llm_w, gt_w)] += 1
            else:
                # New error introduced by LLM
                llm_introduced_pairs[(llm_w, gt_w)] += 1

    # Sort failure samples by severity
    degraded_samples.sort(key=lambda x: x["cer_delta"], reverse=True)
    unchanged_hard_samples.sort(key=lambda x: x["cer_ocr"], reverse=True)

    total = len(records)
    logger.info(
        "Classification: %d perfect, %d improved, %d unchanged, %d degraded (of %d total)",
        bucket_counts["perfect"], bucket_counts["improved"],
        bucket_counts["unchanged"], bucket_counts["degraded"], total,
    )

    # ------------------------------------------------------------------
    # 2. Write word pairs (OCR -> GT)
    # ------------------------------------------------------------------
    top_n = args.top_word_pairs
    ocr_pairs_path = output_dir / "word_pairs_ocr.txt"
    top_ocr_pairs = ocr_word_pairs.most_common(top_n)

    with open(ocr_pairs_path, "w", encoding="utf-8") as f:
        f.write(f"# OCR -> GT Word Substitution Pairs\n")
        f.write(f"# Extracted from {total} training samples\n")
        f.write(f"# Top {len(top_ocr_pairs)} pairs by frequency\n")
        f.write(f"# Format: ocr_word > gt_word  (compatible with WordErrorPairsLoader)\n")
        f.write(f"#\n")
        for (ocr_w, gt_w), count in top_ocr_pairs:
            f.write(f"{ocr_w} > {gt_w}\n")

    logger.info(
        "Wrote %d OCR word pairs to %s (from %d unique pairs total)",
        len(top_ocr_pairs), ocr_pairs_path, len(ocr_word_pairs),
    )

    # ------------------------------------------------------------------
    # 3. Write LLM failure word pairs
    # ------------------------------------------------------------------
    llm_failures_path = output_dir / "word_pairs_llm_failures.txt"
    top_unfixed = llm_unfixed_pairs.most_common(top_n)
    top_introduced = llm_introduced_pairs.most_common(top_n)

    with open(llm_failures_path, "w", encoding="utf-8") as f:
        f.write(f"# LLM Failure Word Pairs\n")
        f.write(f"# Extracted from {total} training samples\n")
        f.write(f"#\n")
        f.write(f"# === UNFIXED: OCR errors the LLM did not correct ===\n")
        f.write(f"# (These are the highest-value targets for prompt improvement)\n")
        f.write(f"# Format: ocr_word > gt_word\n")
        f.write(f"#\n")
        for (ocr_w, gt_w), count in top_unfixed:
            f.write(f"{ocr_w} > {gt_w}\n")
        f.write(f"\n")
        f.write(f"# === INTRODUCED: new errors the LLM created ===\n")
        f.write(f"# (Words the LLM changed incorrectly -- over-correction)\n")
        f.write(f"# Format: llm_word > gt_word\n")
        f.write(f"#\n")
        for (llm_w, gt_w), count in top_introduced:
            f.write(f"{llm_w} > {gt_w}\n")

    logger.info(
        "Wrote LLM failures to %s (%d unfixed, %d introduced)",
        llm_failures_path, len(top_unfixed), len(top_introduced),
    )

    # ------------------------------------------------------------------
    # 4. Write sentence-level failures
    # ------------------------------------------------------------------
    sentences_path = output_dir / "sentence_failures.jsonl"
    with open(sentences_path, "w", encoding="utf-8") as f:
        # Degraded first, then unchanged-hard
        for sample in degraded_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        for sample in unchanged_hard_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote %d sentence failures to %s (%d degraded, %d unchanged-hard)",
        len(degraded_samples) + len(unchanged_hard_samples),
        sentences_path, len(degraded_samples), len(unchanged_hard_samples),
    )

    # ------------------------------------------------------------------
    # 5. Error type analysis (optional, slower)
    # ------------------------------------------------------------------
    error_stats: dict = {}
    if not args.skip_error_types:
        logger.info("Running LLMErrorAnalyzer (char-level error type breakdown) ...")
        from src.analysis.llm_error_analyzer import LLMErrorAnalyzer

        llm_analyzer = LLMErrorAnalyzer()

        pats_results: list[dict] = []
        khatt_results: list[dict] = []

        for r in tqdm(records, desc="Error type analysis", unit="sample"):
            dataset = r.get("dataset", "")
            result = llm_analyzer.analyse_sample(
                ocr_text=r["ocr_text"],
                gt_text=r["gt_text"],
                llm_text=r.get("corrected_text", r["ocr_text"]),
                sample_id=r.get("sample_id", ""),
                dataset=dataset,
            )
            if "KHATT" in dataset:
                khatt_results.append(result)
            else:
                pats_results.append(result)

        if pats_results:
            error_stats["PATS-A01"] = llm_analyzer.aggregate(pats_results, "PATS-A01")
        if khatt_results:
            error_stats["KHATT"] = llm_analyzer.aggregate(khatt_results, "KHATT")

        error_stats_path = output_dir / "error_stats.json"
        with open(error_stats_path, "w", encoding="utf-8") as f:
            json.dump(error_stats, f, ensure_ascii=False, indent=2)
        logger.info("Wrote error type stats to %s", error_stats_path)

    # ------------------------------------------------------------------
    # 6. Sample classification summary
    # ------------------------------------------------------------------
    classification_path = output_dir / "sample_classification.json"
    classification_data = {
        "meta": {
            "total_samples": total,
            "input_file": str(args.input),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "overall": {k: bucket_counts[k] for k in ["perfect", "improved", "unchanged", "degraded"]},
        "overall_pct": {
            k: round(bucket_counts[k] / max(total, 1) * 100, 2)
            for k in ["perfect", "improved", "unchanged", "degraded"]
        },
        "per_dataset": {
            ds: {k: counts[k] for k in ["perfect", "improved", "unchanged", "degraded"]}
            for ds, counts in sorted(per_dataset_counts.items())
        },
    }
    with open(classification_path, "w", encoding="utf-8") as f:
        json.dump(classification_data, f, ensure_ascii=False, indent=2)
    logger.info("Wrote classification to %s", classification_path)

    # ------------------------------------------------------------------
    # 7. Human-readable summary
    # ------------------------------------------------------------------
    summary_path = output_dir / "summary.md"
    _write_summary(
        summary_path,
        total=total,
        bucket_counts=bucket_counts,
        per_dataset_counts=per_dataset_counts,
        ocr_word_pairs=ocr_word_pairs,
        llm_unfixed_pairs=llm_unfixed_pairs,
        llm_introduced_pairs=llm_introduced_pairs,
        degraded_count=len(degraded_samples),
        unchanged_hard_count=len(unchanged_hard_samples),
        error_stats=error_stats,
    )
    logger.info("Wrote summary to %s", summary_path)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Analysis complete. Outputs in %s", output_dir)
    logger.info("")
    logger.info("Key artifacts:")
    logger.info("  word_pairs_ocr.txt           - for Phase 4D / prompt <error_patterns>")
    logger.info("  word_pairs_llm_failures.txt  - LLM blind spots for prompt refinement")
    logger.info("  sentence_failures.jsonl      - for craft_prompt.py --mode refine")
    logger.info("  summary.md                   - human-readable overview")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(
    path: Path,
    total: int,
    bucket_counts: Counter,
    per_dataset_counts: dict[str, Counter],
    ocr_word_pairs: Counter,
    llm_unfixed_pairs: Counter,
    llm_introduced_pairs: Counter,
    degraded_count: int,
    unchanged_hard_count: int,
    error_stats: dict,
) -> None:
    """Write a human-readable Markdown summary."""
    lines: list[str] = []

    lines.append("# Training Corrections Analysis")
    lines.append("")
    lines.append(f"**Total samples**: {total}")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # Classification overview
    lines.append("## Sample Classification")
    lines.append("")
    lines.append("| Bucket | Count | % |")
    lines.append("|--------|------:|--:|")
    for bucket in ["perfect", "improved", "unchanged", "degraded"]:
        count = bucket_counts[bucket]
        pct = count / max(total, 1) * 100
        lines.append(f"| {bucket} | {count} | {pct:.1f}% |")
    lines.append("")

    # Per-dataset breakdown
    lines.append("## Per-Dataset Breakdown")
    lines.append("")
    lines.append("| Dataset | Perfect | Improved | Unchanged | Degraded | Total |")
    lines.append("|---------|--------:|---------:|----------:|---------:|------:|")
    for ds in sorted(per_dataset_counts.keys()):
        counts = per_dataset_counts[ds]
        ds_total = sum(counts.values())
        lines.append(
            f"| {ds} | {counts['perfect']} | {counts['improved']} "
            f"| {counts['unchanged']} | {counts['degraded']} | {ds_total} |"
        )
    lines.append("")

    # Word pairs
    lines.append("## Top OCR Word Errors (by frequency)")
    lines.append("")
    lines.append(f"Total unique OCR->GT word pairs: **{len(ocr_word_pairs)}**")
    lines.append("")
    lines.append("| Rank | OCR Word | GT Word | Count |")
    lines.append("|-----:|----------|---------|------:|")
    for i, ((ocr_w, gt_w), count) in enumerate(ocr_word_pairs.most_common(30), 1):
        lines.append(f"| {i} | {ocr_w} | {gt_w} | {count} |")
    lines.append("")

    # LLM failures
    lines.append("## Top LLM Unfixed Errors")
    lines.append("")
    lines.append(f"Total unique unfixed pairs: **{len(llm_unfixed_pairs)}**")
    lines.append("")
    lines.append("| Rank | OCR/LLM Word | GT Word | Count |")
    lines.append("|-----:|--------------|---------|------:|")
    for i, ((w, gt_w), count) in enumerate(llm_unfixed_pairs.most_common(20), 1):
        lines.append(f"| {i} | {w} | {gt_w} | {count} |")
    lines.append("")

    lines.append("## Top LLM Introduced Errors")
    lines.append("")
    lines.append(f"Total unique introduced pairs: **{len(llm_introduced_pairs)}**")
    lines.append("")
    lines.append("| Rank | LLM Word | GT Word | Count |")
    lines.append("|-----:|----------|---------|------:|")
    for i, ((w, gt_w), count) in enumerate(llm_introduced_pairs.most_common(20), 1):
        lines.append(f"| {i} | {w} | {gt_w} | {count} |")
    lines.append("")

    # Sentence failures
    lines.append("## Sentence-Level Failures")
    lines.append("")
    lines.append(f"- **Degraded** (LLM made worse): {degraded_count}")
    lines.append(f"- **Unchanged hard** (CER > 5% but LLM didn't help): {unchanged_hard_count}")
    lines.append(f"- See `sentence_failures.jsonl` for full details")
    lines.append("")

    # Error type stats
    if error_stats:
        lines.append("## Error Type Fix/Introduction Rates")
        lines.append("")
        for ds_type, stats in error_stats.items():
            overall = stats.get("overall", {})
            by_type = stats.get("by_type", {})
            lines.append(f"### {ds_type}")
            lines.append("")
            fix_r = overall.get("fix_rate")
            intro_r = overall.get("introduction_rate")
            lines.append(
                f"Overall fix rate: **{fix_r*100:.1f}%** | "
                f"Introduction rate: **{intro_r*100:.1f}%**"
                if fix_r is not None and intro_r is not None
                else "Overall stats not available."
            )
            lines.append("")
            lines.append("| Error Type | Baseline | Fixed | Introduced | Fix Rate | Intro Rate |")
            lines.append("|------------|--------:|---------:|----------:|---------:|----------:|")
            for etype, data in sorted(by_type.items(), key=lambda x: x[1].get("baseline", 0), reverse=True):
                b = data.get("baseline", 0)
                if b == 0:
                    continue
                f_rate = data.get("fix_rate")
                i_rate = data.get("introduction_rate")
                lines.append(
                    f"| {etype} | {b} | {data.get('fixed', 0)} | {data.get('introduced', 0)} "
                    f"| {f_rate*100:.1f}% | {i_rate*100:.1f}% |"
                    if f_rate is not None and i_rate is not None
                    else f"| {etype} | {b} | {data.get('fixed', 0)} | {data.get('introduced', 0)} | - | - |"
                )
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
