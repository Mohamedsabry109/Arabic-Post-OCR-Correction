"""Metric calculation for Arabic OCR evaluation: CER and WER.

Uses editdistance for character-level computation and jiwer for word-level.
All metrics are computed per-sample first, then aggregated (mean/std/median/p95).
Never pool characters across samples for CER.
"""

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

import editdistance

from src.data.text_utils import normalise_arabic, tokenise_arabic

if TYPE_CHECKING:
    from src.data.data_loader import OCRSample


@dataclass
class MetricResult:
    """Aggregated CER/WER scores for one dataset."""

    dataset: str
    num_samples: int
    num_chars_ref: int      # total GT characters across all samples
    num_words_ref: int      # total GT words across all samples
    cer: float              # mean CER across samples
    wer: float              # mean WER across samples
    cer_std: float          # standard deviation of per-sample CER
    wer_std: float          # standard deviation of per-sample WER
    cer_median: float
    wer_median: float
    cer_p95: float          # 95th percentile (captures worst-case samples)
    wer_p95: float

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        return {
            "dataset": self.dataset,
            "num_samples": self.num_samples,
            "num_chars_ref": self.num_chars_ref,
            "num_words_ref": self.num_words_ref,
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "cer_std": round(self.cer_std, 6),
            "wer_std": round(self.wer_std, 6),
            "cer_median": round(self.cer_median, 6),
            "wer_median": round(self.wer_median, 6),
            "cer_p95": round(self.cer_p95, 6),
            "wer_p95": round(self.wer_p95, 6),
            # Human-readable percentages
            "cer_pct": f"{self.cer * 100:.2f}%",
            "wer_pct": f"{self.wer * 100:.2f}%",
        }


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate using Levenshtein edit distance.

    Formula:
        CER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions,
              N = number of characters in the reference.

    Both strings are normalised before comparison (normalise_arabic with default
    settings — diacritics preserved, repetitions capped).

    Args:
        reference: Ground truth text.
        hypothesis: OCR prediction text.

    Returns:
        CER value ≥ 0.0. Can exceed 1.0 if hypothesis is much longer.
        Returns 0.0 if reference is empty.

    Example:
        >>> calculate_cer("مرحبا", "مرحيا")
        0.2
        >>> calculate_cer("", "أي شيء")
        0.0
    """
    ref = normalise_arabic(reference)
    hyp = normalise_arabic(hypothesis)

    if len(ref) == 0:
        return 0.0

    distance = editdistance.eval(ref, hyp)
    return distance / len(ref)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using word-level Levenshtein edit distance.

    Tokenises both strings with tokenise_arabic() before comparison.

    Args:
        reference: Ground truth text.
        hypothesis: OCR prediction text.

    Returns:
        WER value ≥ 0.0. Returns 0.0 if reference has no words.

    Example:
        >>> calculate_wer("الكتاب على الطاولة", "الكتب على الطاولة")
        0.333...
    """
    ref_words = tokenise_arabic(normalise_arabic(reference))
    hyp_words = tokenise_arabic(normalise_arabic(hypothesis))

    if len(ref_words) == 0:
        return 0.0

    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words)


def calculate_metrics(
    samples: list["OCRSample"],
    dataset_name: str,
    normalise: bool = True,
) -> MetricResult:
    """Calculate CER and WER over a list of OCRSample objects.

    Computes per-sample scores, then aggregates with mean/std/median/p95.

    Args:
        samples: List of OCRSample with ocr_text and gt_text populated.
        dataset_name: Label for the MetricResult (e.g., "KHATT-train").
        normalise: If True, apply normalise_arabic() before scoring.
                   Should always be True; exposed for testing only.

    Returns:
        Aggregated MetricResult. Returns zero MetricResult if samples is empty.
    """
    if not samples:
        return MetricResult(
            dataset=dataset_name,
            num_samples=0,
            num_chars_ref=0,
            num_words_ref=0,
            cer=0.0, wer=0.0,
            cer_std=0.0, wer_std=0.0,
            cer_median=0.0, wer_median=0.0,
            cer_p95=0.0, wer_p95=0.0,
        )

    per_cer: list[float] = []
    per_wer: list[float] = []
    total_chars = 0
    total_words = 0

    for sample in samples:
        ref = normalise_arabic(sample.gt_text) if normalise else sample.gt_text
        hyp = normalise_arabic(sample.ocr_text) if normalise else sample.ocr_text

        total_chars += len(ref)
        total_words += len(tokenise_arabic(ref))

        per_cer.append(calculate_cer(sample.gt_text, sample.ocr_text))
        per_wer.append(calculate_wer(sample.gt_text, sample.ocr_text))

    return MetricResult(
        dataset=dataset_name,
        num_samples=len(samples),
        num_chars_ref=total_chars,
        num_words_ref=total_words,
        cer=statistics.mean(per_cer),
        wer=statistics.mean(per_wer),
        cer_std=statistics.stdev(per_cer) if len(per_cer) > 1 else 0.0,
        wer_std=statistics.stdev(per_wer) if len(per_wer) > 1 else 0.0,
        cer_median=statistics.median(per_cer),
        wer_median=statistics.median(per_wer),
        cer_p95=_percentile(per_cer, 95),
        wer_p95=_percentile(per_wer, 95),
    )


def calculate_metrics_split(
    samples: list["OCRSample"],
    dataset_name: str,
    runaway_ratio_threshold: float = 5.0,
) -> tuple["MetricResult", "MetricResult", dict]:
    """Calculate metrics for all samples AND for non-runaway samples only.

    A sample is "runaway" when its OCR length exceeds *runaway_ratio_threshold*
    times the GT length (Qaari infinite-repetition failure mode).

    Args:
        samples: List of OCRSample objects.
        dataset_name: Dataset label.
        runaway_ratio_threshold: OCR/GT length ratio above which a sample is
            classified as runaway.

    Returns:
        Tuple of:
          - MetricResult for ALL samples
          - MetricResult for NORMAL (non-runaway) samples only
          - data_quality dict with runaway statistics
    """
    normal_samples = []
    runaway_samples = []

    for s in samples:
        gt_len = max(len(normalise_arabic(s.gt_text)), 1)
        ocr_len = len(normalise_arabic(s.ocr_text))
        if ocr_len / gt_len > runaway_ratio_threshold:
            runaway_samples.append(s)
        else:
            normal_samples.append(s)

    all_metrics = calculate_metrics(samples, dataset_name)
    normal_metrics = calculate_metrics(normal_samples, f"{dataset_name}-normal")

    data_quality = {
        "total_samples": len(samples),
        "normal_samples": len(normal_samples),
        "runaway_samples": len(runaway_samples),
        "runaway_percentage": round(len(runaway_samples) / max(len(samples), 1) * 100, 2),
        "runaway_ratio_threshold": runaway_ratio_threshold,
        "description": (
            f"{len(runaway_samples)} samples ({len(runaway_samples)/max(len(samples),1)*100:.1f}%) "
            f"have OCR output >{runaway_ratio_threshold}x longer than GT (Qaari repetition bug)."
        ),
    }

    return all_metrics, normal_metrics, data_quality


def compare_metrics(baseline: MetricResult, improved: MetricResult) -> dict:
    """Compute delta statistics between two MetricResult objects.

    Positive values mean the *improved* result is better (lower error).

    Args:
        baseline: The reference/baseline MetricResult.
        improved: The result to compare against baseline.

    Returns:
        Dict with absolute and relative improvement for CER and WER.
    """
    cer_delta = baseline.cer - improved.cer          # positive = improved
    wer_delta = baseline.wer - improved.wer

    cer_rel = (cer_delta / baseline.cer * 100) if baseline.cer > 0 else 0.0
    wer_rel = (wer_delta / baseline.wer * 100) if baseline.wer > 0 else 0.0

    return {
        "baseline_dataset": baseline.dataset,
        "improved_dataset": improved.dataset,
        "cer_delta": round(cer_delta, 6),
        "wer_delta": round(wer_delta, 6),
        "cer_relative_improvement_pct": round(cer_rel, 2),
        "wer_relative_improvement_pct": round(wer_rel, 2),
        "cer_baseline": round(baseline.cer, 6),
        "cer_improved": round(improved.cer, 6),
        "wer_baseline": round(baseline.wer, 6),
        "wer_improved": round(improved.wer, 6),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: int) -> float:
    """Return the p-th percentile of *data* (nearest-rank method)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    # Nearest rank: ceil(pct/100 * n) - 1 (0-indexed)
    k = max(0, min(n - 1, int((pct / 100.0) * n)))
    return sorted_data[k]
