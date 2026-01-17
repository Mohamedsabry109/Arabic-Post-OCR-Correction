"""
Metrics calculator for Arabic Post-OCR Correction evaluation.

This module implements Character Error Rate (CER) and Word Error Rate (WER)
metrics using Levenshtein distance for evaluating OCR correction quality.

Example:
    >>> from src.metrics import calculate_cer, calculate_wer, calculate_metrics
    >>> reference = "مرحباً بالعالم"
    >>> hypothesis = "مرحبا بالعالم"
    >>> cer = calculate_cer(reference, hypothesis)
    >>> wer = calculate_wer(reference, hypothesis)
    >>> print(f"CER: {cer:.2%}, WER: {wer:.2%}")
"""

import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for evaluation metrics results."""
    cer: float
    wer: float
    total_samples: int
    total_ref_chars: int
    total_ref_words: int
    total_char_errors: int
    total_word_errors: int


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Uses python-Levenshtein library if available for performance,
    otherwise falls back to a pure Python implementation.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Integer edit distance between the strings.

    Example:
        >>> _levenshtein_distance("hello", "hallo")
        1
        >>> _levenshtein_distance("مرحبا", "مرحباً")
        1
    """
    if HAS_LEVENSHTEIN:
        return Levenshtein.distance(s1, s2)

    # Pure Python fallback using dynamic programming
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True
) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.

    CER = (S + D + I) / N
    Where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of characters in reference

    Args:
        reference: Ground truth text.
        hypothesis: Predicted/OCR text.
        normalize: If True, normalizes whitespace before comparison.
            Defaults to True.

    Returns:
        CER as a float between 0 and potentially > 1
        (if hypothesis is much longer than reference).
        Returns 0.0 if both texts are empty.
        Returns 1.0 if reference is empty but hypothesis is not.

    Example:
        >>> calculate_cer("مرحباً بالعالم", "مرحبا بالعالم")
        0.0625
        >>> calculate_cer("hello", "helo")
        0.2
    """
    # Handle edge cases
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0  # 100% error if reference is empty but hypothesis isn't
    if not hypothesis:
        return 1.0  # 100% error if hypothesis is empty

    # Normalize whitespace if requested
    if normalize:
        reference = ' '.join(reference.split())
        hypothesis = ' '.join(hypothesis.split())

    # Calculate Levenshtein distance at character level
    distance = _levenshtein_distance(reference, hypothesis)

    # CER = edit_distance / reference_length
    cer = distance / len(reference)

    return cer


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True
) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    WER = (S + D + I) / N
    Where:
        S = number of word substitutions
        D = number of word deletions
        I = number of word insertions
        N = number of words in reference

    Args:
        reference: Ground truth text.
        hypothesis: Predicted/OCR text.
        normalize: If True, normalizes whitespace before comparison.
            Defaults to True.

    Returns:
        WER as a float between 0 and potentially > 1.
        Returns 0.0 if both texts are empty.
        Returns 1.0 if reference is empty but hypothesis is not.

    Example:
        >>> calculate_wer("مرحباً بالعالم", "مرحبا بالعالم")
        0.5
        >>> calculate_wer("hello world", "hello word")
        0.5
    """
    # Handle edge cases
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    if not hypothesis:
        return 1.0

    # Normalize and split into words
    if normalize:
        reference = ' '.join(reference.split())
        hypothesis = ' '.join(hypothesis.split())

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Handle empty word lists
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    if not hyp_words:
        return 1.0

    # Calculate Levenshtein distance at word level
    # Join with a character unlikely to appear in Arabic text
    ref_str = '\x00'.join(ref_words)
    hyp_str = '\x00'.join(hyp_words)

    # Word-level distance using standard Levenshtein on word sequences
    distance = _word_levenshtein_distance(ref_words, hyp_words)

    # WER = edit_distance / reference_word_count
    wer = distance / len(ref_words)

    return wer


def _word_levenshtein_distance(ref_words: List[str], hyp_words: List[str]) -> int:
    """
    Calculate Levenshtein distance at the word level.

    Args:
        ref_words: List of reference words.
        hyp_words: List of hypothesis words.

    Returns:
        Word-level edit distance.
    """
    m, n = len(ref_words), len(hyp_words)

    # Create distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first column (deletions)
    for i in range(m + 1):
        dp[i][0] = i

    # Initialize first row (insertions)
    for j in range(n + 1):
        dp[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1   # Substitution
                )

    return dp[m][n]


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate aggregate CER and WER metrics for multiple samples.

    Computes both micro-averaged (total errors / total length) and
    macro-averaged (mean of individual scores) metrics.

    Args:
        predictions: List of predicted/hypothesis texts.
        ground_truths: List of reference/ground truth texts.
        normalize: If True, normalizes whitespace before comparison.

    Returns:
        Dictionary containing:
        - cer: Micro-averaged Character Error Rate
        - wer: Micro-averaged Word Error Rate
        - cer_macro: Macro-averaged CER
        - wer_macro: Macro-averaged WER
        - total_samples: Number of samples
        - total_ref_chars: Total characters in references
        - total_ref_words: Total words in references

    Raises:
        ValueError: If predictions and ground_truths have different lengths.

    Example:
        >>> preds = ["مرحبا", "العالم"]
        >>> truths = ["مرحباً", "العالم"]
        >>> metrics = calculate_metrics(preds, truths)
        >>> print(f"CER: {metrics['cer']:.2%}")
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )

    if not predictions:
        return {
            'cer': 0.0,
            'wer': 0.0,
            'cer_macro': 0.0,
            'wer_macro': 0.0,
            'total_samples': 0,
            'total_ref_chars': 0,
            'total_ref_words': 0,
        }

    total_char_errors = 0
    total_word_errors = 0
    total_ref_chars = 0
    total_ref_words = 0
    cer_scores = []
    wer_scores = []

    for pred, truth in zip(predictions, ground_truths):
        # Normalize if requested
        if normalize:
            pred_norm = ' '.join(pred.split())
            truth_norm = ' '.join(truth.split())
        else:
            pred_norm = pred
            truth_norm = truth

        # Character-level metrics
        if truth_norm:
            char_distance = _levenshtein_distance(truth_norm, pred_norm)
            total_char_errors += char_distance
            total_ref_chars += len(truth_norm)
            cer_scores.append(char_distance / len(truth_norm))
        else:
            cer_scores.append(0.0 if not pred_norm else 1.0)

        # Word-level metrics
        truth_words = truth_norm.split()
        pred_words = pred_norm.split()

        if truth_words:
            word_distance = _word_levenshtein_distance(truth_words, pred_words)
            total_word_errors += word_distance
            total_ref_words += len(truth_words)
            wer_scores.append(word_distance / len(truth_words))
        else:
            wer_scores.append(0.0 if not pred_words else 1.0)

    # Calculate micro-averaged metrics (total errors / total length)
    micro_cer = total_char_errors / total_ref_chars if total_ref_chars > 0 else 0.0
    micro_wer = total_word_errors / total_ref_words if total_ref_words > 0 else 0.0

    # Calculate macro-averaged metrics (mean of individual scores)
    macro_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
    macro_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0

    return {
        'cer': micro_cer,
        'wer': micro_wer,
        'cer_macro': macro_cer,
        'wer_macro': macro_wer,
        'total_samples': len(predictions),
        'total_ref_chars': total_ref_chars,
        'total_ref_words': total_ref_words,
        'total_char_errors': total_char_errors,
        'total_word_errors': total_word_errors,
    }


def calculate_improvement(
    baseline_metrics: Dict[str, float],
    improved_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate improvement percentages between baseline and improved metrics.

    Args:
        baseline_metrics: Metrics before correction.
        improved_metrics: Metrics after correction.

    Returns:
        Dictionary with improvement percentages for CER and WER.
        Positive values indicate improvement (error reduction).

    Example:
        >>> baseline = {'cer': 0.20, 'wer': 0.30}
        >>> improved = {'cer': 0.15, 'wer': 0.25}
        >>> improvement = calculate_improvement(baseline, improved)
        >>> print(f"CER improved by {improvement['cer_improvement']:.1%}")
        CER improved by 25.0%
    """
    cer_baseline = baseline_metrics.get('cer', 0)
    wer_baseline = baseline_metrics.get('wer', 0)
    cer_improved = improved_metrics.get('cer', 0)
    wer_improved = improved_metrics.get('wer', 0)

    # Calculate relative improvement (reduction in error rate)
    cer_improvement = (
        (cer_baseline - cer_improved) / cer_baseline
        if cer_baseline > 0 else 0.0
    )
    wer_improvement = (
        (wer_baseline - wer_improved) / wer_baseline
        if wer_baseline > 0 else 0.0
    )

    # Absolute improvement
    cer_abs_improvement = cer_baseline - cer_improved
    wer_abs_improvement = wer_baseline - wer_improved

    return {
        'cer_improvement': cer_improvement,
        'wer_improvement': wer_improvement,
        'cer_absolute_improvement': cer_abs_improvement,
        'wer_absolute_improvement': wer_abs_improvement,
        'cer_baseline': cer_baseline,
        'cer_improved': cer_improved,
        'wer_baseline': wer_baseline,
        'wer_improved': wer_improved,
    }


def format_metrics(
    metrics: Dict[str, float],
    title: str = "Metrics",
    include_totals: bool = False
) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metrics.
        title: Title for the metrics block.
        include_totals: Include total counts in output.

    Returns:
        Formatted string representation.

    Example:
        >>> metrics = {'cer': 0.15, 'wer': 0.25}
        >>> print(format_metrics(metrics, "Baseline"))
        === Baseline ===
        CER: 15.00%
        WER: 25.00%
    """
    lines = [f"=== {title} ==="]

    if 'cer' in metrics:
        lines.append(f"CER: {metrics['cer']:.2%}")
    if 'wer' in metrics:
        lines.append(f"WER: {metrics['wer']:.2%}")

    if 'cer_macro' in metrics:
        lines.append(f"CER (macro): {metrics['cer_macro']:.2%}")
    if 'wer_macro' in metrics:
        lines.append(f"WER (macro): {metrics['wer_macro']:.2%}")

    if include_totals:
        if 'total_samples' in metrics:
            lines.append(f"Total samples: {metrics['total_samples']}")
        if 'total_ref_chars' in metrics:
            lines.append(f"Total characters: {metrics['total_ref_chars']}")
        if 'total_ref_words' in metrics:
            lines.append(f"Total words: {metrics['total_ref_words']}")

    return '\n'.join(lines)
