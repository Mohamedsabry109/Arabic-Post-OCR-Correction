#!/usr/bin/env python3
"""
Qaari OCR Error Analyzer.

Analyzes OCR errors from Qaari output by comparing predictions with ground truth.
Builds confusion matrices, tracks positional errors, and categorizes error types.

Usage:
    python scripts/analyze_qaari_errors.py --dataset PATS-A01
    python scripts/analyze_qaari_errors.py --dataset PATS-A01 KHATT --output data/

Example Output:
    - confusion_matrix.json: Character-level confusion statistics
    - error_statistics.json: Aggregate error statistics and patterns

Runtime Estimate:
    ~1-2 minutes for 1000 samples depending on text length
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.utils import clean_text, normalize_arabic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Arabic character ranges
ARABIC_LETTERS = set('ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰ')
ARABIC_DIACRITICS = set('ًٌٍَُِّْٰ')


@dataclass
class CharConfusion:
    """Statistics for a single character confusion."""
    count: int = 0
    positions: Dict[str, int] = field(default_factory=lambda: {
        'start': 0, 'middle': 0, 'end': 0, 'isolated': 0
    })
    examples: List[str] = field(default_factory=list)


@dataclass
class ErrorStatistics:
    """Aggregate error statistics."""
    total_chars: int = 0
    total_errors: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    # By position
    start_errors: int = 0
    middle_errors: int = 0
    end_errors: int = 0

    # By character type
    letter_errors: int = 0
    diacritic_errors: int = 0
    punctuation_errors: int = 0
    space_errors: int = 0

    # Common patterns
    top_confusions: List[Dict] = field(default_factory=list)
    top_error_words: List[Dict] = field(default_factory=list)


def align_sequences(ref: str, hyp: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Align two sequences using dynamic programming (edit distance alignment).

    Returns list of (ref_char, hyp_char) tuples where None indicates insertion/deletion.

    Args:
        ref: Reference (ground truth) string.
        hyp: Hypothesis (OCR output) string.

    Returns:
        List of aligned character pairs.

    Example:
        >>> align_sequences("مرحبا", "مرحيا")
        [('م', 'م'), ('ر', 'ر'), ('ح', 'ح'), ('ب', 'ي'), ('ا', 'ا')]
    """
    m, n = len(ref), len(hyp)

    # DP table for edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # Deletion
                    dp[i][j-1] + 1,      # Insertion
                    dp[i-1][j-1] + 1     # Substitution
                )

    # Backtrack to find alignment
    alignment = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            alignment.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            alignment.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion (char in ref not in hyp)
            alignment.append((ref[i-1], None))
            i -= 1
        else:
            # Insertion (char in hyp not in ref)
            alignment.append((None, hyp[j-1]))
            j -= 1

    return list(reversed(alignment))


def get_char_position(word: str, char_idx: int) -> str:
    """
    Determine character position within a word.

    Args:
        word: The word containing the character.
        char_idx: Index of character in the word.

    Returns:
        Position string: 'start', 'middle', 'end', or 'isolated'.
    """
    if len(word) == 1:
        return 'isolated'
    elif char_idx == 0:
        return 'start'
    elif char_idx == len(word) - 1:
        return 'end'
    else:
        return 'middle'


def get_char_type(char: str) -> str:
    """
    Classify character type.

    Args:
        char: Character to classify.

    Returns:
        Type string: 'letter', 'diacritic', 'punctuation', 'space', or 'other'.
    """
    if char in ARABIC_DIACRITICS:
        return 'diacritic'
    elif char in ARABIC_LETTERS:
        return 'letter'
    elif char.isspace():
        return 'space'
    elif not char.isalnum():
        return 'punctuation'
    else:
        return 'other'


class QaariErrorAnalyzer:
    """
    Analyze OCR errors from Qaari output.

    Builds confusion matrices and tracks error patterns.

    Example:
        >>> analyzer = QaariErrorAnalyzer()
        >>> analyzer.add_pair("مرحباً", "مرحيا")
        >>> stats = analyzer.get_statistics()
        >>> print(f"Total errors: {stats.total_errors}")
    """

    def __init__(self, max_examples: int = 5):
        """
        Initialize analyzer.

        Args:
            max_examples: Maximum examples to store per confusion type.
        """
        self.max_examples = max_examples

        # Confusion matrix: {true_char: {ocr_char: CharConfusion}}
        self.confusion_matrix: Dict[str, Dict[str, CharConfusion]] = defaultdict(
            lambda: defaultdict(CharConfusion)
        )

        # Word-level errors
        self.error_words: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # {ground_truth_word: {ocr_word: count}}

        # Statistics
        self.stats = ErrorStatistics()

        # Track processed pairs
        self.num_pairs = 0

    def add_pair(self, ground_truth: str, ocr_text: str) -> None:
        """
        Add a ground truth / OCR pair for analysis.

        Args:
            ground_truth: Reference text.
            ocr_text: OCR output text.
        """
        self.num_pairs += 1

        # Word-level analysis
        self._analyze_words(ground_truth, ocr_text)

        # Character-level analysis
        self._analyze_characters(ground_truth, ocr_text)

    def _analyze_words(self, ground_truth: str, ocr_text: str) -> None:
        """Analyze word-level errors."""
        gt_words = ground_truth.split()
        ocr_words = ocr_text.split()

        # Align words
        word_alignment = align_sequences(gt_words, ocr_words)

        for gt_word, ocr_word in word_alignment:
            if gt_word is not None and ocr_word is not None:
                if gt_word != ocr_word:
                    self.error_words[gt_word][ocr_word] += 1

    def _analyze_characters(self, ground_truth: str, ocr_text: str) -> None:
        """Analyze character-level errors with positional tracking."""
        # Get character alignment
        alignment = align_sequences(ground_truth, ocr_text)

        # Track position in ground truth for word position analysis
        gt_pos = 0
        gt_words = ground_truth.split()
        word_boundaries = self._get_word_boundaries(ground_truth)

        for ref_char, hyp_char in alignment:
            self.stats.total_chars += 1 if ref_char else 0

            # Determine position within word
            position = self._get_position_for_index(gt_pos, word_boundaries, ground_truth)

            if ref_char == hyp_char:
                # Correct - no error
                if ref_char is not None:
                    gt_pos += 1
                continue

            # Error occurred
            self.stats.total_errors += 1

            # Classify error type
            if ref_char is None:
                # Insertion
                self.stats.insertions += 1
                ref_char = '<INS>'
            elif hyp_char is None:
                # Deletion
                self.stats.deletions += 1
                hyp_char = '<DEL>'
            else:
                # Substitution
                self.stats.substitutions += 1

            # Update confusion matrix
            confusion = self.confusion_matrix[ref_char][hyp_char]
            confusion.count += 1
            confusion.positions[position] += 1

            # Store example
            if len(confusion.examples) < self.max_examples:
                context = self._get_context(ground_truth, gt_pos, window=10)
                confusion.examples.append(context)

            # Update position statistics
            if position == 'start':
                self.stats.start_errors += 1
            elif position == 'middle':
                self.stats.middle_errors += 1
            elif position == 'end':
                self.stats.end_errors += 1

            # Update character type statistics
            char_type = get_char_type(ref_char if ref_char != '<INS>' else hyp_char)
            if char_type == 'letter':
                self.stats.letter_errors += 1
            elif char_type == 'diacritic':
                self.stats.diacritic_errors += 1
            elif char_type == 'punctuation':
                self.stats.punctuation_errors += 1
            elif char_type == 'space':
                self.stats.space_errors += 1

            if ref_char not in ('<INS>', '<DEL>'):
                gt_pos += 1

    def _get_word_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Get start and end indices of each word."""
        boundaries = []
        pos = 0
        for word in text.split():
            start = text.find(word, pos)
            end = start + len(word)
            boundaries.append((start, end))
            pos = end
        return boundaries

    def _get_position_for_index(
        self, idx: int,
        word_boundaries: List[Tuple[int, int]],
        text: str
    ) -> str:
        """Determine position type for a character index."""
        for start, end in word_boundaries:
            if start <= idx < end:
                word_len = end - start
                if word_len == 1:
                    return 'isolated'
                elif idx == start:
                    return 'start'
                elif idx == end - 1:
                    return 'end'
                else:
                    return 'middle'
        return 'middle'  # Default

    def _get_context(self, text: str, pos: int, window: int = 10) -> str:
        """Get context around a position."""
        start = max(0, pos - window)
        end = min(len(text), pos + window + 1)
        return text[start:end]

    def get_confusion_matrix(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the confusion matrix as a serializable dictionary.

        Returns:
            Nested dictionary: {true_char: {ocr_char: {count, probability, positions, examples}}}
        """
        result = {}

        for true_char, confusions in self.confusion_matrix.items():
            # Calculate total for this true character
            total = sum(c.count for c in confusions.values())

            result[true_char] = {}
            for ocr_char, confusion in confusions.items():
                result[true_char][ocr_char] = {
                    'count': confusion.count,
                    'probability': confusion.count / total if total > 0 else 0,
                    'positions': dict(confusion.positions),
                    'examples': confusion.examples
                }

        return result

    def get_statistics(self) -> ErrorStatistics:
        """Get aggregate error statistics."""
        # Update top confusions
        all_confusions = []
        for true_char, confusions in self.confusion_matrix.items():
            for ocr_char, confusion in confusions.items():
                all_confusions.append({
                    'true_char': true_char,
                    'ocr_char': ocr_char,
                    'count': confusion.count
                })

        all_confusions.sort(key=lambda x: x['count'], reverse=True)
        self.stats.top_confusions = all_confusions[:50]

        # Update top error words
        all_word_errors = []
        for gt_word, ocr_variants in self.error_words.items():
            for ocr_word, count in ocr_variants.items():
                all_word_errors.append({
                    'ground_truth': gt_word,
                    'ocr_output': ocr_word,
                    'count': count
                })

        all_word_errors.sort(key=lambda x: x['count'], reverse=True)
        self.stats.top_error_words = all_word_errors[:100]

        return self.stats


def analyze_qaari_errors(
    dataset_names: List[str],
    data_loader: DataLoader,
    limit: Optional[int] = None
) -> Tuple[Dict, ErrorStatistics]:
    """
    Analyze Qaari OCR errors for specified datasets.

    Args:
        dataset_names: List of dataset names to analyze.
        data_loader: Initialized DataLoader instance.
        limit: Maximum samples per dataset (None for all).

    Returns:
        Tuple of (confusion_matrix, error_statistics).

    Example:
        >>> loader = DataLoader("data/Original", "data/Predictions")
        >>> matrix, stats = analyze_qaari_errors(["PATS-A01"], loader)
        >>> print(f"Error rate: {stats.total_errors / stats.total_chars:.2%}")
    """
    analyzer = QaariErrorAnalyzer()

    for dataset_name in dataset_names:
        logger.info(f"Analyzing dataset: {dataset_name}")

        try:
            pairs = data_loader.load_dataset(dataset_name, limit=limit)
        except FileNotFoundError as e:
            logger.warning(f"Dataset not found: {dataset_name} - {e}")
            continue

        for ocr_text, ground_truth in tqdm(pairs, desc=f"Analyzing {dataset_name}"):
            analyzer.add_pair(ground_truth, ocr_text)

    return analyzer.get_confusion_matrix(), analyzer.get_statistics()


def save_results(
    confusion_matrix: Dict,
    statistics: ErrorStatistics,
    output_dir: Path
) -> None:
    """
    Save analysis results to JSON files.

    Args:
        confusion_matrix: Character confusion matrix.
        statistics: Error statistics.
        output_dir: Directory to save outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    confusion_path = output_dir / 'confusion_matrix.json'
    with open(confusion_path, 'w', encoding='utf-8') as f:
        json.dump(confusion_matrix, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved confusion matrix to {confusion_path}")

    # Save statistics
    stats_dict = asdict(statistics)
    stats_path = output_dir / 'error_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved error statistics to {stats_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Qaari OCR errors"
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['PATS-A01', 'KHATT'],
        help='Datasets to analyze'
    )
    parser.add_argument(
        '--ground-truth', '-g',
        default='data/Original',
        help='Ground truth base directory'
    )
    parser.add_argument(
        '--predictions', '-p',
        default='data/Predictions',
        help='Predictions base directory'
    )
    parser.add_argument(
        '--output', '-o',
        default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit samples per dataset'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Qaari OCR Error Analysis")
    logger.info("=" * 60)

    # Initialize data loader
    try:
        data_loader = DataLoader(
            ground_truth_base=args.ground_truth,
            predictions_base=args.predictions
        )
    except FileNotFoundError as e:
        logger.error(f"Data directory not found: {e}")
        sys.exit(1)

    # Run analysis
    confusion_matrix, statistics = analyze_qaari_errors(
        args.datasets,
        data_loader,
        limit=args.limit
    )

    # Save results
    output_dir = Path(args.output)
    save_results(confusion_matrix, statistics, output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total characters: {statistics.total_chars:,}")
    logger.info(f"Total errors: {statistics.total_errors:,}")
    if statistics.total_chars > 0:
        logger.info(f"Error rate: {statistics.total_errors / statistics.total_chars:.2%}")
    logger.info(f"  Substitutions: {statistics.substitutions:,}")
    logger.info(f"  Insertions: {statistics.insertions:,}")
    logger.info(f"  Deletions: {statistics.deletions:,}")

    logger.info("\nTop 10 character confusions:")
    for conf in statistics.top_confusions[:10]:
        logger.info(f"  '{conf['true_char']}' -> '{conf['ocr_char']}': {conf['count']}")


if __name__ == '__main__':
    main()
