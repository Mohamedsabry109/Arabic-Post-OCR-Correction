#!/usr/bin/env python3
"""
QALB Corpus Error-Correction Pair Extractor.

Processes the QALB (Qatar Arabic Language Bank) parallel corpus to extract
error-correction pairs with context for few-shot learning examples.

Usage:
    python scripts/extract_qalb.py --qalb-path path/to/qalb
    python scripts/extract_qalb.py --qalb-path data/qalb --few-shot 20

Example Output:
    - qalb_corrections.json: All extracted corrections with context
    - qalb_few_shot.json: Selected diverse examples for few-shot prompting

Runtime Estimate:
    ~1-3 minutes depending on corpus size
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Error type classification patterns
ERROR_PATTERNS = {
    'spelling': [
        (r'[أإآا]', r'[أإآا]'),  # Hamza/Alef confusion
        (r'[هة]', r'[هة]'),      # Taa Marbuta confusion
        (r'[يى]', r'[يى]'),      # Ya/Alef Maksura confusion
    ],
    'punctuation': [
        (r'[،؛:!؟\.\,\;\?\!]', r'[،؛:!؟\.\,\;\?\!]'),
    ],
    'spacing': [
        (r'\s+', r'\s*'),
    ],
    'diacritics': [
        (r'[ًٌٍَُِّْ]', r'[ًٌٍَُِّْ]?'),
    ],
}


@dataclass
class CorrectionPair:
    """A single error-correction pair with context."""
    source: str              # Original (erroneous) text
    target: str              # Corrected text
    error_type: str          # Classification of error
    context_before: str      # Words before the error
    context_after: str       # Words after the error
    source_sentence: str     # Full source sentence
    target_sentence: str     # Full target sentence
    position: int            # Position in sentence
    sentence_id: int         # Sentence identifier


@dataclass
class QALBStatistics:
    """Statistics about the QALB extraction."""
    total_sentences: int = 0
    sentences_with_errors: int = 0
    total_corrections: int = 0
    error_type_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.error_type_counts is None:
            self.error_type_counts = defaultdict(int)


def classify_error_type(source: str, target: str) -> str:
    """
    Classify the type of error based on source and target.

    Args:
        source: Original (erroneous) text.
        target: Corrected text.

    Returns:
        Error type string.

    Example:
        >>> classify_error_type("انا", "أنا")
        'hamza'
        >>> classify_error_type("كتاب.", "كتاب")
        'punctuation'
    """
    # Check for hamza/alef errors
    alef_variants = set('أإآا')
    if any(c in alef_variants for c in source) or any(c in alef_variants for c in target):
        source_alefs = [c for c in source if c in alef_variants]
        target_alefs = [c for c in target if c in alef_variants]
        if source_alefs != target_alefs:
            return 'hamza'

    # Check for taa marbuta
    if ('ه' in source and 'ة' in target) or ('ة' in source and 'ه' in target):
        return 'taa_marbuta'

    # Check for ya/alef maksura
    if ('ي' in source and 'ى' in target) or ('ى' in source and 'ي' in target):
        return 'alef_maksura'

    # Check for diacritics
    diacritics = set('ًٌٍَُِّْ')
    source_no_dia = ''.join(c for c in source if c not in diacritics)
    target_no_dia = ''.join(c for c in target if c not in diacritics)
    if source_no_dia == target_no_dia and source != target:
        return 'diacritics'

    # Check for spacing
    if source.replace(' ', '') == target.replace(' ', ''):
        return 'spacing'

    # Check for punctuation
    punct = set('،؛:!؟.,;?!')
    source_no_punct = ''.join(c for c in source if c not in punct)
    target_no_punct = ''.join(c for c in target if c not in punct)
    if source_no_punct == target_no_punct:
        return 'punctuation'

    # Check for word-level errors
    source_words = source.split()
    target_words = target.split()
    if len(source_words) != len(target_words):
        return 'word_boundary'

    # Default to spelling
    return 'spelling'


def get_context(words: List[str], position: int, window: int = 2) -> Tuple[str, str]:
    """
    Get context words around a position.

    Args:
        words: List of words.
        position: Index of the target word.
        window: Number of context words on each side.

    Returns:
        Tuple of (context_before, context_after).
    """
    start = max(0, position - window)
    end = min(len(words), position + window + 1)

    context_before = ' '.join(words[start:position])
    context_after = ' '.join(words[position + 1:end])

    return context_before, context_after


def align_words(source_words: List[str], target_words: List[str]) -> List[Tuple[int, int, str, str]]:
    """
    Align source and target words to find differences.

    Uses simple position-based alignment with edit distance for mismatches.

    Args:
        source_words: List of source words.
        target_words: List of target words.

    Returns:
        List of (source_idx, target_idx, source_word, target_word) for differences.
    """
    differences = []

    # Simple case: same number of words
    if len(source_words) == len(target_words):
        for i, (sw, tw) in enumerate(zip(source_words, target_words)):
            if sw != tw:
                differences.append((i, i, sw, tw))
        return differences

    # Use DP alignment for different lengths
    m, n = len(source_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )

    # Backtrack
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source_words[i-1] == target_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            differences.append((i-1, j-1, source_words[i-1], target_words[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            differences.append((i-1, -1, source_words[i-1], '<DEL>'))
            i -= 1
        else:
            differences.append((-1, j-1, '<INS>', target_words[j-1]))
            j -= 1

    return list(reversed(differences))


def extract_qalb_corrections(
    qalb_path: str,
    context_window: int = 2
) -> Tuple[List[CorrectionPair], QALBStatistics]:
    """
    Extract error-correction pairs from QALB corpus.

    Args:
        qalb_path: Path to QALB corpus directory.
        context_window: Words of context on each side.

    Returns:
        Tuple of (list of CorrectionPairs, statistics).

    Example:
        >>> corrections, stats = extract_qalb_corrections("data/qalb")
        >>> print(f"Found {len(corrections)} corrections")
    """
    qalb_path = Path(qalb_path)

    if not qalb_path.exists():
        raise FileNotFoundError(f"QALB path not found: {qalb_path}")

    corrections = []
    stats = QALBStatistics()

    # Find source and target files
    # QALB typically has .src and .trg or source.txt/target.txt pairs
    source_files = list(qalb_path.glob('**/*.src')) + \
                   list(qalb_path.glob('**/source*.txt')) + \
                   list(qalb_path.glob('**/*source*.txt'))
    target_files = list(qalb_path.glob('**/*.trg')) + \
                   list(qalb_path.glob('**/target*.txt')) + \
                   list(qalb_path.glob('**/*target*.txt'))

    # Also check for column-based format (source TAB target)
    combined_files = list(qalb_path.glob('**/*.tsv')) + \
                     list(qalb_path.glob('**/*.parallel'))

    logger.info(f"Found {len(source_files)} source files")
    logger.info(f"Found {len(target_files)} target files")
    logger.info(f"Found {len(combined_files)} combined files")

    # Process parallel files (source/target pairs)
    for source_file in tqdm(source_files, desc="Processing parallel files"):
        # Find matching target file
        target_file = None

        # Try different naming conventions
        for ext in ['.trg', '_target.txt', '.target']:
            candidate = source_file.with_suffix(ext)
            if candidate.exists():
                target_file = candidate
                break

        # Try replacing 'source' with 'target' in filename
        if target_file is None:
            target_name = source_file.name.replace('source', 'target').replace('.src', '.trg')
            target_file = source_file.parent / target_name
            if not target_file.exists():
                target_file = None

        if target_file is None:
            logger.warning(f"No target file found for: {source_file}")
            continue

        # Read and process
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            with open(target_file, 'r', encoding='utf-8') as f:
                target_lines = f.readlines()

            if len(source_lines) != len(target_lines):
                logger.warning(
                    f"Line count mismatch: {source_file} ({len(source_lines)}) "
                    f"vs {target_file} ({len(target_lines)})"
                )
                continue

            for i, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
                source_sent = source_line.strip()
                target_sent = target_line.strip()

                if not source_sent or not target_sent:
                    continue

                stats.total_sentences += 1

                if source_sent != target_sent:
                    stats.sentences_with_errors += 1

                    # Extract word-level differences
                    source_words = source_sent.split()
                    target_words = target_sent.split()
                    differences = align_words(source_words, target_words)

                    for src_idx, tgt_idx, src_word, tgt_word in differences:
                        if src_word == '<INS>' or tgt_word == '<DEL>':
                            continue  # Skip insertions/deletions for now

                        error_type = classify_error_type(src_word, tgt_word)
                        stats.error_type_counts[error_type] += 1
                        stats.total_corrections += 1

                        ctx_before, ctx_after = get_context(
                            source_words, src_idx, context_window
                        )

                        correction = CorrectionPair(
                            source=src_word,
                            target=tgt_word,
                            error_type=error_type,
                            context_before=ctx_before,
                            context_after=ctx_after,
                            source_sentence=source_sent,
                            target_sentence=target_sent,
                            position=src_idx,
                            sentence_id=stats.total_sentences
                        )
                        corrections.append(correction)

        except Exception as e:
            logger.warning(f"Error processing {source_file}: {e}")
            continue

    # Process combined (TSV) files
    for combined_file in tqdm(combined_files, desc="Processing combined files"):
        try:
            with open(combined_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue

                    source_sent = parts[0].strip()
                    target_sent = parts[1].strip()

                    if not source_sent or not target_sent:
                        continue

                    stats.total_sentences += 1

                    if source_sent != target_sent:
                        stats.sentences_with_errors += 1

                        source_words = source_sent.split()
                        target_words = target_sent.split()
                        differences = align_words(source_words, target_words)

                        for src_idx, tgt_idx, src_word, tgt_word in differences:
                            if src_word == '<INS>' or tgt_word == '<DEL>':
                                continue

                            error_type = classify_error_type(src_word, tgt_word)
                            stats.error_type_counts[error_type] += 1
                            stats.total_corrections += 1

                            ctx_before, ctx_after = get_context(
                                source_words, src_idx, context_window
                            )

                            correction = CorrectionPair(
                                source=src_word,
                                target=tgt_word,
                                error_type=error_type,
                                context_before=ctx_before,
                                context_after=ctx_after,
                                source_sentence=source_sent,
                                target_sentence=target_sent,
                                position=src_idx,
                                sentence_id=stats.total_sentences
                            )
                            corrections.append(correction)

        except Exception as e:
            logger.warning(f"Error processing {combined_file}: {e}")
            continue

    return corrections, stats


def select_few_shot_examples(
    corrections: List[CorrectionPair],
    n: int = 20,
    seed: int = 42
) -> List[CorrectionPair]:
    """
    Select diverse few-shot examples from corrections.

    Ensures coverage of different error types and contexts.

    Args:
        corrections: List of all correction pairs.
        n: Number of examples to select.
        seed: Random seed for reproducibility.

    Returns:
        List of selected diverse examples.

    Example:
        >>> examples = select_few_shot_examples(corrections, n=20)
        >>> for ex in examples:
        ...     print(f"{ex.source} -> {ex.target} ({ex.error_type})")
    """
    if len(corrections) <= n:
        return corrections

    random.seed(seed)

    # Group by error type
    by_type: Dict[str, List[CorrectionPair]] = defaultdict(list)
    for corr in corrections:
        by_type[corr.error_type].append(corr)

    selected = []
    error_types = list(by_type.keys())

    # First, ensure at least one example of each type
    for error_type in error_types:
        if by_type[error_type] and len(selected) < n:
            # Select one with good context
            candidates = [
                c for c in by_type[error_type]
                if c.context_before and c.context_after
            ]
            if candidates:
                selected.append(random.choice(candidates))
            else:
                selected.append(random.choice(by_type[error_type]))

    # Fill remaining slots with diverse examples
    remaining = n - len(selected)
    if remaining > 0:
        # Flatten all corrections not yet selected
        selected_ids = {(c.source, c.target, c.sentence_id) for c in selected}
        remaining_corrections = [
            c for c in corrections
            if (c.source, c.target, c.sentence_id) not in selected_ids
        ]

        # Prefer examples with context
        with_context = [
            c for c in remaining_corrections
            if c.context_before and c.context_after
        ]

        if len(with_context) >= remaining:
            additional = random.sample(with_context, remaining)
        else:
            additional = with_context + random.sample(
                [c for c in remaining_corrections if c not in with_context],
                min(remaining - len(with_context), len(remaining_corrections) - len(with_context))
            )

        selected.extend(additional)

    return selected[:n]


def save_corrections(
    corrections: List[CorrectionPair],
    statistics: QALBStatistics,
    output_path: Path
) -> None:
    """Save all corrections to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'total_sentences': statistics.total_sentences,
            'sentences_with_errors': statistics.sentences_with_errors,
            'total_corrections': statistics.total_corrections,
            'error_type_counts': dict(statistics.error_type_counts),
            'created_at': datetime.now().isoformat(),
        },
        'corrections': [asdict(c) for c in corrections]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(corrections)} corrections to {output_path}")


def save_few_shot(examples: List[CorrectionPair], output_path: Path) -> None:
    """Save few-shot examples to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'num_examples': len(examples),
            'error_types': list(set(ex.error_type for ex in examples)),
            'created_at': datetime.now().isoformat(),
        },
        'examples': [asdict(ex) for ex in examples]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(examples)} few-shot examples to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract error-correction pairs from QALB corpus"
    )
    parser.add_argument(
        '--qalb-path', '-q',
        required=True,
        help='Path to QALB corpus directory'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='data',
        help='Output directory'
    )
    parser.add_argument(
        '--few-shot', '-n',
        type=int,
        default=20,
        help='Number of few-shot examples to select (default: 20)'
    )
    parser.add_argument(
        '--context-window', '-w',
        type=int,
        default=2,
        help='Context window size in words (default: 2)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("QALB Corpus Processor")
    logger.info("=" * 60)

    try:
        # Extract corrections
        corrections, stats = extract_qalb_corrections(
            args.qalb_path,
            context_window=args.context_window
        )

        if not corrections:
            logger.warning("No corrections extracted. Check corpus format.")
            sys.exit(1)

        # Save all corrections
        output_dir = Path(args.output_dir)
        save_corrections(corrections, stats, output_dir / 'qalb_corrections.json')

        # Select and save few-shot examples
        few_shot = select_few_shot_examples(corrections, n=args.few_shot)
        save_few_shot(few_shot, output_dir / 'qalb_few_shot.json')

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("QALB EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total sentences: {stats.total_sentences:,}")
        logger.info(f"Sentences with errors: {stats.sentences_with_errors:,}")
        logger.info(f"Total corrections: {stats.total_corrections:,}")

        logger.info("\nError type distribution:")
        for error_type, count in sorted(
            stats.error_type_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  {error_type}: {count:,}")

        logger.info(f"\nFew-shot examples selected: {len(few_shot)}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
