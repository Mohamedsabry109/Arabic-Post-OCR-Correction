#!/usr/bin/env python3
"""
Arabic N-gram Statistics Builder from OpenITI Corpus.

Extracts word bigrams and trigrams from the OpenITI corpus
and stores the most frequent ones with their counts.

Usage:
    python scripts/build_ngrams.py --corpus path/to/openiti
    python scripts/build_ngrams.py --corpus data/openiti --top-bigrams 50000 --top-trigrams 50000

Example Output:
    ngrams.json with structure:
    {
        "metadata": {...},
        "bigrams": {"من_الله": 12345, ...},
        "trigrams": {"في_سبيل_الله": 1234, ...}
    }

Runtime Estimate:
    ~10-20 minutes for 1GB of text depending on I/O speed
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Tuple, List, Optional
from datetime import datetime

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Arabic word pattern
ARABIC_WORD_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

# OpenITI metadata markers
OPENITI_MARKERS = {'#META#', '###', 'PageV', 'ms', '~~'}

# N-gram separator
NGRAM_SEP = '_'


def is_metadata_line(line: str) -> bool:
    """Check if a line is OpenITI metadata."""
    stripped = line.strip()
    if not stripped:
        return True

    for marker in OPENITI_MARKERS:
        if stripped.startswith(marker):
            return True

    if re.match(r'PageV\d+P\d+', stripped):
        return True

    return False


def extract_arabic_words(text: str) -> List[str]:
    """
    Extract Arabic words from text.

    Args:
        text: Input text.

    Returns:
        List of Arabic words.
    """
    words = []
    for match in ARABIC_WORD_PATTERN.finditer(text):
        word = match.group()
        if 2 <= len(word) <= 50:
            words.append(word)
    return words


def generate_ngrams(words: List[str], n: int) -> Iterator[str]:
    """
    Generate n-grams from a list of words.

    Args:
        words: List of words.
        n: N-gram size (2 for bigrams, 3 for trigrams).

    Yields:
        N-gram strings joined by separator.

    Example:
        >>> list(generate_ngrams(['الله', 'أكبر', 'من'], 2))
        ['الله_أكبر', 'أكبر_من']
    """
    if len(words) < n:
        return

    for i in range(len(words) - n + 1):
        ngram = NGRAM_SEP.join(words[i:i+n])
        yield ngram


def iter_corpus_files(corpus_path: Path) -> Iterator[Path]:
    """Iterate over text files in corpus directory."""
    extensions = {'.txt', '.ara', '.mARkdown'}

    for root, dirs, files in os.walk(corpus_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            filepath = Path(root) / filename
            if filepath.suffix.lower() in extensions:
                yield filepath


def process_file_for_ngrams(
    filepath: Path,
    bigram_counter: Counter,
    trigram_counter: Counter
) -> Tuple[int, int]:
    """
    Process a single file and update n-gram counters.

    Args:
        filepath: Path to text file.
        bigram_counter: Counter for bigrams.
        trigram_counter: Counter for trigrams.

    Returns:
        Tuple of (num_bigrams, num_trigrams) added.
    """
    encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']
    num_bigrams = 0
    num_trigrams = 0

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                # Process file in chunks (sentences/paragraphs)
                current_words = []

                for line in f:
                    if is_metadata_line(line):
                        # Process accumulated words before metadata break
                        if current_words:
                            for bigram in generate_ngrams(current_words, 2):
                                bigram_counter[bigram] += 1
                                num_bigrams += 1
                            for trigram in generate_ngrams(current_words, 3):
                                trigram_counter[trigram] += 1
                                num_trigrams += 1
                            current_words = []
                        continue

                    words = extract_arabic_words(line)
                    current_words.extend(words)

                    # Process when we have enough words
                    # (to avoid memory issues with very long files)
                    if len(current_words) > 1000:
                        for bigram in generate_ngrams(current_words, 2):
                            bigram_counter[bigram] += 1
                            num_bigrams += 1
                        for trigram in generate_ngrams(current_words, 3):
                            trigram_counter[trigram] += 1
                            num_trigrams += 1
                        # Keep last few words for continuity
                        current_words = current_words[-5:]

                # Process remaining words
                if current_words:
                    for bigram in generate_ngrams(current_words, 2):
                        bigram_counter[bigram] += 1
                        num_bigrams += 1
                    for trigram in generate_ngrams(current_words, 3):
                        trigram_counter[trigram] += 1
                        num_trigrams += 1

            return num_bigrams, num_trigrams

        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return 0, 0

    return 0, 0


def build_ngram_statistics(
    corpus_path: str,
    top_bigrams: int = 50000,
    top_trigrams: int = 50000,
    min_count: int = 2,
    show_progress: bool = True
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, any]]:
    """
    Build n-gram statistics from OpenITI corpus.

    Args:
        corpus_path: Path to OpenITI corpus directory.
        top_bigrams: Number of top bigrams to include.
        top_trigrams: Number of top trigrams to include.
        min_count: Minimum frequency to include n-gram.
        show_progress: Show progress bar.

    Returns:
        Tuple of (bigrams_dict, trigrams_dict, metadata_dict).

    Example:
        >>> bigrams, trigrams, meta = build_ngram_statistics("data/openiti")
        >>> print(f"Top bigram: {list(bigrams.keys())[0]}")
    """
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

    logger.info(f"Building n-gram statistics from: {corpus_path}")

    # Count files
    files = list(iter_corpus_files(corpus_path))
    logger.info(f"Found {len(files):,} text files")

    # Counters
    bigram_counter = Counter()
    trigram_counter = Counter()
    total_bigrams = 0
    total_trigrams = 0
    files_processed = 0

    # Process files
    iterator = tqdm(files, desc="Processing files") if show_progress else files

    for filepath in iterator:
        try:
            num_bi, num_tri = process_file_for_ngrams(
                filepath, bigram_counter, trigram_counter
            )
            total_bigrams += num_bi
            total_trigrams += num_tri
            files_processed += 1
        except Exception as e:
            logger.debug(f"Error processing {filepath}: {e}")

    logger.info(f"Processed {files_processed:,} files")
    logger.info(f"Total bigrams: {total_bigrams:,}")
    logger.info(f"Total trigrams: {total_trigrams:,}")
    logger.info(f"Unique bigrams: {len(bigram_counter):,}")
    logger.info(f"Unique trigrams: {len(trigram_counter):,}")

    # Filter by minimum count
    filtered_bigrams = {
        ng: count for ng, count in bigram_counter.items()
        if count >= min_count
    }
    filtered_trigrams = {
        ng: count for ng, count in trigram_counter.items()
        if count >= min_count
    }

    logger.info(f"Bigrams with count >= {min_count}: {len(filtered_bigrams):,}")
    logger.info(f"Trigrams with count >= {min_count}: {len(filtered_trigrams):,}")

    # Get top N
    top_bigram_dict = dict(
        sorted(filtered_bigrams.items(), key=lambda x: x[1], reverse=True)[:top_bigrams]
    )
    top_trigram_dict = dict(
        sorted(filtered_trigrams.items(), key=lambda x: x[1], reverse=True)[:top_trigrams]
    )

    # Build metadata
    metadata = {
        'corpus_path': str(corpus_path),
        'total_bigrams': total_bigrams,
        'total_trigrams': total_trigrams,
        'unique_bigrams': len(bigram_counter),
        'unique_trigrams': len(trigram_counter),
        'filtered_bigrams': len(filtered_bigrams),
        'filtered_trigrams': len(filtered_trigrams),
        'top_bigrams_count': len(top_bigram_dict),
        'top_trigrams_count': len(top_trigram_dict),
        'min_count': min_count,
        'files_processed': files_processed,
        'ngram_separator': NGRAM_SEP,
        'created_at': datetime.now().isoformat(),
    }

    return top_bigram_dict, top_trigram_dict, metadata


def save_ngrams(
    bigrams: Dict[str, int],
    trigrams: Dict[str, int],
    metadata: Dict[str, any],
    output_path: Path
) -> None:
    """
    Save n-gram statistics to JSON file.

    Args:
        bigrams: Bigram frequency dictionary.
        trigrams: Trigram frequency dictionary.
        metadata: Metadata dictionary.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': metadata,
        'bigrams': bigrams,
        'trigrams': trigrams
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved n-grams to {output_path}")


def load_ngrams(ngrams_path: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, any]]:
    """
    Load n-gram statistics from JSON file.

    Args:
        ngrams_path: Path to n-grams JSON file.

    Returns:
        Tuple of (bigrams_dict, trigrams_dict, metadata_dict).

    Example:
        >>> bigrams, trigrams, meta = load_ngrams("data/ngrams.json")
        >>> print(f"Bigrams: {len(bigrams)}, Trigrams: {len(trigrams)}")
    """
    with open(ngrams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['bigrams'], data['trigrams'], data['metadata']


def split_ngram(ngram: str, sep: str = NGRAM_SEP) -> List[str]:
    """
    Split an n-gram string back into words.

    Args:
        ngram: N-gram string.
        sep: Separator used.

    Returns:
        List of words.

    Example:
        >>> split_ngram("الله_أكبر")
        ['الله', 'أكبر']
    """
    return ngram.split(sep)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Arabic n-gram statistics from OpenITI corpus"
    )
    parser.add_argument(
        '--corpus', '-c',
        required=True,
        help='Path to OpenITI corpus directory'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/ngrams.json',
        help='Output path for n-grams JSON'
    )
    parser.add_argument(
        '--top-bigrams',
        type=int,
        default=50000,
        help='Number of top bigrams to include (default: 50000)'
    )
    parser.add_argument(
        '--top-trigrams',
        type=int,
        default=50000,
        help='Number of top trigrams to include (default: 50000)'
    )
    parser.add_argument(
        '--min-count', '-m',
        type=int,
        default=2,
        help='Minimum n-gram frequency (default: 2)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Arabic N-gram Statistics Builder")
    logger.info("=" * 60)

    try:
        # Build n-grams
        bigrams, trigrams, metadata = build_ngram_statistics(
            args.corpus,
            top_bigrams=args.top_bigrams,
            top_trigrams=args.top_trigrams,
            min_count=args.min_count,
            show_progress=not args.no_progress
        )

        # Save results
        output_path = Path(args.output)
        save_ngrams(bigrams, trigrams, metadata, output_path)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("N-GRAM BUILD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total bigrams processed: {metadata['total_bigrams']:,}")
        logger.info(f"Total trigrams processed: {metadata['total_trigrams']:,}")
        logger.info(f"Final bigrams: {metadata['top_bigrams_count']:,}")
        logger.info(f"Final trigrams: {metadata['top_trigrams_count']:,}")

        # Show top 10 bigrams
        logger.info("\nTop 10 bigrams:")
        for i, (ngram, count) in enumerate(list(bigrams.items())[:10], 1):
            words = split_ngram(ngram)
            logger.info(f"  {i}. {' '.join(words)}: {count:,}")

        # Show top 10 trigrams
        logger.info("\nTop 10 trigrams:")
        for i, (ngram, count) in enumerate(list(trigrams.items())[:10], 1):
            words = split_ngram(ngram)
            logger.info(f"  {i}. {' '.join(words)}: {count:,}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
