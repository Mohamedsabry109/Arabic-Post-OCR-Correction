#!/usr/bin/env python3
"""
Arabic Vocabulary Builder from OpenITI Corpus.

Processes OpenITI corpus text files to extract Arabic word frequencies
and build a vocabulary of the most common words.

Usage:
    python scripts/build_vocabulary.py --corpus path/to/openiti
    python scripts/build_vocabulary.py --corpus data/openiti --top-n 100000

Example Output:
    vocab_100k.json with structure:
    {
        "metadata": {"total_words": 50000000, "unique_words": 500000, ...},
        "vocabulary": {"الله": 123456, "من": 100000, ...}
    }

Runtime Estimate:
    ~5-15 minutes for 1GB of text depending on I/O speed
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional, List
from datetime import datetime

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Arabic word pattern (includes Arabic letters and common diacritics)
ARABIC_WORD_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

# OpenITI metadata markers to skip
OPENITI_MARKERS = {'#META#', '###', 'PageV', 'ms', '~~'}


def is_metadata_line(line: str) -> bool:
    """
    Check if a line is OpenITI metadata to skip.

    Args:
        line: Text line to check.

    Returns:
        True if line is metadata.
    """
    stripped = line.strip()
    if not stripped:
        return True

    # Check for common metadata markers
    for marker in OPENITI_MARKERS:
        if stripped.startswith(marker):
            return True

    # Check for page markers like "PageV01P001"
    if re.match(r'PageV\d+P\d+', stripped):
        return True

    return False


def extract_arabic_words(text: str) -> Iterator[str]:
    """
    Extract Arabic words from text using regex.

    Args:
        text: Input text.

    Yields:
        Arabic words found in text.

    Example:
        >>> list(extract_arabic_words("Hello مرحبا World العالم"))
        ['مرحبا', 'العالم']
    """
    for match in ARABIC_WORD_PATTERN.finditer(text):
        word = match.group()
        # Filter out very short words (likely noise) and very long words
        if 2 <= len(word) <= 50:
            yield word


def iter_corpus_files(corpus_path: Path) -> Iterator[Path]:
    """
    Iterate over text files in corpus directory recursively.

    Args:
        corpus_path: Root directory of corpus.

    Yields:
        Path to each text file.
    """
    extensions = {'.txt', '.ara', '.mARkdown'}

    for root, dirs, files in os.walk(corpus_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            filepath = Path(root) / filename
            if filepath.suffix.lower() in extensions:
                yield filepath


def process_file(filepath: Path) -> Iterator[str]:
    """
    Process a single file and yield Arabic words.

    Args:
        filepath: Path to text file.

    Yields:
        Arabic words from the file.
    """
    encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    # Skip metadata lines
                    if is_metadata_line(line):
                        continue

                    # Extract and yield words
                    for word in extract_arabic_words(line):
                        yield word
            return  # Successfully processed
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return

    logger.warning(f"Could not decode file: {filepath}")


def build_vocabulary_from_openiti(
    corpus_path: str,
    top_n: int = 100000,
    min_count: int = 2,
    show_progress: bool = True
) -> Tuple[Dict[str, int], Dict[str, any]]:
    """
    Build vocabulary from OpenITI corpus.

    Args:
        corpus_path: Path to OpenITI corpus directory.
        top_n: Number of top words to include.
        min_count: Minimum frequency to include word.
        show_progress: Show progress bar.

    Returns:
        Tuple of (vocabulary_dict, metadata_dict).

    Example:
        >>> vocab, meta = build_vocabulary_from_openiti("data/openiti", top_n=50000)
        >>> print(f"Top word: {list(vocab.keys())[0]}")
        >>> print(f"Total words processed: {meta['total_words']}")
    """
    corpus_path = Path(corpus_path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {corpus_path}")

    logger.info(f"Building vocabulary from: {corpus_path}")
    logger.info(f"Target vocabulary size: {top_n:,} words")

    # Count files first for progress bar
    files = list(iter_corpus_files(corpus_path))
    logger.info(f"Found {len(files):,} text files")

    # Word counter
    word_counts = Counter()
    total_words = 0
    files_processed = 0

    # Process files
    iterator = tqdm(files, desc="Processing files") if show_progress else files

    for filepath in iterator:
        try:
            for word in process_file(filepath):
                word_counts[word] += 1
                total_words += 1
            files_processed += 1
        except Exception as e:
            logger.debug(f"Error processing {filepath}: {e}")

    logger.info(f"Processed {files_processed:,} files")
    logger.info(f"Total words: {total_words:,}")
    logger.info(f"Unique words: {len(word_counts):,}")

    # Filter by minimum count
    filtered_counts = {
        word: count for word, count in word_counts.items()
        if count >= min_count
    }
    logger.info(f"Words with count >= {min_count}: {len(filtered_counts):,}")

    # Get top N words
    top_words = dict(
        sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    # Build metadata
    metadata = {
        'corpus_path': str(corpus_path),
        'total_words': total_words,
        'unique_words': len(word_counts),
        'filtered_unique_words': len(filtered_counts),
        'vocabulary_size': len(top_words),
        'min_count': min_count,
        'top_n': top_n,
        'files_processed': files_processed,
        'created_at': datetime.now().isoformat(),
    }

    return top_words, metadata


def save_vocabulary(
    vocabulary: Dict[str, int],
    metadata: Dict[str, any],
    output_path: Path
) -> None:
    """
    Save vocabulary to JSON file.

    Args:
        vocabulary: Word frequency dictionary.
        metadata: Metadata dictionary.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': metadata,
        'vocabulary': vocabulary
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved vocabulary to {output_path}")


def load_vocabulary(vocab_path: str) -> Tuple[Dict[str, int], Dict[str, any]]:
    """
    Load vocabulary from JSON file.

    Args:
        vocab_path: Path to vocabulary JSON file.

    Returns:
        Tuple of (vocabulary_dict, metadata_dict).

    Example:
        >>> vocab, meta = load_vocabulary("data/vocab_100k.json")
        >>> print(f"Vocabulary size: {len(vocab)}")
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['vocabulary'], data['metadata']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Arabic vocabulary from OpenITI corpus"
    )
    parser.add_argument(
        '--corpus', '-c',
        required=True,
        help='Path to OpenITI corpus directory'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/vocab_100k.json',
        help='Output path for vocabulary JSON'
    )
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=100000,
        help='Number of top words to include (default: 100000)'
    )
    parser.add_argument(
        '--min-count', '-m',
        type=int,
        default=2,
        help='Minimum word frequency (default: 2)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Arabic Vocabulary Builder")
    logger.info("=" * 60)

    try:
        # Build vocabulary
        vocabulary, metadata = build_vocabulary_from_openiti(
            args.corpus,
            top_n=args.top_n,
            min_count=args.min_count,
            show_progress=not args.no_progress
        )

        # Save results
        output_path = Path(args.output)
        save_vocabulary(vocabulary, metadata, output_path)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VOCABULARY BUILD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total words processed: {metadata['total_words']:,}")
        logger.info(f"Unique words found: {metadata['unique_words']:,}")
        logger.info(f"Final vocabulary size: {metadata['vocabulary_size']:,}")

        # Show top 10 words
        logger.info("\nTop 10 most frequent words:")
        for i, (word, count) in enumerate(list(vocabulary.items())[:10], 1):
            logger.info(f"  {i}. {word}: {count:,}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
