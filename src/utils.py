"""
Shared utilities for Arabic text processing.

This module provides utility functions for handling Arabic text,
including normalization, cleaning, and encoding operations.

Example:
    >>> from src.utils import normalize_arabic, clean_text
    >>> text = "  مرحباً   بالعالم  "
    >>> clean_text(text)
    'مرحباً بالعالم'
"""

import re
import unicodedata
from typing import Optional


# Arabic Unicode ranges
ARABIC_RANGE = r'\u0600-\u06FF'  # Basic Arabic
ARABIC_SUPPLEMENT = r'\u0750-\u077F'  # Arabic Supplement
ARABIC_EXTENDED_A = r'\u08A0-\u08FF'  # Arabic Extended-A
ARABIC_PRESENTATION_A = r'\uFB50-\uFDFF'  # Arabic Presentation Forms-A
ARABIC_PRESENTATION_B = r'\uFE70-\uFEFF'  # Arabic Presentation Forms-B

# Common Arabic diacritics (tashkeel)
ARABIC_DIACRITICS = (
    '\u064B',  # Fathatan
    '\u064C',  # Dammatan
    '\u064D',  # Kasratan
    '\u064E',  # Fatha
    '\u064F',  # Damma
    '\u0650',  # Kasra
    '\u0651',  # Shadda
    '\u0652',  # Sukun
    '\u0653',  # Maddah
    '\u0654',  # Hamza Above
    '\u0655',  # Hamza Below
    '\u0656',  # Subscript Alef
    '\u0670',  # Superscript Alef
)

# Pattern for Arabic diacritics
DIACRITICS_PATTERN = re.compile(f'[{"".join(ARABIC_DIACRITICS)}]')


def normalize_arabic(text: str, remove_diacritics: bool = False) -> str:
    """
    Normalize Arabic text for consistent processing.

    Performs the following normalizations:
    - Normalizes Alef variants (أ, إ, آ) to bare Alef (ا)
    - Normalizes Teh Marbuta (ة) to Heh (ه) - optional
    - Normalizes Alef Maksura (ى) to Yeh (ي)
    - Optionally removes diacritics (tashkeel)
    - Applies Unicode NFC normalization

    Args:
        text: Arabic text to normalize.
        remove_diacritics: If True, removes all Arabic diacritics.
            Defaults to False to preserve diacritical marks.

    Returns:
        Normalized Arabic text.

    Example:
        >>> normalize_arabic("أَحْمَد")
        'أَحْمَد'
        >>> normalize_arabic("أَحْمَد", remove_diacritics=True)
        'احمد'
    """
    if not text:
        return ""

    # Unicode NFC normalization first
    text = unicodedata.normalize('NFC', text)

    # Normalize Alef variants to bare Alef
    text = re.sub(r'[أإآٱ]', 'ا', text)

    # Normalize Alef Maksura to Yeh
    text = re.sub(r'ى', 'ي', text)

    # Remove diacritics if requested
    if remove_diacritics:
        text = DIACRITICS_PATTERN.sub('', text)

    return text


def clean_text(text: str, normalize_whitespace: bool = True) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Text to clean.
        normalize_whitespace: If True, collapses multiple whitespace
            characters into single spaces. Defaults to True.

    Returns:
        Cleaned text with normalized whitespace.

    Example:
        >>> clean_text("  مرحباً   بالعالم  ")
        'مرحباً بالعالم'
        >>> clean_text("سطر أول\\n\\nسطر ثاني")
        'سطر أول سطر ثاني'
    """
    if not text:
        return ""

    # Strip leading/trailing whitespace
    text = text.strip()

    if normalize_whitespace:
        # Replace multiple whitespace (including newlines) with single space
        text = re.sub(r'\s+', ' ', text)

    return text


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritical marks (tashkeel) from text.

    Args:
        text: Arabic text with diacritics.

    Returns:
        Text with all Arabic diacritics removed.

    Example:
        >>> remove_diacritics("مَرْحَباً")
        'مرحبا'
    """
    if not text:
        return ""
    return DIACRITICS_PATTERN.sub('', text)


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    """
    Check if text contains significant Arabic content.

    Args:
        text: Text to check.
        threshold: Minimum ratio of Arabic characters to consider
            the text as Arabic. Defaults to 0.3 (30%).

    Returns:
        True if Arabic characters exceed the threshold ratio.

    Example:
        >>> is_arabic("مرحبا بالعالم")
        True
        >>> is_arabic("Hello World")
        False
        >>> is_arabic("Hello مرحبا")
        True
    """
    if not text:
        return False

    # Count Arabic characters
    arabic_pattern = re.compile(
        f'[{ARABIC_RANGE}{ARABIC_SUPPLEMENT}{ARABIC_EXTENDED_A}'
        f'{ARABIC_PRESENTATION_A}{ARABIC_PRESENTATION_B}]'
    )
    arabic_chars = len(arabic_pattern.findall(text))

    # Count total non-whitespace characters
    total_chars = len(re.sub(r'\s', '', text))

    if total_chars == 0:
        return False

    return (arabic_chars / total_chars) >= threshold


def ensure_utf8(text: str) -> str:
    """
    Ensure text is properly encoded as UTF-8.

    Handles common encoding issues with Arabic text.

    Args:
        text: Text that may have encoding issues.

    Returns:
        Properly encoded UTF-8 text.

    Example:
        >>> ensure_utf8("مرحبا")
        'مرحبا'
    """
    if not text:
        return ""

    # If already a string, just return it (Python 3 strings are Unicode)
    if isinstance(text, str):
        return text

    # If bytes, decode as UTF-8
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError:
            # Try other common Arabic encodings
            for encoding in ['cp1256', 'iso-8859-6', 'utf-16']:
                try:
                    return text.decode(encoding)
                except UnicodeDecodeError:
                    continue
            # Last resort: decode with replacement
            return text.decode('utf-8', errors='replace')

    return str(text)


def get_text_stats(text: str) -> dict:
    """
    Get statistics about Arabic text.

    Args:
        text: Text to analyze.

    Returns:
        Dictionary with text statistics including:
        - char_count: Total characters
        - word_count: Number of words
        - arabic_char_count: Number of Arabic characters
        - has_diacritics: Whether text contains diacritics

    Example:
        >>> stats = get_text_stats("مَرْحَباً بالعالم")
        >>> stats['word_count']
        2
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'arabic_char_count': 0,
            'has_diacritics': False,
        }

    # Arabic character pattern
    arabic_pattern = re.compile(
        f'[{ARABIC_RANGE}{ARABIC_SUPPLEMENT}{ARABIC_EXTENDED_A}'
        f'{ARABIC_PRESENTATION_A}{ARABIC_PRESENTATION_B}]'
    )

    return {
        'char_count': len(text),
        'word_count': len(text.split()),
        'arabic_char_count': len(arabic_pattern.findall(text)),
        'has_diacritics': bool(DIACRITICS_PATTERN.search(text)),
    }


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, preserving word boundaries.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix. Defaults to 100.
        suffix: String to append when truncating. Defaults to "...".

    Returns:
        Truncated text with suffix if it exceeded max_length.

    Example:
        >>> truncate_text("هذا نص طويل جداً", max_length=10)
        'هذا نص...'
    """
    if not text or len(text) <= max_length:
        return text

    # Find the last space before max_length - len(suffix)
    truncate_at = max_length - len(suffix)

    # Try to find a word boundary
    last_space = text.rfind(' ', 0, truncate_at)
    if last_space > 0:
        truncate_at = last_space

    return text[:truncate_at].rstrip() + suffix


def format_sample_correction(
    index: int,
    ocr_text: str,
    corrected_text: str,
    ground_truth: str,
    max_text_length: int = 200
) -> str:
    """
    Format a sample correction for display/logging.

    Args:
        index: Sample number (1-based).
        ocr_text: Original OCR text.
        corrected_text: LLM-corrected text.
        ground_truth: Ground truth text.
        max_text_length: Maximum text length to display.

    Returns:
        Formatted string showing the correction comparison.

    Example:
        >>> print(format_sample_correction(1, "مرحبا", "مرحباً", "مرحباً"))
        === Sample 1 ===
        OCR:        مرحبا
        Corrected:  مرحباً
        Truth:      مرحباً
        ================
    """
    separator = "=" * 50

    # Truncate long texts
    ocr_display = truncate_text(ocr_text, max_text_length)
    corrected_display = truncate_text(corrected_text, max_text_length)
    truth_display = truncate_text(ground_truth, max_text_length)

    return (
        f"\n{separator}\n"
        f"=== Sample {index} ===\n"
        f"{separator}\n"
        f"OCR:        {ocr_display}\n"
        f"Corrected:  {corrected_display}\n"
        f"Truth:      {truth_display}\n"
        f"{separator}\n"
    )
