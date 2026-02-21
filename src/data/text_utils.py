"""Arabic text utilities: normalisation, tokenisation, and repetition handling.

All functions are pure (no side effects) and stateless.
"""

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for efficiency)
# ---------------------------------------------------------------------------

# Full Arabic Unicode block + supplements
_ARABIC_CHARS = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

# Diacritics: harakat (fatha/damma/kasra + their tanwin), shadda, sukun, maddah,
# hamza above/below, superscript alef, small alef
_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0610-\u061A]')

# Alef variants → plain alef
_ALEF_VARIANTS = re.compile(r'[\u0622\u0623\u0625\u0671]')  # آ أ إ ٱ

# Whitespace normalisation
_MULTI_SPACE = re.compile(r'[ \t\r\n]+')

# Word boundary split: whitespace + common Arabic/general punctuation
_WORD_SPLITTER = re.compile(
    r'[\s\u060C\u061B\u061F\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]+'
)

# Default maximum consecutive identical characters allowed
MAX_RUN: int = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalise_arabic(text: str, remove_diacritics: bool = False) -> str:
    """Normalise Arabic text for consistent metric calculation.

    Normalisation steps (applied in order):
    1. Strip leading/trailing whitespace.
    2. Collapse multiple internal whitespace to a single space.
    3. Normalise Alef variants (آ أ إ ٱ) → ا.
    4. Optionally strip all diacritics (harakat, shadda, sukun …).
    5. Strip runaway character repetitions (> MAX_RUN consecutive identical chars).

    Args:
        text: Raw Arabic string (or mixed Arabic/Latin string).
        remove_diacritics: If True, strip all harakat and other diacritical marks.

    Returns:
        Normalised string suitable for metric computation.
    """
    if not text:
        return ""

    # Step 1 & 2: strip + collapse whitespace
    text = _MULTI_SPACE.sub(" ", text.strip())

    # Step 3: Alef normalisation
    text = _ALEF_VARIANTS.sub("\u0627", text)  # → ا

    # Step 4: optional diacritic removal
    if remove_diacritics:
        text = _DIACRITICS.sub("", text)
        # Also collapse any new consecutive spaces that diacritic removal created
        text = _MULTI_SPACE.sub(" ", text).strip()

    # Step 5: collapse runaway repetitions
    text = strip_repetitions(text, max_run=MAX_RUN)

    return text


def strip_repetitions(text: str, max_run: int = MAX_RUN) -> str:
    """Collapse runs of more than *max_run* consecutive identical characters.

    This handles the KHATT OCR hallucination pattern where Qaari produces
    thousands of repeated characters (e.g., ١٠٠٠٠٠٠٠...) or repeated diacritics.

    Args:
        text: Input string.
        max_run: Maximum allowed consecutive identical characters (must be ≥ 1).

    Returns:
        String with any run longer than max_run collapsed to exactly max_run chars.

    Example:
        >>> strip_repetitions("كلمة١٠٠٠٠٠٠٠٠", max_run=5)
        'كلمة١٠٠٠٠'
        >>> strip_repetitions("سسسسسسلام", max_run=3)
        'سسسلام'
    """
    if not text or max_run < 1:
        return text

    # Use a regex back-reference to match any character repeated > max_run times
    # Replace the entire run with max_run copies of that character
    pattern = re.compile(r'(.)\1{' + str(max_run) + r',}')

    def _replace(match: re.Match) -> str:
        return match.group(1) * max_run

    return pattern.sub(_replace, text)


def tokenise_arabic(text: str) -> list[str]:
    """Split text into word tokens on whitespace and Arabic/Latin punctuation.

    Args:
        text: Arabic (or mixed) string, normalised or raw.

    Returns:
        List of non-empty word strings, preserving their original form.

    Example:
        >>> tokenise_arabic("الكتاب على المكتبة.")
        ['الكتاب', 'على', 'المكتبة']
    """
    if not text:
        return []
    tokens = _WORD_SPLITTER.split(text)
    return [t for t in tokens if t]


def is_arabic_word(word: str) -> bool:
    """Return True if *word* contains at least one Arabic Unicode character.

    Args:
        word: A single token string.

    Returns:
        True if the word contains Arabic codepoints, False otherwise.

    Example:
        >>> is_arabic_word("مرحبا")
        True
        >>> is_arabic_word("hello")
        False
        >>> is_arabic_word("123")
        False
    """
    return bool(_ARABIC_CHARS.search(word))


def remove_diacritics(text: str) -> str:
    """Strip all Arabic diacritical marks from *text*.

    Convenience wrapper around normalise_arabic for callers that only want
    diacritic removal without full normalisation.

    Args:
        text: Arabic text string.

    Returns:
        Text with all harakat, shadda, sukun, etc. removed.
    """
    return _DIACRITICS.sub("", text)


def count_arabic_chars(text: str) -> int:
    """Count the number of Arabic Unicode characters in *text*.

    Used for reporting and sanity checks.

    Args:
        text: Any string.

    Returns:
        Count of characters matching the Arabic Unicode range.
    """
    return len(_ARABIC_CHARS.findall(text))
