"""CAMeL Tools morphological analyser wrapper with LRU caching.

Provides a thin, testable layer over camel_tools.morphology.analyzer.
Degrades gracefully to a disabled (no-op) state when CAMeL is not installed.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel used when CAMeL is unavailable
_CAMEL_UNAVAILABLE = object()


class MorphAnalyzer:
    """Wrapper around camel_tools Morphological Analyzer with in-memory caching.

    Usage::

        analyzer = MorphAnalyzer(db="calima-msa-r13", cache_size=10000)
        if analyzer.is_analysable("كتب"):
            print("Valid Arabic word")

    When ``camel_tools`` is not installed or ``enabled=False``, all methods
    return safe defaults (empty list / False) and log a single warning.
    """

    def __init__(
        self,
        db: str = "calima-msa-r13",
        cache_size: int = 10_000,
        enabled: bool = True,
    ) -> None:
        """Initialise the morphological analyser.

        Args:
            db: CAMeL morphological database name.
            cache_size: LRU cache size for analysed words (number of unique words).
            enabled: If False, all methods return safe no-op values.
                     Automatically set to False when camel_tools is not installed.

        Side effects:
            Logs WARNING if enabled=True but camel_tools is not installed.
            Logs INFO with the database name on successful initialisation.
        """
        self.db = db
        self.cache_size = cache_size
        self.enabled = enabled
        self._analyzer = None

        if not enabled:
            logger.info("MorphAnalyzer: disabled by configuration.")
            return

        try:
            from camel_tools.morphology.database import MorphologyDB
            from camel_tools.morphology.analyzer import Analyzer

            mdb = MorphologyDB.builtin_db(db)
            self._analyzer = Analyzer(mdb, "NOAN_ALL")
            logger.info("MorphAnalyzer: loaded CAMeL database '%s'.", db)

        except ImportError:
            self.enabled = False
            logger.warning(
                "camel_tools is not installed — morphological analysis will be skipped.\n"
                "To enable: pip install camel-tools && camel_data -i morphology-db-msa-r13"
            )
        except Exception as exc:  # noqa: BLE001
            self.enabled = False
            logger.warning(
                "CAMeL Tools failed to initialise (db='%s'): %s\n"
                "Morphological analysis will be skipped.",
                db, exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, word: str) -> Optional[list[dict]]:
        """Return all morphological analyses for *word*.

        Results are cached per word for the lifetime of this object.

        Args:
            word: A single Arabic word (no spaces, no punctuation).

        Returns:
            List of analysis dicts from CAMeL Tools (may be empty if the word
            is not in the lexicon).
            Returns None if the analyser is disabled.
        """
        if not self.enabled or self._analyzer is None:
            return None

        return self._analyse_cached(word)

    def is_analysable(self, word: str) -> bool:
        """Return True if *word* has at least one valid morphological analysis.

        This is the primary validity check: "is this a real Arabic word?"

        Args:
            word: A single Arabic word token.

        Returns:
            True if the word is in the morphological lexicon.
            False if not found or if the analyser is disabled.
        """
        result = self.analyse(word)
        if result is None:
            return False
        return len(result) > 0

    def analyse_batch(self, words: list[str]) -> dict[str, Optional[list[dict]]]:
        """Analyse a list of words, reusing cached results for repeated words.

        Args:
            words: List of Arabic word strings.

        Returns:
            Dict mapping each word → analysis list (or None if disabled).
        """
        return {word: self.analyse(word) for word in words}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _analyse_cached(self, word: str) -> list[dict]:
        """Internal cached analysis call."""
        # We implement manual caching rather than @lru_cache on the method
        # so that cache_size can be set per-instance.
        if not hasattr(self, "_cache"):
            self._cache: dict[str, list[dict]] = {}

        if word not in self._cache:
            if len(self._cache) >= self.cache_size:
                # Evict oldest entry (FIFO approximation — good enough)
                self._cache.pop(next(iter(self._cache)))
            try:
                self._cache[word] = self._analyzer.analyze(word)
            except Exception as exc:  # noqa: BLE001
                logger.debug("CAMeL analysis failed for word '%s': %s", word, exc)
                self._cache[word] = []

        return self._cache[word]
