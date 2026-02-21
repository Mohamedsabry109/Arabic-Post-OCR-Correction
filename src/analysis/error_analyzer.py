"""Character- and word-level error analysis for Arabic OCR output.

Provides:
- Character alignment via edit-distance backtracking
- Error type classification (dot confusion, hamza, taa marbuta, etc.)
- Confusion matrix construction
- Error taxonomy with position analysis
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, TYPE_CHECKING

import editdistance

from src.data.text_utils import normalise_arabic, tokenise_arabic, is_arabic_word

if TYPE_CHECKING:
    from src.data.data_loader import OCRSample

logger = logging.getLogger(__name__)

# Maximum error examples stored per error type in the taxonomy
_MAX_EXAMPLES = 5


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ErrorType(str, Enum):
    DOT_CONFUSION    = "dot_confusion"       # ب↔ت↔ث↔ن, ج↔ح↔خ, etc.
    HAMZA            = "hamza"               # أ↔ا↔إ↔آ↔ء
    TAA_MARBUTA      = "taa_marbuta"         # ة↔ه
    ALEF_MAKSURA     = "alef_maksura"        # ى↔ي
    SIMILAR_SHAPE    = "similar_shape"       # ر↔ز, د↔ذ, و↔ر, ع↔غ, etc.
    MERGED_WORDS     = "merged_words"        # two GT words → one OCR token
    SPLIT_WORD       = "split_word"          # one GT word → two OCR tokens
    DELETION         = "deletion"            # character in GT missing from OCR
    INSERTION        = "insertion"           # extra character in OCR
    OTHER_SUB        = "other_substitution"  # substitution not in above groups
    UNKNOWN          = "unknown"


class ErrorPosition(str, Enum):
    WORD_START  = "word_start"    # first character of a multi-char word
    WORD_MIDDLE = "word_middle"   # neither first nor last
    WORD_END    = "word_end"      # last character of a multi-char word
    SINGLE_CHAR = "single_char"   # word is exactly one character


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CharError:
    """One character-level error between GT and OCR output."""

    gt_char: str              # ground truth character ('' for insertion)
    ocr_char: str             # OCR character ('' for deletion)
    error_type: ErrorType
    position: ErrorPosition
    gt_context: str           # 3-char window around the error in GT


@dataclass
class WordError:
    """One word-level error (substitution, insertion, or deletion)."""

    gt_word: str
    ocr_word: str
    is_merged: bool           # OCR merged two GT words
    is_split: bool            # OCR split one GT word
    char_errors: list[CharError] = field(default_factory=list)


@dataclass
class SampleError:
    """All errors found in one OCRSample."""

    sample_id: str
    dataset: str
    cer: float
    wer: float
    char_errors: list[CharError] = field(default_factory=list)
    word_errors: list[WordError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ErrorAnalyzer
# ---------------------------------------------------------------------------


class ErrorAnalyzer:
    """Align GT and OCR text, classify errors, build confusion matrix and taxonomy.

    Usage::

        analyzer = ErrorAnalyzer()
        errors = [analyzer.analyse_sample(s) for s in samples]
        matrix = analyzer.build_confusion_matrix(errors, dataset="KHATT-train")
        taxonomy = analyzer.build_taxonomy(errors, dataset="KHATT-train")
    """

    # ------------------------------------------------------------------
    # Character group tables for error classification
    # ------------------------------------------------------------------

    # Each frozenset defines one confusable group (dot confusion)
    DOT_GROUPS: list[frozenset] = [
        frozenset("بتثن"),   # ba / ta / tha / nun
        frozenset("جحخ"),    # jim / ha / kha
        frozenset("دذ"),     # dal / dhal
        frozenset("رز"),     # ra / zain
        frozenset("سش"),     # sin / shin
        frozenset("صض"),     # sad / dad
        frozenset("طظ"),     # ta / dha (emphatic)
        frozenset("فق"),     # fa / qaf
        frozenset("يى"),     # ya / alef maqsura (dot position)
    ]

    HAMZA_GROUP: frozenset = frozenset("أاإآءٱ")
    TAA_GROUP: frozenset   = frozenset("ةه")
    ALEF_MAKSURA_GROUP: frozenset = frozenset("ىي")

    # Known visually similar pairs not covered by dot groups
    SIMILAR_SHAPE_PAIRS: set[frozenset] = {
        frozenset("وأ"), frozenset("عغ"), frozenset("من"), frozenset("نم"),
        frozenset("كل"), frozenset("لا"), frozenset("هأ"), frozenset("حم"),
    }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def analyse_sample(self, sample: "OCRSample") -> SampleError:
        """Align GT and OCR text for one sample and extract all errors.

        Steps:
        1. Normalise both strings.
        2. Compute character-level edit operations.
        3. Classify each operation into ErrorType and ErrorPosition.
        4. Detect word-level merges and splits.

        Args:
            sample: OCRSample with gt_text and ocr_text.

        Returns:
            SampleError with complete error breakdown.
        """
        from src.analysis.metrics import calculate_cer, calculate_wer

        ref = normalise_arabic(sample.gt_text)
        hyp = normalise_arabic(sample.ocr_text)

        cer = calculate_cer(sample.gt_text, sample.ocr_text)
        wer = calculate_wer(sample.gt_text, sample.ocr_text)

        # Character-level errors
        char_pairs = self._align_chars(ref, hyp)
        char_errors = self._extract_char_errors(char_pairs, ref)

        # Word-level errors
        ref_words = tokenise_arabic(ref)
        hyp_words = tokenise_arabic(hyp)
        word_errors = self._extract_word_errors(ref_words, hyp_words)

        return SampleError(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            cer=cer,
            wer=wer,
            char_errors=char_errors,
            word_errors=word_errors,
        )

    def build_confusion_matrix(
        self,
        errors: list[SampleError],
        dataset: str,
        min_count: int = 2,
    ) -> dict:
        """Build a character-level confusion matrix from all SampleError objects.

        Only includes substitutions between Arabic characters (ignores
        insertions, deletions, and non-Arabic characters).

        Args:
            errors: List of SampleError from analyse_sample().
            dataset: Dataset label for the metadata block.
            min_count: Minimum occurrence count to include a confusion pair.

        Returns:
            Dict matching the schema in Architecture.md Appendix A.1.
        """
        # raw_counts[gt_char][ocr_char] = count
        raw_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_subs = 0
        total_errors = 0

        for sample_err in errors:
            for ce in sample_err.char_errors:
                total_errors += 1
                if ce.error_type not in (ErrorType.DELETION, ErrorType.INSERTION,
                                         ErrorType.MERGED_WORDS, ErrorType.SPLIT_WORD):
                    if is_arabic_word(ce.gt_char) and is_arabic_word(ce.ocr_char):
                        raw_counts[ce.gt_char][ce.ocr_char] += 1
                        total_subs += 1

        # Filter by min_count and compute probabilities
        confusions: dict[str, dict] = {}
        for gt_char, ocr_dict in raw_counts.items():
            gt_total = sum(ocr_dict.values())
            filtered = {
                ocr_char: cnt
                for ocr_char, cnt in ocr_dict.items()
                if cnt >= min_count
            }
            if not filtered:
                continue
            confusions[gt_char] = {
                ocr_char: {
                    "count": cnt,
                    "probability": round(cnt / gt_total, 4),
                }
                for ocr_char, cnt in sorted(filtered.items(), key=lambda x: -x[1])
            }

        # Top-20 flat list for quick reference
        all_pairs = [
            (gt, ocr, data["count"], data["probability"])
            for gt, ocr_map in confusions.items()
            for ocr, data in ocr_map.items()
        ]
        top_20 = sorted(all_pairs, key=lambda x: -x[2])[:20]

        return {
            "meta": {
                "dataset": dataset,
                "total_char_errors": total_errors,
                "total_substitutions": total_subs,
                "unique_confusions": len(all_pairs),
                "generated_at": _now_iso(),
            },
            "confusions": confusions,
            "top_20": [
                {"gt": gt, "ocr": ocr, "count": cnt, "probability": prob}
                for gt, ocr, cnt, prob in top_20
            ],
        }

    def build_taxonomy(
        self,
        errors: list[SampleError],
        dataset: str,
    ) -> dict:
        """Aggregate error counts by ErrorType and ErrorPosition.

        Args:
            errors: List of SampleError objects.
            dataset: Dataset label for metadata.

        Returns:
            Dict with by_type, by_position, and word_level breakdowns.
        """
        type_counts: dict[str, int] = defaultdict(int)
        pos_counts: dict[str, int] = defaultdict(int)
        type_examples: dict[str, list] = defaultdict(list)

        total_char_errors = 0
        merged_count = 0
        split_count = 0

        for sample_err in errors:
            for ce in sample_err.char_errors:
                total_char_errors += 1
                t = ce.error_type.value
                type_counts[t] += 1
                pos_counts[ce.position.value] += 1

                if len(type_examples[t]) < _MAX_EXAMPLES:
                    type_examples[t].append({
                        "gt": ce.gt_char,
                        "ocr": ce.ocr_char,
                        "context": ce.gt_context,
                    })

            for we in sample_err.word_errors:
                if we.is_merged:
                    merged_count += 1
                if we.is_split:
                    split_count += 1

        # Word-level aggregate
        ref_word_subs = sum(
            1 for se in errors for we in se.word_errors
            if we.gt_word and we.ocr_word and not we.is_merged and not we.is_split
        )
        ref_word_dels = sum(
            1 for se in errors for we in se.word_errors
            if we.gt_word and not we.ocr_word
        )
        ref_word_ins = sum(
            1 for se in errors for we in se.word_errors
            if not we.gt_word and we.ocr_word
        )

        # Build by_type output
        by_type: dict = {}
        for et in ErrorType:
            count = type_counts.get(et.value, 0)
            pct = (count / total_char_errors * 100) if total_char_errors > 0 else 0.0
            by_type[et.value] = {
                "count": count,
                "percentage": round(pct, 2),
                "examples": type_examples.get(et.value, []),
            }

        # Build by_position output
        by_position: dict = {}
        for ep in ErrorPosition:
            count = pos_counts.get(ep.value, 0)
            pct = (count / total_char_errors * 100) if total_char_errors > 0 else 0.0
            by_position[ep.value] = {
                "count": count,
                "percentage": round(pct, 2),
            }

        return {
            "meta": {
                "dataset": dataset,
                "total_samples": len(errors),
                "total_char_errors": total_char_errors,
                "total_word_errors": ref_word_subs + ref_word_dels + ref_word_ins + merged_count + split_count,
                "generated_at": _now_iso(),
            },
            "by_type": by_type,
            "by_position": by_position,
            "word_level": {
                "merged": merged_count,
                "split": split_count,
                "total_word_substitutions": ref_word_subs,
                "total_word_deletions": ref_word_dels,
                "total_word_insertions": ref_word_ins,
            },
        }

    def get_top_confusions(
        self,
        confusion_matrix: dict,
        n: int = 20,
    ) -> list[tuple[str, str, int, float]]:
        """Return top-N confusion pairs sorted by count descending.

        Args:
            confusion_matrix: Output of build_confusion_matrix().
            n: Number of top pairs to return.

        Returns:
            List of (gt_char, ocr_char, count, probability) tuples.
        """
        pairs = []
        for gt_char, ocr_map in confusion_matrix.get("confusions", {}).items():
            for ocr_char, data in ocr_map.items():
                pairs.append((gt_char, ocr_char, data["count"], data["probability"]))
        return sorted(pairs, key=lambda x: -x[2])[:n]

    # ------------------------------------------------------------------
    # Private: character alignment
    # ------------------------------------------------------------------

    def _align_chars(self, ref: str, hyp: str) -> list[tuple[str, str]]:
        """Return aligned (ref_char, hyp_char) pairs via edit-distance backtracking.

        Uses '' (empty string) for insertions/deletions:
          ('ب', 'ت') → substitution
          ('ب', '')  → deletion  (char in GT missing from OCR)
          ('',  'x') → insertion (extra char in OCR)

        Args:
            ref: Normalised ground truth string.
            hyp: Normalised OCR hypothesis string.

        Returns:
            List of (ref_char, hyp_char) aligned character pairs.
        """
        n, m = len(ref), len(hyp)

        # Build DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],     # deletion
                        dp[i][j - 1],     # insertion
                        dp[i - 1][j - 1], # substitution
                    )

        # Backtrack
        alignment: list[tuple[str, str]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
                # Match
                alignment.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                # Substitution
                alignment.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                # Deletion (ref char missing in hyp)
                alignment.append((ref[i - 1], ""))
                i -= 1
            else:
                # Insertion (extra char in hyp)
                alignment.append(("", hyp[j - 1]))
                j -= 1

        alignment.reverse()
        return alignment

    def _extract_char_errors(
        self,
        char_pairs: list[tuple[str, str]],
        ref: str,
    ) -> list[CharError]:
        """Convert aligned char pairs into CharError objects (errors only).

        Only non-matching pairs generate a CharError. Matches are skipped.

        Args:
            char_pairs: Output of _align_chars().
            ref: Original reference string (for context extraction).

        Returns:
            List of CharError objects.
        """
        errors: list[CharError] = []

        # Build a ref position tracker (skipping insertions to track ref idx)
        ref_idx = 0

        for ref_char, ocr_char in char_pairs:
            if ref_char == ocr_char:
                # Exact match — advance ref pointer
                if ref_char:
                    ref_idx += 1
                continue

            error_type = self._classify_error_type(ref_char, ocr_char)
            position = self._classify_position(ref_idx, ref) if ref_char else ErrorPosition.WORD_MIDDLE

            # Build context: 3-char window in ref around current ref_idx
            ctx_start = max(0, ref_idx - 1)
            ctx_end = min(len(ref), ref_idx + 2)
            context = ref[ctx_start:ctx_end]

            errors.append(CharError(
                gt_char=ref_char,
                ocr_char=ocr_char,
                error_type=error_type,
                position=position,
                gt_context=context,
            ))

            if ref_char:
                ref_idx += 1

        return errors

    # ------------------------------------------------------------------
    # Private: word-level alignment
    # ------------------------------------------------------------------

    def _extract_word_errors(
        self,
        ref_words: list[str],
        hyp_words: list[str],
    ) -> list[WordError]:
        """Word-level alignment to detect merges, splits, and substitutions.

        Args:
            ref_words: Tokenised ground-truth words.
            hyp_words: Tokenised OCR words.

        Returns:
            List of WordError objects for non-matching word pairs.
        """
        pairs = self._align_words(ref_words, hyp_words)
        errors: list[WordError] = []

        for ref_w, hyp_w in pairs:
            if ref_w == hyp_w:
                continue

            gt = ref_w or ""
            ocr = hyp_w or ""

            # Detect merge: one OCR token corresponds to two GT words
            is_merged = (
                gt and ocr and
                len(ocr) > len(gt) * 1.5 and
                _words_are_merged(gt, ocr, ref_words)
            )

            # Detect split: OCR produces shorter fragments
            is_split = (
                gt and not ocr and
                len(gt) > 4  # short words may just be deletions
            )

            errors.append(WordError(
                gt_word=gt,
                ocr_word=ocr,
                is_merged=is_merged,
                is_split=is_split,
            ))

        return errors

    def _align_words(
        self,
        ref_words: list[str],
        hyp_words: list[str],
    ) -> list[tuple[Optional[str], Optional[str]]]:
        """Word-level DP alignment.

        Returns:
            List of (ref_word, hyp_word) pairs.
            None in either position indicates insertion/deletion.
        """
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

        # Backtrack
        alignment: list[tuple[Optional[str], Optional[str]]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                alignment.append((ref_words[i - 1], hyp_words[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append((ref_words[i - 1], hyp_words[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                alignment.append((ref_words[i - 1], None))
                i -= 1
            else:
                alignment.append((None, hyp_words[j - 1]))
                j -= 1

        alignment.reverse()
        return alignment

    # ------------------------------------------------------------------
    # Private: error classification
    # ------------------------------------------------------------------

    def _classify_error_type(self, gt_char: str, ocr_char: str) -> ErrorType:
        """Map a (gt_char, ocr_char) pair to an ErrorType.

        Priority order:
        1. Deletion or Insertion (one side empty)
        2. Hamza group
        3. Taa Marbuta group
        4. Alef Maksura group
        5. Dot confusion groups
        6. Similar shape pairs
        7. Other substitution

        Args:
            gt_char: Ground truth character ('' for insertion).
            ocr_char: OCR character ('' for deletion).

        Returns:
            Matching ErrorType.
        """
        if not gt_char:
            return ErrorType.INSERTION
        if not ocr_char:
            return ErrorType.DELETION

        pair = frozenset([gt_char, ocr_char])

        if gt_char in self.HAMZA_GROUP and ocr_char in self.HAMZA_GROUP:
            return ErrorType.HAMZA
        if pair <= self.TAA_GROUP:
            return ErrorType.TAA_MARBUTA
        if pair <= self.ALEF_MAKSURA_GROUP:
            return ErrorType.ALEF_MAKSURA
        for group in self.DOT_GROUPS:
            if gt_char in group and ocr_char in group:
                return ErrorType.DOT_CONFUSION
        if pair in self.SIMILAR_SHAPE_PAIRS:
            return ErrorType.SIMILAR_SHAPE

        return ErrorType.OTHER_SUB

    def _classify_position(self, char_idx: int, word_context: str) -> ErrorPosition:
        """Return position of the character at char_idx within its word.

        Determines which word in word_context contains char_idx, then returns
        its relative position (start, middle, end, or single).

        Args:
            char_idx: 0-based index of the character in word_context (the full ref string).
            word_context: The full reference string.

        Returns:
            ErrorPosition enum value.
        """
        if not word_context or char_idx >= len(word_context):
            return ErrorPosition.WORD_MIDDLE

        # Find word boundaries around char_idx
        # Walk left to find word start
        start = char_idx
        while start > 0 and word_context[start - 1] not in (' ', '\n', '\t'):
            start -= 1

        # Walk right to find word end (exclusive)
        end = char_idx + 1
        while end < len(word_context) and word_context[end] not in (' ', '\n', '\t'):
            end += 1

        word_len = end - start
        pos_in_word = char_idx - start

        if word_len == 1:
            return ErrorPosition.SINGLE_CHAR
        if pos_in_word == 0:
            return ErrorPosition.WORD_START
        if pos_in_word == word_len - 1:
            return ErrorPosition.WORD_END
        return ErrorPosition.WORD_MIDDLE


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _words_are_merged(gt_word: str, ocr_word: str, all_ref_words: list[str]) -> bool:
    """Heuristic: check if ocr_word looks like two GT words concatenated."""
    for i, w in enumerate(all_ref_words):
        if w == gt_word and i + 1 < len(all_ref_words):
            next_word = all_ref_words[i + 1]
            if ocr_word == gt_word + next_word or ocr_word == gt_word + " " + next_word:
                return True
    return False
