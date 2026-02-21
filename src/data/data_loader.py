"""Data loading and alignment for KHATT and PATS-A01 datasets.

Loads OCR predictions and ground-truth text, aligns them by filename stem,
and returns OCRSample objects for downstream analysis.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from src.data.text_utils import strip_repetitions

logger = logging.getLogger(__name__)

# Maximum length (chars) for any single text field after loading.
# Guards against runaway OCR outputs consuming excessive memory/time.
_MAX_TEXT_LEN = 2000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OCRSample:
    """A single aligned OCR prediction / ground-truth pair."""

    sample_id: str              # e.g. "Akhbar_1" or "AHTD3A0001_Para2_3"
    dataset: str                # "PATS-A01" or "KHATT"
    font: Optional[str]         # "Akhbar" | "Andalus" | None (for KHATT)
    split: Optional[str]        # "train" | "validation" | None (for PATS)
    ocr_text: str               # cleaned OCR prediction
    gt_text: str                # ground truth text
    ocr_path: Path              # source file path for OCR
    gt_path: Optional[Path]     # source file path for GT (None if unavailable)


class DataError(Exception):
    """Raised when dataset loading fails due to a data issue (not a code bug)."""


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """Load and align OCR predictions with ground-truth for all datasets.

    All paths are resolved relative to the project root (the directory
    containing configs/). Call site passes a parsed config dict.

    Usage::

        loader = DataLoader(config)
        samples = loader.load_khatt(split="train", limit=100)
        all_data = loader.load_all()
    """

    def __init__(self, config: dict) -> None:
        """Initialise paths from the parsed config dict.

        Expected config keys::

            config['data']['ocr_results']    → base dir for OCR files
            config['data']['ground_truth']   → base dir for GT files
            config['processing']['limit_per_dataset'] → int | None

        PATS-A01 GT files are auto-derived from the ground_truth root:
        ``{ground_truth}/PATS_A01_Dataset/A01-{font}Text.txt`` (cp1256 encoding).

        Args:
            config: Parsed config.yaml as a nested dict.

        Raises:
            DataError: If required config keys are missing.
        """
        try:
            data_cfg = config["data"]
            self._ocr_root = Path(data_cfg["ocr_results"])
            self._gt_root = Path(data_cfg["ground_truth"])
        except KeyError as exc:
            raise DataError(f"Missing required config key: {exc}") from exc

        proc_cfg = config.get("processing", {})
        self._default_limit: Optional[int] = proc_cfg.get("limit_per_dataset")

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_pats(
        self,
        font: str = "Akhbar",
        limit: Optional[int] = None,
    ) -> list[OCRSample]:
        """Load PATS-A01 samples for a specific font.

        GT is loaded from ``{ground_truth}/PATS_A01_Dataset/A01-{font}Text.txt``
        (cp1256 encoding). Line N (1-indexed) corresponds to ``{font}_N.txt``.

        Args:
            font: Font subdirectory name, e.g. "Akhbar" or "Andalus".
                  The OCR directory is expected as ``A01-{font}``.
            limit: If set, return at most this many samples (sorted by ID).

        Returns:
            List of OCRSample sorted by numeric line index.

        Raises:
            DataError: If the OCR directory or GT file does not exist.
        """
        ocr_dir = self._ocr_root / f"pats-a01-data" / f"A01-{font}"
        if not ocr_dir.exists():
            raise DataError(f"PATS OCR directory not found: {ocr_dir}")

        # Discover all OCR .txt files for this font
        ocr_files = sorted(
            ocr_dir.glob("*.txt"),
            key=lambda p: _pats_line_number(p),
        )
        if not ocr_files:
            raise DataError(f"No OCR text files found in {ocr_dir}")

        # Load GT
        gt_lines, gt_file = self._load_pats_gt(font, len(ocr_files))

        effective_limit = limit if limit is not None else self._default_limit
        if effective_limit is not None:
            ocr_files = ocr_files[:effective_limit]

        samples: list[OCRSample] = []
        skipped = 0

        for ocr_path in ocr_files:
            line_num = _pats_line_number(ocr_path)
            ocr_text = _read_ocr_file(ocr_path)

            if ocr_text == "":
                logger.warning("Empty OCR file skipped: %s", ocr_path.name)
                skipped += 1
                continue

            # GT: line_num is 1-indexed; list is 0-indexed
            gt_text = gt_lines[line_num - 1] if gt_lines and (line_num - 1) < len(gt_lines) else ""

            samples.append(OCRSample(
                sample_id=ocr_path.stem,
                dataset="PATS-A01",
                font=font,
                split=None,
                ocr_text=ocr_text,
                gt_text=gt_text,
                ocr_path=ocr_path,
                gt_path=gt_file,
            ))

        if skipped:
            logger.warning("PATS-%s: skipped %d empty OCR files.", font, skipped)

        logger.info("Loaded %d PATS-A01-%s samples.", len(samples), font)
        return samples

    def load_khatt(
        self,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> list[OCRSample]:
        """Load KHATT samples for a given split.

        Pairs files by filename stem. Loads the intersection of OCR and GT
        file sets. Logs any unmatched files as warnings.

        Args:
            split: "train" or "validation".
            limit: If set, return at most this many samples (alphabetical order).

        Returns:
            List of OCRSample sorted alphabetically by sample_id.

        Raises:
            DataError: If the OCR or GT split directory does not exist.
        """
        subfolder = "Training" if split == "train" else "Validation"
        ocr_dir = self._ocr_root / "khatt-data" / split / subfolder
        gt_dir = self._gt_root / "KHATT" / "data" / split / subfolder

        if not ocr_dir.exists():
            raise DataError(f"KHATT OCR directory not found: {ocr_dir}")
        if not gt_dir.exists():
            raise DataError(f"KHATT GT directory not found: {gt_dir}")

        pairs = _pair_by_stem(ocr_dir, gt_dir)

        effective_limit = limit if limit is not None else self._default_limit
        if effective_limit is not None:
            pairs = pairs[:effective_limit]

        samples: list[OCRSample] = []
        skipped = 0

        for ocr_path, gt_path in pairs:
            ocr_text = _read_ocr_file(ocr_path)
            gt_text = _read_gt_file(gt_path)

            if ocr_text == "":
                logger.warning("Empty OCR file skipped: %s", ocr_path.name)
                skipped += 1
                continue

            if gt_text == "":
                logger.warning("Empty GT file skipped: %s", gt_path.name)
                skipped += 1
                continue

            samples.append(OCRSample(
                sample_id=ocr_path.stem,
                dataset="KHATT",
                font=None,
                split=split,
                ocr_text=ocr_text,
                gt_text=gt_text,
                ocr_path=ocr_path,
                gt_path=gt_path,
            ))

        if skipped:
            logger.warning("KHATT-%s: skipped %d empty files.", split, skipped)

        logger.info("Loaded %d KHATT-%s samples.", len(samples), split)
        return samples

    def load_all(
        self,
        limit: Optional[int] = None,
    ) -> dict[str, list[OCRSample]]:
        """Load all available datasets.

        Datasets where GT is unavailable are omitted with a warning.

        Args:
            limit: If set, cap each dataset at this many samples.

        Returns:
            Dict mapping dataset key → list[OCRSample].
            Possible keys: "PATS-A01-Akhbar", "PATS-A01-Andalus",
                           "KHATT-train", "KHATT-validation".
        """
        result: dict[str, list[OCRSample]] = {}

        for font in ("Akhbar", "Andalus"):
            key = f"PATS-A01-{font}"
            try:
                result[key] = self.load_pats(font=font, limit=limit)
            except DataError as exc:
                logger.warning("Skipping %s: %s", key, exc)

        for split in ("train", "validation"):
            key = f"KHATT-{split}"
            try:
                result[key] = self.load_khatt(split=split, limit=limit)
            except DataError as exc:
                logger.warning("Skipping %s: %s", key, exc)

        return result

    def iter_samples(
        self,
        dataset: str,
        limit: Optional[int] = None,
    ) -> Iterator[OCRSample]:
        """Iterate samples for a named dataset without loading all into memory.

        Args:
            dataset: E.g. "KHATT-train", "PATS-A01-Akhbar".
            limit: Stop after this many samples.

        Yields:
            OCRSample objects.

        Raises:
            DataError: If the dataset key is unrecognised.
        """
        if dataset.startswith("PATS-A01-"):
            font = dataset.split("-", 2)[2]
            samples = self.load_pats(font=font, limit=limit)
        elif dataset.startswith("KHATT-"):
            split = dataset.split("-", 1)[1]
            samples = self.load_khatt(split=split, limit=limit)
        else:
            raise DataError(f"Unknown dataset key: '{dataset}'")

        yield from samples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pats_gt(self, font: str, n_lines: int) -> tuple[list[str], Path]:
        """Load PATS GT lines, auto-deriving the GT file path from the font name.

        GT files live at ``{gt_root}/PATS_A01_Dataset/A01-{font}Text.txt``
        and are encoded in cp1256 (Windows Arabic code page).
        Exception: font "Traditional" uses ``TraditionalText.txt`` (no prefix).

        Args:
            font: Font name, e.g. "Akhbar", "Andalus", "Traditional".
            n_lines: Expected number of lines (used for logging).

        Returns:
            Tuple of (lines, gt_path) where lines is 0-indexed
            (line N is at index N-1).

        Raises:
            DataError: If the GT file does not exist or cannot be read.
        """
        gt_filename = (
            "TraditionalText.txt" if font == "Traditional"
            else f"A01-{font}Text.txt"
        )
        gt_path = self._gt_root / "PATS_A01_Dataset" / gt_filename

        if not gt_path.exists():
            raise DataError(
                f"PATS GT file not found: {gt_path}\n"
                f"Expected a cp1256-encoded file where line N → {font}_N.txt."
            )

        try:
            content = gt_path.read_text(encoding="cp1256", errors="replace").strip()
        except OSError as exc:
            raise DataError(
                f"Could not read PATS GT file {gt_path}: {exc}"
            ) from exc

        lines = content.splitlines()
        logger.info(
            "Loaded %d PATS GT lines from %s (expected ~%d for font %s).",
            len(lines), gt_path.name, n_lines, font,
        )
        return lines, gt_path


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _read_ocr_file(path: Path) -> str:
    """Read an OCR text file, apply strip_repetitions, return cleaned string.

    Returns empty string for 0-byte or unreadable files.
    Encoding: UTF-8 with errors='replace' to handle encoding corruption.
    Length is capped at _MAX_TEXT_LEN after repetition stripping.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning("Could not read OCR file %s: %s", path, exc)
        return ""

    if not raw:
        return ""

    # Cap runaway repetitions
    cleaned = strip_repetitions(raw)

    # Hard cap on length to prevent downstream memory issues
    if len(cleaned) > _MAX_TEXT_LEN:
        cleaned = cleaned[:_MAX_TEXT_LEN]

    return cleaned


def _read_gt_file(path: Path) -> str:
    """Read a GT text file.

    Encoding: UTF-8 with errors='replace'.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning("Could not read GT file %s: %s", path, exc)
        return ""


def _pair_by_stem(ocr_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching (ocr_path, gt_path) pairs by filename stem.

    Intersection of OCR and GT .txt files, sorted alphabetically by stem.
    Logs any unmatched stems as warnings.

    Args:
        ocr_dir: Directory containing OCR .txt files.
        gt_dir: Directory containing GT .txt files.

    Returns:
        Sorted list of (ocr_path, gt_path) tuples.
    """
    ocr_stems = {p.stem: p for p in ocr_dir.glob("*.txt")}
    gt_stems = {p.stem: p for p in gt_dir.glob("*.txt")}

    common = set(ocr_stems) & set(gt_stems)
    ocr_only = set(ocr_stems) - common
    gt_only = set(gt_stems) - common

    if ocr_only:
        logger.warning(
            "%d OCR files have no matching GT (showing first 5): %s",
            len(ocr_only),
            sorted(ocr_only)[:5],
        )
    if gt_only:
        logger.warning(
            "%d GT files have no matching OCR (showing first 5): %s",
            len(gt_only),
            sorted(gt_only)[:5],
        )

    return [
        (ocr_stems[stem], gt_stems[stem])
        for stem in sorted(common)
    ]


def _pats_line_number(path: Path) -> int:
    """Extract numeric line index from a PATS filename.

    E.g. "Akhbar_123.txt" → 123, "Andalus_7.txt" → 7.
    Returns 0 if no number is found (sorts unrecognised files first).
    """
    import re
    match = re.search(r'_(\d+)$', path.stem)
    return int(match.group(1)) if match else 0


def _extract_font_from_path(ocr_path: Path) -> str:
    """Derive font name from the OCR directory name.

    E.g., directory "A01-Akhbar" → "Akhbar".
    """
    dir_name = ocr_path.parent.name
    if "-" in dir_name:
        return dir_name.split("-", 1)[1]
    return dir_name
