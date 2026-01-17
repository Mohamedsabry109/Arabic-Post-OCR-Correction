"""
Data loader for Arabic Post-OCR Correction datasets.

This module provides functionality to load OCR predictions and ground truth
text files, validate alignment between them, and return paired data tuples.

Supports multiple datasets with the following structure:
    data/
    ├── Original/           # Ground truth files
    │   ├── PATS-A01/
    │   │   ├── file1.txt
    │   │   └── file2.txt
    │   └── KHATT/
    │       ├── file1.txt
    │       └── file2.txt
    └── Predictions/        # OCR predictions
        ├── PATS-A01/
        │   ├── file1.txt
        │   └── file2.txt
        └── KHATT/
            ├── file1.txt
            └── file2.txt

Example:
    >>> from src.data_loader import DataLoader
    >>> loader = DataLoader("data/Original", "data/Predictions")
    >>> pairs = loader.load_dataset("PATS-A01")
    >>> print(f"Loaded {len(pairs)} pairs")
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .utils import ensure_utf8, clean_text, is_arabic


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    num_samples: int
    ground_truth_path: Path
    predictions_path: Path
    avg_text_length: float
    total_characters: int


class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass


class AlignmentError(DataLoaderError):
    """Raised when ground truth and predictions are misaligned."""
    pass


class DataLoader:
    """
    Load and validate Arabic OCR datasets.

    Loads paired OCR predictions and ground truth text files,
    validates alignment, and handles Arabic text encoding.

    Attributes:
        ground_truth_base: Base path for ground truth files.
        predictions_base: Base path for OCR prediction files.
        file_extension: Extension for text files (default: .txt).

    Example:
        >>> loader = DataLoader(
        ...     ground_truth_base="data/Original",
        ...     predictions_base="data/Predictions"
        ... )
        >>> pairs = loader.load_dataset("PATS-A01")
        >>> for ocr_text, ground_truth in pairs[:5]:
        ...     print(f"OCR: {ocr_text[:50]}...")
        ...     print(f"GT:  {ground_truth[:50]}...")
    """

    def __init__(
        self,
        ground_truth_base: str,
        predictions_base: str,
        file_extension: str = ".txt"
    ):
        """
        Initialize the DataLoader.

        Args:
            ground_truth_base: Base directory containing ground truth datasets.
            predictions_base: Base directory containing OCR predictions.
            file_extension: File extension to look for. Defaults to ".txt".

        Raises:
            FileNotFoundError: If base directories don't exist.
        """
        self.ground_truth_base = Path(ground_truth_base)
        self.predictions_base = Path(predictions_base)
        self.file_extension = file_extension

        # Validate base directories exist
        if not self.ground_truth_base.exists():
            raise FileNotFoundError(
                f"Ground truth directory not found: {self.ground_truth_base}"
            )
        if not self.predictions_base.exists():
            raise FileNotFoundError(
                f"Predictions directory not found: {self.predictions_base}"
            )

        logger.info(
            f"DataLoader initialized with ground_truth={self.ground_truth_base}, "
            f"predictions={self.predictions_base}"
        )

    def list_datasets(self) -> List[str]:
        """
        List available datasets in the ground truth directory.

        Returns:
            List of dataset names (subdirectory names).

        Example:
            >>> loader = DataLoader("data/Original", "data/Predictions")
            >>> datasets = loader.list_datasets()
            >>> print(datasets)
            ['KHATT', 'PATS-A01']
        """
        datasets = []
        for item in self.ground_truth_base.iterdir():
            if item.is_dir():
                # Check if corresponding predictions folder exists
                pred_path = self.predictions_base / item.name
                if pred_path.exists():
                    datasets.append(item.name)
                else:
                    logger.warning(
                        f"Dataset '{item.name}' has no predictions folder"
                    )
        return sorted(datasets)

    def _get_files(self, directory: Path) -> List[Path]:
        """
        Get all text files in a directory, sorted by name.

        Args:
            directory: Directory to search.

        Returns:
            Sorted list of file paths.
        """
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() == self.file_extension.lower()
        )
        return files

    def _read_file(self, file_path: Path) -> str:
        """
        Read a text file with proper UTF-8 handling.

        Args:
            file_path: Path to the text file.

        Returns:
            File contents as a string.

        Raises:
            DataLoaderError: If file cannot be read.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ensure_utf8(content)
        except UnicodeDecodeError:
            # Try alternative encodings for Arabic
            for encoding in ['cp1256', 'iso-8859-6', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.warning(
                        f"File {file_path} read with {encoding} encoding"
                    )
                    return content
                except (UnicodeDecodeError, UnicodeError):
                    continue
            raise DataLoaderError(
                f"Cannot decode file {file_path} with any known encoding"
            )
        except IOError as e:
            raise DataLoaderError(f"Cannot read file {file_path}: {e}")

    def validate_alignment(
        self,
        dataset_name: str,
        strict: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate that ground truth and predictions files are aligned.

        Args:
            dataset_name: Name of the dataset to validate.
            strict: If True, raise error on misalignment. If False,
                return only matched files. Defaults to True.

        Returns:
            Tuple of (matched_files, gt_only, pred_only) where:
            - matched_files: Files present in both directories
            - gt_only: Files only in ground truth
            - pred_only: Files only in predictions

        Raises:
            AlignmentError: If strict=True and files don't align.
            FileNotFoundError: If dataset directory doesn't exist.
        """
        gt_path = self.ground_truth_base / dataset_name
        pred_path = self.predictions_base / dataset_name

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Ground truth dataset not found: {gt_path}"
            )
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Predictions dataset not found: {pred_path}"
            )

        # Get file names (without path)
        gt_files = {f.name for f in self._get_files(gt_path)}
        pred_files = {f.name for f in self._get_files(pred_path)}

        matched = sorted(gt_files & pred_files)
        gt_only = sorted(gt_files - pred_files)
        pred_only = sorted(pred_files - gt_files)

        if strict and (gt_only or pred_only):
            msg = f"Alignment error in dataset '{dataset_name}':\n"
            if gt_only:
                msg += f"  Files only in ground truth: {gt_only[:5]}...\n"
            if pred_only:
                msg += f"  Files only in predictions: {pred_only[:5]}..."
            raise AlignmentError(msg)

        logger.info(
            f"Dataset '{dataset_name}': {len(matched)} matched, "
            f"{len(gt_only)} GT-only, {len(pred_only)} pred-only"
        )

        return matched, gt_only, pred_only

    def load_dataset(
        self,
        dataset_name: str,
        clean: bool = True,
        strict_alignment: bool = False,
        limit: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """
        Load a dataset as pairs of (ocr_text, ground_truth).

        Args:
            dataset_name: Name of the dataset to load.
            clean: If True, clean whitespace in texts. Defaults to True.
            strict_alignment: If True, fail if files don't align.
                Defaults to False (load only matched files).
            limit: Maximum number of pairs to load. None for all.

        Returns:
            List of (ocr_text, ground_truth) tuples.

        Raises:
            FileNotFoundError: If dataset doesn't exist.
            AlignmentError: If strict_alignment=True and files mismatch.

        Example:
            >>> loader = DataLoader("data/Original", "data/Predictions")
            >>> pairs = loader.load_dataset("PATS-A01", limit=100)
            >>> print(f"Loaded {len(pairs)} pairs")
            Loaded 100 pairs
        """
        gt_path = self.ground_truth_base / dataset_name
        pred_path = self.predictions_base / dataset_name

        # Validate alignment
        matched_files, _, _ = self.validate_alignment(
            dataset_name, strict=strict_alignment
        )

        if not matched_files:
            logger.warning(f"No matched files found in dataset '{dataset_name}'")
            return []

        # Apply limit if specified
        if limit is not None:
            matched_files = matched_files[:limit]

        pairs = []
        for filename in matched_files:
            gt_file = gt_path / filename
            pred_file = pred_path / filename

            try:
                gt_text = self._read_file(gt_file)
                pred_text = self._read_file(pred_file)

                if clean:
                    gt_text = clean_text(gt_text)
                    pred_text = clean_text(pred_text)

                # Skip empty pairs
                if not gt_text.strip() or not pred_text.strip():
                    logger.debug(f"Skipping empty file: {filename}")
                    continue

                pairs.append((pred_text, gt_text))

            except DataLoaderError as e:
                logger.warning(f"Error loading {filename}: {e}")
                continue

        logger.info(
            f"Loaded {len(pairs)} pairs from dataset '{dataset_name}'"
        )
        return pairs

    def load_multiple_datasets(
        self,
        dataset_names: List[str],
        clean: bool = True,
        limit_per_dataset: Optional[int] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load multiple datasets at once.

        Args:
            dataset_names: List of dataset names to load.
            clean: If True, clean whitespace in texts.
            limit_per_dataset: Max pairs per dataset. None for all.

        Returns:
            Dictionary mapping dataset name to list of pairs.

        Example:
            >>> loader = DataLoader("data/Original", "data/Predictions")
            >>> data = loader.load_multiple_datasets(
            ...     ["PATS-A01", "KHATT"],
            ...     limit_per_dataset=50
            ... )
            >>> for name, pairs in data.items():
            ...     print(f"{name}: {len(pairs)} pairs")
        """
        results = {}
        for name in dataset_names:
            try:
                pairs = self.load_dataset(
                    name, clean=clean, limit=limit_per_dataset
                )
                results[name] = pairs
            except FileNotFoundError as e:
                logger.error(f"Dataset '{name}' not found: {e}")
                results[name] = []
        return results

    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """
        Get information about a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            DatasetInfo with statistics about the dataset.

        Example:
            >>> loader = DataLoader("data/Original", "data/Predictions")
            >>> info = loader.get_dataset_info("PATS-A01")
            >>> print(f"Samples: {info.num_samples}")
        """
        gt_path = self.ground_truth_base / dataset_name
        pred_path = self.predictions_base / dataset_name

        matched_files, _, _ = self.validate_alignment(
            dataset_name, strict=False
        )

        # Calculate statistics
        total_chars = 0
        for filename in matched_files:
            try:
                gt_text = self._read_file(gt_path / filename)
                total_chars += len(gt_text)
            except DataLoaderError:
                pass

        avg_length = total_chars / len(matched_files) if matched_files else 0

        return DatasetInfo(
            name=dataset_name,
            num_samples=len(matched_files),
            ground_truth_path=gt_path,
            predictions_path=pred_path,
            avg_text_length=avg_length,
            total_characters=total_chars
        )


def load_single_file(file_path: str, clean: bool = True) -> str:
    """
    Convenience function to load a single text file.

    Args:
        file_path: Path to the text file.
        clean: If True, clean whitespace.

    Returns:
        File contents as string.

    Example:
        >>> text = load_single_file("data/sample.txt")
        >>> print(text[:100])
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try cp1256 (common Arabic encoding)
        with open(path, 'r', encoding='cp1256') as f:
            content = f.read()

    if clean:
        content = clean_text(content)

    return content
