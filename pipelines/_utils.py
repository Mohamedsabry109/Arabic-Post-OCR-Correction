"""Shared utilities for pipeline scripts."""

import json
from pathlib import Path
from typing import Optional


def load_sample_list(path: Path) -> tuple[set[str], list[str]]:
    """Load a test-samples JSON file (e.g. ``data/test_samples.json``).

    Returns:
        Tuple of (sample_ids set, dataset_keys list).
        ``sample_ids`` contains all sample IDs from the file.
        ``dataset_keys`` contains the dataset names that have samples in the file.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    sample_ids = set(data["sample_ids"])
    # Dataset keys come from meta.by_dataset (keys with count > 0)
    dataset_keys = list(data.get("meta", {}).get("by_dataset", {}).keys())
    return sample_ids, dataset_keys


def resolve_datasets(config: dict, datasets_arg: Optional[list[str]]) -> list[str]:
    """Return the ordered list of dataset keys to process.

    Priority:
      1. ``--datasets`` CLI arg, if provided (any subset, any order).
      2. ``config['datasets']`` list (names only), as the full default.

    Args:
        config: Parsed config.yaml as a nested dict.
        datasets_arg: List from CLI ``--datasets`` flag, or ``None``.

    Returns:
        Ordered list of dataset key strings, e.g.
        ``["PATS-A01-Akhbar", "KHATT-train"]``.
    """
    if datasets_arg:
        return datasets_arg
    return [entry["name"] for entry in config.get("datasets", [])]
