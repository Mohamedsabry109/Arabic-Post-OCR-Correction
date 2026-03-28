"""Shared utilities for pipeline scripts."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Dataset-group aggregation (PATS-A01 / KHATT)
# ---------------------------------------------------------------------------

# Recognised dataset-type prefixes.  A key that does not match any prefix is
# placed in the "other" group (included in output only if non-empty).
_GROUP_PREFIXES: list[tuple[str, str]] = [
    ("PATS-A01-", "PATS-A01"),
    ("KHATT-",    "KHATT"),
]


def _classify(dataset_key: str) -> str:
    """Return the group name for *dataset_key*."""
    for prefix, group in _GROUP_PREFIXES:
        if dataset_key.startswith(prefix):
            return group
    return "other"


def compute_group_aggregates(
    results: dict[str, dict],
    *,
    cer_key: str = "cer",
    wer_key: str = "wer",
) -> dict[str, dict]:
    """Compute simple (macro-average) CER/WER per dataset group.

    This is a **non-weighted** mean — each dataset contributes equally
    regardless of sample count.

    Args:
        results: Mapping of ``dataset_key`` to a dict that contains at
            least *cer_key* and *wer_key* as ``float`` values.  Typically
            this is either ``{k: MetricResult.to_dict()}`` or a similar
            no-diacritics dict.
        cer_key: Key to read CER from each entry (default ``"cer"``).
        wer_key: Key to read WER from each entry (default ``"wer"``).

    Returns:
        Dict mapping group name (e.g. ``"PATS-A01"``, ``"KHATT"``) to
        ``{"cer": float, "wer": float, "num_datasets": int}``.
        Only groups with at least one dataset are included.
    """
    # Bucket CER/WER values by group.
    buckets: dict[str, list[tuple[float, float]]] = {}
    for ds_key, metrics in results.items():
        group = _classify(ds_key)
        cer = metrics.get(cer_key)
        wer = metrics.get(wer_key)
        if cer is None or wer is None:
            continue
        buckets.setdefault(group, []).append((float(cer), float(wer)))

    aggregates: dict[str, dict] = {}
    for group, values in sorted(buckets.items()):
        n = len(values)
        avg_cer = sum(c for c, _ in values) / n
        avg_wer = sum(w for _, w in values) / n
        aggregates[group] = {
            "cer": round(avg_cer, 6),
            "wer": round(avg_wer, 6),
            "num_datasets": n,
        }
    return aggregates
