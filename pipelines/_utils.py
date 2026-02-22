"""Shared utilities for pipeline scripts."""

from typing import Optional


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
