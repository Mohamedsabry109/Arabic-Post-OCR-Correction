"""Shared utilities for pipeline scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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


def get_train_counterpart(dataset_key: str) -> str:
    """Return the training-split key for a validation-split dataset key.

    Mapping:
      ``PATS-A01-{Font}-val``  ->  ``PATS-A01-{Font}-train``
      ``KHATT-validation``     ->  ``KHATT-train``
    Keys that are already training splits are returned unchanged.
    """
    if dataset_key.endswith("-val"):
        return dataset_key[:-4] + "-train"
    if dataset_key.lower().endswith("-validation"):
        return dataset_key[: -len("-validation")] + "-train"
    return dataset_key


def get_training_dataset_names(all_dataset_names: list[str]) -> list[str]:
    """Return training-split dataset keys from *all_dataset_names*.

    If the list already contains training keys (ending in ``-train``), those
    are returned as-is.  Otherwise every validation key is converted to its
    training counterpart via :func:`get_train_counterpart` so that pipelines
    work correctly even when ``config.yaml`` lists only validation datasets.
    """
    train_keys = [k for k in all_dataset_names if k.lower().endswith("-train")]
    if train_keys:
        return train_keys
    # Derive from validation keys
    derived = [get_train_counterpart(k) for k in all_dataset_names
               if k.lower().endswith("-val") or k.lower().endswith("-validation")]
    if derived:
        logger.info(
            "get_training_dataset_names: config has only val splits; "
            "derived training keys: %s", derived,
        )
    return derived


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


# ---------------------------------------------------------------------------
# Runaway sample splitting
# ---------------------------------------------------------------------------

DEFAULT_RUNAWAY_RATIO_THRESHOLD = 5.0
"""Default OCR/GT length ratio above which a sample is classified as runaway.

Overridden by ``evaluation.runaway_ratio_threshold`` in config.yaml.
"""


def split_runaway_samples(
    samples: list,
    threshold: float = DEFAULT_RUNAWAY_RATIO_THRESHOLD,
) -> tuple[list, list, dict]:
    """Separate samples into normal vs runaway using OCR/GT length ratio.

    Works with both CorrectedSample objects (``cs.sample.gt_text``) and
    plain dicts (``d["gt_text"]``).

    Uses the same criterion as Phase 1's ``calculate_metrics_split``:
    a sample is *runaway* when ``len(ocr_text) / len(gt_text) > threshold``.

    Returns:
        (normal_samples, runaway_samples, data_quality_dict)
    """
    from src.data.text_utils import normalise_arabic

    normal: list = []
    runaway: list = []
    for s in samples:
        # Support both CorrectedSample and dict
        if hasattr(s, "sample"):
            gt_text = s.sample.gt_text
            ocr_text = s.sample.ocr_text
        else:
            gt_text = s.get("gt_text", "")
            ocr_text = s.get("ocr_text", "")
        gt_len = max(len(normalise_arabic(gt_text)), 1)
        ocr_len = len(normalise_arabic(ocr_text))
        if ocr_len / gt_len > threshold:
            runaway.append(s)
        else:
            normal.append(s)
    n = len(samples)
    data_quality = {
        "total_samples": n,
        "normal_samples": len(normal),
        "runaway_samples": len(runaway),
        "runaway_percentage": round(len(runaway) / max(n, 1) * 100, 2),
        "runaway_ratio_threshold": threshold,
        "description": (
            f"{len(runaway)} samples ({len(runaway)/max(n,1)*100:.1f}%) "
            f"have OCR output >{threshold}x longer than GT (Qaari repetition bug)."
        ),
    }
    return normal, runaway, data_quality


# ---------------------------------------------------------------------------
# Phase 2 metrics loading (new format with backward compat)
# ---------------------------------------------------------------------------


def load_phase2_full_metrics(phase2_dir: Path, dataset_key: str) -> Optional[dict]:
    """Load the full Phase 2 per-dataset metrics.json dict.

    Returns the entire parsed JSON dict (not just a sub-key), or None if
    the file does not exist.  Callers can then pick whichever variant
    (all / normal_only, with / without diacritics) they need.
    """
    path = phase2_dir / dataset_key / "metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load Phase 2 metrics for %s: %s", dataset_key, exc)
        return None


def _pick_corrected_key(data: dict, exclude_runaway: bool) -> dict:
    """Pick the best 'corrected' sub-dict from a metrics.json dict.

    Handles both old format (``corrected``) and new format
    (``corrected_all`` / ``corrected_normal_only``).
    """
    if exclude_runaway:
        return (
            data.get("corrected_normal_only")
            or data.get("corrected_all")
            or data.get("corrected", {})
        )
    return data.get("corrected_all") or data.get("corrected", {})


def pick_phase2_variant(
    p2_data: dict,
    exclude_runaway: bool,
) -> tuple[dict, dict, str]:
    """Pick the corrected + no-diacritics Phase 2 variant to compare against.

    Handles both old-format (``corrected`` / ``corrected_no_diacritics``)
    and new-format (``corrected_all`` / ``corrected_normal_only`` etc.) keys.

    Args:
        p2_data: Full metrics.json dict from Phase 2.
        exclude_runaway: If True, prefer normal_only; else prefer all.

    Returns:
        (corrected_dict, corrected_nd_dict, source_label) where source_label
        is "all" or "normal_only".
    """
    if exclude_runaway:
        # Prefer normal_only; fall back to all; fall back to old key
        corrected = (
            p2_data.get("corrected_normal_only")
            or p2_data.get("corrected_all")
            or p2_data.get("corrected", {})
        )
        corrected_nd = (
            p2_data.get("corrected_normal_only_no_diacritics")
            or p2_data.get("corrected_all_no_diacritics")
            or p2_data.get("corrected_no_diacritics", {})
        )
        source = "normal_only" if p2_data.get("corrected_normal_only") else "all"
    else:
        # Prefer all; fall back to old key
        corrected = p2_data.get("corrected_all") or p2_data.get("corrected", {})
        corrected_nd = (
            p2_data.get("corrected_all_no_diacritics")
            or p2_data.get("corrected_no_diacritics", {})
        )
        source = "all"
    return corrected, corrected_nd, source
