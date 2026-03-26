"""LLM error analysis for Phase 4D self-reflective prompting.

Analyses predictions made by the LLM on training-split samples (where
ground truth is available) to identify systematic failure patterns.
These patterns are then injected into the inference prompt for validation
splits, allowing the model to be aware of its own known weaknesses.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.analysis.error_analyzer import ErrorAnalyzer, ErrorType

logger = logging.getLogger(__name__)

# All ErrorType values except UNKNOWN (not meaningful for self-reflection)
_TYPED_ERRORS: list[str] = [
    et.value for et in ErrorType if et != ErrorType.UNKNOWN
]


class LLMErrorAnalyzer:
    """Analyse LLM correction outputs to extract systematic failure patterns.

    Compares LLM predictions against ground truth on training-split samples
    and aggregates per-ErrorType statistics:

    - ``baseline``:     errors in OCR text (what the model is trying to fix)
    - ``residual``:     errors in LLM output (errors that remain or were added)
    - ``fixed``:        baseline errors the LLM removed
    - ``introduced``:   new errors the LLM created that were NOT in OCR

    Usage::

        analyzer = LLMErrorAnalyzer()
        results = []
        for record in training_corrections:
            r = analyzer.analyse_sample(
                ocr_text=record["ocr_text"],
                gt_text=record["gt_text"],
                llm_text=record["corrected_text"],
                sample_id=record["sample_id"],
                dataset=record.get("dataset", ""),
            )
            results.append(r)
        insights = analyzer.aggregate(results, dataset_type="PATS-A01")
    """

    def __init__(self) -> None:
        self._error_analyzer = ErrorAnalyzer()

    def analyse_sample(
        self,
        ocr_text: str,
        gt_text: str,
        llm_text: str,
        sample_id: str = "",
        dataset: str = "",
    ) -> dict:
        """Compute per-ErrorType breakdown for one sample.

        Creates minimal stub objects to call ErrorAnalyzer without needing
        full OCRSample instances (following the existing pattern in run_phase5.py).

        Args:
            ocr_text:  Raw OCR prediction from Qaari.
            gt_text:   Ground truth text (available on training splits).
            llm_text:  Corrected text produced by the LLM.
            sample_id: Optional sample identifier for logging.
            dataset:   Optional dataset label.

        Returns:
            Dict mapping each ErrorType value (except UNKNOWN) to::

                {
                    "baseline":   int,  # OCR errors of this type vs GT
                    "residual":   int,  # LLM output errors of this type vs GT
                    "fixed":      int,  # baseline errors removed by LLM
                    "introduced": int,  # new errors LLM introduced
                }
        """
        class _Stub:
            pass

        s_ocr = _Stub()
        s_ocr.sample_id = sample_id  # type: ignore[attr-defined]
        s_ocr.dataset   = dataset    # type: ignore[attr-defined]
        s_ocr.gt_text   = gt_text    # type: ignore[attr-defined]
        s_ocr.ocr_text  = ocr_text   # type: ignore[attr-defined]

        s_llm = _Stub()
        s_llm.sample_id = sample_id  # type: ignore[attr-defined]
        s_llm.dataset   = dataset    # type: ignore[attr-defined]
        s_llm.gt_text   = gt_text    # type: ignore[attr-defined]
        s_llm.ocr_text  = llm_text   # type: ignore[attr-defined]

        try:
            err_ocr = self._error_analyzer.analyse_sample(s_ocr)  # type: ignore[arg-type]
            err_llm = self._error_analyzer.analyse_sample(s_llm)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Error analysis failed for sample %s: %s", sample_id, exc
            )
            return {
                k: {"baseline": 0, "residual": 0, "fixed": 0, "introduced": 0}
                for k in _TYPED_ERRORS
            }

        # Count per-type errors for OCR (baseline) and LLM (residual)
        baseline: dict[str, int] = {k: 0 for k in _TYPED_ERRORS}
        residual: dict[str, int] = {k: 0 for k in _TYPED_ERRORS}

        for ce in err_ocr.char_errors:
            k = ce.error_type.value
            if k in baseline:
                baseline[k] += 1

        for ce in err_llm.char_errors:
            k = ce.error_type.value
            if k in residual:
                residual[k] += 1

        result: dict = {}
        for k in _TYPED_ERRORS:
            b = baseline[k]
            r = residual[k]
            # fixed = errors that existed in OCR but not in LLM output
            fixed = max(0, b - r)
            # introduced = errors in LLM output not accounted for by unfixed OCR errors
            # = max(0, residual - (baseline - fixed)) = max(0, r - b + fixed)
            introduced = max(0, r - b + fixed)
            result[k] = {
                "baseline":   b,
                "residual":   r,
                "fixed":      fixed,
                "introduced": introduced,
            }

        return result

    def aggregate(
        self,
        all_results: list[dict],
        dataset_type: str,
    ) -> dict:
        """Aggregate per-sample results into dataset-level insights.

        Args:
            all_results:  List of dicts from analyse_sample().
            dataset_type: Dataset type label, e.g. "PATS-A01" or "KHATT".

        Returns:
            Insights JSON dict with meta, overall, and by_type sections::

                {
                    "meta": {"dataset_type": ..., "total_samples": N, "generated_at": ...},
                    "overall": {"total_baseline_errors": N, "total_fixed": N,
                                "total_introduced": N, "fix_rate": 0.XX,
                                "introduction_rate": 0.XX},
                    "by_type": {
                        "taa_marbuta": {"baseline": N, "fixed": N, "introduced": N,
                                        "fix_rate": 0.XX, "introduction_rate": 0.XX},
                        ...
                    }
                }
        """
        totals: dict[str, dict[str, int]] = {
            k: {"baseline": 0, "fixed": 0, "introduced": 0}
            for k in _TYPED_ERRORS
        }

        for sample_result in all_results:
            for k in _TYPED_ERRORS:
                entry = sample_result.get(k, {})
                totals[k]["baseline"]   += entry.get("baseline", 0)
                totals[k]["fixed"]      += entry.get("fixed", 0)
                totals[k]["introduced"] += entry.get("introduced", 0)

        total_baseline   = sum(v["baseline"]   for v in totals.values())
        total_fixed      = sum(v["fixed"]      for v in totals.values())
        total_introduced = sum(v["introduced"] for v in totals.values())

        def _rate(num: int, den: int) -> Optional[float]:
            return round(num / den, 4) if den > 0 else None

        by_type: dict = {}
        for k in _TYPED_ERRORS:
            b = totals[k]["baseline"]
            f = totals[k]["fixed"]
            i = totals[k]["introduced"]
            by_type[k] = {
                "baseline":          b,
                "fixed":             f,
                "introduced":        i,
                "fix_rate":          _rate(f, b),
                "introduction_rate": _rate(i, b),
            }

        return {
            "meta": {
                "dataset_type":  dataset_type,
                "total_samples": len(all_results),
                "generated_at":  _now_iso(),
            },
            "overall": {
                "total_baseline_errors": total_baseline,
                "total_fixed":           total_fixed,
                "total_introduced":      total_introduced,
                "fix_rate":          _rate(total_fixed, total_baseline),
                "introduction_rate": _rate(total_introduced, total_baseline),
            },
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
