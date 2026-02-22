"""Statistical tests for comparing OCR correction systems (Phase 6).

Provides paired t-tests, Cohen's d effect sizes, Bonferroni correction,
and bootstrap confidence intervals for CER/WER comparisons.

Graceful degradation: if scipy is not installed, paired_ttest() falls back to
a simple normal approximation (less accurate for small samples, acceptable for
the typical N > 1000 samples per dataset encountered in this project).
"""

import logging
import math
import random
from typing import Optional

logger = logging.getLogger(__name__)

# Optional heavy dependency
_scipy_available = False
try:
    from scipy import stats as _scipy_stats  # type: ignore[import]
    _scipy_available = True
except ImportError:
    logger.warning(
        "scipy not installed -- statistical tests will use a normal approximation. "
        "Install with: pip install scipy"
    )


class StatsTester:
    """Paired statistical tests for comparing CER/WER across correction systems.

    All test methods accept paired score lists (one score per sample, same order).
    Typically: scores_a = Phase 2 per-sample CERs, scores_b = combo per-sample CERs.

    Usage::

        tester = StatsTester()
        result = tester.paired_ttest(phase2_cers, combo_cers)
        effect = tester.cohens_d(phase2_cers, combo_cers)
        ci = tester.bootstrap_ci(phase2_cers, combo_cers)

        # Bonferroni-corrected comparison of all combos vs Phase 2
        results = tester.compare_all(baseline_cers, {
            "pair_conf_rules": pair_cers,
            "full_system": full_cers,
        })
    """

    # ------------------------------------------------------------------
    # Basic tests
    # ------------------------------------------------------------------

    def paired_ttest(
        self,
        scores_a: list[float],
        scores_b: list[float],
        alpha: float = 0.05,
    ) -> dict:
        """Paired two-tailed t-test comparing scores_a vs scores_b.

        Uses scipy.stats.ttest_rel if available; falls back to a normal
        approximation for large samples (N >= 30) otherwise.

        Args:
            scores_a: Per-sample scores for system A (e.g. Phase 2 CER).
            scores_b: Per-sample scores for system B (e.g. combo CER).
            alpha: Significance level (default 0.05).

        Returns:
            Dict with keys: t_stat, p_value, significant, n, mean_diff, std_diff.
            mean_diff = mean(scores_a - scores_b); positive = A worse than B.
        """
        n = min(len(scores_a), len(scores_b))
        if n < 2:
            return {
                "t_stat": 0.0,
                "p_value": 1.0,
                "significant": False,
                "n": n,
                "mean_diff": 0.0,
                "std_diff": 0.0,
                "note": "Insufficient samples.",
            }

        diffs = [a - b for a, b in zip(scores_a[:n], scores_b[:n])]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)
        std_diff = math.sqrt(var_diff)

        if _scipy_available:
            t_stat, p_value = _scipy_stats.ttest_rel(scores_a[:n], scores_b[:n])
            t_stat = float(t_stat)
            p_value = float(p_value)
        else:
            # Normal approximation (valid for large N)
            se = std_diff / math.sqrt(n)
            t_stat = mean_diff / se if se > 0 else 0.0
            # Two-tailed p from normal CDF approximation
            p_value = _normal_pvalue(abs(t_stat))

        return {
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < alpha,
            "n": n,
            "mean_diff": round(mean_diff, 6),
            "std_diff": round(std_diff, 6),
        }

    def cohens_d(self, scores_a: list[float], scores_b: list[float]) -> float:
        """Cohen's d effect size for paired samples.

        d = mean(scores_a - scores_b) / std(scores_a - scores_b)

        Interpretation: |d| < 0.2 small, 0.2-0.5 medium, > 0.5 large.

        Args:
            scores_a: Per-sample scores for system A.
            scores_b: Per-sample scores for system B.

        Returns:
            Cohen's d (positive if A > B on average).
        """
        n = min(len(scores_a), len(scores_b))
        if n < 2:
            return 0.0

        diffs = [a - b for a, b in zip(scores_a[:n], scores_b[:n])]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)
        std_diff = math.sqrt(var_diff)

        return round(mean_diff / std_diff, 4) if std_diff > 0 else 0.0

    def bonferroni_correct(
        self,
        p_values: list[float],
        alpha: float = 0.05,
    ) -> list[bool]:
        """Bonferroni correction for multiple comparisons.

        Adjusted threshold = alpha / n_tests. A test is significant if
        its p_value < adjusted_threshold.

        Args:
            p_values: List of raw p-values.
            alpha: Family-wise error rate (default 0.05).

        Returns:
            List of booleans — True if the test is significant after correction.
        """
        n = len(p_values)
        if n == 0:
            return []
        threshold = alpha / n
        return [p < threshold for p in p_values]

    def bootstrap_ci(
        self,
        scores_a: list[float],
        scores_b: list[float],
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for the mean CER difference (a - b).

        Args:
            scores_a: Per-sample scores for system A.
            scores_b: Per-sample scores for system B.
            n_bootstrap: Number of bootstrap iterations.
            alpha: Coverage level (default 0.05 -> 95% CI).
            seed: Random seed for reproducibility.

        Returns:
            (lower, upper) bounds of the (1 - alpha) CI for mean(a - b).
        """
        n = min(len(scores_a), len(scores_b))
        if n < 2:
            return (0.0, 0.0)

        pairs = list(zip(scores_a[:n], scores_b[:n]))
        rng = random.Random(seed)
        boot_means: list[float] = []

        for _ in range(n_bootstrap):
            sample = rng.choices(pairs, k=n)
            boot_mean = sum(a - b for a, b in sample) / n
            boot_means.append(boot_mean)

        boot_means.sort()
        lo_idx = int((alpha / 2) * n_bootstrap)
        hi_idx = int((1 - alpha / 2) * n_bootstrap)
        lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
        hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

        return (round(boot_means[lo_idx], 6), round(boot_means[hi_idx], 6))

    # ------------------------------------------------------------------
    # Multi-system comparison
    # ------------------------------------------------------------------

    def compare_all(
        self,
        baseline: list[float],
        systems: dict[str, list[float]],
        alpha: float = 0.05,
    ) -> dict:
        """Bonferroni-corrected paired t-tests for all systems vs baseline.

        Args:
            baseline: Per-sample CER scores for Phase 2 (reference).
            systems: Dict mapping system_name -> per-sample CER scores.
            alpha: Family-wise error rate (default 0.05).

        Returns:
            Dict mapping system_name -> {t_stat, p_value, p_corrected,
            significant, cohens_d, ci_95, mean_diff, n}.
        """
        if not systems:
            return {}

        system_names = list(systems.keys())
        raw_results: dict[str, dict] = {}
        p_values: list[float] = []

        for name in system_names:
            res = self.paired_ttest(baseline, systems[name], alpha=1.0)  # raw p
            d = self.cohens_d(baseline, systems[name])
            ci = self.bootstrap_ci(baseline, systems[name])
            raw_results[name] = {
                "t_stat":    res["t_stat"],
                "p_value":   res["p_value"],
                "cohens_d":  d,
                "ci_95":     list(ci),
                "mean_diff": res["mean_diff"],
                "n":         res["n"],
            }
            p_values.append(res["p_value"])

        sig_flags = self.bonferroni_correct(p_values, alpha=alpha)
        threshold = alpha / max(len(p_values), 1)

        final: dict[str, dict] = {}
        for i, name in enumerate(system_names):
            final[name] = dict(raw_results[name])
            final[name]["p_corrected"] = round(p_values[i] * len(p_values), 6)
            final[name]["significant"] = sig_flags[i]
            final[name]["bonferroni_threshold"] = round(threshold, 6)

        return final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normal_pvalue(z: float) -> float:
    """Two-tailed p-value from standard normal using Abramowitz-Stegun approx."""
    # Only valid for z >= 0
    if z <= 0:
        return 1.0
    # Approximation: Φ(z) from A&S 26.2.17
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    phi_z = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    return round(2.0 * (1.0 - phi_z), 6)
