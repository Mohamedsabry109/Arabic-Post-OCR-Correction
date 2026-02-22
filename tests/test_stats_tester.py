"""Tests for src/analysis/stats_tester.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.analysis.stats_tester import StatsTester, _normal_pvalue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tester() -> StatsTester:
    return StatsTester()


def _repeat(val: float, n: int) -> list[float]:
    return [val] * n


# ---------------------------------------------------------------------------
# _normal_pvalue
# ---------------------------------------------------------------------------

class TestNormalPvalue:
    def test_z_zero_returns_one(self):
        # z=0 means no difference → p=1.0
        assert _normal_pvalue(0.0) == 1.0

    def test_large_z_gives_small_p(self):
        # z=3.0 → p should be < 0.01
        assert _normal_pvalue(3.0) < 0.01

    def test_z_196_gives_approx_005(self):
        # classic significance threshold
        p = _normal_pvalue(1.96)
        assert 0.04 < p < 0.06

    def test_negative_z_treated_as_nonsignificant(self):
        assert _normal_pvalue(-5.0) == 1.0


# ---------------------------------------------------------------------------
# paired_ttest
# ---------------------------------------------------------------------------

class TestPairedTtest:
    def test_identical_lists_not_significant(self, tester):
        a = [0.1, 0.2, 0.3, 0.15, 0.25] * 20
        result = tester.paired_ttest(a, a)
        assert result["significant"] is False
        assert abs(result["mean_diff"]) < 1e-9

    def test_clearly_different_lists_significant(self, tester):
        # a is always 0.1 higher than b
        a = [0.3] * 200
        b = [0.1] * 200
        result = tester.paired_ttest(a, b)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert abs(result["mean_diff"] - 0.2) < 1e-6

    def test_mean_diff_sign(self, tester):
        # mean_diff = mean(a - b); a > b → positive
        a = [0.5] * 50
        b = [0.3] * 50
        result = tester.paired_ttest(a, b)
        assert result["mean_diff"] > 0

    def test_insufficient_samples_fallback(self, tester):
        result = tester.paired_ttest([0.1], [0.2])
        assert result["n"] == 1
        assert result["significant"] is False
        assert result["p_value"] == 1.0

    def test_empty_lists(self, tester):
        result = tester.paired_ttest([], [])
        assert result["n"] == 0
        assert result["significant"] is False

    def test_uses_shorter_list_length(self, tester):
        a = [0.1] * 100
        b = [0.2] * 60
        result = tester.paired_ttest(a, b)
        assert result["n"] == 60

    def test_result_keys_present(self, tester):
        result = tester.paired_ttest([0.1] * 10, [0.2] * 10)
        for key in ("t_stat", "p_value", "significant", "n", "mean_diff", "std_diff"):
            assert key in result, f"Missing key: {key}"

    def test_p_value_between_0_and_1(self, tester):
        a = [0.1 + i * 0.01 for i in range(50)]
        b = [0.2 + i * 0.01 for i in range(50)]
        result = tester.paired_ttest(a, b)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_identical_lists_returns_zero(self, tester):
        a = [0.3, 0.4, 0.5, 0.2] * 10
        assert tester.cohens_d(a, a) == 0.0

    def test_large_effect_size(self, tester):
        # a and b far apart relative to within-pair variance → large |d|
        import random
        rng = random.Random(0)
        # a centred at 1.0, b centred at 0.0, both with std 0.1 → d ≈ 10
        a = [1.0 + rng.gauss(0, 0.1) for _ in range(200)]
        b = [0.0 + rng.gauss(0, 0.1) for _ in range(200)]
        d = tester.cohens_d(a, b)
        assert abs(d) > 1.0

    def test_sign_convention(self, tester):
        # a > b → d > 0
        a = [0.5] * 50
        b = [0.3] * 50
        assert tester.cohens_d(a, b) > 0

        # a < b → d < 0
        assert tester.cohens_d(b, a) < 0

    def test_insufficient_samples(self, tester):
        assert tester.cohens_d([0.5], [0.3]) == 0.0
        assert tester.cohens_d([], []) == 0.0

    def test_medium_effect_range(self, tester):
        # d = mean(a-b) / std(a-b); use independent noisy samples so std > 0
        import random
        rng = random.Random(42)
        # a ~ N(0.4, 0.2), b ~ N(0.2, 0.2) → mean diff ≈ 0.2, std diff > 0
        a = [rng.gauss(0.4, 0.2) for _ in range(500)]
        b = [rng.gauss(0.2, 0.2) for _ in range(500)]
        d = tester.cohens_d(a, b)
        # Effect exists and is measurable
        assert abs(d) > 0.1


# ---------------------------------------------------------------------------
# bonferroni_correct
# ---------------------------------------------------------------------------

class TestBonferroniCorrect:
    def test_all_significant_no_correction_needed(self, tester):
        # 1 test, p=0.01, threshold=0.05/1=0.05 → significant
        assert tester.bonferroni_correct([0.01], alpha=0.05) == [True]

    def test_multiple_tests_raises_threshold(self, tester):
        # 5 tests, alpha=0.05 → threshold=0.01; p=0.02 no longer significant
        result = tester.bonferroni_correct([0.02, 0.02, 0.02, 0.02, 0.02], alpha=0.05)
        assert all(r is False for r in result)

    def test_only_very_small_p_survives_many_comparisons(self, tester):
        ps = [0.001, 0.01, 0.03, 0.04]       # 4 tests → threshold = 0.05/4 = 0.0125
        result = tester.bonferroni_correct(ps, alpha=0.05)
        assert result[0] is True    # 0.001 < 0.0125 → significant
        assert result[1] is True    # 0.010 < 0.0125 → significant
        assert result[2] is False   # 0.030 > 0.0125 → not significant
        assert result[3] is False   # 0.040 > 0.0125 → not significant

    def test_empty_list(self, tester):
        assert tester.bonferroni_correct([]) == []

    def test_output_length_matches_input(self, tester):
        ps = [0.01, 0.05, 0.1, 0.2, 0.5]
        result = tester.bonferroni_correct(ps)
        assert len(result) == len(ps)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_tuple_of_two_floats(self, tester):
        a = [0.3] * 100
        b = [0.2] * 100
        lo, hi = tester.bootstrap_ci(a, b, n_bootstrap=200, seed=42)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_contains_true_mean_diff(self, tester):
        # True mean diff = 0.1
        a = [0.3] * 200
        b = [0.2] * 200
        lo, hi = tester.bootstrap_ci(a, b, n_bootstrap=500, seed=42)
        assert lo <= 0.1 <= hi

    def test_ci_lower_less_than_upper(self, tester):
        # Use noisy data so differences have variance and CI is non-trivial
        import random
        rng = random.Random(7)
        a = [0.3 + rng.gauss(0, 0.05) for _ in range(100)]
        b = [0.2 + rng.gauss(0, 0.05) for _ in range(100)]
        lo, hi = tester.bootstrap_ci(a, b, n_bootstrap=300, seed=0)
        assert lo < hi

    def test_no_difference_ci_straddles_zero(self, tester):
        a = [0.3] * 200
        lo, hi = tester.bootstrap_ci(a, a, n_bootstrap=300, seed=42)
        assert lo <= 0.0 <= hi

    def test_insufficient_samples(self, tester):
        assert tester.bootstrap_ci([0.5], [0.3]) == (0.0, 0.0)

    def test_reproducible_with_same_seed(self, tester):
        a = [0.1 * i for i in range(50)]
        b = [0.05 * i for i in range(50)]
        ci1 = tester.bootstrap_ci(a, b, n_bootstrap=100, seed=99)
        ci2 = tester.bootstrap_ci(a, b, n_bootstrap=100, seed=99)
        assert ci1 == ci2


# ---------------------------------------------------------------------------
# compare_all
# ---------------------------------------------------------------------------

class TestCompareAll:
    def test_empty_systems_returns_empty(self, tester):
        baseline = [0.3] * 50
        assert tester.compare_all(baseline, {}) == {}

    def test_result_has_expected_keys(self, tester):
        baseline = [0.3] * 100
        systems = {"sys_a": [0.2] * 100}
        result = tester.compare_all(baseline, systems)
        assert "sys_a" in result
        for key in ("t_stat", "p_value", "p_corrected", "significant",
                    "cohens_d", "ci_95", "mean_diff", "n", "bonferroni_threshold"):
            assert key in result["sys_a"], f"Missing key: {key}"

    def test_bonferroni_threshold_scales_with_n_systems(self, tester):
        baseline = [0.3] * 200
        systems = {f"s{i}": [0.25] * 200 for i in range(5)}
        result = tester.compare_all(baseline, systems, alpha=0.05)
        for name in systems:
            assert abs(result[name]["bonferroni_threshold"] - 0.01) < 1e-9

    def test_clearly_better_system_is_significant(self, tester):
        baseline = [0.5] * 300
        systems = {"good": [0.1] * 300}
        result = tester.compare_all(baseline, systems)
        assert result["good"]["significant"] is True

    def test_same_system_not_significant(self, tester):
        baseline = [0.3] * 300
        systems = {"same": baseline}
        result = tester.compare_all(baseline, systems)
        assert result["same"]["significant"] is False

    def test_multiple_systems_bonferroni_applied(self, tester):
        # With 10 comparisons, Bonferroni threshold = 0.05/10 = 0.005
        # Use noisy data with a small real difference that is borderline
        import random
        rng = random.Random(99)
        baseline = [0.3 + rng.gauss(0, 0.05) for _ in range(30)]
        # Systems with tiny difference and high noise — should not all be significant
        systems = {
            f"s{i}": [x + rng.gauss(0.002, 0.05) for x in baseline]
            for i in range(10)
        }
        result = tester.compare_all(baseline, systems, alpha=0.05)
        # With high noise and tiny effect, at least some should not survive Bonferroni
        assert any(not r["significant"] for r in result.values())

    def test_ci_95_is_list_of_two(self, tester):
        baseline = [0.4] * 100
        systems = {"a": [0.3] * 100}
        result = tester.compare_all(baseline, systems)
        ci = result["a"]["ci_95"]
        assert isinstance(ci, list)
        assert len(ci) == 2
