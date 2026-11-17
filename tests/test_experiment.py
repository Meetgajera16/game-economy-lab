import numpy as np
import pandas as pd
import pytest
from scipy import stats

from geconomy import experiment as ex


def test_mde_formula_gives_80_percent_power_by_simulation():
    rng = np.random.default_rng(0)
    sigma, n = 10.0, 4000
    delta = ex.mde(sigma, n)
    rejects = 0
    reps = 2500
    for _ in range(reps):
        a = rng.normal(0, sigma, n)
        b = rng.normal(delta, sigma, n)
        rejects += stats.ttest_ind(b, a, equal_var=False).pvalue < 0.05
    assert rejects / reps == pytest.approx(0.80, abs=0.03)


def test_required_n_inverts_mde():
    n = ex.required_n_per_arm(10.0, 0.5)
    assert ex.mde(10.0, n) <= 0.5 < ex.mde(10.0, n - 5)


def test_cuped_variance_reduction_equals_one_minus_rho_squared():
    rng = np.random.default_rng(1)
    n = 200_000
    x = rng.normal(0, 1, n)
    for rho in (0.3, 0.6, 0.9):
        y = rho * x + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
        ya, theta = ex.cuped_adjust(y, x)
        assert np.var(ya) / np.var(y) == pytest.approx(1 - rho**2, rel=0.01)
        assert theta == pytest.approx(rho, rel=0.02)
        assert ya.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_cuped_is_unbiased_and_keeps_coverage_under_rerandomization():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.gamma(2, 2, n)
    y0 = 0.8 * x + rng.gamma(2, 2, n)
    tau = rng.normal(0.5, 0.2, n)
    res = ex.rerandomization_study(y0, tau, x, n_rep=1200, rng=rng)
    assert abs(res["naive"]["bias"]) < 4 * res["naive"]["empirical_sd"] / np.sqrt(1200)
    assert abs(res["cuped"]["bias"]) < 4 * res["cuped"]["empirical_sd"] / np.sqrt(1200)
    assert res["naive"]["coverage"] == pytest.approx(0.95, abs=0.03)
    assert res["cuped"]["coverage"] == pytest.approx(0.95, abs=0.03)
    assert res["se_reduction_cuped"] > 0.1


def test_srm_check_has_nominal_false_alarm_rate():
    rng = np.random.default_rng(3)
    p = [ex.srm_pvalue(int(k), 10_000 - int(k)) for k in rng.binomial(10_000, 0.5, 4000)]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.015)
    assert ex.srm_pvalue(5300, 4700) < 1e-4          # a real 53/47 split is flagged


def test_group_sequential_constants_match_published_values():
    # Jennison & Turnbull (2000), two-sided alpha = 0.05, K = 5 equally spaced looks
    gs = ex.group_sequential_constants(5, 0.05, n_sim=600_000, seed=0)
    assert gs["pocock"] == pytest.approx(2.413, abs=0.02)
    assert gs["obrien_fleming"] == pytest.approx(2.040, abs=0.02)


def test_peeking_inflates_false_positives_and_boundaries_fix_it():
    rng = np.random.default_rng(4)
    y = rng.gamma(2, 3, 6000)
    res = ex.peeking_study(y, n_looks=5, n_rep=2500, seed=1)
    assert res["fpr_peeking_naive"] > 0.10
    assert res["fpr_fixed_horizon"] == pytest.approx(0.05, abs=0.015)
    assert res["fpr_pocock"] == pytest.approx(0.05, abs=0.02)
    assert res["fpr_obrien_fleming"] == pytest.approx(0.05, abs=0.02)


def _synthetic_rct(n, rng):
    x = rng.normal(0, 1, (n, 3))
    tau = 1.5 * np.exp(-(x[:, 0] - 0.5) ** 2 / 0.5)        # non-monotone effect in x0
    w = ex.randomize(n, rng)
    y = x[:, 1] + rng.normal(0, 1, n) + w * tau
    return x, y, w, tau


def test_uplift_models_rank_a_nonmonotone_effect():
    rng = np.random.default_rng(5)
    x, y, w, tau = _synthetic_rct(30_000, rng)
    xt, yt, wt, taut = _synthetic_rct(10_000, np.random.default_rng(6))
    for fit in (ex.fit_t_learner, ex.fit_transformed_outcome, ex.fit_dr_learner):
        pred = fit(x, y, w)(xt)
        assert stats.spearmanr(pred, taut)[0] > 0.5


def _analytic_oracle_qini(tau):
    """Noise-free Qini for perfect ranking: area between the cumulative TRUE effect of the
    top-f segment and the random diagonal."""
    t = np.sort(tau)[::-1]
    n = len(t)
    gain = np.cumsum(t) / n
    frac = np.arange(1, n + 1) / n
    return float(np.trapezoid(gain - frac * gain[-1], frac))


def test_qini_matches_analytic_value_for_oracle_and_zero_for_random_scores():
    rng = np.random.default_rng(7)
    x, y, w, tau = _synthetic_rct(40_000, rng)
    q_rand = np.mean([ex.qini_coefficient(y, w, rng.random(len(y))) for _ in range(20)])
    q_good = ex.qini_coefficient(y, w, tau)
    assert abs(q_rand) < 0.01
    assert q_good == pytest.approx(_analytic_oracle_qini(tau), abs=0.01)


def test_targeting_value_curve_matches_manual_sum():
    net = np.array([5.0, -1.0, 3.0, -2.0])
    score = np.array([0.9, 0.1, 0.5, 0.2])
    v = ex.targeting_value_curve(net, score, [0.25, 0.5, 1.0])
    assert np.allclose(v, np.array([5.0, 8.0, 5.0]) / 4 * 1000)
