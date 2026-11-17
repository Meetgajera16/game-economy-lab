import numpy as np
import pytest

from geconomy import acquisition as acq
from geconomy import simulate

K = {"a": 0.35, "b": 1.8, "c": 0.11}
G = {"a": 0.75, "b": 0.60, "c": 0.90}
V = {"a": 40.0, "b": 18.0, "c": 48.0}
LO = {"a": 5_000.0, "b": 2_500.0, "c": 1_000.0}
HI = {"a": 40_000.0, "b": 20_000.0, "c": 30_000.0}


def test_response_curve_recovery_from_simulated_weeks():
    w = simulate.simulate_world(seed=3)
    fit = acq.fit_response_curves(w.weekly)
    for ch in simulate.PAID:
        tr = simulate.TRUTH[ch]
        assert fit.loc[ch, "gamma_ci_low"] < tr.gamma < fit.loc[ch, "gamma_ci_high"]
        assert fit.loc[ch, "gamma"] == pytest.approx(tr.gamma, abs=0.12)


def test_allocation_respects_budget_and_bounds():
    s = acq.allocate(V, K, G, 35_000.0, LO, HI)
    assert sum(s.values()) <= 35_000.0 + 1e-6
    assert all(LO[c] - 1e-9 <= s[c] <= HI[c] + 1e-9 for c in s)


def test_allocation_satisfies_equal_marginal_return_at_an_interior_optimum():
    # wide bounds -> interior solution; marginal profit per $ must be equal across channels
    lo = {c: 1.0 for c in V}
    hi = {c: 1e7 for c in V}
    g = {"a": 0.75, "b": 0.60, "c": 0.60}
    s = acq.allocate(V, K, g, 35_000.0, lo, hi, force_budget=True)
    marg = [V[c] * g[c] * K[c] * s[c] ** (g[c] - 1) for c in V]
    assert sum(s.values()) == pytest.approx(35_000.0, rel=1e-6)
    assert np.allclose(marg, marg[0], rtol=1e-5)


def test_optimizer_beats_random_feasible_allocations():
    best = acq.allocate(V, K, G, 35_000.0, LO, HI)
    best_profit = acq.weekly_profit(best, V, K, G)
    rng = np.random.default_rng(0)
    for _ in range(300):
        w = rng.dirichlet(np.ones(3)) * 35_000.0
        cand = dict(zip(V, w))
        if all(LO[c] <= cand[c] <= HI[c] for c in V):
            assert acq.weekly_profit(cand, V, K, G) <= best_profit + 1e-6


def test_budget_not_binding_when_marginal_profit_hits_zero_first():
    tiny_value = {c: 0.5 for c in V}
    s = acq.allocate(tiny_value, K, G, 1e9, LO, HI)
    assert sum(s.values()) < 1e9
    assert all(s[c] == pytest.approx(LO[c]) or s[c] <= HI[c] for c in s)


def test_marginal_cac_is_average_cac_over_gamma():
    tab = acq.marginal_table({"a": 20_000.0}, V, K, G)
    assert tab.loc["a", "marginal_cac"] == pytest.approx(tab.loc["a", "avg_cac"] / G["a"])
