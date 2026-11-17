import numpy as np
import pytest
from scipy.special import betaln

from geconomy import ltv, simulate


@pytest.fixture(scope="module")
def world():
    return simulate.simulate_world(seed=11)


@pytest.fixture(scope="module")
def rfm(world):
    return ltv.rfm_summary(world.players, world.transactions, world.n_weeks)


def test_rfm_counts_match_transactions(world, rfm):
    tx = world.transactions
    tx = tx[tx["calendar_week"] <= world.n_weeks]
    repeats = tx[tx["is_first"] == 0].groupby("player_id").size()
    chk = rfm.set_index("player_id")["x"]
    assert (chk.loc[repeats.index] == repeats).all()
    assert (rfm["t_x"] <= rfm["T"] + 1e-9).all()
    assert (rfm.loc[rfm["x"] == 0, "t_x"] == 0).all()


def test_bgnbd_likelihood_matches_numerical_integration():
    """The closed form must equal E_{lambda,p}[ individual-level likelihood ]."""
    r, alpha, a, b = 0.9, 7.0, 1.6, 5.5
    rng = np.random.default_rng(0)
    n = 6_000_000
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    for x, t_x, T in [(0, 0.0, 20.0), (1, 5.0, 20.0), (3, 12.0, 30.0), (6, 28.0, 30.0)]:
        alive = (1 - p) ** x * lam**x * np.exp(-lam * T)
        dead = p * (1 - p) ** (x - 1) * lam**x * np.exp(-lam * t_x) if x > 0 else 0.0
        brute = np.mean(alive + dead)
        closed = np.exp(ltv.bgnbd_loglik((r, alpha, a, b), np.array([x]), np.array([t_x]), np.array([T]))[0])
        assert closed == pytest.approx(brute, rel=0.03)


def test_conditional_equals_unconditional_for_a_new_customer():
    bg = ltv.BGNBD(0.9, 7.0, 1.6, 5.5)
    t = np.array([4.0, 13.0, 52.0])
    cond = bg.cond_expected_purchases(t, np.zeros(3), np.zeros(3), np.zeros(3))
    assert np.allclose(cond, bg.expected_purchases_new(t), rtol=1e-9)


def test_unconditional_expectation_matches_simulation(world):
    p = world.players
    for ch, tr in simulate.TRUTH.items():
        bg = ltv.BGNBD(tr.r, tr.alpha, tr.a, tr.b)
        d = p[(p["channel"] == ch) & (p["signup_time"] < world.n_weeks - 13)]  # fully observed 13 weeks
        tx = world.transactions
        rep = tx[(tx["is_first"] == 0) & (tx["t"] <= 13) & tx["player_id"].isin(d["player_id"])]
        sim_mean = len(rep) / len(d)
        assert sim_mean == pytest.approx(float(bg.expected_purchases_new(13.0)), rel=0.05)


def test_prob_alive_is_a_probability_and_falls_with_recency_gap():
    bg = ltv.BGNBD(0.9, 7.0, 1.6, 5.5)
    pa = bg.prob_alive(np.array([3, 3, 3]), np.array([30.0, 20.0, 5.0]), np.array([30.0, 30.0, 30.0]))
    assert ((pa > 0) & (pa <= 1)).all()
    assert pa[0] > pa[1] > pa[2]
    assert bg.prob_alive(np.array([0]), np.array([0.0]), np.array([10.0]))[0] == 1.0


def test_parameter_recovery(world, rfm):
    models = ltv.fit_models(rfm)
    rec = ltv.parameter_recovery(models, simulate.TRUTH)
    assert all(bg.converged_ for bg, _ in models.values())
    # BG/NBD's (a, b) are only weakly identified; the rate parameters and the
    # implied means must be tight.
    for ch, (bg, gg) in models.items():
        tr = simulate.TRUTH[ch]
        assert bg.r / bg.alpha == pytest.approx(tr.r / tr.alpha, rel=0.08)
        assert bg.a / (bg.a + bg.b) == pytest.approx(tr.a / (tr.a + tr.b), rel=0.12)
        assert gg.mean_spend == pytest.approx(tr.mean_spend, rel=0.05)
    assert rec["rel_error"].abs().max() < 0.6   # no parameter is wildly off


def test_holdout_prediction_is_calibrated(world, rfm):
    models = ltv.fit_models(rfm)
    res = ltv.validate_holdout(rfm, models, world.transactions, world.n_weeks, 13)
    m = res["metrics"]
    assert abs(m["total_purchase_error"]) < 0.05
    assert abs(m["total_revenue_error"]) < 0.06
    assert m["rmse_model"] < m["rmse_rate_baseline"]
    assert m["rmse_model"] <= m["rmse_channel_mean"] * 1.001


def test_gamma_gamma_shrinks_toward_population_mean():
    gg = ltv.GammaGamma(4.0, 4.0, 10.5)
    est = gg.cond_expected_spend(np.array([1, 10, 100]), np.array([50.0, 50.0, 50.0]))
    assert est[0] < est[1] < est[2] < 50.0
    assert est[0] > gg.mean_spend
