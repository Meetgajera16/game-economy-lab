import numpy as np
import pytest
from scipy import stats

from geconomy import matchmaking as mm


def test_elo_is_zero_sum_and_moves_toward_the_result():
    ra, rb = np.array([1500.0, 1700.0]), np.array([1500.0, 1500.0])
    na, nb = mm.elo_update(ra, rb, np.array([1.0, 0.0]), 32.0)
    assert np.allclose(na + nb, ra + rb)
    assert na[0] == pytest.approx(1516.0)           # even match, win: +K/2
    assert na[1] < 1700.0                           # favourite loses: rating falls


def test_expected_score_is_symmetric_and_400_points_is_ten_to_one():
    assert mm.expected_score(1500, 1500) == pytest.approx(0.5)
    assert mm.expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert mm.expected_score(1700, 1500) + mm.expected_score(1500, 1700) == pytest.approx(1.0)


def test_glicko_matches_glickmans_published_example():
    # Glickman (1999) worked example: r=1500, RD=200 vs (1400, 30) win, (1550, 100) loss, (1700, 300) loss
    r, rd = mm.glicko_update(1500.0, 200.0, [1400, 1550, 1700], [30, 100, 300], [1, 0, 0])
    assert r == pytest.approx(1464.1, abs=0.2)
    assert rd == pytest.approx(151.4, abs=0.2)


def test_vectorised_glicko_equals_scalar_for_a_single_opponent():
    rng = np.random.default_rng(0)
    r, rd = rng.normal(1500, 100, 5), rng.uniform(50, 350, 5)
    o, ord_ = rng.normal(1500, 100, 5), rng.uniform(50, 350, 5)
    s = np.array([1, 0, 1, 1, 0.0])
    vr, vrd = mm.glicko_update_vec(r, rd, o, ord_, s)
    for i in range(5):
        sr, srd = mm.glicko_update(r[i], rd[i], [o[i]], [ord_[i]], [s[i]])
        assert vr[i] == pytest.approx(sr) and vrd[i] == pytest.approx(srd)


def test_bradley_terry_recovers_true_skill_scale():
    rng = np.random.default_rng(1)
    n, m = 300, 60_000
    theta = rng.normal(1500, 150, n)
    a = rng.integers(0, n, m)
    b = (a + rng.integers(1, n, m)) % n
    wins = rng.random(m) < mm.expected_score(theta[a], theta[b])
    est = mm.bradley_terry(a, b, wins, n, c=10.0)
    assert stats.spearmanr(est, theta)[0] > 0.97
    slope = np.polyfit(theta, est, 1)[0]
    assert slope == pytest.approx(1.0, abs=0.12)


def test_online_logloss_falls_toward_the_bayes_floor():
    res = mm.run_rating_simulation(n_players=600, rounds=60, policy="random", method="elo", k=24, seed=2)
    assert res["logloss"][:5].mean() > res["logloss"][-10:].mean()
    assert res["logloss"][-10:].mean() > res["bayes_ll"][-10:].mean()      # cannot beat the truth
    assert res["logloss"][0] == pytest.approx(np.log(2), abs=0.02)         # starts at coin-flip


def test_oracle_pairing_makes_fairer_matches_than_random():
    random_ = mm.run_rating_simulation(400, 20, "random", seed=3)["fair_share"].mean()
    oracle = mm.run_rating_simulation(400, 20, "oracle", seed=3)["fair_share"].mean()
    assert oracle > random_ + 0.3


def test_pure_oracle_pairing_leaves_ratings_uninformative():
    """Pairing the same neighbours every round never connects the comparison graph, so Elo
    (zero-sum within each pair) cannot learn a global ranking even with perfect pairing."""
    res = mm.run_rating_simulation(400, 30, "oracle", seed=4)
    assert abs(res["spearman"][-1]) < 0.15


def test_rating_pairing_trades_rating_accuracy_for_fairness_and_exploration_repairs_it():
    kw = dict(n_players=600, rounds=40, seed=5)
    rnd = mm.run_rating_simulation(policy="random", **kw)
    rat = mm.run_rating_simulation(policy="rating", **kw)
    exp = mm.run_rating_simulation(policy="rating_explore", **kw)
    assert rat["fair_share"][10:].mean() > rnd["fair_share"][10:].mean() + 0.15
    assert exp["fair_share"][10:].mean() > rnd["fair_share"][10:].mean() + 0.05
    assert exp["spearman"][-1] > rat["spearman"][-1] - 0.01
    assert exp["spearman"][-1] > rnd["spearman"][-1] - 0.05
