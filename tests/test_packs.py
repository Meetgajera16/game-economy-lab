import numpy as np
import pytest

from geconomy import packs


@pytest.fixture(scope="module")
def spec():
    return packs.default_pack()


def test_slot_probabilities_sum_to_one(spec):
    assert np.allclose(spec.probs().sum(axis=1), 1.0)


def test_exact_payout_distribution_matches_ev(spec):
    v, p = packs.payout_distribution(spec)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (v * p).sum() == pytest.approx(packs.pack_ev(spec), rel=1e-9)


def test_monte_carlo_matches_exact_mean_and_tail(spec):
    rng = np.random.default_rng(1)
    sim = packs.simulate_pack_values(spec, 400_000, rng)
    summ = packs.payout_summary(spec)
    assert sim.mean() == pytest.approx(summ["mean_value"], rel=0.01)
    assert sim.std() == pytest.approx(summ["sd_value"], rel=0.03)
    assert (sim >= spec.price).mean() == pytest.approx(summ["p_value_at_least_price"], abs=0.005)


def test_rarity_calibration_hits_target_rtp(spec):
    for target in (0.70, 0.85, 0.95):
        new, s = packs.calibrate_rarity(spec, target)
        assert packs.rtp(new) == pytest.approx(target, abs=1e-9)
        assert np.allclose(new.probs().sum(axis=1), 1.0)
        assert (new.probs() >= -1e-12).all()


def test_calibration_rejects_unreachable_target(spec):
    with pytest.raises(ValueError):
        packs.calibrate_rarity(spec, 50.0)


def test_pity_preserves_long_run_hit_rate():
    p0 = 0.034
    for n_pity in (40, 60, 90):
        p_new = packs.pity_adjusted_rate(p0, n_pity)
        mean_t = packs.hit_time_stats(p_new, n_pity, n_max=3000)["mean"]
        assert 1.0 / mean_t == pytest.approx(p0, rel=1e-3)
        assert p_new < p0


def test_pity_caps_the_tail_without_changing_the_mean_rate():
    p0 = 0.034
    flat = packs.hit_time_stats(p0)
    p_new = packs.pity_adjusted_rate(p0, 60)
    pity = packs.hit_time_stats(p_new, 60)
    assert pity["p99"] <= 60 < flat["p99"]
    assert pity["sd"] < flat["sd"]


def test_set_completion_exact_vs_monte_carlo(spec):
    exact = packs.set_completion(spec)
    sim = packs.simulate_set_completion(spec, 6000, np.random.default_rng(2))
    assert sim.mean() == pytest.approx(exact["expected_packs"], rel=0.05)
    assert np.quantile(sim, 0.9) == pytest.approx(exact["p90"], rel=0.07)
    assert exact["cdf"][-1] > 0.99999


def test_set_completion_single_card_is_geometric():
    # one chase card only -> expected packs = 1 / p(pack contains that card)
    one = packs.PackSpec(price=1.0, slot_probs=((0.9, 0.0, 0.0, 0.0, 0.1),) * 2,
                         values=(0, 0, 0, 0, 1), cards_per_tier=(1, 1, 1, 1, 1))
    res = packs.set_completion(one, min_tier=4)
    p_pack = 1 - 0.9**2
    assert res["expected_packs"] == pytest.approx(1 / p_pack, rel=1e-9)
