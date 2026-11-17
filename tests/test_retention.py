import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.survfunc import SurvfuncRight

from geconomy import retention


def test_lapse_definition_on_hand_built_cases():
    players = pd.DataFrame({
        "player_id": [0, 1, 2, 3, 4],
        "channel": ["organic"] * 5,
        "signup_time": [0.0] * 5,
    })
    # (player, purchase times since signup)
    times = {0: [0, 3, 20], 1: [0, 2], 2: [0, 1, 2, 3], 3: [0], 4: [0]}
    rows = [(pid, t, 10.0, int(t == 0), float(t)) for pid, ts in times.items() for t in ts]
    tx = pd.DataFrame(rows, columns=["player_id", "t", "amount", "is_first", "calendar_week"])
    ages = {0: 40.0, 1: 5.0, 2: 12.0, 3: 7.0, 4: 9.0}
    out = []
    for pid, age in ages.items():
        players.loc[players.player_id == pid, "signup_time"] = 100.0 - age
        tx.loc[tx.player_id == pid, "calendar_week"] += 100.0 - age
    res = retention.lapse_survival_data(players, tx, 100.0, gap_weeks=8.0).set_index("player_id")
    # 0: gap 3->20 is 17 >= 8 -> lapse at 3 + 8 = 11 (observed)
    assert (res.loc[0, "event"], res.loc[0, "duration"]) == (1, 11.0)
    # 1: last purchase at 2, lapse would be at 10 > age 5 -> censored at 5
    assert (res.loc[1, "event"], res.loc[1, "duration"]) == (0, 5.0)
    # 2: last purchase at 3, lapse at 11 <= age 12 -> event at 11
    assert (res.loc[2, "event"], res.loc[2, "duration"]) == (1, 11.0)
    # 3: only first purchase, lapse at 8 > age 7 -> censored at 7
    assert (res.loc[3, "event"], res.loc[3, "duration"]) == (0, 7.0)
    # 4: only first purchase, lapse at 8 <= age 9 -> event at 8
    assert (res.loc[4, "event"], res.loc[4, "duration"]) == (1, 8.0)


def test_kaplan_meier_matches_statsmodels():
    rng = np.random.default_rng(0)
    t = rng.exponential(10, 5000)
    c = rng.uniform(0, 25, 5000)
    dur, ev = np.minimum(t, c), (t <= c)
    km = retention.kaplan_meier(dur, ev)
    sf = SurvfuncRight(dur, ev)
    assert np.allclose(km["survival"].to_numpy(), sf.surv_prob, atol=1e-10)
    assert np.allclose(km["se"].to_numpy(), sf.surv_prob_se, rtol=1e-6)


def test_kaplan_meier_recovers_exponential_under_censoring():
    rng = np.random.default_rng(1)
    n = 80_000
    t = rng.exponential(10, n)
    c = rng.uniform(0, 30, n)
    km = retention.kaplan_meier(np.minimum(t, c), t <= c)
    for q in (5.0, 10.0, 20.0):
        assert retention.survival_at(km, q) == pytest.approx(np.exp(-q / 10), abs=0.01)


def test_cox_recovers_known_hazard_ratio():
    rng = np.random.default_rng(2)
    n = 40_000
    grp = rng.choice(["organic", "referral"], n)
    rate = np.where(grp == "referral", 0.1 * 0.5, 0.1)   # true HR = 0.5
    t = rng.exponential(1 / rate)
    c = rng.uniform(0, 40, n)
    df = pd.DataFrame({"channel": grp, "duration": np.minimum(t, c), "event": (t <= c).astype(int),
                       "first_amount": rng.gamma(4, 3, n)})
    hr = retention.cox_hazard_ratios(df).set_index("term")
    assert hr.loc["channel_referral", "hazard_ratio"] == pytest.approx(0.5, rel=0.05)
    assert hr.loc["log_first_spend", "hazard_ratio"] == pytest.approx(1.0, abs=0.05)
