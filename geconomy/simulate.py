"""Simulated acquisition + player purchase behaviour with KNOWN ground truth.

Acquisition
    paid channels: weekly signups ~ Poisson(k * spend**gamma * noise)  (diminishing returns)
    organic:       weekly signups ~ Poisson(rate)

Player behaviour (BG/NBD, Fader-Hardie-Lee 2005)
    purchase rate     lambda_i ~ Gamma(r, rate=alpha)
    dropout prob      p_i      ~ Beta(a, b), applied after every repeat purchase
    spend / purchase  x_ij     ~ Gamma(p_gg, rate=nu_i), nu_i ~ Gamma(q_gg, rate=gamma_gg)

The first purchase happens at signup. Parameters differ by channel, so channels
differ in quality as well as in cost. The true parameters are returned with the
data so that every estimator in this repo can be validated against them.
All numbers are assumptions chosen for illustration; none come from a real game.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ("organic", "paid_social", "influencer", "referral")
PAID = ("paid_social", "influencer", "referral")


@dataclass(frozen=True)
class ChannelTruth:
    name: str
    k: float             # signups per week = k * spend**gamma (paid channels)
    gamma: float
    base_spend: float    # typical weekly spend ($)
    organic_rate: float  # signups per week (organic only)
    r: float
    alpha: float
    a: float
    b: float
    gg_p: float
    gg_q: float
    gg_gamma: float

    @property
    def mean_spend(self) -> float:
        return self.gg_p * self.gg_gamma / (self.gg_q - 1.0)


TRUTH: dict[str, ChannelTruth] = {
    "organic": ChannelTruth("organic", 0.0, 0.0, 0.0, 140.0, 0.85, 7.0, 1.6, 5.5, 4.0, 4.0, 10.50),
    "paid_social": ChannelTruth("paid_social", 0.35, 0.75, 20000.0, 0.0, 0.95, 7.5, 1.5, 4.5, 4.0, 4.0, 10.50),
    "influencer": ChannelTruth("influencer", 1.80, 0.60, 10000.0, 0.0, 0.60, 9.0, 1.3, 2.2, 3.0, 4.0, 10.00),
    "referral": ChannelTruth("referral", 0.11, 0.90, 5000.0, 0.0, 1.00, 6.0, 1.8, 7.0, 4.0, 4.0, 11.25),
}


@dataclass
class World:
    players: pd.DataFrame       # one row per player, incl. true_* latent parameters
    transactions: pd.DataFrame  # player_id, t (weeks since signup), calendar_week, amount, is_first
    weekly: pd.DataFrame        # week, channel, spend, signups
    n_weeks: int                # acquisition / calibration window (weeks)
    holdout_weeks: int          # extra weeks simulated after calibration (for validation)
    seed: int


def simulate_weekly_acquisition(rng: np.random.Generator, n_weeks: int, truth=TRUTH,
                                spend_cv: float = 0.35, noise_sd: float = 0.10) -> pd.DataFrame:
    rows = []
    for w in range(n_weeks):
        for ch, tr in truth.items():
            if ch == "organic":
                mu = tr.organic_rate * np.exp(rng.normal(0, 0.05))
                rows.append((w, ch, 0.0, int(rng.poisson(mu))))
            else:
                spend = tr.base_spend * np.exp(rng.normal(-spend_cv**2 / 2, spend_cv))
                mu = tr.k * spend**tr.gamma * np.exp(rng.normal(-noise_sd**2 / 2, noise_sd))
                rows.append((w, ch, float(spend), int(rng.poisson(mu))))
    return pd.DataFrame(rows, columns=["week", "channel", "spend", "signups"])


def simulate_world(seed: int = 7, n_weeks: int = 52, holdout_weeks: int = 13,
                   truth=TRUTH, max_repeats: int = 400) -> World:
    rng = np.random.default_rng(seed)
    weekly = simulate_weekly_acquisition(rng, n_weeks, truth)

    # ---- players -------------------------------------------------------------
    chan = np.repeat(weekly["channel"].to_numpy(), weekly["signups"].to_numpy())
    wk = np.repeat(weekly["week"].to_numpy(), weekly["signups"].to_numpy())
    n = chan.size
    signup = wk + rng.random(n)

    def col(attr):
        return np.array([getattr(truth[c], attr) for c in chan])

    lam = rng.gamma(col("r"), 1.0 / col("alpha"))
    p_drop = rng.beta(col("a"), col("b"))
    nu = rng.gamma(col("gg_q"), 1.0 / col("gg_gamma"))
    gg_p = col("gg_p")

    # ---- repeat purchases: dropout after each repeat purchase (BG/NBD) ----------
    n_rep = np.minimum(rng.geometric(p_drop), max_repeats)          # >= 1
    idx = np.repeat(np.arange(n), n_rep)
    waits = rng.exponential(1.0 / lam[idx])
    cs = np.cumsum(waits)
    starts = np.cumsum(n_rep) - n_rep
    t_rep = cs - np.repeat(cs[starts] - waits[starts], n_rep)
    last_repeat_t = t_rep[np.cumsum(n_rep) - 1]                     # time of the dropout purchase

    t_end = n_weeks + holdout_weeks
    keep = signup[idx] + t_rep <= t_end
    rep = pd.DataFrame({"player_id": idx[keep], "t": t_rep[keep]})
    first = pd.DataFrame({"player_id": np.arange(n), "t": 0.0})
    tx = pd.concat([first.assign(is_first=1), rep.assign(is_first=0)], ignore_index=True)
    tx = tx.sort_values(["player_id", "t"], kind="stable").reset_index(drop=True)
    pid = tx["player_id"].to_numpy()
    tx["calendar_week"] = signup[pid] + tx["t"].to_numpy()
    tx["amount"] = np.round(rng.gamma(gg_p[pid], 1.0 / nu[pid]), 2)

    players = pd.DataFrame({
        "player_id": np.arange(n),
        "channel": chan,
        "signup_week": wk,
        "signup_time": signup,
        "true_lambda": lam,
        "true_p_drop": p_drop,
        "true_nu": nu,
        "true_gg_p": gg_p,
        "true_mean_spend": gg_p / nu,
        "true_dropout_time": last_repeat_t,   # weeks since signup at which the player goes inactive
    })
    return World(players=players, transactions=tx, weekly=weekly,
                 n_weeks=n_weeks, holdout_weeks=holdout_weeks, seed=seed)


def split_calibration(world: World, t_cal: float | None = None):
    """(calibration transactions, holdout transactions) split at calendar week t_cal."""
    t_cal = world.n_weeks if t_cal is None else t_cal
    tx = world.transactions
    cal = tx[tx["calendar_week"] <= t_cal]
    hold = tx[(tx["calendar_week"] > t_cal) & (tx["calendar_week"] <= t_cal + world.holdout_weeks)]
    return cal, hold
