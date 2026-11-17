"""Channel response curves and LTV-aware budget allocation.

Model: signups_c(s) = k_c * s**gamma_c  (gamma < 1: diminishing returns).
Value of a signup in channel c: V_c = contribution-margin LTV over 52 weeks (from ltv.py).
Weekly profit for a spend vector s:   sum_c V_c * k_c * s_c**gamma_c  -  s_c.

First-order condition (interior):  V_c * gamma_c * k_c * s_c**(gamma_c - 1) = lambda,
so  s_c(lambda) = (V_c gamma_c k_c / lambda) ** (1 / (1 - gamma_c)), clipped to the spend range
that was actually observed (extrapolating a fitted power law far outside its data is how
marketing-mix models go wrong). Budget exhaustion fixes lambda by bisection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq


def fit_response_curves(weekly: pd.DataFrame) -> pd.DataFrame:
    """Poisson regression  log E[signups] = log k + gamma log(spend)  per paid channel,
    with heteroskedasticity-robust standard errors."""
    rows = []
    for ch, d in weekly[weekly["spend"] > 0].groupby("channel"):
        x = sm.add_constant(np.log(d["spend"].to_numpy()))
        res = sm.GLM(d["signups"].to_numpy(), x, family=sm.families.Poisson()).fit(cov_type="HC0")
        ci = res.conf_int()
        rows.append({"channel": ch, "k": float(np.exp(res.params[0])), "gamma": float(res.params[1]),
                     "gamma_se": float(res.bse[1]), "gamma_ci_low": float(ci[1, 0]),
                     "gamma_ci_high": float(ci[1, 1]), "n_weeks": int(len(d))})
    return pd.DataFrame(rows).set_index("channel")


def signups(spend, k, gamma):
    return k * np.power(np.maximum(spend, 0.0), gamma)


def weekly_profit(spend: dict, value: dict, k: dict, gamma: dict) -> float:
    return float(sum(value[c] * signups(s, k[c], gamma[c]) - s for c, s in spend.items()))


def weekly_signups(spend: dict, k: dict, gamma: dict) -> float:
    return float(sum(signups(s, k[c], gamma[c]) for c, s in spend.items()))


GAMMA_CAP = 0.99   # the allocation model assumes diminishing returns; a fitted gamma >= 1 is capped


def allocate(value: dict, k: dict, gamma: dict, budget: float, lo: dict, hi: dict,
             force_budget: bool = False) -> dict:
    """Maximize sum_c V_c k_c s_c^gamma_c - s_c  subject to sum s_c <= budget, lo <= s_c <= hi.
    With force_budget=True the whole budget is spent (used for the 'maximize signups' policy).
    Computed in log space because 1 / (1 - gamma) explodes as gamma approaches 1."""
    chans = list(value)
    assert all(lo[c] > 0 for c in chans)

    def spend_at(z):                                   # z = log(lambda)
        out = {}
        for c in chans:
            g = min(gamma[c], GAMMA_CAP)
            coef = value[c] * g * k[c]
            expo = -np.inf if coef <= 0 else (np.log(coef) - z) / (1.0 - g)
            out[c] = float(np.exp(np.clip(expo, np.log(lo[c]), np.log(hi[c]))))
        return out

    total = lambda z: sum(spend_at(z).values()) - budget
    if not force_budget and total(0.0) <= 0:          # budget not binding: spend until marginal profit = 0
        return spend_at(0.0)
    z_lo, z_hi = -60.0, 60.0
    if total(z_lo) < 0:                               # even at the upper bounds the budget is not exhausted
        return spend_at(z_lo)
    if total(z_hi) > 0:                               # lower bounds already exceed the budget
        return spend_at(z_hi)
    z = brentq(total, z_lo, z_hi, xtol=1e-12)
    return spend_at(float(z))


def marginal_table(spend: dict, value: dict, k: dict, gamma: dict) -> pd.DataFrame:
    """Average vs marginal CAC and LTV:CAC at a given spend vector."""
    rows = []
    for c, s in spend.items():
        n = signups(s, k[c], gamma[c])
        avg_cac = s / n
        marg_cac = avg_cac / gamma[c]                 # d spend / d signups = s / (gamma * n)
        rows.append((c, s, n, avg_cac, marg_cac, value[c], value[c] / avg_cac, value[c] / marg_cac))
    return pd.DataFrame(rows, columns=["channel", "spend", "signups", "avg_cac", "marginal_cac",
                                       "ltv_margin", "ltv_to_avg_cac", "ltv_to_marginal_cac"]).set_index("channel")
