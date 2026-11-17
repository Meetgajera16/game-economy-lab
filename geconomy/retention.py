"""Retention / churn analysis for non-contractual customers.

There is no cancellation event in a pack game, so "churn" has to be defined
operationally. Here it is the first lapse: a gap of at least `gap_weeks` weeks with
no purchase. A lapse is only observable once the gap has elapsed, so players whose
gap is still running at the end of the window are right-censored.

(The simulator also knows each player's true latent dropout time. That is NOT used
here: this module only sees what an analyst would see.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff


def lapse_survival_data(players: pd.DataFrame, transactions: pd.DataFrame, t_cal: float,
                        gap_weeks: float = 8.0) -> pd.DataFrame:
    """One row per player: duration (weeks since signup) and event (1 = lapse observed)."""
    p = players[players["signup_time"] < t_cal][["player_id", "channel", "signup_time"]].copy()
    p["age"] = t_cal - p["signup_time"]
    tx = transactions[transactions["calendar_week"] <= t_cal][["player_id", "t", "amount", "is_first"]]
    tx = tx.sort_values(["player_id", "t"], kind="stable")
    nxt = tx.groupby("player_id")["t"].shift(-1).fillna(np.inf)
    lapsed = tx[(nxt - tx["t"]) >= gap_weeks]
    first_lapse_t = lapsed.groupby("player_id")["t"].min() + gap_weeks     # earliest declared lapse
    out = p.set_index("player_id")
    out["lapse_time"] = first_lapse_t
    out["event"] = (out["lapse_time"].notna() & (out["lapse_time"] <= out["age"])).astype(int)
    out["duration"] = np.where(out["event"] == 1, out["lapse_time"], out["age"])
    first_amt = tx[tx["is_first"] == 1].set_index("player_id")["amount"]
    out["first_amount"] = first_amt
    return out.reset_index()


def kaplan_meier(durations, events) -> pd.DataFrame:
    d = np.asarray(durations, float)
    e = np.asarray(events, bool)
    order = np.argsort(d, kind="stable")
    d, e = d[order], e[order]
    times = np.unique(d[e])
    at_risk = len(d) - np.searchsorted(d, times, side="left")
    deaths = np.bincount(np.searchsorted(times, d[e]), minlength=len(times))
    surv = np.cumprod(1.0 - deaths / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(at_risk > deaths, deaths / (at_risk * (at_risk - deaths)), 0.0)
    se = surv * np.sqrt(np.cumsum(g))                                  # Greenwood
    return pd.DataFrame({"time": times, "at_risk": at_risk, "events": deaths, "survival": surv,
                         "se": se, "ci_low": np.clip(surv - 1.96 * se, 0, 1),
                         "ci_high": np.clip(surv + 1.96 * se, 0, 1)})


def survival_at(km: pd.DataFrame, t: float) -> float:
    i = np.searchsorted(km["time"].to_numpy(), t, side="right") - 1
    return 1.0 if i < 0 else float(km["survival"].iloc[i])


def median_survival(km: pd.DataFrame) -> float:
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float("nan")


def logrank_by_channel(df: pd.DataFrame) -> dict:
    chi2, p = survdiff(df["duration"].to_numpy(), df["event"].to_numpy(), df["channel"].to_numpy())
    return {"chi2": float(chi2), "p_value": float(p)}


def cox_hazard_ratios(df: pd.DataFrame, baseline: str = "organic") -> pd.DataFrame:
    """Cox proportional-hazards model: channel dummies (vs baseline) plus log first spend
    (a negative control: first-purchase size is generated independently of lapse)."""
    chans = [c for c in sorted(df["channel"].unique()) if c != baseline]
    x = pd.DataFrame({f"channel_{c}": (df["channel"] == c).astype(float) for c in chans})
    x["log_first_spend"] = np.log(df["first_amount"].clip(lower=0.01))
    res = PHReg(df["duration"].to_numpy(), x.to_numpy(), status=df["event"].to_numpy(),
                ties="breslow").fit()
    ci = np.exp(res.conf_int())
    return pd.DataFrame({"term": x.columns, "hazard_ratio": np.exp(res.params),
                         "ci_low": ci[:, 0], "ci_high": ci[:, 1], "p_value": res.pvalues})
