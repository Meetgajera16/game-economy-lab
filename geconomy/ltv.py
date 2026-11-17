"""Customer lifetime value with BG/NBD (purchases) + Gamma-Gamma (spend).

Both models are implemented from the published formulas:
  Fader, Hardie & Lee (2005), "Counting your customers the easy way"
  Fader & Hardie (2013), "The Gamma-Gamma model of monetary value"
and validated three ways (see tests/test_ltv.py and run_all.py):
  1. the closed-form likelihood against brute-force numerical integration,
  2. parameter recovery against the simulator's true parameters,
  3. out-of-sample prediction on a held-out 13-week window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def rfm_summary(players: pd.DataFrame, transactions: pd.DataFrame, t_cal: float) -> pd.DataFrame:
    """Per-player summary as of calendar week t_cal.

    x    number of REPEAT purchases (excludes the first purchase at signup)
    t_x  weeks since signup of the last repeat purchase (0 if none)
    T    weeks since signup (age) at t_cal
    m    mean spend of repeat purchases (NaN when x == 0)
    """
    p = players[players["signup_time"] < t_cal].copy()
    p["T"] = t_cal - p["signup_time"]
    tx = transactions[transactions["calendar_week"] <= t_cal]
    rep = tx[tx["is_first"] == 0].groupby("player_id").agg(
        x=("t", "size"), t_x=("t", "max"), m=("amount", "mean"))
    first = tx[tx["is_first"] == 1].set_index("player_id")["amount"].rename("first_amount")
    out = p.set_index("player_id").join(rep).join(first)
    out["x"] = out["x"].fillna(0).astype(int)
    out["t_x"] = out["t_x"].fillna(0.0)
    return out.reset_index()


# --------------------------------------------------------------------------- #
# BG/NBD
# --------------------------------------------------------------------------- #
def bgnbd_loglik(params, x, t_x, T) -> np.ndarray:
    """Per-customer log-likelihood L(r, alpha, a, b | x, t_x, T)."""
    r, alpha, a, b = params
    x = np.asarray(x, float)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.log(a) - np.log(b + x - 1.0) - (r + x) * np.log(alpha + t_x)
    a4 = np.where(x > 0, a4, -np.inf)
    return a1 + a2 + np.logaddexp(a3, a4)


class BGNBD:
    def __init__(self, r=None, alpha=None, a=None, b=None):
        self.r, self.alpha, self.a, self.b = r, alpha, a, b

    @property
    def params(self) -> dict:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b}

    def fit(self, x, t_x, T) -> "BGNBD":
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        nll = lambda th: -bgnbd_loglik(np.exp(th), x, t_x, T).sum()
        best = None
        for start in ([1.0, 5.0, 1.5, 3.0], [0.5, 10.0, 2.0, 5.0], [2.0, 3.0, 1.2, 1.2]):
            res = minimize(nll, np.log(start), method="L-BFGS-B", bounds=[(-5, 5)] * 4)
            if best is None or res.fun < best.fun:
                best = res
        self.r, self.alpha, self.a, self.b = np.exp(best.x)
        self.loglik_ = float(-best.fun)
        self.converged_ = bool(best.success)
        return self

    # P(customer still active | history)
    def prob_alive(self, x, t_x, T) -> np.ndarray:
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (self.a / (self.b + x - 1.0)) * ((self.alpha + T) / (self.alpha + t_x)) ** (self.r + x)
        return 1.0 / (1.0 + np.where(x > 0, ratio, 0.0))

    # E[repeat purchases in (T, T+t] | x, t_x, T)
    def cond_expected_purchases(self, t, x, t_x, T) -> np.ndarray:
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        t = np.broadcast_to(np.asarray(t, float), x.shape)
        _a, _b, _c = r + x, b + x, a + b + x - 1.0
        z = t / (alpha + T + t)
        h = hyp2f1(_a, _b, _c, z)
        alt = hyp2f1(_c - _a, _c - _b, _c, z) * (1.0 - z) ** (_c - _a - _b)   # Euler transform
        h = np.where(np.isfinite(h), h, alt)
        first = (a + b + x - 1.0) / (a - 1.0)
        second = 1.0 - h * ((alpha + T) / (alpha + T + t)) ** (r + x)
        denom = 1.0 + np.where(
            x > 0, (a / (b + x - 1.0 + (x == 0))) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return first * second / denom

    # E[X(t)] for a brand-new customer (first purchase at time 0)
    def expected_purchases_new(self, t) -> np.ndarray:
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        t = np.asarray(t, float)
        z = t / (alpha + t)
        return (a + b - 1.0) / (a - 1.0) * (1.0 - (alpha / (alpha + t)) ** r * hyp2f1(r, b, a + b - 1.0, z))


# --------------------------------------------------------------------------- #
# Gamma-Gamma
# --------------------------------------------------------------------------- #
def gg_loglik(params, x, m) -> np.ndarray:
    p, q, v = params
    x = np.asarray(x, float)
    return (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
            + (p * x - 1.0) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(x * m + v))


class GammaGamma:
    def __init__(self, p=None, q=None, v=None):
        self.p, self.q, self.v = p, q, v

    @property
    def params(self) -> dict:
        return {"p": self.p, "q": self.q, "v": self.v}

    @property
    def mean_spend(self) -> float:
        return self.p * self.v / (self.q - 1.0)

    def fit(self, x, m) -> "GammaGamma":
        x, m = np.asarray(x, float), np.asarray(m, float)
        keep = (x > 0) & np.isfinite(m) & (m > 0)
        x, m = x[keep], m[keep]
        nll = lambda th: -gg_loglik(np.exp(th), x, m).sum()
        best = None
        for start in ([4.0, 4.0, 10.0], [2.0, 3.0, 5.0], [6.0, 6.0, 20.0]):
            res = minimize(nll, np.log(start), method="L-BFGS-B", bounds=[(-5, 6)] * 3)
            if best is None or res.fun < best.fun:
                best = res
        self.p, self.q, self.v = np.exp(best.x)
        self.loglik_ = float(-best.fun)
        return self

    def cond_expected_spend(self, x, m) -> np.ndarray:
        """Shrinks the customer's observed mean spend toward the population mean."""
        x = np.asarray(x, float)
        m = np.where(np.isfinite(m), m, 0.0)
        w_pop = (self.q - 1.0) / (self.p * x + self.q - 1.0)
        return w_pop * self.mean_spend + (1.0 - w_pop) * np.where(x > 0, m, 0.0)


# --------------------------------------------------------------------------- #
# channel-level fitting, LTV, validation
# --------------------------------------------------------------------------- #
def fit_models(rfm: pd.DataFrame) -> dict:
    """Fit BG/NBD and Gamma-Gamma separately per channel (the models assume a
    homogeneous population, so a pooled fit across channels would be misspecified)."""
    out = {}
    for ch, d in rfm.groupby("channel"):
        bg = BGNBD().fit(d["x"], d["t_x"], d["T"])
        gg = GammaGamma().fit(d["x"], d["m"])
        out[ch] = (bg, gg)
    return out


def parameter_recovery(models: dict, truth: dict) -> pd.DataFrame:
    rows = []
    for ch, (bg, gg) in models.items():
        t = truth[ch]
        true = {"r": t.r, "alpha": t.alpha, "a": t.a, "b": t.b,
                "p": t.gg_p, "q": t.gg_q, "v": t.gg_gamma}
        est = {**bg.params, **gg.params}
        for k, tv in true.items():
            rows.append((ch, k, tv, est[k], est[k] / tv - 1.0))
    df = pd.DataFrame(rows, columns=["channel", "param", "true", "estimate", "rel_error"])
    return df


def _discount_weights(horizon_weeks: int, annual_discount: float) -> np.ndarray:
    k = np.arange(1, horizon_weeks + 1)
    return (1.0 + annual_discount) ** (-k / 52.0)


def expected_value_existing(bg: BGNBD, gg: GammaGamma, d: pd.DataFrame, horizon_weeks: int,
                            annual_discount: float = 0.0) -> pd.DataFrame:
    """Expected repeat purchases and (discounted) revenue over the next `horizon_weeks`."""
    x, t_x, T = d["x"].to_numpy(), d["t_x"].to_numpy(), d["T"].to_numpy()
    spend = gg.cond_expected_spend(x, d["m"].to_numpy())
    w = _discount_weights(horizon_weeks, annual_discount)
    prev = np.zeros(len(d))
    pv = np.zeros(len(d))
    for k in range(1, horizon_weeks + 1):
        cur = bg.cond_expected_purchases(float(k), x, t_x, T)
        pv += (cur - prev) * w[k - 1]
        prev = cur
    return pd.DataFrame({
        "exp_purchases": prev,
        "exp_spend": spend,
        "exp_revenue": prev * spend,
        "pv_revenue": pv * spend,
        "p_alive": bg.prob_alive(x, t_x, T),
    }, index=d.index)


def expected_value_new(bg: BGNBD, gg: GammaGamma, horizon_weeks: int, annual_discount: float = 0.0,
                       first_spend: float | None = None) -> dict:
    """Expected (discounted) revenue per newly acquired customer over the horizon."""
    first = gg.mean_spend if first_spend is None else first_spend
    w = _discount_weights(horizon_weeks, annual_discount)
    cum = bg.expected_purchases_new(np.arange(0, horizon_weeks + 1, dtype=float))
    inc = np.diff(cum)
    return {"first_purchase": first, "exp_repeat_purchases": float(cum[-1]),
            "repeat_revenue": float((inc * w).sum() * gg.mean_spend),
            "pv_revenue": float(first + (inc * w).sum() * gg.mean_spend)}


def validate_holdout(rfm: pd.DataFrame, models: dict, transactions: pd.DataFrame,
                     t_cal: float, horizon_weeks: int) -> dict:
    """Compare predicted vs actual purchases and revenue in (t_cal, t_cal + horizon]."""
    win = transactions[(transactions["calendar_week"] > t_cal)
                       & (transactions["calendar_week"] <= t_cal + horizon_weeks)]
    actual = win.groupby("player_id").agg(act_purchases=("amount", "size"), act_revenue=("amount", "sum"))
    parts = []
    for ch, d in rfm.groupby("channel"):
        bg, gg = models[ch]
        ev = expected_value_existing(bg, gg, d, horizon_weeks)
        ev["player_id"], ev["channel"], ev["x"] = d["player_id"].to_numpy(), ch, d["x"].to_numpy()
        ev["rate_baseline"] = np.where(d["T"] > 0, d["x"] / d["T"], 0.0) * horizon_weeks
        parts.append(ev)
    df = pd.concat(parts).merge(actual, left_on="player_id", right_index=True, how="left").fillna(
        {"act_purchases": 0, "act_revenue": 0.0})
    df["bucket"] = df["x"].clip(upper=5).map(lambda v: "5+" if v == 5 else str(int(v)))

    by_bucket = df.groupby("bucket").agg(
        n=("player_id", "size"), predicted=("exp_purchases", "mean"), actual=("act_purchases", "mean"))
    by_channel = df.groupby("channel").agg(
        pred_rev=("exp_revenue", "sum"), act_rev=("act_revenue", "sum"),
        pred_purch=("exp_purchases", "sum"), act_purch=("act_purchases", "sum"))
    by_channel["revenue_error"] = by_channel["pred_rev"] / by_channel["act_rev"] - 1.0
    by_channel["purchase_error"] = by_channel["pred_purch"] / by_channel["act_purch"] - 1.0

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    # mean-only baseline: every player gets the channel's average repeat rate
    naive_mean = df.groupby("channel")["act_purchases"].transform("mean")
    metrics = {
        "n_players": int(len(df)),
        "total_revenue_error": float(df["exp_revenue"].sum() / df["act_revenue"].sum() - 1.0),
        "total_purchase_error": float(df["exp_purchases"].sum() / df["act_purchases"].sum() - 1.0),
        "rmse_model": rmse(df["exp_purchases"], df["act_purchases"]),
        "rmse_rate_baseline": rmse(df["rate_baseline"], df["act_purchases"]),
        "rmse_channel_mean": rmse(naive_mean, df["act_purchases"]),
        "corr_purchases": float(np.corrcoef(df["exp_purchases"], df["act_purchases"])[0, 1]),
    }
    return {"player_level": df, "by_bucket": by_bucket, "by_channel": by_channel, "metrics": metrics}
