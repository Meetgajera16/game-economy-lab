"""Experimentation and uplift targeting on a semi-synthetic randomized promo test.

Design
    Baseline outcomes Y(0) are the players' REAL simulated revenue in the 4 weeks after
    the calibration cut-off. A bonus-pack promo then adds an incremental purchase with a
    probability u_i that depends on how long the player has been silent: lapsing-but-
    engaged players respond most, brand-new players and long-gone players barely respond.
    Because the effect is defined by construction, every estimator can be checked
    against the true effect, which a real experiment never allows.

Contents
    ATE estimation (Welch, CUPED), SRM and balance checks, MDE / power (analytic vs
    simulated), re-randomization coverage, peeking and group-sequential boundaries,
    uplift models (T-learner, transformed outcome) and net-value targeting policies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from . import ltv

FEATURES = ["x", "t_x", "T", "gap", "m", "first_amount", "pre_rev4", "pre_n4", "p_alive",
            "exp_purch13", "ch_influencer", "ch_paid_social", "ch_referral"]


# --------------------------------------------------------------------------- #
# the semi-synthetic experiment world
# --------------------------------------------------------------------------- #
def build_experiment_frame(world, models: dict, t_cal: float | None = None, window: int = 4,
                           seed: int = 123, u_max: float = 0.30, gap_peak_weeks: float = 8.0,
                           sigma: float = 0.5) -> pd.DataFrame:
    t_cal = float(world.n_weeks if t_cal is None else t_cal)
    tx = world.transactions
    rfm = ltv.rfm_summary(world.players, tx, t_cal).set_index("player_id")
    pre = tx[(tx["calendar_week"] > t_cal - 4) & (tx["calendar_week"] <= t_cal)].groupby("player_id").agg(
        pre_rev4=("amount", "sum"), pre_n4=("amount", "size"))
    post = tx[(tx["calendar_week"] > t_cal) & (tx["calendar_week"] <= t_cal + window)].groupby(
        "player_id")["amount"].sum().rename("y0")
    df = rfm.join(pre).join(post)
    df[["pre_rev4", "pre_n4", "y0"]] = df[["pre_rev4", "pre_n4", "y0"]].fillna(0.0)
    df["m"] = df["m"].fillna(df["first_amount"])
    df["gap"] = df["T"] - df["t_x"]                       # weeks since last purchase

    df["p_alive"] = np.nan
    df["exp_purch13"] = np.nan
    for ch, (bg, gg) in models.items():
        mask = df["channel"] == ch
        d = df[mask]
        df.loc[mask, "p_alive"] = bg.prob_alive(d["x"], d["t_x"], d["T"])
        df.loc[mask, "exp_purch13"] = bg.cond_expected_purchases(13.0, d["x"], d["t_x"], d["T"])
    for ch in ("influencer", "paid_social", "referral"):
        df[f"ch_{ch}"] = (df["channel"] == ch).astype(float)

    # ---- ground-truth treatment effect (the assumption this whole module rests on) ----
    rng = np.random.default_rng(seed)
    bell = np.exp(-(np.log1p(df["gap"]) - np.log1p(gap_peak_weeks)) ** 2 / (2 * sigma**2))
    ramp = 0.3 + 0.7 * np.minimum(1.0, df["x"] / 3.0)
    df["true_u"] = u_max * bell * ramp                                    # P(extra purchase)
    bought = rng.random(len(df)) < df["true_u"]
    spend = rng.gamma(df["true_gg_p"].to_numpy(), 1.0 / df["true_nu"].to_numpy())
    df["incr"] = np.where(bought, spend, 0.0)                              # realized extra revenue
    df["true_tau"] = df["true_u"] * df["true_mean_spend"]                  # expected extra revenue
    return df.reset_index()


# --------------------------------------------------------------------------- #
# estimation and design
# --------------------------------------------------------------------------- #
def randomize(n: int, rng: np.random.Generator, p_treat: float = 0.5) -> np.ndarray:
    w = np.zeros(n, dtype=int)
    w[rng.permutation(n)[: int(round(n * p_treat))]] = 1
    return w


def estimate_ate(y, w) -> dict:
    y, w = np.asarray(y, float), np.asarray(w)
    y1, y0 = y[w == 1], y[w == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return {"estimate": float(est), "se": float(se), "ci_low": float(est - 1.96 * se),
            "ci_high": float(est + 1.96 * se), "p_value": float(2 * stats.norm.sf(abs(est / se)))}


def cuped_adjust(y, x_pre):
    """Y_adj = Y - theta (X - mean X), theta = Cov(Y, X) / Var(X). Unbiased under randomization."""
    y, x_pre = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x_pre, ddof=1)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean()), float(theta)


def srm_pvalue(n_treat: int, n_control: int, expected_share: float = 0.5) -> float:
    n = n_treat + n_control
    exp = np.array([n * expected_share, n * (1 - expected_share)])
    return float(stats.chisquare([n_treat, n_control], exp).pvalue)


def standardized_mean_differences(df: pd.DataFrame, w, cols) -> pd.Series:
    w = np.asarray(w)
    a, b = df.loc[w == 1, cols], df.loc[w == 0, cols]
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def mde(sigma: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    z = stats.norm.isf(alpha / 2) + stats.norm.ppf(power)
    return float(z * sigma * np.sqrt(2.0 / n_per_arm))


def required_n_per_arm(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    z = stats.norm.isf(alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sigma / delta) ** 2))


def rerandomization_study(y0, tau, x_pre, n_rep: int, rng: np.random.Generator,
                          effect_scale: float = 1.0) -> dict:
    """Repeat the randomization n_rep times on a FIXED finite population with fixed
    potential outcomes. Reports bias, CI coverage of the true sample effect, CI width
    and rejection rate for the naive and the CUPED estimator."""
    y0, tau, x_pre = (np.asarray(v, float) for v in (y0, tau, x_pre))
    tau = tau * effect_scale
    n = len(y0)
    sate = tau.mean()
    out = {k: [] for k in ("est_naive", "se_naive", "est_cuped", "se_cuped")}
    for _ in range(n_rep):
        w = randomize(n, rng)
        y = y0 + w * tau
        e = estimate_ate(y, w)
        ya, _ = cuped_adjust(y, x_pre)
        c = estimate_ate(ya, w)
        out["est_naive"].append(e["estimate"]); out["se_naive"].append(e["se"])
        out["est_cuped"].append(c["estimate"]); out["se_cuped"].append(c["se"])
    res = {"true_sample_effect": float(sate), "n_rep": n_rep}
    for k in ("naive", "cuped"):
        est, se = np.array(out[f"est_{k}"]), np.array(out[f"se_{k}"])
        res[k] = {"mean_estimate": float(est.mean()), "bias": float(est.mean() - sate),
                  "empirical_sd": float(est.std(ddof=1)), "mean_se": float(se.mean()),
                  "coverage": float(np.mean(np.abs(est - sate) <= 1.96 * se)),
                  "rejection_rate": float(np.mean(np.abs(est / se) > 1.96))}
    res["se_reduction_cuped"] = 1.0 - res["cuped"]["mean_se"] / res["naive"]["mean_se"]
    return res


# --------------------------------------------------------------------------- #
# peeking and group-sequential boundaries
# --------------------------------------------------------------------------- #
def group_sequential_constants(n_looks: int, alpha: float = 0.05, n_sim: int = 400_000,
                               seed: int = 0) -> dict:
    """Critical constants for equally spaced two-sided looks, by Monte Carlo over the
    joint law of the interim z statistics Z_k = S_k / sqrt(k).
        Pocock:             reject at look k if |Z_k| > c_P
        O'Brien-Fleming:    reject at look k if |Z_k| > c_B * sqrt(K / k)"""
    rng = np.random.default_rng(seed)
    s = np.cumsum(rng.standard_normal((n_sim, n_looks)), axis=1)
    k = np.arange(1, n_looks + 1)
    z = np.abs(s / np.sqrt(k))
    c_pocock = float(np.quantile(z.max(axis=1), 1 - alpha))
    c_obf = float(np.quantile((z * np.sqrt(k / n_looks)).max(axis=1), 1 - alpha))
    return {"pocock": c_pocock, "obrien_fleming": c_obf,
            "obf_bounds": [c_obf * np.sqrt(n_looks / j) for j in k]}


def peeking_study(y, n_looks: int = 5, n_rep: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """A/A tests on real outcomes: random labels, so any 'significant' result is a false positive.
    Compares fixed-horizon testing with stopping at the first look where p < alpha, and with
    Pocock / O'Brien-Fleming boundaries."""
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    gs = group_sequential_constants(n_looks, alpha, seed=seed + 1)
    looks = (np.arange(1, n_looks + 1) * n // n_looks)
    z_naive = stats.norm.isf(alpha / 2)
    hit_naive = hit_final = hit_pocock = hit_obf = 0
    for _ in range(n_rep):
        yy = y[rng.permutation(n)]
        w = (rng.random(n) < 0.5).astype(float)
        c1, c0 = np.cumsum(w), np.cumsum(1 - w)
        s1, s0 = np.cumsum(yy * w), np.cumsum(yy * (1 - w))
        q1, q0 = np.cumsum(yy**2 * w), np.cumsum(yy**2 * (1 - w))
        i = looks - 1
        m1, m0 = s1[i] / c1[i], s0[i] / c0[i]
        v1 = (q1[i] - c1[i] * m1**2) / (c1[i] - 1)
        v0 = (q0[i] - c0[i] * m0**2) / (c0[i] - 1)
        z = np.abs((m1 - m0) / np.sqrt(v1 / c1[i] + v0 / c0[i]))
        hit_naive += bool((z > z_naive).any())
        hit_final += bool(z[-1] > z_naive)
        hit_pocock += bool((z > gs["pocock"]).any())
        hit_obf += bool((z > np.array(gs["obf_bounds"])).any())
    f = lambda h: h / n_rep
    return {"n_looks": n_looks, "n_rep": n_rep, "fpr_fixed_horizon": f(hit_final),
            "fpr_peeking_naive": f(hit_naive), "fpr_pocock": f(hit_pocock), "fpr_obrien_fleming": f(hit_obf),
            "pocock_z": gs["pocock"], "obrien_fleming_c": gs["obrien_fleming"]}


# --------------------------------------------------------------------------- #
# uplift modelling and targeting
# --------------------------------------------------------------------------- #
def _hgb(seed: int, min_leaf: int):
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=min_leaf, l2_regularization=1.0,
                                         random_state=seed)


def fit_t_learner(x, y, w, seed: int = 0):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w)
    m1 = _hgb(seed, 200).fit(x[w == 1], y[w == 1])
    m0 = _hgb(seed, 200).fit(x[w == 0], y[w == 0])
    return lambda xn: m1.predict(np.asarray(xn, float)) - m0.predict(np.asarray(xn, float))


def fit_transformed_outcome(x, y, w, p_treat: float = 0.5, seed: int = 0):
    """Regress Y* = Y (W - e) / (e (1 - e)); E[Y* | X] = tau(X) in a randomized experiment."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w)
    ystar = y * (w - p_treat) / (p_treat * (1 - p_treat))
    m = _hgb(seed, 400).fit(x, ystar)
    return lambda xn: m.predict(np.asarray(xn, float))


def qini_coefficient(y, w, score) -> float:
    """Area between the cumulative-incremental-revenue curve and the random-targeting
    diagonal, per player. Computable on real RCT data (no ground truth needed).

    The curve is G(f) = (mean_treated - mean_control) among the top-f scored players,
    times the number of players in that segment: the incremental revenue from treating
    the whole segment, per player in the population."""
    y, w, score = np.asarray(y, float), np.asarray(w, float), np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    y, w = y[order], w[order]
    n = len(y)
    nt, nc = np.cumsum(w), np.cumsum(1 - w)
    mean_t = np.cumsum(y * w) / np.maximum(nt, 1)
    mean_c = np.cumsum(y * (1 - w)) / np.maximum(nc, 1)
    k = np.arange(1, n + 1)
    gain = np.where((nt > 0) & (nc > 0), (mean_t - mean_c) * k, 0.0) / n
    frac = k / n
    return float(np.trapezoid(gain - frac * gain[-1], frac))


def targeting_value_curve(net_value: np.ndarray, score: np.ndarray, fractions) -> np.ndarray:
    """Expected net value per 1,000 players when treating the top `fraction` by score."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(net_value[order])
    n = len(net_value)
    idx = np.clip((np.asarray(fractions) * n).astype(int) - 1, 0, n - 1)
    return cum[idx] / n * 1000.0


def uplift_decile_table(test: pd.DataFrame, w, tau_hat, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed (treated - control) vs true effect, by predicted-uplift decile."""
    d = test.assign(w=np.asarray(w), tau_hat=tau_hat)
    d["bin"] = pd.qcut(d["tau_hat"].rank(method="first", ascending=False), n_bins, labels=False) + 1
    rows = []
    for b, g in d.groupby("bin"):
        t, c = g[g["w"] == 1]["y"], g[g["w"] == 0]["y"]
        rows.append((b, len(g), g["tau_hat"].mean(), t.mean() - c.mean(), g["true_tau"].mean()))
    return pd.DataFrame(rows, columns=["decile", "n", "predicted", "observed", "true"])


def fit_dr_learner(x, y, w, p_treat: float = 0.5, seed: int = 0, n_folds: int = 2):
    """Doubly-robust (AIPW) learner. Outcome models are cross-fitted, then the final model
    regresses the pseudo-outcome
        psi = mu1(x) - mu0(x) + w (y - mu1(x)) / e - (1 - w) (y - mu0(x)) / (1 - e)
    on x. Subtracting the outcome models removes most of the outcome noise, which is
    what makes the plain transformed-outcome regression so noisy."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w)
    n = len(y)
    folds = np.random.default_rng(seed).permutation(n) % n_folds
    mu1, mu0 = np.zeros(n), np.zeros(n)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        mu1[te] = _hgb(seed, 200).fit(x[tr & (w == 1)], y[tr & (w == 1)]).predict(x[te])
        mu0[te] = _hgb(seed, 200).fit(x[tr & (w == 0)], y[tr & (w == 0)]).predict(x[te])
    psi = mu1 - mu0 + w * (y - mu1) / p_treat - (1 - w) * (y - mu0) / (1 - p_treat)
    final = _hgb(seed, 400).fit(x, psi)
    return lambda xn: final.predict(np.asarray(xn, float))
