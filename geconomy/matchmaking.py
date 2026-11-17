"""Skill ratings (Elo, Glicko-1, Bradley-Terry MLE) and pairing policies on simulated matches.

Ground truth: each player has a latent skill theta (rating points); P(a beats b) follows the
Elo logistic law in theta. This is a well-specified setting by construction, so it shows how
fast each method converges and what pairing costs or buys - not how Elo fares on a real game
whose results are not Bradley-Terry.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression

Q = np.log(10.0) / 400.0


def expected_score(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(r_b) - np.asarray(r_a)) / 400.0))


# ----------------------------------- Elo ------------------------------------- #
def elo_update(r_a, r_b, score_a, k):
    e = expected_score(r_a, r_b)
    delta = k * (score_a - e)
    return r_a + delta, r_b - delta


# ---------------------------------- Glicko-1 ---------------------------------- #
def _g(rd):
    return 1.0 / np.sqrt(1.0 + 3.0 * Q**2 * np.asarray(rd) ** 2 / np.pi**2)


def glicko_update(r, rd, opp_r, opp_rd, scores):
    """One rating period for one player against arrays of opponents (Glickman 1999)."""
    opp_r, opp_rd, scores = (np.asarray(v, float) for v in (opp_r, opp_rd, scores))
    g = _g(opp_rd)
    e = 1.0 / (1.0 + 10.0 ** (-g * (r - opp_r) / 400.0))
    d2 = 1.0 / (Q**2 * np.sum(g**2 * e * (1.0 - e)))
    denom = 1.0 / rd**2 + 1.0 / d2
    return r + Q / denom * np.sum(g * (scores - e)), float(np.sqrt(1.0 / denom))


def glicko_update_vec(r, rd, opp_r, opp_rd, score):
    """Vectorised single-opponent update: every player plays exactly one match this period."""
    g = _g(opp_rd)
    e = 1.0 / (1.0 + 10.0 ** (-g * (r - opp_r) / 400.0))
    d2 = 1.0 / (Q**2 * g**2 * e * (1.0 - e))
    denom = 1.0 / rd**2 + 1.0 / d2
    return r + Q / denom * g * (score - e), np.sqrt(1.0 / denom)


def glicko_predict(r_a, rd_a, r_b, rd_b):
    g = _g(np.sqrt(np.asarray(rd_a) ** 2 + np.asarray(rd_b) ** 2))
    return 1.0 / (1.0 + 10.0 ** (-g * (np.asarray(r_a) - np.asarray(r_b)) / 400.0))


# ------------------------------- Bradley-Terry -------------------------------- #
def bradley_terry(a, b, a_wins, n_players: int, c: float = 1.0) -> np.ndarray:
    """L2-regularised logistic MLE of skills from all matches so far, in Elo points."""
    m = len(a)
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack([a, b]).ravel()
    vals = np.tile([1.0, -1.0], m)
    x = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n_players))
    clf = LogisticRegression(fit_intercept=False, C=c, max_iter=1000).fit(x, np.asarray(a_wins, int))
    return 1500.0 + clf.coef_[0] * 400.0 / np.log(10.0)


# ------------------------------- pairing + play -------------------------------- #
def pair(policy: str, ratings: np.ndarray, theta: np.ndarray, rng: np.random.Generator,
         explore: float = 0.3):
    n = len(theta)
    if policy == "random":
        order = rng.permutation(n)
    elif policy == "rating":          # sort by current rating, pair neighbours (jitter breaks ties)
        order = np.argsort(ratings + rng.normal(0, 10.0, n))
    elif policy == "rating_explore":  # a share of players is paired at random so that the
        perm = rng.permutation(n)     # comparison graph stays connected; the rest by rating
        n_exp = int(round(explore * n / 2)) * 2
        explore_ids, rest = perm[:n_exp], perm[n_exp:]
        rest = rest[np.argsort(ratings[rest] + rng.normal(0, 10.0, rest.size))]
        order = np.concatenate([explore_ids, rest])
    elif policy == "oracle":          # sort by TRUE skill: the fairness ceiling, but the same
        order = np.argsort(theta + rng.normal(0, 1e-6, n))   # neighbours meet every round
    else:
        raise ValueError(policy)
    return order[0::2], order[1::2]


def run_rating_simulation(n_players: int = 1000, rounds: int = 60, policy: str = "random",
                          method: str = "elo", k: float = 32.0, seed: int = 0,
                          skill_sd: float = 200.0, glicko_c: float = 5.0) -> dict:
    """Every round each player plays one match. Returns per-round diagnostics:
       logloss    online log-loss of the pre-match predicted win probability
       bayes_ll   the log-loss of the TRUE probabilities for the same matches (floor)
       spearman   rank correlation of current ratings with true skill
       fair_share share of matches whose true win probability lies in [0.4, 0.6]"""
    assert n_players % 2 == 0
    rng = np.random.default_rng(seed)
    theta = rng.normal(1500.0, skill_sd, n_players)
    r = np.full(n_players, 1500.0)
    rd = np.full(n_players, 350.0)
    log_loss, bayes, rho, fair = [], [], [], []
    ma, mb, mw = [], [], []
    ll = lambda s, p: float(-np.mean(s * np.log(p) + (1 - s) * np.log(1 - p)))
    for _ in range(rounds):
        a, b = pair(policy, r, theta, rng)
        p_true = expected_score(theta[a], theta[b])
        s = (rng.random(len(a)) < p_true).astype(float)
        if method == "elo":
            p_hat = expected_score(r[a], r[b])
        else:
            p_hat = glicko_predict(r[a], rd[a], r[b], rd[b])
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
        log_loss.append(ll(s, p_hat)); bayes.append(ll(s, p_true))
        fair.append(float(np.mean((p_true >= 0.4) & (p_true <= 0.6))))
        if method == "elo":
            r[a], r[b] = elo_update(r[a], r[b], s, k)
        else:
            ra, rda = glicko_update_vec(r[a], rd[a], r[b], rd[b], s)
            rb_, rdb = glicko_update_vec(r[b], rd[b], r[a], rd[a], 1 - s)
            r[a], rd[a], r[b], rd[b] = ra, rda, rb_, rdb
            rd = np.minimum(np.sqrt(rd**2 + glicko_c**2), 350.0)
        rho.append(float(stats.spearmanr(r, theta)[0]))
        ma.append(a); mb.append(b); mw.append(s)
    return {"theta": theta, "ratings": r, "rd": rd, "logloss": np.array(log_loss),
            "bayes_ll": np.array(bayes), "spearman": np.array(rho), "fair_share": np.array(fair),
            "matches": (np.concatenate(ma), np.concatenate(mb), np.concatenate(mw))}
