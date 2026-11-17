"""Pack economics and probability.

A pack is a fixed number of slots; each slot draws a rarity tier from its own
categorical distribution, and a tier maps to a market value and a number of
distinct cards. Card values and odds are FICTIONAL, chosen for illustration.

Everything here has an exact (closed-form / Markov-chain) answer that is
checked against Monte Carlo in tests/test_packs.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

RARITIES = ("common", "uncommon", "rare", "epic", "legendary")


@dataclass(frozen=True)
class PackSpec:
    price: float
    slot_probs: tuple      # one row of tier probabilities per slot
    values: tuple          # market value ($) per tier
    cards_per_tier: tuple  # number of distinct cards per tier

    @property
    def n_slots(self) -> int:
        return len(self.slot_probs)

    def probs(self) -> np.ndarray:
        return np.asarray(self.slot_probs, dtype=float)

    def vals(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)


def default_pack() -> PackSpec:
    """4 base slots + 1 'hit' slot that is guaranteed rare-or-better."""
    base = (0.700, 0.220, 0.070, 0.009, 0.001)
    hit = (0.000, 0.000, 0.800, 0.170, 0.030)
    return PackSpec(
        price=6.00,
        slot_probs=(base,) * 4 + (hit,),
        values=(0.05, 0.25, 1.50, 8.00, 60.00),
        cards_per_tier=(20, 12, 8, 4, 2),
    )


# --------------------------------------------------------------------------- #
# expected value and payout distribution
# --------------------------------------------------------------------------- #
def slot_ev(spec: PackSpec) -> np.ndarray:
    return spec.probs() @ spec.vals()


def pack_ev(spec: PackSpec) -> float:
    return float(slot_ev(spec).sum())


def rtp(spec: PackSpec) -> float:
    """Return-to-player: expected market value of the cards / pack price."""
    return pack_ev(spec) / spec.price


def payout_distribution(spec: PackSpec, decimals: int = 6):
    """Exact distribution of total pack value (convolution over slots)."""
    dist = {0.0: 1.0}
    for row in spec.probs():
        new: dict[float, float] = {}
        for v0, p0 in dist.items():
            for val, p in zip(spec.vals(), row):
                if p > 0:
                    key = round(v0 + val, decimals)
                    new[key] = new.get(key, 0.0) + p0 * p
        dist = new
    values = np.array(sorted(dist))
    probs = np.array([dist[v] for v in values])
    return values, probs


def payout_summary(spec: PackSpec) -> dict:
    v, p = payout_distribution(spec)
    mean = float((v * p).sum())
    sd = float(np.sqrt(((v - mean) ** 2 * p).sum()))
    cdf = np.cumsum(p)

    def q(a: float) -> float:
        return float(v[np.searchsorted(cdf, a - 1e-12)])

    return {
        "mean_value": mean,
        "sd_value": sd,
        "skew": float(((v - mean) ** 3 * p).sum() / sd**3),
        "rtp": mean / spec.price,
        "p_value_at_least_price": float(p[v >= spec.price - 1e-9].sum()),
        "p10": q(0.10),
        "median": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
    }


# --------------------------------------------------------------------------- #
# rarity calibration: hit a target RTP by scaling the top tiers' odds
# --------------------------------------------------------------------------- #
def scale_top_tiers(spec: PackSpec, s: float, top_from: int = 3) -> PackSpec:
    """Multiply epic+ odds by s in every slot; the freed/needed probability mass
    comes from the lowest tier that slot can produce."""
    probs = spec.probs().copy()
    for row in probs:
        nonzero = np.flatnonzero(row > 0)
        absorber = nonzero[0]
        if absorber >= top_from:
            continue
        top = np.arange(top_from, row.size)
        delta = row[top].sum() * (s - 1.0)
        row[top] = row[top] * s
        row[absorber] -= delta
        if row[absorber] < -1e-12:
            raise ValueError("scale too large: absorber tier would go negative")
    return replace(spec, slot_probs=tuple(tuple(map(float, r)) for r in probs))


def calibrate_rarity(spec: PackSpec, target_rtp: float, top_from: int = 3):
    """Find the scale s on epic+ odds so that RTP == target_rtp. Returns (spec, s)."""
    probs = spec.probs()
    s_max = np.inf
    for row in probs:
        nz = np.flatnonzero(row > 0)
        if nz[0] < top_from and row[top_from:].sum() > 0:
            s_max = min(s_max, 1.0 + row[nz[0]] / row[top_from:].sum())
    f = lambda s: rtp(scale_top_tiers(spec, s, top_from)) - target_rtp
    hi = s_max - 1e-9
    if f(0.0) > 0 or f(hi) < 0:
        raise ValueError("target RTP outside the reachable range")
    s = brentq(f, 0.0, hi, xtol=1e-12)
    return scale_top_tiers(spec, s, top_from), float(s)


# --------------------------------------------------------------------------- #
# chase-card probabilities and pity timers
# --------------------------------------------------------------------------- #
def p_tier_per_pack(spec: PackSpec, tier: int) -> float:
    """P(at least one card of `tier` in a pack)."""
    return float(1.0 - np.prod(1.0 - spec.probs()[:, tier]))


def pity_adjusted_rate(p0: float, n_pity: int) -> float:
    """Per-pack hit probability p' such that, with a guaranteed hit at pack n_pity,
    hits still occur at the original long-run rate p0 (renewal-reward: rate = 1/E[T],
    with T = min(Geometric(p'), n_pity))."""
    if n_pity <= 1 / p0:
        raise ValueError("pity threshold must exceed the mean hit time 1/p0")
    rate = lambda p: p / (1.0 - (1.0 - p) ** n_pity)
    return float(brentq(lambda p: rate(p) - p0, 1e-9, p0, xtol=1e-14))


def hit_time_cdf(p: float, n_max: int, n_pity: int | None = None) -> np.ndarray:
    """P(first hit within n packs), n = 0..n_max, optionally capped by a pity timer."""
    n = np.arange(n_max + 1)
    cdf = 1.0 - (1.0 - p) ** n
    if n_pity is not None:
        cdf = np.where(n >= n_pity, 1.0, cdf)
    return cdf


def hit_time_stats(p: float, n_pity: int | None = None, n_max: int = 5000) -> dict:
    cdf = hit_time_cdf(p, n_max, n_pity)
    pmf = np.diff(cdf, prepend=0.0)
    n = np.arange(n_max + 1)
    mean = float((n * pmf).sum())
    sd = float(np.sqrt(((n - mean) ** 2 * pmf).sum()))
    q = lambda a: int(np.searchsorted(cdf, a - 1e-12))
    return {"mean": mean, "sd": sd, "p50": q(0.5), "p90": q(0.9), "p99": q(0.99)}


# --------------------------------------------------------------------------- #
# exact set completion via an absorbing Markov chain over owned-card bitmasks
# --------------------------------------------------------------------------- #
def chase_cards(spec: PackSpec, min_tier: int = 3) -> list[int]:
    """Tier index of each distinct card in tiers >= min_tier."""
    return [t for t in range(min_tier, len(spec.values)) for _ in range(spec.cards_per_tier[t])]


def pack_transition_matrix(spec: PackSpec, min_tier: int = 3) -> np.ndarray:
    cards = chase_cards(spec, min_tier)
    c = len(cards)
    if c > 10:
        raise ValueError("chase set too large for the exact bitmask chain (max 10 cards)")
    s = 1 << c
    full = np.eye(s)
    for row in spec.probs():
        p_card = np.array([row[t] / spec.cards_per_tier[t] for t in cards])
        p_none = 1.0 - p_card.sum()
        slot = np.zeros((s, s))
        for m in range(s):
            slot[m, m] += p_none
            for j in range(c):
                slot[m, m | (1 << j)] += p_card[j]
        full = full @ slot
    return full


def set_completion(spec: PackSpec, min_tier: int = 3, max_packs: int = 20000) -> dict:
    """Exact distribution of packs needed to own every card in tiers >= min_tier."""
    p = pack_transition_matrix(spec, min_tier)
    s = p.shape[0]
    q = p[: s - 1, : s - 1]
    expected = float(np.linalg.solve(np.eye(s - 1) - q, np.ones(s - 1))[0])
    dist = np.zeros(s)
    dist[0] = 1.0
    cdf = [0.0]
    for _ in range(max_packs):
        dist = dist @ p
        cdf.append(float(dist[s - 1]))
        if cdf[-1] > 0.99999:
            break
    cdf = np.array(cdf)
    qf = lambda a: int(np.searchsorted(cdf, a - 1e-12))
    return {
        "n_cards": len(chase_cards(spec, min_tier)),
        "expected_packs": expected,
        "p50": qf(0.50),
        "p90": qf(0.90),
        "p99": qf(0.99),
        "cdf": cdf,
    }


# --------------------------------------------------------------------------- #
# Monte Carlo counterparts (used to verify the exact results)
# --------------------------------------------------------------------------- #
def _draw_tiers(row: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.cumsum(row), u, side="right").clip(0, row.size - 1)


def simulate_pack_values(spec: PackSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    total = np.zeros(n)
    vals = spec.vals()
    for row in spec.probs():
        total += vals[_draw_tiers(row, rng.random(n))]
    return total


def simulate_set_completion(spec: PackSpec, n_sims: int, rng: np.random.Generator,
                            min_tier: int = 3, max_packs: int = 100000) -> np.ndarray:
    cards = chase_cards(spec, min_tier)
    c = len(cards)
    offsets = {}
    k = 0
    for t in range(min_tier, len(spec.values)):
        offsets[t] = k
        k += spec.cards_per_tier[t]
    counts = np.asarray(spec.cards_per_tier)
    owned = np.zeros((n_sims, c), dtype=bool)
    packs = np.zeros(n_sims, dtype=int)
    active = np.arange(n_sims)
    rows = spec.probs()
    n = 0
    while active.size and n < max_packs:
        n += 1
        packs[active] = n
        for row in rows:
            tiers = _draw_tiers(row, rng.random(active.size))
            for t in range(min_tier, len(spec.values)):
                hit = tiers == t
                if hit.any():
                    idx = offsets[t] + rng.integers(0, counts[t], size=int(hit.sum()))
                    owned[active[hit], idx] = True
        done = owned[active].all(axis=1)
        active = active[~done]
    return packs
