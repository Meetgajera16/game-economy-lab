"""SQL KPI layer (SQLite). Every query is cross-checked against an independent
pandas computation in tests/test_sql_metrics.py and in run_all.py."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .simulate import World

CHANNEL_KPI_SQL = """
WITH tx AS (
    SELECT t.player_id, p.channel, t.amount
    FROM transactions t JOIN players p USING (player_id)
    WHERE t.calendar_week <= :t_cal
), per_player AS (
    SELECT player_id, channel, COUNT(*) AS n_tx, SUM(amount) AS revenue
    FROM tx GROUP BY player_id, channel
), spend AS (
    SELECT channel, SUM(spend) AS spend, SUM(signups) AS signups FROM weekly GROUP BY channel
)
SELECT pp.channel,
       COUNT(*)                                                  AS players,
       SUM(pp.revenue)                                           AS revenue,
       SUM(pp.revenue) * 1.0 / COUNT(*)                          AS revenue_per_player,
       AVG(CASE WHEN pp.n_tx >= 2 THEN 1.0 ELSE 0.0 END)         AS repeat_rate,
       SUM(pp.revenue) * 1.0 / SUM(pp.n_tx)                      AS avg_order_value,
       s.spend * 1.0 / s.signups                                 AS cac
FROM per_player pp JOIN spend s USING (channel)
GROUP BY pp.channel ORDER BY pp.channel
"""

# share of players active (>=1 repeat purchase) in week k after signup, among players
# observed for the whole of week k
RETENTION_SQL = """
WITH RECURSIVE grid(k) AS (SELECT 1 UNION ALL SELECT k + 1 FROM grid WHERE k < :max_k),
obs AS (
    SELECT player_id, channel, :t_cal - signup_time AS age FROM players WHERE signup_time < :t_cal
),
act AS (
    SELECT DISTINCT player_id, CAST(t AS INTEGER) AS k
    FROM transactions WHERE is_first = 0 AND calendar_week <= :t_cal
)
SELECT o.channel, g.k AS week_since_signup,
       COUNT(*)                                   AS n_observed,
       COUNT(a.player_id)                         AS n_active,
       COUNT(a.player_id) * 1.0 / COUNT(*)        AS retention
FROM obs o
JOIN grid g ON o.age >= g.k + 1
LEFT JOIN act a ON a.player_id = o.player_id AND a.k = g.k
GROUP BY o.channel, g.k ORDER BY o.channel, g.k
"""

CUM_REVENUE_SQL = """
WITH wk AS (
    SELECT p.channel, CAST(t.calendar_week AS INTEGER) AS week, SUM(t.amount) AS revenue
    FROM transactions t JOIN players p USING (player_id)
    WHERE t.calendar_week <= :t_cal GROUP BY p.channel, CAST(t.calendar_week AS INTEGER)
)
SELECT channel, week, revenue,
       SUM(revenue) OVER (PARTITION BY channel ORDER BY week) AS cum_revenue
FROM wk ORDER BY channel, week
"""

CONCENTRATION_SQL = """
WITH rev AS (
    SELECT player_id, SUM(amount) AS r FROM transactions WHERE calendar_week <= :t_cal GROUP BY player_id
), ranked AS (
    SELECT r, NTILE(100) OVER (ORDER BY r DESC) AS pct FROM rev
)
SELECT SUM(CASE WHEN pct = 1  THEN r END) * 1.0 / SUM(r) AS top1_share,
       SUM(CASE WHEN pct <= 10 THEN r END) * 1.0 / SUM(r) AS top10_share,
       SUM(CASE WHEN pct <= 20 THEN r END) * 1.0 / SUM(r) AS top20_share
FROM ranked
"""


def build_db(world: World) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    world.players[["player_id", "channel", "signup_time"]].to_sql("players", con, index=False)
    world.transactions[["player_id", "t", "calendar_week", "amount", "is_first"]].to_sql(
        "transactions", con, index=False)
    world.weekly.to_sql("weekly", con, index=False)
    con.execute("CREATE INDEX ix_tx_player ON transactions(player_id)")
    con.execute("CREATE INDEX ix_players_channel ON players(channel)")
    return con


def run_sql_layer(world: World, t_cal: float | None = None, max_k: int = 12) -> dict:
    t_cal = float(world.n_weeks if t_cal is None else t_cal)
    con = build_db(world)
    q = lambda sql, **kw: pd.read_sql_query(sql, con, params={"t_cal": t_cal, **kw})
    out = {
        "channel_kpis": q(CHANNEL_KPI_SQL),
        "retention": q(RETENTION_SQL, max_k=max_k),
        "cum_revenue": q(CUM_REVENUE_SQL),
        "concentration": q(CONCENTRATION_SQL),
    }
    con.close()
    return out


# ----------------------------- pandas cross-checks ---------------------------- #
def pandas_channel_kpis(world: World, t_cal: float) -> pd.DataFrame:
    tx = world.transactions[world.transactions["calendar_week"] <= t_cal]
    pp = tx.groupby("player_id").agg(n_tx=("amount", "size"), revenue=("amount", "sum"))
    pp = pp.join(world.players.set_index("player_id")["channel"])
    g = pp.groupby("channel")
    out = pd.DataFrame({
        "players": g.size(),
        "revenue": g["revenue"].sum(),
        "revenue_per_player": g["revenue"].sum() / g.size(),
        "repeat_rate": g["n_tx"].apply(lambda s: float((s >= 2).mean())),
        "avg_order_value": g["revenue"].sum() / g["n_tx"].sum(),
    })
    wk = world.weekly.groupby("channel")[["spend", "signups"]].sum()
    out["cac"] = wk["spend"] / wk["signups"]
    return out.reset_index()


def pandas_retention(world: World, t_cal: float, max_k: int = 12) -> pd.DataFrame:
    p = world.players[world.players["signup_time"] < t_cal].copy()
    p["age"] = t_cal - p["signup_time"]
    tx = world.transactions
    rep = tx[(tx["is_first"] == 0) & (tx["calendar_week"] <= t_cal)].copy()
    rep["k"] = np.floor(rep["t"]).astype(int)
    active = set(zip(rep["player_id"], rep["k"]))
    rows = []
    for k in range(1, max_k + 1):
        obs = p[p["age"] >= k + 1]
        flag = np.fromiter(((pid, k) in active for pid in obs["player_id"]), bool, len(obs))
        df = pd.DataFrame({"channel": obs["channel"].to_numpy(), "a": flag})
        for ch, d in df.groupby("channel"):
            rows.append((ch, k, len(d), int(d["a"].sum()), float(d["a"].mean())))
    return pd.DataFrame(rows, columns=["channel", "week_since_signup", "n_observed", "n_active", "retention"]
                        ).sort_values(["channel", "week_since_signup"]).reset_index(drop=True)


def _ntile_top_share(values: np.ndarray, pct_max: int) -> float:
    """Replicates SQL NTILE(100) semantics: the first (n mod 100) buckets get one extra row."""
    v = np.sort(values)[::-1]
    n = len(v)
    base, extra = divmod(n, 100)
    rows = sum(base + (1 if i < extra else 0) for i in range(pct_max))
    return float(v[:rows].sum() / v.sum())


def pandas_concentration(world: World, t_cal: float) -> dict:
    tx = world.transactions[world.transactions["calendar_week"] <= t_cal]
    r = tx.groupby("player_id")["amount"].sum().to_numpy()
    return {"top1_share": _ntile_top_share(r, 1), "top10_share": _ntile_top_share(r, 10),
            "top20_share": _ntile_top_share(r, 20)}
