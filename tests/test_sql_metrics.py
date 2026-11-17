import numpy as np
import pandas as pd
import pytest

from geconomy import simulate, sql_metrics


@pytest.fixture(scope="module")
def world():
    return simulate.simulate_world(seed=5, n_weeks=30, holdout_weeks=4)


def test_channel_kpis_match_pandas(world):
    sql = sql_metrics.run_sql_layer(world)["channel_kpis"].set_index("channel")
    pdf = sql_metrics.pandas_channel_kpis(world, world.n_weeks).set_index("channel")
    pd.testing.assert_frame_equal(sql.sort_index(), pdf.sort_index()[sql.columns],
                                  check_dtype=False, rtol=1e-9)


def test_retention_matches_pandas(world):
    sql = sql_metrics.run_sql_layer(world)["retention"]
    pdf = sql_metrics.pandas_retention(world, world.n_weeks)
    pd.testing.assert_frame_equal(sql.reset_index(drop=True), pdf, check_dtype=False, rtol=1e-9)


def test_concentration_matches_pandas(world):
    sql = sql_metrics.run_sql_layer(world)["concentration"].iloc[0].to_dict()
    pdf = sql_metrics.pandas_concentration(world, world.n_weeks)
    for k, v in pdf.items():
        assert sql[k] == pytest.approx(v, rel=1e-9)
    assert 0 < sql["top1_share"] < sql["top10_share"] < sql["top20_share"] < 1


def test_cumulative_revenue_window_function(world):
    cum = sql_metrics.run_sql_layer(world)["cum_revenue"]
    for ch, d in cum.groupby("channel"):
        assert (d["cum_revenue"].diff().dropna() >= 0).all()
        assert d["cum_revenue"].iloc[-1] == pytest.approx(d["revenue"].sum())
