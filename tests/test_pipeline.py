"""End-to-end smoke test: the whole pipeline runs in fast mode and its headline invariants hold.
Writes to a temp directory so the real results/ are untouched."""
import json

import pytest

import run_all


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    s = run_all.main(fast=True, make_figures=True, out_dir=out)
    assert (out / "results" / "summary.json").exists()
    assert len(list((out / "figures").glob("*.png"))) == 7
    return s


def test_summary_is_json_serialisable(summary):
    json.dumps(run_all.py(summary))


def test_sql_layer_matches_pandas_exactly(summary):
    assert max(summary["sql"]["max_abs_diff_vs_pandas"].values()) < 1e-9


def test_pack_rtp_calibrated_and_exact_matches_monte_carlo(summary):
    p = summary["packs"]
    assert p["payout"]["rtp"] == pytest.approx(p["target_rtp"], abs=1e-9)
    assert p["monte_carlo"]["mean_value"] == pytest.approx(p["pack_ev"], rel=0.01)
    assert p["pity"]["mean"] == pytest.approx(p["legendary_flat"]["mean"], rel=1e-6)   # rate-equivalent by construction
    assert p["pity"]["p90"] < p["legendary_flat"]["p90"]


def test_ltv_recovers_known_truth(summary):
    l = summary["ltv"]
    assert l["parameter_recovery"]["max_abs_rel_error_rate_params"] < 0.08
    assert abs(l["holdout_13w"]["total_purchase_error"]) < 0.03
    assert l["holdout_13w"]["rmse_model"] < l["holdout_13w"]["rmse_channel_mean"]


def test_experiment_invariants(summary):
    e = summary["experiment"]
    assert e["srm_p_value"] > 0.01
    assert e["naive"]["ci_low"] < e["true_sample_effect"] < e["naive"]["ci_high"]
    assert e["peeking"]["fpr_peeking_naive"] > 2 * e["peeking"]["fpr_fixed_horizon"]
    assert e["peeking"]["fpr_pocock"] < 0.08 and e["peeking"]["fpr_obrien_fleming"] < 0.08


def test_uplift_beats_heuristics_and_treat_all(summary):
    t = summary["experiment"]["uplift"]["summary"].set_index("policy")
    dr = t.loc["dr_learner", "threshold_net_per_1000_mean"]
    assert dr > 0 > t.loc["treat_all", "threshold_net_per_1000_mean"]
    assert dr < t.loc["oracle", "threshold_net_per_1000_mean"]
    assert t.loc["dr_learner", "spearman_vs_true_mean"] > t.loc["high_ltv_heuristic", "spearman_vs_true_mean"]


def test_ltv_aware_allocation_beats_naive_rules(summary):
    a = summary["acquisition"]
    prof = {r["policy"]: r["mean"] for r in a["panels"]["profit_per_week"]}
    for naive in ("current_allocation", "equal_split", "maximize_signups"):
        assert prof["channel_ltv_aware"] > prof[naive]
    assert prof["channel_ltv_aware"] <= prof["true_optimum"] + 1e-6


def test_matchmaking_ratings_converge(summary):
    comp = summary["matchmaking"]["method_comparison"].set_index("round")
    assert comp.loc[60, "elo_k32"] > 0.9 and comp.loc[60, "glicko1"] > 0.9
    assert comp.loc[60, "bradley_terry_mle"] >= comp.loc[60, "elo_k32"] - 0.01
