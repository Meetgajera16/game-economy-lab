"""Run every analysis end to end and write results/summary.json, results/tables/*.csv
and figures/*.png.

    python run_all.py            # full run (about 2-3 minutes)
    python run_all.py --fast     # fewer replicates, for a smoke test

All data are simulated; seeds are fixed, so a re-run reproduces every number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from geconomy import (acquisition as acq, economics, experiment as ex, ltv, matchmaking as mm,
                      packs, retention, simulate, sql_metrics)

ROOT = Path(__file__).parent
RESULTS, TABLES, FIGURES = ROOT / "results", ROOT / "results" / "tables", ROOT / "figures"
SEED = 7
HORIZON_LTV_WEEKS = 52
PROMO_WINDOW_WEEKS = 4


def py(o):
    """JSON-safe conversion."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return py(o.tolist())
    if isinstance(o, pd.DataFrame):
        return py(o.reset_index().to_dict(orient="records"))
    return o


def save_table(df: pd.DataFrame, name: str, index: bool = False):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index)


# --------------------------------------------------------------------------- #
def run_packs(fast: bool) -> tuple[dict, dict]:
    spec0 = packs.default_pack()
    spec, scale = packs.calibrate_rarity(spec0, economics.TARGET_RTP)
    summ = packs.payout_summary(spec)
    rng = np.random.default_rng(0)
    n_mc = 400_000 if fast else 2_000_000
    sim = packs.simulate_pack_values(spec, n_mc, rng)
    p_leg = packs.p_tier_per_pack(spec, 4)
    n_pity = 60
    p_pity = packs.pity_adjusted_rate(p_leg, n_pity)
    flat, pity = packs.hit_time_stats(p_leg), packs.hit_time_stats(p_pity, n_pity)
    sc = packs.set_completion(spec)
    sc_mc = packs.simulate_set_completion(spec, 4000 if fast else 20_000, np.random.default_rng(1))
    v, p = packs.payout_distribution(spec)
    cdf_flat = packs.hit_time_cdf(p_leg, 200)
    cdf_pity = packs.hit_time_cdf(p_pity, 200, n_pity)
    out = {
        "default_rtp": packs.rtp(spec0),
        "target_rtp": economics.TARGET_RTP,
        "epic_plus_odds_scale": scale,
        "pack_price": spec.price,
        "pack_ev": packs.pack_ev(spec),
        "payout": summ,
        "monte_carlo": {"n": n_mc, "mean_value": float(sim.mean()),
                        "p_value_at_least_price": float((sim >= spec.price - 1e-9).mean())},
        "p_legendary_per_pack": p_leg,
        "p_legendary_in_10_packs": 1 - (1 - p_leg) ** 10,
        "legendary_flat": flat,
        "pity": {"threshold": n_pity, "adjusted_p": p_pity, **pity},
        "chase_set": {"n_cards": sc["n_cards"], "expected_packs": sc["expected_packs"],
                      "expected_spend": sc["expected_packs"] * spec.price,
                      "p50": sc["p50"], "p90": sc["p90"], "p99": sc["p99"],
                      "monte_carlo_mean": float(sc_mc.mean()),
                      "monte_carlo_p90": float(np.quantile(sc_mc, 0.9))},
        "economics": economics.unit_economics(),
    }
    art = {"value_cdf": (v, np.cumsum(p)), "price": spec.price, "cdf_flat": cdf_flat,
           "cdf_pity": cdf_pity, "n_pity": n_pity, "rtp": packs.rtp(spec)}
    save_table(pd.DataFrame({"pack_value": v, "probability": p, "cdf": np.cumsum(p)}), "pack_value_distribution")
    save_table(pd.DataFrame({"packs": np.arange(201), "flat_cdf": cdf_flat, "pity_cdf": cdf_pity}),
               "legendary_hit_time_cdf")
    return out, art


def run_world(fast: bool):
    world = simulate.simulate_world(seed=SEED)
    p, tx = world.players, world.transactions
    t_cal = world.n_weeks
    alive = (p["true_dropout_time"] > (t_cal - p["signup_time"]))
    out = {"seed": SEED, "n_players": int(len(p)), "n_transactions": int(len(tx)),
           "players_by_channel": p.groupby("channel").size().to_dict(),
           "true_share_still_active": float(alive.mean()),
           "weekly_cac": (world.weekly.groupby("channel")[["spend", "signups"]].sum()
                          .assign(cac=lambda d: d["spend"] / d["signups"]).query("spend > 0")["cac"].to_dict())}
    return world, out


def run_sql(world) -> dict:
    t_cal = float(world.n_weeks)
    res = sql_metrics.run_sql_layer(world, t_cal)
    kpi_sql = res["channel_kpis"].set_index("channel").sort_index()
    kpi_pd = sql_metrics.pandas_channel_kpis(world, t_cal).set_index("channel").sort_index()[kpi_sql.columns]
    ret_pd = sql_metrics.pandas_retention(world, t_cal)
    conc_pd = sql_metrics.pandas_concentration(world, t_cal)
    max_kpi = float((kpi_sql - kpi_pd).abs().max().max())
    max_ret = float((res["retention"].reset_index(drop=True)["retention"] - ret_pd["retention"]).abs().max())
    conc = res["concentration"].iloc[0].to_dict()
    max_conc = max(abs(conc[k] - conc_pd[k]) for k in conc_pd)
    save_table(res["channel_kpis"], "sql_channel_kpis")
    save_table(res["retention"], "sql_weekly_retention")
    save_table(res["cum_revenue"], "sql_cumulative_revenue")
    return {"channel_kpis": kpi_sql.reset_index().to_dict(orient="records"), "concentration": conc,
            "max_abs_diff_vs_pandas": {"channel_kpis": max_kpi, "retention": max_ret, "concentration": max_conc},
            "retention_week4": (res["retention"].query("week_since_signup == 4")
                                .set_index("channel")["retention"].to_dict())}


def run_ltv(world) -> tuple[dict, dict]:
    t_cal = float(world.n_weeks)
    rfm = ltv.rfm_summary(world.players, world.transactions, t_cal)
    models = ltv.fit_models(rfm)
    rec = ltv.parameter_recovery(models, simulate.TRUTH)
    val = ltv.validate_holdout(rfm, models, world.transactions, t_cal, 13)
    margin = economics.unit_economics()["margin_rate"]
    rows = []
    cac = (world.weekly.groupby("channel")[["spend", "signups"]].sum().assign(cac=lambda d: d.spend / d.signups)["cac"])
    for ch, (bg, gg) in models.items():
        t = simulate.TRUTH[ch]
        est = ltv.expected_value_new(bg, gg, HORIZON_LTV_WEEKS, economics.ANNUAL_DISCOUNT)
        tru = ltv.expected_value_new(ltv.BGNBD(t.r, t.alpha, t.a, t.b), ltv.GammaGamma(t.gg_p, t.gg_q, t.gg_gamma),
                                     HORIZON_LTV_WEEKS, economics.ANNUAL_DISCOUNT)
        rows.append({"channel": ch, "ltv_revenue_est": est["pv_revenue"], "ltv_revenue_true": tru["pv_revenue"],
                     "ltv_margin_est": margin * est["pv_revenue"], "ltv_margin_true": margin * tru["pv_revenue"],
                     "cac": float(cac[ch]), "bg_loglik": bg.loglik_, "bg_converged": bg.converged_})
    tab = pd.DataFrame(rows).set_index("channel")
    tab["ltv_to_cac_est"] = tab["ltv_margin_est"] / tab["cac"].replace(0, np.nan)
    tab["ltv_to_cac_true"] = tab["ltv_margin_true"] / tab["cac"].replace(0, np.nan)
    tab["ltv_error"] = tab["ltv_margin_est"] / tab["ltv_margin_true"] - 1
    save_table(rec, "ltv_parameter_recovery")
    save_table(val["by_bucket"].reset_index(), "ltv_holdout_by_frequency")
    save_table(val["by_channel"].reset_index(), "ltv_holdout_by_channel")
    save_table(tab.reset_index(), "ltv_by_channel")
    piv = rec.pivot(index="channel", columns="param", values=["true", "estimate"])
    drop_true = piv["true"]["a"] / (piv["true"]["a"] + piv["true"]["b"])
    drop_est = piv["estimate"]["a"] / (piv["estimate"]["a"] + piv["estimate"]["b"])
    out = {
        "n_players": int(len(rfm)),
        "dropout_mean_a_over_a_plus_b": {"true": drop_true.to_dict(), "estimate": drop_est.to_dict(),
                                         "max_abs_rel_error": float(((drop_est / drop_true) - 1).abs().max())},
        "parameter_recovery": {
            "max_abs_rel_error_all": float(rec["rel_error"].abs().max()),
            "max_abs_rel_error_rate_params": float(rec[rec["param"].isin(["r", "alpha"])]["rel_error"].abs().max()),
            "median_abs_rel_error": float(rec["rel_error"].abs().median()),
            "mean_spend_rel_error": {ch: models[ch][1].mean_spend / simulate.TRUTH[ch].mean_spend - 1 for ch in models},
        },
        "holdout_13w": val["metrics"],
        "by_channel": val["by_channel"].reset_index().to_dict(orient="records"),
        "by_bucket": val["by_bucket"].reset_index().to_dict(orient="records"),
        "ltv_by_channel": tab.reset_index().to_dict(orient="records"),
        "horizon_weeks": HORIZON_LTV_WEEKS, "annual_discount": economics.ANNUAL_DISCOUNT,
    }
    art = {"rfm": rfm, "models": models, "by_bucket": val["by_bucket"], "ltv_tab": tab}
    return out, art


def run_retention(world) -> tuple[dict, dict]:
    t_cal = float(world.n_weeks)
    d = retention.lapse_survival_data(world.players, world.transactions, t_cal, gap_weeks=8.0)
    km, rows = {}, []
    for ch, g in d.groupby("channel"):
        k = retention.kaplan_meier(g["duration"], g["event"])
        km[ch] = k
        for t in (8, 16, 26):
            i = np.searchsorted(k["time"].to_numpy(), t, side="right") - 1
            rows.append({"channel": ch, "week": t, "survival": retention.survival_at(k, t),
                         "ci_low": float(k["ci_low"].iloc[i]) if i >= 0 else 1.0,
                         "ci_high": float(k["ci_high"].iloc[i]) if i >= 0 else 1.0})
    tab = pd.DataFrame(rows)
    cox = retention.cox_hazard_ratios(d)
    lr = retention.logrank_by_channel(d)
    save_table(tab, "retention_km_checkpoints")
    save_table(cox, "retention_cox_hazard_ratios")
    return ({"gap_weeks": 8, "n": int(len(d)), "event_share": float(d["event"].mean()),
             "km_checkpoints": tab, "logrank": lr, "cox": cox}, {"km": km})


def run_experiment(world, models, fast: bool) -> tuple[dict, dict]:
    t_cal = float(world.n_weeks)
    eco = economics.unit_economics()
    margin, cost = eco["margin_rate"], eco["bonus_pack_cost"]
    frame = ex.build_experiment_frame(world, models, t_cal, window=PROMO_WINDOW_WEEKS)
    n_exp, train_share = 60_000, 0.6
    fractions = np.round(np.arange(0.01, 0.501, 0.01), 2)

    # ---------------- the ATE analysis on draw 0 ------------------------------
    exp0 = frame.sample(n_exp, random_state=100).reset_index(drop=True)
    w0 = ex.randomize(n_exp, np.random.default_rng(200))
    exp0["y"] = exp0["y0"] + w0 * exp0["incr"]
    sate = float(exp0["incr"].mean())
    naive = ex.estimate_ate(exp0["y"], w0)
    ya, theta = ex.cuped_adjust(exp0["y"], exp0["pre_rev4"])
    cuped = ex.estimate_ate(ya, w0)
    smd = ex.standardized_mean_differences(exp0, w0, ex.FEATURES)
    n_arm = n_exp // 2
    mde_naive = ex.mde(float(exp0["y"].std()), n_arm)
    mde_cuped = ex.mde(float(ya.std()), n_arm)
    n_rep = 300 if fast else 1000
    rer = ex.rerandomization_study(exp0["y0"], exp0["incr"], exp0["pre_rev4"], n_rep, np.random.default_rng(300))
    scale = mde_naive / sate
    pwr = ex.rerandomization_study(exp0["y0"], exp0["incr"], exp0["pre_rev4"], n_rep,
                                   np.random.default_rng(301), effect_scale=scale)
    peek = ex.peeking_study(exp0["y0"], n_looks=5, n_rep=600 if fast else 3000, seed=11)
    ate_out = {
        "n": n_exp, "true_population_ate": float(frame["true_tau"].mean()), "true_sample_effect": sate,
        "mean_y0": float(exp0["y0"].mean()), "sd_y": float(exp0["y"].std()),
        "corr_y0_pre_period": float(np.corrcoef(exp0["y0"], exp0["pre_rev4"])[0, 1]),
        "srm_p_value": ex.srm_pvalue(int(w0.sum()), int((1 - w0).sum())),
        "max_abs_smd": float(smd.abs().max()),
        "naive": naive, "cuped": cuped, "cuped_theta": theta,
        "se_reduction_cuped": 1 - cuped["se"] / naive["se"],
        "mde_naive": mde_naive, "mde_cuped": mde_cuped,
        "required_n_per_arm_for_half_the_effect": ex.required_n_per_arm(float(exp0["y"].std()), sate / 2),
        "rerandomization": rer,
        "power_check": {"effect_scaled_to_naive_mde": float(scale * sate), "analytic_power": 0.80,
                        "empirical_power_naive": pwr["naive"]["rejection_rate"],
                        "empirical_power_cuped": pwr["cuped"]["rejection_rate"], "n_rep": n_rep},
        "peeking": peek,
        "share_profitable_to_treat": float(((frame["true_tau"] * margin - cost) > 0).mean()),
        "promo": {"margin_rate": margin, "bonus_pack_cost": cost, "window_weeks": PROMO_WINDOW_WEEKS},
    }

    # ---------------- uplift across repeated experiment draws ------------------
    n_draws = 2 if fast else 5
    learners = {"t_learner": ex.fit_t_learner, "transformed_outcome": ex.fit_transformed_outcome,
                "dr_learner": ex.fit_dr_learner}
    records, curves, decile0 = [], {}, None
    for d in range(n_draws):
        e = frame.sample(n_exp, random_state=100 + d).reset_index(drop=True)
        w = ex.randomize(n_exp, np.random.default_rng(200 + d))
        e["y"] = e["y0"] + w * e["incr"]
        ntr = int(n_exp * train_share)
        tr, te, wtr, wte = e.iloc[:ntr], e.iloc[ntr:].copy(), w[:ntr], w[ntr:]
        net = te["true_tau"].to_numpy() * margin - cost
        scores = {k: fit(tr[ex.FEATURES], tr["y"], wtr, seed=d)(te[ex.FEATURES]) for k, fit in learners.items()}
        scores["oracle"] = te["true_tau"].to_numpy()
        scores["churn_risk_heuristic"] = (1 - te["p_alive"]).to_numpy()
        scores["high_ltv_heuristic"] = (te["exp_purch13"] * te["m"]).to_numpy()
        for k, s in scores.items():
            thr = (s * margin > cost) if k in (*learners, "oracle") else None
            records.append({
                "draw": d, "policy": k,
                "spearman_vs_true": float(stats.spearmanr(s, te["true_tau"])[0]),
                "qini_observed": ex.qini_coefficient(te["y"], wte, s),
                "net_at_10pct": float(ex.targeting_value_curve(net, s, [0.10])[0]),
                "threshold_share_treated": float(thr.mean()) if thr is not None else np.nan,
                "threshold_net_per_1000": float(net[thr].sum() / len(net) * 1000) if thr is not None else np.nan,
            })
            curves.setdefault(k, []).append(ex.targeting_value_curve(net, s, fractions))
        records.append({"draw": d, "policy": "treat_all", "net_at_10pct": np.nan,
                        "threshold_net_per_1000": float(net.mean() * 1000), "threshold_share_treated": 1.0})
        if d == 0:
            decile0 = ex.uplift_decile_table(te, wte, scores["dr_learner"])
    rec = pd.DataFrame(records)
    summ = (rec.groupby("policy")[["spearman_vs_true", "qini_observed", "net_at_10pct",
                                   "threshold_share_treated", "threshold_net_per_1000"]]
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    curve_mean = {k: np.mean(v, axis=0) for k, v in curves.items()}
    save_table(summ.reset_index(), "uplift_policy_summary")
    save_table(decile0, "uplift_deciles_draw0")
    save_table(pd.DataFrame({"fraction": fractions, **{k: v for k, v in curve_mean.items()}}), "uplift_value_curves")
    out = {**ate_out, "uplift": {"n_draws": n_draws, "train_share": train_share,
                                 "summary": summ.reset_index(), "decile_table_draw0": decile0}}
    return out, {"curves": curve_mean, "fractions": fractions, "peek": peek, "decile": decile0}


def run_acquisition(world, models, fast: bool) -> tuple[dict, dict]:
    paid = list(simulate.PAID)
    eco = economics.unit_economics()
    m = eco["margin_rate"]
    V_est = {c: m * ltv.expected_value_new(*models[c], HORIZON_LTV_WEEKS, economics.ANNUAL_DISCOUNT)["pv_revenue"]
             for c in paid}
    V_true = {}
    for c in paid:
        t = simulate.TRUTH[c]
        V_true[c] = m * ltv.expected_value_new(ltv.BGNBD(t.r, t.alpha, t.a, t.b),
                                               ltv.GammaGamma(t.gg_p, t.gg_q, t.gg_gamma),
                                               HORIZON_LTV_WEEKS, economics.ANNUAL_DISCOUNT)["pv_revenue"]
    k_true = {c: simulate.TRUTH[c].k for c in paid}
    g_true = {c: simulate.TRUTH[c].gamma for c in paid}
    base = {c: simulate.TRUTH[c].base_spend for c in paid}
    budget = float(sum(base.values()))
    lo = {c: 0.5 * base[c] for c in paid}
    hi = {c: 2.0 * base[c] for c in paid}

    fit = acq.fit_response_curves(world.weekly)
    cover = {c: bool(fit.loc[c, "gamma_ci_low"] <= g_true[c] <= fit.loc[c, "gamma_ci_high"]) for c in paid}
    k_est, g_est = fit["k"].to_dict(), fit["gamma"].to_dict()

    def policies(ke, ge):
        n_cur = {c: acq.signups(base[c], ke[c], ge[c]) for c in paid}
        v_blend = sum(n_cur[c] * V_est[c] for c in paid) / sum(n_cur.values())
        return {
            "current_allocation": dict(base),
            "equal_split": {c: budget / len(paid) for c in paid},
            "maximize_signups": acq.allocate({c: 1.0 for c in paid}, ke, ge, budget, lo, hi, force_budget=True),
            "blended_ltv_rule": acq.allocate({c: v_blend for c in paid}, ke, ge, budget, lo, hi),
            "channel_ltv_aware_same_budget": acq.allocate(V_est, ke, ge, budget, lo, hi, force_budget=True),
            "channel_ltv_aware": acq.allocate(V_est, ke, ge, budget, lo, hi),
        }

    opt = acq.allocate(V_true, k_true, g_true, budget, lo, hi)
    opt_profit = acq.weekly_profit(opt, V_true, k_true, g_true)
    main_pol = policies(k_est, g_est)
    main_eval = {n: {"spend": s, "total_spend": float(sum(s.values())),
                     "true_profit_per_week": acq.weekly_profit(s, V_true, k_true, g_true),
                     "true_signups_per_week": acq.weekly_signups(s, k_true, g_true)} for n, s in main_pol.items()}
    main_eval["true_optimum"] = {"spend": opt, "total_spend": float(sum(opt.values())),
                                 "true_profit_per_week": opt_profit,
                                 "true_signups_per_week": acq.weekly_signups(opt, k_true, g_true)}

    # decision quality under estimation noise: repeat the whole fit -> allocate -> evaluate loop
    n_panels = 80 if fast else 300
    rows, cover_rows = [], []
    for i in range(n_panels):
        wk = simulate.simulate_weekly_acquisition(np.random.default_rng(5000 + i), world.n_weeks)
        f = acq.fit_response_curves(wk)
        pol = policies(f["k"].to_dict(), f["gamma"].to_dict())
        rows.append({n: acq.weekly_profit(s, V_true, k_true, g_true) for n, s in pol.items()})
        cover_rows.append({c: float(f.loc[c, "gamma_ci_low"] <= g_true[c] <= f.loc[c, "gamma_ci_high"]) for c in paid}
                          | {f"{c}_est": float(f.loc[c, "gamma"]) for c in paid})
    cov = pd.DataFrame(cover_rows)
    panels = pd.DataFrame(rows)
    panels["true_optimum"] = opt_profit
    summ = pd.DataFrame({"mean": panels.mean(), "p05": panels.quantile(0.05), "p95": panels.quantile(0.95)})
    save_table(summ.reset_index().rename(columns={"index": "policy"}), "acquisition_policy_profit_per_week")
    marg_true = acq.marginal_table(base, V_true, k_true, g_true)
    marg_est = acq.marginal_table(base, V_est, k_est, g_est)
    save_table(marg_true.reset_index(), "acquisition_marginal_economics_true")
    out = {
        "margin_rate": m, "budget_per_week": budget, "spend_bounds": {"low_x_base": 0.5, "high_x_base": 2.0},
        "response_curves": fit.reset_index().to_dict(orient="records"),
        "gamma_true": g_true, "gamma_ci_covers_truth_main_panel": cover,
        "gamma_ci_coverage_across_panels": {c: float(cov[c].mean()) for c in paid},
        "gamma_estimate_mean_across_panels": {c: float(cov[f"{c}_est"].mean()) for c in paid},
        "gamma_estimate_sd_across_panels": {c: float(cov[f"{c}_est"].std()) for c in paid},
        "ltv_margin_est": V_est, "ltv_margin_true": V_true,
        "marginal_economics_true": marg_true.reset_index().to_dict(orient="records"),
        "marginal_economics_est": marg_est.reset_index().to_dict(orient="records"),
        "main_panel": main_eval,
        "panels": {"n": n_panels, "profit_per_week": summ.reset_index().rename(columns={"index": "policy"}).to_dict(orient="records"),
                   "share_same_budget_beats_equal": float((panels["channel_ltv_aware_same_budget"] > panels["equal_split"]).mean()),
                   "share_same_budget_beats_maximize_signups": float((panels["channel_ltv_aware_same_budget"] > panels["maximize_signups"]).mean()),
                   "share_ltv_aware_beats_maximize_signups": float((panels["channel_ltv_aware"] > panels["maximize_signups"]).mean()),
                   "share_ltv_aware_beats_blended": float((panels["channel_ltv_aware"] > panels["blended_ltv_rule"]).mean()),
                   "share_ltv_aware_beats_current": float((panels["channel_ltv_aware"] > panels["current_allocation"]).mean()),
                   "share_ltv_aware_beats_equal": float((panels["channel_ltv_aware"] > panels["equal_split"]).mean()),
                   "mean_regret_vs_optimum": float((panels["true_optimum"] - panels["channel_ltv_aware"]).mean())},
        "optimum_at_bound": {c: bool(np.isclose(opt[c], lo[c]) or np.isclose(opt[c], hi[c])) for c in paid},
    }
    return out, {"summ": summ, "marg_true": marg_true, "main": main_eval}


def run_matchmaking(fast: bool) -> tuple[dict, dict]:
    seeds = [0, 1] if fast else [0, 1, 2]
    n, rounds = 1000, 60

    def avg(f):
        return float(np.mean([f(s) for s in seeds]))

    ksweep = []
    for k in (8, 16, 32, 64):
        res = [mm.run_rating_simulation(n, rounds, "random", "elo", k, seed=s) for s in seeds]
        ksweep.append({"k": k, "logloss_last20": float(np.mean([r["logloss"][-20:].mean() for r in res])),
                       "excess_logloss": float(np.mean([r["logloss"][-20:].mean() - r["bayes_ll"][-20:].mean() for r in res])),
                       "spearman_round10": float(np.mean([r["spearman"][9] for r in res])),
                       "spearman_final": float(np.mean([r["spearman"][-1] for r in res]))})
    checkpoints = [5, 10, 20, 40, 60]
    curves = {"elo": [], "glicko": [], "bt": []}
    ll = {"elo": [], "glicko": []}
    for s in seeds:
        e = mm.run_rating_simulation(n, rounds, "random", "elo", 32, seed=s)
        g = mm.run_rating_simulation(n, rounds, "random", "glicko", seed=s)
        a, b, w = e["matches"]
        curves["elo"].append(e["spearman"]); curves["glicko"].append(g["spearman"])
        bt = [stats.spearmanr(mm.bradley_terry(a[:r * n // 2], b[:r * n // 2], w[:r * n // 2], n), e["theta"])[0]
              for r in range(1, rounds + 1) if r in checkpoints or r % 5 == 0]
        curves["bt"].append(bt)
        ll["elo"].append(e["logloss"][-20:].mean() - e["bayes_ll"][-20:].mean())
        ll["glicko"].append(g["logloss"][-20:].mean() - g["bayes_ll"][-20:].mean())
    bt_rounds = [r for r in range(1, rounds + 1) if r in checkpoints or r % 5 == 0]
    mean_curve = {k: np.mean(v, axis=0) for k, v in curves.items()}
    comp = pd.DataFrame({"round": checkpoints,
                         "elo_k32": [mean_curve["elo"][r - 1] for r in checkpoints],
                         "glicko1": [mean_curve["glicko"][r - 1] for r in checkpoints],
                         "bradley_terry_mle": [mean_curve["bt"][bt_rounds.index(r)] for r in checkpoints]})
    pol_rows = []
    for pol in ("random", "rating", "rating_explore", "oracle"):
        res = [mm.run_rating_simulation(n, rounds, pol, "elo", 32, seed=s) for s in seeds]
        pol_rows.append({"policy": pol,
                         "fair_match_share": float(np.mean([r["fair_share"][20:].mean() for r in res])),
                         "spearman_round20": float(np.mean([r["spearman"][19] for r in res])),
                         "spearman_final": float(np.mean([r["spearman"][-1] for r in res])),
                         "excess_logloss": float(np.mean([r["logloss"][-20:].mean() - r["bayes_ll"][-20:].mean() for r in res]))})
    pol_tab = pd.DataFrame(pol_rows)
    save_table(pd.DataFrame(ksweep), "matchmaking_k_sweep")
    save_table(comp, "matchmaking_method_comparison")
    save_table(pol_tab, "matchmaking_pairing_policies")
    out = {"n_players": n, "rounds": rounds, "seeds": seeds, "k_sweep": ksweep,
           "method_comparison": comp, "pairing_policies": pol_tab,
           "excess_logloss_last20": {"elo_k32": float(np.mean(ll["elo"])), "glicko1": float(np.mean(ll["glicko"]))}}
    art = {"comp": comp, "pol": pol_tab, "curves": mean_curve, "bt_rounds": bt_rounds}
    return out, art


# --------------------------------------------------------------------------- #
def main(fast: bool = False, make_figures: bool = True, out_dir: Path | None = None):
    """Run everything. `out_dir` redirects results/ and figures/ (the smoke test uses a temp dir)."""
    global RESULTS, TABLES, FIGURES
    if out_dir is not None:
        RESULTS, TABLES, FIGURES = Path(out_dir) / "results", Path(out_dir) / "results" / "tables", Path(out_dir) / "figures"
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict = {"fast_mode": fast}
    step = lambda name: print(f"[{time.time() - t0:6.1f}s] {name}", flush=True)

    step("pack economics")
    summary["packs"], art_packs = run_packs(fast)
    step("simulating players")
    world, summary["world"] = run_world(fast)
    step("SQL layer")
    summary["sql"] = run_sql(world)
    step("LTV (BG/NBD + Gamma-Gamma)")
    summary["ltv"], art_ltv = run_ltv(world)
    step("retention")
    summary["retention"], art_ret = run_retention(world)
    step("experiment + uplift")
    summary["experiment"], art_exp = run_experiment(world, art_ltv["models"], fast)
    step("acquisition")
    summary["acquisition"], art_acq = run_acquisition(world, art_ltv["models"], fast)
    step("matchmaking")
    summary["matchmaking"], art_mm = run_matchmaking(fast)

    (RESULTS / "summary.json").write_text(json.dumps(py(summary), indent=2))
    step("wrote results/summary.json")
    if make_figures:
        from geconomy import plots
        FIGURES.mkdir(exist_ok=True)
        plots.make_all(FIGURES, summary, art_packs, art_ltv, art_ret, art_exp, art_acq, art_mm)
        step("wrote figures")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer replicates (smoke test)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    main(fast=args.fast, make_figures=not args.no_figures)
