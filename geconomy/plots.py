"""Figures for the README. Every chart is built from the pipeline's own outputs, so the
numbers in the pictures are the numbers in results/summary.json.

Design rules (dataviz method): one validated categorical order (blue, orange, aqua, yellow),
paid channels always blue / orange / aqua and organic yellow; hairline grid; thin marks; text in
ink tokens, never in series colours; a legend whenever two or more series share a plot.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CHAN = {"paid_social": BLUE, "influencer": ORANGE, "referral": AQUA, "organic": YELLOW}
CHAN_NAME = {"paid_social": "Paid social", "influencer": "Influencer", "referral": "Referral", "organic": "Organic"}
FOOTER = "All data simulated (seed 7). See README for what is and is not being claimed."
LW = 1.7
usd = FuncFormatter(lambda v, _: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "legend.frameon": False,
        "legend.fontsize": 9, "xtick.major.size": 3, "ytick.major.size": 0,
        "lines.solid_capstyle": "round", "text.parse_math": False,
    })


def _finish(fig, path: Path, title: str, subtitle: str | None = None):
    """Reserve a header band (title + wrapped subtitle) above the constrained-layout axes."""
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, int(w * 11.5)) if subtitle else []
    header_in = 0.42 + 0.19 * len(lines)
    fig.get_layout_engine().set(rect=(0, 0.34 / h, 1, 1 - (0.34 + header_in) / h))
    fig.text(0.008, 1 - 0.06 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        fig.text(0.008, 1 - (0.34 + 0.19 * i) / h, ln, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.008, 0.012, FOOTER, ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _bar_gap(bars):
    for b in bars:
        b.set_edgecolor(SURFACE)
        b.set_linewidth(1.5)


# --------------------------------------------------------------------------- #
def fig_packs(path, S, art):
    p, pay = S["packs"], S["packs"]["payout"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    v, c = art["value_cdf"]
    o = np.argsort(v)
    v, c = np.asarray(v)[o], np.asarray(c)[o]
    a.step(v, c, where="post", color=BLUE, lw=LW)
    a.set_xscale("log")
    a.set_xlim(1.4, 400)
    a.set_xticks([2, 3, 6, 10, 25, 100, 300])
    a.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    a.axvline(art["price"], color=INK2, lw=1, ls=(0, (4, 3)))
    a.text(art["price"] * 1.06, 0.06, f"pack price ${art['price']:.2f}", color=INK2, fontsize=9)
    a.annotate(f"only {pay['p_value_at_least_price']:.1%} of packs\nare worth the price or more",
               xy=(art["price"], 1 - pay["p_value_at_least_price"]), xytext=(9, 0.45), color=INK2, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.yaxis.set_major_formatter(PercentFormatter(1.0))
    a.set_ylim(0, 1.02)
    a.set_xlabel("Value of a pack's contents (log scale)")
    a.set_ylabel("Share of packs worth at most this")
    a.set_title(f"Median pack ${pay['median']:.2f}, mean ${pay['mean_value']:.2f} (RTP {pay['rtp']:.0%})")

    n = np.arange(len(art["cdf_flat"]))
    b.plot(n, art["cdf_flat"], color=BLUE, lw=LW, label="No pity timer")
    b.plot(n, art["cdf_pity"], color=ORANGE, lw=LW, label=f"Pity timer at {art['n_pity']} packs")
    fl, pt = p["legendary_flat"], p["pity"]
    for stat, col in ((fl["p90"], BLUE), (pt["p90"], ORANGE)):
        b.plot([stat], [0.9], marker="o", ms=7, color=col, mec=SURFACE, mew=1.5, zorder=3)
    b.annotate(f"p90 = {fl['p90']} packs", xy=(fl["p90"], 0.9), xytext=(104, 0.80), color=INK2, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    b.annotate(f"p90 = {pt['p90']} packs", xy=(pt["p90"], 0.9), xytext=(22, 0.955), color=INK2, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    b.yaxis.set_major_formatter(PercentFormatter(1.0))
    b.set_ylim(0, 1.02)
    b.set_xlim(0, 200)
    b.set_xlabel("Packs opened")
    b.set_ylabel("Share of players with a legendary by now")
    b.set_title(f"Pity timer keeps the mean wait ({fl['mean']:.1f} packs), trims the tail")
    b.legend(loc="lower right")
    _finish(fig, path, "Pack economics: payout distribution and legendary wait time",
            f"Exact distributions, checked against {p['monte_carlo']['n']:,} simulated packs. "
            f"Legendary odds {p['p_legendary_per_pack']:.2%} per pack; rarity odds calibrated to RTP {p['target_rtp']:.0%}.")


def fig_ltv(path, S, art):
    ltv = S["ltv"]
    bb = art["by_bucket"].reset_index() if "bucket" not in art["by_bucket"].columns else art["by_bucket"]
    tab = art["ltv_tab"].loc[["paid_social", "influencer", "referral"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    x = np.arange(len(bb))
    w = 0.36
    r1 = a.bar(x - w / 2, bb["predicted"], w, color=BLUE, label="Model prediction")
    r2 = a.bar(x + w / 2, bb["actual"], w, color=MUTED, label="Actual (held-out 13 weeks)")
    _bar_gap(list(r1) + list(r2))
    a.set_xticks(x, [f"{b_}" for b_ in bb["bucket"]])
    a.set_xlabel("Repeat purchases in the calibration period")
    a.set_ylabel("Purchases per player, next 13 weeks")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper left")
    h = ltv["holdout_13w"]
    a.set_title(f"Holdout: total purchases off by {h['total_purchase_error']:+.1%}")

    x = np.arange(len(tab))
    est, tru = tab["ltv_to_cac_est"].to_numpy(), tab["ltv_to_cac_true"].to_numpy()
    bars = b.bar(x, est, 0.5, color=[CHAN[c] for c in tab.index])
    _bar_gap(bars)
    b.scatter(x, tru, marker="D", s=46, color=INK, zorder=4, edgecolor=SURFACE, linewidth=1.2)
    for xi, e in zip(x, est):
        b.text(xi, e + 0.07, f"{e:.2f}", ha="center", va="bottom", color=INK2, fontsize=9)
    b.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    b.text(2.88, 1.03, "break-even", color=INK2, fontsize=9, ha="right", va="bottom")
    b.set_xticks(x, [CHAN_NAME[c] for c in tab.index])
    b.set_ylabel(f"{ltv['horizon_weeks']}-week margin LTV ÷ average CAC")
    b.set_xlim(-0.5, 2.9)
    b.set_ylim(0, 2.8)
    b.grid(axis="x", visible=False)
    b.legend(handles=[Line2D([], [], marker="s", ls="", color=MUTED, ms=9, label="Estimated from data"),
                      Line2D([], [], marker="D", ls="", color=INK, ms=7, label="True value in the simulator")],
             loc="upper left")
    b.set_title("LTV-to-CAC by paid channel")
    _finish(fig, path, "LTV model (BG/NBD + Gamma-Gamma) checked against a known truth",
            f"{ltv['n_players']:,} simulated players. Holdout RMSE {h['rmse_model']:.2f} vs {h['rmse_channel_mean']:.2f} "
            f"for a channel-average baseline; channel LTVs within {max(abs(r['ltv_error']) for r in ltv['ltv_by_channel']):.1%} of truth.")


def fig_retention(path, S, art):
    r = S["retention"]
    fig, a = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    order = ["referral", "organic", "paid_social", "influencer"]
    for ch in ("organic", "referral", "influencer", "paid_social"):       # organic first: wide halo under paid social
        k = art["km"][ch]
        t = np.r_[8.0, k["time"].to_numpy()]
        sv = np.r_[k["survival"].iloc[0], k["survival"].to_numpy()]
        a.step(t, sv, where="post", color=CHAN[ch], lw=4.2 if ch == "organic" else LW, zorder=2 if ch == "organic" else 3)
    a.set_xlim(8, 40)
    a.set_ylim(0, 0.65)
    a.yaxis.set_major_formatter(PercentFormatter(1.0))
    a.set_xlabel("Weeks since signup")
    a.set_ylabel("Share of players not yet lapsed")
    a.text(17.2, 0.225, "Referral: slowest to lapse", color=INK2, fontsize=9)
    a.text(8.4, 0.03, "Influencer: fastest to lapse", color=INK2, fontsize=9)
    a.annotate("Organic and paid social\noverlap almost exactly", xy=(22.5, 0.065), xytext=(26, 0.17), color=INK2, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.legend(handles=[Line2D([], [], color=CHAN[c], lw=2, label=CHAN_NAME[c]) for c in order], loc="upper right")
    cox = {row["term"]: row for row in r["cox"].to_dict("records")}
    _finish(fig, path, "Retention by acquisition channel (Kaplan-Meier)",
            f"Lapse = first gap of 8+ weeks without a purchase; players still active are censored. "
            f"Cox hazard ratio vs organic: influencer {cox['channel_influencer']['hazard_ratio']:.2f}, "
            f"referral {cox['channel_referral']['hazard_ratio']:.2f}, paid social {cox['channel_paid_social']['hazard_ratio']:.2f}.")


def fig_experiment(path, S, art):
    e = S["experiment"]
    p = e["promo"]
    breakeven = p["bonus_pack_cost"] / p["margin_rate"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    rows = [("Difference in means", e["naive"], BLUE), ("CUPED-adjusted", e["cuped"], BLUE)]
    for i, (lab, r, col) in enumerate(rows[::-1]):
        a.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=2.2, solid_capstyle="round")
        a.plot([r["estimate"]], [i], marker="o", ms=8, color=col, mec=SURFACE, mew=1.5, zorder=3)
    a.axvline(e["true_sample_effect"], color=INK, lw=1.2)
    a.text(e["true_sample_effect"] + 0.05, 1.45, f"true lift ${e['true_sample_effect']:.2f}", color=INK, fontsize=9)
    a.axvline(breakeven, color=INK2, lw=1, ls=(0, (4, 3)))
    a.text(breakeven - 0.05, 1.45, f"lift needed to pay\nfor the bonus pack: ${breakeven:.2f}", color=INK2, fontsize=9, ha="right")
    a.set_yticks([0, 1], [rows[1][0], rows[0][0]])
    a.set_ylim(-0.6, 1.95)
    a.set_xlim(0.4, 3.0)
    a.grid(axis="y", visible=False)
    a.set_xlabel("Extra revenue per player over 4 weeks ($), 95% CI")
    a.set_title(f"Average lift is real but below break-even (n = {e['n']:,})")

    pk = e["peeking"]
    labels = ["One look at\nthe end", "Look 5 times,\nstop at p < 0.05", "Pocock\nboundary", "O'Brien-Fleming\nboundary"]
    vals = [pk["fpr_fixed_horizon"], pk["fpr_peeking_naive"], pk["fpr_pocock"], pk["fpr_obrien_fleming"]]
    bars = b.bar(range(4), vals, 0.55, color=BLUE)
    _bar_gap(bars)
    for xi, vv in enumerate(vals):
        b.text(xi, max(vv, 0.05) + 0.004, f"{vv:.1%}", ha="center", va="bottom", color=INK2, fontsize=9)
    b.axhline(0.05, color=INK2, lw=1, ls=(0, (4, 3)))
    b.set_xlim(-0.5, 4.0)
    b.text(3.98, 0.0515, "nominal 5%", color=INK2, fontsize=9, ha="right", va="bottom")
    b.set_xticks(range(4), labels, fontsize=9)
    b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    b.set_ylim(0, 0.17)
    b.grid(axis="x", visible=False)
    b.set_ylabel("False-positive rate (no true effect)")
    b.set_title(f"Peeking inflates false positives ({pk['n_rep']:,} simulated A/A tests)")
    _finish(fig, path, "Experiment analysis on a semi-synthetic A/B test with a known effect",
            f"Baseline revenue comes from the simulated players; the treatment effect is injected, so the target is known. "
            f"CUPED shrinks the standard error by only {e['se_reduction_cuped']:.1%} (pre-period correlation {e['corr_y0_pre_period']:.2f}).")


def fig_uplift(path, S, art):
    e = S["experiment"]["uplift"]
    summ = e["summary"].set_index("policy")
    fr = art["fractions"] * 100
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    a.axhline(0, color=INK2, lw=1)
    spec = [("oracle", "Oracle (true effect)", INK2, (0, (4, 3))), ("dr_learner", "Doubly-robust learner", BLUE, "-"),
            ("high_ltv_heuristic", "Target high-LTV players", ORANGE, "-"), ("churn_risk_heuristic", "Target likely churners", AQUA, "-")]
    for key, lab, col, ls in spec:
        a.plot(fr, art["curves"][key], color=col, lw=LW, ls=ls, label=lab)
    a.set_xlabel("Share of players given the bonus pack, ranked by score (%)")
    a.set_ylabel("Net value per 1,000 players")
    a.yaxis.set_major_formatter(usd)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    a.set_title("Who gets the bonus pack decides whether it pays")

    names = [("t_learner", "T-learner"), ("transformed_outcome", "Transformed\noutcome"), ("dr_learner", "Doubly\nrobust"), ("oracle", "Oracle")]
    m = [summ.loc[k, "threshold_net_per_1000_mean"] for k, _ in names]
    sd = [summ.loc[k, "threshold_net_per_1000_std"] for k, _ in names]
    cols = [BLUE, BLUE, BLUE, MUTED]
    bars = b.bar(range(4), m, 0.55, color=cols, yerr=sd, ecolor=INK2, error_kw=dict(lw=1.2, capsize=4))
    _bar_gap(bars)
    b.set_xticks(range(4), [n for _, n in names], fontsize=9)
    b.yaxis.set_major_formatter(usd)
    b.grid(axis="x", visible=False)
    b.set_ylabel("Net value per 1,000 players")
    treat_all = summ.loc["treat_all", "threshold_net_per_1000_mean"]
    b.text(0.02, 0.97, f"Treating everyone: {usd(treat_all)} per 1,000", transform=b.transAxes, va="top", color=INK2, fontsize=9)
    b.set_ylim(0, max(np.array(m) + np.array(sd)) * 1.18)
    b.set_title(f"Treat when predicted lift beats cost (mean ± sd, {e['n_draws']} experiments)")
    _finish(fig, path, "Uplift modelling: choosing whom to treat, scored against the true effect",
            f"Each learner trains on {int(S['experiment']['n'] * e['train_share']):,} randomised players and is scored on "
            f"{S['experiment']['n'] - int(S['experiment']['n'] * e['train_share']):,} held-out players. "
            f"Only {S['experiment']['share_profitable_to_treat']:.1%} of players are worth treating at the assumed bonus cost.")


def fig_acquisition(path, S, art):
    a_s = S["acquisition"]
    summ = art["summ"]
    order = ["current_allocation", "equal_split", "maximize_signups", "blended_ltv_rule",
             "channel_ltv_aware_same_budget", "channel_ltv_aware"]
    names = {"current_allocation": "Current split", "equal_split": "Equal split", "maximize_signups": "Maximise signups",
             "blended_ltv_rule": "Blended-LTV rule", "channel_ltv_aware_same_budget": "Channel-LTV, same $35k",
             "channel_ltv_aware": "Channel-LTV, budget can shrink"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(order))[::-1]
    cols = [MUTED] * 4 + [BLUE, BLUE]
    bars = a.barh(y, [summ.loc[k, "mean"] for k in order], 0.55, color=cols)
    _bar_gap(bars)
    for yi, k in zip(y, order):
        lo, hi, mu = summ.loc[k, "p05"], summ.loc[k, "p95"], summ.loc[k, "mean"]
        if hi - lo > 1:
            a.plot([lo, hi], [yi, yi], color=INK, lw=1.2, marker="|", ms=7)
        a.text(14900, yi, f"${mu:,.0f}", va="center", color=INK2, fontsize=9)
    opt = summ.loc["true_optimum", "mean"]
    a.axvline(opt, color=INK2, lw=1, ls=(0, (4, 3)))
    a.text(opt - 120, y[0] + 0.62, "best possible", color=INK2, fontsize=9, ha="right", va="bottom")
    a.text(14900, y[0] + 0.62, "mean", color=INK2, fontsize=9, ha="left", va="bottom")
    a.set_yticks(y, [names[k] for k in order])
    a.xaxis.set_major_formatter(usd)
    a.set_xlim(0, 16800)
    a.grid(axis="y", visible=False)
    a.set_xlabel("True weekly profit after CAC (mean over fresh simulated years; line = 5th-95th percentile)")
    a.set_title("Profit per week under each allocation rule")

    mt = art["marg_true"]
    chs = ["paid_social", "influencer", "referral"]
    vals = [float(mt.loc[c, "ltv_to_marginal_cac"]) for c in chs]
    bars = b.bar(range(3), vals, 0.5, color=[CHAN[c] for c in chs])
    _bar_gap(bars)
    for xi, vv in enumerate(vals):
        b.text(xi, vv + 0.04, f"{vv:.2f}", ha="center", color=INK2, fontsize=9)
    b.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    b.text(2.93, 1.03, "break-even", color=INK2, fontsize=9, ha="right", va="bottom")
    b.set_xlim(-0.5, 2.95)
    b.set_xticks(range(3), [CHAN_NAME[c] for c in chs])
    b.grid(axis="x", visible=False)
    b.set_ylim(0, 2.4)
    b.set_ylabel("Margin LTV ÷ marginal CAC at today's spend")
    b.set_title("Last dollar: only referral pays")
    _finish(fig, path, "Acquisition budget: LTV-aware allocation across channels",
            f"Response curves and LTVs are estimated from {S['world']['n_players']:,} simulated players, then each rule is scored against the "
            f"true curves. Spend limited to 0.5x-2x of today's per channel ({a_s['panels']['n']} fresh panels).")


def fig_matchmaking(path, S, art):
    m = S["matchmaking"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    a, b, c = axs
    r = np.arange(1, m["rounds"] + 1)
    a.plot(r, art["curves"]["elo"], color=BLUE, lw=LW, label="Elo (K = 32)")
    a.plot(r, art["curves"]["glicko"], color=ORANGE, lw=LW, label="Glicko-1")
    a.plot(art["bt_rounds"], art["curves"]["bt"], color=AQUA, lw=LW, label="Bradley-Terry fit")
    a.set_ylim(0.4, 1.0)
    a.set_xlabel("Matches played per player")
    a.set_ylabel("Rank correlation with true skill")
    a.legend(loc="lower right")
    comp20 = art["comp"].set_index("round").loc[20]
    a.set_title(f"Rank correlation reaches {comp20.min():.2f}+ by match 20")

    pol = art["pol"].set_index("policy")
    lab = {"random": "Random pairing", "rating_explore": "Closest rating,\n30% random", "rating": "Closest rating"}
    off = {"random": (0, 12, "center"), "rating_explore": (0, 12, "center"), "rating": (0, 12, "center")}
    for k in lab:
        b.scatter(pol.loc[k, "fair_match_share"], pol.loc[k, "spearman_final"], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        dx, dy, ha = off[k]
        b.annotate(lab[k], (pol.loc[k, "fair_match_share"], pol.loc[k, "spearman_final"]), xytext=(dx, dy),
                   textcoords="offset points", ha=ha, va="bottom", fontsize=9, color=INK2)
    b.set_xlim(0.1, 0.6)
    b.set_ylim(0.915, 0.96)
    b.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    b.set_xlabel("Matches that are close (win prob. 40-60%)")
    b.set_ylabel("Rank correlation after 60 matches")
    b.text(0.99, 0.04, f"Pairing by true skill: {pol.loc['oracle', 'fair_match_share']:.0%} close matches,\nrank correlation {pol.loc['oracle', 'spearman_final']:.2f} (off chart)",
           transform=b.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK2)
    b.set_title("Fairer matches cost some rating accuracy")

    ks = pd.DataFrame(m["k_sweep"])
    c.plot(range(len(ks)), ks["excess_logloss"], color=BLUE, lw=LW, marker="o", ms=8, mec=SURFACE, mew=1.5)
    kbest = int(ks["excess_logloss"].idxmin())
    c.annotate(f"best: K = {int(ks.loc[kbest, 'k'])}", (kbest, ks.loc[kbest, "excess_logloss"]), xytext=(0, -16),
               textcoords="offset points", ha="center", color=INK2, fontsize=9)
    c.set_xticks(range(len(ks)), [str(int(k)) for k in ks["k"]])
    c.set_xlabel("Elo K-factor")
    c.set_ylabel("Excess log loss vs perfect knowledge")
    c.set_ylim(0, ks["excess_logloss"].max() * 1.15)
    c.set_title("K-factor sweep")
    _finish(fig, path, "Matchmaking: rating systems and pairing rules on 1,000 simulated players",
            "Outcomes follow a Bradley-Terry model with known true skills, so rating error is measurable. "
            "Pairing by true skill (oracle) gives fair matches but teaches the ratings nothing.")


def make_all(figdir: Path, S, art_packs, art_ltv, art_ret, art_exp, art_acq, art_mm):
    _style()
    figdir.mkdir(parents=True, exist_ok=True)
    fig_packs(figdir / "1_pack_economics.png", S, art_packs)
    fig_ltv(figdir / "2_ltv_validation.png", S, art_ltv)
    fig_retention(figdir / "3_retention_by_channel.png", S, art_ret)
    fig_experiment(figdir / "4_experiment_analysis.png", S, art_exp)
    fig_uplift(figdir / "5_uplift_targeting.png", S, art_exp)
    fig_acquisition(figdir / "6_acquisition_allocation.png", S, art_acq)
    fig_matchmaking(figdir / "7_matchmaking.png", S, art_mm)
