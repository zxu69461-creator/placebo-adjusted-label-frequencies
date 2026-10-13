#!/usr/bin/env python3
"""Main figures of the manuscript (hierarchical estimator), read only from the project DuckDB, with one Source Data
workbook (manuscript/source_data.xlsx). The Fig. 1a illustration is drawn in code (oajoint.schematics, figkit.art).

1 question, data and estimator | 2 validated by predicting later trials | 3 how much of a label is background
4 where the background lies | 5 from label to patient, and to a better label
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import figures as F  # noqa: E402
from oajoint import paper as P  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
from oajoint import schematics as SCH  # noqa: E402
import figkit as fk  # noqa: E402

SRC = {}
BENCH_C, BENCH_N, BENCH_S = "bench22_confirm33", "generality_newdrug33", "generality_serious33"
ROWS_B = ["AF 3.3", "AF 3.2", "B7 class-Bayes", "B3 Bayes-NN", "B4 AF 2.0", "B6 class-mean", "B2 PM-t", "B1 MH",
          "B5 term-only", "B0 null"]


# ------------------------------------------------------------------------------------------- helpers
def head(ax, letter, text):
    """Panel letter + left-aligned title; figkit places the bold letter after layout (fk.save)."""
    fk.panel(ax, letter, text)


def tx(ax, *a, **k):
    t = ax.text(*a, **k); t.set_in_layout(False); return t


def load(con):
    d = {}
    d["pair"] = con.execute("""SELECT p.*, e.af_mean, e.per100_drug_exp, e.per100_added_exp, e.per100_drug_obs,
                                      e.per100_pl_obs, e.rule_a_v2, e.rule_b_v2, e.rule_c_v2
                               FROM afstd_pair p JOIN afstd_pair_exp e USING (drug, term) ORDER BY drug, term""").df()
    d["sum"] = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_summary").df()
    d["card"] = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_card ORDER BY card_set, rank_drug, rank_term").df()
    d["par"] = con.execute("SELECT kind, soc, name, value FROM af3_param WHERE fit = 'AF 3.3 full'").df()
    d["bench"] = con.execute("""SELECT bench, win, stratum, scoring, method, n_records, n_trials, mean_logscore, diff_vs_ref,
                                       diff_ci_lo, diff_ci_hi, pit_central90 FROM bench_score
                                WHERE method NOT LIKE '%no dose%' AND method NOT LIKE '%nocov%'""").df()
    d["pit"] = con.execute(f"""SELECT method, pit FROM bench_record WHERE bench = '{BENCH_C}' AND scoring = 'hier'
                               AND af1_support AND method IN ('AF 3.3', 'B1 MH') ORDER BY method, nct, drug, term""").df()
    d["wins"] = con.execute("""SELECT win, count(DISTINCT nct) n FROM bench_record WHERE method = 'AF 3.3' AND scoring = 'hier'
                               AND bench IN ('bench21_dev33', 'bench22_confirm33') GROUP BY win ORDER BY win""").df()
    d["sim"] = con.execute("SELECT * FROM af3_sim WHERE version = 'AF 3.3' ORDER BY scenario").df()
    d["n_ctgov"] = con.execute("SELECT count(*) FROM ctgov_trial").fetchone()[0]
    d["n_pc"] = con.execute("SELECT count(DISTINCT nct) FROM atlas_contrast").fetchone()[0]
    d["gate"] = {r[0]: r[1] for r in con.execute("SELECT gate_name, passed FROM prov_decision").fetchall()}
    return d


def S(d, analysis, set_name, grp, metric, col="value"):
    s = d["sum"]
    r = s[(s.analysis == analysis) & (s.set_name == set_name) & (s.grp == grp) & (s.metric == metric)]
    assert len(r) == 1, (analysis, set_name, grp, metric, len(r))
    return float(r[col].iloc[0])


def zone_legend(ax, **k):
    return ax.legend(handles=[mpl.patches.Patch(color=P.ZONE_COLOR[z], label=P.ZONE_LABEL[z]) for z in P.ZONES],
                     frameon=False, **k)


# ============================================================================== Figure 1
def fig1(d):
    pair, par = d["pair"], d["par"]
    fig = fk.figure("double", height=170)
    g0 = fig.add_gridspec(2, 1, height_ratios=[.42, 1])
    # a -- the lead image: the estimator, drawn as an illustration (figkit.art), composed 1:1 into the PDF
    ax = fig.add_subplot(g0[0]); fk.art.place(ax, SCH.fig1a_art())
    head(ax, "a", "A hierarchical estimator of randomized attributable fractions")
    # the 22-organ-system panel (d) is a tall column spanning both rows; b, c above e, f on the left
    g1 = g0[1].subgridspec(2, 3, width_ratios=[1, 1.05, .82])
    # b -- what the attributable fraction means
    ax = fig.add_subplot(g1[0, 0])
    ex = d["card"][d["card"].card_set == "fig1a"].merge(pair[["drug", "term", "af_mean", "per100_added_exp"]],
                                                        on=["drug", "term"])
    y = 0; ticks, tl, rows = [], [], []
    for r in ex.itertuples():
        pl, add = r.per100_pl, r.per100_added_exp
        ax.barh(y + .85, pl, .7, color=P.LGREY)
        tx(ax, pl + .4, y + .85, f"{pl:.1f}", va="center", fontsize=fk.pt("annot"))
        ax.barh(y, pl, .7, color=P.LGREY); ax.barh(y, max(add, 0), .7, left=pl, color=P.VERM)
        ax.errorbar(pl + add, y, xerr=[[max(add - r.per100_added_lo, 0)], [max(r.per100_added_hi - add, 0)]], fmt="none",
                    ecolor=P.BLACK, elinewidth=.5, capsize=1.2)
        tx(ax, pl + r.per100_added_hi + .4, y, f"{pl + add:.1f}", va="center", fontsize=fk.pt("annot"))
        nm = "peripheral oedema" if r.term == "edema peripheral" else r.term
        tx(ax, 0, y + 2.5, f"{r.display_name} · {nm}", fontsize=fk.pt("base"), fontweight="bold", va="center")
        tx(ax, 0, y + 1.75, f"AF {r.af_mean:.2f} ({r.af_lo:.2f} to {r.af_hi:.2f}); P(background) {r.p_af_lt_half:.2f}"
           .replace("-", "\u2212"),
           fontsize=fk.pt("annot"), color=fk.c.grey_dark, va="center")
        ticks += [y + .85, y]; tl += ["placebo", "drug"]
        rows.append(dict(drug=r.display_name, term=r.term, per100_placebo=pl, per100_added=add,
                         added_lo=r.per100_added_lo, added_hi=r.per100_added_hi, af=r.af_mean, af_lo=r.af_lo,
                         af_hi=r.af_hi, p_mostly_background=r.p_af_lt_half, n_trials=r.n_trials))
        y -= 3.9
    ax.set_yticks(ticks); ax.set_yticklabels(tl); ax.set_ylim(y + 3.3, 2.95); ax.set_xlim(0, 22)
    ax.set_xlabel("reports per 100 patients")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fk.legend_top(ax, [mpl.patches.Patch(color=P.LGREY), mpl.patches.Patch(color=P.VERM)],
                  ["on placebo too", "added by the drug"])
    head(ax, "b", "What the attributable fraction (AF) means")
    SRC["Fig 1b"] = pd.DataFrame(rows)
    # c -- from registry to label
    ax = fig.add_subplot(g1[0, 1])
    st = [("randomized trials with posted results", d["n_ctgov"], P.GREY),
          ("placebo-controlled drug trials", d["n_pc"], P.GREY),
          ("trials with arm-level counts used", S(d, "data_v2", "estimator", "all", "trials"), P.GREY),
          ("trial × drug × side-effect records", S(d, "data_v2", "estimator", "all", "records"), P.BLUE),
          ("drug–side-effect pairs estimated", S(d, "counts", "all", "all", "pairs"), P.BLUE),
          ("pairs on a current FDA label", S(d, "counts", "listed", "all", "pairs"), P.VERM)]
    yy = np.arange(len(st))[::-1]
    ax.barh(yy, [s[1] for s in st], .36, color=[s[2] for s in st])
    for yv, (t, n, _) in zip(yy, st):
        tx(ax, 1.1e3, yv + .44, t, fontsize=fk.pt("annot"), va="center")
        tx(ax, n * 1.1, yv - .02, f"{int(n):,}", fontsize=fk.pt("annot"), va="center")
    ax.set_xscale("log"); ax.set_xlim(1e3, 3e6); ax.set_ylim(-.45, len(st) + .1)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("count (log scale)")
    head(ax, "c", "From the registry to the label")
    SRC["Fig 1c"] = pd.DataFrame([dict(stage=t, count=n) for t, n, _ in st])
    # d -- evidence behind each pair
    ax = fig.add_subplot(g1[1, 0])
    sets = [("all pairs", pair), ("on a label", pair[pair.on_fda]),
            ("on a label, conv.\nestimable", pair[pair.on_fda & pair.af1_estimable])]
    shades = fk.sem("oajoint")["ramp"]["blue4"]
    rows = []
    for i, (nm, x) in enumerate(sets):
        left = 0
        for s, c in zip(P.STRATA, shades):
            f = float((x.evidence == s).mean())
            ax.barh(i, f, .62, left=left, color=c, ec="white", lw=.3)
            if f > .13:                       # the share is written inside its segment only where it fits
                tx(ax, left + f / 2, i, f"{100 * f:.0f}%", ha="center", va="center", fontsize=fk.pt("annot"),
                   color="white" if s in ("5-9", ">=10") else P.BLACK)
            left += f
            rows.append(dict(set=nm.replace("\n", " "), stratum=s, share=f, n=int((x.evidence == s).sum())))
        tx(ax, 1.02, i, f"{len(x):,}", va="center", fontsize=fk.pt("annot"))
    ax.set_yticks(range(len(sets))); ax.set_yticklabels([s[0] for s in sets]); ax.invert_yaxis()
    ax.set_xlim(0, 1.2); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100"])
    ax.set_xlabel("share of pairs (%)")
    ax.legend(handles=[mpl.patches.Patch(color=c, label=P.STRATUM_LABEL[s]) for s, c in zip(P.STRATA, shades)],
              title="trials per pair", title_fontsize=fk.pt("legend"), ncol=4, loc="lower center", bbox_to_anchor=(.45, 1.0),
              frameon=False, columnspacing=.7, handlelength=.9)
    head(ax, "e", "")
    SRC["Fig 1e"] = pd.DataFrame(rows)
    # e -- what changes the drug effect between trials
    ax = fig.add_subplot(g1[1, 1])
    bp = par[par.kind == "beta_pooled"].set_index("name").value
    bs = par[par.kind == "beta"]
    covs = sorted([c for c in P.COV_LABEL if c in bp.index], key=lambda c: bp[c])
    rows = []
    for i, c in enumerate(covs):
        v = bs[bs.name == c].value
        lo, hi = v.quantile(.1), v.quantile(.9)
        ax.plot([lo, hi], [i, i], color=P.LGREY, lw=2.2, solid_capstyle="butt")
        ax.plot(bp[c], i, "o", ms=3.2, color=P.VERM if abs(bp[c]) >= .1 else P.BLUE, mec=P.BLACK, mew=.3)
        rows.append(dict(covariate=P.COV_LABEL[c], pooled=bp[c], soc_p10=lo, soc_p90=hi))
    ax.axvline(0, color=P.BLACK, lw=.5)
    ax.set_yticks(range(len(covs))); ax.set_yticklabels([P.COV_LABEL[c] for c in covs])
    ax.set_xlabel("Δ log hazard ratio of the drug effect")
    head(ax, "f", "Trial design shifts the measured effect")
    SRC["Fig 1f"] = pd.DataFrame(rows)
    # f -- dose slope by organ system
    ax = fig.add_subplot(g1[:, 2])
    g = par[par.kind == "gamma"].copy()
    socs = sorted(pair[pair.on_fda].groupby("soc_h").size().loc[lambda s: s >= 30].index)
    g = g[g.soc.isin(socs)].sort_values("value")
    gm = float(par[(par.kind == "misc") & (par.name == "gamma_global")].value.iloc[0])
    gse = float(par[(par.kind == "misc") & (par.name == "gamma_global_se")].value.iloc[0])
    ax.axvspan(gm - 1.96 * gse, gm + 1.96 * gse, color=fk.c["vermilion@t0.8"], lw=0)
    ax.axvline(gm, color=P.VERM, lw=.6); ax.axvline(0, color=P.BLACK, lw=.5)
    ax.plot(g.value, range(len(g)), "o", ms=2.8, color=P.BLUE, mec=P.BLACK, mew=.3)
    ax.set_yticks(range(len(g))); ax.set_yticklabels([P.SOC_SHORT[s] for s in g.soc], fontsize=fk.pt("annot"))
    ax.set_xlabel("dose slope (Δ log HR per e-fold dose)")
    # one-line header (short title, legend reaching back over the organ names): the tall column grows upwards
    fk.legend_top(ax, [mpl.lines.Line2D([], [], color=P.VERM, lw=1.2)],
                  [f"pooled {gm:.3f} (SE {gse:.3f})"], ncol=1, align="right")
    head(ax, "d", "Dose–response")
    SRC["Fig 1d"] = g[["soc", "value"]].rename(columns={"value": "dose_slope"})
    return fig


# ============================================================================== Figure 2
def fig2(d):
    b, pair = d["bench"], d["pair"]
    C = b[(b.bench == BENCH_C) & (b.win == "pooled") & (b.scoring == "hier")]
    fig = fk.figure("double", height=110)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1.05, 1])
    r0 = g0[0].subgridspec(1, 4, width_ratios=[.85, 1.2, .5, .8])
    r1 = g0[1].subgridspec(1, 4, width_ratios=[1.2, 1.1, .8, .9])
    # a -- design
    ax = fig.add_subplot(r0[0])
    wins = [("W2011", 2011.0, 2013.0), ("W2013", 2013.0, 2015.0), ("W2015", 2015.0, 2017.09), ("C2017", 2017.09, 2026.7)]
    nwin = d["wins"].set_index("win").n
    for i, (w, s0, s1) in enumerate(wins):
        y = len(wins) - i
        col = P.VERM if w == "C2017" else P.BLUE
        ax.barh(y, s0 - 2000, .55, left=2000, color=P.LGREY)
        ax.barh(y, s1 - s0, .55, left=s0, color=col)
        if w == "C2017":
            tx(ax, s0 + (s1 - s0) / 2, y, f"{int(nwin[w]):,}", va="center", ha="center", fontsize=fk.pt("annot"), color="white")
        else:
            tx(ax, s1 + .3, y, f"{int(nwin[w]):,}", va="center", ha="left", fontsize=fk.pt("annot"))
    tx(ax, 2000.5, 4.45, "fit on all earlier trials", fontsize=fk.pt("annot"), va="center")
    tx(ax, 2000.5, 1.45, "scored once per estimator version", fontsize=fk.pt("annot"), va="center", color=P.VERM)
    ax.set_yticks(range(1, 5)); ax.set_yticklabels(["confirm", "tune 3", "tune 2", "tune 1"])
    ax.set_xlim(2000, 2027); ax.set_ylim(.5, 4.85); ax.set_xticks([2000, 2010, 2020]); ax.set_xlabel("trial start")
    head(ax, "a", "Predict trials that started later")
    SRC["Fig 2a"] = pd.DataFrame([dict(window=w, start=s0, end=s1, test_trials=int(nwin[w])) for w, s0, s1 in wins])
    # b -- confirmation: gain over each comparator; d shares its rows
    c = C[C.stratum == "AF1-support"].set_index("method").loc[ROWS_B]
    order = ["AF 3.3"] + list(c.drop(index="AF 3.3").sort_values("diff_vs_ref", ascending=False).index)[::-1][::-1]
    order = ["AF 3.3"] + sorted([m for m in ROWS_B if m != "AF 3.3"], key=lambda m: c.loc[m, "diff_vs_ref"])
    yy = np.arange(len(order))[::-1]
    ax = fig.add_subplot(r0[1])
    for yv, m in zip(yy, order):
        r = c.loc[m]
        if m == "AF 3.3":
            ax.plot(0, yv, "D", ms=3.2, color=P.VERM, mec=P.BLACK, mew=.3)
            tx(ax, .012, yv, "reference", va="center", fontsize=fk.pt("annot"), color=P.VERM)
            continue
        ax.errorbar(r.diff_vs_ref, yv, xerr=[[r.diff_vs_ref - r.diff_ci_lo], [r.diff_ci_hi - r.diff_vs_ref]], fmt="o",
                    ms=3, color=P.METHOD_COLOR.get(m, P.GREY), mec=P.BLACK, mew=.3, elinewidth=.8)
        tx(ax, r.diff_ci_hi * 1.25 + .006, yv, f"+{r.diff_vs_ref:.3f}", va="center", fontsize=fk.pt("annot"))
    ax.axvline(0, color=P.BLACK, lw=.5); ax.set_xscale("symlog", linthresh=.05, linscale=.5)
    ax.set_xlim(-.005, 6); ax.set_xticks([0, .05, .2, 1]); ax.set_xticklabels(["0", "0.05", "0.2", "1"])
    ax.set_yticks(yy); ax.set_yticklabels([P.METHOD_SHORT[m] for m in order]); ax.set_ylim(-.6, len(order) - .4)
    ax.set_xlabel("log-score gain per prediction (95% CI)")
    head(ax, "b", f"Better than every alternative ({int(c.loc['AF 3.3', 'n_records']):,} predictions)")
    SRC["Fig 2b"] = c.reset_index()[["method", "n_records", "n_trials", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi"]].assign(
        label=lambda x: x.method.map(P.METHOD_LABEL))
    ax = fig.add_subplot(r0[2], sharey=ax)
    ax.axvspan(.87, .93, color=P.LGREY, lw=0); ax.axvline(.9, color=P.BLACK, lw=.5, ls="--")
    ax.barh(yy, [c.loc[m, "pit_central90"] for m in order], .6, color=[P.METHOD_COLOR.get(m, P.GREY) for m in order])
    for yv, m in zip(yy, order):
        v = c.loc[m, "pit_central90"]
        tx(ax, .712 if v > .77 else v + .005, yv, f"{v:.2f}", va="center", fontsize=fk.pt("annot"),
           color="white" if v > .77 else P.BLACK)
    ax.set_xlim(.7, .95); ax.set_xticks([.7, .8, .9]); ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel("in central 90%")
    head(ax, "c", "Coverage")
    SRC["Fig 2c"] = c.reset_index()[["method", "pit_central90"]]
    # d -- PIT histograms
    ax = fig.add_subplot(r0[3])
    bins = np.linspace(0, 1, 11); rows = []
    for m, col, ls in (("AF 3.3", P.VERM, "-"), ("B1 MH", P.GREY, "--")):
        h, _ = np.histogram(d["pit"][d["pit"].method == m].pit, bins=bins, density=True)
        ax.stairs(h, bins, color=col, lw=1, ls=ls, baseline=None, label=P.METHOD_SHORT[m])
        rows += [dict(method=m, bin_lo=bins[i], density=h[i]) for i in range(10)]
    ax.axhline(1, color=P.BLACK, lw=.4, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(.5, 1.95); ax.set_xlabel("predictive percentile of the\nobserved count (PIT)")
    ax.set_ylabel("density (flat = calibrated)")
    ax.legend(loc="upper center", frameon=False)
    head(ax, "d", "Calibration")
    SRC["Fig 2d"] = pd.DataFrame(rows)
    # e -- gains by evidence
    ax = fig.add_subplot(r1[0])
    strata = ["evidence 1 trial", "evidence 2-4", "evidence 5-9", "evidence >=10", "unseen"]
    rows = []
    for j, m in enumerate(["B7 class-Bayes", "B3 Bayes-NN", "B5 term-only"]):
        t = C[C.method == m].set_index("stratum").reindex(strata)
        x = np.arange(len(strata)) + (j - 1) * .24
        ok = t.diff_vs_ref.notna().values
        ax.errorbar(x[ok], t.diff_vs_ref[ok], yerr=[(t.diff_vs_ref - t.diff_ci_lo)[ok], (t.diff_ci_hi - t.diff_vs_ref)[ok]],
                    fmt="o", ms=2.8, color=P.METHOD_COLOR[m], mec=P.BLACK, mew=.3, elinewidth=.7, label=P.METHOD_SHORT[m])
        rows += [dict(comparator=m, stratum=s, **t.loc[s, ["n_records", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi"]].to_dict())
                 for s in strata]
    ax.axhline(0, color=P.BLACK, lw=.5); ax.set_ylim(-.03, .52)
    ax.set_xticks(range(len(strata))); ax.set_xticklabels(["1", "2–4", "5–9", "≥10", "new"])
    ax.set_xlabel("earlier trials of the pair"); ax.set_ylabel("log-score gain per prediction")
    ax.legend(loc="upper left", frameon=False)
    head(ax, "e", "Gains by evidence")
    SRC["Fig 2e"] = pd.DataFrame(rows)
    # f -- generality
    ax = fig.add_subplot(r1[1])
    rows, ylab, yv = [], [], []
    y = 0
    for bench, stratum, nm in ((BENCH_S, "AF1-support", "Serious events (provisional)"),
                               (BENCH_N, "unseen", "New drugs (10% held out)")):
        t = b[(b.bench == bench) & (b.win == "pooled") & (b.scoring == "hier") & (b.stratum == stratum) & (b.method != "AF 3.3")]
        for r in t.sort_values("diff_vs_ref", ascending=False).itertuples():
            ax.errorbar(r.diff_vs_ref, y, xerr=[[r.diff_vs_ref - r.diff_ci_lo], [r.diff_ci_hi - r.diff_vs_ref]], fmt="o",
                        ms=2.8, color=P.METHOD_COLOR.get(r.method, P.GREY), mec=P.BLACK, mew=.3, elinewidth=.8)
            rows.append(dict(test=nm, method=r.method, d=r.diff_vs_ref, lo=r.diff_ci_lo, hi=r.diff_ci_hi, n=r.n_records))
            ylab.append(P.METHOD_SHORT[r.method]); yv.append(y); y += 1
        ylab.append(nm); yv.append(y); y += 1.2
    G = pd.DataFrame(rows)
    ax.axvline(0, color=P.BLACK, lw=.5); ax.set_xscale("symlog", linthresh=.1, linscale=.6)
    ax.set_xlim(-.06, 1.3); ax.set_xticks([0, .1, .5, 1]); ax.set_xticklabels(["0", "0.1", "0.5", "1"])
    ax.set_yticks(yv); ax.set_yticklabels(ylab, fontsize=fk.pt("annot"))
    for t in ax.get_yticklabels():
        if t.get_text().startswith(("Serious", "New drugs")):
            t.set_fontweight("bold")
    ax.set_ylim(-.6, y - .6)
    ax.set_xlabel("log-score gain per prediction (95% CI)")
    head(ax, "f", "New drugs and serious events")
    SRC["Fig 2f"] = G
    # g -- known-truth simulation
    ax = fig.add_subplot(r1[2])
    s = d["sim"]; x = np.arange(len(s))
    ax.bar(x - .18, s.coverage_af1, .34, color=P.GREY, label="Mantel–Haenszel")
    ax.bar(x + .18, s.coverage_af3, .34, color=P.VERM, label="This estimator")
    ax.axhline(.95, color=P.BLACK, lw=.5, ls="--"); ax.set_ylim(.6, 1.16); ax.set_yticks([.6, .7, .8, .9, 1])
    nm = {"S1": "varying", "S2": "similar", "S3": "skewed"}
    for i, r in enumerate(s.itertuples()):
        tx(ax, i - .18, .62, f"{r.rmse_af1:.2f}", rotation=90, ha="center", va="bottom", fontsize=fk.pt("annot"), color="white")
        tx(ax, i + .18, .62, f"{r.rmse_af3:.2f}", rotation=90, ha="center", va="bottom", fontsize=fk.pt("annot"), color="white")
    ax.set_xticks(x); fk.tilt(ax, [nm.get(r.scenario, r.scenario) for r in s.itertuples()])
    ax.set_xlabel("simulated effects (in bars: RMSE)")
    ax.set_ylabel("coverage of the true AF (95% interval)")
    ax.legend(loc="upper left", frameon=False)          # stacked inside the frame, as in panel h
    head(ax, "g", "Known truth")
    SRC["Fig 2g"] = s[["scenario", "coverage_af1", "coverage_af3", "rmse_af1", "rmse_af3"]]
    # h -- labels agree with the estimates
    ax = fig.add_subplot(r1[3])
    has = pair[pair.has_fda_label]
    rows = []
    for i, st in enumerate(P.STRATA):
        g = has[has.evidence == st]
        for j, (lst, col) in enumerate(((True, P.VERM), (False, P.GREY))):
            v = g[g.on_fda == lst].af_mean
            q = v.quantile([.25, .5, .75])
            xx = i + (j - .5) * .34
            ax.plot([xx, xx], [q[.25], q[.75]], color=col, lw=2.4, solid_capstyle="butt")
            ax.plot(xx, q[.5], "o", ms=2.4, color="white", mec=col, mew=.8)
            rows.append(dict(stratum=st, listed=lst, n=len(v), q25=q[.25], median=q[.5], q75=q[.75]))
    ax.axhline(0, color=P.BLACK, lw=.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{P.STRATUM_LABEL[s]}\n{S(d, 'validity', 'has_label', s, 'auc'):.2f}" for s in P.STRATA],
                       fontsize=fk.pt("annot"))
    ax.set_ylim(-.1, .62); ax.set_xlabel("trials of the pair (AUC)"); ax.set_ylabel("AF, posterior mean (median, IQR)")
    ax.legend(handles=[mpl.lines.Line2D([], [], color=P.VERM, lw=2.4, label="on the label"),
                       mpl.lines.Line2D([], [], color=P.GREY, lw=2.4, label="not on the label")],
              loc="upper left", frameon=False)
    head(ax, "h", "Labels agree")
    SRC["Fig 2h"] = pd.DataFrame(rows)
    return fig


# ============================================================================== Figure 3
def fig3(d):
    pair = d["pair"]; lab_ = pair[pair.on_fda].copy()
    fig = fk.figure("double", height=95)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    r0 = g0[0].subgridspec(1, 2, width_ratios=[1, 1.9])
    r1 = g0[1].subgridspec(1, 2, width_ratios=[1.9, 1])
    # a -- distribution of AF, by zone
    ax = fig.add_subplot(r0[0])
    bins = np.linspace(-.4, 1, 29)
    x = lab_.af_mean.clip(-.399, .999)
    hs = [np.histogram(x[lab_.zone == z], bins=bins)[0] for z in P.ZONES]
    bottom = np.zeros(len(bins) - 1)
    for z, h in zip(P.ZONES, hs):
        ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", bottom=bottom, color=P.ZONE_COLOR[z], ec="white", lw=.2,
               label=f"{P.ZONE_LABEL[z]}: {100 * S(d, 'headline', 'primary', 'all', 'zone_' + z):.0f}%")
        bottom += h
    ax.axvline(.5, color=P.BLACK, lw=.5, ls="--")
    ax.set_xlim(-.4, 1); ax.set_xlabel("AF, posterior mean (values < −0.4 in first bin)")
    ax.set_ylabel("side effects on current labels")
    ax.legend(loc="upper right", frameon=False)
    head(ax, "a", f"{int(len(lab_)):,} side effects on current labels of {lab_.drug.nunique():,} drugs")
    SRC["Fig 3a"] = pd.DataFrame({"bin_lo": bins[:-1], **{z: h for z, h in zip(P.ZONES, hs)}})
    # b -- every listed side effect with its interval
    ax = fig.add_subplot(r0[1])
    o = lab_.sort_values(["af_mean", "drug", "term"]).reset_index(drop=True)
    xs = np.arange(len(o))
    for z in P.ZONES:
        m = (o.zone == z).values
        ax.vlines(xs[m], o.af_hz_lo.values[m].clip(-1), o.af_hz_hi.values[m], color=P.ZONE_COLOR[z], lw=.12, alpha=.6,
                  rasterized=True)
    ax.plot(xs, o.af_mean, color=P.BLACK, lw=.7)
    ax.axhline(.5, color=P.BLACK, lw=.5, ls="--"); ax.axhline(0, color=P.BLACK, lw=.3)
    ax.set_xlim(0, len(o)); ax.set_ylim(-1, 1)
    ax.set_xlabel("side effects on current labels, ranked by AF")
    ax.set_ylabel("AF (95% credible interval)")
    zone_legend(ax, loc="upper left", ncol=2)
    head(ax, "b", "Every listed side effect: posterior mean (line) and 95% credible interval")
    SRC["Fig 3b"] = o[["drug", "term", "n_trials", "af_mean", "af_hz_lo", "af_hz_hi", "p_af_lt_half", "zone"]]
    # c -- headline by evidence
    ax = fig.add_subplot(r1[0])
    grps = ["all"] + P.STRATA
    mets = [("exp_lt_half", "expected share with true AF < 0.5 (95% CI)", P.BLUE),
            ("conf_bg", "confidently mostly background", fk.c["blue@s0.35"]),
            ("exp_no_effect", "expected share with no drug effect", P.SKY),
            ("conf_drug", "confidently mostly drug-caused", P.VERM)]
    rows = []
    for j, (m, nm, col) in enumerate(mets):
        xx = np.arange(len(grps)) + (j - 1.5) * .2
        v = [S(d, "headline", "primary", g, m) for g in grps]
        ax.bar(xx, v, .19, color=col, label=nm)
        if m == "exp_lt_half":
            lo = [S(d, "headline", "primary", g, m, "lo") for g in grps]
            hi = [S(d, "headline", "primary", g, m, "hi") for g in grps]
            ax.errorbar(xx, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="none", ecolor=P.BLACK, elinewidth=.5,
                        capsize=1)
        for xi, vi in zip(xx, v):
            tx(ax, xi, vi + .03, f"{100 * vi:.0f}", ha="center", fontsize=fk.pt("annot"))
        rows += [dict(metric=m, group=g, value=vv) for g, vv in zip(grps, v)]
    n = [int(S(d, "counts", "listed", g, "pairs")) for g in grps]
    ax.set_xticks(range(len(grps)))
    ax.set_xticklabels([f"all\n({n[0]:,})"] + [f"{P.STRATUM_LABEL[g]} trial{'s' if g != '1' else ''}\n({k:,})"
                                             for g, k in zip(P.STRATA, n[1:])])
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel("share of listed side effects")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    head(ax, "c", "Expected background share is stable across evidence; confidence grows with it")
    SRC["Fig 3c"] = pd.DataFrame(rows)
    # d -- sensitivity
    ax = fig.add_subplot(r1[1])
    sens = [("primary", "headline", "all current-label side effects"),
            ("S1", "sensitivity", "conventionally estimable"), ("S2", "sensitivity", "≥2 trials"),
            ("S6", "sensitivity", "≥5 trials"), ("S3", "sensitivity", "2016 labels (SIDER)"),
            ("S7", "sensitivity", "innovator labels only"), ("S8", "sensitivity", "generic terms removed")]
    rows = []
    top = len(sens) - 1
    for i, (k, an, nm) in enumerate(sens):
        v = S(d, an, k, "all", "exp_lt_half"); lo = S(d, an, k, "all", "exp_lt_half", "lo")
        hi = S(d, an, k, "all", "exp_lt_half", "hi"); cb = S(d, an, k, "all", "conf_bg")
        y = top - i
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", ms=3, color=P.BLUE, mec=P.BLACK, mew=.3, elinewidth=.8)
        ax.plot(cb, y, "D", ms=2.8, color=fk.c["blue@s0.35"], mec=P.BLACK, mew=.3)
        rows.append(dict(set=k, label=nm, exp_lt_half=v, lo=lo, hi=hi, conf_bg=cb))
    conv = S(d, "sensitivity", "S5", "all", "conv_afhi_lt_half")
    ax.plot(conv, -1, "s", ms=3, color=P.GREY, mec=P.BLACK, mew=.3)
    rows.append(dict(set="S5", label="conventional estimator, upper 95% limit < 0.5", conf_bg=conv))
    tx(ax, rows[0]["exp_lt_half"], top + .55, "expected", ha="center", fontsize=fk.pt("annot"), color=P.BLUE)
    tx(ax, rows[0]["conf_bg"], top + .55, "confident", ha="center", fontsize=fk.pt("annot"), color=fk.c["blue@s0.35"])
    tx(ax, conv + .04, -1, "upper limit < 0.5", va="center", fontsize=fk.pt("annot"), color=P.GREY)
    ax.axvline(.5, color=P.BLACK, lw=.5, ls="--")
    ax.set_yticks(range(-1, len(sens))); ax.set_yticklabels(["Mantel–Haenszel"] + [s[2] for s in sens][::-1], fontsize=fk.pt("annot"))
    ax.set_ylim(-1.6, top + 1.1)
    ax.set_xlim(0, 1); ax.set_xlabel("share of listed side effects mostly background")
    head(ax, "d", "Sensitivity analyses")
    SRC["Fig 3d"] = pd.DataFrame(rows)
    return fig


# ============================================================================== Figure 4
def fig4(d):
    pair, s = d["pair"], d["sum"]
    lab_ = pair[pair.on_fda]
    fig = fk.figure("double", height=110)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1.2], height_ratios=[1, 1])
    # a -- organ systems
    ax = fig.add_subplot(gs[:, 0])
    bs = s[s.analysis == "by_soc"]
    ex = bs[(bs.set_name == "primary") & (bs.metric == "exp_lt_half")].set_index("grp")
    tab = pd.DataFrame({"exp_lt_half": ex.value, "lo": ex.lo, "hi": ex.hi, "n": ex.n_pairs,
                        "conf_bg": bs[(bs.set_name == "primary") & (bs.metric == "conf_bg")].set_index("grp").value,
                        "pi0": bs[bs.metric == "pi0"].set_index("grp").value}).sort_values("exp_lt_half")
    rows = []
    for i, (soc, r) in enumerate(tab.iterrows()):
        v = lab_[lab_.soc_h == soc].af_mean
        q = v.quantile([.1, .25, .5, .75, .9])
        ax.plot([q[.1], q[.9]], [i, i], color=P.LGREY, lw=.8)
        ax.plot([q[.25], q[.75]], [i, i], color=fk.c["blue@t0.6"], lw=3, solid_capstyle="butt")
        ax.plot(q[.5], i, "|", ms=4, color=P.BLACK, mew=.8)
        rows.append(dict(organ_system=soc, n=int(r.n), af_p10=q[.1], af_q25=q[.25], af_median=q[.5], af_q75=q[.75],
                         af_p90=q[.9], exp_lt_half=r.exp_lt_half, lo=r.lo, hi=r.hi, conf_bg=r.conf_bg, pi0=r.pi0))
    ax.axvline(0, color=P.BLACK, lw=.4); ax.axvline(.5, color=P.BLACK, lw=.4, ls="--")
    ax.set_xlim(-.3, 1.75); ax.set_xticks([0, .25, .5], ["0", "0.25", "0.5"])
    cols = [(1.02, P.BLUE, "exp_lt_half", "exp.\nAF<0.5"), (1.3, fk.c["blue@s0.35"], "conf_bg", "conf.\nbkgd"),
            (1.58, fk.c["blue@t0.2"], "pi0", "no\neffect")]
    for x0, col, key, hd in cols:
        for i, (soc, r) in enumerate(tab.iterrows()):
            tx(ax, x0, i, f"{100 * r[key]:.0f}", ha="center", va="center", fontsize=fk.pt("annot"), color=col)
        tx(ax, x0, len(tab) - .3, hd, ha="center", va="bottom", fontsize=fk.pt("annot"), color=col)
    tx(ax, 1.3, len(tab) + 1.3, "% of listed side effects", ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{P.SOC_SHORT[x]} ({int(n):,})" for x, n in zip(tab.index, tab.n)], fontsize=fk.pt("annot"))
    ax.set_ylim(-.6, len(tab) + 2.2)
    ax.set_xlabel("AF of listed side effects, posterior mean\n(median, IQR, 10th–90th centile)")
    head(ax, "a", "By organ system (listed side effects)")
    SRC["Fig 4a"] = pd.DataFrame(rows)
    # b -- drug classes
    ax = fig.add_subplot(gs[0, 1])
    bc = s[(s.analysis == "by_class") & (s.set_name == "primary") & (s.grp != "coverage")]
    ct = bc.pivot_table(index="grp", columns="metric", values="value")
    ct["n"] = bc[bc.metric == "exp_lt_half"].set_index("grp").n_pairs
    ct = ct.sort_values(["exp_lt_half"]).reset_index()
    ct["atc1"] = ct.grp.str[0]
    for a1, g in ct.groupby("atc1"):
        ax.scatter(g.index, g.exp_lt_half, s=np.sqrt(g.n) * 1.1, color=P.ATC1_COLOR.get(a1, P.GREY), ec=P.BLACK, lw=.2,
                   zorder=3, label=a1)
    short = lambda g: (lambda n: n if len(n) <= 20 else n[:19].rstrip() + "…")(g.split("|")[1].lower())  # noqa: E731
    bots, tops = ct.head(3), ct.iloc[::-1].head(3)
    lo_, hi_ = ct.iloc[0], ct.iloc[-1]                  # the extremes, each pointing at its class
    for (xv, yv, txt, xyt, ha) in ((0, lo_.exp_lt_half, f"least background:\n{short(lo_.grp)} ({100 * lo_.exp_lt_half:.0f}%)",
                                    (.14, .1), "left"),
                                   (len(ct) - 1, hi_.exp_lt_half, f"most background:\n{short(hi_.grp)} ({100 * hi_.exp_lt_half:.0f}%)",
                                    (.97, .42), "right")):
        a_ = ax.annotate(txt, xy=(xv, yv), xytext=xyt, textcoords="axes fraction", ha=ha, va="center",
                         fontsize=fk.pt("annot"), arrowprops=dict(arrowstyle="-", lw=.4, color=fk.c.grey))
        a_.set_in_layout(False)
    ax.axhline(S(d, "headline", "primary", "all", "exp_lt_half"), color=P.BLACK, lw=.5, ls="--")
    ax.set_xlabel(f"{len(ct)} drug classes (ATC level 3), ranked")
    ax.set_ylabel("expected share with AF < 0.5"); ax.set_xticks([]); ax.set_ylim(.1, 1.0)
    head(ax, "b", "By drug class")
    SRC["Fig 4b"] = ct
    # d -- common complaints are background
    ax = fig.add_subplot(gs[1, 1])
    x = np.log10(lab_.per100_pl.clip(.05, 100)); y = lab_.af_mean.clip(-.5, 1)
    ax.hexbin(x, y, gridsize=(26, 18), cmap="Greys", mincnt=1, bins="log", lw=0, extent=(-1.3, 2, -.5, 1))
    edges = np.quantile(x, np.linspace(0, 1, 11))
    mids, med, fr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        mids.append(float(x[m].median())); med.append(float(y[m].median())); fr.append(float(lab_.p_af_lt_half[m].mean()))
    ax.plot(mids, med, "-o", ms=2.4, color=P.VERM, lw=.9, label="median AF")
    ax.plot(mids, fr, "-s", ms=2.4, color=P.BLUE, lw=.9, label="expected share AF < 0.5")
    ax.set_xticks([-1, 0, 1, 2]); ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("reports per 100 on placebo"); ax.set_ylabel("AF (posterior mean), share")
    ax.set_ylim(-.5, 1); ax.legend(loc="lower left", frameon=False)
    head(ax, "d", "Common complaints are background")
    SRC["Fig 4d"] = pd.DataFrame(dict(log10_placebo_per100_mid=mids, median_af=med, exp_share_af_lt_half=fr))
    # c -- ATC1 x organ system
    ax = fig.add_subplot(gs[:, 2])
    h = s[(s.analysis == "by_class") & (s.set_name == "atc1_soc")].copy()
    h[["atc1", "atc1_name", "soc"]] = h.grp.str.split("|", expand=True)
    socs = [x for x in list(tab.index)[::-1] if x in set(h.soc)]
    a1s = sorted(h.atc1.unique(), key=lambda a: -h[h.atc1 == a].n_pairs.sum())
    M = np.full((len(socs), len(a1s)), np.nan)
    for r in h.itertuples():
        if r.soc in socs:
            M[socs.index(r.soc), a1s.index(r.atc1)] = r.value
    cmap = fk.cmap()
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=mpl.colors.Normalize(.4, 1), aspect="auto")
    F.annotate_matrix(ax, 100 * M, fmt="{:.0f}", cmap=cmap, norm=mpl.colors.Normalize(40, 100))
    ax.set_xticks(range(len(a1s))); ax.set_xticklabels(a1s, fontweight="bold")
    for t, a in zip(ax.get_xticklabels(), a1s):
        t.set_color(P.ATC1_COLOR.get(a, P.GREY))
    ax.set_yticks(range(len(socs))); ax.set_yticklabels([P.SOC_SHORT[x] for x in socs], fontsize=fk.pt("annot"))
    ax.set_xlabel("drug group (ATC level 1)"); ax.set_facecolor(fk.c.grey_faint)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=.4, pad=.01, aspect=22, location="bottom")
    cb.set_label("expected share with AF < 0.5", fontsize=fk.pt("annot")); cb.ax.tick_params(labelsize=fk.pt("tick"))
    head(ax, "c", "Drug group × organ system (≥10 listed)")
    SRC["Fig 4c"] = h[["atc1", "atc1_name", "soc", "n_pairs", "value"]]
    return fig


# ============================================================================== Figure 5
def fig5(d):
    pair, card = d["pair"], d["card"]
    card = card[card.card_set == "main"].merge(
        pair[["drug", "term", "per100_drug_exp", "per100_added_exp", "af_mean"]], on=["drug", "term"])
    fig = fk.figure("double", height=150)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])
    ga = g0[0].subgridspec(3, 3, height_ratios=[1, 1, .07])
    gb = g0[1].subgridspec(1, 3, width_ratios=[1.05, .95, 1.4])
    rows = []
    for k, (dn, g) in enumerate(card.groupby("rank_drug")):
        ax = fig.add_subplot(ga[k // 3, k % 3])
        g = g.sort_values("rank_term")
        y = np.arange(len(g))[::-1]
        ax.barh(y, g.per100_pl, .62, color=P.LGREY)
        ax.barh(y, g.per100_added_exp.clip(lower=0), .62, left=g.per100_pl, color=P.VERM)
        ax.errorbar(g.per100_pl + g.per100_added_exp.clip(lower=0), y,
                    xerr=[(g.per100_added_exp.clip(lower=0) - g.per100_added_lo).clip(lower=0),
                          (g.per100_added_hi - g.per100_added_exp.clip(lower=0)).clip(lower=0)],
                    fmt="none", ecolor=P.BLACK, elinewidth=.5, capsize=1)
        for yy, r in zip(y, g.itertuples()):
            ax.plot([r.per100_pl + r.per100_added_pred_lo, r.per100_pl + r.per100_added_pred_hi], [yy - .4, yy - .4],
                    color=P.VERM, lw=.5, alpha=.8)
        xm = float((g.per100_pl + g.per100_added_pred_hi).max()) * 1.02
        for yy, r in zip(y, g.itertuples()):
            tx(ax, 1.0, yy, f"{100 * r.p_af_lt_half:.0f}%", transform=ax.get_yaxis_transform(), va="center", ha="right",
               fontsize=fk.pt("annot"), color=P.ZONE_TEXT[r.zone], fontweight="bold")
        tx(ax, 1.0, len(g) - .35, "P(background)", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=fk.pt("annot"))
        ax.set_yticks(y); ax.set_yticklabels(list(g.term), fontsize=fk.pt("annot"))
        fk.fit_ticklabels(ax, "y")          # two lines if there is room, else abbreviations listed in the legend
        ax.set_xlim(0, xm * 1.2); ax.set_ylim(-.8, len(g) + .1)
        if k >= 3:
            ax.set_xlabel("reports per 100 patients")
        if k == 0:
            head(ax, "a", g.display_name.iloc[0])
        else:
            ax.set_title(g.display_name.iloc[0], loc="left", fontsize=fk.pt("title"), pad=3)
        for r in g.itertuples():
            rows.append(dict(drug=r.display_name, term=r.term, n_trials=r.n_trials, per100_placebo=r.per100_pl,
                             per100_added=r.per100_added_exp, added_lo=r.per100_added_lo, added_hi=r.per100_added_hi,
                             added_pred_lo=r.per100_added_pred_lo, added_pred_hi=r.per100_added_pred_hi,
                             af=r.af_mean, p_mostly_background=r.p_af_lt_half, zone=r.zone))

    lg = fig.add_subplot(ga[2, :]); lg.axis("off")
    lg.legend(handles=[mpl.patches.Patch(color=P.LGREY, label="would occur on placebo"),
                       mpl.patches.Patch(color=P.VERM, label="added by the drug (95% credible interval)"),
                       mpl.lines.Line2D([], [], color=P.VERM, lw=.5, label="95% interval for a new trial"),
                       mpl.lines.Line2D([], [], ls="", marker="$\\%$", color=P.BLACK,
                                        label="P(mostly background), coloured by zone")],
              loc="center", ncol=4, frameon=False)
    SRC["Fig 5a"] = pd.DataFrame(rows)
    # b -- frequency on drug vs added
    ax = fig.add_subplot(gb[0])
    lab_ = pair[pair.on_fda]
    x = lab_.per100_drug_exp.clip(.05, 100); y = lab_.per100_added_exp.clip(.01, 100)
    for z in P.ZONES[::-1]:
        m = lab_.zone == z
        ax.scatter(x[m], y[m], s=.8, color=P.ZONE_COLOR[z], lw=0, alpha=.5, rasterized=True)
    xx = np.logspace(-1.3, 2, 50)
    for f, ls in ((1, "-"), (.5, "--"), (.1, ":")):
        ax.plot(xx, f * xx, color=P.BLACK, lw=.5, ls=ls)
        tx(ax, 1.2, f * 1.2 * 1.25, f"{int(100 * f)}%", fontsize=fk.pt("annot"), ha="left")
    ax.axhline(1, color=P.VERM, lw=.5, ls="-.")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.8, 100); ax.set_ylim(.01, 100)
    ax.set_xlabel("reports per 100 on the drug (label frequency)"); ax.set_ylabel("reports per 100 added by the drug")
    share = S(d, "absolute_v2", "primary", "all", "share_added_exp_lt1")
    tx(ax, .97, .03, f"{100 * share:.0f}% of listed side effects:\n<1 added per 100 (below dash-dot line)",
       transform=ax.transAxes, ha="right", va="bottom", fontsize=fk.pt("annot"), color=P.VERM,
       bbox=dict(fc=fk.c.white, ec="none", alpha=.85, pad=.6))
    zone_legend(ax, loc="upper left", markerscale=1)
    head(ax, "b", "Label frequency overstates what the drug adds")
    SRC["Fig 5b"] = lab_[["drug", "term", "per100_drug_exp", "per100_pl", "per100_added_exp", "zone"]]
    # c -- listing rules
    ax = fig.add_subplot(gb[1])
    rules = [("rule_a_v2", "≥2% and\nabove placebo"), ("rule_b_v2", "evidence\nof excess"),
             ("rule_c_v2", "evidence and\n≥1 added\nper 100")]
    mets = [("retained", "current entries kept", P.GREY), ("bg_share_retained", "kept that are confidently background", P.BLUE),
            ("drug_caused_retained", "confidently drug-caused entries kept", P.VERM)]
    rows = []
    for j, (m, nm, col) in enumerate(mets):
        xx = np.arange(len(rules)) + (j - 1) * .26
        v = [S(d, "listing_rules_v2", r, "all", m) for r, _ in rules]
        ax.bar(xx, v, .25, color=col, label=nm)
        for xi, vi in zip(xx, v):
            tx(ax, xi, vi + .02, f"{100 * vi:.0f}", ha="center", fontsize=fk.pt("annot"))
        rows += [dict(rule=r, metric=m, value=vv) for (r, _), vv in zip(rules, v)]
    cur = S(d, "listing_rules_v2", "current", "all", "bg_share_retained")
    ax.axhline(cur, color=P.BLUE, lw=.5, ls="--", label=f"confidently background on current labels ({100 * cur:.0f}%)")
    ax.set_xticks(range(len(rules))); ax.set_xticklabels([r[1] for r in rules], fontsize=fk.pt("annot"))
    ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_ylabel("share")
    fk.legend_top(ax, ncol=1)
    head(ax, "c", "Listing rules on the full trial record")
    SRC["Fig 5c"] = pd.DataFrame(rows)
    # d -- the same label entries, current format and proposed format (illustration, real rows; figkit.art)
    ax = fig.add_subplot(gb[2])
    show = card[card.rank_drug == 1].sort_values("rank_term")   # all of the drug's entries in panel a (8)
    rows = [dict(term=r.term, drug=r.per100_drug_exp, placebo=r.per100_pl, added=r.per100_added_exp,
                 lo=r.per100_added_lo, hi=r.per100_added_hi, zone=r.zone) for r in show.itertuples()]
    fk.art.place(ax, SCH.fig5d_art(show.display_name.iloc[0], rows, P.ZONE_TEXT))
    head(ax, "d", "A label line that answers the patient's question")
    SRC["Fig 5d"] = show[["display_name", "term", "per100_drug_exp", "per100_pl", "per100_added_exp", "per100_added_lo",
                          "per100_added_hi", "zone"]]
    return fig




def main() -> int:
    with pv.connect(read_only=True) as con:
        d = load(con)
    assert d["gate"].get("afstd_label_match") and d["gate"].get("af33_adopt"), "prerequisite gates not passed"
    made = {}
    for fid, fn in zip([f for f, _ in P.MAIN_FIGS], (fig1, fig2, fig3, fig4, fig5)):
        made[fid] = fk.save(fn(d), ROOT / "figures" / "main" / fid)      # strict: raises on any rule violation
        print(made[fid].report)
    with pv.run_context(__file__, env_name="oajoint-py311", software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','afstd_card','af3_param','bench_score','bench_record',
                   'af3_sim','ctgov_trial','atlas_contrast')""").fetchall()]
        p = ROOT / "manuscript" / "source_data.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            for key in sorted(SRC):
                SRC[key].to_excel(w, sheet_name=key.replace(" ", "_"), index=False)
        run.archive_document("source_data", p, kind="workbook", upstream=up)
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "bench_score"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
        with pv.connect(read_only=True) as con:
            old = [r[0] for r in con.execute("SELECT rendered_relpath FROM fig_panel WHERE rendered_relpath = "
                                             "'figures/main/fig1c_schematic_prompt.md'").fetchall()]
        if old:                       # the schematic is drawn in code now; the image-generation prompt is retired
            run.retire(old, "Fig. 1 schematic now drawn in code (oajoint.schematics, figkit.art)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
