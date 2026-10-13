#!/usr/bin/env python3
"""Supplementary figures of the manuscript, read only from the project DuckDB, with one Source Data workbook
(manuscript/source_data_supplementary.xlsx). Uses the layout helpers of 111_fig_main.py.

1 data and label linkage | 2 how the estimator and its benchmark were developed | 3 model and calibration detail
4 the hierarchical estimate against conventional pooling | 5 further drug cards | 6 analyses with the conventional estimator
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import figures as F  # noqa: E402
from oajoint import paper as P  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
import figkit as fk  # noqa: E402

spec = importlib.util.spec_from_file_location("m111", ROOT / "scripts" / "111_fig_main.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
head, tx, S = M.head, M.tx, M.S
SRC = {}


def load_extra(con, d):
    d["bench_all"] = con.execute("""SELECT bench, win, stratum, scoring, method, n_records, diff_vs_ref, diff_ci_lo,
                                           diff_ci_hi, pit_central90, pit_low10, pit_high10 FROM bench_score""").df()
    d["pit_all"] = con.execute(f"""SELECT method, pit FROM bench_record WHERE bench = '{M.BENCH_C}' AND scoring = 'hier'
                                   AND af1_support AND method NOT LIKE '%no dose%' ORDER BY method, nct, drug, term""").df()
    d["simall"] = con.execute("SELECT * FROM af3_sim ORDER BY version, scenario").df()
    d["gates"] = pd.DataFrame(con.execute("SELECT gate_name, gate_family, passed, evaluated_at FROM prov_decision").fetchall(),
                              columns=["gate_name", "gate_family", "passed", "evaluated_at"])
    d["conv"] = con.execute("SELECT drug, term, n_trials, af, af_lo, af_hi FROM af_cell").df()
    d["names"] = dict(con.execute("SELECT chembl_id, display_name FROM ref_drug_name").fetchall())
    d["split"] = con.execute("SELECT split, half, drug, term, log_rr, se, p FROM af_split_estimate").df()
    d["ext"] = con.execute("SELECT * EXCLUDE (run_id) FROM af_external_ref").df()
    d["route"] = con.execute("SELECT drug, route_family, term, n_trials, af, af_lo, af_hi FROM af_route_cell").df()
    d["ind"] = con.execute("SELECT drug, indication, term, n_trials, af, af_lo, af_hi FROM af_indication_cell").df()
    d["dose"] = con.execute("""SELECT d.drug, d.term, d.rr_hl, c.af FROM af_dose_cell d JOIN af_cell c USING (drug, term)
                               JOIN af_label_match m USING (drug, term) WHERE m.on_fda""").df()
    d["lm"] = con.execute("SELECT drug, term, has_fda_label, label_source, on_fda, sider_mapped, on_sider, term_promiscuity "
                          "FROM afstd_label_match").df()
    d["gd"] = {r[0]: r[1] for r in con.execute("SELECT gate_name, metric_detail FROM prov_decision").fetchall()}
    return d


# ============================================================================== S1 data and labels
def s1(d):
    pair, lm = d["pair"], d["lm"]
    fig = fk.figure("double", height=70, kind="supp")
    gs = fig.add_gridspec(1, 4, width_ratios=[.75, .7, .9, 1.75])
    ax = fig.add_subplot(gs[0])
    cats = [("innovator label", (lm.label_source == "innovator").groupby(lm.drug).first().sum()),
            ("any label (no innovator)", (lm.label_source == "all_labels").groupby(lm.drug).first().sum()),
            ("no current label", (~lm.has_fda_label).groupby(lm.drug).first().sum())]
    ax.bar(range(3), [c[1] for c in cats], .6, color=[P.BLUE, P.SKY, P.LGREY])
    for i, (nm, n) in enumerate(cats):
        tx(ax, i, n * 1.02, f"{int(n):,}", ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax.set_xticks(range(3)); fk.tilt(ax, [c[0] for c in cats]); ax.set_ylabel("drugs")
    ax.set_ylim(0, max(c[1] for c in cats) * 1.15)
    head(ax, "a", "Source of label text")
    SRC["SFig 1a"] = pd.DataFrame(cats, columns=["source", "drugs"])
    ax = fig.add_subplot(gs[1])
    both = lm[lm.has_fda_label & lm.sider_mapped]
    v = [("FDA only", int((both.on_fda & ~both.on_sider).sum())), ("both", int((both.on_fda & both.on_sider).sum())),
         ("SIDER only", int((~both.on_fda & both.on_sider).sum()))]
    ax.bar(range(3), [x[1] for x in v], .6, color=[P.VERM, P.PURP, P.BLUE])
    for i, (nm, n) in enumerate(v):
        tx(ax, i, n * 1.02, f"{n:,}", ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax.set_xticks(range(3)); ax.set_xticklabels(["FDA\nonly", "both", "SIDER\nonly"]); ax.set_ylabel("listed pairs")
    ax.set_ylim(0, max(x[1] for x in v) * 1.12)
    jac = v[1][1] / sum(x[1] for x in v)
    ax.set_xlabel(f"Jaccard index {jac:.2f}")      # below the axis: never on a bar
    head(ax, "b", "Current vs 2016 labels")
    SRC["SFig 1b"] = pd.DataFrame(v, columns=["set", "pairs"])
    ax = fig.add_subplot(gs[2])
    pr = lm.dropna(subset=["term_promiscuity"]).drop_duplicates("term").sort_values("term_promiscuity", ascending=False)
    ax.hist(pr.term_promiscuity, bins=np.linspace(0, 1, 41), color=P.GREY)
    ax.axvline(.5, color=P.VERM, lw=.6, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("share of current label texts matching the term"); ax.set_ylabel("terms")
    top = pr.head(8)
    tx(ax, .97, .95, "\n".join(f"{t} ({100 * v:.0f}%)" for t, v in zip(top.term, top.term_promiscuity)),
       transform=ax.transAxes, ha="right", va="top", fontsize=fk.pt("annot"))
    head(ax, "c", "Generic terms")
    SRC["SFig 1c"] = pr[["term", "term_promiscuity"]]
    ax = fig.add_subplot(gs[3])
    lab_ = pair[pair.on_fda & pair.atc1.notna()]
    tab = pd.crosstab(lab_.soc_h, lab_.atc1)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index[:16], tab.sum(axis=0).sort_values(ascending=False).index]
    im = ax.imshow(np.log10(tab.values + 1), cmap="Greys", aspect="auto")
    F.annotate_matrix(ax, tab.values.astype(float), fmt="{:.0f}", cmap=mpl.colormaps["Greys"],
                      norm=mpl.colors.Normalize(0, np.log10(tab.values.max() + 1)))
    for t in ax.texts:
        try:
            t.set_text(str(int(float(t.get_text())))) if float(t.get_text()) > 0 else t.set_text("")
        except ValueError:
            pass
    ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels(tab.columns)
    ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels([P.SOC_SHORT[x] for x in tab.index], fontsize=fk.pt("annot"))
    ax.set_xlabel("drug group (ATC level 1, primary class)"); ax.tick_params(length=0)
    head(ax, "d", "Listed pairs: drug group × organ system")
    SRC["SFig 1d"] = tab.reset_index()
    return fig


def _status(name, passed, evaluated_at, family):
    if name in P.SUPERSEDED:
        return "superseded"
    if passed is True:
        return "pass"
    if passed is False:
        return "fail"
    if evaluated_at is None or evaluated_at is pd.NaT or (isinstance(evaluated_at, float) and np.isnan(evaluated_at)):
        return "design record" if family in ("descriptive", "pre_registration") else "not yet evaluated"
    return "descriptive or unevaluable"


# ============================================================================== S2 development history
def s2(d):
    b, g = d["bench_all"], d["gates"]
    fig = fk.figure("double", height=105, kind="supp")
    g0 = fig.add_gridspec(2, 1)
    r0 = g0[0].subgridspec(1, 3, width_ratios=[1, 1, 1])
    r1 = g0[1].subgridspec(1, 2, width_ratios=[1, 1.4])
    # a windows
    ax = fig.add_subplot(r0[0])
    hm = ["B7 class-Bayes", "B3 Bayes-NN", "B4 AF 2.0", "B1 MH"]
    D = b[(b.bench == "bench21_dev33") & (b.scoring == "hier") & (b.stratum == "AF1-support") & (b.win != "pooled")]
    Cw = b[(b.bench == M.BENCH_C) & (b.scoring == "hier") & (b.stratum == "AF1-support") & (b.win == "pooled")].assign(win="C2017")
    H = pd.concat([D, Cw]); wins = ["W2011", "W2013", "W2015", "C2017"]; rows = []
    for m in hm:
        t = H[H.method == m].set_index("win").reindex(wins)
        ax.errorbar(range(4), t.diff_vs_ref, yerr=[t.diff_vs_ref - t.diff_ci_lo, t.diff_ci_hi - t.diff_vs_ref], fmt="-o",
                    ms=2.6, lw=.8, color=P.METHOD_COLOR[m], mec=P.BLACK, mew=.3, elinewidth=.6, label=P.METHOD_SHORT[m])
        rows += [dict(method=m, window=w, **t.loc[w, ["diff_vs_ref", "diff_ci_lo", "diff_ci_hi"]].to_dict()) for w in wins]
    ax.axhline(0, color=P.BLACK, lw=.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["2011–12", "2013–14", "2015–16", "2017–\n(confirm)"])
    ax.set_ylabel("log-score gain of this estimator")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-.02, -.3))   # below: the tall 2015 CI stays clear
    head(ax, "a", "Development windows and confirmation")
    SRC["SFig 2a"] = pd.DataFrame(rows)
    # b baseline iteration
    ax = fig.add_subplot(r0[1])
    Dp = b[(b.bench == "bench21_dev33") & (b.win == "pooled") & (b.stratum == "AF1-support")]
    ms = ["AF 3.3", "AF 3.2", "B3 Bayes-NN", "B2 PM-t", "B4 AF 2.0", "B1 MH"]
    x = np.arange(len(ms)); rows = []
    for sc, off, alpha, nm in (("jeffreys", -.17, .45, "own placebo arm"), ("hier", .17, 1, "hierarchical background")):
        t = Dp[Dp.scoring == sc].set_index("method")
        ax.bar(x + off, [t.loc[m, "pit_low10"] for m in ms], .32, color=[P.METHOD_COLOR[m] for m in ms], alpha=alpha)
        rows += [dict(scoring=sc, method=m, pit_low10=t.loc[m, "pit_low10"]) for m in ms]
    ax.axhline(.10, color=P.BLACK, lw=.5, ls="--")
    ax.set_xticks(x); fk.tilt(ax, [P.METHOD_SHORT[m] for m in ms])
    ax.set_ylabel("share below 10th predictive percentile")
    ax.legend(handles=[mpl.patches.Patch(color=P.GREY, alpha=.45, label="own placebo arm"),
                       mpl.patches.Patch(color=P.GREY, label="hierarchical background")], loc="upper left")
    ax.set_ylim(0, .21)
    head(ax, "b", "Iterating the benchmark itself")
    SRC["SFig 2b"] = pd.DataFrame(rows)
    # c ablation / version steps in confirmation
    ax = fig.add_subplot(r0[2])
    C = b[(b.bench == M.BENCH_C) & (b.win == "pooled") & (b.scoring == "hier") & (b.stratum == "AF1-support")].set_index("method")
    steps = ["B1 MH", "B4 AF 2.0", "B3 Bayes-NN", "B7 class-Bayes", "AF 3.2 no dose offsets", "AF 3.2",
             "AF 3.3 no dose offsets", "AF 3.3"]
    labels = ["Mantel–Haenszel", "Empirical-Bayes shrinkage", "Bayesian RE MA", "Class-informed Bayes MA",
              "hierarchical, no dose, no no-effect part", "+ dose offsets", "+ no-effect part, no dose", "full estimator"]
    gains = [-C.loc[m, "diff_vs_ref"] + C.loc["B1 MH", "diff_vs_ref"] for m in steps]
    ax.bar(range(len(steps)), gains, .62, color=[P.METHOD_COLOR.get(m.replace(" no dose offsets", ""), P.GREY)
                                                for m in steps])
    for i, v in enumerate(gains):
        tx(ax, i, v + max(gains) * .02, f"{v:+.3f}", ha="center", va="bottom", rotation=90, fontsize=fk.pt("annot"))
    ax.set_xticks(range(len(steps))); fk.tilt(ax, labels)
    ax.set_ylabel("gain over MH\n(log score)"); ax.set_ylim(0, max(gains) * 1.45)
    head(ax, "c", "Component contributions")
    SRC["SFig 2c"] = pd.DataFrame(dict(step=steps, label=labels, gain_over_mh=gains))
    # d serious events by evidence
    ax = fig.add_subplot(r1[0])
    Ss = b[(b.bench == M.BENCH_S) & (b.win == "pooled") & (b.scoring == "hier")]
    strata = ["evidence 1 trial", "evidence 2-4", "evidence 5-9", "evidence >=10", "unseen"]; rows = []
    for j, m in enumerate(["B5 term-only", "B6 class-mean", "B0 null", "B3 Bayes-NN"]):
        t = Ss[Ss.method == m].set_index("stratum").reindex(strata)
        x = np.arange(len(strata)) + (j - 1.5) * .18
        ok = t.diff_vs_ref.notna().values
        ax.errorbar(x[ok], t.diff_vs_ref[ok], yerr=[(t.diff_vs_ref - t.diff_ci_lo)[ok], (t.diff_ci_hi - t.diff_vs_ref)[ok]],
                    fmt="o", ms=2.6, color=P.METHOD_COLOR[m], mec=P.BLACK, mew=.3, elinewidth=.6, label=P.METHOD_SHORT[m])
        rows += [dict(comparator=m, stratum=s, **t.loc[s, ["n_records", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi"]].to_dict())
                 for s in strata]
    ax.axhline(0, color=P.BLACK, lw=.5)
    ax.set_xticks(range(len(strata))); ax.set_xticklabels(["1", "2–4", "5–9", "≥10", "0 (new)"])
    ax.set_xlabel("earlier trials of the pair"); ax.set_ylabel("log-score gain (serious events)")
    ax.legend(frameon=False, loc="upper right"); ax.set_ylim(-.27, .5)    # headroom: the legend clears every CI cap
    head(ax, "d", "Serious events: provisional scope")
    SRC["SFig 2d"] = pd.DataFrame(rows)
    # e rule ledger
    ax = fig.add_subplot(r1[1])
    rows = []
    for grp, names in P.GATES.items():
        gg = g[g.gate_name.isin(names)]
        st = np.array([_status(r.gate_name, r.passed, r.evaluated_at, r.gate_family) for r in gg.itertuples()])
        for k in ["pass", "fail", "superseded", "descriptive or unevaluable", "design record", "not yet evaluated"]:
            rows.append(dict(group=grp, status=k, n=int((st == k).sum())))
    L = pd.DataFrame(rows)
    cols = {"pass": P.GREEN, "fail": P.VERM, "superseded": P.ORANGE, "descriptive or unevaluable": P.LGREY,
            "design record": fk.c.grey_faint, "not yet evaluated": "white"}
    grps = list(P.GATES)
    left = np.zeros(len(grps))
    for k, c in cols.items():
        v = L[L.status == k].set_index("group").reindex(grps).n.values
        ax.barh(range(len(grps))[::-1], v, left=left, color=c, ec=P.GREY, lw=.3, label=k)
        for i, (lv, vv) in enumerate(zip(left, v)):
            if vv:
                tx(ax, lv + vv / 2, len(grps) - 1 - i, str(vv), ha="center", va="center", fontsize=fk.pt("annot"))
        left += v
    ax.set_yticks(range(len(grps))[::-1]); ax.set_yticklabels(grps, fontsize=fk.pt("annot"))
    ax.set_xlabel("rules registered before evaluation"); ax.legend(frameon=False, loc="center right")
    head(ax, "e", f"The rule ledger ({int(L.n.sum())} rules; Supplementary Table 1)")
    SRC["SFig 2e"] = L
    return fig


# ============================================================================== S3 model and calibration detail
def s3(d):
    b, pair, par = d["bench_all"], d["pair"], d["par"]
    fig = fk.figure("double", height=100, kind="supp")
    g0 = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    r0 = g0[0].subgridspec(2, 5)
    r1 = g0[1].subgridspec(1, 3, width_ratios=[1, 1.2, 1])
    ms = M.ROWS_B; bins = np.linspace(0, 1, 11); rows = []
    for k, m in enumerate(ms):
        ax = fig.add_subplot(r0[k // 5, k % 5])
        h, _ = np.histogram(d["pit_all"][d["pit_all"].method == m].pit, bins=bins, density=True)
        ax.stairs(h, bins, fill=True, color=P.METHOD_COLOR.get(m, P.GREY), alpha=.85)
        ax.axhline(1, color=P.BLACK, lw=.4, ls=":"); ax.set_ylim(0, 2.5); ax.set_xlim(0, 1)
        ax.spines["left"].set_bounds(0, 2)             # range frame: the spine stops at the last tick
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, 1, 2])
        if k % 5:
            ax.tick_params(labelleft=False)
        if k < 5:
            ax.tick_params(labelbottom=False)
        if k == 0:
            fk.panel(ax, "a", P.METHOD_SHORT[m])
            ax._left_title.set_fontsize(fk.pt("tick"))      # keep the letter headroom fk.panel reserved
        else:
            ax.set_title(P.METHOD_SHORT[m], fontsize=fk.pt("tick"), pad=2, loc="left")
        rows += [dict(method=m, bin_lo=bins[i], density=h[i]) for i in range(10)]
    SRC["SFig 3a"] = pd.DataFrame(rows)
    # b coverage by evidence
    ax = fig.add_subplot(r1[0])
    C = b[(b.bench == M.BENCH_C) & (b.win == "pooled") & (b.scoring == "hier")]
    strata = ["evidence 1 trial", "evidence 2-4", "evidence 5-9", "evidence >=10", "unseen"]; rows = []
    ax.axhspan(.87, .93, color=P.LGREY, lw=0)
    for j, m in enumerate(["AF 3.3", "B7 class-Bayes", "B3 Bayes-NN", "B5 term-only"]):
        t = C[C.method == m].set_index("stratum").reindex(strata)
        ax.plot(np.arange(len(strata)) + (j - 1.5) * .12, t.pit_central90, "o", ms=3, color=P.METHOD_COLOR[m],
                mec=P.BLACK, mew=.3, label=P.METHOD_SHORT[m])
        rows += [dict(method=m, stratum=s, pit_central90=t.loc[s, "pit_central90"]) for s in strata]
    ax.set_xticks(range(len(strata))); ax.set_xticklabels(["1", "2–4", "5–9", "≥10", "0 (new)"])
    ax.set_xlabel("earlier trials of the pair"); ax.set_ylabel("share in central 90% interval"); ax.set_ylim(.70, .97)   # a band for the legend
    ax.legend(frameon=False, loc="lower left")
    head(ax, "b", "Coverage by evidence")
    SRC["SFig 3b"] = pd.DataFrame(rows)
    # c organ-system parameters
    ax = fig.add_subplot(r1[1])
    lab_ = pair[pair.on_fda]
    socs = sorted(lab_.groupby("soc_h").size().loc[lambda s: s >= 30].index)
    tau = par[par.kind == "tau"].set_index("soc").value
    pi0 = pair.groupby("soc_h").p_no_effect.mean()
    t = pd.DataFrame(dict(tau=tau.reindex(socs), pi0=pi0.reindex(socs))).sort_values("pi0")
    y = np.arange(len(t))
    ax.barh(y, t.pi0, .6, color=P.SKY, label=r"no-effect weight $\pi_0$")
    ax2 = ax.twiny(); ax2.plot(t.tau, y, "D", ms=2.6, color=P.VERM, mec=P.BLACK, mew=.3, label="heterogeneity τ")
    ax.set_yticks(y); ax.set_yticklabels([P.SOC_SHORT[s] for s in t.index], fontsize=fk.pt("annot"))
    ax.set_xlabel(r"no-effect weight $\pi_0$ (bars)"); ax2.set_xlabel("between-trial SD τ (diamonds)", fontsize=fk.pt("base"), color=P.VERM)
    ax2.tick_params(axis="x", colors=P.VERM, labelsize=fk.pt("tick"))
    head(ax, "c", "")
    SRC["SFig 3c"] = t.reset_index().rename(columns={"index": "organ_system"})
    # d simulation across versions
    ax = fig.add_subplot(r1[2])
    sm = d["simall"]; rows = []
    vers = ["AF 3.0", "AF 3.2", "AF 3.3"]
    vlab = {"AF 3.0": "first hierarchical", "AF 3.2": "+ pair-specific τ", "AF 3.3": "+ no-effect part"}
    for j, v in enumerate(vers):
        t = sm[sm.version == v].sort_values("scenario")
        ax.plot(np.arange(len(t)) + (j - 1) * .18, t.rmse_af3, "o", ms=3, color=[P.SKY, P.ORANGE, P.VERM][j], mec=P.BLACK,
                mew=.3, label=vlab[v])
        rows += t[["version", "scenario", "coverage_af3", "rmse_af3", "coverage_af1", "rmse_af1"]].to_dict("records")
    t = sm[sm.version == "AF 3.3"].sort_values("scenario")
    ax.plot(np.arange(len(t)) + .36, t.rmse_af1, "s", ms=3, color=P.GREY, mec=P.BLACK, mew=.3, label="Mantel–Haenszel")
    ax.set_xticks(range(len(t))); ax.set_xticklabels(["varying", "similar", "skewed"]); ax.set_xlabel("simulated effects")
    ax.set_ylabel("RMSE of log hazard ratio"); ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(.2, .56)                                  # headroom: the legend clears the squares
    head(ax, "d", "Known-truth simulation, by version")
    SRC["SFig 3d"] = pd.DataFrame(rows)
    return fig


# ============================================================================== S4 hierarchical vs conventional
def s4(d):
    pair, conv = d["pair"], d["conv"]
    x = pair[pair.af1_estimable].merge(conv, on=["drug", "term"], suffixes=("", "_mh"))
    fig = fk.figure("double", height=62, kind="supp")
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0])
    ax.scatter(x.af.clip(-1.5, 1), x.af_mean, s=1.2, c=np.log10(x.n_trials_mh), cmap=fk.cmap(), lw=0, rasterized=True)
    ax.plot([-1, 1], [-1, 1], color=P.BLACK, lw=.5, ls="--")
    rho = stats.spearmanr(x.af, x.af_mean).statistic
    ax.set_xlim(-1.5, 1); ax.set_ylim(-.8, 1); ax.set_xlabel("AF, Mantel–Haenszel"); ax.set_ylabel("AF, this estimator")
    tx(ax, .03, .97, f"{len(x):,} pairs\nSpearman ρ {rho:.2f}", transform=ax.transAxes, va="top", fontsize=fk.pt("annot"))
    head(ax, "a", "Same pairs, two estimators")
    SRC["SFig 4a"] = x[["drug", "term", "n_trials_mh", "af", "af_mean"]]
    ax = fig.add_subplot(gs[1])
    wmh = (np.log(1 - x.af_lo.clip(upper=.999)) - np.log(1 - x.af_hi.clip(upper=.999))).abs()
    whz = (np.log(1 - x.af_hz_lo.clip(upper=.999)) - np.log(1 - x.af_hz_hi.clip(upper=.999))).abs()
    q = pd.qcut(x.n_trials_mh, [0, .25, .5, .75, 1], duplicates="drop")
    t = pd.DataFrame(dict(q=q, mh=wmh, hz=whz)).groupby("q", observed=True).median()
    ax.plot(range(len(t)), t.mh, "-s", ms=3, color=P.GREY, label="Mantel–Haenszel 95% CI")
    ax.plot(range(len(t)), t.hz, "-o", ms=3, color=P.VERM, label="this estimator 95% CrI")
    ax.set_xticks(range(len(t))); ax.set_xticklabels([f"{int(i.left) + 1}–{int(i.right)}" for i in t.index], fontsize=fk.pt("annot"))
    ax.set_xlabel("trials of the pair (quartiles)"); ax.set_ylabel("median interval width (log HR / log RR)")
    ax.legend(frameon=False)
    head(ax, "b", "Interval width")
    SRC["SFig 4b"] = t.reset_index().astype({"q": str})
    ax = fig.add_subplot(gs[2])
    lab = x[x.on_fda]
    vals = [("MH upper limit < 0.5", (lab.af_hi < .5).mean()),
            ("confidently background", (lab.p_af_lt_half > .975).mean()),
            ("MH point AF < 0.5", (lab.af < .5).mean()),
            ("expected AF < 0.5", lab.p_af_lt_half.mean())]
    ax.bar(range(4), [v for _, v in vals], .6, color=[P.GREY, fk.c["blue@s0.35"], P.LGREY, P.BLUE])
    for i, (_, v) in enumerate(vals):
        tx(ax, i, v + .02, f"{100 * v:.0f}%", ha="center", fontsize=fk.pt("annot"))
    ax.set_xticks(range(4)); fk.tilt(ax, [v[0] for v in vals]); ax.set_ylim(0, 1)
    ax.set_ylabel(f"share of {len(lab):,} listed pairs")
    head(ax, "c", "Headline on the same pairs")
    SRC["SFig 4c"] = pd.DataFrame(vals, columns=["measure", "share"])
    ax = fig.add_subplot(gs[3])
    shr = (x.af_mean - x.af.clip(-1.5, 1)).clip(-1.5, 1.5)
    ax.scatter(x.n_trials_mh, shr, s=1.2, color=P.BLUE, lw=0, alpha=.4, rasterized=True)
    ax.axhline(0, color=P.BLACK, lw=.5); ax.set_xscale("log"); ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel("trials of the pair"); ax.set_ylabel("AF difference (this − MH; clipped at ±1.5)")
    head(ax, "d", "Shrinkage fades with evidence")
    SRC["SFig 4d"] = pd.DataFrame(dict(n_trials=x.n_trials_mh, diff=shr))
    return fig


# ============================================================================== S5 further drug cards
def s5(d):
    pair, card = d["pair"], d["card"]
    card = card[card.card_set == "supplementary"].merge(pair[["drug", "term", "per100_added_exp"]], on=["drug", "term"])
    fig = fk.figure("double", height=175, kind="supp")
    gs = fig.add_gridspec(4, 3)
    rows = []
    for k, (dn, g) in enumerate(card.groupby("rank_drug")):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        g = g.sort_values("rank_term"); y = np.arange(len(g))[::-1]
        ax.barh(y, g.per100_pl, .62, color=P.LGREY)
        ax.barh(y, g.per100_added_exp.clip(lower=0), .62, left=g.per100_pl, color=P.VERM)
        ax.errorbar(g.per100_pl + g.per100_added_exp.clip(lower=0), y,
                    xerr=[(g.per100_added_exp.clip(lower=0) - g.per100_added_lo).clip(lower=0),
                          (g.per100_added_hi - g.per100_added_exp.clip(lower=0)).clip(lower=0)],
                    fmt="none", ecolor=P.BLACK, elinewidth=.5, capsize=1)
        for yy, r in zip(y, g.itertuples()):
            tx(ax, 1.0, yy, f"{100 * r.p_af_lt_half:.0f}%", transform=ax.get_yaxis_transform(), va="center", ha="right",
               fontsize=fk.pt("annot"), color=P.ZONE_TEXT[r.zone], fontweight="bold")
        xm = float((g.per100_pl + g.per100_added_hi.clip(lower=0)).max())
        ax.set_xlim(0, xm * 1.25); ax.set_yticks(y)
        ax.set_yticklabels(list(g.term), fontsize=fk.pt("annot"))
        fk.fit_ticklabels(ax, "y")          # two lines if there is room, else abbreviations listed in the legend
        ax.set_title(g.display_name.iloc[0], loc="left", fontsize=fk.pt("title"), pad=2)
        if k >= 9:
            ax.set_xlabel("reports per 100 patients")
        rows += g[["display_name", "term", "n_trials", "per100_pl", "per100_added_exp", "per100_added_lo",
                   "per100_added_hi", "af_med", "p_af_lt_half", "zone"]].to_dict("records")
    SRC["SFig 5"] = pd.DataFrame(rows)
    return fig


# ============================================================================== S6 conventional-estimator analyses
def s6(d):
    names = d["names"]
    fig = fk.figure("double", height=100, kind="supp")
    gs = fig.add_gridspec(2, 3)
    sp = d["split"]; rows = {}
    for k, (split, letter, ttl) in enumerate((("random", "a", "Random halves of the trials"),
                                              ("temporal", "b", "Earlier vs later trials"))):
        ax = fig.add_subplot(gs[0, k])
        t = sp[sp.split == split].pivot_table(index=["drug", "term"], columns="half", values="log_rr").dropna()
        ax.scatter(t["A"].clip(-3, 3), t["B"].clip(-3, 3), s=1, color=P.BLUE, lw=0, alpha=.35, rasterized=True)
        ax.plot([-3, 3], [-3, 3], color=P.BLACK, lw=.5, ls="--")
        rho = stats.spearmanr(t["A"], t["B"]).statistic
        tx(ax, .03, .97, f"{len(t):,} pairs; ρ {rho:.2f}", transform=ax.transAxes, va="top", fontsize=fk.pt("annot"))
        ax.set_xlabel("log RR, half A"); ax.set_ylabel("log RR, half B")
        head(ax, letter, ttl)
        rows[f"SFig 6{letter}"] = t.reset_index()
    ax = fig.add_subplot(gs[0, 2])
    e = d["ext"][d["ext"].found.fillna(False)].copy()
    e["atlas_log"] = np.log(e.atlas_rr); e["pub_log"] = np.log(e.estimate)
    ax.scatter(e.pub_log, e.atlas_log, s=10, color=P.VERM, ec=P.BLACK, lw=.3)
    lim = [min(e.pub_log.min(), e.atlas_log.min()) - .2, max(e.pub_log.max(), e.atlas_log.max()) + .2]
    ax.plot(lim, lim, color=P.BLACK, lw=.5, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    fk.callouts(ax, e.pub_log.values, e.atlas_log.values,
                [f"{names.get(r.drug, r.drug)} · {'UTI' if r.term == 'urinary tract infection' else r.term}"
                 for r in e.itertuples()])
    fig._figkit.setdefault("abbreviations", {})["UTI"] = "urinary tract infection"   # -> figure legend
    ax.set_xlabel("published meta-analysis (log ratio)"); ax.set_ylabel("registry estimate (log RR)")
    head(ax, "c", f"Published meta-analyses ({len(e)} pairs)")
    rows["SFig 6c"] = e[["drug", "term", "estimate", "atlas_rr", "pmid"]]
    ax = fig.add_subplot(gs[1, 0])
    ro = d["route"]
    ex = [("aflibercept", "hypertension"), ("varenicline", "cough"), ("aripiprazole", "akathisia"),
          ("semaglutide", "nausea")]
    inv = {v: k for k, v in names.items()}
    y = 0; yl = []; rr = []
    for dn, t in ex:
        g = ro[(ro.drug == inv.get(dn)) & (ro.term == t)].sort_values("route_family")
        for r in g.itertuples():
            ax.errorbar(r.af, y, xerr=[[r.af - max(r.af_lo, -1)], [r.af_hi - r.af]], fmt="o", ms=2.8, color=P.BLUE,
                        mec=P.BLACK, mew=.3, elinewidth=.7)
            yl.append(f"{dn} · {t} ({r.route_family})"); rr.append(r._asdict()); y -= 1
        y -= .5
    ax.axvline(0, color=P.BLACK, lw=.4); ax.axvline(.5, color=P.BLACK, lw=.4, ls="--")
    ypos, yv = [], 0
    for dn, t in ex:
        n = int(((ro.drug == inv.get(dn)) & (ro.term == t)).sum())
        ypos += [yv - i for i in range(n)]; yv -= n + .5
    ax.set_yticks(ypos); ax.set_yticklabels(yl, fontsize=fk.pt("annot"))
    ax.set_xlim(-1, 1); ax.set_xlabel("AF, conventional estimator (95% CI)")
    head(ax, "d", "Same drug, different product")
    rows["SFig 6d"] = pd.DataFrame(rr)
    ax = fig.add_subplot(gs[1, 1])
    ind = d["ind"][d["ind"].n_trials >= 3]
    g = ind.groupby(["drug", "term"]).af.agg(["max", "min", "size"])
    g = g[g["size"] >= 2].assign(spread=lambda x: x["max"] - x["min"]).sort_values("spread", ascending=False).head(8)
    y = 0; yl = []; rr = []
    for (dr, t), _ in g.iterrows():
        tt = ind[(ind.drug == dr) & (ind.term == t)].sort_values("af")
        ax.plot(tt.af.clip(-1, 1), [y] * len(tt), "-o", ms=2.6, color=P.PURP, mec=P.BLACK, mew=.3, lw=.6)
        yl.append(f"{names.get(dr, dr)} · {t}"); rr += tt.to_dict("records"); y -= 1
    ax.set_yticks(range(0, y, -1)); ax.set_yticklabels(yl, fontsize=fk.pt("annot")); ax.set_xlim(-1, 1)
    ax.axvline(0, color=P.BLACK, lw=.4)
    ax.set_xlabel("AF by indication (≥3 trials each), conventional")
    head(ax, "e", "Largest differences between indications")
    rows["SFig 6e"] = pd.DataFrame(rr)
    ax = fig.add_subplot(gs[1, 2])
    ds = d["dose"]
    ax.scatter(ds.af.clip(-1, 1), np.log(ds.rr_hl).clip(-2, 2), s=3, color=P.BLUE, lw=0, alpha=.5)
    rho = stats.spearmanr(ds.af, np.log(ds.rr_hl)).statistic
    ax.axhline(0, color=P.BLACK, lw=.4)
    tx(ax, .03, .97, f"{len(ds):,} listed pairs; ρ {rho:.2f}", transform=ax.transAxes, va="top", fontsize=fk.pt("annot"))
    ax.set_xlabel("AF, drug vs placebo"); ax.set_ylabel("log RR, highest vs lowest dose")
    head(ax, "f", "Within-trial dose contrasts")
    rows["SFig 6f"] = ds
    SRC.update(rows)
    return fig


def main() -> int:
    with pv.connect(read_only=True) as con:
        d = M.load(con)
        d = load_extra(con, d)
    made = {}
    for (fid, _), fn in zip(P.SUPP_FIGS, (s1, s2, s3, s4, s5, s6)):
        made[fid] = fk.save(fn(d), ROOT / "figures" / "supplementary" / fid)   # strict: raises on violations
        print(made[fid].report)
    with pv.run_context(__file__, env_name="oajoint-py311", software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','afstd_card','afstd_label_match','af3_param','bench_score',
                   'bench_record','af3_sim','af_cell','af_split_estimate','af_external_ref','af_route_cell',
                   'af_indication_cell','af_dose_cell')""").fetchall()]
        p = ROOT / "manuscript" / "source_data_supplementary.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            for key in sorted(SRC):
                SRC[key].to_excel(w, sheet_name=key.replace(" ", "_")[:31], index=False)
        run.archive_document("source_data_supplementary", p, kind="workbook", upstream=up)
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "bench_score", "af_cell"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
        live = {str(p.relative_to(ROOT)) for saved in made.values() for p in saved.paths}
        with pv.connect(read_only=True) as con:
            stale = [r[0] for r in con.execute("SELECT rendered_relpath FROM fig_panel WHERE script_path LIKE "
                                               "'%/112_fig_supp.py'").fetchall() if r[0] not in live]
        if stale:
            run.retire(stale, "supplementary figure renamed (112_fig_supp.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
