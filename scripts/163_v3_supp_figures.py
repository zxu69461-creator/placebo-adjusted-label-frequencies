#!/usr/bin/env python3
"""Supporting figures of the PLOS Medicine manuscript, read only from the project DuckDB, with one Source Data workbook
(manuscript/v3/source_data_supplementary.xlsx). Panels are those of the earlier version (112_fig_supp.py) in the v3
terminology and numbering (first citation in the v3 text), with two changes: serious adverse events get their own
figure (from main Fig. 2f of v2 and Supplementary Fig. 2d of v2), and the development figure keeps only its
scientific panels (development windows, benchmark placebo model, component ablation).

1 data and label linkage | 2 calibration and model detail | 3 serious adverse events | 4 hierarchical vs conventional
5 further drug cards | 6 analyses with the conventional estimator | 7 model development and component ablation
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import paper_v3 as P  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
import figkit as fk  # noqa: E402

spec = importlib.util.spec_from_file_location("m112", ROOT / "scripts" / "112_fig_supp.py")
M112 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M112)
M112.P = P                                  # v3 names (zones, event average) in the reused panels
M = M112.M
head, tx = M.head, M.tx
SRC = M112.SRC
OUT = ROOT / P.SUPP_DIR
BENCH_S = M.BENCH_S


def _forest(ax, t, ref="AF 3.3", xmax=None, cov_x=None):
    """Gain of the model over each comparator (rows, best comparator at the bottom) with a 90%-coverage column."""
    comps = [m for m in t.sort_values("diff_vs_ref", ascending=False).index if m != ref]
    order = comps + [ref]
    rows = []
    for yv, m in enumerate(order):
        r = t.loc[m]
        if m == ref:
            ax.plot(0, yv, "D", ms=3.2, color=P.VERM, mec=P.BLACK, mew=.3, clip_on=False, zorder=4)
        else:
            ax.errorbar(r.diff_vs_ref, yv, xerr=[[r.diff_vs_ref - r.diff_ci_lo], [r.diff_ci_hi - r.diff_vs_ref]], fmt="o",
                        ms=3, color=P.METHOD_COLOR.get(m, P.GREY), mec=P.BLACK, mew=.3, elinewidth=.8)
            tx(ax, r.diff_ci_hi + .006, yv, f"{r.diff_vs_ref:+.3f}".replace("-", "\u2212"), va="center",
               fontsize=fk.pt("annot"))
        tx(ax, cov_x, yv, f"{r.pit_central90:.2f}", va="center", ha="right", fontsize=fk.pt("annot"),
           fontweight="bold" if m == ref else "normal")
        rows.append(dict(method=m, label=P.METHOD_LABEL.get(m, m), n_records=int(r.n_records), d=r.diff_vs_ref,
                         lo=r.diff_ci_lo, hi=r.diff_ci_hi, coverage90=r.pit_central90))
    tx(ax, cov_x, len(order) - .35, "90% coverage", va="bottom", ha="right", fontsize=fk.pt("annot"))
    ax.axvline(0, color=P.BLACK, lw=.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([P.METHOD_SHORT[m] for m in order])
    ax.set_ylim(-.6, len(order) + .1)
    return pd.DataFrame(rows)


# ============================================================================== S2 calibration and model detail
def sf2(d):
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
        ax.spines["left"].set_bounds(0, 2)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, 1, 2])
        if k % 5:
            ax.tick_params(labelleft=False)
        if k < 5:
            ax.tick_params(labelbottom=False)
        if k == 0:
            fk.panel(ax, "a", P.METHOD_SHORT[m])
            ax._left_title.set_fontsize(fk.pt("tick"))
        else:
            ax.set_title(P.METHOD_SHORT[m], fontsize=fk.pt("tick"), pad=2, loc="left")
        rows += [dict(method=m, bin_lo=bins[i], density=h[i]) for i in range(10)]
    SRC["SFig 2a"] = pd.DataFrame(rows)
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
    ax.set_xlabel("earlier trials of the pair"); ax.set_ylabel("share in central 90% interval"); ax.set_ylim(.70, .97)
    ax.legend(frameon=False, loc="lower left")
    head(ax, "b", "Coverage by evidence")
    SRC["SFig 2b"] = pd.DataFrame(rows)
    ax = fig.add_subplot(r1[1])
    lab_ = pair[pair.on_fda]
    socs = sorted(lab_.groupby("soc_h").size().loc[lambda s: s >= 30].index)
    tau = par[par.kind == "tau"].set_index("soc").value
    pi0 = pair.groupby("soc_h").p_no_effect.mean()
    t = pd.DataFrame(dict(tau=tau.reindex(socs), pi0=pi0.reindex(socs))).sort_values("pi0")
    y = np.arange(len(t))
    ax.barh(y, t.pi0, .6, color=P.SKY)
    ax2 = ax.twiny(); ax2.plot(t.tau, y, "D", ms=2.6, color=P.VERM, mec=P.BLACK, mew=.3)
    ax.set_yticks(y); ax.set_yticklabels([P.SOC_SHORT[s] for s in t.index], fontsize=fk.pt("annot"))
    ax.set_xlabel("mean posterior probability of zero excess (bars)")
    ax2.set_xlabel("between-trial SD τ (diamonds)", fontsize=fk.pt("base"), color=P.VERM)
    ax2.tick_params(axis="x", colors=P.VERM, labelsize=fk.pt("tick"))
    head(ax, "c", "")
    SRC["SFig 2c"] = t.reset_index().rename(columns={"index": "organ_system"})
    ax = fig.add_subplot(r1[2])
    sm = d["simall"]; rows = []
    vers = ["AF 3.0", "AF 3.2", "AF 3.3"]
    vlab = {"AF 3.0": "hierarchical, common τ", "AF 3.2": "+ pair-specific τ", "AF 3.3": "+ zero-excess component"}
    for j, v in enumerate(vers):
        t = sm[sm.version == v].sort_values("scenario")
        ax.plot(np.arange(len(t)) + (j - 1) * .18, t.rmse_af3, "o", ms=3, color=[P.SKY, P.ORANGE, P.VERM][j], mec=P.BLACK,
                mew=.3, label=vlab[v])
        rows += t.assign(version=vlab[v])[["version", "scenario", "coverage_af3", "rmse_af3", "coverage_af1",
                                           "rmse_af1"]].to_dict("records")
    t = sm[sm.version == "AF 3.3"].sort_values("scenario")
    ax.plot(np.arange(len(t)) + .36, t.rmse_af1, "s", ms=3, color=P.GREY, mec=P.BLACK, mew=.3, label="Mantel–Haenszel")
    ax.set_xticks(range(len(t))); ax.set_xticklabels(["varying", "similar", "skewed"]); ax.set_xlabel("simulated effects")
    ax.set_ylabel("RMSE of log hazard ratio"); ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(.2, .56)
    head(ax, "d", "Known-truth simulation by model component")
    SRC["SFig 2d"] = pd.DataFrame(rows)
    return fig


# ============================================================================== S3 serious adverse events
def sf3(d):
    b = d["bench_all"]
    Sv = b[(b.bench == BENCH_S) & (b.win == "pooled") & (b.scoring == "hier") & ~b.method.str.contains("no dose|nocov")]
    fig = fk.figure("double", height=62, kind="supp")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0])
    t = Sv[Sv.stratum == "AF1-support"].set_index("method")
    SRC["SFig 3a"] = _forest(ax, t, cov_x=.2)
    ax.set_xlim(-.04, .2); ax.set_xticks([0, .05, .1]); ax.spines["bottom"].set_bounds(-.04, .12)
    ax.set_xlabel("log-score gain per prediction (95% CI)")
    head(ax, "a", f"Serious events, temporal validation set ({int(t.loc['AF 3.3', 'n_records']):,} predictions)")
    ax = fig.add_subplot(gs[1])
    strata = ["evidence 1 trial", "evidence 2-4", "evidence 5-9", "evidence >=10", "unseen"]; rows = []
    for j, m in enumerate(["B5 term-only", "B6 class-mean", "B0 null", "B3 Bayes-NN"]):
        tt = Sv[Sv.method == m].set_index("stratum").reindex(strata)
        x = np.arange(len(strata)) + (j - 1.5) * .18
        ok = tt.diff_vs_ref.notna().values
        ax.errorbar(x[ok], tt.diff_vs_ref[ok], yerr=[(tt.diff_vs_ref - tt.diff_ci_lo)[ok], (tt.diff_ci_hi - tt.diff_vs_ref)[ok]],
                    fmt="o", ms=2.6, color=P.METHOD_COLOR[m], mec=P.BLACK, mew=.3, elinewidth=.6, label=P.METHOD_SHORT[m])
        rows += [dict(comparator=m, stratum=s, **tt.loc[s, ["n_records", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi"]].to_dict())
                 for s in strata]
    ax.axhline(0, color=P.BLACK, lw=.5)
    ax.set_xticks(range(len(strata))); ax.set_xticklabels(["1", "2–4", "5–9", "≥10", "0 (new)"])
    ax.set_xlabel("earlier trials of the pair"); ax.set_ylabel("log-score gain of the model")
    ax.legend(frameon=False, loc="upper right"); ax.set_ylim(-.27, .5)
    head(ax, "b", "Serious events by evidence")
    SRC["SFig 3b"] = pd.DataFrame(rows)
    return fig


# ============================================================================== S4 hierarchical vs conventional
def sf4(d):
    pair, conv = d["pair"], d["conv"]
    x = pair[pair.af1_estimable].merge(conv, on=["drug", "term"], suffixes=("", "_mh"))
    fig = fk.figure("double", height=62, kind="supp")
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0])
    ax.scatter(x.af.clip(-1.5, 1), x.af_mean, s=1.2, c=np.log10(x.n_trials_mh), cmap=fk.cmap(), lw=0, rasterized=True)
    ax.plot([-1, 1], [-1, 1], color=P.BLACK, lw=.5, ls="--")
    rho = stats.spearmanr(x.af, x.af_mean).statistic
    ax.set_xlim(-1.5, 1); ax.set_ylim(-.8, 1); ax.set_xlabel("AF, Mantel–Haenszel"); ax.set_ylabel("AF, hierarchical model")
    tx(ax, .03, .97, f"{len(x):,} pairs\nSpearman ρ {rho:.2f}", transform=ax.transAxes, va="top", fontsize=fk.pt("annot"))
    head(ax, "a", "Same pairs, two estimators")
    SRC["SFig 4a"] = x[["drug", "term", "n_trials_mh", "af", "af_mean"]]
    ax = fig.add_subplot(gs[1])
    wmh = (np.log(1 - x.af_lo.clip(upper=.999)) - np.log(1 - x.af_hi.clip(upper=.999))).abs()
    whz = (np.log(1 - x.af_hz_lo.clip(upper=.999)) - np.log(1 - x.af_hz_hi.clip(upper=.999))).abs()
    q = pd.qcut(x.n_trials_mh, [0, .25, .5, .75, 1], duplicates="drop")
    t = pd.DataFrame(dict(q=q, mh=wmh, hz=whz)).groupby("q", observed=True).median()
    ax.plot(range(len(t)), t.mh, "-s", ms=3, color=P.GREY, label="Mantel–Haenszel 95% CI")
    ax.plot(range(len(t)), t.hz, "-o", ms=3, color=P.VERM, label="hierarchical 95% CrI")
    ax.set_xticks(range(len(t))); ax.set_xticklabels([f"{int(i.left) + 1}–{int(i.right)}" for i in t.index], fontsize=fk.pt("annot"))
    ax.set_xlabel("trials of the pair (quartiles)"); ax.set_ylabel("median interval width (log HR / log RR)")
    ax.legend(frameon=False)
    head(ax, "b", "Interval width")
    SRC["SFig 4b"] = t.reset_index().astype({"q": str})
    ax = fig.add_subplot(gs[2])
    lab = x[x.on_fda]
    vals = [("MH upper limit < 0.5", (lab.af_hi < .5).mean()),
            ("confidently background-dominant", (lab.p_af_lt_half > .975).mean()),
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
    ax.set_xlabel("trials of the pair"); ax.set_ylabel("AF difference (hierarchical − MH;\nclipped at ±1.5)")
    head(ax, "d", "Shrinkage fades with evidence")
    SRC["SFig 4d"] = pd.DataFrame(dict(n_trials=x.n_trials_mh, diff=shr))
    return fig


# ============================================================================== S7 development and component ablation
def sf7(d):
    b = d["bench_all"]
    fig = fk.figure("double", height=62, kind="supp")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    ax = fig.add_subplot(gs[0])
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
    ax.set_xticks(range(4)); ax.set_xticklabels(["2011–12", "2013–14", "2015–16", "2017–\n(validation)"])
    ax.set_ylabel("log-score gain of the model")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-.02, -.3))
    head(ax, "a", "Development windows and temporal validation")
    SRC["SFig 7a"] = pd.DataFrame(rows)
    ax = fig.add_subplot(gs[1])
    Dp = b[(b.bench == "bench21_dev33") & (b.win == "pooled") & (b.stratum == "AF1-support")]
    ms = ["AF 3.3", "AF 3.2", "B3 Bayes-NN", "B2 PM-t", "B4 AF 2.0", "B1 MH"]
    x = np.arange(len(ms)); rows = []
    for sc, off, alpha in (("jeffreys", -.17, .45), ("hier", .17, 1)):
        t = Dp[Dp.scoring == sc].set_index("method")
        ax.bar(x + off, [t.loc[m, "pit_low10"] for m in ms], .32, color=[P.METHOD_COLOR[m] for m in ms], alpha=alpha)
        rows += [dict(scoring=sc, method=m, pit_low10=t.loc[m, "pit_low10"]) for m in ms]
    ax.axhline(.10, color=P.BLACK, lw=.5, ls="--")
    ax.set_xticks(x); fk.tilt(ax, [P.METHOD_SHORT[m] for m in ms])
    ax.set_ylabel("share below 10th predictive percentile")
    ax.legend(handles=[mpl.patches.Patch(color=P.GREY, alpha=.45, label="own placebo arm"),
                       mpl.patches.Patch(color=P.GREY, label="hierarchical background")], loc="upper left")
    ax.set_ylim(0, .21)
    head(ax, "b", "Placebo risk of a new trial")
    SRC["SFig 7b"] = pd.DataFrame(rows)
    ax = fig.add_subplot(gs[2])
    C = b[(b.bench == M.BENCH_C) & (b.win == "pooled") & (b.scoring == "hier") & (b.stratum == "AF1-support")].set_index("method")
    steps = ["B1 MH", "B4 AF 2.0", "B3 Bayes-NN", "B7 class-Bayes", "AF 3.2 no dose offsets", "AF 3.2",
             "AF 3.3 no dose offsets", "AF 3.3"]
    labels = ["Mantel–Haenszel", "Empirical-Bayes shrinkage", "Bayesian RE MA", "Class-informed Bayes MA",
              "hierarchical, no dose, no zero-excess", "+ dose offsets", "+ zero-excess, no dose", "full model"]
    gains = [-C.loc[m, "diff_vs_ref"] + C.loc["B1 MH", "diff_vs_ref"] for m in steps]
    ax.bar(range(len(steps)), gains, .62, color=[P.METHOD_COLOR.get(m.replace(" no dose offsets", ""), P.GREY)
                                                for m in steps])
    for i, v in enumerate(gains):
        tx(ax, i, v + max(gains) * .02, f"{v:+.3f}", ha="center", va="bottom", rotation=90, fontsize=fk.pt("annot"))
    ax.set_xticks(range(len(steps))); fk.tilt(ax, labels)
    ax.set_ylabel("gain over MH\n(log score)"); ax.set_ylim(0, max(gains) * 1.45)
    head(ax, "c", "Component ablation")
    SRC["SFig 7c"] = pd.DataFrame(dict(step=steps, label=labels, gain_over_mh=gains))
    return fig


FIGS = {"sfig1_data_labels": M112.s1, "sfig2_calibration": sf2, "sfig3_serious": sf3, "sfig4_conventional": sf4,
        "sfig5_cards": M112.s5, "sfig6_legacy": M112.s6, "sfig7_development": sf7}


def main() -> int:
    only = sys.argv[1:] or [f for f, _ in P.SUPP_FIGS]
    with pv.connect(read_only=True) as con:
        d = M.load(con)
        d = M112.load_extra(con, d)
    OUT.mkdir(parents=True, exist_ok=True)
    made = {}
    for fid in only:
        made[fid] = fk.save(FIGS[fid](d), OUT / fid)
        print(fid, made[fid].report)
    if set(only) != {f for f, _ in P.SUPP_FIGS}:
        return 0
    # v2 sheet names of the reused panels -> v3 numbering
    ren = {"SFig 1": "SFig 1", "SFig 5": "SFig 5", "SFig 6": "SFig 6"}
    with pv.run_context(__file__, env_name="oajoint-py311", software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','afstd_card','afstd_label_match','af3_param','bench_score',
                   'bench_record','af3_sim','af_cell','af_split_estimate','af_external_ref','af_route_cell',
                   'af_indication_cell','af_dose_cell')""").fetchall()]
        p = ROOT / "manuscript" / "v3" / "source_data_supplementary.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            for key in sorted(SRC):
                if key.startswith(("SFig 2", "SFig 3", "SFig 4", "SFig 7")) or key[:6] in ren:
                    SRC[key].to_excel(w, sheet_name=key.replace(" ", "_")[:31], index=False)
        run.archive_document("source_data_supplementary_v3", p, kind="workbook", upstream=up)
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "bench_score", "af_cell"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
