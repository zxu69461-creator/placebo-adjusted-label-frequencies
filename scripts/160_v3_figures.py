#!/usr/bin/env python3
"""Main figures of the PLOS Medicine manuscript, read only from the project
DuckDB, with one Source Data workbook (manuscript/v3/source_data.xlsx). Same data and estimates as the earlier version
(111_fig_main.py, frozen with the Nature Communications submission); what changes is presentation:

  terminology  four posterior zones named as background-/drug-dominant (oajoint.paper_v3), AF < 0.5 = RR < 2
  Fig. 1       flat schematic whose symbols match the Methods; P(AF < 0.5) in b; every stratum share labelled in e
  Fig. 2       temporal validation set labelled as such (a); broken linear axis (b); coverage values in their own
               column (c); f shows drugs held out entirely (serious events move to Supplementary Fig. 7)
  Fig. 3       histogram from -1 with a separate, hatched overflow bin and reference lines at AF = 0 and 0.5 (a)
  Fig. 4       banded rows and spaced columns (a); readable drug-group names and contrast-chosen cell text (c);
               a twin right axis for the expected share (d)
  Fig. 5       P(AF < 0.5) and the nasal-spray origin of varenicline sneezing on the plot (a); boxed note (b);
               the label mock-up uses the four zones and aligned number columns (d)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import figures as F  # noqa: E402,F401
from oajoint import paper_v3 as P  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
from oajoint import schematics_v3 as SCH  # noqa: E402
import figkit as fk  # noqa: E402

_spec = importlib.util.spec_from_file_location("m111", ROOT / "scripts" / "111_fig_main.py")
M111 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M111)
load, S, tx = M111.load, M111.S, M111.tx

SRC = {}
BENCH_C, BENCH_N = M111.BENCH_C, M111.BENCH_N
ROWS_B = M111.ROWS_B
OUT = ROOT / P.FIG_DIR
MS3 = ROOT / "manuscript" / "v3"


def head(ax, letter, text):
    fk.panel(ax, letter, text)


def zone_legend(ax, labels=None, **k):
    labels = labels or P.ZONE_LABEL
    return ax.legend(handles=[mpl.patches.Patch(color=P.ZONE_COLOR[z], label=labels[z]) for z in P.ZONES],
                     frameon=False, **k)


def _lum(c):
    """WCAG relative luminance of a colour."""
    r, g, b = [((v / 12.92) if v <= .04045 else ((v + .055) / 1.055) ** 2.4) for v in mpl.colors.to_rgb(c)]
    return .2126 * r + .7152 * g + .0722 * b


def text_on(c):
    """Black or white, whichever has the higher contrast ratio on fill colour c."""
    L = _lum(c)
    return P.BLACK if (L + .05) / .05 >= 1.05 / (L + .05) else "white"


# ============================================================================== Figure 1
def fig1(d):
    pair, par = d["pair"], d["par"]
    fig = fk.figure("double", height=170)
    g0 = fig.add_gridspec(2, 1, height_ratios=[.46, 1])
    ax = fig.add_subplot(g0[0]); fk.art.place(ax, SCH.fig1a_art())
    head(ax, "a", "A hierarchical model of arm-level adverse-event counts")
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
        tx(ax, 0, y + 1.75, f"AF {r.af_mean:.2f} ({r.af_lo:.2f} to {r.af_hi:.2f}); P(AF < 0.5) {r.p_af_lt_half:.2f}"
           .replace("-", "−"), fontsize=fk.pt("annot"), color=fk.c.grey_dark, va="center")
        ticks += [y + .85, y]; tl += ["placebo", "drug"]
        rows.append(dict(drug=r.display_name, term=r.term, per100_placebo=pl, per100_added=add,
                         added_lo=r.per100_added_lo, added_hi=r.per100_added_hi, af=r.af_mean, af_lo=r.af_lo,
                         af_hi=r.af_hi, p_af_lt_half=r.p_af_lt_half, n_trials=r.n_trials))
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
          ("trial × drug × adverse-event records", S(d, "data_v2", "estimator", "all", "records"), P.BLUE),
          ("drug–adverse-event pairs estimated", S(d, "counts", "all", "all", "pairs"), P.BLUE),
          ("pairs listed on a current FDA label", S(d, "counts", "listed", "all", "pairs"), P.VERM)]
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
    # e -- evidence behind each pair: every stratum share is written, inside its segment when it fits; otherwise
    # the last segment is labelled just right of the bar and any other narrow segment above the bar on a short stem
    ax = fig.add_subplot(g1[1, 0])
    sets = [("all pairs", pair), ("on a label", pair[pair.on_fda]),
            ("on a label,\nconv. estimable", pair[pair.on_fda & pair.af1_estimable])]
    shades = fk.sem("oajoint")["ramp"]["blue4"]
    rows = []
    for i, (nm, x) in enumerate(sets):
        left, segs = 0, []
        for s, c in zip(P.STRATA, shades):
            f = float((x.evidence == s).mean())
            if f > 0:
                ax.barh(i, f, .56, left=left, color=c, ec="white", lw=.3)
                segs.append((left, f, c, f"{100 * f:.0f}%" if f >= .01 else f"{100 * f:.1f}%"))
            left += f
            rows.append(dict(set=nm.replace("\n", " "), stratum=s, share=f, n=int((x.evidence == s).sum())))
        for k, (l0, f, c, lab) in enumerate(segs):
            if f >= .14:
                tx(ax, l0 + f / 2, i, lab, ha="center", va="center", fontsize=fk.pt("annot"), color=text_on(c))
            elif k == len(segs) - 1:
                tx(ax, 1.015, i, lab, ha="left", va="center", fontsize=fk.pt("annot"))
            else:
                ax.plot([l0 + f / 2] * 2, [i - .29, i - .4], color=fk.c.grey_dark, lw=.4)
                tx(ax, l0 + f / 2, i - .41, lab, ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax.set_yticks(range(len(sets))); ax.set_yticklabels([f"{s[0]}\n({len(s[1]):,})" for s in sets])
    ax.set_ylim(len(sets) - .55, -.85)
    ax.set_xlim(0, 1.12); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("share of pairs (%)")
    ax.legend(handles=[mpl.patches.Patch(color=c, label=P.STRATUM_LABEL[s]) for s, c in zip(P.STRATA, shades)],
              title="trials per pair", title_fontsize=fk.pt("legend"), ncol=4, loc="lower center", bbox_to_anchor=(.45, 1.0),
              frameon=False, columnspacing=.7, handlelength=.9)
    head(ax, "e", "")
    SRC["Fig 1e"] = pd.DataFrame(rows)
    # f -- what changes the drug effect between trials
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
    # d -- dose slope by organ system
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
    fk.legend_top(ax, [mpl.lines.Line2D([], [], color=P.VERM, lw=1.2)],
                  [f"pooled {gm:.3f} (SE {gse:.3f})"], ncol=1, align="right")
    head(ax, "d", "Dose–response")
    SRC["Fig 1d"] = g[["soc", "value"]].rename(columns={"value": "dose_slope"})
    return fig


# ============================================================================== Figure 2
BREAK_LO, BREAK_HI, TAIL = .36, .95, .15      # Fig. 2b: linear to 0.36, gap, compressed linear tail from 0.95


def _brk_fwd(x):
    x = np.asarray(x, float)
    gap = BREAK_LO + (x - BREAK_LO) * .015
    tail = BREAK_LO + (BREAK_HI - BREAK_LO) * .015 + (x - BREAK_HI) * TAIL
    return np.where(x <= BREAK_LO, x, np.where(x < BREAK_HI, gap, tail))


def _brk_inv(f):
    f = np.asarray(f, float)
    f_gap = BREAK_LO + (BREAK_HI - BREAK_LO) * .015
    return np.where(f <= BREAK_LO, f, np.where(f < f_gap, BREAK_LO + (f - BREAK_LO) / .015, BREAK_HI + (f - f_gap) / TAIL))


def break_marks(ax, x0, x1):
    """Two short slashes on the x axis between data values x0 and x1 (an axis break), the spine masked between them."""
    tr = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_patch(mpl.patches.Rectangle((x0, -.02), x1 - x0, .04, transform=tr, color="white", clip_on=False, zorder=5,
                                       lw=0))
    for xm in (x0, x1):
        ax.plot([xm, xm], [-.025, .025], transform=tr, color=P.BLACK, lw=.6, clip_on=False, zorder=6)


def fig2(d):
    b, pair = d["bench"], d["pair"]
    C = b[(b.bench == BENCH_C) & (b.win == "pooled") & (b.scoring == "hier")]
    fig = fk.figure("double", height=110)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1.05, 1])
    r0 = g0[0].subgridspec(1, 4, width_ratios=[.9, 1.25, .55, .8])
    r1 = g0[1].subgridspec(1, 4, width_ratios=[1.2, 1.0, .8, .9])
    # a -- design: development windows and the temporal validation set
    ax = fig.add_subplot(r0[0])
    wins = [("W2011", 2011.0, 2013.0, 5.0), ("W2013", 2013.0, 2015.0, 4.0), ("W2015", 2015.0, 2017.09, 3.0),
            ("C2017", 2017.09, 2026.7, 1.0)]
    nwin = d["wins"].set_index("win").n
    for w, s0, s1, y in wins:
        col = P.VERM if w == "C2017" else P.BLUE
        ax.barh(y, s0 - 2000, .58, left=2000, color=P.LGREY)
        ax.barh(y, s1 - s0, .58, left=s0, color=col)
        if w == "C2017":
            tx(ax, s0 + (s1 - s0) / 2, y, f"{int(nwin[w]):,}", va="center", ha="center", fontsize=fk.pt("annot"),
               color="white")
        else:
            tx(ax, s1 + .3, y, f"{int(nwin[w]):,}", va="center", ha="left", fontsize=fk.pt("annot"))
    tx(ax, 2000.4, 5.62, "fit on all earlier trials", fontsize=fk.pt("annot"), va="center")
    tx(ax, 2000.4, 2.02, "temporal validation set", fontsize=fk.pt("annot"), va="center", color=P.VERM, fontweight="bold")
    tx(ax, 2000.4, 1.58, f"(from Feb 2017; n = {int(nwin['C2017']):,})", fontsize=fk.pt("annot"), va="center",
       color=P.VERM)
    ax.set_yticks([1, 3, 4, 5]); ax.set_yticklabels(["validation", "dev. 3", "dev. 2", "dev. 1"])
    ax.set_xlim(2000, 2027); ax.set_ylim(.5, 6.0); ax.set_xticks([2000, 2010, 2020]); ax.set_xlabel("trial start")
    head(ax, "a", "Predicting later trials")
    SRC["Fig 2a"] = pd.DataFrame([dict(window=w, start=s0, end=s1, test_trials=int(nwin[w])) for w, s0, s1, _ in wins])
    # b -- temporal validation: gain over each comparator (linear axis, broken between 0.32 and 1.2)
    c = C[C.stratum == "AF1-support"].set_index("method").loc[ROWS_B]
    order = ["AF 3.3"] + sorted([m for m in ROWS_B if m != "AF 3.3"], key=lambda m: c.loc[m, "diff_vs_ref"])
    top_m = c.diff_vs_ref.idxmax()
    assert c.loc[top_m, "diff_ci_lo"] > BREAK_HI and c.diff_ci_hi.drop(top_m).max() < BREAK_LO, \
        "Fig. 2b axis break does not fit the data"
    yy = np.arange(len(order))[::-1]
    ax = fig.add_subplot(r0[1])
    ax.set_xscale("function", functions=(_brk_fwd, _brk_inv))
    for yv, m in zip(yy, order):
        r = c.loc[m]
        if m == "AF 3.3":
            ax.plot(0, yv, "D", ms=3.2, color=P.VERM, mec=P.BLACK, mew=.3)
            tx(ax, .012, yv, "reference", va="center", fontsize=fk.pt("annot"), color=P.VERM)
            continue
        ax.errorbar(r.diff_vs_ref, yv, xerr=[[r.diff_vs_ref - r.diff_ci_lo], [r.diff_ci_hi - r.diff_vs_ref]], fmt="o",
                    ms=3, color=P.METHOD_COLOR.get(m, P.GREY), mec=P.BLACK, mew=.3, elinewidth=.8)
        if m == top_m:                    # beyond the break: the value is written above its point
            tx(ax, r.diff_vs_ref, yv + .5, f"+{r.diff_vs_ref:.3f}", ha="center", va="center", fontsize=fk.pt("annot"))
        elif yv == 1:                     # the row above it: value left of the interval, clear of the break
            tx(ax, r.diff_ci_lo - .008, yv, f"+{r.diff_vs_ref:.3f}", ha="right", va="center", fontsize=fk.pt("annot"))
        else:
            tx(ax, r.diff_ci_hi + .008, yv, f"+{r.diff_vs_ref:.3f}", va="center", fontsize=fk.pt("annot"))
    ax.axvline(0, color=P.BLACK, lw=.5)
    ax.set_xlim(-.008, 1.68)
    ax.set_xticks([0, .1, .2, .3, 1.3]); ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "1.3"])
    break_marks(ax, BREAK_LO + .004, BREAK_HI - .2)
    ax.set_yticks(yy); ax.set_yticklabels([P.METHOD_SHORT[m] for m in order]); ax.set_ylim(-.6, len(order) - .4)
    ax.set_xlabel("log-score gain per prediction (95% CI; broken axis)")
    head(ax, "b", f"Better than every alternative ({int(c.loc['AF 3.3', 'n_records']):,} predictions)")
    SRC["Fig 2b"] = c.reset_index()[["method", "n_records", "n_trials", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi"]].assign(
        label=lambda x: x.method.map(P.METHOD_LABEL))
    # c -- coverage; values in their own column, right of the target band
    ax = fig.add_subplot(r0[2], sharey=ax)
    ax.axvspan(.87, .93, color=P.LGREY, lw=0); ax.axvline(.9, color=P.BLACK, lw=.5, ls="--")
    ax.barh(yy, [c.loc[m, "pit_central90"] for m in order], .6, color=[P.METHOD_COLOR.get(m, P.GREY) for m in order])
    for yv, m in zip(yy, order):
        tx(ax, 1.045, yv, f"{c.loc[m, 'pit_central90']:.2f}", va="center", ha="right", fontsize=fk.pt("annot"),
           color=P.BLACK)
    ax.set_xlim(.7, 1.045); ax.set_xticks([.7, .8, .9]); ax.tick_params(axis="y", labelleft=False, length=0)
    ax.spines["bottom"].set_bounds(.7, .93)
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
    fk.legend_top(ax, ncol=1)
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
    # f -- drugs held out entirely: gain over every comparator that can predict a new drug, with 90% coverage
    ax = fig.add_subplot(r1[1])
    t = b[(b.bench == BENCH_N) & (b.win == "pooled") & (b.scoring == "hier") & (b.stratum == "unseen")].set_index("method")
    comps = [m for m in t.sort_values("diff_vs_ref", ascending=False).index if m != "AF 3.3"]
    order_f = comps + ["AF 3.3"]                       # bottom to top, reference on top as in b
    rows = []
    XC = 2.0
    for yv, m in enumerate(order_f):
        r = t.loc[m]
        if m == "AF 3.3":
            ax.plot(0, yv, "D", ms=3.2, color=P.VERM, mec=P.BLACK, mew=.3, clip_on=False, zorder=4)
            tx(ax, .07, yv, "reference", va="center", fontsize=fk.pt("annot"), color=P.VERM)
        else:
            ax.errorbar(r.diff_vs_ref, yv, xerr=[[r.diff_vs_ref - r.diff_ci_lo], [r.diff_ci_hi - r.diff_vs_ref]], fmt="o",
                        ms=3, color=P.METHOD_COLOR.get(m, P.GREY), mec=P.BLACK, mew=.3, elinewidth=.8)
            tx(ax, r.diff_ci_hi + .04, yv, f"+{r.diff_vs_ref:.3f}", va="center", fontsize=fk.pt("annot"))
        tx(ax, XC, yv, f"{r.pit_central90:.2f}", va="center", ha="right", fontsize=fk.pt("annot"),
           fontweight="bold" if m == "AF 3.3" else "normal")
        rows.append(dict(method=m, label=P.METHOD_LABEL[m], n_records=int(r.n_records), n_trials=int(r.n_trials),
                         d=r.diff_vs_ref, lo=r.diff_ci_lo, hi=r.diff_ci_hi, coverage90=r.pit_central90))
    tx(ax, XC, len(order_f) - .35, "90% coverage", va="bottom", ha="right", fontsize=fk.pt("annot"))
    ax.set_yticks(range(len(order_f))); ax.set_yticklabels([P.METHOD_SHORT[m] for m in order_f])
    ax.set_ylim(-.6, len(order_f) + .1); ax.set_xlim(0, XC); ax.set_xticks([0, .5, 1])
    ax.spines["bottom"].set_bounds(0, 1.1)
    ax.set_xlabel("log-score gain per prediction (95% CI)")
    head(ax, "f", "Drugs held out entirely (10%)")
    SRC["Fig 2f"] = pd.DataFrame(rows)
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
    ax.legend(loc="upper left", frameon=False)
    head(ax, "g", "Known truth")
    SRC["Fig 2g"] = s[["scenario", "coverage_af1", "coverage_af3", "rmse_af1", "rmse_af3"]]
    # h -- listed reactions have larger estimated effects
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
    ax.set_ylim(-.1, .75); ax.set_xlabel("trials of the pair (AUC)"); ax.set_ylabel("AF, posterior mean (median, IQR)")
    ax.legend(handles=[mpl.lines.Line2D([], [], color=P.VERM, lw=2.4, label="on the label"),
                       mpl.lines.Line2D([], [], color=P.GREY, lw=2.4, label="not on the label")],
              loc="upper left", frameon=False)
    head(ax, "h", "Label validity")
    SRC["Fig 2h"] = pd.DataFrame(rows)
    return fig


# ============================================================================== Figure 3
def fig3(d):
    pair = d["pair"]; lab_ = pair[pair.on_fda].copy()
    fig = fk.figure("double", height=115)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1.1, 1])
    r0 = g0[0].subgridspec(1, 2, width_ratios=[1.35, 1.55])
    r1 = g0[1].subgridspec(1, 2, width_ratios=[1.9, 1])
    # a -- distribution of posterior-mean AF from -1 to 1, by zone; values below -1 in a separate, hatched bin
    ax = fig.add_subplot(r0[0])
    LO, W = -1.0, .05
    bins = np.arange(LO, 1.0 + 1e-9, W)
    of_x, of_w = -1.40, .10                      # the overflow bin, separated from the axis range by a gap
    x = lab_.af_mean
    bottom, bottom_of = np.zeros(len(bins) - 1), 0
    hs, ofs = [], []
    for z in P.ZONES:
        m = lab_.zone == z
        h = np.histogram(x[m].clip(upper=.999), bins=bins)[0]; of = int((x[m] < LO).sum())
        share = S(d, "headline", "primary", "all", "zone_" + z)
        ax.bar(bins[:-1], h, width=W, align="edge", bottom=bottom, color=P.ZONE_COLOR[z], ec="white", lw=.2,
               label=f"{P.ZONE_LABEL[z]}: {100 * share:.0f}%")
        ax.bar(of_x, of, width=of_w, align="edge", bottom=bottom_of, color=P.ZONE_COLOR[z], ec=P.BLACK, lw=.2,
               hatch="////")
        bottom += h; bottom_of += of
        hs.append(h); ofs.append(of)
    ax.axvline(0, color=fk.c.grey_dark, lw=.6, ls="--")
    ax.axvline(.5, color=P.BLACK, lw=.6, ls="--")
    tx(ax, of_x + of_w / 2, bottom_of + 60, f"{bottom_of}", ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax.set_xlim(of_x - .03, 1.0)
    ax.set_xticks([of_x + of_w / 2, -1, -.5, 0, .5, 1]); ax.set_xticklabels(["<−1", "−1", "−0.5", "0", "0.5", "1"])
    ax.spines["bottom"].set_bounds(LO, 1.0)
    ax.plot([of_x, of_x + of_w], [0, 0], color=P.BLACK, lw=.8, clip_on=False)
    ax.set_ylim(0, 1.08 * bottom.max())
    ax.set_xlabel("AF, posterior mean (hatched: below −1)")
    ax.set_ylabel("listed adverse reactions")
    fk.legend_top(ax, ncol=1, title="share in each zone (classified by P(AF < 0.5));\n"
                  "dashed lines, AF = 0 (no excess) and AF = 0.5 (relative risk 2)",
                  title_fontsize=fk.pt("annot"), alignment="left")
    head(ax, "a", f"{int(len(lab_)):,} adverse reactions on current labels of {lab_.drug.nunique():,} drugs")
    SRC["Fig 3a"] = pd.DataFrame({"bin_lo": list(bins[:-1]) + ["< -1 (overflow)"],
                                  **{z: list(h) + [o] for z, h, o in zip(P.ZONES, hs, ofs)}})
    # b -- every listed reaction with its interval
    ax = fig.add_subplot(r0[1])
    o = lab_.sort_values(["af_mean", "drug", "term"]).reset_index(drop=True)
    xs = np.arange(len(o))
    for z in P.ZONES:
        m = (o.zone == z).values
        ax.vlines(xs[m], o.af_hz_lo.values[m].clip(-1), o.af_hz_hi.values[m], color=P.ZONE_COLOR[z], lw=.12, alpha=.6,
                  rasterized=True)
    ax.plot(xs, o.af_mean, color=P.BLACK, lw=.7)
    ax.axhline(.5, color=P.BLACK, lw=.5, ls="--"); ax.axhline(0, color=fk.c.grey_dark, lw=.5, ls="--")
    ax.set_xlim(0, len(o)); ax.set_ylim(-1, 1)
    ax.set_xlabel("listed adverse reactions, ranked by AF")
    ax.set_ylabel("AF (95% credible interval)")
    fk.legend_top(ax, [mpl.patches.Patch(color=P.ZONE_COLOR[z]) for z in P.ZONES], [P.ZONE_LABEL[z] for z in P.ZONES],
                  ncol=2)
    head(ax, "b", "Every listed reaction: posterior mean (line) and 95% credible interval")
    SRC["Fig 3b"] = o[["drug", "term", "n_trials", "af_mean", "af_hz_lo", "af_hz_hi", "p_af_lt_half", "zone"]]
    # c -- headline by evidence
    ax = fig.add_subplot(r1[0])
    grps = ["all"] + P.STRATA
    mets = [("exp_lt_half", "expected share with AF < 0.5 (95% CI)", P.BLUE),
            ("conf_bg", "confidently background-dominant", fk.c["blue@s0.35"]),
            ("exp_no_effect", "mean posterior probability of zero excess", P.SKY),
            ("conf_drug", "confidently drug-dominant", P.VERM)]
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
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("listed reactions (%)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    head(ax, "c", "The expected share is stable across evidence; confidence grows with it")
    SRC["Fig 3c"] = pd.DataFrame(rows)
    # d -- sensitivity
    ax = fig.add_subplot(r1[1])
    sens = [("primary", "headline", "all listed reactions"),
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
    ax.set_yticks(range(-1, len(sens))); ax.set_yticklabels(["Mantel–Haenszel"] + [s[2] for s in sens][::-1],
                                                            fontsize=fk.pt("annot"))
    ax.set_ylim(-1.6, top + 1.1)
    ax.set_xlim(0, 1.06); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("listed reactions background-dominant (%)")
    head(ax, "d", "Sensitivity analyses")
    SRC["Fig 3d"] = pd.DataFrame(rows)
    return fig


# ============================================================================== Figure 4
def fig4(d):
    """a organ systems (banded rows, three spaced number columns); b drug group x organ system, sharing a's rows;
    c drug classes; d common complaints (AF left axis, expected share on a twin right axis)."""
    pair, s = d["pair"], d["sum"]
    lab_ = pair[pair.on_fda]
    fig = fk.figure("double", height=140)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.2, 1.0], height_ratios=[1, 1])
    # a -- organ systems
    ax = fig.add_subplot(gs[:, 0]); axa = ax
    bs = s[s.analysis == "by_soc"]
    ex = bs[(bs.set_name == "primary") & (bs.metric == "exp_lt_half")].set_index("grp")
    tab = pd.DataFrame({"exp_lt_half": ex.value, "lo": ex.lo, "hi": ex.hi, "n": ex.n_pairs,
                        "conf_bg": bs[(bs.set_name == "primary") & (bs.metric == "conf_bg")].set_index("grp").value,
                        "pi0": bs[bs.metric == "pi0"].set_index("grp").value}).sort_values("exp_lt_half")
    XR = 2.84
    for i in range(len(tab)):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color=fk.c.grey_faint, lw=0, zorder=0)
    rows = []
    for i, (soc, r) in enumerate(tab.iterrows()):
        v = lab_[lab_.soc_h == soc].af_mean
        q = v.quantile([.1, .25, .5, .75, .9])
        ax.plot([q[.1], q[.9]], [i, i], color=fk.c.grey, lw=.8)
        ax.plot([q[.25], q[.75]], [i, i], color=fk.c["blue@t0.6"], lw=3, solid_capstyle="butt")
        ax.plot(q[.5], i, "|", ms=4, color=P.BLACK, mew=.8)
        rows.append(dict(organ_system=soc, n=int(r.n), af_p10=q[.1], af_q25=q[.25], af_median=q[.5], af_q75=q[.75],
                         af_p90=q[.9], exp_lt_half=r.exp_lt_half, lo=r.lo, hi=r.hi, conf_bg=r.conf_bg, pi0=r.pi0))
    ax.axvline(0, color=P.BLACK, lw=.4, ymax=(len(tab) - .5 + .6) / (len(tab) + 3.2))
    ax.axvline(.5, color=P.BLACK, lw=.4, ls="--", ymax=(len(tab) - .5 + .6) / (len(tab) + 3.2))
    ax.set_xlim(-.32, XR); ax.set_xticks([0, .5], ["0", "0.5"])
    ax.spines["bottom"].set_bounds(-.32, .75)
    cols = [(1.07, "exp_lt_half", ("Expected", "AF < 0.5")), (1.71, "conf_bg", ("Confident", "AF < 0.5")),
            (2.36, "pi0", ("Zero", "excess"))]
    for x0, key, (h1, h2) in cols:
        for i, (soc, r) in enumerate(tab.iterrows()):
            tx(ax, x0, i, f"{100 * r[key]:.0f}", ha="center", va="center", fontsize=fk.pt("annot"), color=P.BLACK)
        tx(ax, x0, len(tab) + .05, h2, ha="center", va="bottom", fontsize=fk.pt("annot"), fontweight="bold")
        tx(ax, x0, len(tab) + .95, h1, ha="center", va="bottom", fontsize=fk.pt("annot"), fontweight="bold")
    ax.plot([.8, XR - .2], [len(tab) + 2.05] * 2, color=P.BLACK, lw=.4)
    tx(ax, 1.71, len(tab) + 2.2, "% of listed reactions", ha="center", va="bottom", fontsize=fk.pt("annot"))
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{P.SOC_SHORT[x]} ({int(n):,})" for x, n in zip(tab.index, tab.n)], fontsize=fk.pt("annot"))
    ax.set_ylim(-.6, len(tab) + 3.2)
    ax.set_xlabel("AF of listed reactions, posterior mean\n(median, IQR, 10th–90th centile)")
    head(ax, "a", "By organ system")
    SRC["Fig 4a"] = pd.DataFrame(rows)
    # b -- drug group x organ system, on a's rows; full group names, cell text chosen for contrast
    ax = fig.add_subplot(gs[:, 1], sharey=axa)
    h = s[(s.analysis == "by_class") & (s.set_name == "atc1_soc")].copy()
    h[["atc1", "atc1_name", "soc"]] = h.grp.str.split("|", expand=True)
    socs = list(tab.index)
    assert set(h.soc) <= set(socs)
    a1s = sorted(h.atc1.unique(), key=lambda a: -h[h.atc1 == a].n_pairs.sum())
    M = np.full((len(socs), len(a1s)), np.nan)
    for r in h.itertuples():
        M[socs.index(r.soc), a1s.index(r.atc1)] = r.value
    cmap = fk.cmap(); norm = mpl.colors.Normalize(.4, 1)
    ax.add_patch(mpl.patches.Rectangle((-.5, -.5), len(a1s), len(socs), color=fk.c.grey_faint, lw=0, zorder=0))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto", origin="lower",
                   extent=(-.5, len(a1s) - .5, -.5, len(socs) - .5), zorder=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                tx(ax, j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=fk.pt("fine"),
                   color=text_on(cmap(norm(M[i, j]))), zorder=2)
    for j, a in enumerate(a1s):                  # colour key of the drug groups (used in c)
        ax.add_patch(mpl.patches.Rectangle((j - .38, len(socs) - .25), .76, .55, color=P.ATC1_COLOR.get(a, P.GREY),
                                           lw=0))
    ax.set_xlim(-.5, len(a1s) - .5)
    ax.set_xticks(range(len(a1s)))
    ax.set_xticklabels([P.ATC1_AXIS[a] for a in a1s], rotation=90, fontsize=fk.pt("annot"))
    ax.tick_params(axis="y", labelleft=False, length=0); ax.tick_params(axis="x", length=0)
    ax.set_xlabel("drug group (ATC level 1)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([.12, (len(socs) + 1.3) / (len(socs) + 3.8), .76, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([.4, .6, .8, 1]); cb.set_ticklabels(["40", "60", "80", "100"])
    cb.ax.tick_params(labelsize=fk.pt("tick"), length=1.5, pad=1); cb.ax.xaxis.set_ticks_position("top")
    cb.outline.set_linewidth(.3)
    head(ax, "b", "Drug group × organ system (expected share, %)")
    SRC["Fig 4b"] = h[["atc1", "atc1_name", "soc", "n_pairs", "value"]]
    # c -- drug classes
    ax = fig.add_subplot(gs[0, 2])
    bc = s[(s.analysis == "by_class") & (s.set_name == "primary") & (s.grp != "coverage")]
    ct = bc.pivot_table(index="grp", columns="metric", values="value")
    ct["n"] = bc[bc.metric == "exp_lt_half"].set_index("grp").n_pairs
    ct = ct.sort_values(["exp_lt_half"]).reset_index()
    ct["atc1"] = ct.grp.str[0]
    for a1, g in ct.groupby("atc1"):
        ax.scatter(g.index, g.exp_lt_half, s=np.sqrt(g.n) * 1.1, color=P.ATC1_COLOR.get(a1, P.GREY), ec=P.BLACK, lw=.2,
                   zorder=3, label=a1)
    short = lambda g: (lambda n: n if len(n) <= 20 else n[:19].rstrip() + "…")(g.split("|")[1].lower())  # noqa: E731
    lo_, hi_ = ct.iloc[0], ct.iloc[-1]
    for (xv, yv, txt, xyt, ha) in ((0, lo_.exp_lt_half, f"least background:\n{short(lo_.grp)} ({100 * lo_.exp_lt_half:.0f}%)",
                                    (.14, .1), "left"),
                                   (len(ct) - 1, hi_.exp_lt_half, f"most background:\n{short(hi_.grp)} ({100 * hi_.exp_lt_half:.0f}%)",
                                    (.97, .42), "right")):
        a_ = ax.annotate(txt, xy=(xv, yv), xytext=xyt, textcoords="axes fraction", ha=ha, va="center",
                         fontsize=fk.pt("annot"), arrowprops=dict(arrowstyle="-", lw=.4, color=fk.c.grey))
        a_.set_in_layout(False)
    ax.axhline(S(d, "headline", "primary", "all", "exp_lt_half"), color=P.BLACK, lw=.5, ls="--")
    ax.set_xlabel(f"{len(ct)} drug classes (ATC level 3), ranked")
    ax.set_ylabel("expected share with AF < 0.5 (%)"); ax.set_xticks([]); ax.set_ylim(.1, 1.08)
    ax.set_yticks([.2, .4, .6, .8, 1]); ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.spines["left"].set_bounds(.1, 1.0)
    head(ax, "c", "By drug class")
    SRC["Fig 4c"] = ct
    # d -- common complaints: AF on the left axis, expected share on a twin right axis
    ax = fig.add_subplot(gs[1, 2])
    x = np.log10(lab_.per100_pl.clip(.05, 100)); y = lab_.af_mean.clip(-.5, 1)
    ax.hexbin(x, y, gridsize=(26, 18), cmap="Greys", mincnt=1, bins="log", lw=0, extent=(-1.3, 2, -.5, 1))
    edges = np.quantile(x, np.linspace(0, 1, 11))
    mids, med, fr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        mids.append(float(x[m].median())); med.append(float(y[m].median())); fr.append(float(lab_.p_af_lt_half[m].mean()))
    h1, = ax.plot(mids, med, "-o", ms=2.4, color=P.VERM, lw=.9, label="median AF (left)")
    ax.set_xticks([-1, 0, 1, 2]); ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("reports per 100 on placebo"); ax.set_ylabel("AF, posterior mean")
    ax.set_ylim(-.5, 1)
    ax2 = ax.twinx()
    h2, = ax2.plot(mids, 100 * np.array(fr), "-s", ms=2.4, color=P.BLUE, lw=.9, label="expected share AF < 0.5 (right)")
    ax2.set_ylim(0, 100); ax2.set_ylabel("expected share with AF < 0.5 (%)", color=P.BLUE)
    ax2.tick_params(axis="y", colors=P.BLUE); ax2.spines["right"].set_color(P.BLUE); ax2.spines["right"].set_visible(True)
    fk.legend_top(ax, [h1, h2], [h1.get_label(), h2.get_label()], ncol=1)
    head(ax, "d", "Common complaints are background-dominant")
    SRC["Fig 4d"] = pd.DataFrame(dict(log10_placebo_per100_mid=mids, median_af=med, exp_share_af_lt_half=fr))
    return fig


# ============================================================================== Figure 5
def fig5(d):
    pair, card = d["pair"], d["card"]
    card = card[card.card_set == "main"].merge(
        pair[["drug", "term", "per100_drug_exp", "per100_added_exp", "af_mean"]], on=["drug", "term"])
    fig = fk.figure("double", height=170)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 1])
    ga = g0[0].subgridspec(3, 3, height_ratios=[1, 1, .07])
    gb = g0[1].subgridspec(1, 3, width_ratios=[1.0, .85, 1.65])
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
        tx(ax, 1.0, len(g) - .35, "P(AF < 0.5)", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
           fontsize=fk.pt("annot"))
        dname = g.display_name.iloc[0]
        terms = [f"{t} (nasal spray)" if (dname == "varenicline" and t == "sneezing") else t for t in g.term]
        ax.set_yticks(y); ax.set_yticklabels(terms, fontsize=fk.pt("annot"))
        fk.fit_ticklabels(ax, "y")
        ax.set_xlim(0, xm * 1.2); ax.set_ylim(-.8, len(g) + .1)
        if k >= 3:
            ax.set_xlabel("reports per 100 patients")
        if k == 0:
            head(ax, "a", dname)
        else:
            ax.set_title(dname, loc="left", fontsize=fk.pt("title"), pad=3)
        for r, tl in zip(g.itertuples(), terms):
            rows.append(dict(drug=r.display_name, term=r.term, label_on_plot=tl, n_trials=r.n_trials,
                             per100_placebo=r.per100_pl, per100_added=r.per100_added_exp, added_lo=r.per100_added_lo,
                             added_hi=r.per100_added_hi, added_pred_lo=r.per100_added_pred_lo,
                             added_pred_hi=r.per100_added_pred_hi, af=r.af_mean, p_af_lt_half=r.p_af_lt_half,
                             zone=P.ZONE_LABEL[r.zone]))
    lg = fig.add_subplot(ga[2, :]); lg.axis("off")
    lg.legend(handles=[mpl.patches.Patch(color=P.LGREY, label="would occur on placebo"),
                       mpl.patches.Patch(color=P.VERM, label="added by the drug (95% credible interval)"),
                       mpl.lines.Line2D([], [], color=P.VERM, lw=.5, label="95% interval for a new trial"),
                       mpl.lines.Line2D([], [], ls="", marker="$\\%$", color=P.BLACK,
                                        label="P(AF < 0.5), coloured by zone")],
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
        xl_ = 3.4 if f == .5 else 1.15
        tx(ax, xl_, f * xl_ * 1.45, f"AF {f:g}", fontsize=fk.pt("annot"), ha="left", va="bottom",
           bbox=dict(boxstyle="square,pad=0.08", fc=fk.c.white, ec="none", alpha=.85))
    ax.axhline(1, color=P.VERM, lw=.5, ls="-.")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.8, 100); ax.set_ylim(.01, 100)
    ax.set_xlabel("reports per 100 on the drug (label frequency)"); ax.set_ylabel("reports per 100 added by the drug")
    share = S(d, "absolute_v2", "primary", "all", "share_added_exp_lt1")
    tx(ax, .97, .03, f"{100 * share:.0f}% of listed reactions:\n<1 added per 100 (below dash-dot line)",
       transform=ax.transAxes, ha="right", va="bottom", fontsize=fk.pt("annot"), color=P.BLACK, zorder=10,
       bbox=dict(boxstyle="round,pad=0.35", fc=fk.c.white, ec=fk.c.grey, lw=.4, alpha=.95))
    zone_legend(ax, loc="upper left", markerscale=1)
    head(ax, "b", "Label frequency overstates what the drug adds")
    SRC["Fig 5b"] = lab_[["drug", "term", "per100_drug_exp", "per100_pl", "per100_added_exp", "zone"]]
    # c -- listing rules
    ax = fig.add_subplot(gb[1])
    rules = [("rule_a_v2", "≥2% and\nabove placebo"), ("rule_b_v2", "evidence\nof excess"),
             ("rule_c_v2", "evidence and\n≥1 added\nper 100")]
    mets = [("retained", "current entries kept", P.GREY),
            ("bg_share_retained", "kept entries confidently\nbackground-dominant", P.BLUE),
            ("drug_caused_retained", "confidently drug-dominant\nentries kept", P.VERM)]
    rows = []
    for j, (m, nm, col) in enumerate(mets):
        xx = np.arange(len(rules)) + (j - 1) * .26
        v = [S(d, "listing_rules_v2", r, "all", m) for r, _ in rules]
        ax.bar(xx, v, .25, color=col, label=nm)
        for xi, vi in zip(xx, v):
            tx(ax, xi, vi + .02, f"{100 * vi:.0f}", ha="center", fontsize=fk.pt("annot"))
        rows += [dict(rule=r, metric=m, value=vv) for (r, _), vv in zip(rules, v)]
    cur = S(d, "listing_rules_v2", "current", "all", "bg_share_retained")
    ax.axhline(cur, color=P.BLUE, lw=.5, ls="--",
               label=f"current labels: {100 * cur:.0f}% confidently\nbackground-dominant")
    ax.set_xticks(range(len(rules))); ax.set_xticklabels([r[1] for r in rules], fontsize=fk.pt("annot"))
    ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("share (%)")
    fk.legend_top(ax, ncol=1)
    head(ax, "c", "Listing rules on the full trial record")
    SRC["Fig 5c"] = pd.DataFrame(rows)
    # d -- a proposed label block (illustration of real rows; figkit.art)
    ax = fig.add_subplot(gb[2])
    show = card[card.rank_drug == 1].sort_values("rank_term")
    rows = [dict(term=r.term, drug=r.per100_drug_exp, placebo=r.per100_pl, added=r.per100_added_exp,
                 lo=r.per100_added_lo, hi=r.per100_added_hi, zone=r.zone) for r in show.itertuples()]
    fk.art.place(ax, SCH.fig5d_art(show.display_name.iloc[0], rows, P.ZONE_COLOR))
    head(ax, "d", "A label block that reports what the drug adds")
    SRC["Fig 5d"] = show[["display_name", "term", "per100_drug_exp", "per100_pl", "per100_added_exp", "per100_added_lo",
                          "per100_added_hi", "zone"]].assign(zone=lambda x: x.zone.map(P.ZONE_LABEL))
    return fig


FIGS = {"fig1_model_data": fig1, "fig2_validation": fig2, "fig3_distribution": fig3, "fig4_heterogeneity": fig4,
        "fig5_translation": fig5}


def main() -> int:
    only = sys.argv[1:] or [f for f, _ in P.MAIN_FIGS]
    with pv.connect(read_only=True) as con:
        d = load(con)
    assert d["gate"].get("afstd_label_match") and d["gate"].get("af33_adopt"), "prerequisite analyses not completed"
    OUT.mkdir(parents=True, exist_ok=True); MS3.mkdir(parents=True, exist_ok=True)
    made = {}
    for fid in only:
        made[fid] = fk.save(FIGS[fid](d), OUT / fid)      # strict: raises on any rule violation
        print(fid, made[fid].report)
    if set(only) != {f for f, _ in P.MAIN_FIGS}:
        return 0                                          # partial runs are drafts: nothing is archived
    with pv.run_context(__file__, env_name="oajoint-py311", software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','afstd_card','af3_param','bench_score','bench_record',
                   'af3_sim','ctgov_trial','atlas_contrast')""").fetchall()]
        p = MS3 / "source_data.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            for key in sorted(SRC):
                SRC[key].to_excel(w, sheet_name=key.replace(" ", "_"), index=False)
        run.archive_document("source_data_v3", p, kind="workbook", upstream=up)
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "bench_score"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
