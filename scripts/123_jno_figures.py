#!/usr/bin/env python3
"""Main figures of the JAMA Network Open manuscript, read only from the project DuckDB (figkit profile 'jama').

Figure 1  study flow (STROBE-style), drawn with figkit.art
Figure 2  validation: prediction of later trials (A), calibration (B), agreement with individual participant data (C)
Figure 3  primary outcome: distribution (A), placebo frequency (B), sensitivity analyses (C), organ systems (D)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import jama as J  # noqa: E402
from oajoint import paper as P  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
import figkit as fk  # noqa: E402

BENCH_C = "bench22_confirm33"
SEED = 20260927


def tx(ax, *a, **k):
    t = ax.text(*a, **k); t.set_in_layout(False); return t


def load(con):
    d = {}
    d["pair"] = con.execute("""SELECT p.drug, p.term, p.soc_h, p.n_trials, p.evidence, p.on_fda, p.has_fda_label,
                                      p.p_af_lt_half, p.p0_own, p.per100_pl, e.af_mean, e.per100_pl_obs
                               FROM afstd_pair p JOIN afstd_pair_exp e USING (drug, term) ORDER BY drug, term""").df()
    d["sum"] = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_summary").df()
    d["jsum"] = con.execute("SELECT * EXCLUDE (run_id) FROM jno_summary").df()
    d["bench"] = con.execute(f"""SELECT method, n_records, diff_vs_ref, diff_ci_lo, diff_ci_hi, pit_central90
                                 FROM bench_score WHERE bench = '{BENCH_C}' AND win = 'pooled' AND scoring = 'hier'
                                 AND stratum = 'AF1-support'""").df()
    d["ctt"] = con.execute("""SELECT * EXCLUDE (run_id) FROM jno_ext_concordance WHERE benchmark = 'CTT'
                              AND our_loghr IS NOT NULL""").df()
    d["ace"] = con.execute("""SELECT * EXCLUDE (run_id) FROM jno_ext_concordance
                              WHERE benchmark LIKE 'ACEI_cough%'""").df().iloc[0]
    d["half"] = [x[0] for x in con.execute("""SELECT avg(v.p_af_lt_half) FROM jno_cell_variant v
                    JOIN afstd_pair p USING (drug, term) WHERE p.on_fda AND variant LIKE 'half:%' GROUP BY variant""").fetchall()]
    d["gate"] = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT gate_name, passed, metric_detail FROM prov_decision").fetchall()}
    q = lambda s: con.execute(s).fetchone()  # noqa: E731
    d["n"] = dict(ctgov=q("SELECT count(*) FROM ctgov_trial")[0],
                  pc=q("SELECT count(DISTINCT nct), count(*) FROM atlas_contrast"),
                  single=q("SELECT count(DISTINCT nct), count(*) FROM atlas_contrast WHERE single_ingredient"),
                  pairs=q("SELECT count(*), count(DISTINCT drug), count(DISTINCT term) FROM afstd_pair"),
                  nolabel=q("SELECT count(*), count(DISTINCT drug) FROM afstd_pair WHERE NOT has_fda_label"),
                  notlisted=q("SELECT count(*) FROM afstd_pair WHERE has_fda_label AND NOT on_fda")[0],
                  listed=q("SELECT count(*), count(DISTINCT drug) FROM afstd_pair WHERE on_fda"),
                  trial_sec=q("SELECT count(*) FROM jno_label_section WHERE section_class = 'trial'")[0])
    t = d["jsum"]
    d["n"]["trials"] = int(t[(t.analysis == "jno_trial_characteristics") & (t.metric == "n_trials")].value.iloc[0])
    d["n"]["records"] = int(t[(t.analysis == "jno_trial_characteristics") & (t.metric == "n_records")].value.iloc[0])
    import json as _j
    d["n"]["complete"] = int(_j.loads(d["gate"]["jno_complete_reporting"][1])["n_trials"])
    return d


def S(d, analysis, set_name, grp, metric, table="sum"):
    s = d[table]
    r = s[(s.analysis == analysis) & (s.set_name == set_name) & (s.grp == grp) & (s.metric == metric)]
    assert len(r) == 1, (analysis, set_name, grp, metric, len(r))
    return r.iloc[0]


# ============================================================================== Figure 1
def fig1(d):
    n, f = d["n"], J.fmt_n
    fig = fk.figure("double", height=105, profile=J.FIG_PROFILE)
    ax = fig.add_subplot(1, 1, 1)
    W, H = 158.0, 101.0
    A = fk.art.Art(W, H)
    mx, mw, ex, ew = 1.0, 86.0, 96.0, 61.0
    main = [
        [f"{f(n['ctgov'])} Randomized interventional studies with posted results",
         "on ClinicalTrials.gov (retrieved September 24, 2026)"],
        [f"{f(n['pc'][0])} Placebo-controlled drug trials",
         f"({f(n['pc'][1])} drug-placebo comparisons)"],
        [f"{f(n['single'][0])} Trials with single-ingredient comparisons",
         f"({f(n['single'][1])} comparisons)"],
        [f"{f(n['trials'])} Trials with arm-level nonserious adverse event counts",
         f"({f(n['records'])} trial-drug-adverse event records)"],
        [f"{f(n['pairs'][0])} Drug-adverse event pairs estimated by the model",
         f"({f(n['pairs'][1])} drugs; {f(n['pairs'][2])} MedDRA preferred terms)"],
        [f"{f(n['listed'][0])} Adverse reactions listed on current FDA labels",
         f"of {f(n['listed'][1])} drugs (primary analysis set)",
         f"{f(n['trial_sec'])} in the clinical trials experience subsection"],
    ]
    excl = [
        [f"{f(n['ctgov'] - n['pc'][0])} Excluded: no placebo arm or no", "drug arm mapped to a ChEMBL drug"],
        [f"{f(n['pc'][0] - n['single'][0])} Excluded: drug combinations only"],
        [f"{f(n['single'][0] - n['trials'])} Excluded: no nonserious adverse", "event listed in all drug and placebo arms"],
        ["Secondary analysis set:", f"{f(n['complete'])} Trials posting every nonserious event"],
        ["Not listed on a current FDA label:", f"{f(n['nolabel'][0])} Pairs of {f(n['nolabel'][1])} drugs without a label",
         f"{f(n['notlisted'])} Pairs not on the drug's label"],
    ]
    gap, y, pad = 6.4, 1.0, 3.6
    lh = fk.pt("annot") * fk.art.MM_PER_PT * 1.25
    tops, bots = [], []
    for i, lines in enumerate(main):
        bh = len(lines) * lh + pad
        A.rect(mx, y, mw, bh, r=0.8, fill="white" if i < len(main) - 1 else "blue@t0.8",
               stroke="grey_dark" if i < len(main) - 1 else "blue", sw=A.lw)
        y0 = y + bh / 2 - (len(lines) - 1) * lh / 2
        for k, s_ in enumerate(lines):
            A.text(mx + 2.5, y0 + k * lh, s_, role="annot", bold=(k == 0 and i == len(main) - 1))
        tops.append(y); bots.append(y + bh)
        if i < len(main) - 1:
            A.line(mx + mw / 2, y + bh, mx + mw / 2, y + bh + gap, color="grey_dark", sw=A.lw)
            A.path(f"M{mx + mw / 2 - 1.0},{y + bh + gap - 1.6} L{mx + mw / 2},{y + bh + gap} "
                   f"L{mx + mw / 2 + 1.0},{y + bh + gap - 1.6} Z", fill="grey_dark", stroke="none")
        y += bh + gap
    for i, lines in enumerate(excl):
        if lines is None:
            continue
        cy = (bots[i] + tops[i + 1]) / 2
        hh = len(lines) * lh + pad
        A.rect(ex, cy - hh / 2, ew, hh, r=0.8, fill="white", stroke="grey_dark", sw=A.lw)
        y0 = cy - (len(lines) - 1) * lh / 2
        for k, s_ in enumerate(lines):
            A.text(ex + 2.2, y0 + k * lh, s_, role="annot")
        A.line(mx + mw / 2, cy, ex - 0.2, cy, color="grey_dark", sw=A.lw)
        A.path(f"M{ex - 1.7},{cy - 1.0} L{ex - 0.1},{cy} L{ex - 1.7},{cy + 1.0} Z", fill="grey_dark", stroke="none")
    assert y - gap <= H, (y, H)
    probs = A.problems()
    assert not probs, probs
    fk.art.place(ax, A)
    return fig


# ============================================================================== Figure 2
def fig2(d):
    b = d["bench"].set_index("method")
    fig = fk.figure("double", height=165, profile=J.FIG_PROFILE)
    g0 = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    r0 = g0[0].subgridspec(1, 2, width_ratios=[1.45, .55])
    r1 = g0[1].subgridspec(1, 2, width_ratios=[1, 1])
    # A -- gain in log score over each comparator, confirmation set
    order = sorted(J.COMPARATORS, key=lambda m: b.loc[m, "diff_vs_ref"])
    yy = np.arange(len(order))[::-1]
    ax = fig.add_subplot(r0[0])
    for yv, m in zip(yy, order):
        r = b.loc[m]
        ax.errorbar(r.diff_vs_ref, yv, xerr=[[r.diff_vs_ref - r.diff_ci_lo], [r.diff_ci_hi - r.diff_vs_ref]], fmt="s",
                    ms=3.6, color=fk.c.blue, mec=fk.c.blue, elinewidth=1.0, capsize=1.5)
    ax.set_xscale("log"); ax.set_xlim(.004, 3)
    ax.set_xticks([.01, .1, 1]); ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.set_yticks(yy); ax.set_yticklabels([J.METHOD[m] for m in order]); ax.set_ylim(-.6, len(order) - .4)
    ax.set_xlabel("Gain in log score per prediction (95% CI)")
    n_pred = int(b.loc["AF 3.3", "n_records"])
    fk.panel(ax, "a", f"Prediction of later trials, gain of the model ({J.fmt_n(n_pred)} predictions)")
    # B -- calibration: coverage of central 90% predictive intervals
    ax2 = fig.add_subplot(r0[1], sharey=ax)
    ax2.axvspan(.87, .93, color=fk.c.grey_light, lw=0)
    ax2.axvline(.9, color=fk.c.ink, lw=.6, ls="--")
    ax2.plot([b.loc[m, "pit_central90"] for m in order], yy, "s", ms=3.6, color=fk.c.blue, ls="none")
    v = b.loc["AF 3.3", "pit_central90"]
    ax2.axvline(v, color=fk.c.vermilion, lw=1.0)
    ax2.set_xlim(.72, .96); ax2.set_xticks([.75, .85, .95])
    ax2.tick_params(axis="y", labelleft=False)
    ax2.set_xlabel("Coverage of 90% intervals")
    fk.panel(ax2, "b", "Calibration")
    # C -- agreement with the CTT individual participant data
    c = d["ctt"].copy()
    det = __import__("json").loads(d["gate"]["jno_ext_ctt"][1])
    ax3 = fig.add_subplot(r1[0])
    hr, lo, hi = np.exp(c.our_loghr), np.exp(c.our_loghr - 1.96 * c.our_se), np.exp(c.our_loghr + 1.96 * c.our_se)
    sig = c.ref_sig.values
    ax3.plot(c.ref_ratio[~sig], hr[~sig], "o", ms=3.2, color=fk.c.grey_dark, alpha=.7, ls="none",
             label="Not significant in IPD")
    ax3.errorbar(c.ref_ratio[sig], hr[sig], xerr=[c.ref_ratio[sig] - c.ref_lo[sig], c.ref_hi[sig] - c.ref_ratio[sig]],
                 yerr=[hr[sig] - lo[sig], hi[sig] - hr[sig]], fmt="o", ms=3.6, color=fk.c.vermilion,
                 mec=fk.c.vermilion, elinewidth=.8, capsize=1.2, label="Significant in IPD (FDR)")
    ax3.plot([.3, 4], [.3, 4], color=fk.c.ink, lw=.6, ls="--", label="Equal ratios")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlim(.7, 1.7); ax3.set_ylim(.55, 2.6)
    for a_, ticks in ((ax3.xaxis, [.75, 1, 1.25, 1.5]), (ax3.yaxis, [.6, .8, 1, 1.5, 2, 2.5])):
        a_.set_major_locator(mpl.ticker.FixedLocator(ticks))
        a_.set_major_formatter(mpl.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
        a_.set_minor_locator(mpl.ticker.NullLocator())
    ax3.set_xlabel("Rate ratio, individual participant data (95% CI)")
    ax3.set_ylabel("Hazard ratio, registry model (95% CrI)")
    tx(ax3, .72, 2.3, f"Spearman ρ = {det['spearman_rho']:.2f}; {det['n_evaluable']} outcomes", fontsize=fk.pt("annot"),
       va="center")
    ax3.legend(loc="lower right", frameon=False)
    fk.panel(ax3, "c", "Statin label outcomes vs IPD")
    # D -- excess fraction: registry model vs external estimates
    ax4 = fig.add_subplot(r1[1])
    names = {"Abnormal liver transaminases": "Abnormal liver transaminases",
             "Other LFT abnormality": "Other liver test abnormality",
             "Urinary composition alteration": "Urinary composition alteration", "Oedema": "Edema"}
    sc = c[sig].sort_values("ref_ratio", ascending=False)
    rows = [dict(lab=f"Statins: {names[o.outcome]}", ref=o.ref_ef, rlo=1 - 1 / o.ref_lo, rhi=1 - 1 / o.ref_hi,
                 our=o.our_ef, olo=o.our_ef_lo, ohi=o.our_ef_hi) for o in sc.itertuples()]
    ace = d["ace"]
    rows.append(dict(lab="ACE inhibitors: cough", ref=.37, rlo=np.nan, rhi=np.nan, our=ace.our_ef, olo=ace.our_ef_lo,
                     ohi=ace.our_ef_hi))
    R = pd.DataFrame(rows)
    yy4 = np.arange(len(R))[::-1]
    ax4.errorbar(R.ref, yy4 + .16, xerr=[(R.ref - R.rlo).fillna(0), (R.rhi - R.ref).fillna(0)], fmt="D", ms=3.4,
                 color=fk.c.ink, mec=fk.c.ink, elinewidth=.9, capsize=1.5, label="Published estimate")
    ax4.errorbar(R.our, yy4 - .16, xerr=[R.our - R.olo, R.ohi - R.our], fmt="s", ms=3.4, color=fk.c.blue,
                 mec=fk.c.blue, elinewidth=.9, capsize=1.5, label="Registry model")
    ax4.axvline(0, color=fk.c.grey, lw=.5); ax4.axvline(.5, color=fk.c.ink, lw=.6, ls="--")
    ax4.set_yticks(yy4); ax4.set_yticklabels(R.lab); ax4.set_ylim(-.7, len(R) - .3)
    ax4.set_xlim(-.7, .75); ax4.set_xticks([-.5, -.25, 0, .25, .5])
    ax4.set_xticklabels(["-0.5", "-0.25", "0", "0.25", "0.5"])
    ax4.set_xlabel("Excess fraction (95% CrI or CI)")
    ax4.legend(loc="lower left", frameon=False)
    fk.panel(ax4, "d", "Outcomes with a known excess")
    d["fig2d"] = R
    return fig


# ============================================================================== Figure 3
def half_primary(d):
    """Primary estimate with the 95% CI from 20 half-sample refits (jno_r2_subsample): point +/- 1.96 SD."""
    from types import SimpleNamespace
    h = S(d, "headline", "primary", "all", "exp_lt_half")
    hs = d["half"]; assert len(hs) == 20
    sd = float(np.std(hs, ddof=1))
    return SimpleNamespace(value=h.value, lo=h.value - 1.96 * sd, hi=h.value + 1.96 * sd, n_pairs=h.n_pairs)


def boot_share(df, rng, nb=1000):
    g = df.groupby("drug").p_af_lt_half.agg(["sum", "size"])
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=nb)
    v = (w @ g["sum"].values) / np.maximum(w @ g["size"].values, 1)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def fig3(d):
    pair = d["pair"]; L = pair[pair.on_fda].copy()
    fig = fk.figure("double", height=210, profile=J.FIG_PROFILE)
    g0 = fig.add_gridspec(2, 1, height_ratios=[.5, 1])
    top = g0[0].subgridspec(1, 2, width_ratios=[1, 1]); bot = g0[1].subgridspec(1, 2, width_ratios=[1.3, 1])
    # A -- distribution of the excess fraction
    ax = fig.add_subplot(top[0])
    v = L.af_mean.clip(-.5, 1)
    bins = np.linspace(-.5, 1, 31)
    ax.hist(v, bins=bins, color=fk.c.blue, edgecolor="white", linewidth=.3)
    ax.axvline(.5, color=fk.c.vermilion, lw=1.0, ls="--")
    h = half_primary(d)
    tx(ax, .53, ax.get_ylim()[1] * .82, f"Expected proportion <0.5:\n{100 * h.value:.1f}% (95% CI, "
       f"{100 * h.lo:.1f}%-{100 * h.hi:.1f}%)\nPosterior means <0.5:\n{100 * (L.af_mean < .5).mean():.1f}%",
       fontsize=fk.pt("annot"), va="center")
    ax.set_xlim(-.5, 1); ax.set_xlabel("Excess fraction (posterior mean)")
    ax.set_xticks([-.5, -.25, 0, .25, .5, .75, 1]); ax.set_xticklabels(["-0.5", "-0.25", "0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylabel("Listed adverse reactions, No.")
    fk.panel(ax, "a", "Listed reactions")
    # D -- by placebo frequency (top right)
    ax4 = fig.add_subplot(top[1])
    rng = np.random.default_rng(SEED)
    L["p0"] = L.per100_pl                                  # as in the registered decile analysis (afstd_by_soc)
    L["dec"] = pd.qcut(np.log10(L.p0.clip(.05, 100)), 10, labels=False)
    rows = []
    for k, g in L.groupby("dec"):
        lo, hi = boot_share(g, rng)
        rows.append(dict(dec=k, x=g.p0.median(), share=g.p_af_lt_half.mean(), lo=lo, hi=hi, n=len(g)))
    R = pd.DataFrame(rows)
    ax4.errorbar(R.x, R.share, yerr=[R.share - R.lo, R.hi - R.share], fmt="o-", ms=3.4, color=fk.c.blue, lw=.8,
                 elinewidth=.8, capsize=1.5)
    ax4.axhline(.5, color=fk.c.ink, lw=.6, ls="--")
    ax4.set_xscale("log"); ax4.set_xlim(.3, 30); ax4.set_ylim(0, 1.1); ax4.set_yticks([0, .25, .5, .75, 1])
    ax4.spines["left"].set_bounds(0, 1)
    ax4.set_yticklabels(["0", "25", "50", "75", "100"])
    ax4.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax4.set_xlabel("Placebo frequency per 100 patients")
    ax4.set_ylabel("Expected proportion <0.5, % (95% CI)")
    fk.panel(ax4, "b", "Placebo frequency")
    d["fig3b"] = R
    # B -- robustness of the primary outcome
    ax2 = fig.add_subplot(bot[0])
    js = d["jsum"]

    def J_(analysis, set_name, metric="exp_lt_half"):
        return S(d, analysis, set_name, "all", metric, table="jsum")
    items = [("All listed reactions (primary)", half_primary(d), True)]
    for st, lab in (("1", "1 Trial"), ("2-4", "2-4 Trials"), ("5-9", "5-9 Trials"), (">=10", "≥10 Trials")):
        items.append((lab, S(d, "headline", "primary", st, "exp_lt_half"), False))
    items += [("Clinical trials subsection of label", J_("jno_label_section", "trial"), False),
              ("Postmarketing subsection only", J_("jno_label_section", "postmarketing_only"), False),
              ("Complete-reporting trials (refit)", J_("jno_complete_reporting", "listed_complete"), False),
              ("Labels of 2016 (SIDER)", S(d, "sensitivity", "S3", "all", "exp_lt_half"), False),
              ("Innovator labels only", S(d, "sensitivity", "S7", "all", "exp_lt_half"), False),
              ("Terms matching ≤50% of labels", S(d, "sensitivity", "S8", "all", "exp_lt_half"), False),
              ("Estimable by conventional pooling", S(d, "sensitivity", "S1", "all", "exp_lt_half"), False),
              ("≥2 Trials per pair", S(d, "sensitivity", "S2", "all", "exp_lt_half"), False),
              ("≥5 Trials per pair", S(d, "sensitivity", "S6", "all", "exp_lt_half"), False),
              ("Model without zero-excess component",
               S(d, "no_effect_sensitivity", "without no-effect component", "all", "exp_lt_half"), False),
              ("Prior using label status (refit)", J_("jno_rev_label_aware", "label_aware"), False),
              ("No phase 1 or healthy volunteers (refit)", J_("jno_rev_refit_phase", "phase"), False),
              ("No shared placebo arms (refit)", J_("jno_rev_refit_shared", "shared"), False),
              ("Indication terms and vaccines excluded", J_("jno_rev_scope", "excl_both"), False),
              ("Single route in trials and label", J_("jno_r2_route_single", "route_single"), False),
              ("Phase 3 or 4 trials (refit)", J_("jno_r3_phase34", "phase34"), False),
              ("Approved indications (refit)", J_("jno_r3_indication", "indication"), False),
              ("Started by first approval (refit)", J_("jno_r3_approval", "preapproval"), False),
              ("Highest dose per trial (refit)", J_("jno_r3_highdose", "highdose"), False)]
    yy = np.arange(len(items))[::-1]
    for yv, (lab, r, prim) in zip(yy, items):
        col = fk.c.vermilion if prim else fk.c.blue
        ax2.errorbar(r.value, yv, xerr=[[r.value - r.lo], [r.hi - r.value]], fmt="D" if prim else "s", ms=3.6,
                     color=col, mec=col, elinewidth=1.0, capsize=1.5)
    ax2.axvline(.5, color=fk.c.ink, lw=.6, ls="--")
    ax2.axvline(items[0][1].value, color=fk.c.vermilion, lw=.5, ls=":")
    ax2.set_yticks(yy); ax2.set_yticklabels([i[0] for i in items]); ax2.set_ylim(-.6, len(items) - .4)
    ax2.set_xlim(.47, .93); ax2.set_xticks([.5, .6, .7, .8, .9]); ax2.set_xticklabels(["50", "60", "70", "80", "90"])
    ax2.set_xlabel("Expected proportion <0.5, % (95% CI)")
    fk.panel(ax2, "c", "Sensitivity")
    d["fig3c"] = pd.DataFrame([dict(item=i[0], value=i[1].value, lo=i[1].lo, hi=i[1].hi, n=i[1].n_pairs) for i in items])
    # C -- by organ system (>= 30 listed reactions)
    ax3 = fig.add_subplot(bot[1])
    s = d["sum"]
    so = s[(s.analysis == "by_soc") & (s.set_name == "primary") & (s.metric == "exp_lt_half") & (s.n_pairs >= 30)]
    so = so.sort_values("value")
    yy = np.arange(len(so))
    ax3.errorbar(so.value, yy, xerr=[so.value - so.lo, so.hi - so.value], fmt="s", ms=3.4, color=fk.c.blue,
                 mec=fk.c.blue, elinewidth=1.0, capsize=1.5)
    ax3.axvline(.5, color=fk.c.ink, lw=.6, ls="--")
    ax3.axvline(h.value, color=fk.c.vermilion, lw=.5, ls=":")
    ax3.set_yticks(yy)
    ax3.set_yticklabels([f"{P.SOC_SHORT.get(g, g)} ({J.fmt_n(n)})" for g, n in zip(so.grp, so.n_pairs)])
    ax3.set_ylim(-.6, len(so) - .4)
    ax3.set_xlim(.3, 1); ax3.set_xticks([.3, .5, .7, .9]); ax3.set_xticklabels(["30", "50", "70", "90"])
    ax3.set_xlabel("Expected proportion <0.5, % (95% CI)")
    fk.panel(ax3, "d", "Organ system")
    d["fig3d"] = so
    return fig


def main() -> int:
    with pv.connect(read_only=True) as con:
        d = load(con)
    for g in ("afstd_label_match", "af33_adopt", "jno_ext_ctt", "jno_complete_reporting", "jno_label_section"):
        assert d["gate"][g][0], f"prerequisite gate {g} not passed"
    made = {}
    for (fid, _), fn in zip(J.MAIN_FIGS, (fig1, fig2, fig3)):
        made[fid] = fk.save(fn(d), ROOT / "figures" / "jama" / fid)
        print(fid, made[fid].report)
    with pv.run_context(__file__, env_name="oajoint-py311",
                        software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','jno_summary','bench_score','jno_ext_concordance',
                   'jno_label_section','ctgov_trial','atlas_contrast')""").fetchall()]
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "jno_summary", "bench_score",
                                                   "jno_ext_concordance"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
