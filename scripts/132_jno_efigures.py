#!/usr/bin/env python3
"""eFigures of Supplement 1 of the JAMA Network Open manuscript, read only from the project DuckDB (figkit 'jama').

eFigure 1  temporal calibration of the posterior probabilities (A) and the simulation without exact zeros (B)
eFigure 2  discontinuation because of adverse events by drug class (ATC level 1)
eFigure 3  frequencies per 100 patients with the drug vs placebo for listed adverse reactions
eFigure 4  expected proportion with excess fraction < 0.5 by drug class (ATC level 3, >= 150 listed reactions)

ATC level-3 names are taken from drugs with a single ATC code: ChEMBL lists the descriptions of a drug's ATC codes in
alphabetical order, not in code order, so pairing codes and descriptions of multi-code drugs mislabels classes.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import jama as J  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
import figkit as fk  # noqa: E402

ATC1 = {"A": "Alimentary tract and metabolism", "B": "Blood and blood-forming organs", "C": "Cardiovascular system",
        "D": "Dermatologicals", "G": "Genitourinary system and sex hormones",
        "H": "Systemic hormonal preparations", "J": "Anti-infectives for systemic use",
        "L": "Antineoplastic and immunomodulating agents", "M": "Musculoskeletal system", "N": "Nervous system",
        "P": "Antiparasitic products", "R": "Respiratory system", "S": "Sensory organs", "V": "Various",
        "?": "No ATC code"}


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def atc3_names(con):
    rows = con.execute("SELECT atc4, atc4_desc FROM ref_chembl_drug WHERE len(atc4) = 1").fetchall()
    c = {}
    for codes, descs in rows:
        parts = descs[0].split(": ")
        if len(parts) >= 3:
            c.setdefault(codes[0][:4], Counter())[parts[2]] += 1
    fix = {"adhd": "ADHD", "hiv": "HIV", "ace": "ACE"}
    out = {}
    for k, v in c.items():
        name = v.most_common(1)[0][0].capitalize()
        out[k] = " ".join(fix.get(w.strip(",").lower(), w) + ("," if w.endswith(",") and w.strip(",").lower() in fix
                                                               else "") for w in name.split())
    return out


def load(con):
    d = {"gate": {r[0]: (r[1], json.loads(r[2]) if r[2] else {}) for r in con.execute(
        "SELECT gate_name, passed, metric_detail FROM prov_decision").fetchall()}}
    d["pair"] = con.execute("""SELECT p.drug, p.term, p.p_af_lt_half, p.per100_pl, e.per100_drug_exp
                               FROM afstd_pair p JOIN afstd_pair_exp e USING (drug, term) WHERE p.on_fda
                               ORDER BY drug, term""").df()
    d["cls"] = con.execute("""SELECT grp, n_pairs, n_drugs, value, lo, hi FROM afstd_summary WHERE analysis = 'by_class'
                              AND set_name = 'primary' AND metric = 'exp_lt_half' ORDER BY grp""").df()
    d["head"] = con.execute("""SELECT value FROM afstd_summary WHERE analysis = 'headline' AND set_name = 'primary'
                               AND grp = 'all' AND metric = 'exp_lt_half'""").fetchone()[0]
    d["atc3"] = atc3_names(con)
    return d


def efig1(d):
    tc = d["gate"]["jno_rev_temporal_calibration"][1]
    D = pd.DataFrame(tc["deciles"])
    fig = fk.figure("double", height=85, profile=J.FIG_PROFILE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0])
    k = np.round(D.obs * D.n).astype(int)
    lo, hi = wilson(k.values, D.n.values)
    ax.plot([0, 1], [0, 1], color=fk.c.ink, lw=.6, ls="--", label="Perfect calibration")
    ax.errorbar(D.pred, D.obs, yerr=[D.obs - lo, hi - D.obs], fmt="o", ms=3.6, color=fk.c.blue, mec=fk.c.blue,
                elinewidth=.9, capsize=1.5, label="Decile (95% CI)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.1); ax.spines["left"].set_bounds(0, 1); ax.spines["bottom"].set_bounds(0, 1)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(mpl.ticker.FixedLocator([0, .25, .5, .75, 1]))
        a_.set_major_formatter(mpl.ticker.FixedFormatter(["0", "25", "50", "75", "100"]))
    ax.set_xlabel("Predicted probability, % (fit before Feb 2017)")
    ax.set_ylabel("Observed in later trials, %")
    ax.legend(loc="lower right", frameon=False)
    fk.panel(ax, "a", f"Temporal calibration ({J.fmt_n(tc['n_pairs'])} pairs)")
    # B -- simulation: estimated vs true share
    ax2 = fig.add_subplot(gs[1])
    rows = d["gate"]["jno_rev_sim_nozero"][1]["rows"]
    lab = {"S4": "No exact zeros", "S1": "40% exact zeros"}
    yy, ticks = [], []
    for i, scn in enumerate(("S4", "S1")):
        r = {x["estimator"]: x for x in rows if x["scenario"] == scn}
        y0 = 1 - i
        ax2.plot(r["zero-excess component"]["true_share"], y0, "D", ms=4.2, color=fk.c.ink, ls="none",
                 label="True proportion" if i == 0 else None)
        ax2.plot(r["zero-excess component"]["est_share"], y0 + .12, "s", ms=3.8, color=fk.c.blue, ls="none",
                 label="Model" if i == 0 else None)
        ax2.plot(r["without zero-excess component"]["est_share"], y0 - .12, "o", ms=3.8, color=fk.c.vermilion,
                 ls="none", label="Without zero-excess component" if i == 0 else None)
        yy.append(y0); ticks.append(lab[scn])
    ax2.set_yticks(yy); ax2.set_yticklabels(ticks); ax2.set_ylim(-.6, 2.3)
    ax2.set_xlim(.7, .95); ax2.set_xticks([.7, .75, .8, .85, .9, .95])
    ax2.set_xticklabels(["70", "75", "80", "85", "90", "95"])
    ax2.set_xlabel("Pairs with excess fraction <0.5, %")
    ax2.legend(loc="upper left", frameon=False)
    fk.panel(ax2, "b", "Simulation with known truth")
    return fig


def efig2(d):
    v = d["gate"]["jno_rev_discontinuation_v2"][1]
    by = {k: x for k, x in v["by_atc1"].items() if x["n_trials"] >= 10}
    R = pd.DataFrame([dict(g=k, **{kk: x[kk] for kk in ("rr", "rr_lo", "rr_hi", "n_trials")}) for k, x in by.items()])
    R = R.sort_values("rr")
    o = v["restricted"]
    fig = fk.figure("double", height=110, profile=J.FIG_PROFILE)
    ax = fig.add_subplot(1, 1, 1)
    yy = np.arange(len(R)) + 1
    ax.errorbar(R.rr, yy, xerr=[R.rr - R.rr_lo, R.rr_hi - R.rr], fmt="s", ms=3.6, color=fk.c.blue, mec=fk.c.blue,
                elinewidth=.9, capsize=1.5, label="Drug class (95% CI)")
    ax.errorbar([o["rr"]], [0], xerr=[[o["rr"] - o["rr_lo"]], [o["rr_hi"] - o["rr"]]], fmt="D", ms=4.2,
                color=fk.c.vermilion, mec=fk.c.vermilion, elinewidth=.9, capsize=1.5, label="All trials (95% CI)")
    ax.axvline(1, color=fk.c.ink, lw=.6, ls="--")
    lo_ = min(.5, R.rr_lo.min() * .9)
    ax.set_xscale("log"); ax.set_xlim(lo_, 4)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([.5, .75, 1, 1.5, 2, 3, 4]))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["0.5", "0.75", "1", "1.5", "2", "3", "4"]))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_yticks([0, *yy])
    ax.set_yticklabels([f"All ({J.fmt_n(o['n_trials'])} trials)"] +
                       [f"{ATC1.get(g, g)} ({J.fmt_n(n)})" for g, n in zip(R.g, R.n_trials)])
    ax.set_ylim(-.7, len(R) + .7)
    ax.set_xlabel("Risk ratio for discontinuation because of adverse events, drug vs placebo")
    ax.legend(loc="lower right", frameon=False)
    d["efig2"] = R
    return fig


def efig3(d):
    L = d["pair"].copy()
    L = L[(L.per100_pl > 0) & (L.per100_drug_exp > 0)]
    fig = fk.figure("double", height=125, profile=J.FIG_PROFILE)
    ax = fig.add_subplot(1, 1, 1)
    cls = [("Probability excess fraction <0.5, >97.5%", L.p_af_lt_half > .975, fk.c.blue),
           ("Probability excess fraction <0.5, 2.5%-97.5%", (L.p_af_lt_half <= .975) & (L.p_af_lt_half >= .025),
            fk.c.grey),
           ("Probability excess fraction <0.5, <2.5%", L.p_af_lt_half < .025, fk.c.vermilion)]
    for lab, m, col in cls:
        ax.scatter(L.per100_pl[m], L.per100_drug_exp[m], s=2.5, color=col, alpha=.45, lw=0, rasterized=True,
                   label=f"{lab} ({J.fmt_n(m.sum())})")
    x = np.array([.01, 100])
    ax.plot(x, x, color=fk.c.ink, lw=.7, ls="--", label="Drug = placebo")
    ax.plot(x, 2 * x, color=fk.c.ink, lw=.7, ls=":", label="Drug = 2 × placebo (excess fraction 0.5)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.03, 100); ax.set_ylim(.03, 100)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Frequency with placebo per 100 patients")
    ax.set_ylabel("Frequency with the drug per 100 patients")
    ax.legend(loc="upper left", frameon=False, markerscale=3)
    return fig


def efig4(d):
    C = d["cls"].copy()
    C = C[C.n_pairs >= 150].copy()
    C["code"] = C.grp.str.split("|").str[0]
    C["name"] = C.code.map(lambda c: d["atc3"].get(c, c))
    C = C.sort_values("value")
    fig = fk.figure("double", height=215, profile=J.FIG_PROFILE)
    ax = fig.add_subplot(1, 1, 1)
    yy = np.arange(len(C))
    ax.errorbar(C.value, yy, xerr=[C.value - C.lo, C.hi - C.value], fmt="s", ms=3.4, color=fk.c.blue, mec=fk.c.blue,
                elinewidth=.9, capsize=1.3)
    ax.axvline(.5, color=fk.c.ink, lw=.6, ls="--"); ax.axvline(d["head"], color=fk.c.vermilion, lw=.6, ls=":")
    ax.set_yticks(yy); ax.set_yticklabels([f"{n} ({c}; {J.fmt_n(k)})" for n, c, k in zip(C.name, C.code, C.n_pairs)])
    ax.set_ylim(-.7, len(C) - .3)
    ax.set_xlim(.3, 1); ax.set_xticks([.3, .5, .7, .9]); ax.set_xticklabels(["30", "50", "70", "90"])
    ax.set_xlabel("Expected proportion with excess fraction <0.5, % (95% CI)")
    d["efig4"] = C
    return fig


def main() -> int:
    with pv.connect(read_only=True) as con:
        d = load(con)
    for g in ("jno_rev_temporal_calibration", "jno_rev_sim_nozero", "jno_rev_discontinuation_v2"):
        assert d["gate"][g][1], f"gate {g} not evaluated"
    made = {}
    for (fid, _), fn in zip(J.EFIGS, (efig1, efig2, efig3, efig4)):
        made[fid] = fk.save(fn(d), ROOT / "figures" / "jama" / fid)
        print(fid, made[fid].report)
    with pv.run_context(__file__, env_name="oajoint-py311",
                        software={"matplotlib": mpl.__version__, "figkit": fk.version()}) as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_summary','jno_summary','ref_chembl_drug')""").fetchall()]
        for fid, saved in made.items():
            for path in saved.paths:
                run.archive_figure(fid, "all", path, panel_title=fid,
                                   source_objects=["afstd_pair", "afstd_summary", "jno_summary"], upstream=up)
            run.archive_figure_qa(fid, saved)
            run.archive_figure_abbrev(fid, saved.abbreviations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
