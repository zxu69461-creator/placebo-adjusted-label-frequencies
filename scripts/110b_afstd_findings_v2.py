#!/usr/bin/env python3
"""Expected-value point summaries of the manuscript (gates afstd_absolute_v2, afstd_listing_rules_v2, afstd_point_v2),
as registered by 108b_afstd_prereg_v2.py. Reads afstd_pair (unchanged), the full posteriors (Parquet afstd_posterior)
and the observed pooled arm counts of the estimator's data; writes afstd_pair_exp and appends '*_v2' rows to
afstd_summary."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

STRATA = ["1", "2-4", "5-9", ">=10"]


def main() -> int:
    pv.ensure_views()
    with pv.connect(read_only=True) as con:
        p = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_pair ORDER BY drug, term").df()
        post = con.execute("SELECT * FROM v_afstd_posterior ORDER BY drug, term").df()
        conv = con.execute("""SELECT c.drug, c.term, c.af FROM af_cell c JOIN af_label_match m USING (drug, term)
                              WHERE m.on_fda""").df()
        arms, K, J = af3.build(con)
    assert (p.drug.values == post.drug.values).all() and (p.term.values == post.term.values).all()
    G = post[[f"g{i:03d}" for i in range(201)]].to_numpy(np.float64, copy=True)
    G /= G.sum(1, keepdims=True)
    grid = af3.GRID
    # checks against the stored summaries
    af_mean = G @ (1 - np.exp(-grid))
    assert np.nanmax(np.abs(af_mean - p.af_hz.values)) < 1e-4, np.nanmax(np.abs(af_mean - p.af_hz.values))
    p0 = p.p0_own.values[:, None]
    drug_exp = 100 * (G * -np.expm1(np.log1p(-p0) * np.exp(grid)[None, :])).sum(1)
    obs = K.groupby(["drug", "t"]).agg(x_drug_own=("xd", "sum"), n_drug_own=("nd", "sum"))
    p = p.merge(obs, left_on=["drug", "term"], right_index=True, how="left")
    p["af_mean"] = af_mean
    p["per100_drug_exp"] = drug_exp
    p["per100_added_exp"] = drug_exp - p.per100_pl
    p["per100_drug_obs"] = 100 * p.x_drug_own / p.n_drug_own
    p["per100_pl_obs"] = 100 * p.x0_own / p.n0_own
    p["rule_a_v2"] = (p.per100_drug_obs >= 2) & (p.per100_drug_obs > p.per100_pl_obs)
    p["rule_b_v2"] = p.p_no_excess < .025
    p["rule_c_v2"] = p.rule_b_v2 & (p.per100_added_exp >= 1)
    lab, has = p[p.on_fda], p[p.has_fda_label]
    rows = []

    def add(analysis, grp, d, metric, value, lo=np.nan, hi=np.nan, set_name="primary"):
        rows.append(dict(analysis=analysis, set_name=set_name, grp=grp, n_pairs=int(len(d)),
                         n_drugs=int(d.drug.nunique()) if len(d) else 0, metric=metric, value=float(value), lo=lo, hi=hi))

    for s, g in [("all", lab)] + [(s, lab[lab.evidence == s]) for s in STRATA]:
        add("absolute_v2", s, g, "median_per100_drug_exp", g.per100_drug_exp.median())
        add("absolute_v2", s, g, "median_per100_pl", g.per100_pl.median())
        add("absolute_v2", s, g, "median_per100_added_exp", g.per100_added_exp.median(),
            g.per100_added_exp.quantile(.25), g.per100_added_exp.quantile(.75))
        add("absolute_v2", s, g, "share_added_exp_lt1", (g.per100_added_exp < 1).mean())
        add("absolute_v2", s, g, "share_added_hi_lt1", (g.per100_added_hi < 1).mean())
        add("absolute_v2", s, g, "share_added_exp_lt_half_freq", (g.per100_added_exp < .5 * g.per100_drug_exp).mean())
        pos = g[g.per100_added_exp > 0]
        add("absolute_v2", s, pos, "median_nnh", (100 / pos.per100_added_exp).median())
    ref = 100 * (G[lab.index] * -np.expm1(np.log1p(-lab.p0_12wk.values[:, None]) * np.exp(grid)[None, :])).sum(1) \
        - 100 * lab.p0_12wk.values
    add("absolute_v2", "all", lab, "median_per100_added_exp", np.median(ref), set_name="reference_background")
    add("absolute_v2", "all", lab, "share_added_exp_lt1", float((ref < 1).mean()), set_name="reference_background")

    cd = has[has.on_fda & (has.p_af_lt_half < .025)]
    for r in ("rule_a_v2", "rule_b_v2", "rule_c_v2"):
        kept = has[has.on_fda & has[r]]
        add("listing_rules_v2", "all", has, "retained", len(kept) / has.on_fda.sum(), set_name=r)
        add("listing_rules_v2", "all", kept, "bg_share_retained", (kept.p_af_lt_half > .975).mean(), set_name=r)
        add("listing_rules_v2", "all", kept, "exp_lt_half_retained", kept.p_af_lt_half.mean(), set_name=r)
        add("listing_rules_v2", "all", cd, "drug_caused_retained", cd[r].mean(), set_name=r)
        add("listing_rules_v2", "all", has, "unlisted_added", int((~has.on_fda & has[r]).sum()), set_name=r)
    add("listing_rules_v2", "all", lab, "bg_share_retained", (lab.p_af_lt_half > .975).mean(), set_name="current")
    add("listing_rules_v2", "all", lab, "exp_lt_half_retained", lab.p_af_lt_half.mean(), set_name="current")

    def pt(grp, d, set_name="primary"):
        add("point_v2", grp, d, "median_af_mean", d.af_mean.median(), d.af_mean.quantile(.25), d.af_mean.quantile(.75),
            set_name=set_name)
        add("point_v2", grp, d, "share_af_mean_lt_half", (d.af_mean < .5).mean(), set_name=set_name)
    pt("all", lab)
    for s in STRATA:
        pt(s, lab[lab.evidence == s])
    for s, g in lab.groupby("soc_h"):
        if len(g) >= 30:
            pt(s, g, "by_soc")
    atc = {}
    with pv.connect(read_only=True) as con:
        for c, codes in con.execute("SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall():
            atc[c] = sorted({x[:4] for x in (codes or [])})
    cl = pd.DataFrame([(c3, i) for i, dr in zip(lab.index, lab.drug) for c3 in atc.get(dr, [])], columns=["atc3", "i"])
    for c3, g in cl.groupby("atc3"):
        if len(g) >= 20:
            pt(c3, lab.loc[g.i], "by_class")
    both = lab[lab.af1_estimable].merge(conv, on=["drug", "term"])
    add("point_v2", "all", both, "spearman_vs_conventional", stats.spearmanr(both.af_mean, both.af).statistic,
        set_name="S1xS5")
    # the estimator's data, for the text and the data-flow panel
    kk = K.assign(drug=K.drug, term=K.t)
    add("data_v2", "all", kk, "trials", K.nct.nunique(), set_name="estimator")
    add("data_v2", "all", kk, "records", len(K), set_name="estimator")
    add("data_v2", "all", kk, "arms", len(arms), set_name="estimator")
    sm = pd.DataFrame(rows)
    g = lambda a, s, gr, m: float(sm[(sm.analysis == a) & (sm.set_name == s) & (sm.grp == gr) & (sm.metric == m)].value.iloc[0])  # noqa: E731
    print(f"expected per 100: drug {g('absolute_v2','primary','all','median_per100_drug_exp'):.2f}  placebo "
          f"{g('absolute_v2','primary','all','median_per100_pl'):.2f}  added {g('absolute_v2','primary','all','median_per100_added_exp'):.2f}  "
          f"share added<1 {g('absolute_v2','primary','all','share_added_exp_lt1'):.3f}")
    for r in ("rule_a_v2", "rule_b_v2", "rule_c_v2"):
        print(f"  {r}: retained {g('listing_rules_v2',r,'all','retained'):.3f}  bg among kept "
              f"{g('listing_rules_v2',r,'all','bg_share_retained'):.3f}  drug-caused kept "
              f"{g('listing_rules_v2',r,'all','drug_caused_retained'):.3f}  unlisted added "
              f"{g('listing_rules_v2',r,'all','unlisted_added'):.0f}")
    print(f"posterior-mean AF of listed pairs: median {g('point_v2','primary','all','median_af_mean'):.3f}; "
          f"share < 0.5 {g('point_v2','primary','all','share_af_mean_lt_half'):.3f}; rho vs conventional "
          f"{g('point_v2','S1xS5','all','spearman_vs_conventional'):.3f}")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','afstd_posterior','af_cell','af_label_match','ref_chembl_drug')").fetchall()]
        with pv.connect() as con:
            con.execute("DELETE FROM afstd_summary WHERE analysis IN ('absolute_v2','listing_rules_v2','point_v2','data_v2')")
        keep = ["drug", "term", "af_mean", "x_drug_own", "n_drug_own", "per100_drug_obs", "per100_pl_obs",
                "per100_drug_exp", "per100_added_exp", "rule_a_v2", "rule_b_v2", "rule_c_v2"]
        out = p[keep].copy(); out["run_id"] = run.run_id
        run.archive_table("afstd_pair_exp", out, upstream=up, mode="replace_all")
        sm["run_id"] = run.run_id
        run.archive_table("afstd_summary", sm, upstream=up)
        det = lambda a: sm[sm.analysis == a].drop(columns="run_id").replace({np.nan: None}).to_dict("records")  # noqa: E731
        run.evaluate_gate("afstd_absolute_v2", g("absolute_v2", "primary", "all", "median_per100_added_exp"), None,
                          detail=dict(rows=det("absolute_v2")))
        run.evaluate_gate("afstd_listing_rules_v2", g("listing_rules_v2", "rule_b_v2", "all", "retained"), None,
                          detail=dict(rows=det("listing_rules_v2")))
        run.evaluate_gate("afstd_point_v2", g("point_v2", "primary", "all", "median_af_mean"), None,
                          detail=dict(rows=det("point_v2")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
