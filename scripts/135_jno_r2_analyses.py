#!/usr/bin/env python3
"""Gates jno_r2_absolute_complete, jno_r2_route_single and jno_r2_prior_vs_data (rules in 133_jno_r2_prereg.py)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("s110", ROOT / "scripts" / "110_afstd_findings.py")
S110 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S110)
SEED = 20261013
STRATA = ["1", "2-4", "5-9", ">=10"]


def q3(x):
    return [float(np.median(x)), float(np.quantile(x, .25)), float(np.quantile(x, .75))]


def main() -> int:
    rng = np.random.default_rng(SEED)
    pv.ensure_views()
    with pv.connect(read_only=True) as con:
        p = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_pair ORDER BY drug, term").df()
        post = con.execute("SELECT * FROM v_afstd_posterior ORDER BY drug, term").df()
        e = con.execute("SELECT drug, term, per100_drug_exp, per100_added_exp FROM afstd_pair_exp").df()
        cell = con.execute("""SELECT drug, term, prior_mean, lik_mean, lik_sd FROM afstd_cell""").df()
        complete = {r[0] for r in con.execute("SELECT nct FROM ctgov_trial WHERE frequency_threshold = 0").fetchall()}
        route = con.execute("""SELECT drug, route_family FROM atlas_contrast_route
                               WHERE tier IN ('T1', 'T2', 'T3') AND route_family NOT IN ('mixed', 'unresolved')""").df()
        lab_route = con.execute("SELECT generic_name, route_family FROM fda_label_route_pool").df()
        gen = con.execute("SELECT drug, term, fda_generic FROM afstd_label_match WHERE on_fda").df()
        arms, K, J = af3.build(con)
    assert (p.drug.values == post.drug.values).all() and (p.term.values == post.term.values).all()
    G = post[[f"g{i:03d}" for i in range(201)]].to_numpy(np.float64, copy=True); G /= G.sum(1, keepdims=True)
    grid = af3.GRID
    det = {}

    # ------------------------------------------------ absolute frequencies from complete-reporting trials
    Kc = K[K.nct.isin(complete)]
    tdn = Kc.groupby(["nct", "drug"]).agg(n0=("n0", "max"), nd=("nd", "max")).reset_index()   # every trial of the drug
    ev = Kc.groupby(["drug", "t"]).agg(x0=("x0", "sum"), xd=("xd", "sum")).rename_axis(["drug", "term"])
    dn = tdn.groupby("drug").agg(n0c=("n0", "sum"), ndc=("nd", "sum"), ntr_c=("nct", "nunique"))
    L = p[p.on_fda].reset_index().rename(columns={"index": "row"})
    L = L.merge(dn, left_on="drug", right_index=True, how="inner")
    L = L.merge(ev, left_on=["drug", "term"], right_index=True, how="left").fillna({"x0": 0, "xd": 0})
    L = L.merge(e, on=["drug", "term"])
    p0c = ((L.x0 + .5) / (L.n0c + 1)).values
    Gl = G[L.row.values]
    drug_c = 100 * (Gl * -np.expm1(np.log1p(-p0c[:, None]) * np.exp(grid)[None, :])).sum(1)
    L["per100_pl_c"] = 100 * p0c; L["per100_drug_c"] = drug_c; L["per100_added_c"] = drug_c - L.per100_pl_c
    L["per100_pl_c_obs"] = 100 * L.x0 / L.n0c; L["per100_drug_c_obs"] = 100 * L.xd / L.ndc
    det["absolute_complete"] = dict(
        n_pairs=int(len(L)), n_drugs=int(L.drug.nunique()), n_trials_complete=int(tdn.nct.nunique()),
        complete=dict(pl=q3(L.per100_pl_c), drug=q3(L.per100_drug_c), added=q3(L.per100_added_c),
                      share_added_lt1=float((L.per100_added_c < 1).mean()),
                      crude_pl=q3(L.per100_pl_c_obs), crude_drug=q3(L.per100_drug_c_obs)),
        main_same_pairs=dict(pl=q3(L.per100_pl), drug=q3(L.per100_drug_exp), added=q3(L.per100_added_exp),
                             share_added_lt1=float((L.per100_added_exp < 1).mean())))
    allL = p[p.on_fda].merge(e, on=["drug", "term"])
    det["absolute_main_all"] = dict(n_pairs=int(len(allL)), pl=q3(allL.per100_pl), drug=q3(allL.per100_drug_exp),
                                    added=q3(allL.per100_added_exp),
                                    share_added_lt1=float((allL.per100_added_exp < 1).mean()))

    # ------------------------------------------------ formulation-matched subset
    rf = route.groupby("drug").route_family.agg(lambda s: sorted(set(s)))
    lr = lab_route.groupby("generic_name").route_family.agg(lambda s: sorted(set(s)))
    g1 = gen.drop_duplicates("drug").set_index("drug").fda_generic
    ok = {d for d in rf.index if len(rf[d]) == 1 and d in g1.index and g1[d] in lr.index and lr[g1[d]] == rf[d]}
    Ls = p[p.on_fda & p.drug.isin(ok)]
    lo, hi = S110.boot_mean(Ls, "p_af_lt_half", rng)
    det["route_single"] = dict(n_pairs=int(len(Ls)), n_drugs=int(Ls.drug.nunique()),
                               exp_lt_half=float(Ls.p_af_lt_half.mean()), ci=[lo, hi],
                               n_listed_all=int(p.on_fda.sum()))

    # ------------------------------------------------ data vs prior
    Lp = p[p.on_fda].merge(cell, on=["drug", "term"])
    Lp["p_lik"] = stats.norm.cdf((np.log(2) - Lp.lik_mean) / Lp.lik_sd.clip(lower=1e-3))
    Lp["prior_lt"] = (Lp.prior_mean < np.log(2)).astype(float)
    pvd = {}
    for s, g in [("all", Lp)] + [(s, Lp[Lp.evidence == s]) for s in STRATA]:
        pvd[s] = dict(n=int(len(g)), posterior=float(g.p_af_lt_half.mean()), likelihood_only=float(g.p_lik.mean()),
                      prior_mean_below=float(g.prior_lt.mean()))
    det["prior_vs_data"] = pvd
    print(json.dumps(det, indent=1, default=float))

    rows = []

    def add(an, st, grp, n, nd, metric, v, lo_=np.nan, hi_=np.nan):
        rows.append(dict(analysis=an, set_name=st, grp=grp, n_pairs=n, n_drugs=nd, metric=metric, value=float(v),
                         lo=lo_, hi=hi_))
    a = det["absolute_complete"]
    for k in ("pl", "drug", "added"):
        add("jno_r2_absolute_complete", "complete", "all", a["n_pairs"], a["n_drugs"], f"median_{k}",
            a["complete"][k][0], a["complete"][k][1], a["complete"][k][2])
        add("jno_r2_absolute_complete", "main_same_pairs", "all", a["n_pairs"], a["n_drugs"], f"median_{k}",
            a["main_same_pairs"][k][0], a["main_same_pairs"][k][1], a["main_same_pairs"][k][2])
    r = det["route_single"]
    add("jno_r2_route_single", "route_single", "all", r["n_pairs"], r["n_drugs"], "exp_lt_half", r["exp_lt_half"],
        r["ci"][0], r["ci"][1])
    for s, v in pvd.items():
        for k in ("posterior", "likelihood_only", "prior_mean_below"):
            add("jno_r2_prior_vs_data", "listed", s, v["n"], 0, k, v[k])
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','afstd_cell','afstd_label_match','atlas_contrast_route',
                   'fda_label_route_pool','ctgov_trial')""").fetchall()]
        R = pd.DataFrame(rows); R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis IN ('jno_r2_absolute_complete','jno_r2_route_single',"
                        "'jno_r2_prior_vs_data')")
        run.archive_table("jno_summary", R, upstream=up)
        run.evaluate_gate("jno_r2_absolute_complete", a["complete"]["added"][0], None, detail=det["absolute_complete"])
        run.evaluate_gate("jno_r2_route_single", r["exp_lt_half"], None, detail=r)
        run.evaluate_gate("jno_r2_prior_vs_data", None, None, detail=pvd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
