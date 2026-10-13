#!/usr/bin/env python3
"""Route-stratified attributable fractions (gate af_route).

Every v2 rule is kept (US spelling, >=2 trials, >=20 events, MH); the only change is that a
cell is (ingredient, route family, term) and is compared with the labels of THAT route.
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


V2 = load(ROOT / "scripts" / "52_af_v2.py", "v2")
M34, pattern = V2.M34, V2.pattern
V2_HEADLINE = {"af_hi_lt_half": 0.486, "af_le_zero": 0.255}   # registered in af_route's rule text
SEP = "|"


def label_texts(idx):
    with pv.connect(read_only=True) as con:
        pool = con.execute("SELECT generic_name, route_family, pool, adverse_reactions_text "
                           "FROM fda_label_route_pool").df()
    acc = defaultdict(lambda: {"innovator": [], "all_labels": []})
    for g, f, pl, t in zip(pool.generic_name, pool.route_family, pool.pool, pool.adverse_reactions_text):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                acc[c + SEP + f][pl].append(t or "")
    return {k: "\n\n".join(v["innovator"] or v["all_labels"]) for k, v in acc.items()}


def label_cells(cells, text):
    pats = {t: pattern(t) for t in cells.term.unique()}
    cells["has_route_label"] = cells.stratum.isin(text)
    cells["on_route_label"] = [bool(s in text and pats[t].search(text[s])) for s, t in zip(cells.stratum, cells.term)]
    return cells


def summary(cells):
    fc = cells[cells.has_route_label]
    on, off = fc[fc.on_route_label], fc[~fc.on_route_label]
    mw = stats.mannwhitneyu(on.log_rr, off.log_rr, alternative="greater")
    return dict(n_strata_labelled=int(fc.stratum.nunique()), n_on=int(len(on)), n_off=int(len(off)),
                af_hi_lt_half=float((on.af_hi < .5).mean()), af_le_zero=float((on.af <= 0).mean()),
                af_lt_half=float((on.af < .5).mean()),
                off_af_hi_lt_half=float((off.af_hi < .5).mean()), off_af_le_zero=float((off.af <= 0).mean()),
                validity_p=float(mw.pvalue), median_diff=float(on.log_rr.median() - off.log_rr.median()))


def main() -> int:
    rows = V2.canon_rows()
    with pv.connect(read_only=True) as con:
        rt = con.execute("SELECT contrast_id, route_family, tier FROM atlas_contrast_route").df()
    rows = rows.merge(rt, on="contrast_id", how="inner")
    rows = rows[~rows.route_family.isin(["mixed", "unresolved"])].copy()
    rows["ingredient"] = rows.drug
    rows["drug"] = rows.ingredient + SEP + rows.route_family
    print(f"route-coded rows {len(rows):,}; strata {rows.drug.nunique():,} "
          f"(ingredients {rows.ingredient.nunique():,})")
    idx, _ = M34.build_index()
    text = label_texts(idx)

    def run_cells(r):
        c = V2.full_cells(r).rename(columns={"drug": "stratum"})
        c[["drug", "route_family"]] = c.stratum.str.split(SEP, n=1, expand=True)
        return label_cells(c, text)

    cells = run_cells(rows)
    main_s = summary(cells)
    sens = summary(run_cells(rows[rows.tier.isin(["T1", "T2"])]))
    fail = []
    if not (main_s["validity_p"] < 1e-3 and main_s["median_diff"] > 0):
        fail.append("validity")
    d1 = abs(main_s["af_hi_lt_half"] - V2_HEADLINE["af_hi_lt_half"])
    d2 = abs(main_s["af_le_zero"] - V2_HEADLINE["af_le_zero"])
    if not (d1 <= .05 and d2 <= .05):
        fail.append("headline_shift")
    print(f"\ncells {len(cells):,}; labelled strata {main_s['n_strata_labelled']:,}; "
          f"route-matched label cells {main_s['n_on']:,}")
    for k, s in (("MAIN (T1-T3)", main_s), ("text-coded only (T1-T2)", sens)):
        print(f"  {k:<26} on-label n {s['n_on']:>5,}  AF_hi<.5 {s['af_hi_lt_half']:.1%}  AF<=0 {s['af_le_zero']:.1%}  "
              f"AF<.5 {s['af_lt_half']:.1%}  | off-label AF_hi<.5 {s['off_af_hi_lt_half']:.1%}  "
              f"validity P {s['validity_p']:.1e}")
    print(f"  shift vs v2: AF_hi<.5 {main_s['af_hi_lt_half'] - .486:+.3f}, AF<=0 {main_s['af_le_zero'] - .255:+.3f}")
    # ---- between-route heterogeneity -------------------------------------------------
    cells["se"] = (np.log(cells.rr_hi) - np.log(cells.rr_lo)) / 3.92
    het = []
    for (d, t), g in cells.groupby(["drug", "term"]):
        if len(g) < 2:
            continue
        w = 1 / g.se ** 2; mu = (w * g.log_rr).sum() / w.sum()
        q = float((w * (g.log_rr - mu) ** 2).sum())
        het.append(dict(drug=d, term=t, routes=",".join(g.route_family), k=len(g), Q=q,
                        p=float(stats.chi2.sf(q, len(g) - 1)), afs=",".join(f"{a:.2f}" for a in g.af)))
    het = pd.DataFrame(het)
    frac_het = float((het.p < .01).mean()) if len(het) else float("nan")
    print(f"\ndrug-term pairs estimable in >=2 routes: {len(het):,}; heterogeneous (Q P<0.01): {frac_het:.1%}")
    if len(het):
        print(het.sort_values("p").head(12).to_string(index=False))
    # ---- route-specific probability table ----------------------------------------------
    with pv.connect(read_only=True) as con:
        ntr = dict(con.execute("SELECT drug || '|' || route_family, count(DISTINCT nct) FROM atlas_contrast_route "
                               "GROUP BY 1").fetchall())
    nm = {}
    with pv.connect(read_only=True) as con:
        gens = [r[0] for r in con.execute("SELECT DISTINCT generic_name FROM fda_label_route_pool").fetchall()]
    for g in sorted(gens, key=len):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                nm.setdefault(c, g)
    elig = cells[cells.on_route_label].stratum.unique()
    top = sorted(((ntr.get(s, 0), s) for s in elig), reverse=True)[:20]
    pt = pd.DataFrame([dict(chembl_id=s.split(SEP)[0], drug_name=nm.get(s.split(SEP)[0], s.split(SEP)[0]),
                            route_family=s.split(SEP)[1], n_trials_stratum=int(n), term=r.term,
                            n_trials=int(r.n_trials), risk_drug=r.risk_drug, risk_pl=r.risk_pl,
                            af=r.af, af_lo=r.af_lo, af_hi=r.af_hi)
                       for n, s in top for r in cells[(cells.stratum == s) & cells.on_route_label].itertuples()])
    print(f"\nprobability table: {pt[['chembl_id', 'route_family']].drop_duplicates().shape[0]} strata, {len(pt)} rows")
    print(pt.groupby(["drug_name", "route_family"]).n_trials_stratum.first().sort_values(ascending=False).to_string())
    vr = cells[cells.drug.isin(pt[pt.drug_name.str.startswith("varenicline")].chembl_id)]
    if len(vr):
        print("\nvarenicline by route (route-matched label cells):")
        print(vr[vr.on_route_label].sort_values(["route_family", "af"], ascending=[True, False])
              [["route_family", "term", "n_trials", "af", "af_lo", "af_hi"]].head(20).to_string(index=False))
    print(f"\naf_route criteria failed: {fail or 'none'} -> {'PASS' if not fail else 'FAIL'}")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('atlas_contrast_route','fda_label_route_pool','atlas_contrast')").fetchall()]
        keep = ["drug", "route_family", "term", "n_trials", "events_drug", "n_drug", "events_pl", "n_pl",
                "risk_drug", "risk_pl", "rr", "rr_lo", "rr_hi", "af", "af_lo", "af_hi", "p", "fdr",
                "has_route_label", "on_route_label"]
        c = cells[keep].copy(); c["run_id"] = run.run_id; pt["run_id"] = run.run_id
        run.archive_table("af_route_cell", c, upstream=up, mode="replace_all")
        run.archive_table("af_route_probability", pt, upstream=up, mode="replace_all")
        run.evaluate_gate("af_route", float(len(fail)), not fail, detail={
            "failed": fail, "main": main_s, "text_coded_only": sens,
            "shift_vs_v2": {"af_hi_lt_half": main_s["af_hi_lt_half"] - .486, "af_le_zero": main_s["af_le_zero"] - .255},
            "heterogeneity": {"n_pairs_multi_route": int(len(het)), "frac_q_p_lt_01": frac_het},
            "n_cells": int(len(cells)), "n_strata": int(cells.stratum.nunique()),
            "prob_table_strata": int(pt[["chembl_id", "route_family"]].drop_duplicates().shape[0])})
    return 0


if __name__ == "__main__":
    sys.exit(main())
