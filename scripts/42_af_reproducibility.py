#!/usr/bin/env python3
"""Premise test for the attributable-fraction direction: do drug-AE effects reproduce
across independent halves of the trial corpus? (gates af_reproducibility, _temporal)

Non-serious events only. A contrast contributes to a term only if its trial LISTS the
term in the non-serious table -- an absent term means 'below the reporting threshold',
not zero (gate af_design). Risk ratio by Mantel-Haenszel with Greenland-Robins variance;
placebo counts are already divided by k in atlas_contrast.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

SEED = 20260924


def mh_rr(g):
    a, c, n1, n0 = g.a.values, g.c.values, g.n1.values, g.n0.values
    T = n1 + n0
    R, S = (a * n0 / T).sum(), (c * n1 / T).sum()
    if R <= 0 or S <= 0:
        return np.nan, np.nan
    v = ((n1 * n0 * (a + c) - a * c * T) / T ** 2).sum() / (R * S)
    return float(np.log(R / S)), float(np.sqrt(v)) if v > 0 else np.nan


def main() -> int:
    with pv.connect(read_only=True) as con:
        rows = con.execute("""
            WITH ct AS (
              SELECT contrast_id, nct, eg_active, eg_placebo, k, drug_ids[1] AS drug,
                     n_act_other AS n1, n_pl_other AS n0
              FROM atlas_contrast
              WHERE single_ingredient AND n_act_other > 0 AND n_pl_other > 0),
            ev AS (
              SELECT nct, eg_id, lower(trim(regexp_replace(term, '\\s+', ' ', 'g'))) AS term,
                     max(num_affected) AS e
              FROM v_ctgov_ae WHERE kind = 'other' AND num_affected IS NOT NULL
              GROUP BY 1, 2, 3),
            listed AS (SELECT DISTINCT nct, term FROM ev),
            pl AS (
              SELECT ct.contrast_id, ev.term, sum(ev.e) / any_value(ct.k) AS c
              FROM ct, unnest(ct.eg_placebo) AS u(pid)
              JOIN ev ON ev.nct = ct.nct AND ev.eg_id = u.pid
              GROUP BY 1, 2)
            SELECT ct.contrast_id, ct.nct, ct.drug, l.term, ct.n1, ct.n0,
                   coalesce(ea.e, 0) AS a, coalesce(pl.c, 0) AS c
            FROM ct JOIN listed l ON l.nct = ct.nct
            LEFT JOIN ev ea ON ea.nct = ct.nct AND ea.eg_id = ct.eg_active AND ea.term = l.term
            LEFT JOIN pl ON pl.contrast_id = ct.contrast_id AND pl.term = l.term
        """).df()
        start = con.execute("SELECT nct, start_date FROM ctgov_trial").df()
    print(f"contrast x listed-term rows: {len(rows):,}   trials: {rows.nct.nunique():,}   "
          f"drugs: {rows.drug.nunique():,}   terms: {rows.term.nunique():,}")

    trials = np.array(sorted(rows.nct.unique()))
    rng = np.random.default_rng(SEED)
    half_rand = dict(zip(trials, np.where(rng.random(len(trials)) < 0.5, "A", "B")))
    start["d"] = pd.to_datetime(start.start_date, errors="coerce")
    med = start[start.nct.isin(trials)].d.median()
    half_time = {n: ("A" if (d == d and d <= med) else "B")
                 for n, d in zip(start.nct, start.d) if n in half_rand}
    print(f"temporal split at start date {med.date()}")

    out, results = [], {}
    for split, hmap, gate in (("random", half_rand, "af_reproducibility"),
                              ("temporal", half_time, "af_reproducibility_temporal")):
        r = rows.assign(half=rows.nct.map(hmap)).dropna(subset=["half"])
        est = {}
        for (h, d, t), g in r.groupby(["half", "drug", "term"], sort=False):
            if g.nct.nunique() < 2 or (g.a.sum() + g.c.sum()) < 20:
                continue
            lr, se = mh_rr(g)
            if lr != lr or se != se:
                continue
            p = float(2 * stats.norm.sf(abs(lr / se)))
            est[(h, d, t)] = (lr, se, p)
            out.append(dict(split=split, half=h, drug=d, term=t, n_trials=int(g.nct.nunique()),
                            events_drug=float(g.a.sum()), n_drug=float(g.n1.sum()),
                            events_pl=float(g.c.sum()), n_pl=float(g.n0.sum()),
                            log_rr=lr, se=se, p=p))
        both = sorted({(d, t) for (h, d, t) in est if h == "A"} &
                      {(d, t) for (h, d, t) in est if h == "B"})
        A = np.array([est[("A",) + c][0] for c in both]); B = np.array([est[("B",) + c][0] for c in both])
        rho = float(stats.spearmanr(A, B).statistic) if len(both) > 2 else float("nan")
        sigA = [c for c in both if est[("A",) + c][2] < 0.01]
        agree = (np.mean([np.sign(est[("A",) + c][0]) == np.sign(est[("B",) + c][0]) for c in sigA])
                 if sigA else float("nan"))
        enough = len(both) >= 200 and len(sigA) >= 30
        passed = bool(rho >= 0.50 and agree >= 0.80) if enough else None
        results[gate] = dict(split=split, n_cells_both=len(both), spearman=rho,
                             n_sigA=len(sigA), sign_agreement=float(agree), passed=passed)
        print(f"\n[{split}] cells estimable in both halves: {len(both):,}   "
              f"Spearman(logRR A, B) = {rho:.3f}")
        print(f"          cells with P<0.01 in half A: {len(sigA):,}   "
              f"sign agreement in half B: {agree:.1%}")
        print(f"          -> {gate}: "
              f"{'UNEVALUABLE' if passed is None else ('PASS' if passed else 'FAIL')}")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute(
                "SELECT out_id FROM prov_output_asset WHERE object_name='atlas_contrast'").fetchall()]
        df = pd.DataFrame(out); df["run_id"] = run.run_id
        run.archive_table("af_split_estimate", df, upstream=up, mode="replace_all")
        for gate, r in results.items():
            run.evaluate_gate(gate, r["spearman"] if r["spearman"] == r["spearman"] else None,
                              r["passed"], detail=r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
