#!/usr/bin/env python3
"""Indication heterogeneity of attributable fractions (gate af_indication).

Mantel-Haenszel RR with Greenland-Robins variance, vectorised (identical to af_common.mh_rr), so the
500-permutation null is tractable. Rows are first summed to (drug, trial, term): indication is a trial
property, so permutation happens at trial level within each drug.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("v2", ROOT / "scripts" / "52_af_v2.py")
V2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V2)
N_PERM, SEED = 500, 20260924


def trial_rows():
    r = V2.canon_rows()
    T = r.n1 + r.n0
    r["R"], r["S"] = r.a * r.n0 / T, r.c * r.n1 / T
    r["V"] = (r.n1 * r.n0 * (r.a + r.c) - r.a * r.c * T) / T ** 2
    r["ev"] = r.a + r.c
    return r.groupby(["drug", "nct", "term"], as_index=False)[["R", "S", "V", "ev"]].sum()


def cells(tr, ind):
    """ind: Series indexed like tr giving each row's indication."""
    g = tr.assign(ind=ind).groupby(["drug", "ind", "term"], observed=True).agg(
        R=("R", "sum"), S=("S", "sum"), V=("V", "sum"), ev=("ev", "sum"), n=("R", "size")).reset_index()
    g = g[(g.n >= 2) & (g.ev >= 20) & (g.R > 0) & (g.S > 0) & (g.V > 0)].copy()
    g["y"] = np.log(g.R / g.S); g["v"] = g.V / (g.R * g.S)
    return g


def q_share(g):
    g = g.assign(w=1 / g.v)
    g["wy"], g["wy2"] = g.w * g.y, g.w * g.y ** 2
    p = g.groupby(["drug", "term"]).agg(k=("w", "size"), sw=("w", "sum"), swy=("wy", "sum"), swy2=("wy2", "sum"))
    p = p[p.k >= 2]
    q = p.swy2 - p.swy ** 2 / p.sw
    pv_ = stats.chi2.sf(q, p.k - 1)
    return float((pv_ < .01).mean()) if len(p) else np.nan, p.assign(Q=q, p=pv_, I2=np.maximum(0, (q - (p.k - 1)) / q))


def main() -> int:
    tr = trial_rows()
    with pv.connect(read_only=True) as con:
        ind = dict(con.execute("SELECT nct, indication FROM ctgov_condition WHERE indication IS NOT NULL").fetchall())
        names = dict(con.execute("SELECT chembl_id, display_name FROM ref_drug_name").fetchall())
    tr = tr[tr.nct.isin(ind)].reset_index(drop=True)
    tr_ind = tr.nct.map(ind)
    obs_cells = cells(tr, tr_ind)
    obs, pairs = q_share(obs_cells)
    print(f"trial-level rows {len(tr):,}; cells {len(obs_cells):,}; pairs in >=2 indications {len(pairs):,}; "
          f"observed share Q P<0.01 = {obs:.3f}")
    # permutation null: shuffle indication across each drug's trials
    trials = tr[["drug", "nct"]].drop_duplicates().reset_index(drop=True)
    trials["ind"] = trials.nct.map(ind)
    rng = np.random.default_rng(SEED)
    grp = trials.groupby("drug").indices
    null = []
    for b in range(N_PERM):
        perm = trials.ind.values.copy()
        for d, ix in grp.items():
            if len(ix) > 1:
                perm[ix] = perm[rng.permutation(ix)]
        m = dict(zip(zip(trials.drug, trials.nct), perm))
        pind = pd.Series([m[k] for k in zip(tr.drug, tr.nct)])
        null.append(q_share(cells(tr, pind))[0])
        if (b + 1) % 100 == 0:
            print(f"  permutation {b + 1}: running q95 {np.quantile(null, .95):.3f}", flush=True)
    null = np.array(null); q95 = float(np.quantile(null, .95))
    passed = bool(obs > q95)
    print(f"\nobserved {obs:.3f} vs permutation mean {null.mean():.3f}, 95th pct {q95:.3f}, "
          f"P_perm = {(1 + (null >= obs).sum()) / (1 + N_PERM):.4f} -> af_indication {'PASS' if passed else 'FAIL'}")
    pairs = pairs.reset_index()
    pairs["name"] = pairs.drug.map(names)
    top = pairs.sort_values("Q", ascending=False).head(15)
    oc = obs_cells.assign(af=1 - np.exp(-obs_cells.y))
    print("\nmost heterogeneous pairs:")
    for r in top.itertuples():
        sub = oc[(oc.drug == r.drug) & (oc.term == r.term)].sort_values("af")
        print(f"  {r.name} -> {r.term}: Q={r.Q:.1f} k={r.k} I2={r.I2:.2f} | " +
              "; ".join(f"{i[:28]} {a:+.2f} ({n})" for i, a, n in zip(sub.ind, sub.af, sub.n)))
    print(f"\nI2 among multi-indication pairs: median {pairs.I2.median():.2f}; share I2>0.5 {(pairs.I2 > .5).mean():.1%}")
    # healthy volunteers vs patients
    hv = tr_ind.eq("healthy volunteers")
    hc = cells(tr, pd.Series(np.where(hv, "healthy", "patients")))
    w = hc.pivot_table(index=["drug", "term"], columns="ind", values="y").dropna()
    hvp = None
    if len(w):
        diff = (1 - np.exp(-w.patients)) - (1 - np.exp(-w.healthy))
        hvp = dict(n_pairs=int(len(w)), median_af_healthy=float((1 - np.exp(-w.healthy)).median()),
                   median_af_patients=float((1 - np.exp(-w.patients)).median()), median_diff=float(diff.median()))
        print(f"\nhealthy vs patients (same drug-term): {hvp}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('ctgov_condition','atlas_contrast')").fetchall()]
        se = np.sqrt(obs_cells.v)
        out = pd.DataFrame(dict(drug=obs_cells.drug, indication=obs_cells.ind, term=obs_cells.term, n_trials=obs_cells.n,
                                events=obs_cells.ev, rr=np.exp(obs_cells.y), rr_lo=np.exp(obs_cells.y - 1.96 * se),
                                rr_hi=np.exp(obs_cells.y + 1.96 * se)))
        out["af"], out["af_lo"], out["af_hi"] = 1 - 1 / out.rr, 1 - 1 / out.rr_lo, 1 - 1 / out.rr_hi
        out["run_id"] = run.run_id
        run.archive_table("af_indication_cell", out, upstream=up, mode="replace_all")
        run.evaluate_gate("af_indication", obs - q95, passed, detail={
            "observed_share": obs, "perm_mean": float(null.mean()), "perm_q95": q95,
            "p_perm": float((1 + (null >= obs).sum()) / (1 + N_PERM)), "n_pairs": int(len(pairs)),
            "n_cells": int(len(obs_cells)), "i2_median": float(pairs.I2.median()),
            "i2_gt_half": float((pairs.I2 > .5).mean()), "healthy_vs_patients": hvp,
            "top": [dict(drug=r.name, term=r.term, Q=float(r.Q), k=int(r.k)) for r in top.itertuples()]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
