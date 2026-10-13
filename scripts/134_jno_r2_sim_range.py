#!/usr/bin/env python3
"""Gate jno_r2_sim_range (rule in 133_jno_r2_prereg.py): known-truth simulation across true shares.

Real trial structures of af33_sim (15% of trials, seed 20260927); truth without exact zeros as in
129_jno_rev_sim_nozero.py, with intercept 0.45 (true share near 70%), 0.70 (near 50%), and a heavy-tailed scenario
(intercept 0.15, cell residual t(3) scaled to SD 0.35). Each truth is fitted by the adopted estimator.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("s129", ROOT / "scripts" / "129_jno_rev_sim_nozero.py")
S129 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S129)
SEED = 20260927
SCEN = {"I045": dict(icpt=.45, tail=None), "I070": dict(icpt=.70, tail=None), "T3": dict(icpt=.15, tail=3)}


def simulate(K, arms, cf, X, rng, icpt, tail):
    C = len(cf)
    bt = dict(zip(cf.term.unique(), rng.normal(0, .2, cf.term.nunique())))
    bd = dict(zip(cf.drug.unique(), rng.normal(0, .15, cf.drug.nunique())))
    if tail is None:
        e = rng.normal(0, .35, C)
    else:
        e = rng.standard_t(tail, C) * .35 / np.sqrt(tail / (tail - 2))
    psi = icpt + cf.term.map(bt).values + cf.drug.map(bd).values + e
    cell = K.cell.values
    alpha = np.log(-np.log1p(-(K.x0 + .5) / (K.n0 + 1))).values
    b = np.zeros(len(af3.COVS))
    b[af3.COVS.index("lwk")], b[af3.COVS.index("open_label")], b[af3.COVS.index("nonsys")] = -.1, .3, -.15
    zeta = psi[cell] + X.values @ b + rng.normal(0, .2, len(K))
    k = arms.k.values
    p = -np.expm1(-np.exp(np.minimum(alpha[k] + zeta[k] + .3 * arms.ldose_c.values, 3)))
    a = arms.copy(); a["x"] = rng.binomial(a.n.values.astype(int), p)
    KK = K.copy(); KK["x0"] = rng.binomial(K.n0.values.astype(int), -np.expm1(-np.exp(alpha)))
    return KK, a, psi


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        tr = np.array(sorted(K.nct.unique()))
        keep = set(rng.choice(tr, int(len(tr) * .15), replace=False))
        P0 = af3.prepare(con, arms, K, trials=keep)
    ntr = P0["K"].groupby("cell").nct.nunique().reindex(np.arange(len(P0["cf"]))).values
    rows = []
    for i, (scn, par) in enumerate(SCEN.items()):
        r = np.random.default_rng(SEED + 10 + i)
        KK, aa, psi = simulate(P0["K"], P0["arms"], P0["cf"], P0["X"], r, **par)
        with pv.connect(read_only=True) as con:
            P = af3.prepare(con, aa, KK, cf_all=P0["cf"][S129.CF_COLS])
        fit, S, _, _ = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, verbose=False, spike=True)
        m = S129.evaluate(S, psi, ntr)
        m.update(scenario=scn)
        rows.append(m); print(m, flush=True)
    worst = float(max(abs(r_["share_error"]) for r_ in rows))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        sm = []
        for r_ in rows:
            for k in ("coverage", "rmse", "est_share", "true_share", "share_error", "est_share_1trial",
                      "true_share_1trial", "brier"):
                sm.append(dict(analysis="jno_r2_sim_range", set_name=r_["scenario"], grp="all", n_pairs=int(len(ntr)),
                               n_drugs=0, metric=k, value=r_[k], lo=np.nan, hi=np.nan))
        sm = pd.DataFrame(sm); sm["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_r2_sim_range'")
        run.archive_table("jno_summary", sm)
        run.evaluate_gate("jno_r2_sim_range", worst, bool(worst <= .08), detail=dict(rows=rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
