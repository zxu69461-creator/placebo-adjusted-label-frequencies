#!/usr/bin/env python3
"""Gate jno_rev_sim_nozero: known-truth simulation without exact zeros (rule in 126_jno_review_prereg.py).

Real trial structures of af33_sim (15% of trials, seed 20260927). Truth 'S4': psi = 0.15 + term N(0, 0.2) + drug
N(0, 0.15) + N(0, 0.35), no exact zeros; trial level as scenario S1 of 93_af3_sim.py. Truth 'S1' (40% exact zeros) is
repeated for comparison. Each truth is fitted by the adopted estimator (zero-excess component) and without it. The key
metric is the calibration of the primary summary: estimated share (mean P(AF < 0.5)) minus true share (psi < log 2).
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

spec = importlib.util.spec_from_file_location("s93", ROOT / "scripts" / "93_af3_sim.py")
S93 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S93)
SEED = 20260927
CF_COLS = ["drug", "term", "atc3", "atc1", "soc", "mech", "atc1_soc", "atc3_term", "mech_soc"]


def simulate_nozero(K, arms, cf, X, rng):
    C = len(cf)
    bt = dict(zip(cf.term.unique(), rng.normal(0, .2, cf.term.nunique())))
    bd = dict(zip(cf.drug.unique(), rng.normal(0, .15, cf.drug.nunique())))
    psi = .15 + cf.term.map(bt).values + cf.drug.map(bd).values + rng.normal(0, .35, C)
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


def evaluate(S, psi, ntr):
    t = pd.Series(psi, index=np.arange(len(psi)))
    ev = S.loc[t.index]
    true_lt = (t < np.log(2)).astype(float)
    one = (ntr == 1)
    return dict(coverage=float(((t >= ev.psi_lo) & (t <= ev.psi_hi)).mean()),
                rmse=float(np.sqrt(((ev.psi_mean - t) ** 2).mean())),
                est_share=float(ev.p_af_lt_half.mean()), true_share=float(true_lt.mean()),
                share_error=float(ev.p_af_lt_half.mean() - true_lt.mean()),
                est_share_1trial=float(ev.p_af_lt_half[one].mean()), true_share_1trial=float(true_lt[one].mean()),
                brier=float(((ev.p_af_lt_half - true_lt) ** 2).mean()),
                true_zero_share=float((t == 0).mean()))


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        tr = np.array(sorted(K.nct.unique()))
        keep = set(rng.choice(tr, int(len(tr) * .15), replace=False))
        P0 = af3.prepare(con, arms, K, trials=keep)
    ntr = P0["K"].groupby("cell").nct.nunique().reindex(np.arange(len(P0["cf"]))).values
    rows = []
    for scn in ("S4", "S1"):
        r = np.random.default_rng(SEED + (4 if scn == "S4" else 1))
        if scn == "S4":
            KK, aa, psi = simulate_nozero(P0["K"], P0["arms"], P0["cf"], P0["X"], r)
        else:
            KK, aa, psi, _ = S93.simulate("S1", P0["K"], P0["arms"], P0["cf"], P0["X"], r)
        with pv.connect(read_only=True) as con:
            P = af3.prepare(con, aa, KK, cf_all=P0["cf"][CF_COLS])
        for spike in (True, False):
            fit, S, _, _ = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, verbose=False, spike=spike)
            m = evaluate(S, psi, ntr)
            m.update(scenario=scn, estimator="zero-excess component" if spike else "without zero-excess component")
            rows.append(m); print(m, flush=True)
    D = pd.DataFrame(rows)
    main_err = float(D[(D.scenario == "S4") & (D.estimator == "zero-excess component")].share_error.iloc[0])
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        sm = []
        for r_ in rows:
            for k in ("coverage", "rmse", "est_share", "true_share", "share_error", "est_share_1trial",
                      "true_share_1trial", "brier"):
                sm.append(dict(analysis="jno_rev_sim_nozero", set_name=f"{r_['scenario']}|{r_['estimator']}", grp="all",
                               n_pairs=int(len(ntr)), n_drugs=0, metric=k, value=r_[k], lo=np.nan, hi=np.nan))
        sm = pd.DataFrame(sm); sm["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_rev_sim_nozero'")
        run.archive_table("jno_summary", sm)
        run.evaluate_gate("jno_rev_sim_nozero", main_err, bool(abs(main_err) <= .05), detail=dict(rows=rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
