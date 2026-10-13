#!/usr/bin/env python3
"""AF 3.0 benchmark 1 (gate af3_sim): known-truth simulation on real trial structures, three truths (model-consistent,
misspecified link, heavy-tailed heterogeneity), AF 3.0 versus AF 1.0 on the same simulated counts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

SEED = 20260927


def simulate(scn, K, arms, cf, X, rng):
    C = len(cf)
    bt = dict(zip(cf.term.unique(), rng.normal(0, .3, cf.term.nunique())))
    bd = dict(zip(cf.drug.unique(), rng.normal(0, .2, cf.drug.nunique())))
    psi = np.where(rng.random(C) < .4, 0.0, 0.4 + cf.term.map(bt).values + cf.drug.map(bd).values + rng.normal(0, .5, C))
    cell = K.cell.values
    alpha = np.log(-np.log1p(-(K.x0 + .5) / (K.n0 + 1))).values
    b = np.zeros(len(af3.COVS))
    b[af3.COVS.index("lwk")], b[af3.COVS.index("open_label")], b[af3.COVS.index("nonsys")] = -.1, .3, -.15
    if scn == "S1":
        zeta = psi[cell] + X.values @ b + rng.normal(0, .2, len(K)); gam, tau = .3, .2
    elif scn == "S3":
        socs = cf.s.values[cell]
        bs = b[None, :] + rng.normal(0, .1, (cf.s.max() + 1, len(b))) * (b != 0)[None, :]
        zeta = psi[cell] + np.einsum("ij,ij->i", X.values, bs[socs]) + .15 * rng.standard_t(3, len(K)); gam, tau = .3, np.nan
    else:
        zeta = psi[cell] + rng.normal(0, .1, len(K)); gam, tau = 0.0, .1
    k = arms.k.values
    if scn == "S2":
        p0 = -np.expm1(-np.exp(alpha))
        p = np.minimum(p0[k] * np.exp(zeta[k]), .95)
    else:
        p = -np.expm1(-np.exp(np.minimum(alpha[k] + zeta[k] + gam * arms.ldose_c.values, 3)))
    a = arms.copy(); a["x"] = rng.binomial(a.n.values.astype(int), p)
    KK = K.copy(); KK["x0"] = rng.binomial(K.n0.values.astype(int), -np.expm1(-np.exp(alpha)))
    return KK, a, psi, tau


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "AF 3.0"          # 'AF 3.0' (gate af3_sim) or 'AF 3.2' (gate af32_sim)
    gate = {"AF 3.0": "af3_sim", "AF 3.2": "af32_sim", "AF 3.3": "af33_sim"}[version]
    fit_fn = af3.em_fit_mix if version in ("AF 3.2", "AF 3.3") else None
    extra = dict(spike=True) if version == "AF 3.3" else {}
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        tr = np.array(sorted(K.nct.unique()))
        keep = set(rng.choice(tr, int(len(tr) * .15), replace=False))
        P0 = af3.prepare(con, arms, K, trials=keep)
    rows, det = [], {}
    for scn in ("S1", "S2", "S3"):
        r = np.random.default_rng(SEED + int(scn[1]))
        KK, aa, psi, tau_true = simulate(scn, P0["K"], P0["arms"], P0["cf"], P0["X"], r)
        with pv.connect(read_only=True) as con:
            P = af3.prepare(con, aa, KK, cf_all=P0["cf"][["drug", "term", "atc3", "atc1", "soc", "mech", "atc1_soc", "atc3_term", "mech_soc"]])
        fit, S, _, _ = af3.run_fit(P, fit_fn=fit_fn, outer=2, inner=6, verbose=False, **extra)
        g = af3.af1_table(P["K"], P["arms"])
        truth = pd.Series(psi, index=np.arange(len(psi)))                       # cell order of P equals P0 (same K rows)
        ev = S.loc[g.index]; t = truth.loc[g.index]
        cov3 = float(((t >= ev.psi_lo) & (t <= ev.psi_hi)).mean()); cov1 = float((abs(g.theta - t) <= 1.96 * g.se).mean())
        rm3 = float(np.sqrt(((ev.psi_mean - t) ** 2).mean())); rm1 = float(np.sqrt(((g.theta - t) ** 2).mean()))
        ok = bool(.90 <= cov3 <= .98 and rm3 <= rm1)
        rows.append(dict(scenario=scn, n_cells_af1=int(len(g)), coverage_af3=cov3, coverage_af1=cov1, rmse_af3=rm3, rmse_af1=rm1,
                         tau_true=tau_true, tau_est_median=float(np.median(fit["tau"])), passed=ok))
        print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    n_ok = int(D.passed.sum())
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('atlas_contrast','ctgov_event_group','ctgov_trial','ctgov_baseline','ctgov_condition','atlas_contrast_route',
                   'ref_chembl_drug','ref_chembl_mechanism')""").fetchall()]
        D["run_id"] = run.run_id; D["version"] = version
        with pv.connect() as con:
            con.execute("DELETE FROM af3_sim WHERE version = ?", [version])
        run.archive_table("af3_sim", D, upstream=up)
        run.evaluate_gate(gate, float(n_ok), n_ok == 3, detail=dict(version=version, scenarios=rows, gamma_info=P0["gamma_info"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
