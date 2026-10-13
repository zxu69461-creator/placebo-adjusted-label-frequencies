#!/usr/bin/env python3
"""AF 3.0 benchmarks 2-4 (gates af3_prospective, af3_calibration, af3_random_holdout), descriptive headline and
ablations (af3_headline), and the adoption recommendation (af3_adopt). Every method is fitted on training trials only
and scored on the drug-arm counts of held-out trials with the same proper scoring rule."""
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af2, af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

SEED = 20260927
FEATS = ["drug", "term", "atc3", "atc1", "soc", "mech", "atc1_soc", "atc3_term", "mech_soc"]
PIT_STORE = {}


def normal_grid(mu, var):
    v = np.maximum(var, 1e-6)[:, None]
    q = np.exp(-0.5 * (af3.GRID[None, :] - mu[:, None]) ** 2 / v)
    q[~np.isfinite(q)] = 0
    s = q.sum(1, keepdims=True)
    q = np.where(s > 0, q / np.where(s > 0, s, 1), 0)
    zero = s[:, 0] == 0                                    # very precise and off-grid: put all mass on nearest node
    if zero.any():
        idx = np.clip(np.rint((mu[zero] - af3.GRID[0]) / af3.DZ).astype(int), 0, len(af3.GRID) - 1)
        q[np.where(zero)[0], idx] = 1.0
    return q


def af2_train(train):
    spec = importlib.util.spec_from_file_location("s83", ROOT / "scripts" / "83_af2_fit.py")
    s83 = importlib.util.module_from_spec(spec); sys.modules["s83"] = s83; spec.loader.exec_module(s83)
    with pv.connect(read_only=True) as con:
        ct, ae, enf = af2.load_arms(con)
    cells = s83.fit_all(af2.build_trials(ct, ae, enf, keep_ncts=set(train)))
    with pv.connect(read_only=True) as con:
        post, info = s83.shrink(con, cells)
    return post[["drug", "term", "theta2", "se2"]]


def evaluate_split(name, train, test, arms, K, cf_all, fits_extra=(), p0_mode="plugin"):
    with pv.connect(read_only=True) as con:
        P = af3.prepare(con, arms, K, trials=set(train), cf_all=cf_all)
    variants = {"AF 3.0": dict()}
    for v in fits_extra:
        variants[v] = dict(use_cov=False) if v == "AF 3.0 no covariates" else dict(gaussian_prior=True)
    fits = {}
    for v, kw in variants.items():
        print(f"[{name}] fitting {v} on {len(train):,} training trials", flush=True)
        fits[v] = af3.run_fit(P, outer=2, inner=6, verbose=False, **kw)
    g = af3.af1_table(P["K"], P["arms"])
    g = g.join(P["cf"][["drug", "term"]])
    print(f"[{name}] AF 1.0-estimable training cells {len(g):,}; fitting AF 2.0", flush=True)
    a2 = af2_train(train)
    # test records in the common support
    Kt = K[K.nct.isin(set(test))].copy()
    key = pd.MultiIndex.from_frame(P["cf"][["drug", "term"]])
    cmap = pd.Series(np.arange(len(key)), index=key)
    Kt["cell"] = cmap.reindex(pd.MultiIndex.from_frame(Kt[["drug", "t"]])).values
    Kt = Kt[Kt.cell.isin(g.index)].copy()
    Kt = Kt.merge(a2.rename(columns={"term": "t"}), on=["drug", "t"], how="left")
    Kt["kk"] = np.arange(len(Kt))
    at = arms.merge(Kt[["k", "kk", "cell"]], on="k")
    cell = Kt.cell.values.astype(int); soc = P["cf"].s.values[cell]
    Xt = af3.design_covariates(Kt).values
    p0 = ((Kt.x0 + 0.5) / (Kt.n0 + 1)).values
    ak, ax, an = at.kk.values, at.x.values.astype(float), at.n.values.astype(float)
    aoff = P["gamma"][P["cf"].s.values[at.cell.values.astype(int)]] * at.ldose_c.values
    gi = g.reindex(cell)
    preds = {"AF 1.0": ("rr", normal_grid(gi.theta.values, gi.se.values ** 2)),
             "AF 1.0-RE": ("rr", normal_grid(gi.theta_dl.values, gi.se_dl.values ** 2 + gi.tau2_dl.values)),
             "AF 2.0": ("rr", normal_grid(np.where(Kt.theta2.notna(), Kt.theta2, gi.theta.values),
                                          np.where(Kt.se2.notna(), Kt.se2 ** 2, gi.se.values ** 2) + gi.tau2_dl.values))}
    for v, (fit, S, _, _) in fits.items():
        shift = np.einsum("ij,ij->i", Xt, fit["beta"][soc]) if v != "AF 3.0 no covariates" else np.zeros(len(Kt))
        preds[v] = ("hz", af3.predictive_grid(fit["post"][cell].astype(np.float64), shift, fit["tau"][soc]))
    rng = np.random.default_rng(SEED)
    sc, pit = {}, {}
    p0n = af3.beta_nodes(Kt.x0.values.astype(float), Kt.n0.values.astype(float)) if p0_mode == "beta" else None
    for m, (scale, q) in preds.items():
        off_ = aoff if scale == "hz" else np.zeros(len(ak))
        if p0_mode == "beta":
            sc[m] = af3.score_records_beta(q, ax, an, p0n, ak, off_, scale, len(Kt))
            pit[m] = af3.pit_records_beta(q, ax, an, p0n, ak, off_, scale, len(Kt), rng)
        else:
            sc[m] = af3.score_records(q, ax, an, p0, ak, off_, scale, len(Kt))
            pit[m] = af3.pit_records(q, ax, an, p0, ak, off_, scale, len(Kt), rng)
    PIT_STORE[name + ":" + p0_mode] = pd.DataFrame(pit)
    D = pd.DataFrame(sc); D["nct"] = Kt.nct.values
    comp = ["AF 1.0", "AF 1.0-RE", "AF 2.0"]
    best = max(comp, key=lambda m: D[m].mean())
    tsum = D.groupby("nct")[list(sc)].sum(); tcnt = D.groupby("nct").size()
    bs = np.random.default_rng(SEED)
    idx = [bs.integers(0, len(tsum), len(tsum)) for _ in range(2000)]
    rows = []
    for m in sc:
        diffs = np.array([(tsum["AF 3.0"].values[i].sum() - tsum[m].values[i].sum()) / tcnt.values[i].sum() for i in idx])
        rows.append(dict(split=name, method=m, n_records=int(len(D)), n_trials=int(len(tsum)), mean_logscore=float(D[m].mean()),
                         diff_vs_af3=float(D["AF 3.0"].mean() - D[m].mean()), diff_ci_lo=float(np.percentile(diffs, 2.5)),
                         diff_ci_hi=float(np.percentile(diffs, 97.5)),
                         pit_central90=float(((pit[m] >= .05) & (pit[m] <= .95)).mean())))
    R = pd.DataFrame(rows)
    print(R.to_string(), flush=True)
    return R, best


def record(Rp, Rr, best_p, best_r, cut):
    """Evaluate the registered gates from the two score tables (also used with --evaluate-only on the archived scores)."""
    Rp = Rp.drop(columns=[c for c in ("run_id",) if c in Rp]); Rr = Rr.drop(columns=[c for c in ("run_id",) if c in Rr])
    rp = Rp.set_index("method").loc[best_p]; rr = Rr.set_index("method").loc[best_r]
    cal = float(Rp.set_index("method").loc["AF 3.0", "pit_central90"])
    pass_p = bool(rp.diff_ci_lo > 0); pass_c = bool(.87 <= cal <= .93); pass_r = bool(rr.diff_ci_lo > 0)
    with pv.connect(read_only=True) as con:
        q = """SELECT count(*) n, avg((c.p_af_lt_half > .975)::INT) conf_bg, avg((c.af_risk12 < .5)::INT) risk_lt_half,
                      avg((c.af_hz < .5)::INT) hz_lt_half, avg((c.p_no_excess > .5)::INT) no_excess
               FROM af3_cell c JOIN af_label_match m USING (drug, term) WHERE m.on_fda"""
        hd = con.execute(q + " AND c.af1_estimable").df().iloc[0].to_dict()
        ha = con.execute(q).df().iloc[0].to_dict()
        sim = con.execute("SELECT passed FROM prov_decision WHERE gate_name='af3_sim'").fetchone()[0]
        val = con.execute("SELECT passed FROM prov_decision WHERE gate_name='af3_validity'").fetchone()[0]
        up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN ('af3_score','af3_cell')").fetchall()]
    flags = dict(af3_sim=bool(sim), af3_prospective=pass_p, af3_calibration=pass_c, af3_validity=bool(val))
    n_not = sum(not v for v in flags.values())
    return dict(rp=rp, rr=rr, cal=cal, pass_p=pass_p, pass_c=pass_c, pass_r=pass_r, hd=hd, ha=ha, flags=flags, n_not=n_not,
                Rp=Rp, Rr=Rr, best_p=best_p, best_r=best_r, cut=cut, up=up)


def write_gates(run, E):
    Rp, Rr = E["Rp"], E["Rr"]
    run.evaluate_gate("af3_prospective", float(E["rp"].diff_ci_lo), E["pass_p"],
                      detail=dict(best_competitor=E["best_p"], cutoff=E["cut"], table=Rp.to_dict("records")))
    run.evaluate_gate("af3_calibration", E["cal"], E["pass_c"], detail=dict(table=Rp[["method", "pit_central90"]].to_dict("records")))
    run.evaluate_gate("af3_random_holdout", float(E["rr"].diff_ci_lo), E["pass_r"],
                      detail=dict(best_competitor=E["best_r"], table=Rr.to_dict("records")))
    run.evaluate_gate("af3_headline", None, None, detail=dict(label_cells_af1_estimable=E["hd"], label_cells_all=E["ha"],
                      ablations=Rp[Rp.method.str.startswith("AF 3.0")][["method", "mean_logscore", "diff_vs_af3", "diff_ci_lo",
                                                                           "diff_ci_hi", "pit_central90"]].to_dict("records")))
    run.evaluate_gate("af3_adopt", float(E["n_not"]), E["n_not"] == 0, detail=dict(flags=E["flags"],
                      recommendation="AF 3.0 recommended as primary" if E["n_not"] == 0 else "AF 3.0 remains an exploration-branch result"))


def main() -> int:
    if "--evaluate-only" in sys.argv:
        with pv.connect(read_only=True) as con:
            R = con.execute("SELECT * FROM af3_score").df()
            K = af3.build(con)[1]
        st = K.groupby("nct").start.first(); cut = f"{st.dropna().quantile(.7):%Y-%m-%d}"
        Rp, Rr = R[R.split == "prospective"], R[R.split == "random"]
        comp = ["AF 1.0", "AF 1.0-RE", "AF 2.0"]
        best_p = max(comp, key=lambda m: float(Rp.set_index("method").loc[m, "mean_logscore"]))
        best_r = max(comp, key=lambda m: float(Rr.set_index("method").loc[m, "mean_logscore"]))
        E = record(Rp, Rr, best_p, best_r, cut)
        with pv.run_context(__file__, env_name="oajoint-py311") as run:
            write_gates(run, E)
        print(json.dumps(dict(flags=E["flags"], calibration=E["cal"], headline_af1_estimable=E["hd"], headline_all=E["ha"]), indent=1, default=str))
        return 0
    mp.set_start_method("fork", force=True)
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"})
        cf_all = af3.cell_features(con, cells)[FEATS]
    st = K.groupby("nct").start.first()
    cut = st.dropna().quantile(.7)
    train_p = sorted(st.index[(st < cut) | st.isna()]); test_p = sorted(st.index[st >= cut])
    print(f"prospective split at {cut:%Y-%m}: {len(train_p):,} training / {len(test_p):,} test trials", flush=True)
    Rp, best_p = evaluate_split("prospective", train_p, test_p, arms, K, cf_all,
                                fits_extra=("AF 3.0 no covariates", "AF 3.0 Gaussian prior"))
    tr = np.array(sorted(K.nct.unique()))
    rnd = np.random.default_rng(20260924).random(len(tr)) < .8
    Rr, best_r = evaluate_split("random", list(tr[rnd]), list(tr[~rnd]), arms, K, cf_all)
    R = pd.concat([Rp, Rr])
    E = record(Rp, Rr, best_p, best_r, f"{cut:%Y-%m-%d}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        R["run_id"] = run.run_id
        run.archive_table("af3_score", R, upstream=E["up"], mode="replace_all")
        write_gates(run, E)
    print(json.dumps(dict(flags=E["flags"], headline_af1_estimable=E["hd"], headline_all=E["ha"]), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
