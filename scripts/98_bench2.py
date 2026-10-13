#!/usr/bin/env python3
"""BENCH-2 runner. Usage: 98_bench2.py <bench-name> <dev|confirm> <method,...>
Methods: baselines (B0-B5 incl. AF 2.0) always; AF 3 variants by name from VARIANTS. Writes bench_record and bench_score."""
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af2, af3, bench  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

SEED = 20260927
FEATS = ["drug", "term", "atc3", "atc1", "soc", "mech", "atc1_soc", "atc3_term", "mech_soc"]
F31 = FEATS + ["drug_soc"]
M31 = ("mechs", "tokens", "atc3_tokens")
KIND = "other"
import os  # noqa: E402
BENCH22 = os.environ.get("BENCH22") == "1"
VARIANTS = {"AF 3.0": dict(fit=dict(), prep=dict()),
            "AF 3.0 nocov": dict(fit=dict(use_cov=False), prep=dict()),
            "AF 3.2": dict(fit=dict(), prep=dict(), mix=True),
            "AF 3.3": dict(fit=dict(spike=True), prep=dict(), mix=True, kappa=0.1),
            "AF 3.1a": dict(fit=dict(feats=F31, multi=M31), prep=dict()),
            "AF 3.1b": dict(fit=dict(feats=F31, multi=M31, soc_prior=200.0), prep=dict())}


def af2_fit(train):
    spec = importlib.util.spec_from_file_location("s83", ROOT / "scripts" / "83_af2_fit.py")
    s83 = importlib.util.module_from_spec(spec); sys.modules["s83"] = s83; spec.loader.exec_module(s83)
    with pv.connect(read_only=True) as con:
        ct, ae, enf = af2.load_arms(con)
    cells = s83.fit_all(af2.build_trials(ct, ae, enf, keep_ncts=set(train)))
    with pv.connect(read_only=True) as con:
        post, _ = s83.shrink(con, cells)
    return post[["drug", "term", "theta2", "se2"]].rename(columns={"term": "t"})


def run_window(name, cut, end, arms, K, cf_all, variants, prep_fn=None):
    train, test = bench.split(K, cut, end)
    Ktr = K[K.nct.isin(train)]
    c, TT, _ = bench.baselines(Ktr, arms[arms.k.isin(Ktr.k)])
    Kt = K[K.nct.isin(test)].copy()
    seen = set(zip(Ktr.drug, Ktr.t))
    Kt["k_train"] = pd.Series(list(zip(Kt.drug, Kt.t))).map(Ktr.groupby(["drug", "t"]).nct.nunique().to_dict()).fillna(0).astype(int).values
    Kt = Kt.reset_index(drop=True); Kt["kk"] = np.arange(len(Kt))
    at = arms.merge(Kt[["k", "kk"]], on="k")
    keys = list(zip(Kt.drug, Kt.t))
    preds, ci = bench.baseline_predictives(c, TT, keys, Kt.t.values)
    # B6 class-mean and B7 class-informed Bayesian meta-analysis (BENCH-2.2 baselines)
    drug_atc3 = cf_all.drop_duplicates("drug").set_index("drug").atc3.to_dict()
    d_tr = bench.trial_records(Ktr, arms[arms.k.isin(Ktr.k)])
    CM = bench.class_mean(d_tr, drug_atc3)
    cik = CM.reindex(pd.MultiIndex.from_arrays([Kt.drug.map(drug_atc3).values, Kt.t.values]))
    ok6 = cik.cls_mu.notna().values
    q6 = np.full((len(Kt), len(af3.GRID)), np.nan)
    q6[ok6] = bench._grid_normal(cik.cls_mu.values[ok6], cik.cls_se2.values[ok6] + cik.cls_t2.values[ok6])
    preds["B6 class-mean"] = q6
    if BENCH22:
        preds["B7 class-Bayes"] = bench.class_bayes_predictive(c, CM, keys, drug_atc3)
    Kt["af1_support"] = ci.mh_ok.fillna(False).values.astype(bool)
    print(f"[{name}] train {len(train):,} trials, test {len(test):,} trials, {len(Kt):,} records "
          f"({Kt.af1_support.sum():,} in AF 1.0 support, {(Kt.k_train == 0).sum():,} unseen)", flush=True)
    scales = {m: "rr" for m in preds}
    # AF 2.0 (+ Paule-Mandel heterogeneity of the cell); its selection correction is defined for non-serious events only
    if KIND == "other":
        a2 = af2_fit(train).set_index(["drug", "t"]).reindex(pd.MultiIndex.from_tuples(keys, names=["drug", "t"]))
        ok2 = a2.theta2.notna().values & ci.k.notna().values
        q = np.full((len(Kt), len(af3.GRID)), np.nan)
        q[ok2] = bench._grid_normal(a2.theta2.values[ok2], a2.se2.values[ok2] ** 2 + np.nan_to_num(ci.pm_t2.values[ok2]))
        preds["B4 AF 2.0"] = q; scales["B4 AF 2.0"] = "rr"
    # AF 3 variants
    at["off"] = 0.0
    offs = {}
    for v, spec in variants.items():
        with pv.connect(read_only=True) as con:
            P = (prep_fn or af3.prepare)(con, arms, K, trials=train, cf_all=cf_all, **spec.get("prep", {}))
        fit, S, _, _ = af3.run_fit(P, fit_fn=af3.em_fit_mix if spec.get("mix") else None, outer=2, inner=6, verbose=False,
                                   **spec.get("fit", {}))
        key = pd.MultiIndex.from_frame(P["cf"][["drug", "term"]])
        cmap = pd.Series(np.arange(len(key)), index=key)
        cell = cmap.reindex(pd.MultiIndex.from_tuples(keys)).values
        new = np.isnan(cell)
        socmap = dict(zip(P["socs"], range(len(P["socs"]))))
        post = np.zeros((len(Kt), len(af3.GRID)))
        post[~new] = fit["post"][cell[~new].astype(int)]
        soc = np.zeros(len(Kt), int)
        soc[~new] = P["cf"].s.values[cell[~new].astype(int)]
        if new.any():
            nc = pd.DataFrame(dict(drug=Kt.drug.values[new], term=Kt.t.values[new])).drop_duplicates()
            nc = nc.merge(cf_all, on=["drug", "term"], how="left")
            mnew = af3.prior_for_new(P["cf"], fit, nc, feats=af3.FEATS)
            mm = dict(zip(zip(nc.drug, nc.term), mnew)); sm = dict(zip(zip(nc.drug, nc.term), nc.soc))
            G = len(af3.GRID); z0 = np.argmin(np.abs(af3.GRID))
            for i in np.where(new)[0]:
                kk_ = (Kt.drug.values[i], Kt.t.values[i])
                sh = int(np.rint(mm[kk_] / af3.DZ))
                idx = np.clip(np.arange(G) - sh, 0, G - 1)
                post[i] = fit["gprior"][idx]; post[i] /= post[i].sum()
                soc[i] = socmap.get(sm[kk_], 0)
                if fit.get("pi0") is not None and np.any(fit["pi0"] > 0):
                    p0_ = fit["pi0"][soc[i]]
                    post[i] = (1 - p0_) * post[i]; post[i, z0] += p0_
        Xt = af3.design_covariates(Kt).values
        shift = np.einsum("ij,ij->i", Xt, fit["beta"][soc]) if spec.get("fit", {}).get("use_cov", True) else np.zeros(len(Kt))
        if spec.get("mix"):
            tw = np.zeros((len(Kt), len(fit["tau_grid"])))
            tw[~new] = fit["tau_w"][cell[~new].astype(int)]
            tw[new] = fit["tau_pi"][soc[new]]
            preds[v] = af3.predictive_grid_mix(post, shift, tw, fit["tau_grid"])
            if spec.get("kappa"):                            # predictive spread for new trials, tuned in the development zone
                qk = preds[v] @ af3._kernels(spec["kappa"])[0].T.astype(np.float64)
                preds[v] = qk / qk.sum(1, keepdims=True)
        else:
            preds[v] = af3.predictive_grid(post, shift, fit["tau"][soc])
        scales[v] = "hz"
        offs[v] = P["gamma"][soc[at.kk.values]] * at.ldose_c.values
        Kt[f"soc_{v}"] = [P["socs"][s] for s in soc]
    return dict(preds={m: q.astype(np.float32) for m, q in preds.items()}, scales=scales, offs=offs, Kt=Kt, at=at,
                name=name, train=train)


def score_window(W, scheme, soc_of_term, K):
    """Score a cached window under a placebo scheme: 'jeffreys' (each placebo arm alone; BENCH-2) or 'hier' (hierarchical
    background model fitted on the window's training trials; BENCH-2.1). AF 3 variants are also scored without dose offsets."""
    preds, scales, offs, Kt, at, name = W["preds"], W["scales"], W["offs"], W["Kt"], W["at"].copy(), W["name"]
    p0n = None
    if scheme == "hier":
        pp = bench.placebo_prior(K[K.nct.isin(W["train"])], soc_of_term)
        mu, sd = pp(Kt)
        p0n = bench.placebo_nodes(Kt, mu, sd)
    rng = np.random.default_rng(SEED)
    res = {}
    for m, q in preds.items():
        q = q.astype(np.float64)
        at["off"] = offs.get(m, 0.0)
        res.update(bench.score_all({m: q}, {m: scales[m]}, Kt, at, rng, p0n=p0n))
        if m in offs:
            at["off"] = 0.0
            res.update({f"{m} no dose offsets": v for v in bench.score_all({m: q}, {m: scales[m]}, Kt, at, rng, p0n=p0n).values()})
    rec = []
    socs = Kt[[c_ for c_ in Kt.columns if c_.startswith("soc_")][0]] if any(c_.startswith("soc_") for c_ in Kt.columns) else ""
    for m, (sc, pit) in res.items():
        rec.append(pd.DataFrame(dict(window=name, nct=Kt.nct.values, drug=Kt.drug.values, term=Kt.t.values, soc=socs,
                                     k_train=Kt.k_train.values, af1_support=Kt.af1_support.values, method=m, logscore=sc, pit=pit,
                                     scoring=scheme)))
    return pd.concat(rec)


def summarise(R, bench_name, reference):
    out = []
    R = R.reset_index(drop=True).copy()
    R["stratum_ev"] = pd.cut(R.k_train, [-1, 0, 1, 4, 9, 1e9], labels=["unseen", "1 trial", "2-4", "5-9", ">=10"]).astype(str)
    strata = {"AF1-support": R.af1_support.astype(bool).values, "seen": (R.k_train > 0).values, "unseen": (R.k_train == 0).values}
    for ev in ["1 trial", "2-4", "5-9", ">=10"]:
        strata[f"evidence {ev}"] = (R.stratum_ev == ev).values
    rng = np.random.default_rng(SEED)
    for win in list(R.window.unique()) + ["pooled"]:
        wm = np.ones(len(R), bool) if win == "pooled" else (R.window == win).values
        for sname, mask in strata.items():
            Rs = R[wm & mask]
            W = Rs.pivot_table(index=["window", "nct", "drug", "term"], columns="method", values="logscore")
            if reference not in W:
                continue
            for m in W.columns:
                sub = W[[m]].join(W[[reference]].rename(columns={reference: "_ref"})).dropna()
                sub = sub.rename(columns={"_ref": "__ref"})
                if len(sub) < 30:
                    continue
                d = sub["__ref"] - sub[m]
                tr = d.groupby(level=[0, 1]).agg(["sum", "count"])
                bs = [tr.iloc[i].sum() for i in [rng.integers(0, len(tr), len(tr)) for _ in range(1000)]]
                bs = np.array([b["sum"] / b["count"] for b in bs])
                P = Rs[Rs.method == m].pit.dropna()
                out.append(dict(bench=bench_name, window=win, stratum=sname, method=m, reference=reference, n_records=int(len(sub)),
                                n_trials=int(len(tr)), mean_logscore=float(sub[m].mean()), diff_vs_ref=float(d.mean()),
                                diff_ci_lo=float(np.percentile(bs, 2.5)), diff_ci_hi=float(np.percentile(bs, 97.5)),
                                pit_central90=float(((P >= .05) & (P <= .95)).mean()), pit_low10=float((P < .1).mean()),
                                pit_high10=float((P > .9).mean())))
    return pd.DataFrame(out)


def main() -> int:
    mp.set_start_method("fork", force=True)
    bench_name, zone, methods = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    kind = sys.argv[4] if len(sys.argv) > 4 else "other"
    global KIND
    KIND = kind
    variants = {m: VARIANTS[m] for m in methods if m in VARIANTS}
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con, kind=kind)
        cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"})
        cf_all = af3.cell_features_v31(con, cells)[F31 + list(M31)]
    wins = bench.DEV_WINDOWS if zone == "dev" else [bench.CONFIRM]
    parts = []
    SPD = Path(os.environ.get("OAJOINT_SCRATCH", "/tmp"))
    soc_of_term = dict(zip(cf_all.term, cf_all.soc))
    for n, c_, e in wins:
        cache = SPD / f"benchfit_{bench_name}_{n}.pkl"
        if cache.exists():
            W = pd.read_pickle(cache)
        else:
            W = run_window(n, c_, e, arms, K, cf_all, variants)
            pd.to_pickle(W, cache)
        for scheme in ("jeffreys", "hier"):
            parts.append(score_window(W, scheme, soc_of_term, K))
    R = pd.concat(parts, ignore_index=True)
    ref = methods[0]
    S = pd.concat([summarise(R[R.scoring == sc_].drop(columns="scoring"), bench_name, ref).assign(scoring=sc_)
                   for sc_ in R.scoring.unique()], ignore_index=True)
    show = S[(S.window == "pooled") & S.stratum.isin(["AF1-support", "seen", "unseen"])]
    print(show[["scoring", "stratum", "method", "n_records", "mean_logscore", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi", "pit_central90",
                "pit_low10", "pit_high10"]].to_string(), flush=True)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN ('atlas_contrast',
                  'ctgov_event_group','ctgov_trial','ctgov_baseline','ctgov_condition','atlas_contrast_route','ref_chembl_drug',
                  'ref_chembl_mechanism')""").fetchall()]
            con.execute("SELECT 1")
        R["run_id"] = run.run_id; R["bench"] = bench_name; S["run_id"] = run.run_id
        run.archive_table("bench_record", R.rename(columns={"window": "win"}), upstream=up)
        run.archive_table("bench_score", S.rename(columns={"window": "win"}), upstream=up)
    return 0


if __name__ == "__main__":
    sys.exit(main())
