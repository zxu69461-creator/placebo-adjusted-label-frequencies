#!/usr/bin/env python3
"""Benchmark 1 (gate af2_sim): simulation with known truth on real trial structures -- does the selection-corrected
stage 1 of AF 2.0 remove the bias that listing-by-threshold induces in AF 1.0?"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af2  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

SEED, N_CELLS, N_REP = 20260927, 300, 30
G = {}


def _init(structs):
    G["s"] = structs


def run_cell(args):
    ci, theta, p0med = args
    trials = G["s"][ci]
    rng = np.random.default_rng(SEED + ci)
    out = []
    for rep in range(N_REP):
        recs = []
        for tr in trials:
            p0 = float(np.clip(np.exp(np.log(p0med) + rng.normal(0, .7)), .002, .4))
            p = np.where(tr["is_drug"], min(p0 * np.exp(theta), .95), p0)
            x = rng.binomial(tr["n"].astype(int), p).astype(float)
            m = x.sum()
            listed = (x >= tr["thr"]).any() if tr["enforcing"] else m > 0
            if listed:
                recs.append(dict(nct=tr["nct"], x=x, n=tr["n"], is_drug=tr["is_drug"], thr=tr["thr"],
                                 truncate=tr["enforcing"]))
        a = af2.fit_cell(recs); b = af2.mh_cell(recs)
        if a and b and b["se"] == b["se"]:
            out.append((ci, rep, theta, a["theta"], a["se"], b["theta"], b["se"]))
    return out


def main() -> int:
    with pv.connect(read_only=True) as con:
        ct, ae, enf = af2.load_arms(con)
    recs = af2.build_trials(ct, ae, enf)
    est = [k for k, v in recs.items() if af2.mh_cell(v)]
    rng = np.random.default_rng(SEED)
    pick = [est[i] for i in rng.choice(len(est), N_CELLS, replace=False)]
    # per-trial arm structure of every trial of the drug (listed or not for this term)
    narm = ae.groupby(["nct", "eg_id"]).n.max().to_dict()
    tau = ae.groupby("nct").tau.first().to_dict()
    structs, jobs = [], []
    for ci, (drug, term) in enumerate(pick):
        trials = []
        for nct, g in ct[ct.drug == drug].groupby("nct"):
            D = sorted(set(g.eg_active)); P = sorted({p for lst in g.eg_placebo for p in lst})
            n = np.array([narm.get((nct, e), np.nan) for e in D + P], float)
            if np.isnan(n).any() or (n <= 0).any():
                continue
            t_ = tau.get(nct, 0) or 0
            thr = np.where(t_ > 0, np.ceil(t_ * n / 100 - 1e-9), 0).astype(int)
            trials.append(dict(nct=nct, n=n, is_drug=np.array([True] * len(D) + [False] * len(P)), thr=thr,
                               enforcing=nct in enf))
        structs.append(trials)
        pr = [r["x"][~r["is_drug"]].sum() / r["n"][~r["is_drug"]].sum() for r in recs[(drug, term)]]
        p0med = float(np.clip(np.median(pr) if pr else .05, .005, .3))
        theta = 0.0 if rng.random() < .4 else float(rng.normal(.5, .5))
        jobs.append((ci, theta, p0med))
    with Pool(6, initializer=_init, initargs=(structs,)) as pool:
        res = [r for chunk in pool.map(run_cell, jobs) for r in chunk]
    d = pd.DataFrame(res, columns=["cell", "rep", "truth", "th2", "se2", "th1", "se1"])
    b1, b2 = float((d.th1 - d.truth).mean()), float((d.th2 - d.truth).mean())
    cov1 = float((abs(d.th1 - d.truth) <= 1.96 * d.se1).mean()); cov2 = float((abs(d.th2 - d.truth) <= 1.96 * d.se2).mean())
    rm1, rm2 = float(np.sqrt(((d.th1 - d.truth) ** 2).mean())), float(np.sqrt(((d.th2 - d.truth) ** 2).mean()))
    ratio = abs(b2) / abs(b1) if b1 else np.nan
    passed = bool(ratio <= .5 and .90 <= cov2 <= .98)
    by = d.assign(null=d.truth == 0).groupby("null").apply(lambda g: pd.Series(dict(
        bias1=(g.th1 - g.truth).mean(), bias2=(g.th2 - g.truth).mean()))).to_dict()
    det = dict(n_cells=int(d.cell.nunique()), n_replicates=int(len(d)), bias_af1=b1, bias_af2=b2, bias_ratio=ratio,
               coverage_af1=cov1, coverage_af2=cov2, rmse_af1=rm1, rmse_af2=rm2, by_null_truth=by)
    print(det); print(f"-> af2_sim {'PASS' if passed else 'FAIL'}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.evaluate_gate("af2_sim", ratio, passed, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
