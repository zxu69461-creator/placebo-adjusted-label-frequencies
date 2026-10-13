#!/usr/bin/env python3
"""Sampling-error-corrected headline (gate af_deconvolution_v2, superseding af_deconvolution): NPMLE of the
distribution of true log RR among current-label side effects; calibration on pi_half under three known G."""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import npmle  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

GRID = np.round(np.arange(-4, 4.0001, 0.01), 2)
LOG2 = np.log(2)


def summ(w):
    return float(w[GRID <= 0].sum()), float(w[GRID < LOG2].sum())


MAX_IT = 5000
from scipy.stats import norm  # noqa: E402
SCEN = {"A": (.40, .6, .5), "B": (0.0, .4, .6), "C": (.20, .9, .6)}      # (point mass at 0, mean, sd)
TRUTH = {k: p0 + (1 - p0) * norm.cdf((LOG2 - m) / sd) for k, (p0, m, sd) in SCEN.items()}


def _calib(args):
    sc, k, s = args
    p0, m, sd = SCEN[sc]
    rng = np.random.default_rng(1000 + 100 * ord(sc) + k)
    theta = np.where(rng.random(len(s)) < p0, 0.0, rng.normal(m, sd, len(s)))
    x = rng.normal(theta, s)
    w, *_ = npmle.fit(x, s, GRID, max_iter=MAX_IT)
    return sc, summ(w)[1]


def _boot(args):
    k, x, s, w0 = args
    rng = np.random.default_rng(5000 + k)
    ix = rng.integers(0, len(x), len(x))
    w, *_ = npmle.fit(x[ix], s[ix], GRID, w0=w0, max_iter=MAX_IT)
    return summ(w)


def main() -> int:
    with pv.connect(read_only=True) as con:
        d = con.execute("""SELECT c.drug, c.term, c.rr, c.rr_lo, c.rr_hi, c.af FROM af_cell c
                           JOIN af_label_match m USING (drug, term) WHERE m.on_fda""").df()
    x = np.log(d.rr.values); s = ((np.log(d.rr_hi) - np.log(d.rr_lo)) / 3.92).values
    ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
    d, x, s = d[ok].reset_index(drop=True), x[ok], s[ok]
    print(f"label cells {len(d):,}; raw: AF<=0 {(d.af <= 0).mean():.1%}", flush=True)
    with Pool(5) as pool:
        cal = pool.map(_calib, [(sc, k, s) for sc in SCEN for k in range(10)])
    err = {sc: float(np.mean([abs(v - TRUTH[sc]) for c, v in cal if c == sc])) for sc in SCEN}
    est = {sc: float(np.mean([v for c, v in cal if c == sc])) for sc in SCEN}
    calibrated = all(e <= .05 for e in err.values())
    print(f"CALIBRATION (pi_half): truth {TRUTH}; mean estimate {est}; mean |error| {err} -> "
          f"{'PASS' if calibrated else 'FAIL'}", flush=True)
    if not calibrated:
        with pv.run_context(__file__, env_name="oajoint-py311") as run:
            run.evaluate_gate("af_deconvolution_v2", None, None, detail={
                "calibration": {"truth": TRUTH, "mean_estimate": est, "mean_abs_error": err, "passed": False},
                "status": "UNEVALUABLE: calibration failed; real data not fitted"})
        return 0
    w, ll, it, L = npmle.fit(x, s, GRID, max_iter=MAX_IT)
    pi0, pi_half = summ(w)
    print(f"FIT: iterations {it}, loglik {ll:.2f}; pi0 = {pi0:.3f}; pi_half = {pi_half:.3f}", flush=True)
    with Pool(5) as pool:
        bs = np.array(pool.map(_boot, [(k, x, s, w) for k in range(100)]))
    ci0, cih = np.quantile(bs[:, 0], [.025, .975]), np.quantile(bs[:, 1], [.025, .975])
    print(f"bootstrap 95% CI: pi0 [{ci0[0]:.3f}, {ci0[1]:.3f}]; pi_half [{cih[0]:.3f}, {cih[1]:.3f}]", flush=True)
    P = npmle.posterior(L, w, GRID)
    d["af_post_mean"] = P @ (1 - np.exp(-GRID))
    d["p_no_excess"] = P[:, GRID <= 0].sum(1)
    d["p_af_lt_half"] = P[:, GRID < LOG2].sum(1)
    survives = pi_half >= .35
    passed = bool(survives)
    print(f"posterior: median shrunken AF {d.af_post_mean.median():.3f}; cells with P(AF<0.5) > 0.9: "
          f"{(d.p_af_lt_half > .9).mean():.1%}; P(no excess) > 0.5: {(d.p_no_excess > .5).mean():.1%}")
    print(f"-> af_deconvolution: {'UNEVALUABLE (calibration failed)' if passed is None else ('PASS' if passed else 'FAIL: headline downgraded')}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('af_cell','af_label_match')").fetchall()]
        e = d[["drug", "term", "af", "af_post_mean", "p_no_excess", "p_af_lt_half"]].rename(columns={"af": "af_raw"})
        e["run_id"] = run.run_id
        pr = pd.DataFrame({"theta": GRID, "weight": w, "run_id": run.run_id})
        run.archive_table("af_eb_cell", e, upstream=up, mode="replace_all")
        run.archive_table("af_eb_prior", pr, upstream=up, mode="replace_all")
        run.evaluate_gate("af_deconvolution_v2", pi_half, passed, detail={
            "n_cells": int(len(d)), "pi0": pi0, "pi_half": pi_half, "pi0_ci": list(map(float, ci0)),
            "pi_half_ci": list(map(float, cih)), "iterations": int(it), "loglik": ll,
            "calibration": {"truth": TRUTH, "mean_estimate": est, "mean_abs_error": err, "passed": calibrated},
            "pi0_note": "descriptive only; weakly identified (af_deconvolution calibration failure)",
            "raw_af_le_zero": float((d.af <= 0).mean()),
            "posterior_share_p_af_lt_half_gt_09": float((d.p_af_lt_half > .9).mean()),
            "posterior_share_p_no_excess_gt_05": float((d.p_no_excess > .5).mean())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
