#!/usr/bin/env python3
"""Fit AF 2.0 on the full data and evaluate benchmark 2 (af2_anchor, complete-reporting anchor), benchmark 3
(af2_holdout, out-of-sample accuracy) and the adoption decision (af2_adopt)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af2  # noqa: E402
from oajoint import af_common as afc  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

G = {}


def _init(recs):
    G["r"] = recs


def _fit(key):
    v = G["r"][key]
    b = af2.mh_cell(v)
    a = af2.fit_cell(v) if b else None
    return key, a, b


def fit_all(recs, pool_n=6):
    with Pool(pool_n, initializer=_init, initargs=(recs,)) as pool:
        res = pool.map(_fit, list(recs), chunksize=200)
    rows = []
    for (d, t), a, b in res:
        if a and b and b["se"] == b["se"]:
            rows.append(dict(drug=d, term=t, k=a["k"], events=a["events"], n_trunc=a["n_trunc"],
                             theta_mh=b["theta"], se_mh=b["se"], theta_s1=a["theta"], se_s1=a["se"]))
    return pd.DataFrame(rows)


def shrink(con, cells):
    c = af2.features(con, cells.rename(columns={"theta_s1": "theta", "se_s1": "se"}))
    out, info = af2.stage2(c)
    return out, info


def main() -> int:
    with pv.connect(read_only=True) as con:
        ct, ae, enf = af2.load_arms(con)
        rows_trials = sorted(afc.load_rows().nct.unique())
    # ---- full fit ---------------------------------------------------------------------------------------------
    full = fit_all(af2.build_trials(ct, ae, enf))
    with pv.connect(read_only=True) as con:
        post, info = shrink(con, full)
    print(f"cells {len(full):,}; stage-2 lambda {info['lambda_']}, tau2 {info['tau2']:.3f}; "
          f"median shrinkage {post.shrink.median():.2f}")
    # ---- benchmark 2: complete-reporting anchor ---------------------------------------------------------------
    tau = ae.groupby("nct").tau.first()
    thr_trials, comp_trials = set(tau[tau > 0].index), set(tau[tau == 0].index)
    A = fit_all(af2.build_trials(ct, ae, enf, keep_ncts=thr_trials))
    B = fit_all(af2.build_trials(ct, ae, enf, keep_ncts=comp_trials))
    m = A.merge(B, on=["drug", "term"], suffixes=("_thr", "_comp"))
    d_mh = (m.theta_mh_thr - m.theta_mh_comp).values; d_s1 = (m.theta_s1_thr - m.theta_s1_comp).values
    rng = np.random.default_rng(20260927)
    bs = np.array([[d_mh[i].mean(), d_s1[i].mean()] for i in (rng.integers(0, len(m), len(m)) for _ in range(1000))])
    anchor = dict(n_cells=int(len(m)), mean_diff_af1=float(d_mh.mean()), mean_diff_af2_s1=float(d_s1.mean()),
                  ci_af1=list(map(float, np.percentile(bs[:, 0], [2.5, 97.5]))),
                  ci_af2_s1=list(map(float, np.percentile(bs[:, 1], [2.5, 97.5]))))
    print("anchor:", anchor)
    # ---- benchmark 3: out-of-sample accuracy ------------------------------------------------------------------
    rr = np.random.default_rng(20260924)
    half = dict(zip(rows_trials, np.where(rr.random(len(rows_trials)) < .5, "A", "B")))
    hA = {n for n, h in half.items() if h == "A"}; hB = {n for n, h in half.items() if h == "B"}
    fa = fit_all(af2.build_trials(ct, ae, enf, keep_ncts=hA))
    fb = fit_all(af2.build_trials(ct, ae, enf, keep_ncts=hB))
    with pv.connect(read_only=True) as con:
        pa, info_a = shrink(con, fa)
    ev = pa.merge(fb[["drug", "term", "theta_s1", "se_s1"]].rename(columns={"theta_s1": "tB", "se_s1": "sB"}),
                  on=["drug", "term"])
    meth = {"AF 1.0": (ev.theta_mh, ev.se_mh), "AF 2.0 stage 1": (ev.theta, ev.se), "AF 2.0 full": (ev.theta2, ev.se2)}
    hold = {}
    for k, (th, se) in meth.items():
        mse = float(((th - ev.tB) ** 2 - ev.sB ** 2).mean())
        cov = float((abs(th - ev.tB) <= 1.96 * np.sqrt(se ** 2 + ev.sB ** 2)).mean())
        hold[k] = dict(mse=mse, coverage=cov)
    ratio = hold["AF 2.0 full"]["mse"] / hold["AF 1.0"]["mse"]
    cov2 = hold["AF 2.0 full"]["coverage"]
    hold_pass = bool(ratio <= .8 and .92 <= cov2 <= .98)
    print("holdout:", hold, f"ratio {ratio:.3f} ->", "PASS" if hold_pass else "FAIL")
    with pv.connect(read_only=True) as con:
        sim = con.execute("SELECT passed FROM prov_decision WHERE gate_name='af2_sim'").fetchone()[0]
    n_not = int(not sim) + int(not hold_pass)
    print(f"adoption: {'AF 2.0 PRIMARY' if n_not == 0 else 'AF 1.0 stays primary; AF 2.0 supplementary'}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('atlas_contrast','ctgov_event_group','ctgov_trial')").fetchall()]
        keep = ["drug", "term", "k", "events", "n_trunc", "theta_mh", "se_mh", "theta", "se", "mu", "tau2", "shrink",
                "theta2", "se2", "af2", "af2_lo", "af2_hi"]
        out = post[keep].rename(columns={"theta": "theta_s1", "se": "se_s1"})
        out["run_id"] = run.run_id
        run.archive_table("af2_cell", out, upstream=up, mode="replace_all")
        run.evaluate_gate("af2_anchor", anchor["mean_diff_af2_s1"], None, detail=anchor)
        run.evaluate_gate("af2_holdout", ratio, hold_pass, detail=dict(n_cells=int(len(ev)), methods=hold,
                                                                       stage2_half_a=info_a, stage2_full=info))
        run.evaluate_gate("af2_adopt", float(n_not), n_not == 0,
                          detail=dict(af2_sim=bool(sim), af2_holdout=hold_pass,
                                      decision="AF 2.0 primary" if n_not == 0 else "AF 1.0 primary; AF 2.0 supplementary"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
