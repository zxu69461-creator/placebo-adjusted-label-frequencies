#!/usr/bin/env python3
"""Archive the full posterior of every pair under the adopted hierarchical estimator (AF 3.3).

afstd_cell stores posterior summaries only. Because the prior has a no-effect component, the posterior of a pair is a
point mass at psi = 0 plus a continuous part, and posterior medians are exactly 0 for most pairs; expected values of
non-linear quantities (reports per 100 added by the drug) need the whole posterior. This script re-runs the full-data
fit exactly as 102_af32_fit.py 'AF 3.3' does, checks that it reproduces afstd_cell, and writes the posterior mass on the
grid to the Parquet domain afstd_posterior (one row per pair, columns g000..g200 for psi = -5.00..5.00).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

KAPPA = 0.1                     # as 102_af32_fit.KAPPA['AF 3.3']
TOL = 1e-4


def main() -> int:
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        P = af3.prepare(con, arms, K)
        ref = con.execute("SELECT drug, term, psi_mean, psi_med, p_af_lt_half, p_no_effect, pred_lo FROM afstd_cell").df()
    fit, S, base12, binfo = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, spike=True)
    et2 = (fit["tau_w"] * fit["tau_grid"] ** 2).sum(1) + KAPPA ** 2
    S["p_no_effect"] = fit["p_null"]
    S["pred_lo"] = 1 - np.exp(-(S.psi_mean - 1.96 * np.sqrt(S.psi_sd ** 2 + et2)))
    m = S[["drug", "term", "psi_mean", "psi_med", "p_af_lt_half", "p_no_effect", "pred_lo"]].merge(
        ref, on=["drug", "term"], suffixes=("", "_ref"), validate="1:1")
    assert len(m) == len(ref) == len(S), (len(m), len(ref), len(S))
    diff = {c: float(np.nanmax(np.abs(m[c] - m[c + "_ref"]))) for c in ("psi_mean", "psi_med", "p_af_lt_half",
                                                                        "p_no_effect", "pred_lo")}
    print("max |refit - afstd_cell|:", diff)
    if max(diff.values()) > TOL:
        print("refit does not reproduce afstd_cell; not archiving")
        return 1
    post = fit["post"].astype(np.float32)
    assert post.shape[1] == len(af3.GRID) == 201
    w = pd.DataFrame(post, columns=[f"g{i:03d}" for i in range(post.shape[1])])
    w.insert(0, "term", S.term.values); w.insert(0, "drug", S.drug.values)
    w = w.sort_values(["drug", "term"]).reset_index(drop=True)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_cell','atlas_contrast','ctgov_event_group','ctgov_trial',"
                                            "'ctgov_baseline','ctgov_condition','atlas_contrast_route','ref_chembl_drug',"
                                            "'ref_chembl_mechanism')").fetchall()]
        dest = run.staging_dir() / "afstd_posterior"; dest.mkdir(parents=True, exist_ok=True)
        with pv.connect(read_only=True) as con:
            con.register("w", w)
            con.execute(f"COPY w TO '{dest / 'posterior.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        run.ingest_staging("afstd_posterior", upstream=up)
    return 0


if __name__ == "__main__":
    sys.exit(main())
