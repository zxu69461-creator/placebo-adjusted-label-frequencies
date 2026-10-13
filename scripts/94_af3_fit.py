#!/usr/bin/env python3
"""AF 3.0 on all trials: per-cell posteriors (table af3_cell), parameters (af3_param) and gate af3_validity."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402


def params(fit, P, label):
    rows = []
    for s, soc in enumerate(P["socs"]):
        rows.append(dict(fit=label, kind="tau", soc=soc, name="tau", value=float(fit["tau"][s])))
        rows.append(dict(fit=label, kind="gamma", soc=soc, name="dose_slope", value=float(P["gamma"][s])))
        for j, c in enumerate(af3.COVS):
            rows.append(dict(fit=label, kind="beta", soc=soc, name=c, value=float(fit["beta"][s, j])))
    for j, c in enumerate(af3.COVS):
        rows.append(dict(fit=label, kind="beta_pooled", soc=None, name=c, value=float(fit["beta_g"][j])))
    for i, v in enumerate(fit["gprior"]):
        rows.append(dict(fit=label, kind="prior_shape", soc=None, name=f"{af3.GRID[i]:.2f}", value=float(v)))
    rows.append(dict(fit=label, kind="misc", soc=None, name="ridge_lambda", value=float(fit["lam"])))
    rows.append(dict(fit=label, kind="misc", soc=None, name="marginal_loglik", value=float(fit["marginal_loglik"])))
    rows.append(dict(fit=label, kind="misc", soc=None, name="gamma_global", value=P["gamma_info"]["global_slope"]))
    rows.append(dict(fit=label, kind="misc", soc=None, name="gamma_global_se", value=P["gamma_info"]["global_se"]))
    return pd.DataFrame(rows)


def main() -> int:
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        P = af3.prepare(con, arms, K)
        lab = con.execute("SELECT drug, term, has_fda_label, on_fda FROM af_label_match").df()
    print(f"records {len(P['K']):,}; cells {len(P['cf']):,}; gamma {P['gamma_info']}", flush=True)
    fit, S, base12, binfo = af3.run_fit(P, outer=2, inner=6)
    g = af3.af1_table(P["K"], P["arms"])
    S["af1_estimable"] = S.index.isin(g.index)
    S = S.merge(lab, on=["drug", "term"], how="left")
    e = S[S.af1_estimable & (S.has_fda_label == True)]  # noqa: E712
    mw = stats.mannwhitneyu(e[e.on_fda == True].psi_mean, e[e.on_fda != True].psi_mean, alternative="greater")  # noqa: E712
    z = P["gamma_info"]["global_slope"] / P["gamma_info"]["global_se"]
    fail = []
    if not mw.pvalue < 1e-3:
        fail.append("a_label_validity")
    if not z > 3.29:
        fail.append("b_dose")
    beta = {c: float(fit["beta_g"][j]) for j, c in enumerate(af3.COVS)}
    det = dict(label_validity_p=float(mw.pvalue), n_listed=int((e.on_fda == True).sum()), n_unlisted=int((e.on_fda != True).sum()),  # noqa: E712
               median_psi_listed=float(e[e.on_fda == True].psi_mean.median()), median_psi_unlisted=float(e[e.on_fda != True].psi_mean.median()),  # noqa: E712
               gamma_global=P["gamma_info"]["global_slope"], gamma_z=float(z), beta_pooled=beta,
               tau_by_soc=dict(zip(P["socs"], map(float, fit["tau"]))), baseline=binfo, failed=fail)
    print(json.dumps(det, indent=1)[:3000])
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('atlas_contrast','ctgov_event_group','ctgov_trial','ctgov_baseline','ctgov_condition','atlas_contrast_route',
                   'ref_chembl_drug','ref_chembl_mechanism','af_label_match')""").fetchall()]
        keep = ["drug", "term", "soc", "n_trials", "psi_mean", "psi_sd", "psi_lo", "psi_med", "psi_hi", "af_hz", "af_hz_lo",
                "af_hz_hi", "af_risk12", "p0_12wk", "p_af_lt_half", "p_no_excess", "pred_lo", "pred_hi", "tau_soc",
                "prior_mean", "lik_mean", "lik_sd", "af1_estimable"]
        out = S[keep].copy(); out["run_id"] = run.run_id
        out["n_trials"] = out.n_trials.fillna(0).astype(int)
        run.archive_table("af3_cell", out, upstream=up, mode="replace_all")
        pr = params(fit, P, "full"); pr["run_id"] = run.run_id
        run.archive_table("af3_param", pr, upstream=up, mode="replace_all")
        run.evaluate_gate("af3_validity", float(len(fail)), not fail, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
