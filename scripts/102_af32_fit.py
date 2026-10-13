#!/usr/bin/env python3
"""Adopted AF-STD estimator on all trials. Usage: 102_af32_fit.py [AF 3.2|AF 3.3]. AF 3.2 writes af32_cell (as before);
any version also writes afstd_cell (the standard's published estimates, with the estimator named)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("s94", ROOT / "scripts" / "94_af3_fit.py")
s94 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s94)


KAPPA = {"AF 3.2": 0.0, "AF 3.3": 0.1}


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else "AF 3.2"
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        P = af3.prepare(con, arms, K)
    fit, S, base12, binfo = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, **(dict(spike=True) if version == "AF 3.3" else {}))
    et2 = (fit["tau_w"] * fit["tau_grid"] ** 2).sum(1) + KAPPA[version] ** 2
    S["p_no_effect"] = fit["p_null"] if version == "AF 3.3" else np.nan
    S["tau_cell"] = np.sqrt(et2)
    S["pred_lo"] = 1 - np.exp(-(S.psi_mean - 1.96 * np.sqrt(S.psi_sd ** 2 + et2)))
    S["pred_hi"] = 1 - np.exp(-(S.psi_mean + 1.96 * np.sqrt(S.psi_sd ** 2 + et2)))
    g = af3.af1_table(P["K"], P["arms"])
    S["af1_estimable"] = S.index.isin(g.index)
    keep = ["drug", "term", "soc", "n_trials", "psi_mean", "psi_sd", "psi_lo", "psi_med", "psi_hi", "af_hz", "af_hz_lo",
            "af_hz_hi", "af_risk12", "p0_12wk", "p_af_lt_half", "p_no_excess", "pred_lo", "pred_hi", "tau_cell",
            "prior_mean", "lik_mean", "lik_sd", "af1_estimable"]
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('atlas_contrast','ctgov_event_group','ctgov_trial','ctgov_baseline','ctgov_condition','atlas_contrast_route',
                   'ref_chembl_drug','ref_chembl_mechanism')""").fetchall()]
        out = S[keep].copy(); out["run_id"] = run.run_id; out["n_trials"] = out.n_trials.fillna(0).astype(int)
        if version == "AF 3.2":
            run.archive_table("af32_cell", out, upstream=up, mode="replace_all")
        std = S[keep + ["p_no_effect"]].copy(); std["run_id"] = run.run_id; std["estimator"] = version
        std["n_trials"] = std.n_trials.fillna(0).astype(int); std = std.rename(columns={"tau_cell": "tau_cell"})
        run.archive_table("afstd_cell", std, upstream=up, mode="replace_all")
        pr = s94.params(fit, P, f"{version} full"); pr["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM af3_param WHERE fit = ?", [f"{version} full"])
        run.archive_table("af3_param", pr, upstream=up)
    print("cells", len(out), "median tau_cell", float(np.median(S.tau_cell)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
