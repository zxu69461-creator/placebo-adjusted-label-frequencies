#!/usr/bin/env python3
"""Gate jno_complete_reporting: refit the adopted estimator (AF 3.3) on trials that posted every non-serious adverse
event (frequency threshold 0) and recompute the headline on the label-listed pairs estimable in that refit.

Settings are those of 102_af32_fit.py 'AF 3.3' (em_fit_mix, outer=2, inner=6, spike); only the trial set changes.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("s110", ROOT / "scripts" / "110_afstd_findings.py")
s110 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s110)
SEED = 20260927


def main() -> int:
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        complete = {r[0] for r in con.execute("SELECT nct FROM ctgov_trial WHERE frequency_threshold = 0").fetchall()}
        P = af3.prepare(con, arms, K, trials=complete)
        pairs = con.execute("SELECT drug, term, on_fda, p_af_lt_half AS p_full, af_hz AS af_full FROM afstd_pair").df()
    n_used = int(P["K"].nct.nunique())
    print(f"threshold-0 registry trials {len(complete):,}; contributing {n_used:,}; records {len(P['K']):,}; "
          f"pairs {len(P['cf']):,}")
    fit, S, base12, binfo = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, spike=True)
    S["p_no_effect"] = fit["p_null"]
    S["n_trials"] = S.n_trials.fillna(0).astype(int)
    m = S.merge(pairs, on=["drug", "term"], how="left", validate="1:1")
    lst = m[m.on_fda.fillna(False).astype(bool)].copy()
    rng = np.random.default_rng(SEED)
    lo, hi = s110.boot_mean(lst, "p_af_lt_half", rng)
    rng2 = np.random.default_rng(SEED)
    flo, fhi = s110.boot_mean(lst.assign(p_af_lt_half=lst.p_full), "p_af_lt_half", rng2)
    share, full = float(lst.p_af_lt_half.mean()), float(lst.p_full.mean())
    q = lst.af_hz.quantile([.25, .5, .75]).tolist()
    det = dict(n_trials_registry_threshold0=len(complete), n_trials=n_used, n_records=int(len(P["K"])), n_pairs=int(len(S)), n_listed=int(len(lst)),
               n_listed_drugs=int(lst.drug.nunique()), expected_share=share, ci=[lo, hi], full_fit_same_pairs=full,
               full_fit_ci=[flo, fhi], conf_bg=float((lst.p_af_lt_half > .975).mean()),
               conf_drug=float((lst.p_af_lt_half < .025).mean()), af_median_iqr=q,
               spearman_af_vs_full=float(lst[["af_hz", "af_full"]].corr(method="spearman").iloc[0, 1]))
    print(det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','atlas_contrast','ctgov_event_group','ctgov_trial')").fetchall()]
        out = S[["drug", "term", "n_trials", "psi_mean", "psi_sd", "af_hz", "p_af_lt_half", "p_no_effect"]].copy()
        out["run_id"] = run.run_id
        run.archive_table("jno_cell_complete", out, upstream=up, mode="replace_all")
        rows = [dict(analysis="jno_complete_reporting", set_name="listed_complete", grp="all", n_pairs=len(lst),
                     n_drugs=lst.drug.nunique(), metric=k, value=v, lo=a, hi=b)
                for k, v, a, b in [("exp_lt_half", share, lo, hi), ("exp_lt_half_fullfit", full, flo, fhi),
                                   ("conf_bg", det["conf_bg"], np.nan, np.nan),
                                   ("conf_drug", det["conf_drug"], np.nan, np.nan),
                                   ("median_af", q[1], q[0], q[2])]]
        import pandas as pd
        sm = pd.DataFrame(rows); sm["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_complete_reporting'")
        run.archive_table("jno_summary", sm, upstream=up)
        run.evaluate_gate("jno_complete_reporting", share, bool(share > 0.5), detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
