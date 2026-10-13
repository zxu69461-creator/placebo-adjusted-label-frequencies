#!/usr/bin/env python3
"""Evaluate the refit-based review analyses (rules in 126_jno_review_prereg.py): jno_rev_label_aware,
jno_rev_refit_phase, jno_rev_refit_shared, jno_rev_bootstrap and jno_rev_temporal_calibration, from jno_cell_variant
(127_jno_rev_refits.py). Run after all refits have finished."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("s110", ROOT / "scripts" / "110_afstd_findings.py")
S110 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S110)
SEED = 20260927


def classify(d):
    return dict(n=int(len(d)), drugs=int(d.drug.nunique()), exp_lt_half=float(d.p_af_lt_half.mean()),
                conf_below=float((d.p_af_lt_half > .975).mean()), conf_above=float((d.p_af_lt_half < .025).mean()),
                exp_no_excess=float(d.p_no_excess.mean()), conf_excess=float((d.p_no_excess < .025).mean()),
                exp_no_effect=float(d.p_no_effect.mean()))


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        V = con.execute("SELECT * EXCLUDE (run_id) FROM jno_cell_variant").df()
        L = con.execute("SELECT drug, term, evidence, p_af_lt_half AS p_main FROM afstd_pair WHERE on_fda").df()
        js = con.execute("SELECT * EXCLUDE (run_id) FROM jno_summary").df()
        sm_ = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_summary").df()
        gd = {r[0]: json.loads(r[1]) if r[1] else {} for r in
              con.execute("SELECT gate_name, metric_detail FROM prov_decision").fetchall()}
    need = {"label_aware", "phase", "shared", "pre2017"} | {f"boot:{b}" for b in range(1, 21)}
    have = set(V.variant)
    assert need <= have, f"missing refits: {sorted(need - have)}"
    rows, det = [], {}

    def add(analysis, set_name, d, res, lo=np.nan, hi=np.nan):
        for k, v in res.items():
            if k in ("n", "drugs"):
                continue
            rows.append(dict(analysis=analysis, set_name=set_name, grp="all", n_pairs=res["n"], n_drugs=res["drugs"],
                             metric=k, value=v, lo=lo if k == "exp_lt_half" else np.nan,
                             hi=hi if k == "exp_lt_half" else np.nan))
    for var, gate in (("label_aware", "jno_rev_label_aware"), ("phase", "jno_rev_refit_phase"),
                      ("shared", "jno_rev_refit_shared")):
        d = V[V.variant == var].merge(L[["drug", "term", "evidence"]], on=["drug", "term"])
        lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
        res = classify(d); res["ci"] = [lo, hi]
        if var == "label_aware":
            res["by_stratum"] = {s: classify(d[d.evidence == s]) for s in ("1", "2-4", "5-9", ">=10")}
        det[gate] = res
        add(gate, var, d, {k: v for k, v in res.items() if k not in ("ci", "by_stratum")}, lo, hi)
    # bootstrap
    B = V[V.variant.str.startswith("boot:")].groupby("variant").p_af_lt_half.mean()
    point = float(L.p_main.mean())
    se = float(B.std(ddof=1))
    spec_vals = {"primary": point, "label-aware prior": det["jno_rev_label_aware"]["exp_lt_half"],
                 "without zero-excess component": float(sm_[(sm_.analysis == "no_effect_sensitivity") & (sm_.set_name == "without no-effect component") & (sm_.metric == "exp_lt_half")].value.iloc[0]),
                 "complete-reporting trials": float(js[(js.analysis == "jno_complete_reporting") & (js.metric == "exp_lt_half")].value.iloc[0]),
                 "excluding phase 1 and healthy volunteers": det["jno_rev_refit_phase"]["exp_lt_half"],
                 "excluding shared placebo arms": det["jno_rev_refit_shared"]["exp_lt_half"]}
    det["jno_rev_bootstrap"] = dict(B=int(len(B)), replicates=B.round(4).to_dict(), se=se,
                                    pct=[float(B.quantile(.025)), float(B.quantile(.975))],
                                    normal_ci=[point - 1.96 * se, point + 1.96 * se], point=point,
                                    spec_range=[min(spec_vals.values()), max(spec_vals.values())], specs=spec_vals)
    rows.append(dict(analysis="jno_rev_bootstrap", set_name="trial_cluster", grp="all", n_pairs=int(len(L)), n_drugs=0,
                     metric="exp_lt_half", value=point, lo=point - 1.96 * se, hi=point + 1.96 * se))
    rows.append(dict(analysis="jno_rev_bootstrap", set_name="spec_range", grp="all", n_pairs=int(len(L)), n_drugs=0,
                     metric="exp_lt_half", value=point, lo=min(spec_vals.values()), hi=max(spec_vals.values())))
    # temporal calibration: pre-2017 posterior vs later conventional estimate
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        post = set(K[K.start >= pd.Timestamp("2017-02-01")].nct)
        Pp = af3.prepare(con, arms, K, trials=post)
    g = af3.af1_table(Pp["K"], Pp["arms"])
    cf = Pp["cf"]
    g = g.join(cf[["drug", "term"]], how="left")
    g["obs_lt_half"] = (1 - np.exp(-g.theta) < .5).astype(int)
    pre = V[V.variant == "pre2017"][["drug", "term", "p_af_lt_half"]]
    c = g.merge(pre, on=["drug", "term"])
    c["dec"] = pd.qcut(c.p_af_lt_half.rank(method="first"), 10, labels=False)
    cal = c.groupby("dec").agg(pred=("p_af_lt_half", "mean"), obs=("obs_lt_half", "mean"), n=("obs_lt_half", "size"))
    x = np.log(np.clip(c.p_af_lt_half, 1e-4, 1 - 1e-4) / (1 - np.clip(c.p_af_lt_half, 1e-4, 1 - 1e-4)))
    lg = sm.Logit(c.obs_lt_half, sm.add_constant(x)).fit(disp=0)
    det["jno_rev_temporal_calibration"] = dict(n_pairs=int(len(c)), mean_pred=float(c.p_af_lt_half.mean()),
                                               mean_obs=float(c.obs_lt_half.mean()), slope=float(lg.params.iloc[1]),
                                               intercept=float(lg.params.iloc[0]),
                                               deciles=cal.reset_index().round(3).to_dict("records"))
    rows.append(dict(analysis="jno_rev_temporal_calibration", set_name="pre2017_vs_later", grp="all",
                     n_pairs=int(len(c)), n_drugs=int(c.drug.nunique()), metric="mean_pred_minus_obs",
                     value=float(c.p_af_lt_half.mean() - c.obs_lt_half.mean()), lo=np.nan, hi=np.nan))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("replicates", "deciles", "by_stratum")}
                      for k, v in det.items()}, indent=1, default=float))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x_[0] for x_ in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                              "('jno_cell_variant','afstd_pair','afstd_summary','jno_summary')").fetchall()]
        R = pd.DataFrame(rows); R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis IN ('jno_rev_label_aware','jno_rev_refit_phase',"
                        "'jno_rev_refit_shared','jno_rev_bootstrap','jno_rev_temporal_calibration')")
        run.archive_table("jno_summary", R, upstream=up)
        la = det["jno_rev_label_aware"]["exp_lt_half"]
        run.evaluate_gate("jno_rev_label_aware", la, bool(la > .5), detail=det["jno_rev_label_aware"])
        run.evaluate_gate("jno_rev_refit_phase", det["jno_rev_refit_phase"]["exp_lt_half"], None,
                          detail=det["jno_rev_refit_phase"])
        run.evaluate_gate("jno_rev_refit_shared", det["jno_rev_refit_shared"]["exp_lt_half"], None,
                          detail=det["jno_rev_refit_shared"])
        run.evaluate_gate("jno_rev_bootstrap", se, None, detail=det["jno_rev_bootstrap"])
        run.evaluate_gate("jno_rev_temporal_calibration", det["jno_rev_temporal_calibration"]["slope"], None,
                          detail=det["jno_rev_temporal_calibration"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
