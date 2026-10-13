#!/usr/bin/env python3
"""POST HOC (registered after jno_rev_temporal_calibration was evaluated and showed a logistic calibration slope of
0.40 with mean predicted 80.9% vs observed 79.6%): how much of the low slope is expected from the imprecision of the
later trials' own Mantel-Haenszel estimates?

For each pair of jno_rev_temporal_calibration, the pre-2017 posterior is approximated by a 2-component mixture with the
fit's zero-excess weight at psi = 0 and a normal slab whose first 2 moments match the posterior mean and SD; the
probability that a later Mantel-Haenszel log HR (standard error se) is below log 2 is then
  p_obs = w0 * Phi(log 2 / se) + (1 - w0) * Phi((log 2 - m1) / sqrt(v1 + se^2)).
Reported: mean p_obs vs observed share, and the logistic calibration slope of the observed indicator on logit(p_obs),
alongside the registered slope on logit(P(EF < 0.5)). Between-trial heterogeneity of the later trials is ignored, which
makes p_obs slightly overconfident. Descriptive; no pass criterion."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

GATE = "jno_rev_temporal_calibration_noise"
RULE = ("POST HOC, registered after jno_rev_temporal_calibration was evaluated (slope 0.40; mean predicted 0.809 vs "
        "observed 0.796). For the same pairs, approximate the pre-2017 posterior by its zero-excess weight at 0 and a "
        "moment-matched normal slab, compute the probability that the later Mantel-Haenszel log HR (with its standard "
        "error) is below log 2, and report its mean and the logistic calibration slope of the observed indicator on its "
        "logit, beside the registered slope. Descriptive.")


def slope(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    x = np.log(p / (1 - p))
    m = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    return float(m.params.iloc[1]), float(m.params.iloc[0])


def main() -> int:
    with pv.connect(read_only=True) as con:
        pre = con.execute("""SELECT drug, term, psi_mean, psi_sd, p_af_lt_half, p_no_effect FROM jno_cell_variant
                             WHERE variant = 'pre2017'""").df()
        arms, K, J = af3.build(con)
        post = set(K[K.start >= pd.Timestamp("2017-02-01")].nct)
        Pp = af3.prepare(con, arms, K, trials=post)
    g = af3.af1_table(Pp["K"], Pp["arms"]).join(Pp["cf"][["drug", "term"]], how="left")
    g["obs"] = (1 - np.exp(-g.theta) < .5).astype(int)
    c = g.merge(pre, on=["drug", "term"])
    w0 = c.p_no_effect.clip(0, .999)
    m1 = c.psi_mean / (1 - w0)
    v1 = np.maximum((c.psi_sd ** 2 + c.psi_mean ** 2) / (1 - w0) - m1 ** 2, 1e-6)
    L2 = np.log(2)
    c["p_obs"] = w0 * stats.norm.cdf(L2 / c.se) + (1 - w0) * stats.norm.cdf((L2 - m1) / np.sqrt(v1 + c.se ** 2))
    s_reg, i_reg = slope(c.obs, c.p_af_lt_half)
    s_noise, i_noise = slope(c.obs, c.p_obs)
    c["dec"] = pd.qcut(c.p_obs.rank(method="first"), 10, labels=False)
    dec = c.groupby("dec").agg(pred=("p_obs", "mean"), obs=("obs", "mean"), n=("obs", "size")).reset_index()
    det = dict(n_pairs=int(len(c)), mean_pred_true=float(c.p_af_lt_half.mean()), mean_pred_noise=float(c.p_obs.mean()),
               mean_obs=float(c.obs.mean()), slope_registered=s_reg, slope_noise_aware=s_noise,
               intercept_noise_aware=i_noise, deciles=dec.round(3).to_dict("records"),
               median_se=float(c.se.median()))
    print(json.dumps({k: v for k, v in det.items() if k != "deciles"}, indent=1))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate(GATE, RULE, gate_family="descriptive", metric_name="slope_noise_aware")
        run.evaluate_gate(GATE, s_noise, None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
