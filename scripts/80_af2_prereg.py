#!/usr/bin/env python3
"""Prespecify AF 2.0 and the benchmarks that decide whether it replaces AF 1.0. Registered before any AF 2.0
estimate, simulation or benchmark was computed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

DESIGN = (
    "AF 2.0 DESIGN (af2_design). Data: arm-level non-serious adverse-event counts and numbers at risk. For a drug d and "
    "term t (US spelling, maximum over spellings), a trial contributes if it lists t and has >= 1 single-ingredient arm "
    "of d and >= 1 placebo arm; drug arms D (all single-ingredient arms of d, doses pooled under one rate ratio), "
    "placebo arms P, other arms O. STAGE 1 (selection-corrected conditional likelihood): given the total m of events in "
    "D and P, counts are multinomial with p_i proportional to n_i*psi_i (psi = exp(theta) for D, 1 for P). A trial is "
    "THRESHOLD-ENFORCING if its reporting threshold tau > 0 and every non-serious term it lists reaches the threshold in "
    "at least one arm (count >= ceil(tau*n/100); if fewer than 90% of listed terms in tau>0 trials comply under this "
    "rule but more comply under count > tau*n/100, the stricter rule is used instead). For enforcing trials in which no "
    "arm in O reaches the threshold for t, the likelihood is divided by P(some arm in D or P reaches its threshold | m, "
    "theta), computed exactly by truncated-series convolution (equal to 1 when m exceeds the sum of (threshold - 1)). "
    "theta is estimated per cell by maximising the summed log-likelihood with a weak N(0, 4^2) penalty (finite estimates "
    "under separation); SE = sqrt(max(model-based, trial-cluster sandwich x k/(k-1))). Cells need >= 2 trials and >= 20 "
    "events. STAGE 2 (structured empirical Bayes): prior mean mu_c from a weighted ridge regression of theta_hat on "
    "one-hot drug, term, drug ATC level-3 class, ATC level 1, term system organ class and ATC1 x SOC, weights "
    "1/(s^2+tau^2), five-fold cross-fitted so that no cell informs its own prior; ridge penalty chosen by cross-validated "
    "predictive log-likelihood; tau^2 by method of moments. Posterior theta_tilde = mu + B(theta_hat - mu), B = "
    "tau^2/(tau^2+s^2), posterior variance B*s^2. AF2 = 1 - exp(-theta_tilde) with 95% posterior interval.")
SIM = (
    "BENCHMARK 1, SIMULATION (af2_sim). 300 cells sampled at random (seed 20260927) from cells estimable in AF 1.0; for "
    "each, all trials of the drug with their real arm sizes, thresholds and enforcement status; true theta 40% = 0 and "
    "60% ~ N(0.5, 0.5^2); per-trial placebo risk ~ lognormal around the term's median listed placebo risk (sd 0.7 on log "
    "scale, clipped to 0.002-0.4); binomial counts; listing applied as defined in af2_design; 30 replicates per cell. "
    "Compared on replicates where both are estimable: AF 1.0 (Mantel-Haenszel on listed trials) versus AF 2.0 stage 1. "
    "PASS iff |overall mean bias of theta| (over all cell-replicates) for AF 2.0 <= 50% of AF 1.0's AND AF 2.0 95% "
    "interval coverage is between 0.90 and 0.98. RMSE reported. [Amended before evaluation: 30 replicates per cell and "
    "an overall signed-bias criterion, because a per-cell absolute bias over few replicates is dominated by Monte-Carlo "
    "noise and cannot resolve a directional selection bias.]")
ANCHOR = (
    "BENCHMARK 2, COMPLETE-REPORTING ANCHOR (af2_anchor, supportive). Cells estimable both from trials with threshold "
    "> 0 only and from trials with threshold 0 only: mean of (theta from thresholded trials - theta from complete-"
    "reporting trials), for AF 1.0 and AF 2.0 stage 1, with 1,000 cell-bootstrap CIs. Reported; supportive if AF 2.0's "
    "|mean difference| is smaller than AF 1.0's.")
HOLDOUT = (
    "BENCHMARK 3, OUT-OF-SAMPLE ACCURACY (af2_holdout). Trials split into random halves (seed 20260924, as before). "
    "Each method is fitted on half A and evaluated against the AF 2.0 stage-1 estimate in half B (selection-corrected, "
    "independent) on cells estimable under both: excess squared error MSE = mean[(theta_A - theta_B)^2 - s_B^2]; "
    "coverage = share of theta_B within theta_A +/- 1.96*sqrt(v_A + s_B^2). Methods: AF 1.0; AF 2.0 stage 1; AF 2.0 "
    "full. PASS iff AF 2.0 full MSE <= 0.8 x AF 1.0 MSE AND AF 2.0 full coverage between 0.92 and 0.98.")
ADOPT = ("ADOPTION (af2_adopt). AF 2.0 replaces AF 1.0 as the primary estimator of the paper iff af2_sim and af2_holdout "
         "both pass; otherwise AF 1.0 stays primary and AF 2.0 is reported as a supplementary sensitivity analysis. "
         "Metric = number of the two gates not passed.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af2_design", DESIGN, gate_family="pre_registration",
                          notes="registered before any AF 2.0 estimate")
        run.register_gate("af2_sim", SIM, gate_family="pre_registration", metric_name="bias_ratio",
                          comparator="<=", threshold=0.5)
        run.register_gate("af2_anchor", ANCHOR, gate_family="descriptive")
        run.register_gate("af2_holdout", HOLDOUT, gate_family="pre_registration", metric_name="mse_ratio",
                          comparator="<=", threshold=0.8)
        run.register_gate("af2_adopt", ADOPT, gate_family="decision", metric_name="n_not_passed",
                          comparator="==", threshold=0.0)
        print("  registered af2_design, af2_sim, af2_anchor, af2_holdout, af2_adopt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
