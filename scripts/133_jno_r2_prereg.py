#!/usr/bin/env python3
"""Analyses added in response to a second internal review (2026-10-13), registered BEFORE any of
them is computed. Seen before registration (disclosed): the second review report, which quoted only numbers already in
Supplement 1 and Supplement 3; no new estimate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

FIT = "the adopted estimator (AF 3.3; em_fit_mix, outer=2, inner=6, spike) "
PRIMARY = ("expected share with AF < 0.5 among label-listed pairs (mean of P(AF < 0.5) over afstd_pair.on_fda pairs "
           "estimable in the fit)")
R = {
    "jno_r2_subsample": ("descriptive", "subsample_se", None, None, f"""Uncertainty from refitting without duplicated
trials. 20 half-samples of trials drawn without replacement (seed 20261013 + 100 + b), each refitting {FIT}; report the
{PRIMARY} per replicate, its mean and SD. Under delete-half subsampling the SD of the half-sample estimates
approximates the standard error of the full-data estimate; report the full-data estimate +/- 1.96 SD as an
approximate 95% interval, and the mean shift of the half-sample estimates from the full-data estimate (less data,
more shrinkage)."""),
    "jno_r2_sim_range": ("validation", "max_abs_share_error", 0.08, "<=", """Simulation across true shares. On the real
trial structures of af33_sim (15% of trials, seed 20260927), simulate drug effects without exact zeros as in
jno_rev_sim_nozero but with intercepts 0.45 (true share near 70%) and 0.70 (near 50%), and a heavy-tailed scenario
(intercept 0.15, cell residual t with 3 degrees of freedom scaled to SD 0.35). Fit the adopted estimator; report the
estimated (mean P(AF < 0.5)) and true share overall and among cells with 1 trial, coverage and RMSE. Pass if the
absolute error of the overall share is <= 0.08 in every scenario."""),
    "jno_r2_absolute_complete": ("descriptive", "median_per100_added_complete", None, None, """Absolute frequencies
without the posting threshold. For listed pairs whose drug was tested in at least 1 trial that posted every
nonserious event (registry frequency threshold 0), compute the placebo frequency per 100 from all such trials of the
drug (an event not posted counts as 0) and the drug frequency per 100 implied by the pair's posterior (same formula as
afstd_pair), and report medians (IQR) of placebo, drug and excess per 100 and the share with excess < 1 per 100, beside
the threshold-conditioned values of afstd_pair for the same pairs."""),
    "jno_r2_route_single": ("descriptive", "expected_share_lt_half", None, None, f"""Formulation-matched subset.
Restrict listed pairs to drugs whose coded trial contrasts (atlas_contrast_route, tiers T1-T3; mixed and unresolved
ignored) all have one route family and whose current FDA labels (fda_label_route_pool, for the drug's label generic
name) cover exactly that route family. Report the {PRIMARY} in this subset with a 95% CI from 1000 bootstrap resamples
of drugs, and n."""),
    "jno_r2_prior_vs_data": ("descriptive", None, None, None, """How much the data and the prior contribute, by
evidence stratum of listed pairs: expected share from the posterior; expected share implied by the pair's likelihood
alone (normal approximation with the archived likelihood mean and SD, P = Phi((log 2 - mean) / SD)); and the share of
pairs whose cross-fitted prior mean is below log 2. Descriptive."""),
}


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, thr, comp, rule) in R.items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=thr,
                              comparator=comp)
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
