#!/usr/bin/env python3
"""Analyses requested by a third internal review (statistical, 2026-10-14), registered
BEFORE any of them is computed. Seen before registration: the reviewer report and the manuscript/Additional file 2
(numbers already reported there); no new estimate. Computed by 153_bmc_r4_refits.py (refits) and
154_bmc_r4_analyses.py (summaries)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

FIT = "the adopted estimator (AF 3.3; em_fit_mix, outer=2, inner=6, spike) "
EXP = "expected share with AF < 0.5 (mean of P(AF < 0.5))"
STRATA = "by number of trials of the pair (1, 2-4, 5-9, >=10)"
R = {
    "bmc_r4_downsample": ("descriptive", "bias_k1", f"""Downsampling. Take the label-listed pairs with >= 10 trials in
the primary fit. For each such pair keep the records of k randomly chosen trials of that pair and drop its other
records (k = 1 and k = 3; 3 independent draws each, seeds 20261014 + 10k + draw), then refit {FIT}to all remaining
records, so that the dropped records also leave hyperparameter estimation. Report, for these pairs, the {EXP} in each
downsampled fit and in the primary fit, the difference (downsampled minus primary; mean and range over draws) for each
k, and the share of posterior-mean AF < 0.5. The difference estimates the bias of sparse-pair estimates relative to
data-rich estimates of the same pairs."""),
    "bmc_r4_prior_only": ("descriptive", "prior_exp_1trial", f"""Prior-only probabilities. Refit the primary
specification (variant bmc_full; must reproduce the primary {EXP} within 0.005) and compute, for each listed pair, the
prior probability that AF < 0.5 integrating the zero-excess component of its organ system, the cross-fitted prior mean
and the estimated slab distribution G. Report its mean {STRATA}, beside the posterior and the likelihood-alone values
of Table S7."""),
    "bmc_r4_pi0_profile": ("descriptive", "exp_pi0_x0", f"""Sensitivity to the zero-excess component. From the bmc_full
fit, recompute each listed pair's posterior with the organ-system zero-excess weights multiplied by c = 0, 0.5 and 1.5
(capped at 0.95), holding the pair likelihoods, prior means and G fixed (pair likelihood = posterior / prior on the
grid); report the {EXP} for each c and the mean posterior zero-excess weight among listed pairs {STRATA}."""),
    "bmc_r4_pivotal": ("descriptive", "expected_share_lt_half", f"""Pivotal-like trials. Refit {FIT}to records of
trials that are phase 3, 2/3 or 4 AND double-blind or more AND in an approved indication of the drug (rule of
jno_r3_indication) AND started no later than the drug's first approval year (rule of jno_r3_approval); report the
{EXP} among listed pairs estimable in the fit, with a 95% CI from 1000 drug-cluster bootstrap resamples, and the
numbers of records, trials and listed pairs."""),
    "bmc_r4_addon": ("descriptive", "expected_share_lt_half", f"""Add-on designs. A trial has an add-on design if any
of its placebo event groups maps to at least 1 ChEMBL ingredient (placebo plus active background therapy). Report the
share of included trials with an add-on design and refit {FIT}without them; report the {EXP} among listed pairs, with a
drug-cluster bootstrap 95% CI."""),
    "bmc_r4_complete_same": ("descriptive", "exp_primary_same_pairs", f"""Like-for-like comparison with the
complete-reporting refit (jno_rev complete-reporting variant): the primary fit's {EXP} for exactly the listed pairs
estimated in that refit, beside the refit's value, with the numbers of pairs and drugs."""),
    "bmc_r4_threshold_curve": ("descriptive", "exp_lt_half", f"""Threshold curve. From the archived posterior grids of
the primary fit, the expected share with AF < t among listed pairs for t = 0, 0.2, 1/3, 0.5, 2/3 and 0.8, overall and
{STRATA}."""),
    "bmc_r4_joint_abs": ("descriptive", "share_lt_half_and_added_ge1", """Relative and absolute excess together: listed
pairs cross-classified by posterior-mean AF (< 0.5 vs >= 0.5) and posterior-mean excess per 100 patients in trials
that reported the reaction (< 1 vs >= 1); report the 4 shares."""),
    "bmc_r4_weighted": ("descriptive", "exp_drug_weighted", f"""Weighting. The {EXP} with each drug weighted equally
(mean over drugs of the within-drug mean) and with each term weighted equally; and the report-weighted share, the sum
over listed pairs of the placebo frequency per 100 divided by the sum of the expected drug frequency per 100 (trials
that reported the reaction), i.e. the share of all listed-reaction reports with the drug that would be expected with
placebo."""),
    "bmc_r4_term_length": ("descriptive", "exp_single_word", f"""Term length (linkage recall differs by it): the {EXP}
for listed pairs whose term is a single word vs more than one word, with drug-cluster bootstrap 95% CIs."""),
    "bmc_r4_posctrl": ("descriptive", "mean_p_lt_half", """Positive controls (pairs of jno_rev_positive_controls, >= 2
trials): mean posterior probability that AF < 0.5 and share with posterior-mean AF < 0.5, overall, by number of trials
(2-4, 5-9, >= 10) and excluding mirabegron and vibegron (beta-3 agonists in the antimuscarinic ATC class)."""),
    "bmc_r4_temporal_strata": ("descriptive", "slope_listed", """Temporal calibration (definitions of
jno_rev_temporal_calibration and jno_rev_calibration_noise) restricted to listed pairs and stratified by the number of
pre-February-2017 trials of the pair (1, 2-4, >= 5): n, mean predicted P(AF < 0.5), mean noise-aware predicted
probability, observed share, and the logistic calibration slopes."""),
    "bmc_r4_recalibrated": ("descriptive", "exp_recalibrated", """POST HOC illustration of recalibrated individual
probabilities: logit p' = a + b logit p with b = the noise-aware calibration slope of jno_rev_calibration_noise and a
chosen so that the mean of p' over listed pairs equals the mean of p (calibration in the large was adequate); report the
shares of listed pairs with p' > 0.975 and p' < 0.025."""),
    "bmc_r4_disc_re": ("descriptive", "rr_random", """Discontinuation because of adverse events, random effects: on the
comparisons of jno_rev_discontinuation_v2 (restricted set) with events in both arms, DerSimonian-Laird pooled risk
ratio with 95% CI and 95% prediction interval, and tau^2, beside the Mantel-Haenszel estimate."""),
    "bmc_r4_missed_ci": ("descriptive", "exp_missed", f"""Recall audit: the {EXP} among the false-negative pairs of the
recall audit (pairs listed under other wording), with n and a 95% bootstrap CI over pairs (2000 resamples)."""),
    "bmc_r4_listed_effect": ("descriptive", "auc_listed", """Effect size for listed vs unlisted: among drugs with both,
the area under the curve of posterior-mean AF for listed vs unlisted pairs and the difference in medians, overall and
for pairs with >= 10 trials."""),
}


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, rule) in R.items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=None,
                              comparator=None)
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
