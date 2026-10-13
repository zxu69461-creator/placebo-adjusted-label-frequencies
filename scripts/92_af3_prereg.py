#!/usr/bin/env python3
"""AF 3.0 exploration branch: register the model and its validation rules BEFORE any fit to real outcomes.
(The estimator was developed on data simulated from real trial structures only.)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

GATES = {
    "af3_design": ("descriptive", None, None, None, """AF 3.0 DESIGN. A generative hierarchical model of adverse-event reporting in placebo-controlled trials,
fitted by empirical Bayes on a log-hazard-ratio grid (src/oajoint/af3.py). Units: k = trial x drug x term (a trial contributes to a
term only if it lists it, as in AF 1.0), j = trial x term. LEVEL 1 (exact trial likelihood): every arm's count is binomial with
p = 1 - exp(-exp(eta)) (complementary log-log, i.e. a proportional-hazards model over the trial's follow-up); placebo arms
eta = alpha_j, drug arm a eta = alpha_j + zeta_k + gamma_s * (log dose_a - trial mean log dose); alpha_j is profiled out;
a placebo arm shared by m drugs is split equally between them. gamma_s (within-trial dose slope per system organ class, SOC) is
estimated beforehand by the exact conditional multinomial likelihood among drug arms (baseline and drug effect cancel), shrunk
to the pooled slope. LEVEL 2 (between-trial heterogeneity with effect modifiers): zeta_k ~ N(psi_c + x_k' beta_s, tau_s^2),
x_k = log(follow-up/12 weeks) [parsed from the AE time frame] and missing indicator, open-label, single-blind, share of the
term's events under non-systematic assessment, non-industry sponsor, phase-1-only, healthy volunteers, (start year - 2010)/10
and missing indicator, mean log dose relative to the drug's median dose and unknown-dose indicator, (mean age - 50)/10 and
missing indicator, share female - 0.5 and missing indicator, log(enrolment/200); all zero at the REFERENCE TRIAL (12 weeks,
double-blind, systematic assessment, industry, phase 2-4 patients, 2010, median dose, age 50, 50% women, 200 participants).
Post-randomisation quantities (dropout) are deliberately not adjusted for. beta_s by Fisher scoring with a ridge towards the
pooled effect (prior sd 0.25); tau_s by maximising the within-cell (flat-prior, REML-like) marginal likelihood, penalised
towards the geometric mean over organ systems (log-normal, sd 0.5; weights = numbers of multi-trial cells). LEVEL 3
(structured nonparametric prior): psi_c = m_c + u_c, m_c = 5-fold cross-fitted weighted ridge on drug, term, ATC3, ATC1, SOC,
ATC1 x SOC, ATC3 x term and primary molecular target x SOC (ChEMBL), u_c ~ G estimated nonparametrically on the grid (NPMLE-EM).
Grid -5..5 step 0.05; 2 outer x 6 inner iterations. OUTPUTS per drug x term cell: posterior of psi_c (log HR in the reference
trial), AF_hz = 1 - exp(-psi), the risk-scale excess fraction over 12 weeks at the term's typical placebo hazard (AF_risk12),
P(AF < 0.5), P(no excess), and a 95% predictive interval for a new reference trial (psi + N(0, tau_s^2))."""),
    "af3_sim": ("validation", "scenarios_passed", 3, ">=", """BENCHMARK 1, SIMULATION WITH KNOWN TRUTH (af3_sim). Real structures of a random 15% of the trials in the AF 3.0
data (seed 20260927); placebo counts regenerated from their observed cloglog risk. Three truths: S1 model-consistent (psi = 0 for 40%
of cells, else 0.4 + term N(0,0.3^2) + drug N(0,0.2^2) + N(0,0.5^2); beta: log follow-up -0.1, open-label +0.3, non-systematic -0.15;
tau 0.2; gamma 0.3); S2 misspecified link (drug-arm risk = min(placebo risk x RR, 0.95), log RR drawn as psi in S1, no covariate
effects, tau 0.1, no dose effect); S3 heavy-tailed heterogeneity (as S1 but trial effects t with 3 df scaled by 0.15 and
covariate effects varying by SOC with sd 0.1). Evaluated on cells estimable under AF 1.0 (>= 2 trials, >= 20 events; Mantel-Haenszel
on the same simulated counts). A scenario passes iff AF 3.0's 95% interval coverage of the true psi is between 0.90 and 0.98 AND
its root-mean-square error is <= that of AF 1.0. PASS iff all three scenarios pass."""),
    "af3_prospective": ("validation", "logscore_diff_ci_low", 0.0, ">", """BENCHMARK 2 (PRIMARY), PROSPECTIVE PREDICTION OF NEW TRIALS (af3_prospective). Trials of the AF 3.0 data with a
start date are ordered by it; TRAINING = start before the 70th-percentile start date (trials without a date also go to training),
TEST = the rest. Every method is fitted on training trials only: AF 1.0 (Mantel-Haenszel, cells with >= 2 trials and >= 20
events), AF 1.0-RE (DerSimonian-Laird random effects on the same cells), AF 2.0 (stage 1 + stage 2 exactly as scripts/83) and
AF 3.0. Test records = trial x drug x term records of test trials whose cell is estimable under AF 1.0 in training. Each method
gives a predictive density for the record's effect: AF 1.0 N(theta, se^2); AF 1.0-RE N(theta_DL, se_DL^2 + tau_DL^2); AF 2.0
N(theta2, se2^2 + tau_DL^2); AF 3.0 its cell posterior shifted by x_k' beta_s and convolved with N(0, tau_s^2), with arm dose
offsets. SCORE = log predictive probability of the drug-arm counts given the observed placebo risk (x0 + 0.5)/(n0 + 1)
(risk-ratio scale p = p0 e^z for AF 1.0/AF 2.0, hazard scale p = 1 - (1 - p0)^exp(z) for AF 3.0). Metric = mean score of AF 3.0
minus mean score of the best competitor, with a 95% bootstrap CI over test trials (2,000 resamples, seed 20260927). PASS iff the
lower limit is > 0."""),
    "af3_calibration": ("validation", "pit_central90", None, "between", """BENCHMARK 3, CALIBRATION (af3_calibration). On the af3_prospective test records: randomised probability integral
transform of the pooled drug-arm count under AF 3.0's predictive distribution (arms pooled at their mean risk); metric = share of
records with PIT in [0.05, 0.95]. PASS iff between 0.87 and 0.93. The same metric is reported for every competitor."""),
    "af3_random_holdout": ("validation", "logscore_diff_ci_low", 0.0, ">", """BENCHMARK 4 (SECONDARY), RANDOM HOLDOUT (af3_random_holdout). As af3_prospective, but trials are split at random
80/20 (seed 20260924). Supportive; reported with its CI."""),
    "af3_validity": ("validation", "n_failed", 0, "==", """BENCHMARK 5, VALIDITY RETAINED (af3_validity), on the full-data fit. (a) Label validity: posterior mean psi of
cells on the current FDA label (af_label_match.on_fda) exceeds that of unlisted cells of labelled drugs, one-sided Mann-Whitney
P < 0.001, on cells estimable under AF 1.0. (b) Dose: the pooled within-trial dose slope gamma is positive with z > 3.29
(P < 0.001). PASS iff both hold. Covariate effects (beta) are reported descriptively."""),
    "af3_headline": ("descriptive", None, None, None, """DESCRIPTIVE (af3_headline). On cells estimable under AF 1.0 and on the current FDA label: share with posterior
P(AF_hz < 0.5) > 0.975 (upper 95% credible limit below 0.5), share with posterior mean AF_risk12 < 0.5, share with P(no excess) > 0.5,
and the same shares for all label-listed cells (AF 3.0 estimates cells AF 1.0 cannot). Ablations reported descriptively on the
prospective split: AF 3.0 without covariates, and with a Gaussian instead of a nonparametric prior."""),
    "af3_adopt": ("decision", "n_not_passed", 0, "==", """ADOPTION (af3_adopt). AF 3.0 is RECOMMENDED as the paper's primary estimator iff af3_sim, af3_prospective,
af3_calibration and af3_validity all pass; otherwise it remains an exploration-branch result. (Recommendation only: changing the
manuscript's primary estimator is the authors' decision.) Metric = number of the four gates not passed."""),
}


AMEND = {"af3_design": "Amendment before any predictive validation: the first full-data fit (run 01M3HWW4RBCT8PRD9JTGQ97XMF; "
                        "af3_validity PASS, label P = 2.4e-111, dose z = 29.2) returned implausible between-trial SDs in organ "
                        "systems with few, sparse multi-trial cells (ear 2.0, immune system 2.2, pregnancy 1.16); tau_s is now "
                        "penalised towards the pooled value. af3_sim was first evaluated with the unpenalised estimator (3/3 "
                        "scenarios passed; coverage 0.952/0.950/0.962) and is re-run with the penalised one."}


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, thr, comp, rule) in GATES.items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=thr, comparator=comp,
                              notes=AMEND.get(name))
            print("registered", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
