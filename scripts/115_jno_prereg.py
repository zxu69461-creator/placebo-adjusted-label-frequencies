#!/usr/bin/env python3
"""Analysis rules for the JAMA Network Open revision, registered BEFORE any of them is computed (2026-10-12).

Seen before registration, and disclosed in the rules that depend on it:
  - the published results of the CTT individual-participant-data meta-analysis of statin label effects
    (Lancet 2026;407:689-703, appendix webtable 1 and webfigures 1a-1d) and of the ACE-inhibitor cough meta-analysis
    (Clin Pharmacol Ther 2019;105:652-660), which are the external benchmarks;
  - counts only, no estimates: 729 distinct terms among statin pairs; 4,018 trials with a posted non-serious
    adverse-event frequency threshold of 0; 17,906 of 21,972 single-ingredient contrasts whose active event group has a
    flow group of the same title;
  - one exploratory crude pooling of adverse-event discontinuations over flow groups whose titles contain or lack
    'placebo' (9,727 trials; 2.05% vs 1.40%), made to judge feasibility before this rule was written.
The primary outcome of the paper is unchanged (afstd_headline: expected share of label-listed pairs with AF < 0.5).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

CTT = "Cholesterol Treatment Trialists' Collaboration, Lancet 2026;407:689-703 (PMID 41655587), appendix (PMC7619005)"
STATINS = "drugs whose ChEMBL ATC level-4 codes include C10AA (HMG-CoA reductase inhibitors), single-ingredient pairs"
POOL = ("outcome-level estimate = precision-weighted mean of the pairs' posterior-mean log hazard ratios (afstd_cell "
        "psi_mean, weights 1/psi_sd^2), standard error (sum of weights)^-1/2, excess fraction EF = 1 - exp(-estimate) "
        "with 95% limits from estimate +/- 1.96 SE")
TERM = "terms compared after the study's normalization (lower case, whitespace collapsed, UK to US spelling)"

R = {
    "jno_ctt_reference": ("qc", "n_outcomes", 65.0, "==", f"""Parse the CTT appendix ({CTT}) into one row per statin
versus placebo outcome: outcome name, organ system, MedDRA preferred terms (webtable 1), events on statin and placebo,
observed-minus-expected and its variance, rate ratio and 95% CI, and the FDR-significance flag (webfigures 1a-1d).
Pass if 65 outcomes are parsed (the 66th has fewer than 10 events and is not reported), every outcome has at least 1
preferred term, and the FDR-significant outcomes are exactly abnormal liver transaminases, other liver function test
abnormality, urinary composition alteration and oedema."""),
    "jno_ext_ctt": ("validation", "spearman_rho", 0.0, ">", f"""External concordance with individual-participant data.
Benchmark: the 65 CTT statin-versus-placebo outcomes of jno_ctt_reference. Our side: pairs of {STATINS} whose term
equals one of the outcome's preferred terms ({TERM}); an outcome is evaluable if it has at least 1 such pair; {POOL}.
Metrics: (1) Spearman correlation between our outcome-level log hazard ratio and the CTT log rate ratio over evaluable
outcomes, one-sided permutation P (10,000 permutations, seed 20261012); (2) agreement on 'mostly background': share of
evaluable outcomes whose point EF is below 0.5 (every CTT outcome has a rate ratio below 2, ie EF < 0.5); (3) our
estimates and ranks for the 4 CTT FDR-significant outcomes; (4) mean difference of our log hazard ratio minus the CTT
log rate ratio. Pass if (1) rho > 0 with one-sided P < 0.05 AND (2) agreement >= 0.80; UNEVALUABLE if fewer than 10
outcomes are evaluable. Also report the number of statin trials in our data with at least 1,000 participants and the
statin trial records shared with CTT where identifiable by name or NCT number. Seen before registration: the CTT
results and the count of distinct statin terms (729); no estimate of ours for any statin pair."""),
    "jno_ext_acei_cough": ("descriptive", "ef_pooled", None, None, f"""Benchmark: Vukadinovic et al., Clin Pharmacol
Ther 2019;105:652-660 (22 placebo-controlled trials, 65,054 patients): 37% of cough reports on ACE inhibitors exceeded
placebo (placebo-adjusted fraction 0.37). Our side: pairs of drugs with ATC level-4 code C09AA and term 'cough'; {POOL}.
Descriptive; consistent if our 95% interval contains 0.37. Seen before registration: the benchmark only."""),
    "jno_complete_reporting": ("validation", "expected_share_lt_half", 0.5, ">", """Sensitivity to the reporting
threshold for non-serious events. Refit the adopted estimator (AF 3.3 with the settings of 102_af32_fit.py: em_fit_mix,
outer=2, inner=6, spike, kappa 0.1) on trials whose posted non-serious adverse-event frequency threshold is 0
(ctgov_trial.frequency_threshold = 0), so that every event observed in those trials is posted. For label-listed pairs
(afstd_pair.on_fda) estimable in the refit: expected share with AF < 0.5 (mean of P(AF < 0.5)) with a 95% CI from 1,000
bootstrap resamples of drugs (seed 20260927); the same quantity from the full-data fit on the same pairs; the share with
P(AF < 0.5) > 0.975; and the median and IQR of posterior-mean AF. Pass (wording rule): the refit expected share exceeds
0.5, so that 'most' remains supported. Seen before registration: the count of threshold-0 trials (4,018), no estimate."""),
    "jno_label_section": ("descriptive", "n_trial_section", None, None, """Where on the label the listed side effects
come from. Segment each current label text in the pool of afstd_label_match into trial spans (from a heading matching
6.1 Clinical Trial(s)/Studies Experience to the next 6.2 heading or the next 6.1 heading), postmarketing spans (from a
heading matching 6.2 Post(-)marketing Experience to the next 6.1 heading or the end) and other text (everything else);
labels without a 6.1 heading are unsegmented. Re-apply the afstd_label_match term rule within each kind of span. Classes
of listed pairs: 'trial' (matched in a trial span), 'postmarketing_only', 'other_only' (eg, cross-references to warnings
only), 'unsegmented'. QC: for segmented ingredients, matches in any span must equal on_fda. Report n per class and the
expected share with AF < 0.5 (bootstrap CI as in afstd_headline) per class. Secondary; the primary outcome stays the
registered set of all listed pairs."""),
    "jno_discontinuation": ("descriptive", "ef_mh", None, None, """Secondary outcome: discontinuation because of
adverse events. Each single-ingredient contrast (atlas_contrast) is linked to flow groups (ctgov_flow_group) whose title,
lower-cased and trimmed, equals the title or arm label of its active event group and of each placebo event group;
evaluable if the active and every placebo event group link to exactly 1 flow group each, with participants started > 0
and adverse-event withdrawals recorded (drop_ae not missing). When m evaluable contrasts of a trial share placebo arms,
the placebo participants and withdrawals are divided by m. Measures: pooled withdrawals per 100 participants on the drug
and on placebo; Mantel-Haenszel risk ratio stratified by contrast; excess fraction 1 - 1/RR with 95% CI; withdrawals per
100 in excess of placebo; the same by ATC level-1 group of the drug and, for drugs with at least 5 evaluable contrasts,
by drug (descriptive). Seen before registration: one exploratory crude pooling by flow-group title (2.05% vs 1.40%)."""),
    "jno_trial_characteristics": ("descriptive", "n_trials", None, None, """Table of the trials contributing to the
214,010 pairs: number of trials and participants (sum of enrolment), median (IQR) enrolment, phase, masking, industry
sponsor, healthy volunteers, start year, follow-up weeks, mean age (median of trial means), percentage of women
(participant-weighted) and ATC level-1 group of the drug; and of the label-listed pairs: number, drugs, terms and
evidence strata."""),
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
