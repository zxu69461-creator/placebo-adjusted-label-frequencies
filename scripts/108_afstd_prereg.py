#!/usr/bin/env python3
"""Register every rule of the manuscript-v2 label analyses (hierarchical estimator, AF 3.3) BEFORE 109/110 compute them.

Disclosure written into afstd_definitions: before registration, AF 3.3 summaries on the label-listed pairs that the
conventional estimator can estimate were seen twice during planning on 2026-09-27 (6,316 pairs: mean P(AF<0.5) 0.783,
share with P(AF<0.5) > 0.975 0.491, share with posterior-mean AF < 0.5 0.836, mean P(no effect) 0.404; 6,284 pairs:
about 49% confidently background, about 46% with P(no effect) > 0.5), as were the AF 3.0 shares in af3_headline and the
per-drug AF 3.3 values of docs/AF-STD_label_examples.md. The headline analyses are therefore descriptive; the one rule that
decides wording (afstd_robustness) says so.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

SEEN = ("Seen before registration (planning, 2026-09-27): AF 3.3 shares on the conventionally estimable label-listed "
        "pairs (6,316 pairs: mean P(AF<0.5) 0.783; P(AF<0.5)>0.975 in 0.491; posterior-mean AF<0.5 in 0.836; mean "
        "P(no effect) 0.404; 6,284 pairs: ~49% confidently background, ~46% P(no effect)>0.5), the AF 3.0 shares of "
        "af3_headline and the per-drug AF 3.3 values of docs/AF-STD_label_examples.md.")

R = {
    "afstd_definitions": ("descriptive", None, None, None, f"""Definitions for every analysis of manuscript v2. Estimator:
the adopted hierarchical estimator (afstd_cell, estimator 'AF 3.3', latest run of 102_af32_fit.py). Pair = drug x MedDRA
preferred term with at least one placebo-controlled trial listing the event among non-serious adverse events (all 214,010
rows of afstd_cell). AF = 1 - 1/HR under reference trial conditions (12 weeks, double-blind, systematic assessment,
industry sponsor, phase 2-4, the drug's median dose); the posterior median of AF is 1 - exp(-psi_med). Zones by
p = P(AF < 0.5) (p_af_lt_half): confidently mostly background p > 0.975; probably mostly background 0.5 < p <= 0.975;
probably mostly drug-caused 0.025 <= p <= 0.5; confidently mostly drug-caused p < 0.025. Expected share with true AF < 0.5
= mean of p_af_lt_half over the set; its 95% interval from a drug-cluster bootstrap (1,000 resamples, seed 20260927).
No effect: expected share = mean p_no_effect; 'more likely than not no effect' = p_no_effect > 0.5; 'no excess' =
p_no_excess > 0.5. Evidence strata: 1, 2-4, 5-9, >=10 trials. Label-listed = on_fda in afstd_label_match (rule B of
52_af_v2.py: the term, UK or US spelling, optional plural, as a whole word in the adverse-reactions section of any current
innovator label of the ingredient, or of any label when no innovator label exists); the primary set is every on_fda pair.
Organ system: the term's most frequent MedDRA SOC, with 'general disorders and administration site conditions' merged into
'general disorders' for display; organ systems reported with >= 30 listed pairs, ATC level-3 classes with >= 20 (a drug
counts in every class it belongs to). Absolute risks per 100 treated patients over the pooled follow-up of the drug's own
trials: placebo risk p0 = (x0 + 0.5)/(n0 + 1) pooled over the drug's placebo arms listing the event; on drug
1 - (1 - p0)^exp(psi) mapped through the posterior quantiles of psi; added = on drug - placebo; predictive limits from
pred_lo/pred_hi; reference-background version uses p0_12wk. Sensitivity sets: S1 conventionally estimable pairs
(af1_estimable); S2 >= 2 trials; S3 2016 SIDER labels instead of current FDA labels; S4 risk scale (af_risk12 < 0.5 vs
af_hz < 0.5); S5 conventional Mantel-Haenszel estimates (upper 95% limit < 0.5) on af_cell x af_label_match; S6 >= 5 trials;
S7 ingredients whose label text comes from innovator labels only; S8 excluding promiscuous terms (whole-word match in > 50%
of current label texts). {SEEN}"""),
    "afstd_label_match": ("qc", "n_disagreements", 0, "==", """Rule B of 52_af_v2.py (current FDA labels) and the SIDER 4.1
mapping of af_common.label_pairs(), applied unchanged to all afstd_cell pairs and written to afstd_label_match. PASS iff
has_fda_label, on_fda and on_sider agree for every pair shared with af_label_match (0 disagreements). On failure no
downstream analysis is run and a superseding rule is registered."""),
    "afstd_headline": ("descriptive", "exp_lt_half", None, None, f"""Primary set (every on_fda pair). Report the number of
pairs and drugs; share confidently mostly background; expected share with true AF < 0.5 (with drug-cluster bootstrap 95%
interval); expected share with no effect; share with p_no_effect > 0.5; share with p_no_excess > 0.5; share confidently
mostly drug-caused; the four zone shares; median posterior-median AF; each overall and by evidence stratum. Metric =
expected share with true AF < 0.5. {SEEN}"""),
    "afstd_sensitivity": ("descriptive", None, None, None, """Expected share with true AF < 0.5 and share confidently mostly
background in sensitivity sets S1-S8 as defined in afstd_definitions (S4 reports the posterior-mean AF shares on the hazard
and 12-week risk scales; S5 reports the conventional estimator's share with upper 95% limit < 0.5)."""),
    "afstd_robustness": ("validation", "n_sets_failed", 0, "==", f"""Wording rule for title and abstract. The text may say that
'most' label-listed side effects are mostly background only if the expected share with true AF < 0.5 exceeds 0.5 in the
primary set and in S1, S2, S3, S7 and S8; otherwise the neutral wording ('a large share') is used. {SEEN}"""),
    "afstd_label_validity": ("validation", "validity_p", 0.001, "<", """Among pairs of drugs with a current FDA label, posterior
mean log hazard ratio (psi_mean) of listed pairs is larger than that of unlisted pairs: one-sided Mann-Whitney P < 0.001
and median difference > 0 (the criteria of af_fda_validity and af3_validity). Descriptive extras: the same by evidence
stratum and the drug-stratified AUC."""),
    "afstd_by_soc": ("descriptive", None, None, None, """By organ system (>= 30 listed pairs): number of listed pairs, expected
share with true AF < 0.5, share confidently mostly background, median posterior-median AF and its interquartile range,
no-effect share pi0 (mean p_no_effect over all pairs of the organ system, equal to the organ-system spike weight at
convergence of the EM), between-trial tau (af3_param, fit 'AF 3.3 full') and dose slope. No P values (pairs share prior
terms)."""),
    "afstd_by_class": ("descriptive", None, None, None, """By ATC level-3 class (>= 20 listed pairs; a drug counts in every
class it belongs to): number of listed pairs and drugs, expected share with true AF < 0.5, share confidently mostly
background, median posterior-median AF; ATC coverage of listed pairs; ATC level-1 x organ-system expected shares for cells
with >= 10 listed pairs."""),
    "afstd_absolute": ("descriptive", None, None, None, """Per 100 treated patients, for listed pairs: median frequency on the
drug, median added by the drug (posterior median), share with added < 1 per 100, share whose upper 97.5% limit of added is
< 1 per 100, median number needed to harm (100 / added among pairs with added > 0); overall and by evidence stratum; with
the drug's own placebo risk and with the reference background p0_12wk."""),
    "afstd_listing_rules": ("descriptive", None, None, None, """Three listing rules applied to all pairs of drugs with a
current label: A (current form) frequency on the drug >= 2 per 100 and posterior-median AF > 0; B (evidence of excess)
p_no_excess < 0.025; C rule B and added >= 1 per 100. For each: share of current entries retained, share of retained
entries confidently mostly background, share of confidently drug-caused listed pairs retained, and the number of unlisted
pairs the rule would add."""),
    "afstd_examples": ("descriptive", None, None, None, """Fixed examples. Fig. 1a: amlodipine - peripheral oedema and
atorvastatin - myalgia. Fig. 5 cards: the 6 drugs with a current label and the most placebo-controlled trials among drugs
with >= 8 listed pairs with >= 2 trials; for each, the 8 such pairs with the highest frequency per 100 on the drug.
Supplementary cards: drugs ranked 7-18 by the same rule."""),
    "afstd_future_prospective": ("validation", "n_criteria_failed", 0, "==", """Fresh prospective test, evaluated later. The
current fit (afstd_cell and the BENCH-2.2 predictive for new trials, estimator frozen as of 2026-09-27) predicts the
drug-arm counts of placebo-controlled trials whose results are first posted on ClinicalTrials.gov after 2026-09-24 (the
data freeze), under BENCH-2.2 scoring. PASS iff its mean log score beats Mantel-Haenszel pooling and class-informed
Bayesian meta-analysis (bootstrap 95% CI lower limit > 0) and the share of randomised PIT in [0.05, 0.95] lies in
0.87-0.93."""),
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
