#!/usr/bin/env python3
"""Analyses added in response to an internal review (2026-10-12), registered BEFORE any of them is
computed. Seen before registration (disclosed): the review report, which quoted from the database the share of listed
pairs with P(no excess) > 0.5 (66%), a crude drug/placebo frequency ratio below 2 for about 53% of single-trial listed
pairs and 77% of listed pairs with >= 10 trials, and the AE-discontinuation details it criticized. No new estimate.
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
    "jno_rev_label_aware": ("validation", "expected_share_lt_half", 0.5, ">", f"""Label-aware prior. Refit {FIT}with
label status as information: organ-system groups of the model (zero-excess probability, heterogeneity distribution,
trial-covariate and dose effects) are split by status (listed on a current FDA label; labelled but not listed; no
current label), and the cross-fitted ridge prior mean gains a label indicator (listed in the clinical trials subsection;
listed elsewhere only; labelled but not listed; no label) and its interaction with organ system. Report the {PRIMARY},
the three-way classification and the expected share with no excess under this fit. Wording rule: 'most' stays in the
title-free text only if the share exceeds 0.5; the abstract reports this result beside the primary one."""),
    "jno_rev_refit_phase": ("descriptive", "expected_share_lt_half", None, None, f"""Refit {FIT}excluding phase 1-only
trials and trials in healthy volunteers; report the {PRIMARY}."""),
    "jno_rev_refit_shared": ("descriptive", "expected_share_lt_half", None, None, f"""Refit {FIT}excluding trials in which
a placebo arm is shared by more than 1 single-ingredient drug comparison; report the {PRIMARY}."""),
    "jno_rev_bootstrap": ("descriptive", "bootstrap_se", None, None, f"""Uncertainty from refitting. 20 trial-cluster
bootstrap replicates (trials resampled with replacement, duplicates kept as separate trials, seed 20261013 + b), each
refitting {FIT}; report the {PRIMARY} per replicate, its SD (bootstrap standard error) and 2.5th-97.5th percentiles, and
the range across model specifications (primary, label-aware, without zero-excess component, complete reporting,
phase/healthy-volunteer and shared-placebo refits)."""),
    "jno_rev_sim_nozero": ("validation", "share_error", 0.05, "<=", """Simulation without exact zeros. On the real trial
structures of af33_sim (15% of trials, seed 20260927), simulate drug effects with no exact zeros: psi = 0.15 + term
effect N(0, 0.2) + drug effect N(0, 0.15) + N(0, 0.35); otherwise as scenario S1. Fit the adopted estimator and the
same without the zero-excess component. Metrics: coverage of 95% intervals for psi, RMSE, and the calibration of the
primary summary: estimated share (mean P(AF < 0.5)) minus true share (psi < log 2), overall and among cells with 1
trial. Pass if |estimated - true| <= 0.05 overall for the adopted estimator. Repeat the share calibration on the
original scenario S1 (40% exact zeros) for comparison."""),
    "jno_rev_temporal_calibration": ("descriptive", "calibration_slope", None, None, f"""Temporal calibration of the
posterior probabilities. Fit {FIT}on trials started before February 2017 only. For pairs with >= 2 trials started from
February 2017 and >= 20 events in those trials, compare the pre-2017 P(AF < 0.5) with the later trials' conventional
Mantel-Haenszel AF (< 0.5 or not): calibration by deciles of the predicted probability, calibration intercept and slope
(logistic), and the mean predicted vs observed share."""),
    "jno_rev_classification": ("descriptive", "share_conf_excess", None, None, """Classification of label-listed pairs
from the archived posteriors: share with P(AF < 0.5) > 0.975 (confidently below 0.5), < 0.025 (confidently above), and
uncertain; expected share with no excess (mean of P(no excess)); share with P(no excess) < 0.025 (confident excess);
share with expected excess per 100 >= 1; by evidence stratum. Post hoc descriptive additions requested by review."""),
    "jno_rev_positive_controls": ("descriptive", "detection_share", None, None, """Positive controls for nonserious
reactions, fixed by rule from pharmacology before any estimate is viewed: urinary antimuscarinics (ATC4 G04BD) x dry
mouth; opioids (N02A*) x constipation and nausea; dihydropyridine calcium channel blockers (C08CA) x edema peripheral;
GLP-1 receptor agonists (A10BJ) x nausea and vomiting; SGLT2 inhibitors (A10BK) x genital mycotic terms (any term
containing 'mycotic', 'candid' or 'fungal' with genital, vulvovaginal, balan or penile); ACE inhibitors (C09AA) x
cough; alpha-2 delta ligands (pregabalin, gabapentin) x dizziness and somnolence; triptans (N02CC) x paraesthesia.
For pairs with >= 2 trials: share with P(no excess) < 0.025 (excess detected), share with P(AF < 0.5) > 0.5, median
posterior-mean AF, and the conventional pooled AF where estimable. Descriptive."""),
    "jno_rev_scope": ("descriptive", "expected_share_lt_half", None, None, """Scope sensitivity on label-listed pairs:
exclude vaccines (ATC J07), vitamins and mineral supplements (A11, A12); exclude pairs whose term occurs in the drug's
registered trial conditions (indication or study population matched as a label 'reaction'); both. Report the
expected share with AF < 0.5 and n."""),
    "jno_rev_label_audit": ("descriptive", "precision", None, None, """Label-linkage audit: a random sample of 400
label-listed pairs (seed 20261013, stratified proportionally by label subsection class) with the label text around
each whole-word match (+/- 160 characters). Each match is classified as (a) listed as an adverse reaction (including
tables, lists and cross-references to warnings), (b) indication or study population, (c) negation or statement of no
difference, (d) other mention. Classifications are proposed with language-model assistance and must be verified by an
author before publication; precision = share (a). Report precision by subsection class with a 95% Wilson CI."""),
    "jno_rev_discontinuation_v2": ("descriptive", "rr_mh", None, None, """Supersedes jno_discontinuation for the text.
Contrasts as in jno_discontinuation, restricted to trials in which at least 1 linked flow group (any arm) records an
adverse-event withdrawal, excluding trials with enrollment >= 50 000 and vaccines (ATC J07). Report only the
Mantel-Haenszel risk ratio and risk difference (per 100) with 95% CIs, Cochran Q and I-squared, the excess fraction
1 - 1/RR, by ATC level-1 group (>= 10 trials), and the same with all contrasts for comparison."""),
    "jno_rev_descriptives": ("descriptive", None, None, None, """Further descriptive summaries requested by review,
from archived tables: (1) excess fraction on the risk scale at 12 weeks (afstd sensitivity S4) for the primary set;
(2) complete-reporting refit (jno_cell_complete) by evidence stratum with the share confidently below 0.5; (3) listed
pairs with >= 10 trials: model expected share and the conventional Mantel-Haenszel share with AF < 0.5 where
estimable; (4) label-printed placebo frequencies (SIDER pairs with drug and placebo frequency): label-implied AF vs
model posterior-mean AF (Spearman, median difference); (5) Table 1 denominators for follow-up, age and sex; (6) the
share of listed pairs whose evidence includes only 1 trial with zero placebo events."""),
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
