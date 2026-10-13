#!/usr/bin/env python3
"""Register the confirmation and generality rules of the estimator version chosen in the development zone (BENCH-2.1),
BEFORE the confirmation zone is touched. Usage: 99b_af32_prereg.py '<version>' '<predecessor>'"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def rules(v, prev, pre="af32", gpre="gen"):
    R = {
        f"{pre}_design": ("descriptive", None, None, None, f"""{v} = AF 3.0 with drug x event-specific between-trial heterogeneity:
tau of each cell integrated over the grid {{0.03, 0.08, 0.15, 0.25, 0.4, 0.65, 1.0}}, the distribution of tau within each organ
system estimated nonparametrically from multi-trial cells (EM, initialised log-normal around 0.25 with sd 0.7, Dirichlet
pseudo-count 0.5); predictive for a new trial = cell posterior shifted by x'beta and convolved with the cell's tau mixture.
Chosen over AF 3.0 and AF 3.1a/b in the development zone of BENCH-2.1 (rolling origins 2011, 2013, 2015). BENCH-2.1 = BENCH-2
with the placebo risk of each test record integrated over a hierarchical background model fitted on the training trials
(exact binomial EM on a logit grid: event + organ-system effects + trial covariates; spread x 1.2, tuned in the development
zone)."""),
        f"{pre}_sim": ("validation", "scenarios_passed", 3, ">=", f"""Known-truth simulation of {v}, identical to af3_sim (same seeds,
structures, three truths and criteria: coverage 0.90-0.98 and RMSE <= AF 1.0 in every scenario)."""),
        f"{pre}_confirm": ("validation", "n_comparisons_failed", 0, "==", f"""CONFIRMATION ZONE (trials started from 2017-02-01; training =
all earlier trials), BENCH-2.1 scoring. On records in AF 1.0 support, {v}'s mean log score minus that of each comparator
(B0 null, B1 Mantel-Haenszel, B2 Paule-Mandel-t, B3 Bayesian normal-normal, B4 AF 2.0, B5 term-only, {prev}) has a bootstrap
95% CI (test trials, 1,000 resamples) with lower limit > 0. PASS iff every comparison passes."""),
        f"{pre}_seen": ("validation", "n_comparisons_failed", 0, "==", f"""As af32_confirm, on all seen cells (>= 1 training trial), against
the comparators that cover them (B0, B2, B3, B5, {prev}). Supportive."""),
        f"{pre}_calibration": ("validation", "pit_central90", None, "between", f"""Calibration of {v} in the confirmation zone under
BENCH-2.1 scoring, AF 1.0 support: share of randomised PIT in [0.05, 0.95] between 0.87 and 0.93."""),
        f"{pre}_adopt": ("decision", "n_not_passed", 0, "==", f"""{v} becomes the estimator of AF-STD v1.0 iff {pre}_sim, {pre}_confirm and
{pre}_calibration pass (recommendation; the manuscript decision rests with the authors)."""),
        f"{gpre}_newdrug": ("validation", "n_comparisons_failed", 0, "==", f"""NEW-DRUG GENERALITY. All trials of a random 10% of drugs with >= 3
trials (seed 20260927) are held out; {v} is fitted on the remaining trials and predicts the held-out drugs' records from drug
class, molecular target and side-effect features only. PASS iff {v}'s mean log score beats B6 class-mean (random-effects mean
of the side effect in the same ATC3 class), B5 term-only and B0 null (bootstrap CI lower limit > 0, BENCH-2.1 scoring) and its
calibration (share of PIT in [0.05, 0.95]) is between 0.85 and 0.95."""),
        f"{gpre}_serious": ("validation", "n_comparisons_failed", 0, "==", f"""SERIOUS-EVENT GENERALITY. Serious adverse events (reported in full,
no frequency threshold), confirmation split (trials from 2017-02 vs earlier), BENCH-2.1 scoring. PASS iff on AF 1.0 support
{v} beats B0, B1, B2, B3 and B5 (CI lower limit > 0) and its calibration is between 0.85 and 0.95."""),
    }
    if pre != "af32":
        R[f"{pre}_design"] = ("descriptive", None, None, None, f"""{v} = {prev} plus (i) a no-effect spike in the prior of each drug x event effect:
prior = pi0_SOC * point mass at log HR 0 + (1 - pi0_SOC) * structured nonparametric slab, pi0 estimated by EM per organ system;
(ii) predictive spread kappa = 0.1 (log HR) added for new trials. Both chosen in the development zone (non-serious and serious
events; rolling origins 2011, 2013, 2015) before the confirmation zone was touched for {v}. Confirmation zone looks so far:
AF 3.0 (af3_prospective), AF 3.2 (af32_confirm); this is the third, disclosed.""")
        R[f"{pre}_confirm"] = ("validation", "n_comparisons_failed", 0, "==", f"""CONFIRMATION ZONE (trials started from 2017-02-01),
BENCH-2.2 scoring (BENCH-2.1 placebo model; baselines incl. B6 class-mean and B7 class-informed Bayesian meta-analysis). On
AF 1.0 support, {v}'s mean log score minus that of each comparator (B0, B1, B2, B3, B4 AF 2.0, B5, B6, B7, {prev}) has a bootstrap
95% CI lower limit > 0. PASS iff every comparison passes.""")
        R[f"{pre}_seen"] = ("validation", "n_comparisons_failed", 0, "==", f"""As {pre}_confirm on all seen cells, against the
comparators that cover them (B0, B2, B3, B5, B6, B7, {prev}). Supportive.""")
    return R


def main() -> int:
    v, prev = sys.argv[1], sys.argv[2]
    pre = sys.argv[3] if len(sys.argv) > 3 else "af32"; gpre = sys.argv[4] if len(sys.argv) > 4 else "gen"
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, thr, comp, rule) in rules(v, prev, pre, gpre).items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=thr, comparator=comp)
            print("registered", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
