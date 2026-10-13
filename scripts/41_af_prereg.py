#!/usr/bin/env python3
"""Pre-register the attributable-fraction (nocebo) premise test before any estimate.

Question: for a patient on drug X who reports adverse event Y, what fraction of such
events would have occurred on placebo anyway? Estimated per (ingredient, MedDRA preferred
term) from within-trial drug vs placebo comparisons, in the NON-SERIOUS stratum, where
the corpus has already shown it can recover a known effect (aromatase inhibitors ->
arthralgia) and where events are common enough to estimate.

Prior art is per class only (statins, obesity drugs, antidepressants' 'nocebo index',
COVID vaccines); a cross-class, 12,000-trial atlas checked against drug labels does not
exist. The premise that decides whether to continue is REPRODUCIBILITY: estimates from two
independent random halves of the trials must agree.

THE THRESHOLD TRAP, fixed here: non-serious events are listed only above a trial-chosen
frequency threshold, so a term ABSENT from a trial's table means 'below threshold', NOT
zero. For non-serious outcomes only trials that LIST the term contribute to it.

Disclosed prior exposure: one non-serious drug-AE estimate has been seen (aromatase
inhibitors -> arthralgia, OR 1.15, from an earlier exploratory analysis). No other non-serious estimate exists.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

GATES = [
 ("af_design", "pre_registration",
  "UNIT: (ingredient, MedDRA preferred-term string, case/whitespace-normalised), "
  "single-ingredient placebo-controlled contrasts from atlas_contrast, NON-SERIOUS events "
  "('other'), at-risk = event group otherNumAtRisk. A contrast contributes to a term ONLY "
  "if the trial's non-serious table lists that term (absence = below threshold, not zero). "
  "ESTIMATOR: Mantel-Haenszel risk ratio (Greenland-Robins variance); attributable "
  "fraction among the exposed AF = 1 - 1/RR (negative AF = fewer events on drug). A cell "
  "is ESTIMABLE with >=2 contributing trials and >=20 events in total.", None, None, None),
 ("af_reproducibility", "pre_registration",
  "PREMISE GATE. Trials (not contrasts) are split into two halves at random (seed "
  "20260924). log-RR is estimated separately in each half for every cell estimable in "
  "BOTH. PASS requires all of: >=200 cells estimable in both halves; Spearman correlation "
  "of log-RR between halves >= 0.50; and among cells with P<0.01 in half A (>=30 such "
  "cells), the sign of log-RR agrees in half B for >=80%. Fewer cells -> UNEVALUABLE. "
  "If this fails, the attributable-fraction direction is not pursued.",
  "spearman_logrr_halves", ">=", 0.50),
 ("af_reproducibility_temporal", "pre_registration",
  "Supportive, same rule, split by trial start date at the corpus median instead of at "
  "random: a stricter, quasi-prospective test (later trials predicted by earlier ones).",
  "spearman_logrr_halves", ">=", 0.50),
]


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, fam, rule, metric, cmp_, thr in GATES:
            run.register_gate(name, rule, gate_family=fam, metric_name=metric,
                              comparator=cmp_, threshold=thr,
                              notes="registered before any non-serious estimate beyond the "
                                    "disclosed aromatase-inhibitor arthralgia result")
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
