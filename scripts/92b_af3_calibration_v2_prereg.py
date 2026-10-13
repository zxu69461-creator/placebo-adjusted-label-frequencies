#!/usr/bin/env python3
"""Superseding rules registered AFTER af3_calibration failed (0.779), BEFORE the new test is computed. Defect: the
calibration benchmark treated each test trial's observed placebo risk as known, so every method's predictive interval
omitted placebo sampling error (all methods 0.67-0.78). The superseding test integrates it; the pass criterion is the
original one. af3_calibration and af3_adopt keep their recorded outcomes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

GATES = {
    "af3_calibration_v2": ("validation", "pit_central90", None, "between", """SUPERSEDES af3_calibration (defect: placebo risk
treated as known). Same test records, training fits and methods as af3_prospective (refitted with the same code and seeds). The
predictive distribution of a record's drug-arm counts integrates the placebo risk over its Jeffreys posterior
Beta(x0 + 0.5, n0 - x0 + 0.5) (20 equal-probability nodes) as well as the method's effect density. Metric = share of records
with randomised PIT of the pooled drug-arm count in [0.05, 0.95] under AF 3.0. PASS iff between 0.87 and 0.93 (criterion of
af3_calibration unchanged). Log scores under the integrated predictive are reported for every method (descriptive)."""),
    "af3_adopt_v2": ("decision", "n_not_passed", 0, "==", """SUPERSEDES af3_adopt, with af3_calibration replaced by
af3_calibration_v2: AF 3.0 is RECOMMENDED as the paper's primary estimator iff af3_sim, af3_prospective, af3_calibration_v2 and
af3_validity all pass (recommendation only; the decision rests with the authors). Both adoption outcomes are reported."""),
}


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, thr, comp, rule) in GATES.items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=thr, comparator=comp,
                              notes="registered 2026-09-27 after af3_calibration FAIL (0.779; all methods 0.67-0.78), before computing")
            print("registered", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
