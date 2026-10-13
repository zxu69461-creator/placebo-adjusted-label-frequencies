#!/usr/bin/env python3
"""Register the BENCH-2 protocol (iteration framework for the estimator and its baselines)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = """BENCH-2 PROTOCOL (iteration framework; src/oajoint/bench.py, scripts/98_bench2.py). ZONES: development = trials started
before 2017-02-01, used for all tuning through three rolling origins (train on trials started before 2011-01-01, 2013-01-01,
2015-01-01; test on trials started in [2011, 2013), [2013, 2015), [2015, 2017-02)); confirmation = trials started from
2017-02-01 (train on everything earlier), touched ONCE per registered estimator version, after that version's gates are
registered (AF 3.0 used it once in af3_prospective). METHODS: B0 null; B1 Mantel-Haenszel (AF 1.0); B2 Paule-Mandel random
effects with t(k-2) predictive; B3 Bayesian normal-normal meta-analysis (flat mean, half-normal(0.5) heterogeneity); B4 AF 2.0
with Paule-Mandel heterogeneity; B5 term-only random-effects mean; AF 3.x versions. Baselines are iterated whenever a stronger
standard exists; they are never weakened. SCORE: log predictive probability of each test record's drug-arm counts given its
placebo arm, placebo risk integrated over Beta(x0 + 0.5, n0 - x0 + 0.5) (20 nodes); calibration = share of randomised PIT in
[0.05, 0.95] (target 0.90) and tails PIT < 0.1 and > 0.9 (target 0.10 each). STRATA: AF 1.0 support (cells estimable by AF 1.0 in
training), seen cells (>= 1 training trial), unseen cells (none), and training evidence 1, 2-4, 5-9, >= 10 trials. Differences
use bootstrap CIs over test trials (1,000 resamples, seed 20260927). A new estimator version is ADOPTED iff, in the confirmation
zone, it beats every baseline and the previous version on AF 1.0 support (CI lower limit > 0) and its calibration is within
0.87-0.93; each version registers its own gates before its confirmation run."""


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("bench2_protocol", " ".join(RULE.split()), gate_family="descriptive")
        print("registered bench2_protocol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
