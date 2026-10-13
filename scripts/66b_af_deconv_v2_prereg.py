#!/usr/bin/env python3
"""Pre-register af_deconvolution_v2, superseding af_deconvolution after its calibration failed.

Only the estimand of the calibration changes: the point-mass-at-zero fraction pi0 proved weakly identified, so the
calibration now targets pi_half = G(theta < log 2), the quantity on which the headline criterion was already
defined. The headline criterion (pi_half >= 0.35) is copied unchanged. pi0 is reported only descriptively.
Registered before any fit of the real data has been seen."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = (
    "DECONVOLUTION v2 (supersedes af_deconvolution, whose pi0 calibration failed). Same data and model: log RR_i ~ "
    "N(theta_i, se_i^2) for current-FDA-label cells, theta ~ G by NPMLE on a grid -4..4 step 0.01, EM with SQUAREM, "
    "stop at log-likelihood gain < 1e-8 or 5,000 iterations (identical settings for calibration, real fit and "
    "bootstrap). CALIBRATION on pi_half = G(theta < log 2): 10 simulated data sets with the observed SEs under each of "
    "three known G -- (A) 40% at 0 + 60% N(0.6,0.5^2) [true pi_half 0.745]; (B) N(0.4,0.6^2) [0.688]; (C) 20% at 0 + "
    "80% N(0.9,0.6^2) [0.492]; passes if the mean absolute error of pi_half <= 0.05 in every scenario. If calibration "
    "passes: report pi_half with 100 bootstrap 95% CI; HEADLINE SURVIVES if pi_half >= 0.35 (unchanged); pi0 reported "
    "descriptively as weakly identified. If calibration fails: UNEVALUABLE, and the paper reports only the "
    "confidence-bound fractions.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_deconvolution_v2", RULE, gate_family="pre_registration", metric_name="pi_half",
                          comparator=">=", threshold=0.35,
                          notes="registered after af_deconvolution calibration failure; real-data fit never inspected")
        print("  registered af_deconvolution_v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
