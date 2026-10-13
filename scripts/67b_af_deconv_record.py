#!/usr/bin/env python3
"""Record the outcome of gate af_deconvolution: the pre-registered calibration FAILED (mean |error| of the
estimated null fraction 0.104 > 0.05), so the gate is UNEVALUABLE. The run of 67_af_deconvolution.py was stopped
by hand after the calibration line and BEFORE the fit to the real data was printed, so that a superseding rule could
be registered without knowledge of the real-data result. Nothing from the real-data fit was archived."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.evaluate_gate("af_deconvolution", None, None, detail={
            "calibration": {"pi0_mean_estimate": 0.358, "truth": 0.40, "mean_abs_error": 0.104, "passed": False},
            "status": "UNEVALUABLE: calibration failed; real-data fit stopped before output and never inspected",
            "diagnosis": "a point mass at exactly theta=0 is weakly identified under the observed standard errors: "
                         "NPMLE spreads it over small positive and negative effects",
            "superseded_by": "af_deconvolution_v2"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
