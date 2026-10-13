#!/usr/bin/env python3
"""Pre-register the sampling-error correction of the headline (af_deconvolution).

Why: 'AF upper bound < 0.5' is conservative, but 'AF <= 0' counts point estimates, which noise alone pushes
below zero. A reviewer's first question is how much of the headline is sampling error. Registered before
any deconvolution was run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = (
    "DECONVOLUTION (af_deconvolution). Input: log RR and SE of every cell listed on the current FDA label (v2 "
    "pooled-innovator rule, af_label_match.on_fda). Model: log RR_i ~ N(theta_i, se_i^2), theta ~ G, G estimated by "
    "nonparametric maximum likelihood (Kiefer-Wolfowitz) on a grid from -4 to 4 in steps of 0.01 by EM "
    "(stop when the log-likelihood gain < 1e-8 or 20,000 iterations). Reported: pi0 = G(theta <= 0) (no excess), "
    "pi_half = G(theta < log 2) (true AF < 0.5), with 200 nonparametric bootstrap CIs (resampling cells). "
    "Calibration BEFORE the real fit: simulate 20 datasets with the observed SEs from a known G (40% at 0, 60% from "
    "N(0.6, 0.5^2)); the method passes calibration if mean |estimated pi0 - 0.40| <= 0.05. HEADLINE SURVIVES if "
    "pi_half >= 0.35 (the robustness threshold registered for af_robustness). Posterior-mean (shrunken) AFs are "
    "computed for every label cell and used for the clinical probability table as a sensitivity alongside the raw "
    "estimates. If the headline does not survive, the paper's main claim is downgraded.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_deconvolution", RULE, gate_family="pre_registration", metric_name="pi_half",
                          comparator=">=", threshold=0.35, notes="registered before any deconvolution fit")
        print("  registered af_deconvolution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
