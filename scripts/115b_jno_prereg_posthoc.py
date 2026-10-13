#!/usr/bin/env python3
"""Post hoc rule, registered after jno_ext_ctt was evaluated (2026-10-12) and BEFORE the refit it describes.

jno_ext_ctt found 2 of the 19 CTT statin-versus-placebo trials in our data (AURORA, NCT00240331; JUPITER, NCT00239681),
so the concordance is not fully independent. This rule removes them and repeats the comparison. It is post hoc and is
reported as such; the registered gate jno_ext_ctt stands as evaluated.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = """POST HOC (registered after jno_ext_ctt was evaluated: rho 0.330, one-sided P 0.013, agreement 1.00 over 46
evaluable outcomes; 2 CTT trials, AURORA NCT00240331 and JUPITER NCT00239681, contribute to our statin estimates).
Refit the adopted estimator (AF 3.3, settings of 102_af32_fit.py) on all trials except NCT00240331 and NCT00239681 and
repeat the jno_ext_ctt comparison unchanged (pairs, pooling, metrics, permutation seed). Pass criteria as jno_ext_ctt
(rho > 0 with one-sided P < 0.05 and agreement >= 0.80); reported as a post hoc sensitivity analysis."""


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("jno_ext_ctt_independent", " ".join(RULE.split()), gate_family="validation",
                          metric_name="spearman_rho", threshold=0.0, comparator=">")
        print("  registered jno_ext_ctt_independent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
