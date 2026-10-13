#!/usr/bin/env python3
"""Pre-register the indication-heterogeneity analysis: is a drug's attributable fraction for a side
effect a property of the drug, or of the population it is given to? Registered before condition MeSH
terms were downloaded."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = (
    "INDICATION HETEROGENEITY (af_indication). Indication of a trial = the first condition MeSH term in the "
    "CT.gov conditionBrowseModule 'meshes' list; trials whose conditions text is only healthy volunteers form "
    "'healthy volunteers'; trials with no condition MeSH are excluded. Cells (ingredient, indication, term) use "
    "the v2 rows and rules (US spelling, >=2 trials, >=20 events, Mantel-Haenszel RR). For each drug-term pair "
    "estimable in >= 2 indications, Cochran Q across indications. Statistic = share of such pairs with Q P < 0.01. "
    "NULL: within each drug, indication labels are permuted across that drug's trials (group sizes preserved), "
    "500 permutations (seed 20260924), statistic recomputed among pairs estimable in each permutation. H_ind "
    "PASSES (indication matters beyond trial-to-trial noise) if the observed share exceeds the 95th percentile of "
    "the permutation distribution. Reported as a finding either way: PASS -> AFs must be read per indication; "
    "FAIL -> AFs are transportable across indications. DESCRIPTIVE: the heterogeneous pairs with largest Q; "
    "healthy-volunteer vs patient AF for the same drug-term where both are estimable.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_indication", RULE, gate_family="pre_registration", metric_name="obs_minus_perm_q95",
                          comparator=">", threshold=0.0,
                          notes="registered before condition MeSH download and before any indication-level estimate")
        print("  registered af_indication")
    return 0


if __name__ == "__main__":
    sys.exit(main())
