#!/usr/bin/env python3
"""Further analyses requested by the third internal review (clinical and editorial,
2026-10-14), registered BEFORE they are computed (154_bmc_r4_analyses.py). Seen before registration: the two reports
and the manuscript; no new estimate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

EXP = "expected share with AF < 0.5 (mean of P(AF < 0.5))"
R = {
    "bmc_r4_ctt_analog": ("descriptive", "share_precise_negligible", """Analogue of the CTT statin question (no
meaningful excess): among listed pairs with >= 10 trials, the share whose upper 95% credible limit of the excess per
100 patients (trials that reported the reaction) is below 1, and the share whose posterior probability of no excess is
above 0.5; with n."""),
    "bmc_r4_nnh_bands": ("descriptive", "share_added_ge5", """Absolute excess in number-needed-to-harm bands: the
distribution of the posterior-mean excess per 100 patients (trials that reported the reaction) among listed pairs:
< 1 (NNH > 100), 1 to < 5, >= 5 (NNH <= 20); and, among drugs with >= 5 listed pairs, the median number of listed
pairs per drug with an excess >= 5 per 100 and the median share of each drug's listed pairs with an excess < 1."""),
    "bmc_r4_common_terms": ("descriptive", "n_terms", f"""The 20 terms listed for the most drugs (among estimated
listed pairs): number of drugs, median placebo frequency per 100, median excess per 100 and the {EXP}."""),
    "bmc_r4_unlisted_benchmark": ("descriptive", "exp_unlisted", f"""Benchmarks for the primary metric: the {EXP}
among pairs not listed on the current label of drugs that have a label (same drugs as listed pairs), beside the listed
value and the positive controls (bmc_r4_posctrl)."""),
    "bmc_r4_coverage": ("descriptive", "coverage_sider", """Coverage of the estimable set: among the SIDER 4.1
(2016 label) pairs of drugs with at least 1 estimated listed pair, the share estimated by the model at all (any trial
contributed counts), overall and for single-word vs multiword terms."""),
    "bmc_r4_disclaimer": ("descriptive", "expected_share_lt_half", f"""Listings in disclaimer lists: in the label text
used for linkage, a disclaimer region starts at a statement that the events listed were not necessarily caused by the
drug (regex 'not necessarily (been )?(caused|related)') and ends at the next postmarketing heading, the next numbered
subsection heading or the end of the text. A listed pair is disclaimer-only if every match of its term lies in a
disclaimer region. Report the number of such pairs and the {EXP} after excluding them, with a drug-cluster bootstrap
95% CI."""),
}


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, rule) in R.items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=None,
                              comparator=None)
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
