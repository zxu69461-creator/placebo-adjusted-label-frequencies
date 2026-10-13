#!/usr/bin/env python3
"""Pre-register the main attributable-fraction atlas and the drug-label analysis.

Registered after the reproducibility premise passed (af_reproducibility: Spearman 0.569,
93.5% sign agreement; temporal: 0.503, 90.9%) and BEFORE any full-corpus estimate or any
label comparison was computed. The split-half estimates already archived are the only
non-serious estimates seen, and no label mapping has been run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

GATES = [
 ("af_main", "pre_registration",
  "FULL-CORPUS ATLAS: every (ingredient, PT) cell estimable under af_design (>=2 trials, "
  ">=20 events, non-serious, listed-term rule). Per cell: MH risk ratio with Greenland-"
  "Robins CI; attributable fraction among the exposed AF = 1 - 1/RR with CI transformed "
  "from the RR CI; pooled crude risks on drug and on placebo. SIGNALS: BH-FDR < 5% on "
  "log-RR across all estimable cells.", "fdr", "<", 0.05),
 ("af_label_validity", "pre_registration",
  "VALIDITY CHECK with a predicted direction. SIDER 4.1 drugs are mapped to ChEMBL through "
  "the same name index as the corpus (mapping rate reported); label pairs = SIDER PT-level "
  "(drug, side effect). Among ESTIMABLE cells, label-listed pairs must have HIGHER log-RR "
  "than non-listed pairs of the same drugs: one-sided Mann-Whitney P < 0.001 and a "
  "positive median difference. If this fails, the label analysis (af_h1_label) is not "
  "reported as a finding.", "mw_p_one_sided", "<", 0.001),
 ("af_h1_label", "pre_registration",
  "H1, descriptive, the clinical headline. Among label-listed pairs that are estimable: "
  "(a) fraction that are harm SIGNALS (RR>1, FDR<5%); (b) PRIMARY: fraction whose AF "
  "point estimate is < 0.5, i.e. under randomization fewer than half of the reported "
  "events are attributable to the drug; (c) fraction whose AF 95% upper bound is < 0.5 "
  "('confidently majority-placebo'); (d) fraction with AF point estimate <= 0. The same "
  "four fractions for non-listed estimable pairs are the reference. SECONDARY (with an "
  "explicit circularity caveat, since labels draw on the same pivotal trials): for SIDER "
  "rows that state a placebo frequency, the label-implied AF = 1 - f_placebo/f_drug is "
  "compared with the atlas AF by Spearman correlation.", None, None, None),
]


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, fam, rule, metric, cmp_, thr in GATES:
            run.register_gate(name, rule, gate_family=fam, metric_name=metric,
                              comparator=cmp_, threshold=thr,
                              notes="registered after the reproducibility premise passed, before "
                                    "any full-corpus estimate or label mapping")
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
