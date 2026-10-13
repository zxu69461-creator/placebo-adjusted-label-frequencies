#!/usr/bin/env python3
"""Superseding rules for the point summaries of the manuscript, registered after afstd_absolute and afstd_listing_rules
were evaluated and BEFORE any expected-value summary is computed.

Defect found at evaluation (2026-09-27): the prior's no-effect component puts a point mass at psi = 0, so the posterior
median of psi is exactly 0 for 60% of label-listed pairs (68%, 60%, 57% and 48% in the 1, 2-4, 5-9 and >=10-trial
strata). Posterior medians of AF and of reports added per 100 are then 0 for most pairs and cannot summarise or rank them,
and rule A as registered (posterior-median AF > 0) is not the current listing rule. The superseding rules keep every
set, stratum and criterion and change only the point summary. Seen before registration: the v1 outputs of
afstd_absolute (median added 0 per 100) and afstd_listing_rules (rule A retained 24%, rule B 5.4%, rule C 5.3%)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

WHY = ("Supersedes the point summaries of afstd_definitions: posterior medians collapse onto the no-effect point mass "
       "(posterior-median psi = 0 for 60% of listed pairs). Seen before registration: v1 outputs of afstd_absolute "
       "(median added 0 per 100) and afstd_listing_rules (rule A 24%, B 5.4%, C 5.3% retained).")
R = {
    "afstd_definitions_v2": ("descriptive", None, None, None, f"""Point summaries use posterior expectations, computed from
the full posterior on the grid (Parquet afstd_posterior, verified to reproduce afstd_cell): AF point = posterior mean of
1 - exp(-psi) (= af_hz); expected reports per 100 on the drug = sum_g post_g * 100 * (1 - (1 - p0)^exp(g)) with p0 the
drug's own pooled placebo risk as in afstd_definitions; expected added = that minus 100 * p0; credible and predictive
limits of added unchanged (posterior and predictive quantiles). Rule A (current form) is evaluated as labels are, on the
observed pooled data of the drug's trials listing the event: drug-arm risk >= 2% and greater than the placebo risk.
Rule B unchanged (P(no excess) < 0.025). Rule C = rule B and expected added >= 1 per 100. Distributions and medians of
AF (overall, by stratum, organ system, class) use the posterior mean. All sets, strata, zones and thresholds unchanged.
{WHY}"""),
    "afstd_absolute_v2": ("descriptive", "median_per100_added_exp", None, None, f"""As afstd_absolute with expected values:
median expected frequency per 100 on the drug and on placebo, median expected added, share with expected added < 1 per
100, share whose upper 97.5% limit of added is < 1 per 100, median number needed to harm (100 / expected added among
pairs with expected added > 0); overall and by evidence stratum; also with the reference background p0_12wk. {WHY}"""),
    "afstd_listing_rules_v2": ("descriptive", None, None, None, f"""As afstd_listing_rules with rule A on the observed data
and rule C on expected added (afstd_definitions_v2): share of current entries retained, share of retained entries
confidently mostly background, expected share with true AF < 0.5 among retained, share of confidently drug-caused listed
pairs retained, and unlisted pairs each rule would add. {WHY}"""),
    "afstd_point_v2": ("descriptive", None, None, None, f"""Posterior-mean AF of label-listed pairs: median and
interquartile range overall, by evidence stratum, by organ system (>= 30 listed pairs) and by ATC level-3 class (>= 20);
Spearman correlation with the conventional Mantel-Haenszel AF on the conventionally estimable pairs. {WHY}"""),
}


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, (fam, metric, thr, comp, rule) in R.items():
            run.register_gate(name, " ".join(rule.split()), gate_family=fam, metric_name=metric, threshold=thr,
                              comparator=comp)
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
