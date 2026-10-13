#!/usr/bin/env python3
"""Larger analyses requested by the second internal review (labeled use and linkage recall), registered BEFORE any of
them is computed (2026-10-13). Seen before registration: nothing new (the round-2 review report and Supplement 1 only).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

FIT = "the adopted estimator (AF 3.3; em_fit_mix, outer=2, inner=6, spike) "
PRIMARY = ("expected share with AF < 0.5 among label-listed pairs (mean of P(AF < 0.5) over afstd_pair.on_fda pairs "
           "estimable in the fit), with a 95% CI from 1000 drug-cluster bootstrap resamples, and n")
R = {
    "jno_r3_phase34": ("descriptive", "expected_share_lt_half", None, None, f"""Trials closest to labeled use. Refit
{FIT}to trials whose registry phase is phase 3, phase 2/3 or phase 4 only; report the {PRIMARY}."""),
    "jno_r3_approval": ("descriptive", "expected_share_lt_half", None, None, f"""Registration-era vs later trials.
Using the drug's first approval year in ChEMBL, refit {FIT}separately to (a) records of trials that started no later
than the drug's first approval year and (b) records of trials that started after it (drugs without an approval year
excluded from both); report the {PRIMARY} for each."""),
    "jno_r3_indication": ("descriptive", "expected_share_lt_half", None, None, f"""Labeled indication. Approved
indications are the ChEMBL drug_indication records with max_phase_for_ind = 4 (MeSH identifiers), retrieved from the
ChEMBL web services. A record is in a labeled indication if any MeSH identifier of the trial's registered conditions,
or any of their MeSH ancestors, equals an approved-indication MeSH identifier of the drug. Refit {FIT}to these records
only; report the {PRIMARY}, the number of records and trials used, and the share of all records that matched."""),
    "jno_r3_highdose": ("descriptive", "expected_share_lt_half", None, None, f"""Dose. Refit {FIT}keeping, in every
trial, only the drug arm with the highest dose among arms with a known dose (arms with unknown dose kept only when no
arm of the drug in that trial has a known dose); report the {PRIMARY}."""),
    "jno_r3_recall": ("descriptive", "recall_sider_proxy", None, None, """Recall of the label linkage. (1) Proxy: among
pairs that SIDER 4.1 lists for the drug (2016 labels), whose drug has a current FDA label and that the model
estimated, the share that the current-label text match flags as listed, overall and for MedDRA terms of 1 word vs
more. (2) False-negative audit: a random sample of 200 pairs with a current FDA label but not flagged as listed (seed
20261013), with the drug's adverse-reactions label text; each is classified as (a) the reaction is listed under
another wording (synonym, plural, broader or narrower term that clearly covers it), (b) not listed. Classifications
are proposed with language-model assistance and must be verified by an author; report the share (a) with a Wilson 95%
CI and the implied recall of the linkage."""),
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
