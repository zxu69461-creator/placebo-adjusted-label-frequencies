#!/usr/bin/env python3
"""Pre-register the current-label (openFDA), drug-class and probability-table analyses.

Why current labels: SIDER 4.1 was mined from labels around 2015/16. A reviewer's first
objection to 'X% of label-listed side effects are placebo-level' is that the labels are
out of date. openFDA's drug-label export (2026-09-24) is public-domain FDA data needing
no application. Registered BEFORE any openFDA label was downloaded or matched.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

GATES = [
 ("af_fda_labels", "pre_registration",
  "LABEL SOURCE: openFDA drug-label export dated 2026-09-24; HUMAN PRESCRIPTION DRUG labels "
  "with an ADVERSE REACTIONS section; single-ingredient only (one openfda.generic_name "
  "without ' and ' or ','). Generic names are mapped to ChEMBL through the same name index "
  "as the trial corpus. ONE label per ingredient: prefer application numbers starting "
  "NDA or BLA (the innovator label that generics copy), then the latest effective_time. "
  "TERM MATCHING: every MedDRA term in the atlas is searched case-insensitively with word "
  "boundaries in the ADVERSE REACTIONS text, also under its US spelling (ae/oe -> e: "
  "diarrhoea->diarrhea, oedema->edema, haemorrhage->hemorrhage, anaemia->anemia, "
  "oesophag->esophag). AMENDED BEFORE ANY MATCHING: the final word of a term may also appear "
  "in plural form ('s' or 'es'), because labels write 'headaches', 'infections'; strict "
  "word boundaries would otherwise under-count label-listed effects and bias H1. The full "
  "British->US rule set is: aemia/aem->em, oea->ea, oedema->edema, oesoph->esoph, "
  "paed->ped, foet->fet, faec->fec, oestr->estr, our->or at word end, sulph->sulf. "
  "Known limitation stated in advance: dictionary matching also counts "
  "terms mentioned in negation or in cross-references; no attempt is made to remove them.",
  None, None, None),
 ("af_fda_validity", "pre_registration",
  "Same rule as af_label_validity, with openFDA labels: among estimable cells of drugs with "
  "a current label, label-listed log-RR > non-listed log-RR, one-sided Mann-Whitney "
  "P < 0.001 with positive median difference. If it fails, af_fda_h1 is not a finding.",
  "mw_p_one_sided", "<", 0.001),
 ("af_fda_h1", "pre_registration",
  "PRIMARY H1 for the paper (current labels): the four fractions of af_h1_label computed on "
  "openFDA-label-listed estimable cells; reported next to the SIDER-based numbers, with "
  "the overlap of the two label sets (Jaccard) and the H1 fractions on pairs listed in BOTH.",
  None, None, None),
 ("af_class", "pre_registration",
  "DRUG CLASSES: WHO-ATC level 3 (first 4 characters) of each ingredient. For every class "
  "with >=20 label-listed (openFDA) estimable cells: fraction with AF upper bound < 0.5, "
  "fraction with AF <= 0, median AF. Heterogeneity of the AF_hi<0.5 fraction across these "
  "classes is tested by chi-square; descriptive otherwise.", "chi2_p", None, None),
 ("af_probability_table", "pre_registration",
  "CLINICIAN / PATIENT TABLE, selection fixed in advance to prevent cherry-picking: the 20 "
  "ingredients with the most contributing trials in the atlas; for each, every label-"
  "listed (openFDA) estimable side effect, reported as 'probability the drug caused it' = "
  "AF with 95% CI (truncated to [0,1] for display, untruncated in the archive).",
  None, None, None),
]


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, fam, rule, metric, cmp_, thr in GATES:
            run.register_gate(name, rule, gate_family=fam, metric_name=metric,
                              comparator=cmp_, threshold=thr,
                              notes="registered before any openFDA label was downloaded")
            print(f"  registered {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
