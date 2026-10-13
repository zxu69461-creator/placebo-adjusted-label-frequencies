#!/usr/bin/env python3
"""Pre-register v2 of the conventional analysis, BEFORE any v2 computation or re-download.

Two defects were found by inspecting Fig C3 AFTER the v1 gates had been evaluated:
  1. SPELLING SPLITS. Sponsors report the same MedDRA term in UK and US spelling; the atlas
     treated 'diarrhoea' and 'diarrhea' as different side effects. 29 variant groups, 97
     (drug, concept) pairs split, 194 cells (1.1%).
  2. LABEL MISMATCH. 'One label per ingredient, innovator first, newest first' picked the
     wrong PRODUCT for several drugs: varenicline -> Tyrvaya nasal spray instead of Chantix;
     acetaminophen -> IV (Ofirmev); lidocaine -> infiltration; naltrexone -> injectable.
     Mismatches push label-listed effects toward looking placebo-like, so they could
     inflate H1 and cannot be left in.
v1 outputs are frozen in db/parquet/af_v1_snapshot and keep their evaluated gates.
v2 changes ONLY the two implementation details below; every PASS criterion is copied
verbatim from v1. v2 is reported whichever way it comes out.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = (
 "DIRECTION C v2. CHANGES vs v1 (only these): (A) every MedDRA term is canonicalised to US "
 "spelling with the rule set of af_fda_labels (aemia/aem->em, oea->ea, oedema->edema, "
 "oesoph->esoph, paed->ped, foet->fet, faec->fec, oestr->estr, our->or at word end, "
 "sulph->sulf) BEFORE grouping; if a trial lists both spellings, the per-arm count is their "
 "maximum. (B) A side effect is on the CURRENT label of an ingredient if it appears in the "
 "ADVERSE REACTIONS section of ANY current human-prescription innovator (NDA/BLA) label of "
 "that ingredient, across all formulations, routes and salt forms; ingredients with no "
 "innovator label use all their labels. Matching rule otherwise as af_fda_labels. "
 "PASS CRITERIA, copied from v1: (1) af_reproducibility and _temporal: >=200 cells in both "
 "halves, Spearman >= 0.50, sign agreement >= 80% among >=30 P<0.01 cells; (2) validity, "
 "SIDER and current labels separately: one-sided Mann-Whitney P < 0.001, positive median "
 "difference; (3) af_robustness R1-R4 each: AF_hi<0.5 fraction >= 0.35, AF<=0 fraction "
 ">= 0.20, validity P < 0.001. v2 PASSES only if (1), (2) and (3) all pass. Descriptive "
 "analyses (H1 fractions, D2, classes with chi-square, probability table) are recomputed "
 "under v2 and reported beside v1.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_v2", RULE, gate_family="pre_registration",
                          metric_name="n_criteria_failed", comparator="==", threshold=0.0,
                          notes="registered after v1 evaluation, before any v2 computation; "
                                "v1 frozen in af_v1_snapshot")
        print("  registered af_v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
