#!/usr/bin/env python3
"""Pre-register the route/formulation-stratified re-analysis of the conventional estimates.

Why: v2 pools every formulation of an ingredient on BOTH sides (trials and labels). That is
internally consistent but clinically wrong for drugs sold in several forms -- 'varenicline'
mixes oral smoking-cessation trials with nasal-spray dry-eye trials, so the probability table
shows 'varenicline -> sneezing AF 0.74'. A patient asks about ONE product.
Registered before any route is coded for any contrast and before route-level label pools exist.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

FAMILIES = ("oral, parenteral, inhaled, nasal, ophthalmic, topical, transdermal, vaginal, rectal, "
            "local_injection")

CODING = (
    "ROUTE CODING (QC). Each single-ingredient contrast entering the v2 AF rows gets ONE route family "
    f"({FAMILIES}) for its active arm, by the first tier that yields a family: (T1) text of the trial "
    "interventions that map to the drug of interest (name, description, other names); (T2) the linked arm "
    "description plus the event-group title and description; (T3) 'label_inferred' = the single route "
    "family of all current FDA human-prescription labels of the ingredient. Formulation-specific patterns "
    "(e.g. intravitreal, intra-articular, nasal spray, inhaler) take precedence over generic words "
    "('injection', 'solution'); two or more families at the deciding tier = 'mixed' (excluded); nothing "
    "found = 'unresolved' (excluded). PASS if >= 70% of contrasts get a family (T1-T3); the share coded "
    "from trial text alone (T1-T2) is reported beside it.")

ANALYSIS = (
    "ROUTE-STRATIFIED AF (af_route). Cells are (ingredient, route family, MedDRA term) with every v2 rule "
    "(US spelling, >= 2 trials, >= 20 events, Mantel-Haenszel RR). A side effect is ON THE ROUTE-MATCHED "
    "LABEL if it is in the ADVERSE REACTIONS section of ANY current innovator (NDA/BLA) label of the "
    "ingredient WITH THE SAME ROUTE FAMILY (no innovator label in that family -> all labels in that "
    "family); strata with no label in that family are excluded from label analyses. PASS CRITERIA: "
    "(1) validity: one-sided Mann-Whitney P < 0.001 and positive median log-RR difference, on vs not on "
    "the route-matched label; (2) headline robust to formulation: among route-matched label-listed cells "
    "the AF_hi<0.5 fraction lies within 5 percentage points of v2's 48.6% AND the AF<=0 fraction within 5 "
    "points of v2's 25.5%. Metric = number of failed criteria; PASS iff 0. SENSITIVITY (reported, not "
    "gating): text-coded routes only (T1-T2). DESCRIPTIVE: route-specific probability table (the 20 "
    "ingredient-route strata with the most trials among those with >= 1 route-matched label-listed "
    "estimable cell); for drug-term pairs estimable in >= 2 route families, the share with between-route "
    "heterogeneity (Cochran Q P < 0.01).")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_route_coding", CODING, gate_family="qc", metric_name="frac_coded",
                          comparator=">=", threshold=0.70,
                          notes="registered before any contrast route was coded")
        run.register_gate("af_route", ANALYSIS, gate_family="pre_registration",
                          metric_name="n_criteria_failed", comparator="==", threshold=0.0,
                          notes="registered after af_v2 evaluation, before route-level labels or cells exist")
        print("  registered af_route_coding, af_route")
    return 0


if __name__ == "__main__":
    sys.exit(main())
