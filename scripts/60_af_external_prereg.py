#!/usr/bin/env python3
"""Pre-register an external-concordance check of the conventional estimates against published meta-analyses.

Pairs are chosen by rule, not by looking: the 10 drug-route strata with the most trials in
af_route_probability and, within each, the 2 label-listed side effects estimable in the most
trials (ties broken alphabetically). For each pair a published meta-analysis of randomized
placebo-controlled trials of THAT drug reporting a pooled RR or OR for THAT event (or its
MedDRA-equivalent wording) is sought on PubMed; the estimate is taken from the abstract.
DISCLOSURE: some atlas AF values for these drugs were already seen in Figs C1-C4; the pair
list is fixed by the rule above, which does not use AF.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = (
    "EXTERNAL CONCORDANCE. Pair list fixed by rule: top-10 drug-route strata of af_route_probability by "
    "n_trials_stratum; within each, the 2 route-matched label-listed terms with the most trials (ties "
    "alphabetical). For each pair, the first PubMed-indexed meta-analysis of randomized placebo-controlled "
    "trials of that drug (drug-specific, not class-pooled) whose abstract reports a pooled RR or OR vs "
    "placebo for that event (synonyms allowed, e.g. diarrhoea/diarrhea, dizziness/vertigo not allowed) is "
    "used; search query and PMID recorded per pair; pairs with no such abstract are 'not found' and "
    "excluded. PASS if >= 8 pairs are found AND Spearman rho(published log ratio, atlas log RR) >= 0.50 AND "
    "sign agreement (both > 0 or both <= 0) >= 80%. Fewer than 8 found -> UNEVALUABLE. OR is used as "
    "reported (not converted); non-independence (shared trials) is a stated limitation.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_external_concordance", RULE, gate_family="pre_registration",
                          metric_name="spearman_rho", comparator=">=", threshold=0.50,
                          notes="registered before the pair list was generated and before any literature search")
        with pv.connect(read_only=True) as con:
            pairs = con.execute("""
                WITH s AS (SELECT DISTINCT chembl_id, route_family, drug_name, n_trials_stratum FROM af_route_probability
                           ORDER BY n_trials_stratum DESC, chembl_id LIMIT 10),
                r AS (SELECT p.chembl_id, p.route_family, n.display_name, p.term, p.n_trials,
                             row_number() OVER (PARTITION BY p.chembl_id, p.route_family ORDER BY p.n_trials DESC, p.term) rk
                      FROM af_route_probability p JOIN s USING (chembl_id, route_family)
                      LEFT JOIN ref_drug_name n ON n.chembl_id = p.chembl_id)
                SELECT display_name, route_family, term, n_trials FROM r WHERE rk <= 2 ORDER BY display_name, rk""").fetchall()
        for p in pairs:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
