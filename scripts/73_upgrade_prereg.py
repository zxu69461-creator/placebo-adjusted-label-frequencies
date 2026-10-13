#!/usr/bin/env python3
"""Prespecify the three upgrade analyses before any of them is computed:
  af_dose_response  -- within-trial randomized dose contrasts as label-independent evidence of causation
  af_excess_risk    -- absolute excess risk / number needed to harm for label-listed side effects (descriptive)
  af_label_rules    -- counterfactual listing rules applied to the full trial record (descriptive, policy)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

DOSE = (
    "DOSE-RESPONSE (af_dose_response). Trials with >= 2 single-ingredient active arms of the SAME ingredient against "
    "the same placebo. Dose of each arm = first number+unit (mg, mcg/ug/µg, g, mg/kg, mg/m2, IU, units) in, in order, "
    "the event-group title, the arm label and the arm description; arms of one trial must share the unit and have "
    "distinct values, otherwise the trial is excluded. Per trial the highest- and lowest-dose arms are compared "
    "directly (randomized, same trial). Terms canonicalised as in v2; a trial contributes to a term only if it lists "
    "it. Per (ingredient, term): Mantel-Haenszel RR high vs low dose, >= 2 trials and >= 20 events in the two arms. "
    "Analysed set: cells on the current FDA label (v2 rule). PASS iff (a) Spearman rho between the cell's "
    "drug-vs-placebo AF and its log RR(high vs low) > 0 with one-sided P < 0.01, AND (b) among cells with AF lower "
    "95% limit > 0.5 (clearly drug-caused), log RR(high vs low) > 0 in more than half (one-sided sign test P < 0.01). "
    "Descriptive: the same distribution among cells with AF upper limit < 0.5.")
EXCESS = (
    "EXCESS RISK (af_excess_risk, descriptive). For current-FDA-label cells: excess risk = pooled placebo risk x (RR - 1), "
    "i.e. extra reports per 100 treated attributable to the drug; drug-arm frequency = pooled placebo risk x RR; number "
    "needed to harm = 1 / excess risk for cells with RR > 1. Reported: medians and the share of listed side effects "
    "whose excess is < 1 per 100 treated.")
RULES = (
    "COUNTERFACTUAL LISTING RULES (af_label_rules, descriptive). Applied to every current-FDA-label cell using the full "
    "trial record: rule A (frequency and direction, the form of current practice) -- drug-arm frequency >= 2% and "
    "RR > 1; rule B (evidence of excess) -- lower 95% limit of RR > 1; rule C (evidence and size) -- rule B and excess "
    "risk >= 1 per 100. For each rule: share of listed side effects retained, share of the retained that are "
    "confidently mostly background (AF upper limit < 0.5), and share of clearly drug-caused cells (AF lower limit > "
    "0.5) retained.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_dose_response", DOSE, gate_family="pre_registration", metric_name="n_criteria_failed",
                          comparator="==", threshold=0.0, notes="registered before dose parsing or any dose estimate")
        run.register_gate("af_excess_risk", EXCESS, gate_family="descriptive")
        run.register_gate("af_label_rules", RULES, gate_family="descriptive")
        print("  registered af_dose_response, af_excess_risk, af_label_rules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
