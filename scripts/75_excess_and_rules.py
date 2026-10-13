#!/usr/bin/env python3
"""Absolute excess risk / number needed to harm (af_excess_risk) and counterfactual listing rules
(af_label_rules) for side effects on current FDA labels. Descriptive; rules prespecified in 73_upgrade_prereg.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def main() -> int:
    with pv.connect(read_only=True) as con:
        c = con.execute("""SELECT c.drug, c.term, c.risk_pl, c.rr, c.rr_lo, c.af, c.af_lo, c.af_hi FROM af_cell c
                           JOIN af_label_match m USING (drug, term) WHERE m.on_fda""").df()
    c["freq_drug"] = c.risk_pl * c.rr                  # MH-consistent drug-arm frequency
    c["excess"] = c.risk_pl * (c.rr - 1)               # extra reports per treated patient
    pos = c[c.rr > 1]
    ex = dict(n=int(len(c)), median_freq_drug_per100=float(100 * c.freq_drug.median()),
              median_excess_per100=float(100 * c.excess.median()),
              share_excess_lt1_per100=float((c.excess < .01).mean()),
              median_nnh=float((1 / pos.excess).median()), share_nnh_gt100=float(((1 / pos.excess) > 100).mean() * len(pos) / len(c)
                                                                                  + (c.rr <= 1).mean()))
    rules = {"A_frequency_direction": (c.freq_drug >= .02) & (c.rr > 1),
             "B_evidence": c.rr_lo > 1,
             "C_evidence_and_size": (c.rr_lo > 1) & (c.excess >= .01)}
    clear = c.af_lo > .5
    rr = {}
    for k, keep in rules.items():
        rr[k] = dict(retained=float(keep.mean()), n_retained=int(keep.sum()),
                     retained_confidently_background=float((c[keep].af_hi < .5).mean()) if keep.any() else None,
                     clear_drug_caused_retained=float(keep[clear].mean()) if clear.any() else None)
    print(ex); [print(k, v) for k, v in rr.items()]
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.evaluate_gate("af_excess_risk", ex["median_excess_per100"], None, detail=ex)
        run.evaluate_gate("af_label_rules", rr["B_evidence"]["retained"], None, detail=rr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
