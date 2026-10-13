#!/usr/bin/env python3
"""POST HOC (prompted by the recall audit, 2026-10-13): drugs whose current-label text came from a different molecule
(a deuterated analogue, the racemate of an enantiomer, an unrelated product sharing a name) or from a combination
product only, identified by comparing each drug's display name with the generic name of its label pool. The list is
fixed below after manual review of the 51 drugs whose names differ (the others are INN vs USAN names, salts or brand
names of the same molecule). Reports the expected share with AF < 0.5 after excluding these drugs. Descriptive."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("s110", ROOT / "scripts" / "110_afstd_findings.py")
S110 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S110)
WRONG = {"CHEMBL1789941": "ruxolitinib -> deuruxolitinib label", "CHEMBL1082407": "deutenzalutamide -> enzalutamide label",
         "CHEMBL1201191": "levocetirizine -> cetirizine label", "CHEMBL117785": "tetrabenazine -> deutetrabenazine label",
         "CHEMBL1743079": "onilcamotide -> teprotumumab label", "CHEMBL1644695": "sodium iodide (125I) -> potassium iodide label",
         "CHEMBL1201335": "glycopyrronium -> combination inhaler label only"}
GATE = "jno_r3_label_molecule"
RULE = ("POST HOC, after the recall audit found labels assigned to a different molecule: exclude the drugs listed in "
        "141b_jno_label_molecule_check.py (wrong molecule or combination-only label, fixed after manual review of the 51 "
        "drugs whose display and label names differ) and report the expected share with AF < 0.5 among the remaining "
        "listed pairs with a drug-cluster bootstrap 95% CI. Descriptive.")


def main() -> int:
    with pv.connect(read_only=True) as con:
        L = con.execute("SELECT drug, term, p_af_lt_half FROM afstd_pair WHERE on_fda").df()
    keep = L[~L.drug.isin(WRONG)]
    lo, hi = S110.boot_mean(keep, "p_af_lt_half", np.random.default_rng(20261013))
    det = dict(drugs=WRONG, n_excluded=int(len(L) - len(keep)), n=int(len(keep)), exp_lt_half=float(keep.p_af_lt_half.mean()),
               ci=[lo, hi], excluded_exp_lt_half=float(L[L.drug.isin(WRONG)].p_af_lt_half.mean()))
    print(det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate(GATE, RULE, gate_family="descriptive", metric_name="expected_share_lt_half")
        R = pd.DataFrame([dict(analysis=GATE, set_name="excl_wrong_molecule", grp="all", n_pairs=det["n"],
                               n_drugs=int(keep.drug.nunique()), metric="exp_lt_half", value=det["exp_lt_half"], lo=lo, hi=hi)])
        R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute(f"DELETE FROM jno_summary WHERE analysis = '{GATE}'")
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','afstd_label_match')").fetchall()]
        run.archive_table("jno_summary", R, upstream=up)
        run.evaluate_gate(GATE, det["exp_lt_half"], None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
