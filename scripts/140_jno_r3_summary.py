#!/usr/bin/env python3
"""Evaluate jno_r3_phase34, jno_r3_approval, jno_r3_indication and jno_r3_highdose (rules in 137_jno_r3_prereg.py)
from the refits of 127_jno_rev_refits.py (variants phase34, preapproval, postapproval, indication, highdose)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("s110", ROOT / "scripts" / "110_afstd_findings.py")
S110 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S110)
SEED = 20261013
VARIANTS = {"phase34": "jno_r3_phase34", "preapproval": "jno_r3_approval", "postapproval": "jno_r3_approval",
            "indication": "jno_r3_indication", "highdose": "jno_r3_highdose"}


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        V = con.execute("SELECT * EXCLUDE (run_id) FROM jno_cell_variant WHERE variant IN ('phase34','preapproval',"
                        "'postapproval','indication','highdose')").df()
        L = con.execute("SELECT drug, term, p_af_lt_half AS p_main FROM afstd_pair WHERE on_fda").df()
    assert set(VARIANTS) <= set(V.variant), sorted(set(VARIANTS) - set(V.variant))
    det, rows = {}, []
    for var in VARIANTS:
        d = V[V.variant == var].merge(L, on=["drug", "term"])
        lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
        r = dict(n=int(len(d)), drugs=int(d.drug.nunique()), exp_lt_half=float(d.p_af_lt_half.mean()), ci=[lo, hi],
                 main_same_pairs=float(d.p_main.mean()), conf_below=float((d.p_af_lt_half > .975).mean()),
                 conf_above=float((d.p_af_lt_half < .025).mean()))
        det[var] = r
        rows.append(dict(analysis=VARIANTS[var], set_name=var, grp="all", n_pairs=r["n"], n_drugs=r["drugs"],
                         metric="exp_lt_half", value=r["exp_lt_half"], lo=lo, hi=hi))
        print(var, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('jno_cell_variant','afstd_pair','ref_chembl_indication')").fetchall()]
        R = pd.DataFrame(rows); R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis IN ('jno_r3_phase34','jno_r3_approval',"
                        "'jno_r3_indication','jno_r3_highdose')")
        run.archive_table("jno_summary", R, upstream=up)
        run.evaluate_gate("jno_r3_phase34", det["phase34"]["exp_lt_half"], None, detail=det["phase34"])
        run.evaluate_gate("jno_r3_approval", det["preapproval"]["exp_lt_half"], None,
                          detail=dict(preapproval=det["preapproval"], postapproval=det["postapproval"]))
        run.evaluate_gate("jno_r3_indication", det["indication"]["exp_lt_half"], None, detail=det["indication"])
        run.evaluate_gate("jno_r3_highdose", det["highdose"]["exp_lt_half"], None, detail=det["highdose"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
