#!/usr/bin/env python3
"""Descriptive companion of jno_r3_indication: share of all records whose trial conditions (or their MeSH ancestors)
match an approved indication of the drug, with the same matching as 127_jno_rev_refits.py variant 'indication'."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402


def main() -> int:
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        app = {}
        for mol, par, h in con.execute("""SELECT molecule_chembl_id, parent_molecule_chembl_id, lower(mesh_heading)
                                          FROM ref_chembl_indication WHERE mesh_heading IS NOT NULL""").fetchall():
            for d_ in {mol, par} - {None}:
                app.setdefault(d_, set()).add(h)
        cond = {}
        for n, meshes, anc in con.execute("SELECT nct, meshes, ancestors FROM ctgov_condition").fetchall():
            cond.setdefault(n, set()).update(x.lower() for x in (meshes or []) + (anc or []))
    m = np.array([bool(app.get(d_, set()) & cond.get(n, set())) for n, d_ in zip(K.nct, K.drug)])
    det = dict(n_records=int(len(K)), matched=int(m.sum()), share=float(m.mean()),
               trials_matched=int(K[m].nct.nunique()), drugs_with_indication=int(K.drug.isin(app).mean() * 1e6) / 1e6)
    print(det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        R = pd.DataFrame([dict(analysis="jno_r3_indication", set_name="record_match_share", grp="all",
                               n_pairs=det["n_records"], n_drugs=0, metric="share", value=det["share"],
                               lo=np.nan, hi=np.nan)])
        R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_r3_indication' AND set_name = 'record_match_share'")
        run.archive_table("jno_summary", R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
