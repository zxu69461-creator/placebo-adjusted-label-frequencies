#!/usr/bin/env python3
"""Build the placebo-controlled contrast table that every atlas analysis reads from.

One row per (trial, active arm). The drug of interest is the active arm's ChEMBL set minus
the placebo arm's, so background therapy given to both arms is never credited to the new
drug (an earlier defect). Placebo event groups are pooled and their counts divided by
k, the number of active arms sharing them (gate atlas_design). The placebo event-group IDs
are stored explicitly so any outcome can later be computed exactly from the term-level
corpus, including term-set outcomes that need the per-arm MAX across terms.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def num(x):
    return pd.to_numeric(x, errors="coerce").astype(float)


def main() -> int:
    with pv.connect(read_only=True) as con:
        eg = con.execute("SELECT * FROM ctgov_event_group").df()
        atc_of = {r[0]: list(r[1] or []) for r in con.execute(
            "SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
    eg["chembl_ids"] = eg["chembl_ids"].apply(lambda x: list(x) if x is not None else [])
    rows = []
    for nct, g in eg.groupby("nct"):
        pl = g[g.role == "placebo"]
        ac = g[(g.role == "active") & (g.chembl_ids.str.len() > 0)]
        if pl.empty or ac.empty:
            continue
        k = len(ac)
        pl_ids = set().union(*[set(x) for x in pl.chembl_ids])
        for _, a in ac.iterrows():
            doi = sorted(set(a.chembl_ids) - pl_ids)
            if not doi:
                continue
            rows.append(dict(
                contrast_id=f"{nct}:{a.eg_id}", nct=nct, eg_active=a.eg_id,
                eg_placebo=sorted(pl.eg_id.tolist()), k=k, drug_ids=doi,
                atc4=sorted({c for cid in doi for c in atc_of.get(cid, [])}),
                single_ingredient=len(doi) == 1,
                n_act_serious=float(num(pd.Series([a.n_risk_serious])).iloc[0]),
                n_act_other=float(num(pd.Series([a.n_risk_other])).iloc[0]),
                n_pl_serious=float(num(pl.n_risk_serious).sum()) / k,
                n_pl_other=float(num(pl.n_risk_other).sum()) / k))
    df = pd.DataFrame(rows)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute(
                "SELECT out_id FROM prov_output_asset WHERE object_name IN "
                "('ctgov_event_group','ref_chembl_drug')").fetchall()]
        df["run_id"] = run.run_id
        run.archive_table("atlas_contrast", df, upstream=up, mode="replace_all")
    print(f"contrasts: {len(df):,} from {df.nct.nunique():,} trials")
    print(f"  single-ingredient (ingredient-level eligible): {int(df.single_ingredient.sum()):,}")
    print(f"  with >=1 ATC-4 class (class-level eligible)   : {int((df.atc4.str.len() > 0).sum()):,}")
    print(f"  distinct drug-of-interest molecules           : {len({d for x in df.drug_ids for d in x}):,}")
    print(f"  shared-placebo contrasts (k>1)                : {int((df.k > 1).sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
