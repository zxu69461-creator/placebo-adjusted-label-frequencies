#!/usr/bin/env python3
"""Sensitivity of the headline to the no-effect component (gate afstd_no_effect_sensitivity, registered here before the
numbers are computed): the same label-listed pairs summarised under the full-data fit of the estimator without the
no-effect component (af32_cell)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = ("Descriptive. For the primary set of manuscript v2 (every on_fda pair of afstd_pair), the expected share with "
        "true AF < 0.5 (mean P(AF<0.5); drug-cluster bootstrap 95% interval, 1,000 resamples, seed 20260927) and the share "
        "confidently mostly background (P(AF<0.5) > 0.975) under the full-data fit of the same estimator without the "
        "no-effect component (af32_cell, AF 3.2), compared with the adopted estimator. Registered before computing; the "
        "AF 3.2 values for these pairs have not been summarised before.")


def main() -> int:
    with pv.connect(read_only=True) as con:
        x = con.execute("""SELECT p.drug, p.term, p.evidence, p.p_af_lt_half AS p33, a.p_af_lt_half AS p32
                           FROM afstd_pair p JOIN af32_cell a USING (drug, term) WHERE p.on_fda ORDER BY drug, term""").df()
        n_primary = con.execute("SELECT count(*) FROM afstd_pair WHERE on_fda").fetchone()[0]
    rng = np.random.default_rng(20260927)
    rows = []
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("afstd_no_effect_sensitivity", RULE, gate_family="descriptive", metric_name="exp_lt_half_af32")
        assert len(x) == n_primary, (len(x), n_primary)
        for est, col in (("with no-effect component", "p33"), ("without no-effect component", "p32")):
            g = x.groupby("drug")[col].agg(["sum", "size"])
            w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=1000)
            v = (w @ g["sum"].values) / (w @ g["size"].values)
            rows.append(dict(analysis="no_effect_sensitivity", set_name=est, grp="all", n_pairs=len(x),
                             n_drugs=int(x.drug.nunique()), metric="exp_lt_half", value=float(x[col].mean()),
                             lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975))))
            rows.append(dict(analysis="no_effect_sensitivity", set_name=est, grp="all", n_pairs=len(x),
                             n_drugs=int(x.drug.nunique()), metric="conf_bg", value=float((x[col] > .975).mean()),
                             lo=np.nan, hi=np.nan))
        sm = pd.DataFrame(rows); print(sm.to_string())
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','af32_cell')").fetchall()]
        with pv.connect() as con:
            con.execute("DELETE FROM afstd_summary WHERE analysis = 'no_effect_sensitivity'")
        sm["run_id"] = run.run_id
        run.archive_table("afstd_summary", sm, upstream=up)
        run.evaluate_gate("afstd_no_effect_sensitivity", float(rows[2]["value"]), None,
                          detail=dict(rows=sm.drop(columns="run_id").replace({np.nan: None}).to_dict("records")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
