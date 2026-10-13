#!/usr/bin/env python3
"""Gate jno_r2_subsample (rule in 133_jno_r2_prereg.py): uncertainty of the primary outcome from 20 refits to random
halves of the trials (127_jno_rev_refits.py variants 'half:1'..'half:20')."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def main() -> int:
    with pv.connect(read_only=True) as con:
        H = con.execute("""SELECT variant, avg(v.p_af_lt_half) AS share, count(*) AS n FROM jno_cell_variant v
                           JOIN afstd_pair p USING (drug, term) WHERE p.on_fda AND variant LIKE 'half:%'
                           GROUP BY 1 ORDER BY 1""").df()
        point = con.execute("""SELECT value FROM afstd_summary WHERE analysis = 'headline' AND set_name = 'primary'
                               AND grp = 'all' AND metric = 'exp_lt_half'""").fetchone()[0]
    assert len(H) == 20, len(H)
    sd = float(H.share.std(ddof=1))
    det = dict(point=float(point), replicates=dict(zip(H.variant, H.share.round(5))),
               n_listed_per_replicate=[int(x) for x in H.n], mean=float(H.share.mean()), sd=sd,
               min=float(H.share.min()), max=float(H.share.max()), ci=[point - 1.96 * sd, point + 1.96 * sd],
               mean_shift=float(H.share.mean() - point))
    print({k: v for k, v in det.items() if k not in ("replicates", "n_listed_per_replicate")})
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('jno_cell_variant','afstd_pair','afstd_summary')").fetchall()]
        R = pd.DataFrame([dict(analysis="jno_r2_subsample", set_name="half_sample", grp="all", n_pairs=int(H.n.mean()),
                               n_drugs=0, metric="exp_lt_half", value=float(point), lo=det["ci"][0], hi=det["ci"][1]),
                          dict(analysis="jno_r2_subsample", set_name="half_sample_mean", grp="all",
                               n_pairs=int(H.n.mean()), n_drugs=0, metric="exp_lt_half", value=det["mean"],
                               lo=det["min"], hi=det["max"])])
        R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_r2_subsample'")
        run.archive_table("jno_summary", R, upstream=up)
        run.evaluate_gate("jno_r2_subsample", sd, None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
