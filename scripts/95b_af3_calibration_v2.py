#!/usr/bin/env python3
"""Superseding calibration test (gate af3_calibration_v2) and adoption (af3_adopt_v2): the af3_prospective split and fits,
scored with the placebo risk integrated over its posterior."""
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("v95", ROOT / "scripts" / "95_af3_validate.py")
V = importlib.util.module_from_spec(spec); sys.modules["v95"] = V; spec.loader.exec_module(V)


def main() -> int:
    mp.set_start_method("fork", force=True)
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"})
        cf_all = af3.cell_features(con, cells)[V.FEATS]
    st = K.groupby("nct").start.first(); cut = st.dropna().quantile(.7)
    train = sorted(st.index[(st < cut) | st.isna()]); test = sorted(st.index[st >= cut])
    R, best = V.evaluate_split("prospective_v2", train, test, arms, K, cf_all, p0_mode="beta")
    cal = float(R.set_index("method").loc["AF 3.0", "pit_central90"])
    ok = bool(.87 <= cal <= .93)
    P = V.PIT_STORE["prospective_v2:beta"]
    hist = {m: np.histogram(P[m], bins=10, range=(0, 1))[0].tolist() for m in P}
    with pv.connect(read_only=True) as con:
        g = {n: con.execute("SELECT passed FROM prov_decision WHERE gate_name=?", [n]).fetchone()[0]
             for n in ("af3_sim", "af3_prospective", "af3_validity")}
    flags = dict(af3_sim=bool(g["af3_sim"]), af3_prospective=bool(g["af3_prospective"]), af3_calibration_v2=ok,
                 af3_validity=bool(g["af3_validity"]))
    n_not = sum(not v for v in flags.values())
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            prev = con.execute("SELECT * FROM af3_score WHERE split <> 'prospective_v2'").df()
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('atlas_contrast','ctgov_event_group','ctgov_trial','ctgov_baseline','ctgov_condition','atlas_contrast_route',
                   'ref_chembl_drug','ref_chembl_mechanism','af_label_match','af3_cell','af3_score')""").fetchall()]
        R["run_id"] = run.run_id
        run.archive_table("af3_score", pd.concat([prev, R[prev.columns]]), upstream=up, mode="replace_all")   # v1 rows kept
        run.evaluate_gate("af3_calibration_v2", cal, ok, detail=dict(table=R.drop(columns="run_id").to_dict("records"),
                                                                        pit_histograms=hist, best_competitor=best))
        run.evaluate_gate("af3_adopt_v2", float(n_not), n_not == 0, detail=dict(flags=flags,
                          recommendation="AF 3.0 recommended as primary" if n_not == 0 else "AF 3.0 remains an exploration-branch result"))
    print(json.dumps(dict(calibration_v2=cal, flags=flags, pit_hist=hist), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
