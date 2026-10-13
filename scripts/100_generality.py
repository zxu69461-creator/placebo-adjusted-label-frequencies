#!/usr/bin/env python3
"""Generality benchmarks (gates registered by scripts/99b_generality_prereg.py).
  newdrug  : hold out every trial of a random 10% of drugs with >= 3 trials (seed 20260927); predict their records from
             drug class, molecular target and side-effect features only (all cells unseen).
  serious  : serious adverse events (fully reported), confirmation split (trials from 2017-02 vs earlier).
Usage: 100_generality.py <newdrug|serious> <AF version name>"""
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import importlib.util  # noqa: E402
from oajoint import af3, bench  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("b98", ROOT / "scripts" / "98_bench2.py")
B = importlib.util.module_from_spec(spec); sys.modules["b98"] = B; spec.loader.exec_module(B)
SEED = 20260927


def main() -> int:
    mp.set_start_method("fork", force=True)
    mode, version = sys.argv[1], sys.argv[2]
    kind = "serious" if mode == "serious" else "other"
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con, kind=kind)
        cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"})
        cf_all = af3.cell_features_v31(con, cells)[B.F31 + list(B.M31)]
    soc_of_term = dict(zip(cf_all.term, cf_all.soc)); drug_atc3 = cf_all.drop_duplicates("drug").set_index("drug").atc3.to_dict()
    if mode == "newdrug":
        nt = K.groupby("drug").nct.nunique()
        elig = np.array(sorted(nt.index[nt >= 3]))
        held = set(np.random.default_rng(SEED).choice(elig, int(round(.1 * len(elig))), replace=False))
        test = set(K.nct[K.drug.isin(held)]); train = set(K.nct) - test
        name = "newdrug"
    else:
        train, test = bench.split(K, "2017-02-01", "2100-01-01"); name = "serious_C2017"
    tag = "" if version == "AF 3.2" else version.replace("AF ", "").replace(".", "")
    variants = {version: B.VARIANTS[version]}
    B.KIND = kind
    # fit window with explicit train/test sets (reuse the runner with a patched split)
    orig = bench.split
    bench.split = lambda K_, c_, e_: (train, test)
    try:
        W = B.run_window(name, None, None, arms, K, cf_all, variants)
    finally:
        bench.split = orig
    Kt = W["Kt"]
    if mode == "newdrug":
        Kt_mask = Kt.drug.isin(held).values
        for m in list(W["preds"]):
            W["preds"][m][~Kt_mask] = np.nan
    # B6 class-mean baseline
    d = bench.trial_records(K[K.nct.isin(train)], arms[arms.k.isin(K.k[K.nct.isin(train)])])
    CM = bench.class_mean(d, drug_atc3)
    keys = pd.MultiIndex.from_arrays([Kt.drug.map(drug_atc3).values, Kt.t.values])
    ci = CM.reindex(keys)
    ok = ci.cls_mu.notna().values
    q = np.full((len(Kt), len(af3.GRID)), np.nan)
    q[ok] = bench._grid_normal(ci.cls_mu.values[ok], ci.cls_se2.values[ok] + ci.cls_t2.values[ok])
    if mode == "newdrug":
        q[~Kt_mask] = np.nan
    W["preds"]["B6 class-mean"] = q.astype(np.float32); W["scales"]["B6 class-mean"] = "rr"
    R = pd.concat([B.score_window(W, sc, soc_of_term, K) for sc in ("jeffreys", "hier")], ignore_index=True)
    R = R[R.logscore.notna()]
    S = pd.concat([B.summarise(R[R.scoring == sc].drop(columns="scoring"), f"generality_{mode}{tag}", version).assign(scoring=sc)
                   for sc in R.scoring.unique()], ignore_index=True)
    show = S[S.window == "pooled"]
    print(show[["scoring", "stratum", "method", "n_records", "mean_logscore", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi",
                "pit_central90"]].to_string(), flush=True)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN ('atlas_contrast',
                  'ctgov_event_group','ctgov_trial','ctgov_baseline','ctgov_condition','ref_chembl_drug','ref_chembl_mechanism')""").fetchall()]
        R["run_id"] = run.run_id; R["bench"] = f"generality_{mode}{tag}"; S["run_id"] = run.run_id
        run.archive_table("bench_record", R.rename(columns={"window": "win"}), upstream=up)
        run.archive_table("bench_score", S.rename(columns={"window": "win"}), upstream=up)
        pd.to_pickle(S, Path(os.environ.get("OAJOINT_SCRATCH", "/tmp")) / f"gen_{mode}.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
