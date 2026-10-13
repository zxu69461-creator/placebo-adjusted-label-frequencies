#!/usr/bin/env python3
"""BENCH-2.2 baseline iteration: the class-informed Bayesian meta-analysis (B7) added after AF 3.2's confirmation. Gate
bench22_af32_vs_b7 (registered here before scoring): AF 3.2's cached confirmation-zone predictions versus B7 fitted on the
same training trials, BENCH-2.1 scoring. AF 3.2 is not refitted."""
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
SP = Path(os.environ.get("OAJOINT_SCRATCH", "/tmp"))
RULE = """BENCH-2.2 baseline iteration (registered after AF 3.2's confirmation, before this comparison is computed): B7 =
class-informed Bayesian normal-normal meta-analysis (prior mean of each drug x event effect = its ATC3 class's random-effects
mean for the event, variance = class se^2 + class tau^2; half-normal(0.5) heterogeneity; unseen cells get the class prior).
Fitted on the confirmation split's training trials; AF 3.2's cached confirmation predictions are re-scored, not refitted.
PASS iff AF 3.2 minus B7 mean log score on AF 1.0 support has bootstrap 95% CI lower limit > 0 (BENCH-2.1 scoring). Also
reported: all seen cells and unseen cells."""


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("bench22_af32_vs_b7", " ".join(RULE.split()), gate_family="validation", metric_name="diff_ci_lo",
                          threshold=0.0, comparator=">")
    W = pd.read_pickle(SP / "benchfit_bench21_confirm_C2017.pkl")
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"})
        cf_all = af3.cell_features(con, cells)
    drug_atc3 = cf_all.drop_duplicates("drug").set_index("drug").atc3.to_dict(); soc_of_term = dict(zip(cf_all.term, cf_all.soc))
    Kt = W["Kt"]; Ktr = K[K.nct.isin(W["train"])]
    c, TT, d = bench.baselines(Ktr, arms[arms.k.isin(Ktr.k)])
    CM = bench.class_mean(d, drug_atc3)
    keys = list(zip(Kt.drug, Kt.t))
    W["preds"] = {m: W["preds"][m] for m in ("AF 3.2", "AF 3.0", "B3 Bayes-NN")}
    W["preds"]["B7 class-Bayes"] = bench.class_bayes_predictive(c, CM, keys, drug_atc3).astype(np.float32)
    W["scales"]["B7 class-Bayes"] = "rr"
    R = B.score_window(W, "hier", soc_of_term, K)
    R = R[~R.method.str.contains("no dose")]
    S = B.summarise(R.drop(columns="scoring"), "bench22_confirm", "AF 3.2").assign(scoring="hier")
    t = S[(S.window == "pooled") & (S.stratum.isin(["AF1-support", "seen", "unseen"]))]
    print(t[["stratum", "method", "n_records", "mean_logscore", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi", "pit_central90"]].to_string())
    r = t[(t.stratum == "AF1-support") & (t.method == "B7 class-Bayes")].iloc[0]
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN ('bench_score','bench_record')").fetchall()]
        S["run_id"] = run.run_id
        run.archive_table("bench_score", S.rename(columns={"window": "win"}), upstream=up)
        run.evaluate_gate("bench22_af32_vs_b7", float(r.diff_ci_lo), bool(r.diff_ci_lo > 0),
                          detail=t.drop(columns=["run_id"], errors="ignore").to_dict("records"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
