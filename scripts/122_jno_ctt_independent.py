#!/usr/bin/env python3
"""Gate jno_ext_ctt_independent (post hoc): refit AF 3.3 without the 2 CTT trials found in our data (AURORA, JUPITER)
and repeat the jno_ext_ctt comparison unchanged."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("s117", ROOT / "scripts" / "117_jno_external.py")
s117 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s117)
DROP = {"NCT00240331", "NCT00239681"}


def main() -> int:
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        keep = set(K.nct) - DROP
        assert DROP <= set(K.nct), "expected both CTT trials among contributing trials"
        P = af3.prepare(con, arms, K, trials=keep)
        atc = con.execute("SELECT chembl_id AS drug, atc4 FROM ref_chembl_drug").df()
        ctt = con.execute("SELECT outcome, pts, rr, fdr_sig FROM ext_ctt_outcome").df()
    fit, S, base12, binfo = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, spike=True)
    S = S.merge(atc, on="drug", how="left")
    st = S[S.atc4.map(lambda a: a is not None and "C10AA" in list(a))].copy()
    st["k"] = st.term.map(s117.key)
    rows = []
    for _, o in ctt.iterrows():
        d = st[st.k.isin({s117.key(p) for p in o.pts})]
        if len(d):
            est, se = s117.pool(d)
            rows.append(dict(outcome=o.outcome, est=est, ef=s117.ef(est), lrr=np.log(o.rr), sig=bool(o.fdr_sig)))
    ev = pd.DataFrame(rows)
    x, y = ev.est.values, ev.lrr.values
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(s117.SEED)
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(s117.NPERM)])
    p_one = float((1 + (perm >= rho).sum()) / (s117.NPERM + 1))
    agree = float((ev.ef < 0.5).mean())
    ev["rank"] = ev.est.rank(ascending=False).astype(int)
    det = dict(dropped=sorted(DROP), n_evaluable=len(ev), spearman_rho=rho, p_one_sided=p_one,
               agreement_ef_lt_half=agree, pearson_r=float(np.corrcoef(x, y)[0, 1]),
               fdr_sig=ev[ev.sig][["outcome", "ef", "rank"]].to_dict("records"))
    print(det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.evaluate_gate("jno_ext_ctt_independent", rho, bool(rho > 0 and p_one < 0.05 and agree >= 0.80), detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
