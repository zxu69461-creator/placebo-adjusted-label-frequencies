#!/usr/bin/env python3
"""Gates jno_ext_ctt and jno_ext_acei_cough: external concordance of the adopted estimator (AF 3.3).

CTT: our statin pairs (ATC4 C10AA) whose term is one of an outcome's MedDRA preferred terms are pooled per outcome
(precision-weighted mean of posterior-mean log hazard ratios) and compared with the CTT individual-participant-data rate
ratios (ext_ctt_outcome). ACE inhibitors (ATC4 C09AA) x cough are pooled the same way and compared with the
placebo-adjusted fraction of 0.37 from Vukadinovic et al. Rules as registered in 115_jno_prereg.py.
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("m50", ROOT / "scripts" / "50_af_fda_analysis.py")
M50 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M50)
SEED, NPERM = 20261012, 10_000
CTT_TRIALS = ["4S", "WOSCOPS", "CARE", "AFCAPS", "TexCAPS", "LIPID", "LIPS", "HPS", "Heart Protection", "PROSPER",
              "ASCOT", "ALERT", "CARDS", "4D", "ASPEN", "SPARCL", "CORONA", "GISSI-HF", "GISSI HF", "JUPITER",
              "AURORA", "HOPE-3", "HOPE 3"]


def key(t):
    return M50.us(" ".join(str(t).lower().split()))


def pool(d):
    w = 1 / d.psi_sd.clip(lower=1e-6) ** 2
    est = float((w * d.psi_mean).sum() / w.sum()); se = float(w.sum() ** -0.5)
    return est, se


def ef(x):
    return 1 - np.exp(-x)


def main() -> int:
    with pv.connect(read_only=True) as con:
        cell = con.execute("""SELECT c.drug, c.term, c.n_trials, c.psi_mean, c.psi_sd, d.atc4
                              FROM afstd_cell c JOIN ref_chembl_drug d ON d.chembl_id = c.drug
                              WHERE c.estimator = 'AF 3.3'""").df()
        ctt = con.execute("SELECT outcome, soc, pts, rr, rr_lo, rr_hi, fdr_sig FROM ext_ctt_outcome").df()
        trials = con.execute("""SELECT DISTINCT a.nct, t.title, t.enrollment, list_extract(a.drug_ids, 1) AS drug
                                FROM atlas_contrast a JOIN ctgov_trial t USING (nct) WHERE a.single_ingredient""").df()
    has = lambda codes, c: c in list(codes) if codes is not None else False  # noqa: E731
    st = cell[cell.atc4.map(lambda a: has(a, "C10AA"))].copy()
    ace = cell[cell.atc4.map(lambda a: has(a, "C09AA"))].copy()
    st["k"] = st.term.map(key)
    rows = []
    for _, o in ctt.iterrows():
        pts = {key(p) for p in o.pts}
        d = st[st.k.isin(pts)]
        r = dict(benchmark="CTT", outcome=o.outcome, n_pairs=len(d), n_drugs=d.drug.nunique(),
                 n_trials=int(d.n_trials.sum()), ref_ratio=o.rr, ref_lo=o.rr_lo, ref_hi=o.rr_hi,
                 ref_ef=1 - 1 / o.rr, ref_sig=bool(o.fdr_sig))
        if len(d):
            est, se = pool(d)
            r.update(our_loghr=est, our_se=se, our_ef=ef(est), our_ef_lo=ef(est - 1.96 * se), our_ef_hi=ef(est + 1.96 * se))
        rows.append(r)
    res = pd.DataFrame(rows)
    ev = res.dropna(subset=["our_loghr"]).copy()
    n_ev = len(ev)
    det = dict(n_outcomes=len(res), n_evaluable=n_ev, n_statin_pairs_used=int(ev.n_pairs.sum()))
    passed, rho = None, np.nan
    if n_ev >= 10:
        x, y = ev.our_loghr.values, np.log(ev.ref_ratio.values)
        rho = float(stats.spearmanr(x, y).statistic)
        rng = np.random.default_rng(SEED)
        perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(NPERM)])
        p_one = float((1 + (perm >= rho).sum()) / (NPERM + 1))
        agree = float((ev.our_ef < 0.5).mean())
        ev["rank_our"] = ev.our_loghr.rank(ascending=False).astype(int)
        sig = ev[ev.ref_sig][["outcome", "our_ef", "our_ef_lo", "our_ef_hi", "rank_our", "ref_ef"]]
        passed = bool(rho > 0 and p_one < 0.05 and agree >= 0.80)
        det.update(spearman_rho=rho, p_one_sided=p_one, agreement_ef_lt_half=agree,
                   mean_diff_loghr=float((x - y).mean()), fdr_sig_outcomes=sig.to_dict("records"),
                   pearson_r=float(np.corrcoef(x, y)[0, 1]))
    # overlap with CTT trials among statin trials in our data
    stat_drugs = set(st.drug)
    tr = trials[trials.drug.isin(stat_drugs)].copy()
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in CTT_TRIALS) + r")\b", re.I)
    tr["ctt_name"] = tr.title.fillna("").map(lambda s: (pat.search(s).group(1) if pat.search(s) else None))
    det.update(n_statin_trials=int(tr.nct.nunique()), n_statin_trials_ge1000=int((tr.enrollment >= 1000).sum()),
               statin_trials_ge1000=tr[tr.enrollment >= 1000][["nct", "title", "enrollment"]].to_dict("records"),
               possible_ctt_overlap=tr.dropna(subset=["ctt_name"])[["nct", "title", "enrollment", "ctt_name"]]
               .to_dict("records"))
    print({k: v for k, v in det.items() if k not in ("statin_trials_ge1000",)})

    a = ace[ace.term.map(key) == "cough"]
    est, se = pool(a)
    ace_det = dict(n_pairs=len(a), drugs=sorted(a.drug), n_trials=int(a.n_trials.sum()), loghr=est, se=se,
                   ef=ef(est), ef_lo=ef(est - 1.96 * se), ef_hi=ef(est + 1.96 * se), benchmark=0.37,
                   consistent=bool(ef(est - 1.96 * se) <= 0.37 <= ef(est + 1.96 * se)))
    print("ACE-I cough", ace_det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_cell','ext_ctt_outcome','ref_chembl_drug')").fetchall()]
        acer = pd.DataFrame([dict(benchmark="ACEI_cough_Vukadinovic2019", outcome="Cough", n_pairs=len(a),
                                  n_drugs=a.drug.nunique(), n_trials=ace_det["n_trials"], our_loghr=est, our_se=se,
                                  our_ef=ace_det["ef"], our_ef_lo=ace_det["ef_lo"], our_ef_hi=ace_det["ef_hi"],
                                  ref_ratio=np.nan, ref_lo=np.nan, ref_hi=np.nan, ref_ef=0.37, ref_sig=True)])
        out = pd.concat([res, acer], ignore_index=True); out["run_id"] = run.run_id
        cols = ["run_id", "benchmark", "outcome", "n_pairs", "n_drugs", "n_trials", "our_loghr", "our_se", "our_ef",
                "our_ef_lo", "our_ef_hi", "ref_ratio", "ref_lo", "ref_hi", "ref_ef", "ref_sig"]
        run.archive_table("jno_ext_concordance", out[cols], upstream=up, mode="replace_all")
        run.evaluate_gate("jno_ext_ctt", rho if rho == rho else None, passed, detail=det)
        run.evaluate_gate("jno_ext_acei_cough", ace_det["ef"], None if not len(a) else ace_det["consistent"],
                          detail=ace_det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
