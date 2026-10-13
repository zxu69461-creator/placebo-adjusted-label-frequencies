#!/usr/bin/env python3
"""Evaluate gate af_external_concordance.

The literature table below was extracted by hand from PubMed abstracts under the registered rule
(earliest qualifying drug-specific meta-analysis of placebo-controlled RCTs whose abstract gives a
pooled RR/OR vs placebo for the event; composites such as 'nausea and vomiting' do NOT qualify).
Queries: '<drug>[Title] AND placebo AND (meta-analysis[Publication Type] OR "meta-analysis"[Title])
AND (<event terms>[Title/Abstract])', run 2026-09-24.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

Q = '{d}[Title] AND placebo AND (meta-analysis[Publication Type] OR "meta-analysis"[Title]) AND ({e})'
# drug, route, term, pmid, year, measure, estimate, context
FOUND = [
    ("pregabalin", "oral", "dizziness", "25209095", 2014, "RR", 1.33, "perioperative (acute/persistent postoperative pain)"),
    ("pregabalin", "oral", "headache", "28642088", 2017, "RR", 0.95, "pre-emptive, laparoscopic cholecystectomy"),
    ("gabapentin", "oral", "dizziness", "24806220", 2014, "RR", 2.68, "post-herpetic neuralgia, 1800 mg"),
    ("gabapentin", "oral", "nausea", "24807337", 2014, "RR", 0.76, "pre-emptive, abdominal hysterectomy"),
    ("metformin", "oral", "diarrhea", "30039871", 2018, "RR", 2.34, "overweight/obese pregnancy (Cochrane)"),
    ("dapagliflozin", "oral", "urinary tract infection", "31348290", 2019, "RR", 1.74, "type 2 diabetes monotherapy"),
    ("naltrexone", "oral", "nausea", "15850502", 2005, "RR", 2.14, "alcohol dependence, short-term"),
    ("naltrexone", "oral", "headache", "40337423", 2025, "RR", 1.68, "low-dose naltrexone, fibromyalgia"),
]
NOT_FOUND = [("adalimumab", "parenteral", "headache"), ("adalimumab", "parenteral", "upper respiratory tract infection"),
             ("dapagliflozin", "oral", "nasopharyngitis"), ("liraglutide", "parenteral", "nausea"),
             ("liraglutide", "parenteral", "diarrhea"), ("metformin", "oral", "nausea"),
             ("pioglitazone", "oral", "headache"), ("pioglitazone", "oral", "diarrhea"),
             ("secukinumab", "parenteral", "nasopharyngitis"), ("secukinumab", "parenteral", "headache"),
             ("sitagliptin", "oral", "headache"), ("sitagliptin", "oral", "diarrhea")]


def main() -> int:
    with pv.connect(read_only=True) as con:
        cells = con.execute("""SELECT n.display_name AS drug, c.route_family, c.term, c.rr, c.rr_lo, c.rr_hi, c.n_trials
                               FROM af_route_cell c JOIN ref_drug_name n ON n.chembl_id = c.drug""").df()
    key = cells.set_index(["drug", "route_family", "term"])
    rows = []
    for d, r, t, pm, y, m, est, ctx in FOUND:
        a = key.loc[(d, r, t)]
        rows.append(dict(drug=d, route_family=r, term=t, found=True, pmid=pm, pub_year=y, measure=m, estimate=est,
                         context=ctx, query=Q.format(d=d, e=t), atlas_rr=float(a.rr), atlas_rr_lo=float(a.rr_lo),
                         atlas_rr_hi=float(a.rr_hi), atlas_n_trials=int(a.n_trials)))
    for d, r, t in NOT_FOUND:
        a = key.loc[(d, r, t)]
        rows.append(dict(drug=d, route_family=r, term=t, found=False, pmid=None, pub_year=None, measure=None,
                         estimate=None, context=None, query=Q.format(d=d, e=t), atlas_rr=float(a.rr),
                         atlas_rr_lo=float(a.rr_lo), atlas_rr_hi=float(a.rr_hi), atlas_n_trials=int(a.n_trials)))
    df = pd.DataFrame(rows)
    f = df[df.found]
    x, y = np.log(f.estimate.values), np.log(f.atlas_rr.values)
    rho = float(stats.spearmanr(x, y).statistic)
    sign = float(np.mean((x > 0) == (y > 0)))
    n = len(f)
    passed = None if n < 8 else bool(rho >= .5 and sign >= .8)
    print(f.assign(atlas_af=1 - 1 / f.atlas_rr, pub_af=1 - 1 / f.estimate)[
        ["drug", "term", "estimate", "pub_af", "atlas_rr", "atlas_af", "atlas_n_trials", "context"]].to_string(index=False))
    print(f"\nfound {n}/20; Spearman rho = {rho:.3f}; sign agreement = {sign:.0%} -> "
          f"{'UNEVALUABLE' if passed is None else ('PASS' if passed else 'FAIL')}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x_[0] for x_ in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                              "('af_route_cell','ref_drug_name')").fetchall()]
        df["run_id"] = run.run_id
        run.archive_table("af_external_ref", df, upstream=up, mode="replace_all")
        run.evaluate_gate("af_external_concordance", rho, passed, detail={
            "n_found": n, "n_pairs": 20, "spearman": rho, "sign_agreement": sign,
            "rule_interpretation": "'first' = earliest publication among qualifying abstracts; composite "
                                   "events (e.g. nausea and vomiting) do not qualify"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
