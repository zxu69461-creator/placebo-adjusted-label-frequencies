#!/usr/bin/env python3
"""Gate jno_trial_characteristics: Table 1 of the JAMA Network Open manuscript (trials contributing to the 214,010 pairs
and the label-listed pairs), written to jno_summary (analysis 'jno_trial_characteristics')."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

ATC1 = {"A": "Alimentary tract and metabolism", "B": "Blood and blood-forming organs", "C": "Cardiovascular system",
        "D": "Dermatologicals", "G": "Genitourinary system and sex hormones", "H": "Systemic hormonal preparations",
        "J": "Anti-infectives for systemic use", "L": "Antineoplastic and immunomodulating agents",
        "M": "Musculoskeletal system", "N": "Nervous system", "P": "Antiparasitic products", "R": "Respiratory system",
        "S": "Sensory organs", "V": "Various", "?": "Not classified"}


def main() -> int:
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        atc = con.execute("SELECT chembl_id AS drug, list_extract(atc4, 1) AS atc4 FROM ref_chembl_drug").df()
        phase = con.execute("SELECT nct, phase FROM ctgov_trial").df()
        pairs = con.execute("SELECT drug, term, n_trials, evidence, on_fda FROM afstd_pair").df()
    T = K.drop_duplicates("nct")[["nct", "wk", "open_label", "single_blind", "industry", "phase1", "healthy", "year",
                                  "age", "pct_female", "enrollment"]].copy()
    T = T.merge(phase, on="nct", how="left")
    n = len(T)
    rows = []

    def add(grp, metric, value, lo=np.nan, hi=np.nan, n_pairs=n):
        rows.append(dict(analysis="jno_trial_characteristics", set_name="trials", grp=grp, n_pairs=int(n_pairs),
                         n_drugs=0, metric=metric, value=float(value), lo=float(lo), hi=float(hi)))

    add("all", "n_trials", n)
    add("all", "participants", T.enrollment.sum())
    q = T.enrollment.quantile([.25, .5, .75])
    add("all", "enrollment_median", q[.5], q[.25], q[.75])
    ph = T.phase.map(lambda p: ",".join(sorted(p)) if p is not None and len(p) else "NA")

    def phase_class(p):
        if "PHASE4" in p:
            return "Phase 4"
        if "PHASE3" in p:
            return "Phase 3" if "PHASE2" not in p else "Phase 2/3"
        if "PHASE2" in p:
            return "Phase 2" if "PHASE1" not in p else "Phase 1/2"
        if "PHASE1" in p or "EARLY_PHASE1" in p:
            return "Phase 1"
        return "Not applicable"
    for k, v in ph.map(phase_class).value_counts().items():
        add(k, "phase_n", v)
    mask = np.select([T.open_label == 1, T.single_blind == 1], ["Open label", "Single blind"], "Double blind or more")
    for k, v in pd.Series(mask).value_counts().items():
        add(k, "masking_n", v)
    add("Industry", "sponsor_n", (T.industry == 1).sum())
    add("Healthy volunteers", "population_n", (T.healthy == 1).sum())
    yr = T.year.dropna()
    for lab, a, b in [("Before 2008", 0, 2008), ("2008-2012", 2008, 2013), ("2013-2017", 2013, 2018),
                      ("2018 or later", 2018, 3000)]:
        add(lab, "start_n", ((yr >= a) & (yr < b)).sum())
    add("Missing", "start_n", T.year.isna().sum())
    w = T.wk.dropna(); qw = w.quantile([.25, .5, .75])
    add("all", "followup_weeks_median", qw[.5], qw[.25], qw[.75], n_pairs=len(w))
    a = T.age.dropna(); qa = a.quantile([.25, .5, .75])
    add("all", "age_median_of_means", qa[.5], qa[.25], qa[.75], n_pairs=len(a))
    f = T.dropna(subset=["pct_female", "enrollment"])
    add("all", "women_pct_weighted", 100 * (f.pct_female * f.enrollment).sum() / f.enrollment.sum(), n_pairs=len(f))
    # drug class of trials (first ATC level-1 letter of each drug in the trial; a trial counts once per class)
    kd = K[["nct", "drug"]].drop_duplicates().merge(atc, on="drug", how="left")
    kd["atc1"] = kd.atc4.map(lambda s: s[0] if isinstance(s, str) and s else "?")
    for k, v in kd.drop_duplicates(["nct", "atc1"]).atc1.value_counts().items():
        add(ATC1.get(k, k), "atc1_trials_n", v)
    add("all", "n_drugs", K.drug.nunique())
    add("all", "n_records", len(K))
    # label-listed pairs
    L = pairs[pairs.on_fda.fillna(False).astype(bool)]
    for grp, metric, val in [("listed", "n_pairs", len(L)), ("listed", "n_drugs", L.drug.nunique()),
                             ("listed", "n_terms", L.term.nunique()), ("all_pairs", "n_pairs", len(pairs)),
                             ("all_pairs", "n_terms", pairs.term.nunique())]:
        rows.append(dict(analysis="jno_trial_characteristics", set_name="pairs", grp=grp, n_pairs=int(val),
                         n_drugs=0, metric=metric, value=float(val), lo=np.nan, hi=np.nan))
    for ev, v in L.evidence.value_counts().items():
        rows.append(dict(analysis="jno_trial_characteristics", set_name="pairs", grp=f"listed_evidence_{ev}",
                         n_pairs=int(v), n_drugs=0, metric="n_pairs", value=float(v), lo=np.nan, hi=np.nan))
    sm = pd.DataFrame(rows)
    print(sm[["set_name", "grp", "metric", "value", "lo", "hi"]].to_string())
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','atlas_contrast','ctgov_trial','ctgov_baseline',"
                                            "'ref_chembl_drug')").fetchall()]
        sm["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_trial_characteristics'")
        run.archive_table("jno_summary", sm, upstream=up)
        run.evaluate_gate("jno_trial_characteristics", float(n), None,
                          detail=dict(n_trials=n, participants=float(T.enrollment.sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
