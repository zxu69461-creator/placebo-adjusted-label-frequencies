#!/usr/bin/env python3
"""Refits of the adopted estimator (AF 3.3) registered in 126_jno_review_prereg.py. Usage:
    127_jno_rev_refits.py label_aware | phase | shared | pre2017 | boot:<b>
Writes one row per estimated pair to jno_cell_variant (variant name; bootstrap replicates keep label-listed pairs only).
Settings as 102_af32_fit.py 'AF 3.3' (em_fit_mix, outer=2, inner=6, spike); only the trials or the prior change."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

COLS = ["drug", "term", "n_trials", "psi_mean", "psi_sd", "af_hz", "p_af_lt_half", "p_no_effect", "p_no_excess"]


def label_cf(con, K):
    cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"}).reset_index(drop=True)
    cf = af3.cell_features(con, cells)
    lm = con.execute("SELECT drug, term, has_fda_label, on_fda FROM afstd_label_match").df()
    sec = con.execute("SELECT drug, term, section_class FROM jno_label_section").df()
    cf = cf.merge(lm, on=["drug", "term"], how="left").merge(sec, on=["drug", "term"], how="left")
    lab3 = np.select([cf.on_fda.fillna(False).astype(bool), cf.has_fda_label.fillna(False).astype(bool)],
                     ["listed", "unlisted"], "nolabel")
    lab4 = np.where(cf.section_class == "trial", "listed_trial", np.where(lab3 == "listed", "listed_other", lab3))
    cf["soc_orig"] = cf.soc
    cf["soc"] = cf.soc + "|" + lab3
    cf["label"] = lab4
    cf["label_soc"] = [f"{a}|{s}" for a, s in zip(lab4, cf.soc_orig)]
    return cf[["drug", "term", "atc3", "atc1", "soc", "mech", "atc1_soc", "atc3_term", "mech_soc", "label", "label_soc"]]


def boot_tables(arms, K, b):
    rng = np.random.default_rng(20261013 + b)
    trials = np.array(sorted(K.nct.unique()))
    pick = rng.choice(trials, len(trials), replace=True)
    Ks, As, nk = [], [], 0
    byK = {n: g for n, g in K.groupby("nct", sort=False)}
    byA = {n: g for n, g in arms.groupby("nct", sort=False)}
    for i, n in enumerate(pick):
        g = byK[n].copy(); remap = dict(zip(g.k, range(nk, nk + len(g))))
        g["k"] = g.k.map(remap); g["nct"] = f"{n}#{i}"; g["j"] = g.j.astype(str) + f"#{i}"
        a = byA[n].copy(); a["k"] = a.k.map(remap); a["nct"] = f"{n}#{i}"
        Ks.append(g); As.append(a); nk += len(g)
    Kb = pd.concat(Ks, ignore_index=True); Ab = pd.concat(As, ignore_index=True)
    Kb["j"] = pd.factorize(Kb.j)[0]
    Ab["j"] = Ab.k.map(dict(zip(Kb.k, Kb.j)))
    return Ab, Kb


def main() -> int:
    variant = sys.argv[1]
    extra = {}
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
        if variant == "label_aware":
            P = af3.prepare(con, arms, K, cf_all=label_cf(con, K))
            extra = dict(feats=af3.FEATS + ("label", "label_soc"))
        elif variant == "phase":
            P = af3.prepare(con, arms, K, trials=set(K[(K.phase1 == 0) & (K.healthy == 0)].nct))
        elif variant == "shared":
            ms = K.groupby("nct").m_share.max()
            P = af3.prepare(con, arms, K, trials=set(ms[ms <= 1].index))
        elif variant == "pre2017":
            P = af3.prepare(con, arms, K, trials=set(K[K.start < pd.Timestamp("2017-02-01")].nct))
        elif variant == "phase34":                            # jno_r3_phase34 (137_jno_r3_prereg.py)
            ph = con.execute("SELECT nct, phase FROM ctgov_trial").fetchall()
            keep = {n for n, p_ in ph if p_ is not None and any(x in ("PHASE3", "PHASE4") for x in p_)}
            P = af3.prepare(con, arms, K, trials=keep)
        elif variant in ("preapproval", "postapproval"):      # jno_r3_approval
            appr = dict(con.execute("SELECT chembl_id, first_approval FROM ref_chembl_drug "
                                    "WHERE first_approval IS NOT NULL").fetchall())
            yr = K.drug.map(appr); sy = K.start.dt.year
            m = yr.notna() & sy.notna() & ((sy <= yr) if variant == "preapproval" else (sy > yr))
            Ks = K[m.values]
            P = af3.prepare(con, arms[arms.k.isin(Ks.k)], Ks)
        elif variant == "indication":                         # jno_r3_indication
            ind = con.execute("""SELECT molecule_chembl_id, parent_molecule_chembl_id, lower(mesh_heading)
                                 FROM ref_chembl_indication WHERE mesh_heading IS NOT NULL""").fetchall()
            app = {}
            for mol, par, h in ind:
                for d_ in {mol, par} - {None}:
                    app.setdefault(d_, set()).add(h)
            cond = {}
            for n, meshes, anc in con.execute("SELECT nct, meshes, ancestors FROM ctgov_condition").fetchall():
                cond.setdefault(n, set()).update(x.lower() for x in (meshes or []) + (anc or []))
            m = np.array([bool(app.get(d_, set()) & cond.get(n, set())) for n, d_ in zip(K.nct, K.drug)])
            Ks = K[m]
            print(f"indication: matched records {m.sum():,} of {len(K):,} ({m.mean():.3f}); trials "
                  f"{Ks.nct.nunique():,}", flush=True)
            P = af3.prepare(con, arms[arms.k.isin(Ks.k)], Ks)
        elif variant == "highdose":                           # jno_r3_highdose
            # one drug arm per trial-drug: the highest known dose (lowest eg_id on ties); arms of unknown dose only when
            # no arm of that drug in the trial has a known dose. The kept dose becomes the reference (no dose offset).
            a_ = arms.copy()
            hk = a_.groupby(["nct", "drug"]).dose_known.transform("max") > 0
            top = a_.ldose.where(a_.dose_known > 0).groupby([a_.nct, a_.drug]).transform("max")
            cand = a_[~hk | ((a_.dose_known > 0) & (a_.ldose >= top - 1e-9))].sort_values(["nct", "drug", "eg_id"])
            fe = cand.groupby(["nct", "drug"]).eg_id.transform("first")
            hk2 = cand.groupby(["nct", "drug"]).dose_known.transform("max") > 0
            sel = cand[(cand.eg_id == fe) | ~hk2].copy()
            sel["ldose_c"] = 0.0
            Ks = K[K.k.isin(sel.k)].copy()
            Ks.loc[Ks.dose_known > 0, "ldose_mean"] = 0.0
            print(f"highdose: arm rows {len(sel):,} of {len(arms):,}", flush=True)
            P = af3.prepare(con, sel, Ks)
        elif variant.startswith("half:"):                     # jno_r2_subsample (133_jno_r2_prereg.py)
            b = int(variant.split(":")[1])
            tr = np.array(sorted(K.nct.unique()))
            keep = set(np.random.default_rng(20261013 + 100 + b).choice(tr, len(tr) // 2, replace=False))
            P = af3.prepare(con, arms, K, trials=keep)
        elif variant.startswith("boot:"):
            Ab, Kb = boot_tables(arms, K, int(variant.split(":")[1]))
            P = af3.prepare(con, Ab, Kb)
        else:
            raise SystemExit(f"unknown variant {variant}")
        listed = con.execute("SELECT drug, term FROM afstd_pair WHERE on_fda").df()
    print(f"{variant}: records {len(P['K']):,}; trials {P['K'].nct.nunique():,}; pairs {len(P['cf']):,}", flush=True)
    fit, S, _, _ = af3.run_fit(P, fit_fn=af3.em_fit_mix, outer=2, inner=6, spike=True, verbose=False, **extra)
    S["p_no_effect"] = fit["p_null"]
    S["n_trials"] = S.n_trials.fillna(0).astype(int)
    out = S[COLS].copy()
    if variant.startswith(("boot:", "half:")):
        out = out.merge(listed, on=["drug", "term"])
    m = out.merge(listed, on=["drug", "term"])
    print(f"{variant}: listed pairs {len(m):,}; expected share AF<0.5 {m.p_af_lt_half.mean():.4f}", flush=True)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','afstd_label_match','jno_label_section','atlas_contrast',"
                                            "'ctgov_event_group','ctgov_trial')").fetchall()]
        out["run_id"] = run.run_id; out["variant"] = variant
        with pv.connect() as con:
            con.execute("DELETE FROM jno_cell_variant WHERE variant = ?", [variant])
        run.archive_table("jno_cell_variant", out[["run_id", "variant"] + COLS], upstream=up)
    return 0


if __name__ == "__main__":
    sys.exit(main())
