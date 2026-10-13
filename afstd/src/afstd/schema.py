"""Input schema of AF-STD and conversion into model records.

Required tables (pandas DataFrames):
  arms    trial_id, arm_id, role ('placebo' | 'drug'), drug_id (drug arms), event, affected, at_risk,
          optional dose, dose_unit
  trials  trial_id, optional: start_date, followup_weeks, masking ('NONE','SINGLE','DOUBLE','TRIPLE','QUADRUPLE'),
          sponsor_class ('INDUSTRY' or other), phase1_only (bool), healthy_volunteers (bool), mean_age, pct_female, enrollment
  drugs   drug_id, optional atc (list of ATC codes), targets (list of 'target|action' strings)
  events  event, soc (organ system)
Optional:
  assessment  trial_id, event, nonsystematic_share (0-1)
An event contributes from a trial only if the trial reports a row for it for every one of its placebo and drug arms."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = {"arms": ["trial_id", "arm_id", "role", "event", "affected", "at_risk"], "trials": ["trial_id"],
            "drugs": ["drug_id"], "events": ["event", "soc"]}


def _norm(t):
    return " ".join(str(t).lower().split())


@dataclass
class Records:
    arms: pd.DataFrame          # one row per drug arm x event: k, x, n, ldose, ldose_c
    K: pd.DataFrame             # one row per trial x drug x event (record k) with placebo counts and covariates
    cf: pd.DataFrame            # one row per drug x event (cell) with prior features

    @classmethod
    def from_frames(cls, arms, trials, drugs, events, assessment=None):
        for name, df in (("arms", arms), ("trials", trials), ("drugs", drugs), ("events", events)):
            miss = [c for c in REQUIRED[name] if c not in df.columns]
            if miss:
                raise ValueError(f"{name}: missing columns {miss}")
        a = arms.copy()
        a["t"] = a.event.map(_norm)
        a["role"] = a.role.str.lower()
        if not set(a.role.unique()) <= {"placebo", "drug"}:
            raise ValueError("arms.role must be 'placebo' or 'drug'")
        if (a.affected > a.at_risk).any() or (a.at_risk <= 0).any():
            raise ValueError("arms: affected must be <= at_risk and at_risk > 0")
        a = a.rename(columns={"trial_id": "nct", "affected": "x", "at_risk": "n"})
        # placebo pooled per trial x event (every placebo arm must report the event)
        pl = a[a.role == "placebo"]
        npl = pl.groupby("nct").arm_id.nunique()
        P = pl.groupby(["nct", "t"]).agg(x0=("x", "sum"), n0=("n", "sum"), npa=("arm_id", "nunique")).reset_index()
        P = P[P.npa == P.nct.map(npl)]
        # drug arms, relative dose
        d = a[a.role == "drug"].copy()
        if "dose" not in d:
            d["dose"] = np.nan
        if "dose_unit" not in d:
            d["dose_unit"] = "unit"
        d.loc[~(d.dose > 0), "dose"] = np.nan
        ref = d.dropna(subset=["dose"]).drop_duplicates(["nct", "arm_id"]).groupby(["drug_id", "dose_unit"]).dose.median().rename("ref")
        d = d.join(ref, on=["drug_id", "dose_unit"])
        d["ldose"] = np.log(d.dose / d.ref)
        d["dose_known"] = d.ldose.notna().astype(float)
        d["ldose"] = d.ldose.fillna(0.0).clip(-3, 3)
        nd = d.groupby(["nct", "drug_id"]).arm_id.nunique()
        d["n_arms_term"] = d.groupby(["nct", "drug_id", "t"]).arm_id.transform("nunique")
        d = d[d.n_arms_term.values == pd.MultiIndex.from_frame(d[["nct", "drug_id"]]).map(nd.to_dict().get).values]
        d = d.merge(P[["nct", "t"]], on=["nct", "t"]).rename(columns={"drug_id": "drug"})
        J = P.merge(d[["nct", "t"]].drop_duplicates(), on=["nct", "t"]).reset_index(drop=True); J["j"] = np.arange(len(J))
        K = d.groupby(["nct", "drug", "t"], as_index=False).agg(ldose_mean=("ldose", "mean"), dose_known=("dose_known", "min"),
                                                               n_arms=("arm_id", "nunique"))
        K = K.merge(J[["nct", "t", "j", "x0", "n0"]], on=["nct", "t"]); K["k"] = np.arange(len(K))
        ar = d.merge(K[["nct", "drug", "t", "k", "j"]], on=["nct", "drug", "t"])
        ar["ldose_c"] = ar.ldose - ar.k.map(K.set_index("k").ldose_mean)
        # trial covariates
        tr = trials.rename(columns={"trial_id": "nct"}).copy()
        g = lambda c, dflt: tr[c] if c in tr else dflt
        cv = pd.DataFrame({"nct": tr.nct})
        cv["start"] = pd.to_datetime(g("start_date", pd.Series([None] * len(tr))), errors="coerce")
        cv["year"] = cv.start.dt.year + cv.start.dt.dayofyear / 365.25
        cv["wk"] = pd.to_numeric(g("followup_weeks", np.nan), errors="coerce")
        mk = g("masking", pd.Series(["DOUBLE"] * len(tr))).fillna("DOUBLE").str.upper()
        cv["open_label"] = (mk == "NONE").astype(float).values; cv["single_blind"] = (mk == "SINGLE").astype(float).values
        cv["industry"] = (g("sponsor_class", pd.Series(["INDUSTRY"] * len(tr))).fillna("INDUSTRY").str.upper() == "INDUSTRY").astype(float).values
        cv["phase1"] = g("phase1_only", pd.Series([False] * len(tr))).fillna(False).astype(float).values
        cv["healthy"] = g("healthy_volunteers", pd.Series([False] * len(tr))).fillna(False).astype(float).values
        cv["age"] = pd.to_numeric(g("mean_age", np.nan), errors="coerce")
        cv["pct_female"] = pd.to_numeric(g("pct_female", np.nan), errors="coerce")
        cv["enrollment"] = pd.to_numeric(g("enrollment", np.nan), errors="coerce")
        K = K.merge(cv, on="nct", how="left")
        if assessment is not None:
            asd = assessment.rename(columns={"trial_id": "nct", "nonsystematic_share": "nonsys"}).copy()
            asd["t"] = asd.event.map(_norm)
            K = K.merge(asd[["nct", "t", "nonsys"]], on=["nct", "t"], how="left")
        else:
            K["nonsys"] = np.nan
        K["nonsys"] = K.nonsys.fillna(0.0)
        K["m_share"] = K.groupby("j").drug.transform("nunique")
        # cell features
        ev = events.copy(); ev["t"] = ev.event.map(_norm); smap = dict(zip(ev.t, ev.soc.astype(str).str.lower()))
        dr = drugs.set_index("drug_id")
        cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"}).reset_index(drop=True)
        atc = lambda x: sorted(x)[0] if isinstance(x, (list, tuple)) and len(x) else None
        cells["atc3"] = [atc(dr.atc.get(x)) [:4] if "atc" in dr and atc(dr.atc.get(x)) else None for x in cells.drug]
        cells["atc1"] = [a[0] if isinstance(a, str) else None for a in cells.atc3]
        cells["soc"] = cells.term.map(smap).fillna("unknown")
        cells["mech"] = [sorted(dr.targets.get(x))[0] if "targets" in dr and isinstance(dr.targets.get(x), (list, tuple)) and len(dr.targets.get(x)) else None for x in cells.drug]
        cells["atc1_soc"] = [f"{a}|{s}" if a else None for a, s in zip(cells.atc1, cells.soc)]
        cells["atc3_term"] = [f"{a}|{t}" if a else None for a, t in zip(cells.atc3, cells.term)]
        cells["mech_soc"] = [f"{m}|{s}" if isinstance(m, str) else None for m, s in zip(cells.mech, cells.soc)]
        return cls(arms=ar[["nct", "drug", "t", "k", "j", "arm_id", "x", "n", "ldose", "ldose_c"]], K=K, cf=cells)
