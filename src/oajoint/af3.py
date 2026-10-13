"""AF 3.0 (exploration branch): a generative hierarchical model of adverse-event reporting in placebo-controlled trials.

Data layer. Units:
  j  trial x term                          (baseline cumulative hazard alpha_j, shared by all arms of the trial)
  k  trial x drug x term                   (trial-specific log hazard ratio zeta_k at the trial's covariates)
  a  arm rows: pooled placebo per j; one row per drug arm (event group) per k, with its relative dose
Covariates are built from the registry records only (never from outcomes)."""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import af2

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "cache" / "ctgov" / "pages"

# ------------------------------------------------------------------------------------------------ duration
_W = {"minute": 1 / 10080, "hour": 1 / 168, "day": 1 / 7, "week": 1.0, "month": 4.345, "year": 52.18}
_UNIT = {"min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute", "hr": "hour", "hrs": "hour",
         "hour": "hour", "hours": "hour", "d": "day", "day": "day", "days": "day", "wk": "week", "wks": "week",
         "week": "week", "weeks": "week", "mo": "month", "mos": "month", "month": "month", "months": "month",
         "yr": "year", "yrs": "year", "year": "year", "years": "year"}
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "sixteen": 16, "eighteen": 18, "twenty": 20,
         "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(_NUMW, key=len, reverse=True)) + r")"
_UN = r"(" + "|".join(sorted(_UNIT, key=len, reverse=True)) + r")\b"
_RX_A = re.compile(_NUM + r"(?:\s*(?:-|to|or)\s*\d+(?:\.\d+)?)?\s*(?:[a-z]+\s+){0,2}?" + _UN, re.I)   # "28 calendar days"
_RX_B = re.compile(r"\b(day|week|month|year)\s*(\d+(?:\.\d+)?)", re.I)                                  # "Week 52"


def weeks(text):
    """Longest adverse-event collection period stated in the registry time-frame text, in weeks (NaN if none)."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    vals = []
    for m in _RX_A.finditer(text):
        v = m.group(1).lower()
        v = float(v) if v[0].isdigit() else _NUMW.get(v)
        u = _UNIT.get(m.group(2).lower())
        if v is not None and u:
            vals.append(v * _W[u])
    for m in _RX_B.finditer(text):
        vals.append(float(m.group(2)) * _W[m.group(1).lower()])
    vals = [v for v in vals if 1 / 168 <= v <= 520]
    return max(vals) if vals else np.nan


# ------------------------------------------------------------------------------------------------ dose
_DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg/kg|mg/m2|mg/m²|mg|mcg|µg|μg|ug|g|iu|units?)\b", re.I)
_TO_MG = {"mg": 1.0, "g": 1000.0, "mcg": 1e-3, "µg": 1e-3, "μg": 1e-3, "ug": 1e-3}


def dose(text):
    m = _DOSE.search(text if isinstance(text, str) else "")
    if not m:
        return None
    v, u = float(m.group(1).replace(",", ".")), m.group(2).lower()
    if u in _TO_MG:
        return v * _TO_MG[u], "mg"
    return v, {"mg/m²": "mg/m2", "unit": "units"}.get(u, u)


# ------------------------------------------------------------------------------------------------ registry fields
def registry_fields(ncts):
    """Design fields and per-term assessment type from the cached registry pages."""
    need, trial, assess, armdesc = set(ncts), [], [], {}
    for f in sorted(PAGES.glob("page_*.json.gz")):
        for s in json.load(gzip.open(f))["studies"]:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            if nct not in need:
                continue
            dm = ps.get("designModule") or {}
            di = dm.get("designInfo") or {}
            mi = di.get("maskingInfo") or {}
            trial.append(dict(nct=nct, sponsor_class=((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("class"),
                              masking=mi.get("masking"), who_masked=",".join(mi.get("whoMasked") or []),
                              phases=",".join(dm.get("phases") or []), purpose=di.get("primaryPurpose"),
                              completion=((ps.get("statusModule") or {}).get("completionDateStruct") or {}).get("date")))
            for a in (ps.get("armsInterventionsModule") or {}).get("armGroups") or []:
                armdesc[(nct, a.get("label"))] = a.get("description") or ""
            for e in ((s.get("resultsSection") or {}).get("adverseEventsModule") or {}).get("otherEvents") or []:
                assess.append((nct, e.get("term"), e.get("assessmentType")))
    A = pd.DataFrame(assess, columns=["nct", "term", "assessment"])
    A["t"] = A.term.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    A["t"] = A.t.map({t: af2.us(t) for t in A.t.unique()})
    A["nonsys"] = (A.assessment == "NON_SYSTEMATIC_ASSESSMENT").astype(float)
    A = A.groupby(["nct", "t"], as_index=False).nonsys.mean()
    return pd.DataFrame(trial).drop_duplicates("nct"), A, armdesc


# ------------------------------------------------------------------------------------------------ build
def build(con, kind="other"):
    """Return (arms, K, J): arm-level rows and the trial-drug-term (K) and trial-term (J) tables with covariates.
    kind: 'other' (non-serious events, default) or 'serious'."""
    ct, ae, enf = af2.load_arms(con, kind=kind)
    eg = con.execute("SELECT nct, eg_id, eg_title, arm_label FROM ctgov_event_group").df()
    tr = con.execute("SELECT nct, start_date, ae_time_frame, enrollment FROM ctgov_trial").df()
    cond = con.execute("SELECT nct, indication FROM ctgov_condition").df()
    route = con.execute("SELECT contrast_id, route_family FROM atlas_contrast_route").df()
    base = con.execute("SELECT nct, age_value, age_unit, pct_female FROM ctgov_baseline").df() \
        if con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='ctgov_baseline'").fetchone()[0] else None
    ncts = sorted(ct.nct.unique())
    T, A, armdesc = registry_fields(ncts)

    # drug arms (event groups) with doses; placebo arms
    ct = ct.merge(route, on="contrast_id", how="left")
    darm = ct[["nct", "drug", "eg_active", "route_family"]].rename(columns={"eg_active": "eg_id"}).drop_duplicates(["nct", "drug", "eg_id"])
    darm = darm.merge(eg, on=["nct", "eg_id"], how="left")
    ds = [dose(a) or dose(b) or dose(armdesc.get((n, b), "")) for n, a, b in zip(darm.nct, darm.eg_title, darm.arm_label)]
    darm["dose"] = [d[0] if d else np.nan for d in ds]; darm["unit"] = [d[1] if d else None for d in ds]
    ref = darm.dropna(subset=["dose"]).groupby(["drug", "unit"]).dose.median().rename("ref").reset_index()
    darm = darm.merge(ref, on=["drug", "unit"], how="left")
    darm.loc[~(darm.dose > 0), "dose"] = np.nan
    darm["ldose"] = np.log(darm.dose / darm.ref)
    darm["dose_known"] = darm.ldose.notna().astype(float)
    darm["ldose"] = darm.ldose.fillna(0.0).clip(-3, 3)
    parm = ct.explode("eg_placebo")[["nct", "eg_placebo"]].rename(columns={"eg_placebo": "eg_id"}).drop_duplicates()

    ae_i = ae.set_index(["nct", "eg_id"])
    # placebo pooled per trial-term
    pl = parm.merge(ae, on=["nct", "eg_id"])
    npl = parm.merge(ae.groupby(["nct", "eg_id"]).n.max().reset_index(), on=["nct", "eg_id"]).groupby("nct").n.sum()
    P = pl.groupby(["nct", "t"], as_index=False).agg(x0=("x", "sum"), n0=("n", "sum"), n_pl_arms=("eg_id", "nunique"))
    P = P[P.n0 == P.nct.map(npl)]                                    # every placebo arm has a row for the term
    # drug arms per trial-drug-term
    D = darm.merge(ae, on=["nct", "eg_id"])[["nct", "drug", "eg_id", "t", "x", "n", "ldose", "dose_known", "route_family", "tau"]]
    nd = darm.merge(ae.groupby(["nct", "eg_id"]).n.max().reset_index(), on=["nct", "eg_id"]).groupby(["nct", "drug"]).eg_id.nunique()
    D["n_arms_term"] = D.groupby(["nct", "drug", "t"]).eg_id.transform("nunique")
    D = D[D.n_arms_term == pd.MultiIndex.from_frame(D[["nct", "drug"]]).map(nd.to_dict().get).values]
    D = D.merge(P[["nct", "t"]], on=["nct", "t"])                     # term listed and complete in placebo too
    D = D[(D.n > 0)]
    # indices
    J = P.merge(D[["nct", "t"]].drop_duplicates(), on=["nct", "t"]).reset_index(drop=True)
    J["j"] = np.arange(len(J))
    K = D.groupby(["nct", "drug", "t"], as_index=False).agg(xd=("x", "sum"), nd=("n", "sum"), ldose_mean=("ldose", "mean"),
                                                             dose_known=("dose_known", "min"), route_family=("route_family", "first"),
                                                             n_arms=("eg_id", "nunique"))
    K = K.merge(J[["nct", "t", "j", "x0", "n0"]], on=["nct", "t"])
    K["k"] = np.arange(len(K))
    arms = D.merge(K[["nct", "drug", "t", "k", "j"]], on=["nct", "drug", "t"])
    arms["ldose_c"] = arms.ldose - arms.k.map(K.set_index("k").ldose_mean)   # within-trial dose deviation

    # trial covariates
    tr["start"] = pd.to_datetime(tr.start_date.fillna("").str.slice(0, 7) + "-01", format="%Y-%m-%d", errors="coerce")
    tr["wk"] = tr.ae_time_frame.map(weeks)
    cv = tr.merge(T, on="nct", how="right").merge(cond, on="nct", how="left")
    if base is not None:
        b = base.copy()
        b["age"] = np.where(b.age_unit.fillna("").str.startswith("year"), b.age_value, np.nan)
        cv = cv.merge(b[["nct", "age", "pct_female"]], on="nct", how="left")
    else:
        cv["age"], cv["pct_female"] = np.nan, np.nan
    cv["open_label"] = (cv.masking == "NONE").astype(float)
    cv["single_blind"] = (cv.masking == "SINGLE").astype(float)
    cv["industry"] = (cv.sponsor_class == "INDUSTRY").astype(float)
    cv["phase1"] = cv.phases.fillna("").str.contains("PHASE1").astype(float) * (~cv.phases.fillna("").str.contains("PHASE2|PHASE3")).astype(float)
    cv["healthy"] = cv.indication.fillna("").str.lower().str.contains("healthy").astype(float)
    cv["year"] = cv.start.dt.year + cv.start.dt.dayofyear / 365.25
    K = K.merge(cv[["nct", "wk", "open_label", "single_blind", "industry", "phase1", "healthy", "year", "age", "pct_female",
                    "indication", "start", "enrollment"]], on="nct", how="left")
    K["m_share"] = K.groupby("j").drug.transform("nunique")         # drugs sharing the placebo arm of trial-term j
    K = K.merge(A.rename(columns={"t": "t"}), on=["nct", "t"], how="left")
    K["nonsys"] = K.nonsys.fillna(0.0)
    return arms, K, J


# ================================================================================================ model
GRID = np.round(np.arange(-5.0, 5.0 + 1e-9, 0.05), 4)          # log hazard-ratio grid (also the psi grid)
DZ = 0.05
COVS = ["lwk", "wk_missing", "open_label", "single_blind", "nonsys", "nonindustry", "phase1", "healthy",
        "year10", "year_missing", "ldose_mean", "dose_unknown", "age10", "age_missing", "female_c", "female_missing",
        "lenroll"]


def design_covariates(K):
    """Trial-level covariates centred at the reference trial: 12 weeks of follow-up, double-blind, systematic
    assessment, industry-sponsored, phase 2-4 patients, started 2010, the drug's median dose, age 50, 50% women,
    200 participants."""
    X = pd.DataFrame(index=K.index)
    X["lwk"] = np.log(K.wk / 12).where(K.wk.notna(), 0.0)
    X["wk_missing"] = K.wk.isna().astype(float)
    X["open_label"], X["single_blind"], X["nonsys"] = K.open_label, K.single_blind, K.nonsys
    X["nonindustry"] = 1.0 - K.industry
    X["phase1"], X["healthy"] = K.phase1, K.healthy
    X["year10"] = ((K.year - 2010) / 10).where(K.year.notna(), 0.0)
    X["year_missing"] = K.year.isna().astype(float)
    X["ldose_mean"] = K.ldose_mean.where(K.dose_known > 0, 0.0)
    X["dose_unknown"] = 1.0 - K.dose_known
    X["age10"] = ((K.age - 50) / 10).where(K.age.notna(), 0.0).clip(-5, 5)
    X["age_missing"] = K.age.isna().astype(float)
    X["female_c"] = (K.pct_female - 0.5).where(K.pct_female.notna(), 0.0)
    X["female_missing"] = K.pct_female.isna().astype(float)
    X["lenroll"] = np.log(K.enrollment.clip(lower=10).fillna(200) / 200)
    return X[COVS].astype(float)


def rows_for(arms, K, gamma_soc, soc_of_k):
    """Flat arm rows for the profile likelihood: placebo pooled per k (shared placebo split equally between the drugs
    that share it), then each drug arm with its within-trial dose offset gamma_s * (log dose - trial mean)."""
    nk = len(K)
    pk = np.arange(nk)
    px = (K.x0 / K.m_share).values; pn = (K.n0 / K.m_share).values
    dk = arms.k.values
    doff = gamma_soc[soc_of_k[dk]] * arms.ldose_c.values
    rk = np.concatenate([pk, dk]); rx = np.concatenate([px, arms.x.values.astype(float)])
    rn = np.concatenate([pn, arms.n.values.astype(float)]); isd = np.concatenate([np.zeros(nk, bool), np.ones(len(dk), bool)])
    off = np.concatenate([np.zeros(nk), doff])
    return rk, rx, rn, isd, off


def _cll(eta, x, n):
    lam = np.exp(np.minimum(eta, 5.0))
    lp = np.where(x > 0, np.log(-np.expm1(-lam)), 0.0)
    return x * lp - (n - x) * lam


def profile_loglik(rk, rx, rn, isd, off, nk, grid=GRID, newton=6):
    """log L_k(z) = max_alpha sum_rows [x log p + (n - x) log(1 - p)], p = 1 - exp(-exp(alpha + z*drug + offset)),
    for every trial-drug-term k and every z on the grid. Returned normalised to a row maximum of 0 (float32)."""
    tx = np.bincount(rk, rx, nk); tn = np.bincount(rk, rn, nk)
    pbar = (tx + 0.5) / (tn + 1.0)
    alpha = np.log(-np.log1p(-pbar))
    out = np.empty((nk, len(grid)), np.float32)
    order = np.argsort(np.abs(grid))                                   # start at z = 0, walk outwards (warm starts)
    a_pos, a_neg = alpha.copy(), alpha.copy()
    for gi in order:
        z = grid[gi]
        a = (a_pos if z >= 0 else a_neg).copy()
        for _ in range(newton if z != 0 else 12):
            eta = np.clip(a[rk] + isd * (z + off), -30, 5.0)
            lam = np.exp(eta)
            em = np.expm1(lam)
            r1 = np.where(lam > 1e-10, lam / np.where(em > 0, em, 1.0), 1.0)
            g = rx * r1 - (rn - rx) * lam
            h = rx * (r1 - r1 * r1 * np.exp(lam)) - (rn - rx) * lam
            G = np.bincount(rk, g, nk); H = np.bincount(rk, h, nk)
            step = np.clip(-G / np.minimum(H, -1e-10), -3, 3)
            a = np.clip(a + step, -30, 4.0)
        ll = np.bincount(rk, _cll(np.clip(a[rk] + isd * (z + off), -30, 5.0), rx, rn), nk)
        out[:, gi] = ll
        if z >= 0:
            a_pos = a
        if z <= 0:
            a_neg = a
    out -= out.max(1, keepdims=True)
    return out


def dose_slopes(arms, K, soc_of_k, n_soc, iters=25):
    """Within-trial dose-response slope per system organ class, in log rate ratio per unit log relative dose.
    Among the drug arms of a trial-drug-term with two or more known, distinct doses, and conditional on their total
    count, arm counts are multinomial with probabilities proportional to n_a exp(gamma d_a): the trial's baseline and
    drug effect cancel exactly. Fitted per organ class by Newton's method, then shrunk to the pooled slope."""
    a = arms[["k", "x", "n", "ldose_c"]].copy()
    a["d_ok"] = a.groupby("k").ldose_c.transform(lambda v: v.max() - v.min()) > 0.1
    a = a[a.d_ok & (a.groupby("k").x.transform("sum") > 0)]
    s_ = soc_of_k[a.k.values]; k_ = a.k.values; x = a.x.values.astype(float); n = a.n.values.astype(float); d = a.ldose_c.values
    ks, kinv = np.unique(k_, return_inverse=True)
    m = np.bincount(kinv, x)
    soc_k = soc_of_k[ks]
    def fit(mask_k):
        g = 0.0
        for _ in range(iters):
            w = n * np.exp(g * d); W = np.bincount(kinv, w)
            md = np.bincount(kinv, w * d) / W; md2 = np.bincount(kinv, w * d * d) / W
            sc = np.sum((x * d)[mask_k[kinv]]) - np.sum((m * md)[mask_k])
            info = np.sum((m * (md2 - md ** 2))[mask_k])
            if info <= 0:
                return np.nan, np.inf
            g += np.clip(sc / info, -1, 1)
        return g, 1 / info
    if len(ks) == 0:                                   # no trial with two or more known, distinct doses
        return np.zeros(n_soc), dict(global_slope=0.0, global_se=float("inf"), tau2_between_soc=0.0, n_trial_terms=0)
    est = [fit(soc_k == s) if np.any(soc_k == s) else (np.nan, np.inf) for s in range(n_soc)]
    bs = np.array([e[0] for e in est]); vs = np.array([e[1] for e in est])
    bg, vg = fit(np.ones(len(ks), bool))
    ok = np.isfinite(bs) & np.isfinite(vs)
    q = np.sum((bs[ok] - bg) ** 2 / vs[ok]); c = np.sum(1 / vs[ok]) - np.sum(1 / vs[ok] ** 2) / np.sum(1 / vs[ok])
    t2 = max(0.0, (q - (ok.sum() - 1)) / c) if c > 0 else 0.0
    B = np.where(ok, t2 / (t2 + vs), 0.0)
    gam = np.where(ok, bg + B * (np.nan_to_num(bs) - bg), bg)
    return gam, dict(global_slope=float(bg), global_se=float(np.sqrt(vg)), tau2_between_soc=float(t2), n_trial_terms=int(len(ks)))


def _kernels(tau, grid=GRID):
    """Gaussian smoothing kernels K[i, j] = phi(z_i - v_j; tau), normalised over j for every z_i (mass-conserving on the
    grid: without this, records with flat likelihoods lose mass off the grid edges as tau grows, which biases tau to 0)."""
    d = grid[:, None] - grid[None, :]                    # z_i - v_j
    k0 = np.exp(-0.5 * (d / tau) ** 2)
    k0 /= k0.sum(1, keepdims=True)
    return k0.astype(np.float32), (d * k0).astype(np.float32), ((d ** 2) * k0).astype(np.float32)


def _shift_idx(sh, G):
    return np.clip(np.arange(G)[None, :] + sh[:, None], 0, G - 1)


def cell_features(con, cells):
    """Drug, side effect, ATC level-3 and level-1 class, organ system, class x organ system, class x side effect and
    primary molecular target x organ system, for the structured prior mean."""
    atc = {r[0]: sorted(r[1] or []) for r in con.execute("SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
    mech = con.execute("""SELECT coalesce(parent_chembl_id, chembl_id) AS d, target_chembl_id || '|' || coalesce(action_type,'') AS m
                          FROM ref_chembl_mechanism WHERE target_chembl_id IS NOT NULL""").df()
    mech = mech.sort_values(["d", "m"]).drop_duplicates("d").set_index("d").m.to_dict()
    soc = con.execute("""SELECT lower(trim(regexp_replace(term,'\\s+',' ','g'))) t, organ_system s, count(*) n
                         FROM v_ctgov_ae WHERE kind='other' GROUP BY 1,2""").df()
    soc["t"] = soc.t.map({t: af2.us(t) for t in soc.t.unique()})
    soc = soc.groupby(["t", "s"], as_index=False).n.sum().sort_values("n", ascending=False).drop_duplicates("t")
    smap = dict(zip(soc.t, soc.s.str.lower()))
    c = cells.copy()
    c["atc3"] = [atc[d][0][:4] if atc.get(d) else None for d in c.drug]
    c["atc1"] = [a[0] if isinstance(a, str) else None for a in c.atc3]
    c["soc"] = c.term.map(smap).fillna("unknown")
    c["mech"] = c.drug.map(mech)
    c["atc1_soc"] = [f"{a}|{s}" if a else None for a, s in zip(c.atc1, c.soc)]
    c["atc3_term"] = [f"{a}|{t}" if a else None for a, t in zip(c.atc3, c.term)]
    c["mech_soc"] = [f"{m}|{s}" if isinstance(m, str) else None for m, s in zip(c.mech, c.soc)]
    return c


FEATS = ("drug", "term", "atc3", "atc1", "soc", "atc1_soc", "atc3_term", "mech_soc")


def prior_mean(cf, y, v, var_g, lam=None, lams=(3, 10, 30, 100, 300, 1000), folds=5, seed=20260927, feats=FEATS, multi=()):
    """Cross-fitted weighted ridge prediction of each cell's effect from its features (no cell informs its own prior).
    y, v: likelihood-based mean and variance of each cell's log hazard ratio; var_g: spread of the prior around it."""
    X = af2._design(cf, feats) if not multi else design_multi(cf, feats, multi)
    w = 1 / (v + var_g)
    fold = np.random.default_rng(seed).integers(0, folds, len(y))
    def xfit(l):
        mu = np.empty(len(y))
        for f in range(folds):
            tr, te = fold != f, fold == f
            ym, b = af2._ridge(X[tr], y[tr], w[tr], l)
            mu[te] = ym + X[te] @ b
        return mu
    info = {}
    if lam is None:
        best = None
        inf_ = v < 1.0                                             # score on informative cells only
        for l in lams:
            mu = xfit(l)
            pll = float(np.sum(-0.5 * np.log(2 * np.pi * (v[inf_] + var_g)) - 0.5 * (y[inf_] - mu[inf_]) ** 2 / (v[inf_] + var_g)))
            info[l] = pll
            if best is None or pll > best[0]:
                best = (pll, l, mu)
        return best[2], best[1], info
    return xfit(lam), lam, info


def em_fit(LL, X, soc_k, cell_k, cf, n_soc, *, outer=3, inner=6, omega=0.25, tau0=0.3, use_cov=True,
           gaussian_prior=False, verbose=True, lam=None, tau_steps=(-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5),
           tau_prior_sd=0.5, feats=FEATS, multi=(), soc_prior=0.0):
    """Empirical-Bayes fit on the log-HR grid.
    Per iteration: heterogeneity tau_s of each organ system by direct maximisation of the within-cell (flat-prior,
    REML-like) marginal likelihood, penalised towards the pooled value (log-normal, sd `tau_prior_sd`), over a
    multiplicative step set; covariate effects by one Fisher-scoring step; the nonparametric prior shape by its EM
    update; every `inner` iterations the structured prior mean is refitted (cross-fitted ridge)."""
    G = len(GRID); nk, P = X.shape; C = len(cf)
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    beta = np.zeros((n_soc, P)); beta_g = np.zeros(P)
    tau = np.full(n_soc, tau0)
    gpri = np.exp(-0.5 * GRID ** 2); gpri /= gpri.sum()
    m = np.zeros(C)
    by_soc = [np.where(soc_k == s)[0] for s in range(n_soc)]
    cells_of = [np.unique(cell_k[ks]) for ks in by_soc]
    loc = np.empty(C, np.int64)
    for cs in cells_of:
        loc[cs] = np.arange(len(cs))
    hist = []
    gsoc = None
    ll_cell = np.zeros((C, G))
    for it_o in range(outer):
        for it in range(inner):
            shm = np.rint(m / DZ).astype(np.int64)
            pidx = np.arange(G)[None, :] - shm[:, None]
            valid = (pidx >= 0) & (pidx < G)
            if soc_prior and gsoc is not None:
                lgs = np.log(np.maximum(gsoc, 1e-300))
                lp = np.where(valid, lgs[cf.s.values[:, None], np.clip(pidx, 0, G - 1)], -np.inf)
            else:
                lg = np.log(np.maximum(gpri, 1e-300))
                lp = np.where(valid, lg[np.clip(pidx, 0, G - 1)], -np.inf)
            store = {}
            # hierarchical penalty on log tau_s towards the geometric mean over organ systems, weighted by the number
            # of multi-trial cells (which alone inform tau); stabilises systems with few, sparse multi-trial cells
            wts = np.array([np.sum(np.bincount(loc[cell_k[ks]], minlength=len(cells_of[s])) >= 2) if len(ks) else 0
                            for s, ks in enumerate(by_soc)], float)
            log_ref = np.sum(wts * np.log(tau)) / max(wts.sum(), 1)
            for s, ks in enumerate(by_soc):
                if len(ks) == 0:
                    continue
                L = np.exp(LL[ks]); cs = cells_of[s]; lc = loc[cell_k[ks]]
                sh = np.rint((Xv[ks] @ beta[s]) / DZ).astype(np.int64) if use_cov else np.zeros(len(ks), np.int64)
                idx = _shift_idx(sh, G)
                best = None
                for st in tau_steps:
                    t = max(tau[s] * np.exp(st), DZ)
                    k0, _, _ = _kernels(t)
                    A = np.log(np.maximum(np.take_along_axis(L @ k0, idx, 1), 1e-30))
                    llc = np.zeros((len(cs), G)); _add_rows(llc, lc, A)
                    # heterogeneity is identified by within-cell dispersion only: integrate each cell's effect under
                    # a flat prior (single-trial cells then contribute a constant), so that the flexible prior shape
                    # cannot absorb between-trial variation
                    mx = llc.max(1, keepdims=True)
                    marg = float(np.sum(np.log(np.exp(llc - mx).sum(1)) + mx[:, 0]))
                    if tau_prior_sd:
                        marg -= (np.log(t) - log_ref) ** 2 / (2 * tau_prior_sd ** 2)
                    if best is None or marg > best[0]:
                        best = (marg, t, llc)
                tau[s] = best[1]; ll_cell[cs] = best[2]
                _, k1, k2 = _kernels(tau[s]); k0, _, _ = _kernels(tau[s])
                store[s] = (idx, L @ k0, L @ k1, L @ k2)
            lpost = lp + ll_cell
            mx = lpost.max(1, keepdims=True)
            post = np.exp(lpost - mx); zs = post.sum(1, keepdims=True); post /= zs
            marg = float(np.sum(np.log(zs[:, 0]) + mx[:, 0]))
            E1 = np.zeros(nk); E2 = np.zeros(nk)
            for s, (idx, M0, M1, M2) in store.items():
                ks = by_soc[s]
                M0g = np.maximum(np.take_along_axis(M0, idx, 1), 1e-30)
                pc = post[cell_k[ks]]
                E1[ks] = (pc * np.take_along_axis(M1, idx, 1) / M0g).sum(1)
                E2[ks] = (pc * np.take_along_axis(M2, idx, 1) / M0g).sum(1)
            # prior shape (global; optionally organ-system-specific, shrunk to the global shape with pseudo-count soc_prior)
            gnew = np.zeros(G); np.add.at(gnew, pidx[valid], post[valid]); gnew /= gnew.sum()
            if gaussian_prior:
                mu_ = np.sum(gnew * GRID); sd_ = np.sqrt(np.sum(gnew * (GRID - mu_) ** 2))
                gnew = np.exp(-0.5 * ((GRID - mu_) / sd_) ** 2); gnew /= gnew.sum()
            gpri = gnew
            if soc_prior:
                socc = cf.s.values
                gs = np.zeros((n_soc, G))
                for s_ in range(n_soc):
                    r_ = socc == s_
                    if r_.any():
                        acc = np.zeros(G); np.add.at(acc, pidx[r_][valid[r_]], post[r_][valid[r_]])
                        gs[s_] = (acc + soc_prior * gnew) / (r_.sum() + soc_prior)
                    else:
                        gs[s_] = gnew
                gsoc = gs
            # covariate effects: one Fisher-scoring step per organ system, ridge towards the pooled effect
            if use_cov:
                Vk = np.maximum(E2 - E1 ** 2, 0.0)
                for s, ks in enumerate(by_soc):
                    if len(ks) == 0:
                        continue
                    t2 = tau[s] ** 2
                    w = np.clip(1 - Vk[ks] / t2, 1e-4, 1) / t2
                    Xs = Xv[ks]
                    A_ = (Xs * w[:, None]).T @ Xs + np.eye(P) / omega ** 2
                    beta[s] = beta[s] + np.linalg.solve(A_, Xs.T @ (E1[ks] / t2) - (beta[s] - beta_g) / omega ** 2)
                cnt = np.array([len(k) for k in by_soc], float)
                beta_g = (beta * cnt[:, None]).sum(0) / cnt.sum()
            hist.append(dict(outer=it_o, inner=it, marginal_loglik=marg, tau_median=float(np.median(tau[tau > 0]))))
            if verbose:
                print(f"  iter {it_o}.{it}: marginal loglik {marg:,.1f}; median tau {np.median(tau):.3f}", flush=True)
        # structured prior mean, from each cell's likelihood summary
        lik = np.exp(ll_cell - ll_cell.max(1, keepdims=True)); lik /= lik.sum(1, keepdims=True)
        ym = (lik * GRID).sum(1); yv = np.maximum((lik * GRID ** 2).sum(1) - ym ** 2, 1e-4)
        g_mu = np.sum(gpri * GRID); var_g = float(np.sum(gpri * (GRID - g_mu) ** 2))
        if it_o < outer - 1:
            m_new, lam, lam_info = prior_mean(cf, ym, yv, var_g, lam=lam, feats=feats, multi=multi)
            m = m_new + g_mu
            gidx = np.clip(np.rint((GRID - g_mu) / DZ).astype(int) + np.argmin(np.abs(GRID)), 0, G - 1)
            gpri = np.bincount(gidx, gpri, G); gpri /= gpri.sum()
            if verbose:
                print(f"  outer {it_o}: ridge lambda {lam}; prior spread sd {np.sqrt(var_g):.3f}", flush=True)
    return dict(beta=beta, beta_g=beta_g, tau=tau, gprior=gpri, gsoc=gsoc, m=m, post=post.astype(np.float32), lam=lam,
                history=hist, lik_mean=ym, lik_var=yv, marginal_loglik=marg, feats=feats, multi=multi)


def _add_rows(acc, rows, vals):
    o = np.argsort(rows, kind="stable")
    r = rows[o]; v = vals[o]
    starts = np.r_[0, np.nonzero(np.diff(r))[0] + 1]
    acc[r[starts]] += np.add.reduceat(v, starts, axis=0)


# ================================================================================================ outputs
def baseline_terms(K, X):
    """Term-level baseline cumulative hazard at 12 weeks: cloglog placebo risk of each trial-term, regressed on log
    follow-up within term (pooled slope), term means shrunk to the overall mean (moment estimate of the spread)."""
    p = (K.x0 + 0.5) / (K.n0 + 1)
    a = np.log(-np.log1p(-p))
    d = pd.DataFrame(dict(t=K.t.values, a=a.values, lwk=X.lwk.values, known=1 - X.wk_missing.values)).drop_duplicates()
    dk = d[d.known > 0]
    dm = dk.groupby("t")[["a", "lwk"]].transform("mean")
    slope = float(((dk.a - dm.a) * (dk.lwk - dm.lwk)).sum() / max(((dk.lwk - dm.lwk) ** 2).sum(), 1e-9))
    d["a12"] = d.a - slope * d.lwk * d.known
    g = d.groupby("t").a12.agg(["mean", "count", "var"])
    grand = float(d.a12.mean()); within = float(np.nanmedian(g["var"]))
    between = max(float(g["mean"].var() - within / g["count"].mean()), 0.01)
    B = between / (between + within / g["count"])
    return (grand + B * (g["mean"] - grand)).to_dict(), dict(duration_slope=slope, grand=grand)


def summarise(fit, cells, soc_of_cell, base12):
    """Per-cell posterior summaries of psi (log HR at reference conditions) and the derived fractions."""
    post = fit["post"].astype(np.float64); G = len(GRID)
    cdf = np.cumsum(post, 1)
    q = lambda p: GRID[np.minimum((cdf < p).sum(1), G - 1)]
    mean = (post * GRID).sum(1); sd = np.sqrt(np.maximum((post * GRID ** 2).sum(1) - mean ** 2, 0))
    afh = 1 - np.exp(-GRID)
    out = cells.copy()
    out["psi_mean"], out["psi_sd"], out["psi_lo"], out["psi_med"], out["psi_hi"] = mean, sd, q(.025), q(.5), q(.975)
    out["af_hz"] = (post * afh).sum(1)
    out["af_hz_lo"], out["af_hz_hi"] = 1 - np.exp(-out.psi_lo), 1 - np.exp(-out.psi_hi)
    out["p_af_lt_half"] = post[:, GRID < np.log(2)].sum(1)
    out["p_no_excess"] = post[:, GRID <= 0].sum(1)
    # risk-scale excess fraction over 12 weeks at the term's typical placebo hazard
    a12 = out.term.map(base12).fillna(np.nanmedian(list(base12.values()))).values
    p0 = -np.expm1(-np.exp(a12))[:, None]
    p1 = -np.expm1(-np.exp(a12[:, None] + GRID[None, :]))
    afr = 1 - p0 / p1
    out["p0_12wk"] = p0[:, 0]
    out["af_risk12"] = (post * afr).sum(1)
    # predictive for a new reference trial: psi + N(0, tau_s^2)
    tau = fit["tau"][soc_of_cell]
    pv_sd = np.sqrt(sd ** 2 + tau ** 2)
    out["pred_lo"], out["pred_hi"] = 1 - np.exp(-(mean - 1.96 * pv_sd)), 1 - np.exp(-(mean + 1.96 * pv_sd))
    out["tau_soc"] = tau
    out["prior_mean"] = fit["m"]
    out["lik_mean"], out["lik_sd"] = fit["lik_mean"], np.sqrt(fit["lik_var"])
    return out


def predictive_grid(post_c, shift, tau):
    """Predictive density of a new trial's log HR on the grid: psi-posterior shifted by x'beta, convolved with N(0, tau^2)."""
    G = len(GRID)
    sh = np.rint(shift / DZ).astype(np.int64)
    idx = np.clip(np.arange(G)[None, :] - sh[:, None], 0, G - 1)
    p = np.take_along_axis(post_c, idx, 1)
    out = np.empty_like(p)
    for t in np.unique(tau):
        r = tau == t
        k0, _, _ = _kernels(max(t, DZ))
        out[r] = p[r] @ k0.T
    out /= out.sum(1, keepdims=True)
    return out


def score_records(q, x, n, p0, arm_k, arm_off, scale, nrec):
    """Log predictive probability of the drug-arm counts of each record given its placebo risk p0 (plug-in) and a
    predictive density q (nrec x G) of the record's log effect. scale 'rr': p = p0 e^z; 'hz': p = 1 - (1-p0)^exp(z+off)."""
    from scipy.special import gammaln
    ll = np.zeros((nrec, len(GRID)))
    z = GRID[None, :]
    pa = p0[arm_k][:, None]
    if scale == "rr":
        p = np.clip(pa * np.exp(z), 1e-12, 1 - 1e-9)
    else:
        p = np.clip(-np.expm1(np.log1p(-pa) * np.exp(z + arm_off[:, None])), 1e-12, 1 - 1e-9)
    lb = (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1))[:, None] + x[:, None] * np.log(p) + (n - x)[:, None] * np.log1p(-p)
    np.add.at(ll, arm_k, lb)
    mx = ll.max(1, keepdims=True)
    return np.log(np.maximum((q * np.exp(ll - mx)).sum(1), 1e-300)) + mx[:, 0]


def pit_records(q, x, n, p0, arm_k, arm_off, scale, nrec, rng):
    """Randomised PIT of the pooled drug-arm count (arms pooled at their mean risk) under the predictive q."""
    from scipy import stats
    xs = np.bincount(arm_k, x, nrec); ns = np.bincount(arm_k, n, nrec)
    z = GRID[None, :]
    pa = p0[:, None]
    offm = np.bincount(arm_k, arm_off * n, nrec) / np.maximum(ns, 1)
    if scale == "rr":
        p = np.clip(pa * np.exp(z), 1e-12, 1 - 1e-9)
    else:
        p = np.clip(-np.expm1(np.log1p(-pa) * np.exp(z + offm[:, None])), 1e-12, 1 - 1e-9)
    F1 = (q * stats.binom.cdf(xs[:, None], ns[:, None], p)).sum(1)
    F0 = (q * stats.binom.cdf(xs[:, None] - 1, ns[:, None], p)).sum(1)
    return F0 + rng.random(nrec) * (F1 - F0)


# ================================================================================================ pipeline helpers
def prepare(con, arms, K, trials=None, cf_all=None):
    """Subset to `trials` (None = all), re-index, and build everything the fit needs."""
    if trials is not None:
        K = K[K.nct.isin(trials)].copy()
        arms = arms[arms.k.isin(K.k)].copy()
    K = K.copy(); arms = arms.copy()
    remap = dict(zip(K.k.values, range(len(K))))
    K["k"] = np.arange(len(K)); arms["k"] = arms.k.map(remap).values
    cells = K[["drug", "t"]].drop_duplicates().rename(columns={"t": "term"}).reset_index(drop=True)
    if cf_all is None:
        cf = cell_features(con, cells)
    else:
        cf = cells.merge(cf_all, on=["drug", "term"], how="left")
    cidx = pd.Series(np.arange(len(cf)), index=pd.MultiIndex.from_frame(cf[["drug", "term"]]))
    K["cell"] = cidx.reindex(pd.MultiIndex.from_frame(K[["drug", "t"]])).values
    socs = sorted(cf.soc.unique()); sidx = {s: i for i, s in enumerate(socs)}
    cf["s"] = cf.soc.map(sidx).values
    soc_k = cf.s.values[K.cell.values]
    X = design_covariates(K)
    gam, ginfo = dose_slopes(arms, K, soc_k, len(socs))
    rk, rx, rn, isd, off = rows_for(arms, K, gam, soc_k)
    LL = profile_loglik(rk, rx, rn, isd, off, len(K))
    return dict(K=K, arms=arms, cf=cf, socs=socs, soc_k=soc_k, X=X, gamma=gam, gamma_info=ginfo, LL=LL)


def run_fit(P, fit_fn=None, **kw):
    fit = (fit_fn or em_fit)(P["LL"], P["X"], P["soc_k"], P["K"].cell.values, P["cf"], len(P["socs"]), **kw)
    base12, binfo = baseline_terms(P["K"], P["X"])
    S = summarise(fit, P["cf"], P["cf"].s.values, base12)
    S["n_trials"] = P["K"].groupby("cell").nct.nunique().reindex(np.arange(len(S))).values
    return fit, S, base12, binfo


def af1_table(K, arms):
    """AF 1.0 on the same records: Mantel-Haenszel over trials (drug arms pooled; shared placebo split), cells with
    >= 2 trials and >= 20 events; DerSimonian-Laird random effects on per-trial log RR (0.5 correction for zeros)."""
    a = arms.groupby("k").agg(xd=("x", "sum"), nd=("n", "sum"))
    d = K[["k", "cell", "nct", "x0", "n0", "m_share"]].join(a, on="k")
    d["x0s"], d["n0s"] = d.x0 / d.m_share, d.n0 / d.m_share
    N = d.nd + d.n0s; T = d.xd + d.x0s
    d["R"] = d.xd * d.n0s / N; d["S"] = d.x0s * d.nd / N; d["V"] = (d.nd * d.n0s * T - d.xd * d.x0s * N) / N ** 2
    zc = (d.xd == 0) | (d.x0s == 0)
    xa, xb = d.xd + 0.5 * zc, d.x0s + 0.5 * zc; na, nb = d.nd + 1.0 * zc, d.n0s + 1.0 * zc
    d["lr"] = np.log((xa / na) / (xb / nb)); d["vr"] = 1 / xa - 1 / na + 1 / xb - 1 / nb
    d = d[(d.xd + d.x0s) > 0]
    g = d.groupby("cell").agg(R=("R", "sum"), S=("S", "sum"), V=("V", "sum"), k=("nct", "nunique"), ev=("xd", "sum"), ev0=("x0s", "sum"))
    g = g[(g.k >= 2) & (g.ev + g.ev0 >= 20) & (g.R > 0) & (g.S > 0)].copy()
    g["theta"] = np.log(g.R / g.S); g["se"] = np.sqrt(g.V / (g.R * g.S))
    dd = d[d.cell.isin(g.index)].copy()
    dd["w"] = 1 / dd.vr
    s = dd.groupby("cell").agg(sw=("w", "sum"), sw2=("w", lambda w: (w ** 2).sum()), swy=("lr", lambda y: 0))
    swy = (dd.w * dd.lr).groupby(dd.cell).sum(); s["swy"] = swy
    mu = s.swy / s.sw
    dd["mu"] = dd.cell.map(mu)
    Q = (dd.w * (dd.lr - dd.mu) ** 2).groupby(dd.cell).sum()
    kk = dd.groupby("cell").size()
    t2 = np.maximum(0, (Q - (kk - 1)) / (s.sw - s.sw2 / s.sw))
    dd["wr"] = 1 / (dd.vr + dd.cell.map(t2))
    th_dl = (dd.wr * dd.lr).groupby(dd.cell).sum() / dd.wr.groupby(dd.cell).sum()
    se_dl = np.sqrt(1 / dd.wr.groupby(dd.cell).sum())
    g["tau2_dl"] = t2.reindex(g.index).fillna(0).values
    g["theta_dl"] = th_dl.reindex(g.index).values; g["se_dl"] = se_dl.reindex(g.index).values
    return g


def beta_nodes(x0, n0, M=20):
    """Equal-probability nodes of the Jeffreys posterior Beta(x0 + 0.5, n0 - x0 + 0.5) of each placebo risk (nrec x M)."""
    from scipy import stats
    u = (np.arange(M) + 0.5) / M
    return stats.beta.ppf(u[None, :], (x0 + 0.5)[:, None], (n0 - x0 + 0.5)[:, None])


def _arm_p(p0, z, off, scale):
    if scale == "rr":
        return np.clip(p0[:, None] * np.exp(z), 1e-12, 1 - 1e-9)
    return np.clip(-np.expm1(np.log1p(-p0[:, None]) * np.exp(z + off[:, None])), 1e-12, 1 - 1e-9)


def score_records_beta(q, x, n, p0n, arm_k, arm_off, scale, nrec):
    """As score_records, integrating the placebo risk over its posterior nodes p0n (nrec x M)."""
    from scipy.special import gammaln, logsumexp
    z = GRID[None, :]
    lc = (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1))[:, None]
    out = []
    for m in range(p0n.shape[1]):
        p = _arm_p(p0n[arm_k, m], z, arm_off, scale)
        lb = lc + x[:, None] * np.log(p) + (n - x)[:, None] * np.log1p(-p)
        ll = np.zeros((nrec, len(GRID))); _add_rows(ll, arm_k, lb)
        mx = ll.max(1, keepdims=True)
        out.append(np.log(np.maximum((q * np.exp(ll - mx)).sum(1), 1e-300)) + mx[:, 0])
    return logsumexp(np.array(out), axis=0) - np.log(p0n.shape[1])


def pit_records_beta(q, x, n, p0n, arm_k, arm_off, scale, nrec, rng):
    """As pit_records, integrating the placebo risk over its posterior nodes."""
    from scipy import stats
    xs = np.bincount(arm_k, x, nrec); ns = np.bincount(arm_k, n, nrec)
    offm = np.bincount(arm_k, arm_off * n, nrec) / np.maximum(ns, 1)
    z = GRID[None, :]
    F1 = np.zeros(nrec); F0 = np.zeros(nrec)
    for m in range(p0n.shape[1]):
        p = _arm_p(p0n[:, m], z, offm, scale)
        F1 += (q * stats.binom.cdf(xs[:, None], ns[:, None], p)).sum(1)
        F0 += (q * stats.binom.cdf(xs[:, None] - 1, ns[:, None], p)).sum(1)
    F1 /= p0n.shape[1]; F0 /= p0n.shape[1]
    return F0 + rng.random(nrec) * (F1 - F0)


def prior_for_new(cf_train, fit, cf_new, feats=None):
    """Structured prior mean for cells without training records: ridge fitted on all training cells' likelihood
    summaries (lambda from the fit), evaluated on the new cells' features; returns m (prior mean of psi)."""
    feats = feats or fit.get("feats", FEATS); multi = fit.get("multi", ())
    cols = list(feats) + list(multi)
    allc = pd.concat([cf_train[cols], cf_new[cols]], ignore_index=True)
    X = af2._design(allc, feats) if not multi else design_multi(allc, feats, multi)
    n = len(cf_train)
    g_mu = float(np.sum(fit["gprior"] * GRID)); var_g = float(np.sum(fit["gprior"] * (GRID - g_mu) ** 2))
    w = 1 / (fit["lik_var"] + var_g)
    lam = fit["lam"]
    if lam is None:                                     # fit ran a single outer iteration: select lambda now
        _, lam, _ = prior_mean(cf_train, fit["lik_mean"], fit["lik_var"], var_g, lam=None, feats=feats, multi=multi)
    ym, b = af2._ridge(X[:n], fit["lik_mean"], w, lam)
    return ym + X[n:] @ b + g_mu


# ================================================================================================ AF 3.1 candidates
_STOP = {"and", "or", "of", "the", "in", "to", "with", "nos", "not", "other", "unspecified", "disorder", "disorders",
         "site", "on", "due", "a", "an", "by", "for", "at"}


def term_tokens(term):
    toks = re.findall(r"[a-z]+", str(term).lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 3:
            continue
        out.append(t[:-1] if t.endswith("s") and len(t) > 4 else t)
    return sorted(set(out))


def cell_features_v31(con, cells):
    """AF 3.0 features plus: drug x organ system, all molecular targets (multi-hot) and side-effect text tokens (multi-hot),
    also crossed with drug class (ATC3 x token)."""
    c = cell_features(con, cells)
    mech = con.execute("""SELECT DISTINCT coalesce(parent_chembl_id, chembl_id) AS d, target_chembl_id || '|' || coalesce(action_type,'') AS m
                          FROM ref_chembl_mechanism WHERE target_chembl_id IS NOT NULL""").df()
    mm = mech.groupby("d").m.apply(lambda s: sorted(set(s))).to_dict()
    c["drug_soc"] = c.drug + "|" + c.soc
    c["mechs"] = [mm.get(d, []) for d in c.drug]
    tk = {t: term_tokens(t) for t in c.term.unique()}
    c["tokens"] = c.term.map(tk)
    c["atc3_tokens"] = [[f"{a}|{t}" for t in ts] if isinstance(a, str) else [] for a, ts in zip(c.atc3, c.tokens)]
    return c


def design_multi(cf, single, multi):
    """Sparse design: one-hot for `single` features, multi-hot (each row scaled to unit norm per feature) for `multi`."""
    from scipy import sparse
    rows, cols, vals, idx = [], [], [], {}
    for f in single:
        v = cf[f].astype(str).values
        for i, x in enumerate(v):
            if x in ("nan", "None", ""):
                continue
            j = idx.setdefault((f, x), len(idx)); rows.append(i); cols.append(j); vals.append(1.0)
    for f in multi:
        for i, lst in enumerate(cf[f].values):
            lst = list(lst) if isinstance(lst, (list, tuple, np.ndarray)) else []
            if not lst:
                continue
            w = 1 / np.sqrt(len(lst))
            for x in lst:
                j = idx.setdefault((f, x), len(idx)); rows.append(i); cols.append(j); vals.append(w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(cf), len(idx)))


# ================================================================================================ AF 3.2
TAU_GRID = np.array([0.03, 0.08, 0.15, 0.25, 0.4, 0.65, 1.0])


def em_fit_mix(LL, X, soc_k, cell_k, cf, n_soc, *, outer=2, inner=6, omega=0.25, use_cov=True, verbose=True, lam=None,
               feats=FEATS, multi=(), tg=TAU_GRID, pi_pseudo=0.5, pi_init_center=0.25, pi_init_sd=0.7, spike=False,
               spike_init=0.3):
    """AF 3.2: as em_fit, but between-trial heterogeneity is specific to each drug x event cell and integrated over a
    grid of tau values; the distribution of tau across cells is estimated nonparametrically within each organ system
    (EM over the grid, informed by multi-trial cells, with a Dirichlet pseudo-count towards uniform)."""
    G = len(GRID); nk, P = X.shape; C = len(cf); T = len(tg)
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    beta = np.zeros((n_soc, P)); beta_g = np.zeros(P)
    p0_ = np.exp(-0.5 * ((np.log(tg) - np.log(pi_init_center)) / pi_init_sd) ** 2); p0_ /= p0_.sum()
    pi = np.tile(p0_, (n_soc, 1))
    gpri = np.exp(-0.5 * GRID ** 2); gpri /= gpri.sum()
    m = np.zeros(C)
    by_soc = [np.where(soc_k == s)[0] for s in range(n_soc)]
    cells_of = [np.unique(cell_k[ks]) for ks in by_soc]
    loc = np.empty(C, np.int64)
    for cs in cells_of:
        loc[cs] = np.arange(len(cs))
    nrec_c = np.bincount(cell_k, minlength=C)
    kern = [_kernels(t) for t in tg]
    wt = np.zeros((C, T)); hist = []
    ll_cell = np.zeros((C, G)); post = None
    z0 = int(np.argmin(np.abs(GRID)))
    pi0 = np.full(n_soc, spike_init if spike else 0.0)          # AF 3.3: organ-system probability of no drug effect
    soc_c = cf.s.values if "s" in cf else np.zeros(C, int)
    r0 = np.zeros(C)
    for it_o in range(outer):
        for it in range(inner):
            shm = np.rint(m / DZ).astype(np.int64)
            pidx = np.arange(G)[None, :] - shm[:, None]
            valid = (pidx >= 0) & (pidx < G)
            slab = np.where(valid, gpri[np.clip(pidx, 0, G - 1)], 0.0)
            if spike:
                pr = (1 - pi0[soc_c])[:, None] * slab
                pr[:, z0] += pi0[soc_c]
                lp = np.log(np.maximum(pr, 1e-300))
            else:
                lp = np.where(valid, np.log(np.maximum(gpri, 1e-300))[np.clip(pidx, 0, G - 1)], -np.inf)
            E1 = np.zeros(nk); E2 = np.zeros(nk); ET2 = np.zeros(nk)
            post = np.zeros((C, G))
            marg = 0.0
            for s, ks in enumerate(by_soc):
                if len(ks) == 0:
                    continue
                L = np.exp(LL[ks]); cs = cells_of[s]; lc = loc[cell_k[ks]]
                sh = np.rint((Xv[ks] @ beta[s]) / DZ).astype(np.int64) if use_cov else np.zeros(len(ks), np.int64)
                idx = _shift_idx(sh, G)
                llg = np.zeros((T, len(cs), G))
                for g_, (k0, _, _) in enumerate(kern):
                    A = np.log(np.maximum(np.take_along_axis(L @ k0, idx, 1), 1e-30))
                    _add_rows(llg[g_], lc, A)
                # joint over (tau, psi) per cell
                J_ = np.log(np.maximum(pi[s], 1e-300))[:, None, None] + lp[cs][None, :, :] + llg
                mx = J_.max(axis=(0, 2), keepdims=True)
                e = np.exp(J_ - mx)
                zc = e.sum(axis=(0, 2))
                w = e.sum(2) / zc[None, :]                         # T x cells: posterior over tau
                pc = e.sum(0) / zc[:, None]                        # cells x G: posterior over psi
                post[cs] = pc; wt[cs] = w.T
                marg += float(np.sum(np.log(zc) + mx[0, :, 0]))
                # marginal (over tau) cell log-likelihood, for the prior-mean regression
                mxl = llg.max(axis=(0, 2), keepdims=True)
                ll_cell[cs] = np.log(np.maximum((pi[s][:, None, None] * np.exp(llg - mxl)).sum(0), 1e-300)) + mxl[0]
                # record moments: mixture over tau of E[eps | psi, tau] under the joint posterior
                pcg = e / zc[None, :, None]                         # T x cells x G joint posterior
                for g_, (k0, k1, k2) in enumerate(kern):
                    M0 = np.maximum(np.take_along_axis(L @ k0, idx, 1), 1e-30)
                    R1 = np.take_along_axis(L @ k1, idx, 1) / M0; R2 = np.take_along_axis(L @ k2, idx, 1) / M0
                    wj = pcg[g_][lc]
                    E1[ks] += (wj * R1).sum(1); E2[ks] += (wj * R2).sum(1); ET2[ks] += wj.sum(1) * tg[g_] ** 2
                # tau distribution in this organ system: multi-trial cells inform it
                inf_ = nrec_c[cs] >= 2
                if inf_.any():
                    pi[s] = (w[:, inf_].sum(1) + pi_pseudo * p0_) / (inf_.sum() + pi_pseudo)
            if spike:
                # responsibility of the no-effect spike for each cell's posterior mass at psi = 0
                share = pi0[soc_c] / np.maximum(pi0[soc_c] + (1 - pi0[soc_c]) * slab[:, z0], 1e-300)
                r0 = post[:, z0] * share
                slabpost = post.copy(); slabpost[:, z0] -= r0
                pi0 = np.clip(np.bincount(soc_c, r0, n_soc) / np.maximum(np.bincount(soc_c, minlength=n_soc), 1), 1e-3, .95)
                gnew = np.zeros(G); np.add.at(gnew, pidx[valid], slabpost[valid]); gpri = gnew / gnew.sum()
            else:
                gnew = np.zeros(G); np.add.at(gnew, pidx[valid], post[valid]); gpri = gnew / gnew.sum()
            if use_cov:
                Vk = np.maximum(E2 - E1 ** 2, 0.0)
                for s, ks in enumerate(by_soc):
                    if len(ks) == 0:
                        continue
                    t2 = np.maximum(ET2[ks], DZ ** 2)
                    wv = np.clip(1 - Vk[ks] / t2, 1e-4, 1) / t2
                    Xs = Xv[ks]
                    A_ = (Xs * wv[:, None]).T @ Xs + np.eye(P) / omega ** 2
                    beta[s] = beta[s] + np.linalg.solve(A_, Xs.T @ (E1[ks] / t2) - (beta[s] - beta_g) / omega ** 2)
                cnt = np.array([len(k) for k in by_soc], float)
                beta_g = (beta * cnt[:, None]).sum(0) / cnt.sum()
            hist.append(dict(outer=it_o, inner=it, marginal_loglik=marg))
            if verbose:
                print(f"  iter {it_o}.{it}: marginal loglik {marg:,.1f}; median E[tau] {np.median(np.sqrt((wt * tg ** 2).sum(1))):.3f}", flush=True)
        lik = np.exp(ll_cell - ll_cell.max(1, keepdims=True)); lik /= lik.sum(1, keepdims=True)
        ym = (lik * GRID).sum(1); yv = np.maximum((lik * GRID ** 2).sum(1) - ym ** 2, 1e-4)
        g_mu = np.sum(gpri * GRID); var_g = float(np.sum(gpri * (GRID - g_mu) ** 2))
        if it_o < outer - 1:
            m_new, lam, _ = prior_mean(cf, ym, yv, var_g, lam=lam, feats=feats, multi=multi)
            m = m_new + g_mu
            gidx = np.clip(np.rint((GRID - g_mu) / DZ).astype(int) + np.argmin(np.abs(GRID)), 0, G - 1)
            gpri = np.bincount(gidx, gpri, G); gpri /= gpri.sum()
    tau_soc = np.sqrt((pi * tg ** 2).sum(1))
    return dict(beta=beta, beta_g=beta_g, tau=tau_soc, tau_grid=tg, tau_w=wt.astype(np.float32), tau_pi=pi, gprior=gpri, gsoc=None,
                pi0=pi0, p_null=r0.astype(np.float32),
                m=m, post=post.astype(np.float32), lam=lam, history=hist, lik_mean=ym, lik_var=yv, marginal_loglik=marg,
                feats=feats, multi=multi)


def predictive_grid_mix(post_c, shift, tau_w, tg=TAU_GRID):
    """Predictive of a new trial's log HR: cell posterior shifted by x'beta, convolved with the cell's tau mixture."""
    G = len(GRID)
    sh = np.rint(shift / DZ).astype(np.int64)
    idx = np.clip(np.arange(G)[None, :] - sh[:, None], 0, G - 1)
    p = np.take_along_axis(post_c, idx, 1)
    out = np.zeros_like(p)
    for g_, t in enumerate(tg):
        k0, _, _ = _kernels(t)
        out += tau_w[:, [g_]] * (p @ k0.T)
    out /= out.sum(1, keepdims=True)
    return out
