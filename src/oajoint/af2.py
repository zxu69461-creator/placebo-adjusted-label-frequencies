"""AF 2.0 (gate af2_design): selection-corrected conditional likelihood (stage 1) and structured empirical Bayes
(stage 2) for randomized attributable fractions.

Stage 1. For one drug and one MedDRA term, a trial contributes its drug arms D (doses pooled under one rate ratio
psi = exp(theta)) and placebo arms P. Given the total m of events in D and P, arm counts are multinomial with
p_i proportional to n_i * psi_i. This conditional likelihood removes the per-trial baseline risk, and its score
equation is unbiased for the risk ratio even for binomial data. Registries list a non-serious event only if it
reaches the trial's frequency threshold in some arm; for threshold-enforcing trials in which no other arm reaches it,
the likelihood is divided by P(some arm in D or P reaches its threshold | m, theta), which removes the selection bias
of analysing listed events only.

Stage 2. theta_hat is shrunk towards a cross-fitted prior mean from a weighted ridge regression on drug, term, drug
class and organ system.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats
from scipy.sparse.linalg import lsqr

from . import provenance as pv

PENALTY_SD = 4.0
BOUND = 8.0
US_RULES = [(r"aemia", "emia"), (r"aem", "em"), (r"oea", "ea"), (r"oedema", "edema"), (r"oesoph", "esoph"),
            (r"paed", "ped"), (r"foet", "fet"), (r"faec", "fec"), (r"oestr", "estr"), (r"our\b", "or"), (r"sulph", "sulf")]


def us(t):
    for a, b in US_RULES:
        t = re.sub(a, b, t)
    return t


# ----------------------------------------------------------------------------------------------- data assembly
def load_arms(con, kind="other"):
    """Contrasts, arm-level counts and thresholds for every trial with a single-ingredient drug-placebo contrast.
    kind='other' (non-serious events, listed above the trial's frequency threshold; the default, as published) or
    'serious' (serious adverse events, reported in full: no threshold applies)."""
    sfx = "other" if kind == "other" else "serious"
    ct = con.execute(f"""SELECT contrast_id, nct, drug_ids[1] AS drug, eg_active, eg_placebo FROM atlas_contrast
                        WHERE single_ingredient AND n_act_{sfx} > 0 AND n_pl_{sfx} > 0""").df()
    ae = con.execute(f"""
        WITH tr AS (SELECT DISTINCT nct FROM atlas_contrast WHERE single_ingredient AND n_act_{sfx} > 0 AND n_pl_{sfx} > 0)
        SELECT a.nct, a.eg_id, lower(trim(regexp_replace(a.term,'\\s+',' ','g'))) AS t,
               max(a.num_affected) AS x, max(a.num_at_risk) AS n
        FROM v_ctgov_ae a JOIN tr USING (nct) WHERE a.kind = '{kind}' AND a.num_affected IS NOT NULL
        GROUP BY 1, 2, 3""").df()
    eg = con.execute(f"SELECT nct, eg_id, n_risk_{sfx} AS nr FROM ctgov_event_group").df()
    tau = dict(con.execute("SELECT nct, frequency_threshold FROM ctgov_trial").fetchall())
    if kind != "other":
        tau = {k: 0.0 for k in tau}
    ae["t"] = ae.t.map({t: us(t) for t in ae.t.unique()})
    ae = ae.groupby(["nct", "eg_id", "t"], as_index=False).agg(x=("x", "max"), n=("n", "max"))
    nmap = dict(zip(zip(eg.nct, eg.eg_id), eg.nr))
    ae["n"] = [n if n == n and n > 0 else nmap.get((a, b), np.nan) for a, b, n in zip(ae.nct, ae.eg_id, ae.n)]
    ae = ae[ae.n > 0]
    ae["tau"] = ae.nct.map(tau)
    ae["thr"] = np.where(ae.tau > 0, np.ceil(ae.tau * ae.n / 100 - 1e-9), 0).astype(int)
    ae["meets"] = (ae.tau > 0) & (ae.x >= ae.thr)
    # threshold-enforcing trials: tau > 0 and every listed term reaches the threshold in some arm
    tm = ae[ae.tau > 0].groupby(["nct", "t"]).meets.max()
    enforcing = set(tm.groupby("nct").min().loc[lambda s: s].index) if len(tm) else set()
    return ct, ae, enforcing


def build_trials(ct, ae, enforcing, keep_ncts=None):
    """Per (drug, term): list of trial records (x, n, is_drug, thr, truncate)."""
    if keep_ncts is not None:
        ct = ct[ct.nct.isin(keep_ncts)]
    arms_all = ae.groupby("nct").eg_id.unique().to_dict()
    recs = {}
    by = {(n, e, t): (x, nn, thr, me) for n, e, t, x, nn, thr, me in
          zip(ae.nct, ae.eg_id, ae.t, ae.x, ae.n, ae.thr, ae.meets)}
    terms_by_trial = ae.groupby("nct").t.unique().to_dict()
    for (nct, drug), g in ct.groupby(["nct", "drug"]):
        D = sorted(set(g.eg_active))
        P = sorted({p for lst in g.eg_placebo for p in lst})
        O = [e for e in arms_all.get(nct, []) if e not in D and e not in P]
        enf = nct in enforcing
        for t in terms_by_trial.get(nct, []):
            rows = [by.get((nct, e, t)) for e in D + P]
            if any(r is None for r in rows):
                continue
            x = np.array([r[0] for r in rows], float); n = np.array([r[1] for r in rows], float)
            thr = np.array([r[2] for r in rows], int)
            other_meets = any((by.get((nct, e, t)) or (0, 0, 0, False))[3] for e in O)
            recs.setdefault((drug, t), []).append(dict(
                nct=nct, x=x, n=n, is_drug=np.array([True] * len(D) + [False] * len(P)), thr=thr,
                truncate=bool(enf and not other_meets)))
    return recs


# ----------------------------------------------------------------------------------------------- stage 1
def p_below(m, p, thr):
    """P(every arm count <= thr_i - 1 | total m, multinomial probabilities p)."""
    cap = thr - 1
    if m > cap.sum():
        return 0.0
    f = np.zeros(m + 1); f[m] = 1.0          # distribution of remaining events
    rem = 1.0
    for i in range(len(p) - 1):
        q = min(max(p[i] / rem, 0.0), 1.0) if rem > 0 else 0.0
        g = np.zeros(m + 1)
        for r in np.nonzero(f > 0)[0]:
            k = np.arange(0, min(cap[i], r) + 1)
            g[r - k] += f[r] * stats.binom.pmf(k, r, q)
        f = g; rem -= p[i]
    return float(f[: cap[-1] + 1].sum())


def trial_ll(theta, r):
    psi = np.where(r["is_drug"], np.exp(theta), 1.0)
    w = r["n"] * psi
    p = w / w.sum()
    m = int(round(r["x"].sum()))
    ll = float((r["x"] * np.log(np.maximum(p, 1e-300))).sum())
    if r["truncate"] and m > 0:
        pb = p_below(m, p, r["thr"])
        ll -= np.log(max(1.0 - pb, 1e-300))
    return ll


def fit_cell(trials, min_trials=2, min_events=20):
    trials = [r for r in trials if r["x"].sum() > 0]
    k = len({r["nct"] for r in trials}); ev = float(sum(r["x"].sum() for r in trials))
    if k < min_trials or ev < min_events:
        return None
    obj = lambda th: -(sum(trial_ll(th, r) for r in trials) - th ** 2 / (2 * PENALTY_SD ** 2))
    res = optimize.minimize_scalar(obj, bounds=(-BOUND, BOUND), method="bounded", options={"xatol": 1e-5})
    th = float(res.x); h = 1e-3
    u = np.array([(trial_ll(th + h, r) - trial_ll(th - h, r)) / (2 * h) for r in trials])
    info = -(obj(th + h) - 2 * obj(th) + obj(th - h)) / h ** 2 * -1   # observed information of the penalised ll
    info = max(info, 1e-6)
    v_model = 1 / info
    v_sand = (u ** 2).sum() / info ** 2 * k / (k - 1)
    se = float(np.sqrt(max(v_model, v_sand)))
    return dict(theta=th, se=se, k=k, events=ev, n_trunc=int(sum(r["truncate"] for r in trials)),
                boundary=abs(th) > BOUND - 0.05)


def mh_cell(trials, min_trials=2, min_events=20):
    """AF 1.0 on the same arm-level records: Mantel-Haenszel RR, drug arms pooled vs placebo arms pooled."""
    trials = [r for r in trials if r["x"].sum() > 0]
    k = len({r["nct"] for r in trials}); ev = float(sum(r["x"].sum() for r in trials))
    if k < min_trials or ev < min_events:
        return None
    a = np.array([r["x"][r["is_drug"]].sum() for r in trials]); c = np.array([r["x"][~r["is_drug"]].sum() for r in trials])
    n1 = np.array([r["n"][r["is_drug"]].sum() for r in trials]); n0 = np.array([r["n"][~r["is_drug"]].sum() for r in trials])
    T = n1 + n0; R, S = (a * n0 / T).sum(), (c * n1 / T).sum()
    if R <= 0 or S <= 0:
        return None
    v = ((n1 * n0 * (a + c) - a * c * T) / T ** 2).sum() / (R * S)
    return dict(theta=float(np.log(R / S)), se=float(np.sqrt(v)) if v > 0 else np.nan, k=k, events=ev)


# ----------------------------------------------------------------------------------------------- stage 2
def _design(cells, feats):
    cols, idx = [], {}
    rows, cidx = [], []
    for j, f in enumerate(feats):
        vals = cells[f].astype(str).values
        for i, v in enumerate(vals):
            if v in ("nan", "None", ""):
                continue
            key = (f, v)
            if key not in idx:
                idx[key] = len(idx)
            rows.append(i); cidx.append(idx[key])
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cidx)), shape=(len(cells), len(idx)))
    return X


def _ridge(X, y, w, lam):
    sw = np.sqrt(w)
    Xa = sparse.diags(sw) @ X
    ym = np.average(y, weights=w)
    b = lsqr(Xa, sw * (y - ym), damp=np.sqrt(lam), atol=1e-8, btol=1e-8, iter_lim=5000)[0]
    return ym, b


def stage2(cells, feats=("drug", "term", "atc3", "atc1", "soc", "atc1_soc"), lams=(0.3, 1, 3, 10, 30, 100),
           folds=5, seed=20260927):
    """Cross-fitted structured EB. cells: DataFrame with theta, se and feature columns."""
    y, s2 = cells.theta.values, cells.se.values ** 2
    X = _design(cells, feats)
    rng = np.random.default_rng(seed); fold = rng.integers(0, folds, len(cells))
    tau2 = max(np.var(y) - np.mean(s2), 0.01)
    best = None
    for lam in lams:
        for _ in range(3):
            mu = np.empty(len(y))
            for f in range(folds):
                tr, te = fold != f, fold == f
                ym, b = _ridge(X[tr], y[tr], 1 / (s2[tr] + tau2), lam)
                mu[te] = ym + X[te] @ b
            tau2 = max(np.mean((y - mu) ** 2 - s2), 0.005)
        pll = stats.norm.logpdf(y, mu, np.sqrt(s2 + tau2)).sum()
        if best is None or pll > best[0]:
            best = (pll, lam, mu.copy(), tau2)
    pll, lam, mu, tau2 = best
    B = tau2 / (tau2 + s2)
    post = mu + B * (y - mu); pv_ = B * s2
    out = cells.copy()
    out["mu"], out["tau2"], out["shrink"] = mu, tau2, 1 - B
    out["theta2"], out["se2"] = post, np.sqrt(pv_)
    out["af2"] = 1 - np.exp(-post)
    out["af2_lo"] = 1 - np.exp(-(post - 1.96 * np.sqrt(pv_)))
    out["af2_hi"] = 1 - np.exp(-(post + 1.96 * np.sqrt(pv_)))
    return out, dict(lambda_=lam, tau2=float(tau2), cv_pll=float(pll))


def features(con, cells):
    atc = {r[0]: sorted(r[1] or []) for r in con.execute("SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
    soc = con.execute("""SELECT lower(trim(regexp_replace(term,'\\s+',' ','g'))) t, organ_system s, count(*) n
                         FROM v_ctgov_ae WHERE kind='other' GROUP BY 1,2""").df()
    soc["t"] = soc.t.map(us)
    soc = soc.groupby(["t", "s"], as_index=False).n.sum().sort_values("n", ascending=False).drop_duplicates("t")
    smap = dict(zip(soc.t, soc.s.str.lower()))
    c = cells.copy()
    c["atc3"] = [atc.get(d, [None])[0][:4] if atc.get(d) else None for d in c.drug]
    c["atc1"] = [a[0] if isinstance(a, str) else None for a in c.atc3]
    c["soc"] = c.term.map(smap)
    c["atc1_soc"] = [f"{a}|{s}" if a and s else None for a, s in zip(c.atc1, c.soc)]
    return c
