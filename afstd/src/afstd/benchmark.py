"""afstd.benchmark -- BENCH-2.1, VENDORED VERBATIM from oajoint/bench.py (sha256 bb77f048273b99e2).
Score any estimator: give each test record a predictive density of its log effect on core.GRID (scale 'rr' or 'hz'); the
placebo risk is integrated over the hierarchical background model; proper log score and PIT calibration."""
import numpy as np
import pandas as pd
from scipy import stats

from . import core as af3
from . import core_extra as _cx
for _n in ('beta_nodes', 'score_records_beta', 'pit_records_beta', '_arm_p'):
    setattr(af3, _n, getattr(_cx, _n))

BASE_COVS = ['lwk', 'wk_missing', 'nonsys', 'phase1', 'healthy', 'age10', 'age_missing', 'female_c', 'female_missing', 'year10']

def trial_records(K, arms):
    """Per trial-drug-term log RR (drug arms pooled, shared placebo split) with 0.5 correction for zero cells."""
    a = arms.groupby("k").agg(xd=("x", "sum"), nd=("n", "sum"))
    d = K[["k", "nct", "drug", "t", "x0", "n0", "m_share"]].join(a, on="k")
    d["x0s"], d["n0s"] = d.x0 / d.m_share, d.n0 / d.m_share
    z = (d.xd == 0) | (d.x0s == 0) | (d.xd >= d.nd) | (d.x0s >= d.n0s)       # zero events or zero non-events
    xa, xb = d.xd + .5 * z, d.x0s + .5 * z; na, nb = d.nd + 1. * z, d.n0s + 1. * z
    d["y"] = np.log((xa / na) / (xb / nb)); d["v"] = np.maximum(1 / xa - 1 / na + 1 / xb - 1 / nb, 1e-4)
    return d[(d.xd + d.x0s) > 0]


def _pm_tau2(y, v):
    """Paule-Mandel between-trial variance: tau^2 >= 0 solving Q(tau^2) = k - 1 (bisection; 0 if Q(0) <= k - 1)."""
    k = len(y)
    if k < 2:
        return 0.0
    def Q(t2):
        w = 1 / (v + t2); mu = np.sum(w * y) / np.sum(w)
        return np.sum(w * (y - mu) ** 2) - (k - 1)
    if Q(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while Q(hi) > 0 and hi < 1e3:
        hi *= 4
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Q(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def baselines(K, arms):
    """Per-cell predictive specifications of B1, B2, B3 and per-term B5, fitted on the given (training) records."""
    d = trial_records(K, arms)
    # B1 Mantel-Haenszel on the same pooled records
    N = d.nd + d.n0s; T = d.xd + d.x0s
    d["R"] = d.xd * d.n0s / N; d["S"] = d.x0s * d.nd / N; d["V"] = (d.nd * d.n0s * T - d.xd * d.x0s * N) / N ** 2
    g = d.groupby(["drug", "t"])
    c = g.agg(R=("R", "sum"), S=("S", "sum"), V=("V", "sum"), k=("nct", "nunique"), ev=("xd", "sum"), ev0=("x0s", "sum"))
    c["mh_ok"] = (c.k >= 2) & (c.ev + c.ev0 >= 20) & (c.R > 0) & (c.S > 0)
    c["mh_mu"] = np.log(c.R / c.S); c["mh_se"] = np.sqrt(c.V / (c.R * c.S))
    # B2 Paule-Mandel + t predictive, and B3 Bayesian normal-normal (tau grid with half-normal(0.5) prior)
    tg = np.linspace(0, 3, 61); lp_tau = stats.halfnorm.logpdf(tg, scale=.5); lp_tau[0] = np.log(stats.halfnorm.pdf(0, scale=.5))
    rows = []
    for (dr, t), q in g:
        y, v = q.y.values, q.v.values; k = len(y)
        t2 = _pm_tau2(y, v) if k >= 2 else 0.0
        w = 1 / (v + t2); mu = np.sum(w * y) / np.sum(w); se2 = 1 / np.sum(w)
        # Bayes-NN: for each tau, mu | tau ~ N(m, s2); marginal likelihood with flat prior on mu
        W = 1 / (v[None, :] + tg[:, None] ** 2)
        s2 = 1 / W.sum(1); m = (W * y[None, :]).sum(1) * s2
        ll = 0.5 * np.log(s2) + 0.5 * np.log(W).sum(1) - 0.5 * (W * (y[None, :] - m[:, None]) ** 2).sum(1)
        lw = ll + lp_tau; lw -= lw.max(); pw = np.exp(lw); pw /= pw.sum()
        rows.append((dr, t, k, mu, se2, t2, m, s2, pw))
    B = pd.DataFrame(rows, columns=["drug", "t", "k_rec", "pm_mu", "pm_se2", "pm_t2", "bn_m", "bn_s2", "bn_w"]).set_index(["drug", "t"])
    c = c.join(B)
    # B5 term-only: DerSimonian-Laird across every record of the term (all drugs)
    tt = []
    for t, q in d.groupby("t"):
        y, v = q.y.values, q.v.values
        w = 1 / v; mu0 = np.sum(w * y) / np.sum(w); Q = np.sum(w * (y - mu0) ** 2)
        t2 = max(0.0, (Q - (len(y) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if len(y) > 1 else 0.25
        w2 = 1 / (v + t2); tt.append((t, np.sum(w2 * y) / np.sum(w2), 1 / np.sum(w2), t2, len(y)))
    TT = pd.DataFrame(tt, columns=["t", "term_mu", "term_se2", "term_t2", "term_n"]).set_index("t")
    return c, TT, d


def _grid_normal(mu, var):
    v = np.maximum(var, 1e-6)[:, None]
    q = np.exp(-0.5 * (af3.GRID[None, :] - mu[:, None]) ** 2 / v)
    s = q.sum(1, keepdims=True); bad = s[:, 0] <= 0
    q = q / np.where(s > 0, s, 1)
    if bad.any():
        idx = np.clip(np.rint((mu[bad] - af3.GRID[0]) / af3.DZ).astype(int), 0, len(af3.GRID) - 1)
        q[np.where(bad)[0]] = 0; q[np.where(bad)[0], idx] = 1
    return q


def _grid_t(mu, scale2, df):
    q = stats.t.pdf((af3.GRID[None, :] - mu[:, None]) / np.sqrt(np.maximum(scale2, 1e-6))[:, None], df[:, None])
    s = q.sum(1, keepdims=True)
    return q / np.where(s > 0, s, 1)


def baseline_predictives(c, TT, keys, terms):
    """Grid predictive densities (nrec x G) of each baseline for test records with cell keys and terms; NaN rows where a
    baseline has no prediction."""
    G = len(af3.GRID); n = len(keys)
    ci = c.reindex(pd.MultiIndex.from_tuples(keys, names=["drug", "t"]))
    out = {}
    q = np.zeros((n, G)); q[:, np.argmin(np.abs(af3.GRID))] = 1; out["B0 null"] = q
    ok = ci.mh_ok.fillna(False).values.astype(bool)
    q = np.full((n, G), np.nan); q[ok] = _grid_normal(ci.mh_mu.values[ok], ci.mh_se.values[ok] ** 2); out["B1 MH"] = q
    has = ci.k.notna().values
    q = np.full((n, G), np.nan)
    k = ci.k.values
    m3 = has & (k >= 3)
    q[m3] = _grid_t(ci.pm_mu.values[m3], ci.pm_se2.values[m3] + ci.pm_t2.values[m3], k[m3] - 2)
    m2 = has & (k < 3)
    q[m2] = _grid_normal(ci.pm_mu.values[m2], ci.pm_se2.values[m2] + ci.pm_t2.values[m2])
    out["B2 PM-t"] = q
    q = np.full((n, G), np.nan)
    tg = np.linspace(0, 3, 61)
    for i in np.where(has)[0]:
        w, m, s2 = ci.bn_w.iloc[i], ci.bn_m.iloc[i], ci.bn_s2.iloc[i]
        dens = (w[:, None] * stats.norm.pdf(af3.GRID[None, :], m[:, None], np.sqrt(s2 + tg ** 2)[:, None])).sum(0)
        q[i] = dens / dens.sum()
    out["B3 Bayes-NN"] = q
    ti = TT.reindex(terms)
    okt = ti.term_mu.notna().values
    q = np.full((n, G), np.nan); q[okt] = _grid_normal(ti.term_mu.values[okt], ti.term_se2.values[okt] + ti.term_t2.values[okt])
    out["B5 term-only"] = q
    return out, ci


def score_all(preds, scales, Kt, arms_t, rng, M=20, p0n=None):
    """Integrated-placebo log score and PIT for each method's predictive (rows with NaN are skipped). p0n: placebo-risk
    nodes (nrec x M); default Jeffreys posterior of each placebo arm alone (BENCH-2), or hierarchical (BENCH-2.1)."""
    if p0n is None:
        p0n = af3.beta_nodes(Kt.x0.values.astype(float), Kt.n0.values.astype(float), M)
    ak, ax, an = arms_t.kk.values, arms_t.x.values.astype(float), arms_t.n.values.astype(float)
    res = {}
    for m, q in preds.items():
        ok = ~np.isnan(q).any(1)
        sc = np.full(len(Kt), np.nan); pit = np.full(len(Kt), np.nan)
        if ok.any():
            idx = np.where(ok)[0]; rm = -np.ones(len(Kt), int); rm[idx] = np.arange(len(idx))
            sel = ok[ak]
            off = arms_t.off.values[sel] if scales[m] == "hz" else np.zeros(sel.sum())
            sc[idx] = af3.score_records_beta(q[idx], ax[sel], an[sel], p0n[idx], rm[ak[sel]], off, scales[m], len(idx))
            pit[idx] = af3.pit_records_beta(q[idx], ax[sel], an[sel], p0n[idx], rm[ak[sel]], off, scales[m], len(idx), rng)
        res[m] = (sc, pit)
    return res


def placebo_prior(Ktr, soc_of_term, iters=8, inflate=1.2):
    """Hierarchical background (placebo) risk model fitted on training trials by EM with the EXACT binomial likelihood on
    a logit grid: logit p0 ~ N(ym + term effect + organ-system effect + trial covariates, s^2_organ system); effects by
    weighted ridge on posterior means. Returns a predictor giving (mean, sd) of logit p0 for new records."""
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    d = Ktr.drop_duplicates(["nct", "t"]).copy()
    X = af3.design_covariates(d)[BASE_COVS].values
    soc = d.t.map(soc_of_term).fillna("unknown").values; terms = d.t.values
    ti = {t: i for i, t in enumerate(sorted(set(terms)))}; si = {s: i for i, s in enumerate(sorted(set(soc)))}
    nT, nS = len(ti), len(si); rows = np.arange(len(d))
    A = sparse.hstack([sparse.csr_matrix((np.ones(len(d)), (rows, [ti[t] for t in terms])), shape=(len(d), nT)),
                       sparse.csr_matrix((np.ones(len(d)), (rows, [si[s] for s in soc])), shape=(len(d), nS)),
                       sparse.csr_matrix(X)]).tocsr()
    g = np.arange(-12, 2.0001, 0.1)
    pg = 1 / (1 + np.exp(-g))
    x0 = d.x0.values[:, None].astype(float); n0 = d.n0.values[:, None].astype(float)
    ll = x0 * np.log(pg)[None, :] + (n0 - x0) * np.log1p(-pg)[None, :]
    ll -= ll.max(1, keepdims=True); L = np.exp(ll)
    p_ = (d.x0 + .5) / (d.n0 + 1); m = np.log(p_ / (1 - p_)).values
    s2 = np.ones(len(d)); ym = float(np.mean(m)); b = np.zeros(A.shape[1])
    socs = pd.Series(soc)
    for _ in range(iters):
        post = L * np.exp(-0.5 * (g[None, :] - m[:, None]) ** 2 / s2[:, None])
        post /= post.sum(1, keepdims=True)
        E = (post * g).sum(1); V = (post * g ** 2).sum(1) - E ** 2
        ym = float(np.mean(E))
        b = lsqr(A, E - ym, damp=np.sqrt(3.0), atol=1e-8, btol=1e-8, iter_lim=3000, x0=b)[0]
        m = ym + A @ b
        s2s = pd.Series(V + (E - m) ** 2).groupby(socs.values).mean().clip(lower=0.01)
        s2 = socs.map(s2s).values
    bt, bs, bc = b[:nT], b[nT:nT + nS], b[nT + nS:]
    def predict(Kt):
        Xt = af3.design_covariates(Kt)[BASE_COVS].values
        st = Kt.t.map(soc_of_term).fillna("unknown").values
        mu = ym + np.array([bt[ti[t]] if t in ti else 0.0 for t in Kt.t.values]) \
            + np.array([bs[si[s]] if s in si else 0.0 for s in st]) + Xt @ bc
        sd = inflate * np.sqrt(np.array([s2s.get(s, float(s2s.median())) for s in st]))   # inflation tuned on dev windows
        return mu, sd
    return predict


def placebo_nodes(Kt, mu, sd, M=20, G=121):
    """Posterior nodes of each test record's placebo risk: prior logit-normal(mu, sd) x binomial(x0, n0), on a logit grid;
    returns M equal-probability nodes (nrec x M)."""
    from scipy.special import expit
    z = mu[:, None] + sd[:, None] * np.linspace(-6, 6, G)[None, :]
    p = expit(z)
    x0 = Kt.x0.values[:, None].astype(float); n0 = Kt.n0.values[:, None].astype(float)
    lw = -0.5 * ((z - mu[:, None]) / sd[:, None]) ** 2 + x0 * np.log(p) + (n0 - x0) * np.log1p(-p)
    lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
    cdf = np.cumsum(w, 1)
    u = (np.arange(M) + .5) / M
    idx = np.array([np.searchsorted(c, u) for c in cdf]).clip(0, G - 1)
    return np.take_along_axis(p, idx, 1)


def class_mean(d, drug_atc3):
    """B6 class-mean baseline: random-effects (DerSimonian-Laird) mean of the side effect across training records of drugs
    in the same ATC level-3 class; returns a frame indexed by (atc3, t)."""
    d = d.assign(atc3=d.drug.map(drug_atc3)).dropna(subset=["atc3"])
    rows = []
    for (a, t), q in d.groupby(["atc3", "t"]):
        y, v = q.y.values, q.v.values
        w = 1 / v; mu0 = np.sum(w * y) / np.sum(w)
        Q = np.sum(w * (y - mu0) ** 2)
        t2 = max(0.0, (Q - (len(y) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if len(y) > 1 else 0.25
        w2 = 1 / (v + t2); rows.append((a, t, np.sum(w2 * y) / np.sum(w2), 1 / np.sum(w2), t2, len(y)))
    return pd.DataFrame(rows, columns=["atc3", "t", "cls_mu", "cls_se2", "cls_t2", "cls_n"]).set_index(["atc3", "t"])
