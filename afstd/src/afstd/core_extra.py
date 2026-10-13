"""Vendored scoring primitives (from oajoint/af3.py)."""
import numpy as np
from .core import GRID, _add_rows


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
