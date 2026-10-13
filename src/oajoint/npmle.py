"""Kiefer-Wolfowitz NPMLE of a mixing distribution for heteroscedastic normal means (gate af_deconvolution).

x_i ~ N(theta_i, s_i^2), theta ~ G supported on a fixed grid. EM fixed point, accelerated with SQUAREM
(Varadhan & Roland 2008), which converges to the same fixed point as plain EM.
"""
from __future__ import annotations

import numpy as np


def lik_matrix(x, s, grid):
    z = (x[:, None] - grid[None, :]) / s[:, None]
    return np.exp(-0.5 * z * z) / (s[:, None] * np.sqrt(2 * np.pi))


def _em(w, L):
    f = L @ w
    return w * (L.T @ (1 / f)) / len(f), float(np.log(f).sum())


def fit(x, s, grid, w0=None, tol=1e-8, max_iter=20000):
    L = lik_matrix(np.asarray(x, float), np.asarray(s, float), grid)
    w = np.full(len(grid), 1 / len(grid)) if w0 is None else w0.copy()
    ll_old = -np.inf
    for it in range(max_iter):
        w1, _ = _em(w, L); w2, _ = _em(w1, L)
        r, v = w1 - w, w2 - 2 * w1 + w
        a = -np.sqrt((r @ r) / max(v @ v, 1e-300))
        wn = np.clip(w - 2 * a * r + a * a * v, 0, None); wn /= wn.sum()
        wn, ll = _em(wn, L)                       # stabilising EM step
        if not np.isfinite(ll) or ll < ll_old - 1e-6:   # SQUAREM step failed -> plain EM
            wn, ll = _em(w2, L)
        if ll - ll_old < tol and it > 5:
            w = wn; break
        w, ll_old = wn, ll
    return w, ll_old, it + 1, L


def posterior(L, w, grid):
    P = L * w[None, :]
    P /= P.sum(1, keepdims=True)
    return P
