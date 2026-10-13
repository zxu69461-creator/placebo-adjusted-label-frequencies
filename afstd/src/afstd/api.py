"""User-facing API of the AF-STD reference implementation."""
import numpy as np
import pandas as pd

from . import core
from .schema import Records

SPEC_VERSION = "AF-STD 1.1 (estimator AF 3.3; benchmark BENCH-2.2; validated scope: non-serious adverse events; serious events provisional)"


def _prepare(rec: Records):
    K = rec.K.copy(); cf = rec.cf.copy()
    cidx = pd.Series(np.arange(len(cf)), index=pd.MultiIndex.from_frame(cf[["drug", "term"]]))
    K["cell"] = cidx.reindex(pd.MultiIndex.from_frame(K[["drug", "t"]])).values
    socs = sorted(cf.soc.unique()); sidx = {s: i for i, s in enumerate(socs)}
    cf["s"] = cf.soc.map(sidx).values
    soc_k = cf.s.values[K.cell.values]
    X = core.design_covariates(K)
    gam, ginfo = core.dose_slopes(rec.arms, K, soc_k, len(socs))
    rk, rx, rn, isd, off = core.rows_for(rec.arms, K, gam, soc_k)
    LL = core.profile_loglik(rk, rx, rn, isd, off, len(K))
    return dict(K=K, arms=rec.arms, cf=cf, socs=socs, soc_k=soc_k, X=X, gamma=gam, gamma_info=ginfo, LL=LL)


KAPPA = {"3.3": 0.1}          # predictive spread for new trials (development-zone tuned); 0 for earlier versions


def fit(rec: Records, version="3.3", **kw):
    """Fit AF 3.x ('3.0': organ-system heterogeneity; '3.2': cell-specific heterogeneity mixture; '3.3': 3.2 plus a
    no-effect spike in the prior, organ-system-specific, and predictive spread kappa for new trials)."""
    P = _prepare(rec)
    fn = core.em_fit_mix if version in ("3.2", "3.3") else core.em_fit
    if version == "3.3":
        kw.setdefault("spike", True)
    fitd = fn(P["LL"], P["X"], P["soc_k"], P["K"].cell.values, P["cf"], len(P["socs"]), verbose=kw.pop("verbose", False), **kw)
    base12, binfo = core.baseline_terms(P["K"], P["X"])
    return dict(P=P, fit=fitd, version=version, base12=base12, baseline=binfo, spec=SPEC_VERSION)


def estimates(model):
    """One row per drug x event: reference-trial log HR, excess fractions, probability mostly background, predictive
    interval for a new population, and whether direct trial evidence exists."""
    P, f = model["P"], model["fit"]
    S = core.summarise(f, P["cf"], P["cf"].s.values, model["base12"])
    S["n_trials"] = P["K"].groupby("cell").nct.nunique().reindex(np.arange(len(S))).fillna(0).astype(int).values
    if "p_null" in f and f.get("pi0") is not None and np.any(f["pi0"] > 0):
        S["p_no_effect"] = f["p_null"]
    if "tau_w" in f:
        et2 = (f["tau_w"] * f["tau_grid"] ** 2).sum(1) + KAPPA.get(model["version"], 0.0) ** 2
        S["pred_lo"] = 1 - np.exp(-(S.psi_mean - 1.96 * np.sqrt(S.psi_sd ** 2 + et2)))
        S["pred_hi"] = 1 - np.exp(-(S.psi_mean + 1.96 * np.sqrt(S.psi_sd ** 2 + et2)))
    S["spec"] = model["spec"]; S["estimator"] = f"AF {model['version']}"
    return S.rename(columns={"term": "event"})


def predict_trial(model, rec_new: Records):
    """Predictive density (on core.GRID) of the log HR for every record of new trials, given their design; cells unseen
    in the fit use the structured prior."""
    P, f = model["P"], model["fit"]
    Kt = rec_new.K.copy()
    key = pd.MultiIndex.from_frame(P["cf"][["drug", "term"]])
    cell = pd.Series(np.arange(len(key)), index=key).reindex(pd.MultiIndex.from_frame(Kt[["drug", "t"]])).values
    new = np.isnan(cell)
    G = len(core.GRID)
    post = np.zeros((len(Kt), G)); soc = np.zeros(len(Kt), int)
    post[~new] = f["post"][cell[~new].astype(int)]; soc[~new] = P["cf"].s.values[cell[~new].astype(int)]
    if new.any():
        nc = rec_new.cf.merge(Kt.loc[new, ["drug", "t"]].drop_duplicates().rename(columns={"t": "term"}), on=["drug", "term"])
        m = core.prior_for_new(P["cf"], f, nc)
        mm = dict(zip(zip(nc.drug, nc.term), m)); sm = {s: i for i, s in enumerate(P["socs"])}
        smap = dict(zip(zip(nc.drug, nc.term), nc.soc))
        for i in np.where(new)[0]:
            kk = (Kt.drug.values[i], Kt.t.values[i])
            idx = np.clip(np.arange(G) - int(np.rint(mm[kk] / core.DZ)), 0, G - 1)
            post[i] = f["gprior"][idx] / f["gprior"][idx].sum(); soc[i] = sm.get(smap[kk], 0)
            if f.get("pi0") is not None and np.any(f["pi0"] > 0):
                p0_ = f["pi0"][soc[i]]; post[i] = (1 - p0_) * post[i]; post[i, int(np.argmin(np.abs(core.GRID)))] += p0_
    shift = np.einsum("ij,ij->i", core.design_covariates(Kt).values, f["beta"][soc])
    if "tau_w" in f:
        tw = np.zeros((len(Kt), len(f["tau_grid"]))); tw[~new] = f["tau_w"][cell[~new].astype(int)]; tw[new] = f["tau_pi"][soc[new]]
        q = core.predictive_grid_mix(post, shift, tw, f["tau_grid"])
        if KAPPA.get(model["version"]):
            q = q @ core._kernels(KAPPA[model["version"]])[0].T.astype(np.float64); q /= q.sum(1, keepdims=True)
    else:
        q = core.predictive_grid(post, shift, f["tau"][soc])
    return Kt.assign(seen=~new), q


def report(model, drug, event, table=None):
    """Checklist-conformant report (AF-STD section 5) for one drug x event."""
    t = table if table is not None else estimates(model)
    r = t[(t.drug == drug) & (t.event == " ".join(str(event).lower().split()))]
    if r.empty:
        raise KeyError(f"{drug} x {event} not in the fitted cells")
    r = r.iloc[0]
    P = model["P"]
    k = P["K"][(P["K"].drug == drug) & (P["K"].t == r.event)]
    direct = "direct trial evidence" if r.n_trials > 0 else "predicted from the structured prior only (no trial of this pair)"
    lines = [
        f"Specification: {model['spec']}; estimator {r.estimator}",
        f"Drug x event: {drug} x {r.event} (organ system: {r.soc})",
        f"Evidence: {int(r.n_trials)} trials, {int(k.n0.sum() if len(k) else 0):,} placebo participants; {direct}",
        f"Excess fraction (hazard scale, reference trial): {r.af_hz:.2f} (95% CrI {r.af_hz_lo:.2f} to {r.af_hz_hi:.2f})",
        f"Excess fraction over 12 weeks (risk scale): {r.af_risk12:.2f}",
        f"Probability the event is mostly background (AF < 0.5): {r.p_af_lt_half:.2f}",
        f"95% predictive interval in a new population: {r.pred_lo:.2f} to {r.pred_hi:.2f}",
        "Reference trial: 12 weeks, double-blind, systematic assessment, industry-sponsored, phase 2-4 patients, median dose, "
        "age 50, 50% women, 200 participants",
    ]
    return "\n".join(lines)
