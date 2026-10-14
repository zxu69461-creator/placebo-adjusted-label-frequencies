#!/usr/bin/env python3
"""Summaries for the third internal review: evaluates the rules registered in 152_bmc_r4_prereg.py and
152b_bmc_r4_prereg.py, from the primary fit (afstd_pair, afstd_pair_exp, archived posterior grids), the refits of
153_bmc_r4_refits.py (jno_cell_variant 'bmc_*') and earlier tables. Writes jno_summary rows (analysis = rule name) and
evaluates each rule (all descriptive)."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import af_common as af  # noqa: E402
from oajoint import provenance as pv  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S110 = load("s110", "110_afstd_findings.py")
M128 = load("m128", "128_jno_rev_analyses.py")
M119 = load("m119", "119_jno_label_section.py")
M130b = load("m130b", "130b_jno_rev_calibration_noise.py")
SEED = 20261014
STRATA = ["1", "2-4", "5-9", ">=10"]
ROWS, DET = [], {}


def row(analysis, set_name, grp, metric, value, lo=np.nan, hi=np.nan, n=0, nd=0):
    ROWS.append(dict(analysis=analysis, set_name=set_name, grp=str(grp), n_pairs=int(n), n_drugs=int(nd), metric=metric,
                     value=float(value), lo=float(lo), hi=float(hi)))


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        P = con.execute("""SELECT p.*, e.af_mean, e.per100_drug_exp, e.per100_added_exp FROM afstd_pair p
                           JOIN afstd_pair_exp e USING (drug, term)""").df()
        V = con.execute("SELECT * EXCLUDE (run_id) FROM jno_cell_variant WHERE variant LIKE 'bmc_%' OR variant = 'pre2017'").df()
        info = con.execute("SELECT set_name, metric, value FROM jno_summary WHERE analysis = 'bmc_r4_refit_info'").df()
        comp = con.execute("SELECT drug, term, p_af_lt_half FROM jno_cell_complete").df()
        atc = {r[0]: [a for a in (r[1] or [])] for r in con.execute("SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
        names = dict(con.execute("SELECT chembl_id, display_name FROM ref_drug_name").fetchall())
        disc = con.execute("SELECT * EXCLUDE (run_id) FROM jno_discontinuation_contrast").df()
        enr = con.execute("SELECT nct, enrollment FROM ctgov_trial").df()
        pool = con.execute("SELECT generic_name, pool, adverse_reactions_text FROM fda_label_pool "
                           "ORDER BY generic_name, pool").df()
        cal = json.loads(con.execute("SELECT metric_detail FROM prov_decision WHERE gate_name = "
                                     "'jno_rev_temporal_calibration_noise'").fetchone()[0])
        G = con.execute("""SELECT g.* FROM v_afstd_posterior g JOIN afstd_pair p USING (drug, term) WHERE p.on_fda""").df()
    L = P[P.on_fda].copy()
    L["ev"] = L.evidence
    prim = float(L.p_af_lt_half.mean())
    var = {v: d for v, d in V.groupby("variant")}

    # ------------------------------------------------------------------ downsampling
    dense = L[L.ev == ">=10"][["drug", "term", "p_af_lt_half", "af_mean"]]
    ds = {}
    for k in (1, 3):
        diffs, shares, means = [], [], []
        for draw in (1, 2, 3):
            d = var.get(f"bmc_ds_{k}_{draw}")
            if d is None:
                continue
            m = dense.merge(d[["drug", "term", "p_af_lt_half", "af_hz", "n_trials"]], on=["drug", "term"],
                            suffixes=("_full", "_ds"))
            means.append(m.p_af_lt_half_ds.mean()); diffs.append(m.p_af_lt_half_ds.mean() - m.p_af_lt_half_full.mean())
            shares.append((m.af_hz < .5).mean())
            row("bmc_r4_downsample", f"k{k}", f"draw{draw}", "exp_lt_half_ds", m.p_af_lt_half_ds.mean(), n=len(m),
                nd=m.drug.nunique())
            row("bmc_r4_downsample", f"k{k}", f"draw{draw}", "exp_lt_half_full", m.p_af_lt_half_full.mean(), n=len(m))
        if diffs:
            ds[k] = dict(n_draws=len(diffs), full=float(dense.p_af_lt_half.mean()), ds_mean=float(np.mean(means)),
                         bias_mean=float(np.mean(diffs)), bias_min=float(np.min(diffs)), bias_max=float(np.max(diffs)),
                         share_point_lt_half=float(np.mean(shares)))
            row("bmc_r4_downsample", f"k{k}", "all", "bias", np.mean(diffs), np.min(diffs), np.max(diffs), n=len(dense))
    DET["bmc_r4_downsample"] = dict(n_dense=len(dense), by_k=ds)

    # ------------------------------------------------------------------ prior only and pi0 profile (bmc_full)
    full = var.get("bmc_full")
    if full is not None:
        chk = L[["drug", "term", "p_af_lt_half"]].merge(full[["drug", "term", "p_af_lt_half"]], on=["drug", "term"],
                                                        suffixes=("", "_re"))
        repro = float(chk.p_af_lt_half_re.mean() - chk.p_af_lt_half.mean())
        po = L[["drug", "term", "ev"]].merge(var["bmc_prior_only"][["drug", "term", "p_af_lt_half"]], on=["drug", "term"])
        det = dict(reproduce_diff=repro, prior_all=float(po.p_af_lt_half.mean()))
        row("bmc_r4_prior_only", "listed", "all", "prior_exp", po.p_af_lt_half.mean(), n=len(po))
        for s in STRATA:
            v = po[po.ev == s].p_af_lt_half.mean(); det[f"prior_{s}"] = float(v)
            row("bmc_r4_prior_only", "listed", s, "prior_exp", v, n=(po.ev == s).sum())
        DET["bmc_r4_prior_only"] = det
        pdet = {}
        fz = L[["drug", "term", "ev"]].merge(full[["drug", "term", "p_no_effect"]], on=["drug", "term"])
        for c in ("0", "0.5", "1.5"):
            d = var[f"bmc_pi0x{c}"]
            pdet[f"x{c}"] = float(d.p_af_lt_half.mean())
            row("bmc_r4_pi0_profile", f"x{c}", "all", "exp_lt_half", d.p_af_lt_half.mean(), n=len(d))
        pdet["x1"] = float(full.merge(L[["drug", "term"]]).p_af_lt_half.mean())
        pdet["spike_weight"] = {s: float(fz[fz.ev == s].p_no_effect.mean()) for s in STRATA}
        pdet["spike_weight_all"] = float(fz.p_no_effect.mean())
        for s in STRATA:
            row("bmc_r4_pi0_profile", "spike_weight", s, "mean_p_no_effect", pdet["spike_weight"][s], n=(fz.ev == s).sum())
        DET["bmc_r4_pi0_profile"] = pdet

    # ------------------------------------------------------------------ pivotal-like and add-on refits
    for gate, v in (("bmc_r4_pivotal", "bmc_pivotal"), ("bmc_r4_addon", "bmc_addon")):
        d = var.get(v)
        if d is None:
            continue
        lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
        inf = info[info.set_name == v].set_index("metric").value.to_dict()
        DET[gate] = dict(exp=float(d.p_af_lt_half.mean()), lo=lo, hi=hi, n_pairs=len(d), n_drugs=int(d.drug.nunique()),
                         **{k: float(x) for k, x in inf.items()})
        row(gate, "listed", "all", "exp_lt_half", d.p_af_lt_half.mean(), lo, hi, n=len(d), nd=d.drug.nunique())

    # ------------------------------------------------------------------ like-for-like complete reporting
    cm = comp.merge(L[["drug", "term", "p_af_lt_half"]], on=["drug", "term"], suffixes=("_complete", "_primary"))
    DET["bmc_r4_complete_same"] = dict(n_pairs=len(cm), n_drugs=int(cm.drug.nunique()),
                                       complete=float(cm.p_af_lt_half_complete.mean()),
                                       primary_same_pairs=float(cm.p_af_lt_half_primary.mean()))
    row("bmc_r4_complete_same", "same_pairs", "all", "exp_complete", cm.p_af_lt_half_complete.mean(), n=len(cm))
    row("bmc_r4_complete_same", "same_pairs", "all", "exp_primary", cm.p_af_lt_half_primary.mean(), n=len(cm))

    # ------------------------------------------------------------------ threshold curve
    gcols = [c for c in G.columns if re.fullmatch(r"g\d{3}", c)]
    assert len(gcols) == len(af3.GRID)
    Gm = G.merge(L[["drug", "term", "ev"]], on=["drug", "term"])
    post = Gm[gcols].values.astype(float); post /= post.sum(1, keepdims=True)
    tc = {}
    for t in (0.0, 0.2, 1 / 3, 0.5, 2 / 3, 0.8):
        if t == 0:
            p = post[:, af3.GRID <= 0].sum(1); lab = "le0"
        else:
            p = post[:, af3.GRID < -np.log(1 - t)].sum(1); lab = f"{t:.2f}"
        tc[lab] = {"all": float(p.mean()), **{s: float(p[Gm.ev.values == s].mean()) for s in STRATA}}
        row("bmc_r4_threshold_curve", lab, "all", "exp_lt_t", p.mean(), n=len(p))
        for s in STRATA:
            row("bmc_r4_threshold_curve", lab, s, "exp_lt_t", p[Gm.ev.values == s].mean(), n=(Gm.ev.values == s).sum())
    assert abs(tc["0.50"]["all"] - prim) < 0.002, (tc["0.50"]["all"], prim)
    DET["bmc_r4_threshold_curve"] = tc

    # ------------------------------------------------------------------ relative x absolute
    lo_ = L.af_mean < .5; ab = L.per100_added_exp >= 1
    ja = {"lt_half_added_lt1": float((lo_ & ~ab).mean()), "lt_half_added_ge1": float((lo_ & ab).mean()),
          "ge_half_added_lt1": float((~lo_ & ~ab).mean()), "ge_half_added_ge1": float((~lo_ & ab).mean())}
    for k, v in ja.items():
        row("bmc_r4_joint_abs", "listed", k, "share", v, n=len(L))
    DET["bmc_r4_joint_abs"] = ja

    # ------------------------------------------------------------------ weighting
    dw = float(L.groupby("drug").p_af_lt_half.mean().mean()); tw = float(L.groupby("term").p_af_lt_half.mean().mean())
    rw = float(L.per100_pl.sum() / L.per100_drug_exp.sum())
    DET["bmc_r4_weighted"] = dict(drug_weighted=dw, term_weighted=tw, report_weighted_placebo_share=rw,
                                  n_drugs=int(L.drug.nunique()), n_terms=int(L.term.nunique()))
    for k, v in DET["bmc_r4_weighted"].items():
        if not k.startswith("n_"):
            row("bmc_r4_weighted", "listed", k, "value", v, n=len(L))

    # ------------------------------------------------------------------ term length
    one = L.term.str.split().str.len() == 1
    tl = {}
    for nm, d in (("single", L[one]), ("multi", L[~one])):
        lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
        tl[nm] = dict(n=len(d), exp=float(d.p_af_lt_half.mean()), lo=lo, hi=hi)
        row("bmc_r4_term_length", nm, "all", "exp_lt_half", d.p_af_lt_half.mean(), lo, hi, n=len(d), nd=d.drug.nunique())
    DET["bmc_r4_term_length"] = tl

    # ------------------------------------------------------------------ positive controls
    Pc = P[P.n_trials >= 2].copy()
    pcs = []
    for lab, pre, rx in M128.PC:
        d = Pc[Pc.drug.map(lambda x: any(a.startswith(pre) for a in atc.get(x, []))) & Pc.term.map(lambda t: bool(re.search(rx, t)))]
        pcs.append(d.assign(control=lab))
    PCd = pd.concat(pcs).drop_duplicates(["drug", "term"])
    PCd["ev"] = S110.evidence(PCd.n_trials.values)
    PCd["name"] = PCd.drug.map(names)
    pc = {"all": dict(n=len(PCd), mean_p_lt_half=float(PCd.p_af_lt_half.mean()), share_point_lt_half=float((PCd.af_mean < .5).mean()))}
    for s in ["2-4", "5-9", ">=10"]:
        d = PCd[PCd.ev == s]
        pc[s] = dict(n=len(d), mean_p_lt_half=float(d.p_af_lt_half.mean()), share_point_lt_half=float((d.af_mean < .5).mean()))
    d = PCd[~PCd.name.isin(["mirabegron", "vibegron"])]
    pc["excl_beta3"] = dict(n=len(d), mean_p_lt_half=float(d.p_af_lt_half.mean()), share_point_lt_half=float((d.af_mean < .5).mean()))
    pc["by_control"] = PCd.groupby("control").agg(n=("drug", "size"), mean_p=("p_af_lt_half", "mean"),
                                                  share_point=("af_mean", lambda x: float((x < .5).mean()))).round(3).reset_index().to_dict("records")
    for k, v in pc.items():
        if k != "by_control":
            row("bmc_r4_posctrl", "positive_controls", k, "mean_p_lt_half", v["mean_p_lt_half"], n=v["n"])
            row("bmc_r4_posctrl", "positive_controls", k, "share_point_lt_half", v["share_point_lt_half"], n=v["n"])
    DET["bmc_r4_posctrl"] = pc

    # ------------------------------------------------------------------ temporal calibration by earlier evidence
    pre = var["pre2017"][["drug", "term", "psi_mean", "psi_sd", "p_af_lt_half", "p_no_effect", "n_trials"]]
    with pv.connect(read_only=True) as con:
        arms, K, _ = af3.build(con)
        post_tr = set(K[K.start >= pd.Timestamp("2017-02-01")].nct)
        Pp = af3.prepare(con, arms, K, trials=post_tr)
    g = af3.af1_table(Pp["K"], Pp["arms"]).join(Pp["cf"][["drug", "term"]], how="left")
    g["obs"] = (1 - np.exp(-g.theta) < .5).astype(int)
    c = g.merge(pre, on=["drug", "term"])
    w0 = c.p_no_effect.clip(0, .999)
    m1 = c.psi_mean / (1 - w0)
    v1 = np.maximum((c.psi_sd ** 2 + c.psi_mean ** 2) / (1 - w0) - m1 ** 2, 1e-6)
    L2 = np.log(2)
    c["p_obs"] = w0 * stats.norm.cdf(L2 / c.se) + (1 - w0) * stats.norm.cdf((L2 - m1) / np.sqrt(v1 + c.se ** 2))
    c = c.merge(L[["drug", "term"]].assign(listed=1), on=["drug", "term"], how="left")
    c["listed"] = c.listed.fillna(0).astype(bool)
    c["pre_ev"] = np.select([c.n_trials <= 1, c.n_trials <= 4], ["1", "2-4"], ">=5")
    assert abs(c.p_af_lt_half.mean() - cal["mean_pred"]) < 1e-6 if "mean_pred" in cal else True
    tsd = {}
    for nm, d in [("all", c), ("listed", c[c.listed])] + [(f"listed_{s}", c[c.listed & (c.pre_ev == s)]) for s in ("1", "2-4", ">=5")] \
            + [(f"all_{s}", c[c.pre_ev == s]) for s in ("1", "2-4", ">=5")]:
        if len(d) < 30:
            tsd[nm] = dict(n=len(d)); continue
        s_reg, _ = M130b.slope(d.obs, d.p_af_lt_half); s_noi, _ = M130b.slope(d.obs, d.p_obs)
        tsd[nm] = dict(n=len(d), pred=float(d.p_af_lt_half.mean()), pred_noise=float(d.p_obs.mean()),
                       obs=float(d.obs.mean()), slope=s_reg, slope_noise=s_noi)
        for k_ in ("pred", "pred_noise", "obs", "slope", "slope_noise"):
            row("bmc_r4_temporal_strata", nm, "all", k_, tsd[nm][k_], n=len(d))
    DET["bmc_r4_temporal_strata"] = tsd

    # ------------------------------------------------------------------ recalibrated probabilities (post hoc)
    b = float(cal["slope_noise_aware"])
    lp = np.log(np.clip(L.p_af_lt_half, 1e-6, 1 - 1e-6) / (1 - np.clip(L.p_af_lt_half, 1e-6, 1 - 1e-6)))
    a = optimize.brentq(lambda a_: (1 / (1 + np.exp(-(a_ + b * lp)))).mean() - prim, -10, 10)
    pr = 1 / (1 + np.exp(-(a + b * lp)))
    DET["bmc_r4_recalibrated"] = dict(slope=b, intercept=float(a), exp=float(pr.mean()), share_gt975=float((pr > .975).mean()),
                                      share_lt025=float((pr < .025).mean()), orig_gt975=float((L.p_af_lt_half > .975).mean()),
                                      orig_lt025=float((L.p_af_lt_half < .025).mean()))
    for k in ("share_gt975", "share_lt025"):
        row("bmc_r4_recalibrated", "listed", k, "share", DET["bmc_r4_recalibrated"][k], n=len(L))

    # ------------------------------------------------------------------ discontinuation, random effects
    D = disc.merge(enr, on="nct", how="left")
    D["atc_all"] = D.drug.map(lambda d: atc.get(d, []))
    rep = D.groupby("nct").apply(lambda g_: (g_.x_drug.sum() + g_.x_pl.sum()) > 0, include_groups=False)
    keep = D.nct.map(rep).astype(bool) & (D.enrollment.fillna(0) < 50000) & ~D.atc_all.map(lambda a_: any(x.startswith("J07") for x in a_))
    Dv = D[keep]
    z = Dv[(Dv.x_drug > 0) & (Dv.x_pl > 0) & (Dv.x_drug < Dv.n_drug) & (Dv.x_pl < Dv.n_pl)]
    y = np.log((z.x_drug / z.n_drug) / (z.x_pl / z.n_pl)).values
    v = (1 / z.x_drug - 1 / z.n_drug + 1 / z.x_pl - 1 / z.n_pl).values
    w = 1 / v; mu_f = (w * y).sum() / w.sum(); Q = (w * (y - mu_f) ** 2).sum(); k_ = len(y)
    t2 = max(0.0, (Q - (k_ - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (v + t2); mu = (wr * y).sum() / wr.sum(); se = np.sqrt(1 / wr.sum())
    tq = stats.t.ppf(.975, k_ - 2)
    DET["bmc_r4_disc_re"] = dict(n=k_, rr=float(np.exp(mu)), lo=float(np.exp(mu - 1.96 * se)), hi=float(np.exp(mu + 1.96 * se)),
                                 pi_lo=float(np.exp(mu - tq * np.sqrt(t2 + se ** 2))), pi_hi=float(np.exp(mu + tq * np.sqrt(t2 + se ** 2))),
                                 tau2=float(t2), I2=float(max(0, (Q - (k_ - 1)) / Q)))
    row("bmc_r4_disc_re", "restricted", "all", "rr_random", np.exp(mu), np.exp(mu - 1.96 * se), np.exp(mu + 1.96 * se), n=k_)
    row("bmc_r4_disc_re", "restricted", "all", "rr_pred_interval", np.exp(mu), DET["bmc_r4_disc_re"]["pi_lo"],
        DET["bmc_r4_disc_re"]["pi_hi"], n=k_)

    # ------------------------------------------------------------------ recall audit: missed reactions
    rs = pd.read_csv(ROOT / "manuscript" / "jama" / "review" / "recall_sample.csv")
    rf = pd.read_csv(ROOT / "manuscript" / "jama" / "review" / "recall_final.csv")
    rs = rs.reset_index().rename(columns={"index": "row"}).merge(rf, on="row")
    fn = rs[rs.category == "a"]
    bs = [rng.choice(fn.p_af_lt_half.values, len(fn)).mean() for _ in range(2000)]
    DET["bmc_r4_missed_ci"] = dict(n=len(fn), exp=float(fn.p_af_lt_half.mean()), lo=float(np.quantile(bs, .025)),
                                   hi=float(np.quantile(bs, .975)))
    row("bmc_r4_missed_ci", "false_negatives", "all", "exp_lt_half", fn.p_af_lt_half.mean(), np.quantile(bs, .025),
        np.quantile(bs, .975), n=len(fn))

    # ------------------------------------------------------------------ listed vs unlisted effect size and benchmark
    U = P[P.has_fda_label & ~P.on_fda & P.drug.isin(L.drug)]
    le = {}
    for nm, (a_, b_) in (("all", (L, U)), (">=10", (L[L.ev == ">=10"], U[U.evidence == ">=10"]))):
        u = stats.mannwhitneyu(a_.af_mean, b_.af_mean, alternative="two-sided")
        le[nm] = dict(n_listed=len(a_), n_unlisted=len(b_), auc=float(u.statistic / (len(a_) * len(b_))),
                      median_listed=float(a_.af_mean.median()), median_unlisted=float(b_.af_mean.median()))
        row("bmc_r4_listed_effect", nm, "all", "auc", le[nm]["auc"], n=len(a_))
    DET["bmc_r4_listed_effect"] = le
    lo, hi = S110.boot_mean(U, "p_af_lt_half", rng)
    DET["bmc_r4_unlisted_benchmark"] = dict(unlisted=float(U.p_af_lt_half.mean()), lo=lo, hi=hi, n_unlisted=len(U),
                                            listed=prim, positive_controls=pc["all"]["mean_p_lt_half"])
    row("bmc_r4_unlisted_benchmark", "unlisted_same_drugs", "all", "exp_lt_half", U.p_af_lt_half.mean(), lo, hi, n=len(U),
        nd=U.drug.nunique())

    # ------------------------------------------------------------------ CTT analogue and NNH bands
    d10 = L[L.ev == ">=10"]
    DET["bmc_r4_ctt_analog"] = dict(n=len(d10), share_hi_lt1=float((d10.per100_added_hi < 1).mean()),
                                    share_p_no_excess_gt_half=float((d10.p_no_excess > .5).mean()),
                                    n_hi_lt1=int((d10.per100_added_hi < 1).sum()))
    row("bmc_r4_ctt_analog", "listed_ge10", "all", "share_hi_lt1", (d10.per100_added_hi < 1).mean(), n=len(d10))
    a_ = L.per100_added_exp
    bands = {"lt1": float((a_ < 1).mean()), "1to5": float(((a_ >= 1) & (a_ < 5)).mean()), "ge5": float((a_ >= 5).mean())}
    per = L.groupby("drug").agg(n=("term", "size"), ge5=("per100_added_exp", lambda x: int((x >= 5).sum())),
                                lt1=("per100_added_exp", lambda x: float((x < 1).mean())))
    per = per[per.n >= 5]
    bands.update(n_drugs_ge5=len(per), median_ge5_per_drug=float(per.ge5.median()), median_share_lt1_per_drug=float(per.lt1.median()),
                 iqr_ge5=[float(per.ge5.quantile(.25)), float(per.ge5.quantile(.75))])
    for k in ("lt1", "1to5", "ge5"):
        row("bmc_r4_nnh_bands", "listed", k, "share", bands[k], n=len(L))
    DET["bmc_r4_nnh_bands"] = bands

    # ------------------------------------------------------------------ common terms
    ct = L.groupby("term").agg(n_drugs=("drug", "nunique"), pl=("per100_pl", "median"), added=("per100_added_exp", "median"),
                               exp=("p_af_lt_half", "mean")).sort_values(["n_drugs", "term"], ascending=[False, True]).head(20)
    DET["bmc_r4_common_terms"] = ct.round(4).reset_index().to_dict("records")
    for t, r_ in ct.iterrows():
        row("bmc_r4_common_terms", t, "all", "exp_lt_half", r_.exp, n=r_.n_drugs)

    # ------------------------------------------------------------------ coverage (SIDER proxy)
    label_raw, mapped = af.label_pairs()
    us = load("m109", "109_afstd_label_match.py").us
    sid = pd.DataFrame(sorted({(c_, us(t)) for c_, t in label_raw}), columns=["drug", "term"])
    sid = sid[sid.drug.isin(set(L.drug))]
    est = set(zip(P.drug, P.term))
    sid["estimated"] = [(d_, t) in est for d_, t in zip(sid.drug, sid.term)]
    one_ = sid.term.str.split().str.len() == 1
    DET["bmc_r4_coverage"] = dict(n_sider_pairs=len(sid), n_drugs=int(sid.drug.nunique()), coverage=float(sid.estimated.mean()),
                                  coverage_single=float(sid[one_].estimated.mean()), coverage_multi=float(sid[~one_].estimated.mean()))
    row("bmc_r4_coverage", "sider_pairs", "all", "coverage", sid.estimated.mean(), n=len(sid), nd=sid.drug.nunique())

    # ------------------------------------------------------------------ disclaimer lists
    idx, _ = M119.M109.M34.build_index()
    text, _, _ = M119.M109.pool_text(pool, idx)
    DIS = re.compile(r"not\s+necessarily\s+(?:been\s+)?(?:caused|related)", re.I)
    END = re.compile(r"post-?\s?marketing|(?<![\d.])6\.\d\s+[A-Z]", re.I)
    regions = {}
    for d_, t in text.items():
        rr = []
        for m in DIS.finditer(t):
            e = END.search(t, m.end())
            rr.append((m.start(), e.start() if e else len(t)))
        regions[d_] = rr
    only = []
    pats = {}
    for d_, t in zip(L.drug, L.term):
        rr = regions.get(d_) or []
        if not rr or d_ not in text:
            only.append(False); continue
        p = pats.setdefault(t, M119.M109.pattern(t))
        pos = [m.start() for m in p.finditer(text[d_])]
        only.append(bool(pos) and all(any(a_ <= x < b_ for a_, b_ in rr) for x in pos))
    L["disclaimer_only"] = only
    d = L[~L.disclaimer_only]
    lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
    DET["bmc_r4_disclaimer"] = dict(n_disclaimer_only=int(L.disclaimer_only.sum()),
                                    n_drugs_disclaimer=int(L[L.disclaimer_only].drug.nunique()),
                                    exp_disclaimer_only=float(L[L.disclaimer_only].p_af_lt_half.mean()) if L.disclaimer_only.any() else None,
                                    exp_excluding=float(d.p_af_lt_half.mean()), lo=lo, hi=hi, n=len(d))
    row("bmc_r4_disclaimer", "excluding_disclaimer_only", "all", "exp_lt_half", d.p_af_lt_half.mean(), lo, hi, n=len(d))

    for k, v in DET.items():
        print(k, json.dumps(v, default=float)[:600])
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [r_[0] for r_ in con.execute("""SELECT out_id FROM prov_output_asset WHERE object_name IN
                  ('afstd_pair','afstd_pair_exp','jno_cell_variant','jno_cell_complete','jno_discontinuation_contrast',
                   'fda_label_pool','ref_chembl_drug')""").fetchall()]
        R = pd.DataFrame(ROWS); R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis LIKE 'bmc_r4_%' AND analysis <> 'bmc_r4_refit_info'")
        run.archive_table("jno_summary", R, upstream=up)
        metric = {"bmc_r4_downsample": ("by_k", 1, "bias_mean"), "bmc_r4_prior_only": ("prior_1",),
                  "bmc_r4_pi0_profile": ("x0",), "bmc_r4_pivotal": ("exp",), "bmc_r4_addon": ("exp",),
                  "bmc_r4_complete_same": ("primary_same_pairs",), "bmc_r4_threshold_curve": ("0.50", "all"),
                  "bmc_r4_joint_abs": ("lt_half_added_ge1",), "bmc_r4_weighted": ("drug_weighted",),
                  "bmc_r4_term_length": ("single", "exp"), "bmc_r4_posctrl": ("all", "mean_p_lt_half"),
                  "bmc_r4_temporal_strata": ("listed", "slope"), "bmc_r4_recalibrated": ("exp",),
                  "bmc_r4_disc_re": ("rr",), "bmc_r4_missed_ci": ("exp",), "bmc_r4_listed_effect": ("all", "auc"),
                  "bmc_r4_ctt_analog": ("share_hi_lt1",), "bmc_r4_nnh_bands": ("ge5",), "bmc_r4_common_terms": None,
                  "bmc_r4_unlisted_benchmark": ("unlisted",), "bmc_r4_coverage": ("coverage",),
                  "bmc_r4_disclaimer": ("exp_excluding",)}
        for gate, path in metric.items():
            if gate not in DET:
                print("NOT EVALUATED (inputs missing):", gate); continue
            val = None
            if path is None:
                val = float(len(DET[gate]))
            else:
                x = DET[gate]
                try:
                    for p_ in path:
                        x = x[p_]
                    val = float(x)
                except (KeyError, TypeError):
                    val = None
            run.evaluate_gate(gate, val, None, detail=json.loads(json.dumps(DET[gate], default=float)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
