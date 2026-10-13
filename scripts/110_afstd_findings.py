#!/usr/bin/env python3
"""Manuscript-v2 findings from the hierarchical estimator (AF 3.3), exactly as registered by 108_afstd_prereg.py.

Writes afstd_pair (one row per estimated drug-side-effect pair with label flags, evidence stratum, zone, absolute risks
per 100 and listing rules), afstd_summary (long format: every number the paper quotes) and afstd_card (drug cards), and
evaluates gates afstd_headline, _sensitivity, _robustness, _label_validity, _by_soc, _by_class, _absolute,
_listing_rules and _examples. Reads only; writes no table that a live figure depends on.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

SEED, NBOOT = 20260927, 1000
STRATA = ["1", "2-4", "5-9", ">=10"]
ZONES = ["confident_bg", "probable_bg", "probable_drug", "confident_drug"]
GD = {"general disorders and administration site conditions": "general disorders"}
EX1 = [("amlodipine", "edema peripheral"), ("atorvastatin", "myalgia")]


def evidence(n):
    return np.select([n <= 1, n <= 4, n <= 9], ["1", "2-4", "5-9"], ">=10")


def zone(p):
    return np.select([p > .975, p > .5, p >= .025], ZONES[:3], ZONES[3])


def per100(p0, psi):
    return 100 * -np.expm1(np.log1p(-p0) * np.exp(psi))


def boot_mean(df, col, rng):
    """Drug-cluster bootstrap interval of the mean of `col` (pairs share hyperparameters, so resample drugs)."""
    g = df.groupby("drug")[col].agg(["sum", "size"])
    if len(g) < 2:
        return np.nan, np.nan
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=NBOOT)
    v = (w @ g["sum"].values) / np.maximum(w @ g["size"].values, 1)
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


class Summary:
    def __init__(self):
        self.rows = []

    def add(self, analysis, set_name, grp, d, metric, value, lo=np.nan, hi=np.nan):
        self.rows.append(dict(analysis=analysis, set_name=set_name, grp=grp, n_pairs=int(len(d)),
                              n_drugs=int(d.drug.nunique()) if len(d) else 0, metric=metric,
                              value=float(value) if value == value else np.nan, lo=lo, hi=hi))

    def headline(self, analysis, set_name, grp, d, rng, boot=True):
        lo, hi = boot_mean(d, "p_af_lt_half", rng) if boot else (np.nan, np.nan)
        self.add(analysis, set_name, grp, d, "exp_lt_half", d.p_af_lt_half.mean(), lo, hi)
        self.add(analysis, set_name, grp, d, "conf_bg", (d.p_af_lt_half > .975).mean())
        self.add(analysis, set_name, grp, d, "conf_drug", (d.p_af_lt_half < .025).mean())
        self.add(analysis, set_name, grp, d, "exp_no_effect", d.p_no_effect.mean())
        self.add(analysis, set_name, grp, d, "prob_no_effect", (d.p_no_effect > .5).mean())
        self.add(analysis, set_name, grp, d, "prob_no_excess", (d.p_no_excess > .5).mean())
        self.add(analysis, set_name, grp, d, "median_af", d.af_med.median())
        for z in ZONES:
            self.add(analysis, set_name, grp, d, f"zone_{z}", (d.zone == z).mean())

    def get(self, analysis, set_name, grp, metric):
        for r in self.rows:
            if (r["analysis"], r["set_name"], r["grp"], r["metric"]) == (analysis, set_name, grp, metric):
                return r["value"]
        raise KeyError((analysis, set_name, grp, metric))


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        est = con.execute("SELECT DISTINCT estimator, run_id FROM afstd_cell").fetchall()
        last102 = con.execute("""SELECT run_id FROM prov_script_run WHERE script_path LIKE '%102_af32_fit.py'
                                 AND exit_status = 0 ORDER BY started_at DESC LIMIT 1""").fetchone()[0]
        assert est == [("AF 3.3", last102)], f"afstd_cell is not the latest AF 3.3 fit: {est} vs {last102}"
        a = con.execute("SELECT * EXCLUDE (run_id, estimator) FROM afstd_cell").df()
        lm = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_label_match").df()
        drugs = con.execute("SELECT chembl_id, atc4, atc4_desc FROM ref_chembl_drug").df()
        names = dict(con.execute("SELECT chembl_id, display_name FROM ref_drug_name").fetchall())
        par = con.execute("SELECT kind, soc, name, value FROM af3_param WHERE fit = 'AF 3.3 full'").df()
        conv = con.execute("""SELECT c.drug, c.term, c.af, c.af_hi FROM af_cell c JOIN af_label_match m USING (drug, term)
                              WHERE m.on_fda""").df()
        arms, K, J = af3.build(con)
    # ---------------------------------------------------------------- per pair
    ntr = K.groupby(["drug", "t"]).nct.nunique().rename("n_k")
    pl = K.groupby(["drug", "t"]).agg(x0_own=("x0", "sum"), n0_own=("n0", "sum"))
    d = a.merge(lm, on=["drug", "term"], how="left", validate="1:1")
    d = d.merge(ntr, left_on=["drug", "term"], right_index=True, how="left").merge(
        pl, left_on=["drug", "term"], right_index=True, how="left")
    bad = int((d.n_k != d.n_trials).sum())
    assert bad == 0, f"{bad} pairs whose trial count differs from the estimator's data"
    atc = {c: sorted({x[:4] for x in (v if v is not None else [])}) for c, v in zip(drugs.chembl_id, drugs.atc4)}
    atc3_name, atc1_name = {}, {}
    for codes, descs in zip(drugs.atc4, drugs.atc4_desc):
        for c, s in zip(codes if codes is not None else [], descs if descs is not None else []):
            parts = (s or "").split(": ")
            if len(parts) >= 3:
                atc3_name.setdefault(c[:4], parts[2].capitalize()); atc1_name.setdefault(c[0], parts[0].capitalize())
    d["soc_h"] = d.soc.replace(GD)
    d["atc3"] = [atc.get(x, [None])[0] if atc.get(x) else None for x in d.drug]
    d["atc1"] = [x[0] if isinstance(x, str) else None for x in d.atc3]
    d["evidence"] = evidence(d.n_trials.values)
    d["zone"] = zone(d.p_af_lt_half.values)
    d["af_med"] = 1 - np.exp(-d.psi_med)
    d["promiscuous"] = d.term_promiscuity.fillna(0) > .5
    d["p0_own"] = (d.x0_own + .5) / (d.n0_own + 1)
    d["per100_pl"] = 100 * d.p0_own
    for q, c in (("lo", "psi_lo"), ("med", "psi_med"), ("hi", "psi_hi")):
        d[f"per100_drug_{q}"] = per100(d.p0_own, d[c])
        d[f"per100_added_{q}"] = d[f"per100_drug_{q}"] - d.per100_pl
    for q, c in (("lo", "pred_lo"), ("hi", "pred_hi")):
        d[f"per100_added_pred_{q}"] = per100(d.p0_own, -np.log1p(-d[c].clip(upper=1 - 1e-9))) - d.per100_pl
    d["per100_added_ref_med"] = per100(d.p0_12wk, d.psi_med) - 100 * d.p0_12wk
    d["rule_a"] = (d.per100_drug_med >= 2) & (d.af_med > 0)
    d["rule_b"] = d.p_no_excess < .025
    d["rule_c"] = d.rule_b & (d.per100_added_med >= 1)
    lab = d[d.on_fda].copy()                     # primary set
    haslab = d[d.has_fda_label].copy()
    print(f"pairs {len(d):,}; label-listed {len(lab):,} ({lab.drug.nunique():,} drugs); drugs with label "
          f"{haslab.drug.nunique():,}")

    S = Summary()
    # ---------------------------------------------------------------- counts
    S.add("counts", "all", "all", d, "pairs", len(d))
    S.add("counts", "all", "all", d, "terms", d.term.nunique())
    S.add("counts", "has_label", "all", haslab, "pairs", len(haslab))
    S.add("counts", "listed", "all", lab, "pairs", len(lab))
    S.add("counts", "listed", "all", lab, "terms", lab.term.nunique())
    S.add("counts", "listed_af1", "all", lab[lab.af1_estimable], "pairs", int(lab.af1_estimable.sum()))
    for s in STRATA:
        S.add("counts", "all", s, d[d.evidence == s], "pairs", int((d.evidence == s).sum()))
        S.add("counts", "listed", s, lab[lab.evidence == s], "pairs", int((lab.evidence == s).sum()))
    # ---------------------------------------------------------------- headline
    S.headline("headline", "primary", "all", lab, rng)
    for s in STRATA:
        S.headline("headline", "primary", s, lab[lab.evidence == s], rng)
    S.headline("headline", "unlisted", "all", haslab[~haslab.on_fda], rng)
    # ---------------------------------------------------------------- sensitivity
    sets = {"S1": lab[lab.af1_estimable], "S2": lab[lab.n_trials >= 2],
            "S3": d[d.sider_mapped & d.on_sider], "S6": lab[lab.n_trials >= 5],
            "S7": lab[lab.label_source == "innovator"], "S8": lab[~lab.promiscuous]}
    for k, x in sets.items():
        S.headline("sensitivity", k, "all", x, rng)
    S.add("sensitivity", "S4", "all", lab, "hz_mean_lt_half", (lab.af_hz < .5).mean())
    S.add("sensitivity", "S4", "all", lab, "risk_mean_lt_half", (lab.af_risk12 < .5).mean())
    S.add("sensitivity", "S5", "all", conv, "conv_afhi_lt_half", (conv.af_hi < .5).mean())
    S.add("sensitivity", "S5", "all", conv, "conv_af_lt_half", (conv.af < .5).mean())
    both = lab[lab.af1_estimable].merge(conv, on=["drug", "term"])
    S.add("sensitivity", "S1xS5", "all", both, "spearman_af", stats.spearmanr(both.af_med, both.af).statistic)
    rob_sets = {"primary": lab, **{k: sets[k] for k in ("S1", "S2", "S3", "S7", "S8")}}
    rob = {k: float(x.p_af_lt_half.mean()) for k, x in rob_sets.items()}
    rob_fail = [k for k, v in rob.items() if not v > .5]
    # ---------------------------------------------------------------- label validity
    x1, x0 = haslab[haslab.on_fda].psi_mean, haslab[~haslab.on_fda].psi_mean
    mw = stats.mannwhitneyu(x1, x0, alternative="greater")
    med_diff = float(x1.median() - x0.median())
    S.add("validity", "has_label", "all", haslab, "median_psi_listed", x1.median())
    S.add("validity", "has_label", "all", haslab, "median_psi_unlisted", x0.median())
    S.add("validity", "has_label", "all", haslab, "validity_p", mw.pvalue)
    S.add("validity", "has_label", "all", haslab, "auc", mw.statistic / (len(x1) * len(x0)))
    u = n = 0.0
    for _, g in haslab.groupby("drug"):
        a1, a0 = g[g.on_fda].psi_mean, g[~g.on_fda].psi_mean
        if len(a1) and len(a0):
            u += stats.mannwhitneyu(a1, a0).statistic; n += len(a1) * len(a0)
    S.add("validity", "has_label", "all", haslab, "auc_drug_stratified", u / n)
    for s in STRATA:
        g = haslab[haslab.evidence == s]
        g1, g0 = g[g.on_fda].psi_mean, g[~g.on_fda].psi_mean
        S.add("validity", "has_label", s, g, "median_psi_listed", g1.median())
        S.add("validity", "has_label", s, g, "median_psi_unlisted", g0.median())
        S.add("validity", "has_label", s, g, "auc", stats.mannwhitneyu(g1, g0).statistic / (len(g1) * len(g0)))
    # ---------------------------------------------------------------- organ systems
    tau = par[par.kind == "tau"].set_index("soc").value.to_dict()
    gam = par[par.kind == "gamma"].set_index("soc").value.to_dict()
    socs = sorted(s for s, g in lab.groupby("soc_h") if len(g) >= 30)
    for s in socs:
        g = lab[lab.soc_h == s]; allg = d[d.soc_h == s]
        lo, hi = boot_mean(g, "p_af_lt_half", rng)
        S.add("by_soc", "primary", s, g, "exp_lt_half", g.p_af_lt_half.mean(), lo, hi)
        S.add("by_soc", "primary", s, g, "conf_bg", (g.p_af_lt_half > .975).mean())
        S.add("by_soc", "primary", s, g, "median_af", g.af_med.median(), g.af_med.quantile(.25), g.af_med.quantile(.75))
        S.add("by_soc", "all_pairs", s, allg, "pi0", allg.p_no_effect.mean())
        S.add("by_soc", "model", s, allg, "tau", tau.get(s, np.nan))
        S.add("by_soc", "model", s, allg, "dose_slope", gam.get(s, np.nan))
    # ---------------------------------------------------------------- classes
    rows = [(c3, i) for i, dr in zip(lab.index, lab.drug) for c3 in atc.get(dr, [])]
    cl = pd.DataFrame(rows, columns=["atc3", "i"])
    n_cov = int(lab.drug.map(lambda x: bool(atc.get(x))).sum())
    S.add("by_class", "primary", "coverage", lab, "share_with_atc", n_cov / len(lab))
    n_class = 0
    for c3, g in cl.groupby("atc3"):
        if len(g) < 20:
            continue
        x = lab.loc[g.i]; n_class += 1
        lo, hi = boot_mean(x, "p_af_lt_half", rng)
        grp = f"{c3}|{atc3_name.get(c3, '')}"
        S.add("by_class", "primary", grp, x, "exp_lt_half", x.p_af_lt_half.mean(), lo, hi)
        S.add("by_class", "primary", grp, x, "conf_bg", (x.p_af_lt_half > .975).mean())
        S.add("by_class", "primary", grp, x, "median_af", x.af_med.median())
    rows = [(c3[0], i) for i, dr in zip(lab.index, lab.drug) for c3 in atc.get(dr, [])]
    c1 = pd.DataFrame(rows, columns=["atc1", "i"]).drop_duplicates()
    c1["soc_h"] = lab.loc[c1.i, "soc_h"].values
    for (x1_, s), g in c1.groupby(["atc1", "soc_h"]):
        if len(g) >= 10:
            x = lab.loc[g.i]
            S.add("by_class", "atc1_soc", f"{x1_}|{atc1_name.get(x1_, '')}|{s}", x, "exp_lt_half", x.p_af_lt_half.mean())
    # ---------------------------------------------------------------- absolute
    for s, g in [("all", lab)] + [(s, lab[lab.evidence == s]) for s in STRATA]:
        S.add("absolute", "primary", s, g, "median_per100_drug", g.per100_drug_med.median())
        S.add("absolute", "primary", s, g, "median_per100_pl", g.per100_pl.median())
        S.add("absolute", "primary", s, g, "median_per100_added", g.per100_added_med.median())
        S.add("absolute", "primary", s, g, "share_added_lt1", (g.per100_added_med < 1).mean())
        S.add("absolute", "primary", s, g, "share_added_hi_lt1", (g.per100_added_hi < 1).mean())
        pos = g[g.per100_added_med > 0]
        S.add("absolute", "primary", s, pos, "median_nnh", (100 / pos.per100_added_med).median())
    S.add("absolute", "reference_background", "all", lab, "median_per100_added", lab.per100_added_ref_med.median())
    S.add("absolute", "reference_background", "all", lab, "share_added_lt1", (lab.per100_added_ref_med < 1).mean())
    # ---------------------------------------------------------------- listing rules
    cd = haslab[haslab.on_fda & (haslab.p_af_lt_half < .025)]
    for r in ("rule_a", "rule_b", "rule_c"):
        kept = haslab[haslab.on_fda & haslab[r]]
        S.add("listing_rules", r, "all", haslab, "retained", len(kept) / haslab.on_fda.sum())
        S.add("listing_rules", r, "all", kept, "bg_share_retained", (kept.p_af_lt_half > .975).mean())
        S.add("listing_rules", r, "all", kept, "exp_lt_half_retained", kept.p_af_lt_half.mean())
        S.add("listing_rules", r, "all", cd, "drug_caused_retained", cd[r].mean())
        S.add("listing_rules", r, "all", haslab, "unlisted_added", int((~haslab.on_fda & haslab[r]).sum()))
    S.add("listing_rules", "current", "all", lab, "bg_share_retained", (lab.p_af_lt_half > .975).mean())
    S.add("listing_rules", "current", "all", lab, "exp_lt_half_retained", lab.p_af_lt_half.mean())
    # ---------------------------------------------------------------- examples and cards
    d["display_name"] = d.drug.map(names)
    ex = []
    for nm, t in EX1:
        r = d[(d.display_name == nm) & (d.term == t)]
        assert len(r) == 1, (nm, t)
        ex.append(r.iloc[0])
    ntr_d = K.groupby("drug").nct.nunique()
    elig = lab[lab.n_trials >= 2].groupby("drug").size()
    elig = elig[elig >= 8].index
    order = sorted(elig, key=lambda x: (-ntr_d.get(x, 0), names.get(x, x)))
    cards = []
    for rank, dr in enumerate(order[:18], 1):
        g = lab[(lab.drug == dr) & (lab.n_trials >= 2)].sort_values(["per100_drug_med", "term"], ascending=[False, True]).head(8)
        for rt, r in enumerate(g.itertuples(), 1):
            cards.append(dict(card_set="main" if rank <= 6 else "supplementary", rank_drug=rank, rank_term=rt, drug=dr,
                              display_name=names.get(dr, dr), term=r.term, n_trials=int(r.n_trials),
                              per100_pl=r.per100_pl, per100_drug_med=r.per100_drug_med,
                              per100_added_med=r.per100_added_med, per100_added_lo=r.per100_added_lo,
                              per100_added_hi=r.per100_added_hi, per100_added_pred_lo=r.per100_added_pred_lo,
                              per100_added_pred_hi=r.per100_added_pred_hi, af_med=r.af_med, af_lo=r.af_hz_lo,
                              af_hi=r.af_hz_hi, p_af_lt_half=r.p_af_lt_half, p_no_effect=r.p_no_effect, zone=r.zone))
    for i, r in enumerate(ex):
        cards.append(dict(card_set="fig1a", rank_drug=i + 1, rank_term=1, drug=r.drug, display_name=r.display_name,
                          term=r.term, n_trials=int(r.n_trials), per100_pl=r.per100_pl, per100_drug_med=r.per100_drug_med,
                          per100_added_med=r.per100_added_med, per100_added_lo=r.per100_added_lo,
                          per100_added_hi=r.per100_added_hi, per100_added_pred_lo=r.per100_added_pred_lo,
                          per100_added_pred_hi=r.per100_added_pred_hi, af_med=r.af_med, af_lo=r.af_hz_lo,
                          af_hi=r.af_hz_hi, p_af_lt_half=r.p_af_lt_half, p_no_effect=r.p_no_effect, zone=r.zone))
    cards = pd.DataFrame(cards)
    main_names = [names.get(x, x) for x in order[:6]]

    sm = pd.DataFrame(S.rows)
    g = lambda *k: S.get(*k)  # noqa: E731
    print(f"headline: exp_lt_half {g('headline','primary','all','exp_lt_half'):.3f}  conf_bg "
          f"{g('headline','primary','all','conf_bg'):.3f}  exp_no_effect {g('headline','primary','all','exp_no_effect'):.3f}  "
          f"conf_drug {g('headline','primary','all','conf_drug'):.3f}")
    for s in STRATA:
        print(f"  {s:>4}: n {int((lab.evidence == s).sum()):>6,}  exp {g('headline','primary',s,'exp_lt_half'):.3f}  "
              f"conf {g('headline','primary',s,'conf_bg'):.3f}")
    print(f"robustness exp_lt_half: {rob}  failed {rob_fail}")
    print(f"validity P {mw.pvalue:.2e}  median diff {med_diff:.3f}  AUC {g('validity','has_label','all','auc'):.3f}")
    print(f"organ systems {len(socs)}; classes {n_class}; cards {main_names}")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_cell','afstd_label_match','ref_drug_name','ref_chembl_drug',"
                                            "'af3_param','af_cell','af_label_match')").fetchall()]
        keep = ["drug", "term", "soc_h", "atc3", "atc1", "n_trials", "evidence", "zone", "has_fda_label", "on_fda",
                "sider_mapped", "on_sider", "label_source", "promiscuous", "af1_estimable", "psi_mean", "psi_med", "af_hz",
                "af_med", "af_hz_lo", "af_hz_hi", "af_risk12", "p_af_lt_half", "p_no_excess", "p_no_effect", "x0_own",
                "n0_own", "p0_own", "p0_12wk", "per100_pl", "per100_drug_med", "per100_drug_lo", "per100_drug_hi",
                "per100_added_med", "per100_added_lo", "per100_added_hi", "per100_added_pred_lo", "per100_added_pred_hi",
                "per100_added_ref_med", "rule_a", "rule_b", "rule_c"]
        out = d[keep].copy(); out["run_id"] = run.run_id
        run.archive_table("afstd_pair", out, upstream=up, mode="replace_all")
        sm["run_id"] = run.run_id
        run.archive_table("afstd_summary", sm, upstream=up, mode="replace_all")
        cards["run_id"] = run.run_id
        run.archive_table("afstd_card", cards, upstream=up, mode="replace_all")

        def det(analysis):
            x = sm[sm.analysis == analysis].drop(columns="run_id")
            return x.replace({np.nan: None}).to_dict("records")
        run.evaluate_gate("afstd_headline", g("headline", "primary", "all", "exp_lt_half"), None, detail=dict(rows=det("headline")))
        run.evaluate_gate("afstd_sensitivity", None, None, detail=dict(rows=det("sensitivity")))
        run.evaluate_gate("afstd_robustness", float(len(rob_fail)), not rob_fail,
                          detail=dict(exp_lt_half=rob, failed=rob_fail,
                                      wording="most" if not rob_fail else "a large share"))
        run.evaluate_gate("afstd_label_validity", float(mw.pvalue), bool(mw.pvalue < 1e-3 and med_diff > 0),
                          detail=dict(rows=det("validity"), median_diff=med_diff, n_listed=len(x1), n_unlisted=len(x0)))
        run.evaluate_gate("afstd_by_soc", float(len(socs)), None, detail=dict(rows=det("by_soc")))
        run.evaluate_gate("afstd_by_class", float(n_class), None, detail=dict(rows=det("by_class")))
        run.evaluate_gate("afstd_absolute", g("absolute", "primary", "all", "median_per100_added"), None,
                          detail=dict(rows=det("absolute")))
        run.evaluate_gate("afstd_listing_rules", g("listing_rules", "rule_b", "all", "retained"), None,
                          detail=dict(rows=det("listing_rules")))
        run.evaluate_gate("afstd_examples", float(len(order)), None,
                          detail=dict(fig1a=[f"{r.display_name} | {r.term}" for r in ex], cards_main=main_names,
                                      cards_supplementary=[names.get(x, x) for x in order[6:18]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
