#!/usr/bin/env python3
"""Post-review analyses that need no refit (rules in 126_jno_review_prereg.py): jno_rev_classification,
jno_rev_positive_controls, jno_rev_scope, jno_rev_discontinuation_v2, jno_rev_descriptives, and the sample file of
jno_rev_label_audit (classified separately, then evaluated by 128b_jno_rev_label_audit.py)."""
import gzip
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af3  # noqa: E402
from oajoint import provenance as pv  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S110 = load(ROOT / "scripts" / "110_afstd_findings.py", "s110")
S120 = load(ROOT / "scripts" / "120_jno_discontinuation.py", "s120")
M109 = load(ROOT / "scripts" / "109_afstd_label_match.py", "m109")
S119 = load(ROOT / "scripts" / "119_jno_label_section.py", "s119")
REV = ROOT / "manuscript" / "jama" / "review"
SEED = 20260927
STRATA = ["1", "2-4", "5-9", ">=10"]
PC = [  # (label, ATC prefixes, term regex) fixed by rule before viewing estimates
    ("Urinary antimuscarinics: dry mouth", ("G04BD",), r"^dry mouth$"),
    ("Opioids: constipation", ("N02A",), r"^constipation$"),
    ("Opioids: nausea", ("N02A",), r"^nausea$"),
    ("Dihydropyridine CCBs: peripheral edema", ("C08CA",), r"^edema peripheral$"),
    ("GLP-1 receptor agonists: nausea", ("A10BJ",), r"^nausea$"),
    ("GLP-1 receptor agonists: vomiting", ("A10BJ",), r"^vomiting$"),
    ("SGLT2 inhibitors: genital mycotic infection", ("A10BK",),
     r"(mycotic|candid|fungal).*(genital|vulvovaginal|balan|penile)|(genital|vulvovaginal|balan|penile).*(mycotic|candid|fungal)"),
    ("ACE inhibitors: cough", ("C09AA",), r"^cough$"),
    ("Pregabalin/gabapentin: dizziness", ("N03AX12", "N03AX16", "N02BF"), r"^dizziness$"),
    ("Pregabalin/gabapentin: somnolence", ("N03AX12", "N03AX16", "N02BF"), r"^somnolence$"),
    ("Triptans: paresthesia", ("N02CC",), r"^paresthesia$"),
]


def summ(rows, analysis, set_name, grp, d, metrics):
    for m, v, lo, hi in metrics:
        rows.append(dict(analysis=analysis, set_name=set_name, grp=grp, n_pairs=int(len(d)),
                         n_drugs=int(d.drug.nunique()) if len(d) else 0, metric=m, value=float(v),
                         lo=float(lo) if lo == lo else np.nan, hi=float(hi) if hi == hi else np.nan))


def main() -> int:
    rng = np.random.default_rng(SEED)
    with pv.connect(read_only=True) as con:
        P = con.execute("""SELECT p.*, e.af_mean, e.per100_drug_exp, e.per100_added_exp FROM afstd_pair p
                           JOIN afstd_pair_exp e USING (drug, term)""").df()
        atc = {r[0]: [a for a in (r[1] or [])] for r in con.execute("SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
        conv = con.execute("SELECT drug, term, af, n_trials AS n_conv FROM af_cell").df()
        names = dict(con.execute("SELECT chembl_id, display_name FROM ref_drug_name").fetchall())
        comp = con.execute("SELECT drug, term, n_trials, p_af_lt_half FROM jno_cell_complete").df()
        sm = con.execute("SELECT * EXCLUDE (run_id) FROM afstd_summary").df()
        disc = con.execute("SELECT * EXCLUDE (run_id) FROM jno_discontinuation_contrast").df()
        enr = con.execute("SELECT nct, enrollment, conditions FROM ctgov_trial").df()
        sec = con.execute("SELECT drug, term, section_class FROM jno_label_section").df()
        pool = con.execute("SELECT generic_name, pool, adverse_reactions_text FROM fda_label_pool "
                           "ORDER BY generic_name, pool").df()
        dtr = con.execute("""SELECT DISTINCT list_extract(drug_ids, 1) AS drug, nct FROM atlas_contrast
                             WHERE single_ingredient""").df()
    L = P[P.on_fda].copy()
    L["atc"] = L.drug.map(lambda d: atc.get(d, []))
    rows, det = [], {}

    # ---------------------------------------------------------------- classification
    for grp, d in [("all", L)] + [(s, L[L.evidence == s]) for s in STRATA]:
        cb, cd = (d.p_af_lt_half > .975).mean(), (d.p_af_lt_half < .025).mean()
        summ(rows, "jno_rev_classification", "listed", grp, d, [
            ("conf_below_half", cb, np.nan, np.nan), ("conf_above_half", cd, np.nan, np.nan),
            ("uncertain", 1 - cb - cd, np.nan, np.nan), ("exp_no_excess", d.p_no_excess.mean(), np.nan, np.nan),
            ("conf_excess", (d.p_no_excess < .025).mean(), np.nan, np.nan),
            ("prob_no_excess_gt_half", (d.p_no_excess > .5).mean(), np.nan, np.nan),
            ("exp_no_effect", d.p_no_effect.mean(), np.nan, np.nan),
            ("added_ge1", (d.per100_added_exp >= 1).mean(), np.nan, np.nan)])
    cls = {r["metric"]: r["value"] for r in rows if r["grp"] == "all"}
    det["classification"] = cls

    # ---------------------------------------------------------------- positive controls
    pcs = []
    for lab, pre, rx in PC:
        r_ = re.compile(rx)
        d = P[(P.n_trials >= 2) & P.drug.map(lambda x: any(a.startswith(pre) for a in atc.get(x, []))) &
              P.term.map(lambda t: bool(r_.search(t)))].merge(conv, on=["drug", "term"], how="left")
        if not len(d):
            pcs.append(dict(control=lab, n_pairs=0)); continue
        pcs.append(dict(control=lab, n_pairs=len(d), drugs=", ".join(sorted({names.get(x, x) for x in d.drug})),
                        n_trials=int(d.n_trials.sum()), detected=float((d.p_no_excess < .025).mean()),
                        prob_lt_half=float((d.p_af_lt_half > .5).mean()), median_af=float(d.af_mean.median()),
                        median_conv_af=float(d.af.median()) if d.af.notna().any() else np.nan,
                        n_conv=int(d.af.notna().sum())))
    PCD = pd.DataFrame(pcs)
    allpc = pd.concat([P[(P.n_trials >= 2) & P.drug.map(lambda x: any(a.startswith(pre) for a in atc.get(x, []))) &
                         P.term.map(lambda t: bool(re.search(rx, t)))] for _, pre, rx in PC]).drop_duplicates(["drug", "term"])
    det["positive_controls"] = dict(n_pairs=len(allpc), detected=float((allpc.p_no_excess < .025).mean()),
                                    prob_lt_half=float((allpc.p_af_lt_half > .5).mean()),
                                    median_af=float(allpc.af_mean.median()), by_control=PCD.to_dict("records"))
    summ(rows, "jno_rev_positive_controls", "all", "all", allpc, [
        ("detected", det["positive_controls"]["detected"], np.nan, np.nan),
        ("prob_lt_half", det["positive_controls"]["prob_lt_half"], np.nan, np.nan),
        ("median_af", det["positive_controls"]["median_af"], np.nan, np.nan)])
    print(PCD.to_string())

    # ---------------------------------------------------------------- scope
    cond = enr.set_index("nct").conditions.to_dict()
    dcond = defaultdict(set)
    for d_, n_ in zip(dtr.drug, dtr.nct):
        for c in (cond.get(n_) if cond.get(n_) is not None else []):
            dcond[d_].add(str(c).lower())
    L["indication_match"] = [any(re.search(r"\b" + re.escape(t) + r"\b", c) for c in dcond.get(d, ())) for d, t in
                             zip(L.drug, L.term)]
    L["vacc_suppl"] = L.atc.map(lambda a: any(x.startswith(("J07", "A11", "A12")) for x in a))
    for nm, keep in (("excl_vaccines_supplements", ~L.vacc_suppl), ("excl_indication_terms", ~L.indication_match),
                     ("excl_both", ~L.vacc_suppl & ~L.indication_match)):
        d = L[keep]
        lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
        summ(rows, "jno_rev_scope", nm, "all", d, [("exp_lt_half", d.p_af_lt_half.mean(), lo, hi)])
    det["scope"] = dict(n_vacc_suppl=int(L.vacc_suppl.sum()), n_indication=int(L.indication_match.sum()),
                        examples_indication=L[L.indication_match].assign(dn=lambda x: x.drug.map(names))
                        .sort_values("af_mean")[["dn", "term"]].head(12).values.tolist())

    # ---------------------------------------------------------------- discontinuation v2
    D = disc.merge(enr[["nct", "enrollment"]], on="nct", how="left")
    D["atc_all"] = D.drug.map(lambda d: atc.get(d, []))
    rep = D.groupby("nct").apply(lambda g: (g.x_drug.sum() + g.x_pl.sum()) > 0, include_groups=False)
    keep = D.nct.map(rep).astype(bool) & (D.enrollment.fillna(0) < 50000) & ~D.atc_all.map(
        lambda a: any(x.startswith("J07") for x in a))
    Dv = D[keep]

    def mh_q(d):
        m = S120.mh(d)
        z = d[(d.x_drug > 0) & (d.x_pl > 0) & (d.x_drug < d.n_drug) & (d.x_pl < d.n_pl)]
        lr = np.log((z.x_drug / z.n_drug) / (z.x_pl / z.n_pl))
        v = 1 / z.x_drug - 1 / z.n_drug + 1 / z.x_pl - 1 / z.n_pl
        w = 1 / v
        Q = float((w * (lr - np.log(m["rr"])) ** 2).sum()); dfq = len(z) - 1
        m.update(Q=Q, df=dfq, I2=float(max(0, (Q - dfq) / Q)) if Q > 0 else 0.0, n_q=int(len(z)))
        return m
    o2, o1 = mh_q(Dv), mh_q(D)
    det["discontinuation_v2"] = dict(restricted=o2, all_contrasts=o1, n_excluded_trials=int(D.nct.nunique() - Dv.nct.nunique()))
    by = {}
    Dv = Dv.assign(atc1=Dv.atc1.fillna("?"))
    for g, d in Dv.groupby("atc1"):
        if d.nct.nunique() >= 10:
            by[g] = mh_q(d)
    det["discontinuation_v2"]["by_atc1"] = by
    for nm, o in (("restricted", o2), ("all", o1)):
        summ(rows, "jno_rev_discontinuation_v2", nm, "all", pd.DataFrame({"drug": Dv.drug if nm == "restricted" else D.drug}),
             [("rr", o["rr"], o["rr_lo"], o["rr_hi"]), ("rd_per100", o["rd"], o["rd_lo"], o["rd_hi"]),
              ("ef", o["ef"], o["ef_lo"], o["ef_hi"]), ("I2", o["I2"], np.nan, np.nan)])
    print("discontinuation v2:", {k: round(v, 3) if isinstance(v, float) else v for k, v in o2.items()})

    # ---------------------------------------------------------------- descriptives
    s4 = sm[(sm.analysis == "sensitivity") & (sm.set_name == "S4") & (sm.metric == "risk_mean_lt_half")].value.iloc[0]
    cm = comp.merge(L[["drug", "term"]], on=["drug", "term"])
    cm["ev"] = S110.evidence(cm.n_trials.values)
    cstrata = {s: dict(n=int((cm.ev == s).sum()), exp=float(cm[cm.ev == s].p_af_lt_half.mean()),
                       conf=float((cm[cm.ev == s].p_af_lt_half > .975).mean())) for s in STRATA}
    d10 = L[L.evidence == ">=10"].merge(conv, on=["drug", "term"], how="left")
    dense = dict(n=len(d10), model_exp=float(d10.p_af_lt_half.mean()), n_conv=int(d10.af.notna().sum()),
                 conv_share_lt_half=float((d10.af.dropna() < .5).mean()))
    zero1 = float(((L.evidence == "1") & (L.x0_own == 0)).mean())
    # label-printed placebo frequencies (SIDER 4.1 meddra_freq; logic of 68_label_placebo_info.py)
    M68 = load(ROOT / "scripts" / "68_label_placebo_info.py", "m68")
    SIDER = ROOT / "cache" / "sider"
    labfreq = {}
    if (SIDER / "meddra_freq.tsv.gz").exists():
        idx, _ = M68.M34.build_index()
        snames = dict(l.split("\t") for l in (SIDER / "drug_names.tsv").read_text().splitlines())
        cid2 = {}
        for cid, nm in snames.items():
            hit, how = M68.M34.match(nm, [], idx)
            if hit and how in ("exact", "research_code", "combination_parts"):
                cid2[cid] = hit
        fr = defaultdict(lambda: {"drug": [], "pl": []})
        for l in gzip.open(SIDER / "meddra_freq.tsv.gz", "rt"):
            f = l.rstrip("\n").split("\t")
            if len(f) < 10 or f[7] != "PT" or f[0] not in cid2:
                continue
            try:
                v = (float(f[5]) + float(f[6])) / 2
            except ValueError:
                continue
            for c in cid2[f[0]]:
                fr[(c, M68.M50.us(f[9].strip().lower()))]["pl" if f[3] == "placebo" else "drug"].append(v)
        S_ = P[P.on_sider].copy()
        S_["both"] = [bool(fr.get((d, t), {}).get("drug")) and bool(fr.get((d, t), {}).get("pl")) for d, t in zip(S_.drug, S_.term)]
        B_ = S_[S_.both].copy()
        B_["label_af"] = [1 - np.median(fr[(d, t)]["pl"]) / np.median(fr[(d, t)]["drug"]) if np.median(fr[(d, t)]["drug"]) > 0
                          else np.nan for d, t in zip(B_.drug, B_.term)]
        B_ = B_.dropna(subset=["label_af"])
        labfreq = dict(n_sider_listed=int(len(S_)), share_with_placebo_freq=float(S_.both.mean()), n_both=int(len(B_)),
                       median_label_af=float(B_.label_af.median()), median_model_af=float(B_.af_mean.median()),
                       spearman=float(stats.spearmanr(B_.label_af, B_.af_mean).statistic),
                       model_share_lt_half=float(B_.p_af_lt_half.mean()), label_share_lt_half=float((B_.label_af < .5).mean()))
    with pv.connect(read_only=True) as con:
        arms, K, J = af3.build(con)
    T = K.drop_duplicates("nct")
    denom = dict(n_trials=int(len(T)), followup=int(T.wk.notna().sum()), age=int(T.age.notna().sum()),
                 sex=int(T.pct_female.notna().sum()))
    det["descriptives"] = dict(risk_scale_share=float(s4), complete_by_stratum=cstrata, dense=dense,
                               single_trial_zero_placebo=zero1, label_frequencies=labfreq, table1_denominators=denom)
    print(json.dumps(det["descriptives"], indent=1))

    # ---------------------------------------------------------------- label audit sample (400, stratified by class)
    REV.mkdir(parents=True, exist_ok=True)
    Ls = L.merge(sec, on=["drug", "term"])
    shares = Ls.section_class.value_counts(normalize=True)
    nper = (shares * 400).round().astype(int)
    samp = pd.concat([Ls[Ls.section_class == c].sample(n=int(k), random_state=20261013) for c, k in nper.items() if k > 0])
    idx_, _ = M109.M34.build_index()
    text, src, gen = M109.pool_text(pool, idx_)
    ctx = []
    for r in samp.itertuples():
        t = text.get(r.drug, "")
        pat = M109.pattern(r.term)
        m = pat.search(t)
        c = re.sub(r"\s+", " ", t[max(0, m.start() - 160): m.end() + 160]) if m else ""
        ctx.append(dict(drug=r.drug, drug_name=names.get(r.drug, r.drug), term=r.term, section_class=r.section_class,
                        af_mean=round(r.af_mean, 3), context=c))
    pd.DataFrame(ctx).to_csv(REV / "label_audit_sample.csv", index=False)
    print("audit sample", len(ctx), nper.to_dict())

    # ---------------------------------------------------------------- archive and evaluate
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','afstd_pair_exp','af_cell','jno_cell_complete','afstd_summary',"
                                            "'jno_discontinuation_contrast','ctgov_trial','jno_label_section','fda_label_pool')")
                  .fetchall()]
        sm_ = pd.DataFrame(rows); sm_["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis LIKE 'jno_rev_%' AND analysis NOT IN "
                        "('jno_rev_label_aware','jno_rev_bootstrap','jno_rev_refit_phase','jno_rev_refit_shared',"
                        "'jno_rev_sim_nozero','jno_rev_temporal_calibration','jno_rev_label_audit')")
        run.archive_table("jno_summary", sm_, upstream=up)
        run.archive_document("jama_label_audit_sample", REV / "label_audit_sample.csv", kind="csv", upstream=up)
        run.evaluate_gate("jno_rev_classification", cls["conf_excess"], None, detail=cls)
        run.evaluate_gate("jno_rev_positive_controls", det["positive_controls"]["detected"], None,
                          detail=det["positive_controls"])
        run.evaluate_gate("jno_rev_scope", float(L[~L.vacc_suppl & ~L.indication_match].p_af_lt_half.mean()), None,
                          detail=det["scope"])
        run.evaluate_gate("jno_rev_discontinuation_v2", o2["rr"], None, detail=det["discontinuation_v2"])
        run.evaluate_gate("jno_rev_descriptives", None, None, detail=det["descriptives"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
