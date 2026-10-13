#!/usr/bin/env python3
"""Conventional analysis, v2 (gate af_v2): every analysis recomputed with
  (A) MedDRA terms canonicalised to US spelling before grouping, and
  (B) 'on the current label' = listed on ANY current innovator label of the ingredient.
Pass criteria are copied verbatim from v1. v1 is frozen in db/parquet/af_v1_snapshot.
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af_common as af  # noqa: E402
from oajoint import provenance as pv  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M50 = load(ROOT / "scripts" / "50_af_fda_analysis.py", "m50")
M34 = load(ROOT / "scripts" / "34_ctgov_parse.py", "m34")
us, pattern = M50.us, M50.pattern
SETS = json.loads((ROOT / "config" / "af_symptom_sets.json").read_text())


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def canon_rows():
    rows = af.load_rows()
    cmap = {t: us(t) for t in rows.term.unique()}
    rows["term"] = rows.term.map(cmap)
    keys = ["contrast_id", "nct", "drug", "term", "n1", "n0", "masking", "thr"]
    rows = rows.groupby(keys, dropna=False, as_index=False).agg(a=("a", "max"), c=("c", "max"))
    return rows


def full_cells(rows):
    out = []
    for (d, t), g in rows.groupby(["drug", "term"], sort=False):
        if g.nct.nunique() < 2 or (g.a.sum() + g.c.sum()) < 20:
            continue
        lr, se = af.mh_rr(g)
        if lr != lr or se != se:
            continue
        lo, hi = np.exp(lr - 1.96 * se), np.exp(lr + 1.96 * se)
        out.append(dict(drug=d, term=t, n_trials=int(g.nct.nunique()),
                        events_drug=float(g.a.sum()), n_drug=float(g.n1.sum()),
                        events_pl=float(g.c.sum()), n_pl=float(g.n0.sum()),
                        risk_drug=float(g.a.sum() / g.n1.sum()), risk_pl=float(g.c.sum() / g.n0.sum()),
                        rr=float(np.exp(lr)), rr_lo=float(lo), rr_hi=float(hi), log_rr=lr,
                        af=float(1 - np.exp(-lr)), af_lo=float(1 - 1 / lo), af_hi=float(1 - 1 / hi),
                        p=float(2 * stats.norm.sf(abs(lr / se)))))
    c = pd.DataFrame(out); c["fdr"] = bh(c.p.values); return c


def repro(rows, hmap):
    r = rows.assign(half=rows.nct.map(hmap)).dropna(subset=["half"])
    est, out = {}, []
    for (h, d, t), g in r.groupby(["half", "drug", "term"], sort=False):
        if g.nct.nunique() < 2 or (g.a.sum() + g.c.sum()) < 20:
            continue
        lr, se = af.mh_rr(g)
        if lr != lr or se != se:
            continue
        p = float(2 * stats.norm.sf(abs(lr / se)))
        est[(h, d, t)] = (lr, p)
        out.append(dict(half=h, drug=d, term=t, n_trials=int(g.nct.nunique()),
                        events_drug=float(g.a.sum()), n_drug=float(g.n1.sum()),
                        events_pl=float(g.c.sum()), n_pl=float(g.n0.sum()), log_rr=lr, se=se, p=p))
    both = sorted({k[1:] for k in est if k[0] == "A"} & {k[1:] for k in est if k[0] == "B"})
    A = np.array([est[("A",) + c][0] for c in both]); B = np.array([est[("B",) + c][0] for c in both])
    sigA = [c for c in both if est[("A",) + c][1] < 0.01]
    agree = float(np.mean([np.sign(est[("A",) + c][0]) == np.sign(est[("B",) + c][0]) for c in sigA])) if sigA else np.nan
    rho = float(stats.spearmanr(A, B).statistic)
    ok = len(both) >= 200 and len(sigA) >= 30 and rho >= .5 and agree >= .8
    return dict(n_cells_both=len(both), spearman=rho, n_sigA=len(sigA), sign_agreement=agree, passed=bool(ok)), out


def main() -> int:
    rows = canon_rows()
    print(f"canonical rows {len(rows):,}; terms {rows.term.nunique():,}")
    with pv.connect(read_only=True) as con:
        start = con.execute("SELECT nct, start_date FROM ctgov_trial").df()
        pool = con.execute("SELECT generic_name, pool, adverse_reactions_text FROM fda_label_pool").df()
        atc = {r[0]: list(r[1] or []) for r in con.execute("SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
        ntr = dict(con.execute("SELECT drug_ids[1], count(DISTINCT nct) FROM atlas_contrast "
                               "WHERE single_ingredient GROUP BY 1").fetchall())
        soc = con.execute("""SELECT lower(trim(regexp_replace(term,'\\s+',' ','g'))) AS t, organ_system, count(*) AS n
                             FROM v_ctgov_ae WHERE kind='other' GROUP BY 1,2""").df()
    fail = []
    # ---- (1) reproducibility -----------------------------------------------------------
    trials = np.array(sorted(rows.nct.unique()))
    rng = np.random.default_rng(20260924)
    hr = dict(zip(trials, np.where(rng.random(len(trials)) < .5, "A", "B")))
    start["d"] = pd.to_datetime(start.start_date, errors="coerce")
    med = start[start.nct.isin(trials)].d.median()
    ht = {n: ("A" if (d == d and d <= med) else "B") for n, d in zip(start.nct, start.d) if n in hr}
    split_rows, rep = [], {}
    for name, hm in (("random", hr), ("temporal", ht)):
        s, out = repro(rows, hm); rep[name] = s
        split_rows += [dict(o, split=name) for o in out]
        print(f"[{name}] cells {s['n_cells_both']:,}  rho {s['spearman']:.3f}  sigA {s['n_sigA']}  "
              f"agree {s['sign_agreement']:.1%}  -> {'PASS' if s['passed'] else 'FAIL'}")
        if not s["passed"]:
            fail.append(f"reproducibility_{name}")
    # ---- atlas + SIDER ---------------------------------------------------------------
    cells = full_cells(rows)
    label_raw, mapped = af.label_pairs()
    sider = {(c, us(t)) for c, t in label_raw}
    cells["sider_mapped"] = cells.drug.isin(mapped)
    cells["on_label"] = [(d, t) in sider for d, t in zip(cells.drug, cells.term)]
    print(f"\natlas v2: {len(cells):,} cells, {cells.drug.nunique():,} drugs, {cells.term.nunique():,} terms")
    s1 = af.h1_summary(cells, sider, mapped)
    print(f"SIDER: label {s1['n_label']:,}  AF_hi<.5 {s1['frac_afhi_lt_half']:.1%}  AF<=0 {s1['frac_af_le_zero']:.1%}  "
          f"validity P {s1['validity_p']:.1e}")
    if not (s1["validity_p"] < 1e-3 and s1["median_diff"] > 0):
        fail.append("validity_sider")
    # ---- current labels (pool) -------------------------------------------------------
    idx, _ = M34.build_index()
    # rule B: union of ALL innovator labels of the ingredient (any salt/route/formulation);
    # only an ingredient with no innovator label at all falls back to the all-labels pool
    text = defaultdict(lambda: {"innovator": [], "all_labels": []})
    for g, pl_, t in zip(pool.generic_name, pool.pool, pool.adverse_reactions_text):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                text[c][pl_].append(t or "")
    src = {c: ("innovator" if v["innovator"] else "all_labels") for c, v in text.items()}
    text = {c: "\n\n".join(v[src[c]]) for c, v in text.items()}
    print(f"ingredients with pooled label text: {len(text):,} "
          f"({sum(x == 'innovator' for x in src.values()):,} innovator, "
          f"{sum(x == 'all_labels' for x in src.values()):,} all-label fallback)")
    pats = {t: pattern(t) for t in cells.term.unique()}
    cells["has_fda"] = cells.drug.isin(text)
    cells["on_fda"] = [bool(d in text and pats[t].search(text[d])) for d, t in zip(cells.drug, cells.term)]
    fc = cells[cells.has_fda]
    mw = stats.mannwhitneyu(fc[fc.on_fda].log_rr, fc[~fc.on_fda].log_rr, alternative="greater")
    dfd = float(fc[fc.on_fda].log_rr.median() - fc[~fc.on_fda].log_rr.median())
    fda = dict(n=int(fc.on_fda.sum()), af_hi_lt_half=float((fc[fc.on_fda].af_hi < .5).mean()),
               af_le_zero=float((fc[fc.on_fda].af <= 0).mean()), af_lt_half=float((fc[fc.on_fda].af < .5).mean()),
               validity_p=float(mw.pvalue), median_diff=dfd,
               n_not=int((~fc.on_fda).sum()), not_af_hi_lt_half=float((fc[~fc.on_fda].af_hi < .5).mean()),
               not_af_le_zero=float((fc[~fc.on_fda].af <= 0).mean()))
    both = cells[cells.has_fda & cells.sider_mapped]
    a_, b_ = set(both[both.on_fda].index), set(both[both.on_label].index)
    fda["jaccard_fda_sider"] = len(a_ & b_) / max(len(a_ | b_), 1)
    print(f"CURRENT labels (pool): drugs {cells[cells.has_fda].drug.nunique():,}; label {fda['n']:,}  "
          f"AF_hi<.5 {fda['af_hi_lt_half']:.1%}  AF<=0 {fda['af_le_zero']:.1%}  validity P {fda['validity_p']:.1e}  "
          f"Jaccard vs SIDER {fda['jaccard_fda_sider']:.2f}")
    if not (fda["validity_p"] < 1e-3 and dfd > 0):
        fail.append("validity_fda")
    # ---- robustness --------------------------------------------------------------------
    blinded = rows.masking.isin(["DOUBLE", "TRIPLE", "QUADRUPLE"]); complete = rows.thr.fillna(-1) == 0
    rob = {"R1 blinded only": af.estimate(rows[blinded]),
           "R2 complete reporting (threshold 0)": af.estimate(rows[complete]),
           "R3 >=3 trials, >=30 events": af.estimate(rows, 3, 30),
           "R4 DerSimonian-Laird": af.estimate(rows, method="DL")}
    robres = {"MAIN (reference)": s1}
    for k, c in rob.items():
        s = af.h1_summary(c, sider, mapped); s["n_cells"] = len(c); robres[k] = s
        ok = s["frac_afhi_lt_half"] >= .35 and s["frac_af_le_zero"] >= .20 and s["validity_p"] < 1e-3
        print(f"  {k:<38} label {s['n_label']:>5,}  AF_hi<.5 {s['frac_afhi_lt_half']:.1%}  AF<=0 "
              f"{s['frac_af_le_zero']:.1%}  P {s['validity_p']:.1e}  {'ok' if ok else 'FAIL'}")
        if not ok:
            fail.append(f"robust_{k.split()[0]}")
    # ---- D2, classes, probability table ------------------------------------------------
    soc["t"] = soc.t.map(us)
    soc = soc.groupby(["t", "organ_system"], dropna=False, as_index=False).n.sum()
    socmap = (soc.sort_values("n", ascending=False).drop_duplicates("t").set_index("t").organ_system
              .map(lambda s: "general disorders" if (s or "").lower().startswith("general disorders") else (s or "").lower()))
    lab = cells[cells.on_label].copy(); lab["soc"] = lab.term.map(socmap)
    d2 = (lab.groupby("soc").agg(n=("af", "size"), median_af=("af", "median"),
                                 frac_afhi_lt_half=("af_hi", lambda x: float((x < .5).mean()))).query("n >= 30")
          .sort_values("median_af").reset_index())
    names = {}
    for fpath in sorted((ROOT / "cache" / "chembl").glob("drug_*.json")):
        for x in json.loads(fpath.read_text())["drugs"]:
            for a in x.get("atc_classification") or []:
                parts = (a.get("description") or "").split(": ")
                if len(a.get("code", "")) >= 4 and len(parts) >= 3:
                    names.setdefault(a["code"][:4], parts[2])
    lf = fc[fc.on_fda]; by = defaultdict(list)
    for i, d in zip(lf.index, lf.drug):
        for c3 in {a[:4] for a in atc.get(d, [])}:
            by[c3].append(i)
    cls = pd.DataFrame([dict(atc3=c3, class_label=names.get(c3, ""), n_label_cells=len(ix),
                             frac_afhi_lt_half=float((lf.loc[ix].af_hi < .5).mean()),
                             frac_af_le_zero=float((lf.loc[ix].af <= 0).mean()),
                             median_af=float(lf.loc[ix].af.median())) for c3, ix in by.items() if len(ix) >= 20])
    k = (cls.frac_afhi_lt_half * cls.n_label_cells).round().astype(int)
    chi_p = float(stats.chi2_contingency(np.vstack([k, cls.n_label_cells - k])).pvalue)
    nm = {}
    for g in pool.generic_name:
        hit, how = M34.match(g, [], idx)
        for c in (hit or []):
            nm.setdefault(c, g)
    top = sorted(((ntr.get(d, 0), d) for d in fc.drug.unique()), reverse=True)[:20]
    pt = pd.DataFrame([dict(chembl_id=d, drug_name=nm.get(d, d), n_trials_drug=int(n), term=r.term,
                            n_trials=int(r.n_trials), risk_drug=r.risk_drug, risk_pl=r.risk_pl,
                            af=r.af, af_lo=r.af_lo, af_hi=r.af_hi)
                       for n, d in top for r in fc[(fc.drug == d) & fc.on_fda].itertuples()])
    print(f"\nclasses {len(cls)} (chi2 P {chi_p:.1e}); probability table {pt.chembl_id.nunique()} drugs, {len(pt)} rows")
    print(f"\nv2 criteria failed: {fail or 'none'}  -> af_v2 {'PASS' if not fail else 'FAIL'}")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('atlas_contrast','fda_label_pool','ref_chembl_drug')").fetchall()]
        sp = pd.DataFrame(split_rows)
        keep_cell = ["drug", "term", "n_trials", "events_drug", "n_drug", "events_pl", "n_pl", "risk_drug",
                     "risk_pl", "rr", "rr_lo", "rr_hi", "af", "af_lo", "af_hi", "p", "fdr", "sider_mapped", "on_label"]
        m = cells[["drug", "term", "on_fda", "on_label", "has_fda"]].rename(
            columns={"on_label": "on_sider", "has_fda": "has_fda_label"})
        m["fda_generic"] = [nm.get(d) for d in m.drug]; m["fda_set_id"] = None
        for df in (sp, cells, m, cls, pt):
            df["run_id"] = run.run_id
        run.archive_table("af_split_estimate", sp, upstream=up, mode="replace_all")
        run.archive_table("af_cell", cells[keep_cell + ["run_id"]], upstream=up, mode="replace_all")
        run.archive_table("af_label_match", m, upstream=up, mode="replace_all")
        run.archive_table("af_class_summary", cls, upstream=up, mode="replace_all")
        run.archive_table("af_probability", pt, upstream=up, mode="replace_all")
        run.evaluate_gate("af_v2", float(len(fail)), not fail, detail={
            "failed": fail, "reproducibility": rep, "sider": s1, "fda": fda, "robustness": robres,
            "d2_by_soc": json.loads(d2.to_json(orient="records")), "class_chi2_p": chi_p,
            "n_classes": int(len(cls)), "prob_table_drugs": int(pt.chembl_id.nunique()),
            "atlas": {"n_cells": int(len(cells)), "n_drugs": int(cells.drug.nunique()),
                      "n_terms": int(cells.term.nunique()),
                      "n_harm": int(((cells.rr > 1) & (cells.fdr < .05)).sum()),
                      "n_protective": int(((cells.rr < 1) & (cells.fdr < .05)).sum())}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
