"""Shared machinery for the attributable-fraction analyses.

Factored out of scripts 42/44 so later analyses do not re-implement the contrast-by-term
query, the estimators or the SIDER label mapping. The earlier scripts are left untouched
so their recorded SHA256 in prov_script_run stays valid.
"""
from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import provenance as pv

ROOT = Path(__file__).resolve().parents[2]
SIDER = ROOT / "cache" / "sider"

ROWS_SQL = """
WITH ct AS (
  SELECT c.contrast_id, c.nct, c.eg_active, c.eg_placebo, c.k, c.drug_ids[1] AS drug,
         c.n_act_other AS n1, c.n_pl_other AS n0, t.masking, t.frequency_threshold AS thr
  FROM atlas_contrast c JOIN ctgov_trial t USING (nct)
  WHERE c.single_ingredient AND c.n_act_other > 0 AND c.n_pl_other > 0),
ev AS (
  SELECT nct, eg_id, lower(trim(regexp_replace(term, '\\s+', ' ', 'g'))) AS term,
         max(num_affected) AS e
  FROM v_ctgov_ae WHERE kind = 'other' AND num_affected IS NOT NULL GROUP BY 1, 2, 3),
listed AS (SELECT DISTINCT nct, term FROM ev),
pl AS (
  SELECT ct.contrast_id, ev.term, sum(ev.e) / any_value(ct.k) AS c
  FROM ct, unnest(ct.eg_placebo) AS u(pid)
  JOIN ev ON ev.nct = ct.nct AND ev.eg_id = u.pid GROUP BY 1, 2)
SELECT ct.contrast_id, ct.nct, ct.drug, l.term, ct.n1, ct.n0, ct.masking, ct.thr,
       coalesce(ea.e, 0) AS a, coalesce(pl.c, 0) AS c
FROM ct JOIN listed l ON l.nct = ct.nct
LEFT JOIN ev ea ON ea.nct = ct.nct AND ea.eg_id = ct.eg_active AND ea.term = l.term
LEFT JOIN pl ON pl.contrast_id = ct.contrast_id AND pl.term = l.term
"""


def load_rows() -> pd.DataFrame:
    with pv.connect(read_only=True) as con:
        return con.execute(ROWS_SQL).df()


def mh_rr(g):
    a, c, n1, n0 = g.a.values, g.c.values, g.n1.values, g.n0.values
    T = n1 + n0
    R, S = (a * n0 / T).sum(), (c * n1 / T).sum()
    if R <= 0 or S <= 0:
        return np.nan, np.nan
    v = ((n1 * n0 * (a + c) - a * c * T) / T ** 2).sum() / (R * S)
    return float(np.log(R / S)), float(np.sqrt(v)) if v > 0 else np.nan


def dl_rr(g):
    """DerSimonian-Laird random-effects log-RR; 0.5 correction only for single zeros."""
    a, c, n1, n0 = [g[x].values.astype(float) for x in ("a", "c", "n1", "n0")]
    keep = (a + c) > 0
    a, c, n1, n0 = a[keep], c[keep], n1[keep], n0[keep]
    if len(a) < 2:
        return np.nan, np.nan
    z = (a == 0) | (c == 0)
    a, c = np.where(z, a + .5, a), np.where(z, c + .5, c)
    n1, n0 = np.where(z, n1 + 1, n1), np.where(z, n0 + 1, n0)
    y = np.log((a / n1) / (c / n0))
    v = 1 / a - 1 / n1 + 1 / c - 1 / n0
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    return float((ws * y).sum() / ws.sum()), float(np.sqrt(1 / ws.sum()))


def estimate(rows, min_trials=2, min_events=20, method="MH") -> pd.DataFrame:
    fn = mh_rr if method == "MH" else dl_rr
    out = []
    for (d, t), g in rows.groupby(["drug", "term"], sort=False):
        if g.nct.nunique() < min_trials or (g.a.sum() + g.c.sum()) < min_events:
            continue
        lr, se = fn(g)
        if lr != lr or se != se:
            continue
        lo, hi = np.exp(lr - 1.96 * se), np.exp(lr + 1.96 * se)
        out.append(dict(drug=d, term=t, n_trials=int(g.nct.nunique()), log_rr=lr, se=se,
                        rr=float(np.exp(lr)), af=float(1 - np.exp(-lr)),
                        af_lo=float(1 - 1 / lo), af_hi=float(1 - 1 / hi),
                        p=float(2 * stats.norm.sf(abs(lr / se)))))
    return pd.DataFrame(out)


def label_pairs():
    """SIDER PT-level (chembl_id, term) pairs and the set of mapped ChEMBL ids."""
    spec = importlib.util.spec_from_file_location("m34", ROOT / "scripts" / "34_ctgov_parse.py")
    m34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m34)
    idx, _ = m34.build_index()
    names = dict(l.split("\t") for l in (SIDER / "drug_names.tsv").read_text().splitlines())
    cid2 = {}
    for cid, nm in names.items():
        hit, how = m34.match(nm, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            cid2[cid] = hit
    label = set()
    for line in gzip.open(SIDER / "meddra_all_se.tsv.gz", "rt"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6 and f[3] == "PT" and f[0] in cid2:
            for c in cid2[f[0]]:
                label.add((c, f[5].strip().lower()))
    return label, {c for s in cid2.values() for c in s}


def h1_summary(cells: pd.DataFrame, label, mapped) -> dict:
    sm = cells[cells.drug.isin(mapped)].copy()
    sm["on_label"] = [(d, t) in label for d, t in zip(sm.drug, sm.term)]
    lab, non = sm[sm.on_label], sm[~sm.on_label]
    mw = stats.mannwhitneyu(lab.log_rr, non.log_rr, alternative="greater") \
        if len(lab) > 5 and len(non) > 5 else None
    return dict(n_label=int(len(lab)), n_nonlabel=int(len(non)),
                frac_afhi_lt_half=float((lab.af_hi < 0.5).mean()) if len(lab) else np.nan,
                frac_af_le_zero=float((lab.af <= 0).mean()) if len(lab) else np.nan,
                frac_af_lt_half=float((lab.af < 0.5).mean()) if len(lab) else np.nan,
                validity_p=float(mw.pvalue) if mw else np.nan,
                median_diff=float(lab.log_rr.median() - non.log_rr.median()) if mw else np.nan)


def gate_details(con) -> dict:
    """Gate-level summaries for figures, from the CURRENT pipeline version.

    Once gate af_v2 is evaluated, the result tables hold v2 rows, so figures must read the
    v2 summaries too -- mixing v1 gate numbers with v2 tables would put two pipeline
    versions on one panel. Returns the v1 key layout the figure scripts already use.
    """
    import json

    def det(name):
        r = con.execute("SELECT metric_detail, metric_value FROM prov_decision WHERE gate_name=?",
                        [name]).fetchone()
        return (json.loads(r[0]) if r and r[0] else {}), (r[1] if r else None)

    v2, _ = det("af_v2")
    if not v2:
        out = {g: det(g)[0] for g in ("af_reproducibility", "af_reproducibility_temporal",
                                      "af_robustness", "af_d2_by_soc", "af_h1_label", "af_fda_h1")}
        out["af_label_validity"] = {"p": det("af_label_validity")[1]}
        out["af_fda_h1"]["validity_p"] = det("af_fda_validity")[1]
        out["af_class"] = {"p": det("af_class")[1], **det("af_class")[0]}
        out["version"] = "v1"
        return out
    c = con.execute("SELECT c.af, c.af_hi, c.on_label, c.sider_mapped, m.on_fda, m.has_fda_label "
                    "FROM af_cell c JOIN af_label_match m USING (drug, term)").df()

    def fr(d):
        return dict(n=int(len(d)), af_hi_lt_half=float((d.af_hi < .5).mean()),
                    af_le_zero=float((d.af <= 0).mean()), af_lt_half=float((d.af < .5).mean()))
    s = c[c.sider_mapped]; f = c[c.has_fda_label]
    return {
        "version": "v2",
        "af_reproducibility": v2["reproducibility"]["random"],
        "af_reproducibility_temporal": v2["reproducibility"]["temporal"],
        "af_label_validity": {"p": v2["sider"]["validity_p"]},
        "af_robustness": v2["robustness"],
        "af_d2_by_soc": v2["d2_by_soc"],
        "af_h1_label": {"on_label": fr(s[s.on_label]), "not_on_label": fr(s[~s.on_label]),
                        "validity_p": v2["sider"]["validity_p"]},
        "af_fda_h1": {"fda_on_label": fr(f[f.on_fda]), "fda_not_on_label": fr(f[~f.on_fda]),
                      "jaccard_fda_sider": v2["fda"]["jaccard_fda_sider"],
                      "validity_p": v2["fda"]["validity_p"]},
        "af_class": {"p": v2["class_chi2_p"], "n_classes": v2["n_classes"]},
    }
