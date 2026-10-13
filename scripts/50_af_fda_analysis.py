#!/usr/bin/env python3
"""Current-label H1, drug classes and the probability table (gates af_fda_validity,
af_fda_h1, af_class, af_probability_table), under the matching rule of af_fda_labels
(amended before any matching to allow plurals and the full British->US spelling set).

Order: validity is evaluated BEFORE the H1 fractions are reported as a finding.
"""
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
from oajoint import af_common as af  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

US_RULES = [(r"aemia", "emia"), (r"aem", "em"), (r"oea", "ea"), (r"oedema", "edema"),
            (r"oesoph", "esoph"), (r"paed", "ped"), (r"foet", "fet"), (r"faec", "fec"),
            (r"oestr", "estr"), (r"our\b", "or"), (r"sulph", "sulf")]


def us(term):
    t = term
    for a, b in US_RULES:
        t = re.sub(a, b, t)
    return t


def pattern(term):
    """Word-bounded, flexible whitespace, optional plural on the final word, UK or US."""
    forms = {term, us(term)}
    pats = []
    for f in forms:
        words = [re.escape(w) for w in f.split()]
        if not words:
            continue
        words[-1] = words[-1] + r"(?:e?s)?"
        pats.append(r"\b" + r"\s+".join(words) + r"\b")
    return re.compile("|".join(pats), re.I)


def rank(app, eff):
    a = (app or "").upper()
    return (0 if a.startswith(("NDA", "BLA")) else 1, -(int(eff) if (eff or "").isdigit() else 0))


def main() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("m34", ROOT / "scripts" / "34_ctgov_parse.py")
    m34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m34)
    idx, _ = m34.build_index()
    with pv.connect(read_only=True) as con:
        cells = con.execute("SELECT drug, term, n_trials, risk_drug, risk_pl, rr, af, af_lo, af_hi, "
                            "on_label AS on_sider, sider_mapped FROM af_cell").df()
        labels = con.execute("SELECT generic_name, set_id, application_number, effective_time, "
                             "adverse_reactions_text FROM fda_label").df()
        atc = {r[0]: list(r[1] or []) for r in con.execute(
            "SELECT chembl_id, atc4 FROM ref_chembl_drug").fetchall()}
        ntr = dict(con.execute("""SELECT drug_ids[1], count(DISTINCT nct) FROM atlas_contrast
                                  WHERE single_ingredient GROUP BY 1""").fetchall())
    cells["log_rr"] = np.log(cells.rr)
    # ---- map labels to ChEMBL, one label per ingredient -------------------------------
    per_drug = {}
    for r in labels.itertuples():
        hit, how = m34.match(r.generic_name, [], idx)
        if not hit or how not in ("exact", "research_code", "combination_parts"):
            continue
        for c in hit:
            if c not in per_drug or rank(r.application_number, r.effective_time) < \
                    rank(per_drug[c].application_number, per_drug[c].effective_time):
                per_drug[c] = r
    print(f"FDA labels: {len(labels):,} generics; mapped to {len(per_drug):,} ChEMBL ingredients; "
          f"atlas drugs with a current label: {cells.drug.isin(per_drug).sum() and cells[cells.drug.isin(per_drug)].drug.nunique():,} "
          f"of {cells.drug.nunique():,}")
    # ---- match every estimable term against that drug's label ------------------------
    pats = {t: pattern(t) for t in cells.term.unique()}
    on_fda, has = [], []
    for d, t in zip(cells.drug, cells.term):
        lab = per_drug.get(d)
        has.append(lab is not None)
        on_fda.append(bool(lab is not None and pats[t].search(lab.adverse_reactions_text or "")))
    cells["on_fda"], cells["has_fda"] = on_fda, has
    fc = cells[cells.has_fda]
    print(f"estimable cells for labelled drugs: {len(fc):,};  on current label: {fc.on_fda.sum():,}")

    # ---- validity FIRST ---------------------------------------------------------------
    lab_, non = fc[fc.on_fda].log_rr, fc[~fc.on_fda].log_rr
    mw = stats.mannwhitneyu(lab_, non, alternative="greater")
    diff = float(lab_.median() - non.median())
    valid = bool(mw.pvalue < 0.001 and diff > 0)
    print(f"\nVALIDITY (current labels): median log-RR {lab_.median():+.3f} vs {non.median():+.3f}; "
          f"one-sided P = {mw.pvalue:.1e}  -> {'PASS' if valid else 'FAIL'}")

    def fr(d):
        return dict(n=int(len(d)), harm_signal=None, af_lt_half=float((d.af < .5).mean()),
                    af_hi_lt_half=float((d.af_hi < .5).mean()), af_le_zero=float((d.af <= 0).mean()))
    h1 = {"fda_on_label": fr(fc[fc.on_fda]), "fda_not_on_label": fr(fc[~fc.on_fda])}
    both = cells[cells.has_fda & cells.sider_mapped]
    sf, ss = set(both[both.on_fda].index), set(both[both.on_sider].index)
    jac = len(sf & ss) / max(len(sf | ss), 1)
    h1["listed_in_both"] = fr(both[both.on_fda & both.on_sider])
    h1["jaccard_fda_sider"] = jac
    print(f"\nH1 on CURRENT labels {'(finding)' if valid else '(NOT a finding)'}:")
    print(f"  {'':<18}{'n':>7}{'AF_hi<.5':>10}{'AF<=0':>8}{'AF<.5':>8}")
    for k in ("fda_on_label", "fda_not_on_label", "listed_in_both"):
        v = h1[k]
        print(f"  {k:<18}{v['n']:>7,}{v['af_hi_lt_half']:>10.1%}{v['af_le_zero']:>8.1%}{v['af_lt_half']:>8.1%}")
    print(f"  overlap of current (FDA) and 2016 (SIDER) label sets: Jaccard = {jac:.2f}")

    # ---- drug classes -----------------------------------------------------------------
    rows = []
    lab_cells = fc[fc.on_fda]
    by = defaultdict(list)
    for i, d in zip(lab_cells.index, lab_cells.drug):
        for c in {a[:4] for a in atc.get(d, [])}:
            by[c].append(i)
    import json as _j
    names = {}
    for fpath in sorted((ROOT / "cache" / "chembl").glob("drug_*.json")):
        for x in _j.loads(fpath.read_text())["drugs"]:
            for a in x.get("atc_classification") or []:
                parts = (a.get("description") or "").split(": ")
                if len(a.get("code", "")) >= 4 and len(parts) >= 3:
                    names.setdefault(a["code"][:4], parts[2])
    for c, ix in by.items():
        d = lab_cells.loc[ix]
        if len(d) >= 20:
            rows.append(dict(atc3=c, class_label=names.get(c, ""), n_label_cells=len(d),
                             frac_afhi_lt_half=float((d.af_hi < .5).mean()),
                             frac_af_le_zero=float((d.af <= 0).mean()), median_af=float(d.af.median())))
    cls = pd.DataFrame(rows).sort_values("frac_afhi_lt_half", ascending=False)
    k = (cls.frac_afhi_lt_half * cls.n_label_cells).round().astype(int)
    chi = stats.chi2_contingency(np.vstack([k, cls.n_label_cells - k]))
    print(f"\nDRUG CLASSES (ATC-3, >=20 current-label cells): {len(cls)} classes; "
          f"heterogeneity chi-square P = {chi.pvalue:.1e}")
    print(cls.head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("  ...")
    print(cls.tail(5).to_string(index=False, header=False, float_format=lambda v: f"{v:.2f}"))

    # ---- probability table: 20 drugs with the most trials, fixed in advance -----------
    top = sorted(((ntr.get(d, 0), d) for d in fc.drug.unique()), reverse=True)[:20]
    prob = []
    for n, d in top:
        lab = per_drug.get(d)
        for r in fc[(fc.drug == d) & fc.on_fda].itertuples():
            prob.append(dict(chembl_id=d, drug_name=lab.generic_name if lab is not None else d,
                             n_trials_drug=int(n), term=r.term, n_trials=int(r.n_trials),
                             risk_drug=float(r.risk_drug), risk_pl=float(r.risk_pl),
                             af=float(r.af), af_lo=float(r.af_lo), af_hi=float(r.af_hi)))
    pt = pd.DataFrame(prob)
    print(f"\nPROBABILITY TABLE: {pt.chembl_id.nunique()} drugs, {len(pt)} label-listed side effects")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('af_cell','fda_label','atlas_contrast')").fetchall()]
        m = cells[["drug", "term", "on_fda", "on_sider", "has_fda"]].rename(columns={"has_fda": "has_fda_label"})
        m["fda_generic"] = [per_drug[d].generic_name if d in per_drug else None for d in m.drug]
        m["fda_set_id"] = [per_drug[d].set_id if d in per_drug else None for d in m.drug]
        for df in (m, cls, pt):
            df["run_id"] = run.run_id
        run.archive_table("af_label_match", m, upstream=up, mode="replace_all")
        run.archive_table("af_class_summary", cls, upstream=up, mode="replace_all")
        run.archive_table("af_probability", pt, upstream=up, mode="replace_all")
        run.evaluate_gate("af_fda_labels", None, True, detail={
            "labels": int(len(labels)), "mapped_ingredients": len(per_drug),
            "estimable_cells_labelled_drugs": int(len(fc)), "on_label": int(fc.on_fda.sum())})
        run.evaluate_gate("af_fda_validity", float(mw.pvalue), valid,
                          detail={"median_diff": diff, "n_label": int(len(lab_)), "n_not": int(len(non))})
        run.evaluate_gate("af_fda_h1", h1["fda_on_label"]["af_hi_lt_half"], True if valid else None,
                          detail={**h1, "reported_as_finding": valid})
        run.evaluate_gate("af_class", float(chi.pvalue), True, detail={"n_classes": int(len(cls))})
        run.evaluate_gate("af_probability_table", None, True,
                          detail={"n_drugs": int(pt.chembl_id.nunique()), "n_rows": int(len(pt))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
