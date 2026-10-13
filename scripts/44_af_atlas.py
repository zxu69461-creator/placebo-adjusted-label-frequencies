#!/usr/bin/env python3
"""Full attributable-fraction atlas and the drug-label analysis (gates af_main,
af_label_validity, af_h1_label), all registered before this script was run.

Order: the atlas is estimated first, the label mapping second, and the validity check
(label-listed pairs should carry HIGHER risk ratios than unlisted pairs of the same drugs)
is evaluated BEFORE any H1 fraction is reported. If validity fails, H1 is archived but not
presented as a finding.
"""
import gzip
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

SIDER = ROOT / "cache" / "sider"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main() -> int:
    m42 = load(ROOT / "scripts" / "42_af_reproducibility.py", "m42")
    m34 = load(ROOT / "scripts" / "34_ctgov_parse.py", "m34")
    # ---- 1. atlas -------------------------------------------------------------------
    src42 = (ROOT / "scripts" / "42_af_reproducibility.py").read_text()
    sql = src42[src42.index('rows = con.execute("""') + len('rows = con.execute("""'):
                src42.index('""").df()\n        start')]
    with pv.connect(read_only=True) as con:
        rows = con.execute(sql).df()
    cells = []
    for (d, t), g in rows.groupby(["drug", "term"], sort=False):
        if g.nct.nunique() < 2 or (g.a.sum() + g.c.sum()) < 20:
            continue
        lr, se = m42.mh_rr(g)
        if lr != lr or se != se:
            continue
        rr, lo, hi = np.exp(lr), np.exp(lr - 1.96 * se), np.exp(lr + 1.96 * se)
        cells.append(dict(drug=d, term=t, n_trials=int(g.nct.nunique()),
                          events_drug=float(g.a.sum()), n_drug=float(g.n1.sum()),
                          events_pl=float(g.c.sum()), n_pl=float(g.n0.sum()),
                          risk_drug=float(g.a.sum() / g.n1.sum()),
                          risk_pl=float(g.c.sum() / g.n0.sum()),
                          rr=rr, rr_lo=lo, rr_hi=hi,
                          af=1 - 1 / rr, af_lo=1 - 1 / lo, af_hi=1 - 1 / hi,
                          p=float(2 * stats.norm.sf(abs(lr / se)))))
    at = pd.DataFrame(cells)
    at["fdr"] = bh(at.p.values)
    print(f"ATLAS: {len(at):,} estimable (ingredient, PT) cells from {at.drug.nunique():,} drugs, "
          f"{at.term.nunique():,} terms")
    print(f"  harm signals (RR>1, FDR<5%): {((at.rr > 1) & (at.fdr < .05)).sum():,}   "
          f"protective (RR<1, FDR<5%): {((at.rr < 1) & (at.fdr < .05)).sum():,}")

    # ---- 2. label mapping -----------------------------------------------------------
    idx, _ = m34.build_index()
    names = dict(l.split("\t") for l in (SIDER / "drug_names.tsv").read_text().splitlines())
    cid2chembl = {}
    for cid, nm in names.items():
        hit, how = m34.match(nm, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            cid2chembl[cid] = hit
    label = set()
    for l in gzip.open(SIDER / "meddra_all_se.tsv.gz", "rt"):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 6 and f[3] == "PT" and f[0] in cid2chembl:
            for c in cid2chembl[f[0]]:
                label.add((c, f[5].strip().lower()))
    mapped = {c for s in cid2chembl.values() for c in s}
    at["sider_mapped"] = at.drug.isin(mapped)
    at["on_label"] = [(d, t) in label for d, t in zip(at.drug, at.term)]
    sm = at[at.sider_mapped]
    print(f"\nSIDER drugs mapped to ChEMBL: {len(cid2chembl):,}/{len(names):,}; atlas cells for "
          f"mapped drugs: {len(sm):,} (on label {sm.on_label.sum():,}, not on label "
          f"{(~sm.on_label).sum():,})")

    # ---- 3. validity check FIRST ----------------------------------------------------
    lab, non = np.log(sm[sm.on_label].rr), np.log(sm[~sm.on_label].rr)
    mw = stats.mannwhitneyu(lab, non, alternative="greater")
    diff = float(np.median(lab) - np.median(non))
    valid = bool(mw.pvalue < 0.001 and diff > 0)
    print(f"\nVALIDITY: median log-RR on label {np.median(lab):+.3f} vs not on label "
          f"{np.median(non):+.3f} (diff {diff:+.3f});  one-sided MW P = {mw.pvalue:.2e}  -> "
          f"{'PASS' if valid else 'FAIL'}")

    # ---- 4. H1 fractions --------------------------------------------------------------
    def frac(d):
        return dict(n=len(d),
                    harm_signal=float(((d.rr > 1) & (d.fdr < .05)).mean()),
                    af_lt_half=float((d.af < 0.5).mean()),
                    af_hi_lt_half=float((d.af_hi < 0.5).mean()),
                    af_le_zero=float((d.af <= 0).mean()))
    h1 = {"on_label": frac(sm[sm.on_label]), "not_on_label": frac(sm[~sm.on_label])}
    print(f"\nH1 {'(finding)' if valid else '(NOT a finding: validity failed)'}")
    print(f"  {'':<14}{'n':>7}{'harm signal':>13}{'AF<0.5':>9}{'AF_hi<0.5':>11}{'AF<=0':>8}")
    for k, v in h1.items():
        print(f"  {k:<14}{v['n']:>7,}{v['harm_signal']:>13.1%}{v['af_lt_half']:>9.1%}"
              f"{v['af_hi_lt_half']:>11.1%}{v['af_le_zero']:>8.1%}")

    # secondary: label-implied AF from SIDER placebo frequencies
    fr = defaultdict(lambda: {"drug": [], "pl": []})
    for l in gzip.open(SIDER / "meddra_freq.tsv.gz", "rt"):
        f = l.rstrip("\n").split("\t")
        if len(f) < 10 or f[7] != "PT" or f[0] not in cid2chembl:
            continue
        try:
            v = (float(f[5]) + float(f[6])) / 2
        except ValueError:
            continue
        for c in cid2chembl[f[0]]:
            fr[(c, f[9].strip().lower())]["pl" if f[3] == "placebo" else "drug"].append(v)
    pairs = []
    for (c, t), v in fr.items():
        if v["drug"] and v["pl"] and np.median(v["drug"]) > 0:
            pairs.append((c, t, 1 - np.median(v["pl"]) / np.median(v["drug"])))
    lp = pd.DataFrame(pairs, columns=["drug", "term", "label_af"]).merge(at, on=["drug", "term"])
    rho = stats.spearmanr(lp.label_af, lp.af) if len(lp) > 10 else None
    print(f"\nlabel-implied AF vs atlas AF (circular -- labels use the same pivotal trials): "
          f"n={len(lp):,}, Spearman={rho.statistic:.3f}" if rho else "\nlabel AF: too few pairs")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute(
                "SELECT out_id FROM prov_output_asset WHERE object_name='atlas_contrast'").fetchall()]
        ins = [run.register_input(
            "sider:4.1", asset_kind="drug_label_side_effects", local_path=SIDER,
            accession="SIDER-4.1", source_registry="EMBL SIDER",
            url="http://sideeffects.embl.de/download/", genome_build="not_applicable",
            build_evidence="drug label side-effect data", licence="CC BY-NC-SA 4.0",
            licence_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
            notes="label side effects mined from FDA package inserts (Kuhn et al. 2016)")]
        at["run_id"] = run.run_id
        run.archive_table("af_cell", at, inputs=ins, upstream=up, mode="replace_all")
        run.evaluate_gate("af_main", float(at.fdr.min()),
                          bool((at.fdr < .05).any()),
                          detail={"n_cells": len(at), "n_drugs": int(at.drug.nunique()),
                                  "n_harm": int(((at.rr > 1) & (at.fdr < .05)).sum()),
                                  "n_protective": int(((at.rr < 1) & (at.fdr < .05)).sum())})
        run.evaluate_gate("af_label_validity", float(mw.pvalue), valid,
                          detail={"median_diff": diff, "n_label": int(len(lab)),
                                  "n_not_label": int(len(non)),
                                  "sider_mapped_drugs": len(cid2chembl)})
        run.evaluate_gate("af_h1_label", h1["on_label"]["af_lt_half"], None if not valid else True,
                          detail={**h1, "reported_as_finding": valid,
                                  "label_af_spearman": float(rho.statistic) if rho else None,
                                  "label_af_n": int(len(lp))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
