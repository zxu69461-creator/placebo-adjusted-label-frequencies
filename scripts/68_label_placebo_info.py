#!/usr/bin/env python3
"""How often does a label itself give the placebo rate of a listed side effect? (descriptive gate
af_label_placebo_info, registered in this script before computation). SIDER 4.1 frequency table, PT level,
restricted to v2 SIDER label-listed atlas cells. Also: label-implied AF (1 - placebo/drug frequency) versus atlas AF."""
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

spec = importlib.util.spec_from_file_location("m50", ROOT / "scripts" / "50_af_fda_analysis.py")
M50 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M50)
spec = importlib.util.spec_from_file_location("m34", ROOT / "scripts" / "34_ctgov_parse.py")
M34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M34)
SIDER = ROOT / "cache" / "sider"
RULE = ("DESCRIPTIVE (af_label_placebo_info). Among v2 atlas cells listed on the 2016 label (SIDER, PT level), the "
        "share for which the label reports (a) any frequency on the drug and (b) a placebo frequency (SIDER "
        "meddra_freq 'placebo' rows). For cells with both, label-implied AF = 1 - median placebo / median drug "
        "frequency; report Spearman correlation with atlas AF and the median difference (label minus atlas). No pass "
        "criterion; reported as description.")


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("af_label_placebo_info", RULE, gate_family="descriptive")
        idx, _ = M34.build_index()
        names = dict(l.split("\t") for l in (SIDER / "drug_names.tsv").read_text().splitlines())
        cid2 = {}
        for cid, nm in names.items():
            hit, how = M34.match(nm, [], idx)
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
                fr[(c, M50.us(f[9].strip().lower()))]["pl" if f[3] == "placebo" else "drug"].append(v)
        with pv.connect(read_only=True) as con:
            lab = con.execute("SELECT drug, term, af FROM af_cell WHERE on_label").df()
        has_any = [bool(fr.get((d, t), {}).get("drug")) for d, t in zip(lab.drug, lab.term)]
        has_pl = [bool(fr.get((d, t), {}).get("pl")) for d, t in zip(lab.drug, lab.term)]
        lab["has_freq"], lab["has_pl"] = has_any, has_pl
        both = lab[lab.has_freq & lab.has_pl].copy()
        both["label_af"] = [1 - np.median(fr[(d, t)]["pl"]) / np.median(fr[(d, t)]["drug"]) if np.median(fr[(d, t)]["drug"]) > 0
                            else np.nan for d, t in zip(both.drug, both.term)]
        both = both.dropna(subset=["label_af"])
        rho = stats.spearmanr(both.label_af, both.af).statistic
        det = dict(n_label_cells=int(len(lab)), share_any_frequency=float(lab.has_freq.mean()),
                   share_placebo_frequency=float(lab.has_pl.mean()), n_both=int(len(both)), spearman=float(rho),
                   median_label_af=float(both.label_af.median()), median_atlas_af=float(both.af.median()),
                   median_diff_label_minus_atlas=float((both.label_af - both.af).median()))
        print(det)
        run.evaluate_gate("af_label_placebo_info", float(lab.has_pl.mean()), None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
