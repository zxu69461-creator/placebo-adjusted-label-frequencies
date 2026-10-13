#!/usr/bin/env python3
"""Gate jno_rev_label_audit (rule in 126_jno_review_prereg.py): precision of the label linkage in the 400-pair audit
sample (128_jno_rev_analyses.py). Classes (a) listed reaction, (b) indication or study population, (c) negation,
(d) other mention were proposed with language-model assistance (manuscript/jama/review/label_audit_classified.csv) and
must be verified by an author before publication; the gate detail records author_verified = false until then.
Reports precision (share a) overall and by label subsection class with 95% Wilson CIs, and, post hoc, the expected
share with excess fraction < 0.5 among audited pairs by class."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

REV = ROOT / "manuscript" / "jama" / "review"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(c - h), float(c + h)


def main() -> int:
    s = pd.read_csv(REV / "label_audit_sample.csv")
    final = REV / "audit_final.csv"                 # authors' blinded double coding, adjudicated (141)
    c = pd.read_csv(final) if final.exists() else pd.read_csv(REV / "label_audit_classified.csv")
    assert len(s) == len(c) == 400
    if not final.exists():
        assert (s.drug.values == c.drug.values).all() and (s.term.values == c.term.values).all()
    assert set(c.category) <= set("abcd")
    d = s.assign(category=c.category.values, note=c.note.values if "note" in c else "")
    with pv.connect(read_only=True) as con:
        p = con.execute("SELECT drug, term, p_af_lt_half FROM afstd_pair WHERE on_fda").df()
    d = d.merge(p, on=["drug", "term"], how="left")
    assert d.p_af_lt_half.notna().all()
    det, rows = {"author_verified": final.exists(), "counts": d.category.value_counts().to_dict()}, []
    for grp, g in [("all", d)] + list(d.groupby("section_class")):
        k = int((g.category == "a").sum())
        pr, lo, hi = wilson(k, len(g))
        det[grp] = dict(n=int(len(g)), n_listed=k, precision=pr, lo=lo, hi=hi,
                        counts=g.category.value_counts().to_dict())
        rows.append(dict(analysis="jno_rev_label_audit", set_name="precision", grp=grp, n_pairs=int(len(g)),
                         n_drugs=int(g.drug.nunique()), metric="precision", value=pr, lo=lo, hi=hi))
    ok = d.category == "a"
    det["post_hoc_exp_lt_half"] = dict(listed=float(d.p_af_lt_half[ok].mean()), other=float(d.p_af_lt_half[~ok].mean()),
                                       all=float(d.p_af_lt_half.mean()), n_other=int((~ok).sum()))
    for k_, m in (("listed", ok), ("other", ~ok)):
        rows.append(dict(analysis="jno_rev_label_audit", set_name="exp_lt_half_post_hoc", grp=k_, n_pairs=int(m.sum()),
                         n_drugs=int(d[m].drug.nunique()), metric="exp_lt_half", value=float(d.p_af_lt_half[m].mean()),
                         lo=np.nan, hi=np.nan))
    print(det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input("jama_label_audit_classified", asset_kind="config",
                                 local_path=REV / "label_audit_classified.csv", source_registry="manuscript",
                                 genome_build="not_applicable",
                                 build_evidence="label text classifications, not genomic data")
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair', 'jama_label_audit_sample')").fetchall()]
        R = pd.DataFrame(rows); R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_rev_label_audit'")
        run.archive_table("jno_summary", R, upstream=up, inputs=[aid])
        run.evaluate_gate("jno_rev_label_audit", det["all"]["precision"], None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
