#!/usr/bin/env python3
"""Gate jno_r3_recall, part 2 (rule in 137_jno_r3_prereg.py): share of 200 unlisted pairs that the label lists under
another wording, with a Wilson 95% CI, and the implied recall of the linkage.

Classes come from manuscript/jama/review/recall_classified.csv (language-model proposals) unless an adjudicated author
file manuscript/jama/review/recall_final.csv (columns row, category) exists; the gate detail records which.
Implied recall = TP / (TP + FN) with TP = precision x listed pairs (precision from jno_rev_label_audit) and
FN = share (a) x unlisted pairs with a current label. Post hoc companion: expected share with AF < 0.5 among sampled
pairs classified (a), beside the primary set."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

REV = ROOT / "manuscript" / "jama" / "review"


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(c - h), float(c + h)


def main() -> int:
    s = pd.read_csv(REV / "recall_sample.csv")
    final = REV / "recall_final.csv"
    src = "author_adjudicated" if final.exists() else "language_model_proposal"
    c = pd.read_csv(final if final.exists() else REV / "recall_classified.csv")
    assert len(s) == len(c) == 200 and set(c.category) <= {"a", "b"}
    d = s.assign(category=c.category.values)
    with pv.connect(read_only=True) as con:
        n_unlisted = con.execute("SELECT count(*) FROM afstd_label_match WHERE has_fda_label AND NOT on_fda").fetchone()[0]
        n_listed = con.execute("SELECT count(*) FROM afstd_pair WHERE on_fda").fetchone()[0]
        prec = json.loads(con.execute("SELECT metric_detail FROM prov_decision WHERE gate_name = 'jno_rev_label_audit'")
                          .fetchone()[0])["all"]["precision"]
        p1 = con.execute("""SELECT value FROM jno_summary WHERE analysis = 'jno_r3_recall' AND set_name = 'sider_proxy'
                            AND grp = 'all'""").fetchone()[0]
        head = con.execute("""SELECT value FROM afstd_summary WHERE analysis = 'headline' AND set_name = 'primary'
                              AND grp = 'all' AND metric = 'exp_lt_half'""").fetchone()[0]
    k = int((d.category == "a").sum())
    share, lo, hi = wilson(k, len(d))
    tp = prec * n_listed

    def rec(f):
        return tp / (tp + f * n_unlisted)
    det = dict(source=src, author_verified=src == "author_adjudicated", n=int(len(d)), n_a=k, share_a=share,
               ci=[lo, hi], n_unlisted=int(n_unlisted), n_listed=int(n_listed), precision=prec,
               implied_recall=rec(share), implied_recall_ci=[rec(hi), rec(lo)], sider_proxy=p1,
               exp_lt_half_missed=float(d.p_af_lt_half[d.category == "a"].mean()) if k else None,
               exp_lt_half_primary=float(head))
    print(json.dumps(det, indent=1))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input("jama_recall_classes", asset_kind="config",
                                 local_path=final if final.exists() else REV / "recall_classified.csv",
                                 source_registry="manuscript", genome_build="not_applicable",
                                 build_evidence="label text classifications, not genomic data")
        R = pd.DataFrame([dict(analysis="jno_r3_recall", set_name="false_negative_audit", grp="all", n_pairs=200,
                               n_drugs=int(d.drug.nunique()), metric="share_listed_other_wording", value=share, lo=lo,
                               hi=hi),
                          dict(analysis="jno_r3_recall", set_name="implied_recall", grp="all", n_pairs=int(n_listed),
                               n_drugs=0, metric="recall", value=det["implied_recall"],
                               lo=det["implied_recall_ci"][0], hi=det["implied_recall_ci"][1])])
        R["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_r3_recall' AND set_name IN "
                        "('false_negative_audit', 'implied_recall')")
        run.archive_table("jno_summary", R, inputs=[aid])
        run.evaluate_gate("jno_r3_recall", p1, None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
