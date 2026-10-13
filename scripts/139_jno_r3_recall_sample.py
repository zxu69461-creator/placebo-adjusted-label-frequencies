#!/usr/bin/env python3
"""jno_r3_recall, part 1 and the sample of part 2 (rule in 137_jno_r3_prereg.py).

Part 1: among pairs listed by SIDER 4.1 (2016 labels) whose drug has a current FDA label and that the model estimated,
the share flagged as listed by the current-label text match, overall and by number of words in the term.
Part 2: a random sample of 200 pairs with a current FDA label but not flagged as listed (seed 20261013), written with
the drug's adverse-reactions label text (manuscript/jama/review/recall_sample.csv and recall_texts/), to be classified
(a) listed under another wording or (b) not listed; evaluated by 139b after classification and author verification.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402
import importlib.util  # noqa: E402

_sp = importlib.util.spec_from_file_location("s109", ROOT / "scripts" / "109_afstd_label_match.py")
S109 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(S109)

REV = ROOT / "manuscript" / "jama" / "review"
SEED = 20261013


def main() -> int:
    with pv.connect(read_only=True) as con:
        m = con.execute("""SELECT m.drug, m.term, m.on_fda, m.on_sider, m.has_fda_label, m.fda_generic, m.label_source,
                                  p.n_trials, p.p_af_lt_half
                           FROM afstd_label_match m JOIN afstd_pair p USING (drug, term)""").df()
        pool = con.execute("SELECT generic_name, pool, adverse_reactions_text FROM fda_label_pool").df()
        names = dict(con.execute("SELECT chembl_id, display_name FROM ref_drug_name").fetchall())
    s = m[m.on_sider & m.has_fda_label]
    nw = s.term.str.split().str.len()
    p1 = dict(n=int(len(s)), recall=float(s.on_fda.mean()), n_1word=int((nw == 1).sum()),
              recall_1word=float(s.on_fda[nw == 1].mean()), n_multi=int((nw > 1).sum()),
              recall_multi=float(s.on_fda[nw > 1].mean()))
    print(p1)
    u = m[m.has_fda_label & ~m.on_fda].sort_values(["drug", "term"]).reset_index(drop=True)
    smp = u.iloc[np.random.default_rng(SEED).choice(len(u), 200, replace=False)].sort_values(["drug", "term"])
    idx, _ = S109.M34.build_index()
    text, src_map, gen = S109.pool_text(pool, idx)                 # the same drug -> label text map as the linkage
    tdir = REV / "recall_texts"
    for f in tdir.glob("*.txt"):
        f.unlink()
    tdir.mkdir(exist_ok=True)
    rows = []
    for r in smp.itertuples():
        src = src_map[r.drug]
        txt = text[r.drug]
        assert txt, (r.drug, r.fda_generic, src)
        fn = re.sub(r"[^a-z0-9]+", "_", r.fda_generic.lower()) + f"__{src}.txt"
        (tdir / fn).write_text(txt)
        rows.append(dict(drug=r.drug, drug_name=names.get(r.drug, r.fda_generic), term=r.term, fda_generic=r.fda_generic,
                         label_pool=src, label_text_file=f"recall_texts/{fn}", n_trials=int(r.n_trials),
                         p_af_lt_half=round(float(r.p_af_lt_half), 3)))
    pd.DataFrame(rows).to_csv(REV / "recall_sample.csv", index=False)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_label_match','afstd_pair','fda_label_pool')").fetchall()]
        run.archive_document("jama_recall_sample", REV / "recall_sample.csv", kind="csv", upstream=up)
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_r3_recall' AND set_name = 'sider_proxy'")
        R = pd.DataFrame([dict(analysis="jno_r3_recall", set_name="sider_proxy", grp=g, n_pairs=n, n_drugs=0,
                               metric="recall", value=v, lo=np.nan, hi=np.nan)
                          for g, n, v in (("all", p1["n"], p1["recall"]), ("1 word", p1["n_1word"], p1["recall_1word"]),
                                          ("2+ words", p1["n_multi"], p1["recall_multi"]))])
        R["run_id"] = run.run_id
        run.archive_table("jno_summary", R, upstream=up)
    (REV / "recall_part1.json").write_text(json.dumps(p1, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
