#!/usr/bin/env python3
"""Gate jno_label_section: which part of the label each listed side effect comes from (clinical-trial subsection,
postmarketing subsection, other text), and the headline per part. Rule as amended by 115c_jno_prereg_amend.py."""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M109 = load(ROOT / "scripts" / "109_afstd_label_match.py", "m109")
S110 = load(ROOT / "scripts" / "110_afstd_findings.py", "s110")
HEAD = re.compile(r"(?<![\d.])6(?:\.(\d))?\s+(ADVERSE REACTIONS|Clinical\s+(?:Trials?|Studies)\s+Experience|"
                  r"Post-?\s?[Mm]arketing|Immunogenicity)")
SEED = 20260927


def kind(num, title):
    t = title.lower()
    if num is None:
        return "other" if t.startswith("adverse reactions") else None
    if re.match(r"clinical\s+(trials?|studies)\s+experience", t):
        return "trial"
    if re.match(r"post-?\s?marketing", t):
        return "post"
    return "other"                                   # immunogenicity


def segment(text):
    """Return {'trial': str, 'post': str, 'other': str, 'segmented': bool}."""
    cuts = []
    for m in HEAD.finditer(text):
        k = kind(m.group(1), m.group(2))
        if k is not None:
            cuts.append((m.start(), k))
    spans = {"trial": [], "post": [], "other": []}
    bounds = [(0, "other")] + cuts + [(len(text), None)]
    for (a, k), (b, _) in zip(bounds[:-1], bounds[1:]):
        spans[k].append(text[a:b])
    out = {k: "\n".join(v) for k, v in spans.items()}
    out["segmented"] = any(k == "trial" for _, k in cuts)
    return out


def main() -> int:
    with pv.connect(read_only=True) as con:
        pairs = con.execute("""SELECT drug, term, on_fda, p_af_lt_half, p_no_effect FROM afstd_pair
                               WHERE on_fda""").df()
        pool = con.execute("SELECT generic_name, pool, adverse_reactions_text FROM fda_label_pool "
                           "ORDER BY generic_name, pool").df()
    idx, _ = M109.M34.build_index()
    text, src, gen = M109.pool_text(pool, idx)
    seg = {c: segment(t) for c, t in text.items()}
    pats = {}
    rows = []
    for d, t in zip(pairs.drug, pairs.term):
        s = seg.get(d)
        p = pats.setdefault(t, M109.pattern(t))
        if s is None:
            rows.append(dict(drug=d, term=t, segmented=False, on_trial=False, on_postmarketing=False, on_other=False))
            continue
        rows.append(dict(drug=d, term=t, segmented=s["segmented"], on_trial=bool(p.search(s["trial"])),
                         on_postmarketing=bool(p.search(s["post"])), on_other=bool(p.search(s["other"]))))
    r = pd.DataFrame(rows)
    r["section_class"] = np.select([~r.segmented, r.on_trial, r.on_postmarketing, r.on_other],
                                   ["unsegmented", "trial", "postmarketing_only", "other_only"], "unmatched")
    seg_r = r[r.segmented]
    n_unmatched = int((seg_r.section_class == "unmatched").sum())
    m = r.merge(pairs, on=["drug", "term"])
    rng = np.random.default_rng(SEED)
    summ, det = [], dict(n_listed=len(r), n_unmatched_qc=n_unmatched,
                         n_ingredients_segmented=int(sum(v["segmented"] for v in seg.values())),
                         n_ingredients=len(seg))
    for cls in ["trial", "postmarketing_only", "other_only", "unsegmented", "all"]:
        d = m if cls == "all" else m[m.section_class == cls]
        if not len(d):
            continue
        lo, hi = S110.boot_mean(d, "p_af_lt_half", rng)
        val = float(d.p_af_lt_half.mean())
        det[cls] = dict(n=len(d), drugs=int(d.drug.nunique()), exp_lt_half=val, ci=[lo, hi],
                        conf_bg=float((d.p_af_lt_half > .975).mean()))
        summ.append(dict(analysis="jno_label_section", set_name=cls, grp="all", n_pairs=len(d),
                         n_drugs=int(d.drug.nunique()), metric="exp_lt_half", value=val, lo=lo, hi=hi))
    print(det)
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','fda_label_pool','afstd_label_match')").fetchall()]
        out = r.copy(); out["run_id"] = run.run_id
        run.archive_table("jno_label_section", out[["run_id", "drug", "term", "segmented", "on_trial",
                                                    "on_postmarketing", "on_other", "section_class"]],
                          upstream=up, mode="replace_all")
        sm = pd.DataFrame(summ); sm["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_label_section'")
        run.archive_table("jno_summary", sm, upstream=up)
        run.evaluate_gate("jno_label_section", float(det.get("trial", {}).get("n", 0)), n_unmatched == 0, detail=det)
    return 0 if n_unmatched == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
