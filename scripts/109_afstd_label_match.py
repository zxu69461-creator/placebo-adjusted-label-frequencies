#!/usr/bin/env python3
"""Label linkage for every pair estimated by the hierarchical estimator (gate afstd_label_match).

Rule B of 52_af_v2.py, unchanged: a pair is on the current FDA label if its term (UK or US spelling, optional plural)
occurs as a whole word in the adverse-reactions section of any current innovator (NDA/BLA) label of the ingredient, or of
any label when the ingredient has no innovator label; SIDER 4.1 pairs via af_common.label_pairs(). The result must agree
exactly with af_label_match on the pairs the two tables share. Also records, for each matched term, the share of all
current label texts it matches (term_promiscuity, for sensitivity set S8) and adds display names for drugs that lack one.
"""
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af_common as af  # noqa: E402
from oajoint import provenance as pv  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M50 = load(ROOT / "scripts" / "50_af_fda_analysis.py", "m50")
M34 = load(ROOT / "scripts" / "34_ctgov_parse.py", "m34")
M58 = load(ROOT / "scripts" / "58_drug_names.py", "m58")
us, pattern = M50.us, M50.pattern


def pool_text(pool, idx):
    """Rule B text per ChEMBL ingredient (verbatim logic of 52_af_v2.py lines 131-144)."""
    text = defaultdict(lambda: {"innovator": [], "all_labels": []})
    gen = {}
    for g, pl_, t in zip(pool.generic_name, pool.pool, pool.adverse_reactions_text):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                text[c][pl_].append(t or "")
                gen.setdefault(c, g)
    src = {c: ("innovator" if v["innovator"] else "all_labels") for c, v in text.items()}
    return {c: "\n\n".join(v[src[c]]) for c, v in text.items()}, src, gen


def names_for(ids, pool, idx):
    """Display names for drugs missing from ref_drug_name (the preference order of 58_drug_names.py)."""
    name, src = {}, {}
    with pv.connect(read_only=True) as con:
        syn = dict(con.execute("SELECT chembl_id, synonyms FROM ref_chembl_drug").fetchall())
    for fpath in sorted((ROOT / "cache" / "chembl").glob("drug_*.json")):
        for d in json.loads(fpath.read_text())["drugs"]:
            c = d["molecule_chembl_id"]
            if c not in ids or c in name:
                continue
            for tag in ("INN", "USAN", "BAN"):
                hit = [re.sub(r"\s*\([^)]*\)\s*$", "", x) for x in (d.get("synonyms") or [])
                       if re.search(r"\(([^)]*\b%s\b[^)]*)\)\s*$" % tag, x or "")]
                if hit:
                    name[c], src[c] = hit[0].lower(), "chembl_" + tag.lower()
                    break
    for g in sorted(pool.generic_name, key=lambda x: (len(M58.clean(x)), x)):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                if c in ids and c not in name:
                    name[c], src[c] = M58.clean(g), "fda_generic"
    for c in sorted(ids - set(name)):
        plain = [s for s in (syn.get(c) or []) if s and re.fullmatch(r"[A-Za-z][a-z\- ]{3,}", s)]
        name[c], src[c] = (plain[0].lower(), "chembl_synonym") if plain else (c, "chembl_id")
    return pd.DataFrame({"chembl_id": sorted(name), "display_name": [name[c] for c in sorted(name)],
                         "source": [src[c] for c in sorted(name)]})


def main() -> int:
    with pv.connect(read_only=True) as con:
        cells = con.execute("SELECT drug, term FROM afstd_cell WHERE estimator = 'AF 3.3' ORDER BY drug, term").df()
        pool = con.execute("SELECT generic_name, pool, adverse_reactions_text FROM fda_label_pool ORDER BY generic_name, pool").df()
        old = con.execute("SELECT drug, term, has_fda_label, on_fda, on_sider FROM af_label_match").df()
        named = {r[0] for r in con.execute("SELECT chembl_id FROM ref_drug_name").fetchall()}
    print(f"pairs {len(cells):,}; drugs {cells.drug.nunique():,}; terms {cells.term.nunique():,}")
    idx, _ = M34.build_index()
    text, src, gen = pool_text(pool, idx)
    low = {c: t.lower() for c, t in text.items()}
    print(f"ingredients with label text {len(text):,} ({sum(v == 'innovator' for v in src.values()):,} innovator)")

    # a term with no words (4 pairs with an empty term) cannot occur as a whole word; re.compile("") would match anything
    pats = {t: pattern(t) for t in sorted(cells.term.unique()) if t.split()}
    stem = {t: t.split()[-1][:max(3, len(t.split()[-1]) - 2)] for t in pats}   # cheap substring prefilter
    has = cells.drug.isin(text).values
    on = []
    for d, t, h in zip(cells.drug, cells.term, has):
        on.append(bool(h and t in pats and stem[t] in low[d] and pats[t].search(text[d])))
    cells["has_fda_label"] = has
    cells["on_fda"] = on
    cells["label_source"] = cells.drug.map(src)
    cells["fda_generic"] = cells.drug.map(gen)
    label_raw, mapped = af.label_pairs()
    sider = {(c, us(t)) for c, t in label_raw}
    cells["sider_mapped"] = cells.drug.isin(mapped)
    cells["on_sider"] = [(d, t) in sider for d, t in zip(cells.drug, cells.term)]

    # promiscuity of matched terms over all current label texts
    texts = sorted(text); prom = {}
    for t in sorted(cells.loc[cells.on_fda, "term"].unique()):
        cand = [c for c in texts if stem[t] in low[c]]
        prom[t] = sum(1 for c in cand if pats[t].search(text[c])) / len(texts)
    cells["term_promiscuity"] = cells.term.map(prom)
    print(f"on_fda {cells.on_fda.sum():,} pairs ({cells[cells.on_fda].drug.nunique():,} drugs); on_sider "
          f"{cells.on_sider.sum():,}; terms matched {len(prom):,}; promiscuous (>50%) "
          f"{sorted(t for t, v in prom.items() if v > .5)}")

    m = cells.merge(old, on=["drug", "term"], suffixes=("", "_old"))
    dis = {k: int((m[k] != m[k + "_old"]).sum()) for k in ("has_fda_label", "on_fda", "on_sider")}
    n_dis = sum(dis.values())
    print(f"agreement check on {len(m):,} shared pairs: {dis}")

    miss = set(cells.drug) - named
    names = names_for(miss, pool, idx) if miss else pd.DataFrame()
    print(f"display names added for {len(names):,} drugs: {names.source.value_counts().to_dict() if len(names) else {}}")

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_cell','fda_label_pool','ref_chembl_drug','af_label_match')").fetchall()]
        run.evaluate_gate("afstd_label_match", float(n_dis), n_dis == 0,
                          detail=dict(n_pairs=len(cells), n_shared=len(m), disagreements=dis,
                                      n_on_fda=int(cells.on_fda.sum()), n_on_sider=int(cells.on_sider.sum()),
                                      n_drugs_with_label=int(cells[cells.has_fda_label].drug.nunique()),
                                      n_drugs_on_fda=int(cells[cells.on_fda].drug.nunique()),
                                      n_pairs_empty_term=int((~cells.term.isin(pats)).sum()),
                                      promiscuous_terms=sorted(t for t, v in prom.items() if v > .5)))
        if n_dis:
            return 1
        cols = ["drug", "term", "has_fda_label", "label_source", "on_fda", "sider_mapped", "on_sider", "fda_generic",
                "term_promiscuity"]
        out = cells[cols].copy(); out["run_id"] = run.run_id
        run.archive_table("afstd_label_match", out, upstream=up, mode="replace_all")
        if len(names):
            names["run_id"] = run.run_id
            run.archive_table("ref_drug_name", names, upstream=up)
    return 0


if __name__ == "__main__":
    sys.exit(main())
