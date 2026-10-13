#!/usr/bin/env python3
"""ref_drug_name: one display name per ChEMBL id used in the results.

Preference: the ChEMBL synonym tagged INN (then USAN/BAN); else the shortest current FDA generic
name that maps to the id (salts, routes, biosimilar suffixes stripped); else a plain ChEMBL synonym.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("m34", ROOT / "scripts" / "34_ctgov_parse.py")
M34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M34)
STRIP = re.compile(r"\b(hydrochloride|hcl|hci|calcium|sodium|potassium|oral|tartrate|mesylate|maleate|"
                   r"fumarate|succinate|besylate|citrate|sulfate|phosphate|acetate|bromide|injection|"
                   r"extended-release|delayed-release)\b", re.I)


def clean(n):
    n = STRIP.sub(" ", (n or "").lower())
    n = re.sub(r"-[a-z]{4}$", "", n.strip())
    return re.sub(r"\s+", " ", n).strip()


def main() -> int:
    with pv.connect(read_only=True) as con:
        ids = {r[0] for r in con.execute("SELECT DISTINCT drug FROM af_cell UNION SELECT DISTINCT drug FROM af_route_cell").fetchall()}
        gens = [r[0] for r in con.execute("SELECT generic_name FROM fda_label_pool UNION "
                                          "SELECT generic_name FROM fda_label_route_pool").fetchall()]
        syn = dict(con.execute("SELECT chembl_id, synonyms FROM ref_chembl_drug").fetchall())
    idx, _ = M34.build_index()
    name, src = {}, {}
    import json
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
    for g in sorted(gens, key=lambda x: (len(clean(x)), x)):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                if c in ids and c not in name:
                    name[c], src[c] = clean(g), "fda_generic"
    for c in ids - set(name):
        plain = [s for s in (syn.get(c) or []) if s and re.fullmatch(r"[A-Za-z][a-z\- ]{3,}", s)]
        name[c], src[c] = (plain[0].lower(), "chembl_synonym") if plain else (c, "chembl_id")
    df = pd.DataFrame({"chembl_id": list(name), "display_name": [name[c] for c in name], "source": [src[c] for c in name]})
    print(df.source.value_counts().to_string())
    print(df[df.chembl_id.isin(["CHEMBL1742982", "CHEMBL1076903", "CHEMBL1112", "CHEMBL521", "CHEMBL2108724"])].to_string(index=False))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('fda_label_pool','fda_label_route_pool','ref_chembl_drug')").fetchall()]
        df["run_id"] = run.run_id
        run.archive_table("ref_drug_name", df, upstream=up, mode="replace_all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
