#!/usr/bin/env python3
"""ChEMBL approved indications (drug_indication records with max_phase_for_ind = 4) for jno_r3_indication: all pages
from the ChEMBL web services, cached as JSON under cache/chembl/, registered as an input, archived as
ref_chembl_indication (without the reference lists)."""
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

BASE = "https://www.ebi.ac.uk"
URL = BASE + "/chembl/api/data/drug_indication.json?max_phase_for_ind=4&limit=1000&offset=0"
CACHE = ROOT / "cache" / "chembl_indication" / "drug_indication_phase4.json"   # kept out of cache/chembl/drug_*.json (name index)


def get(url):
    ctx = ssl.create_default_context(cafile=certifi.where())
    for a in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=120, context=ctx) as r:
                return json.loads(r.read())
        except Exception:
            if a == 4:
                raise
            time.sleep(5 * (a + 1))


def main() -> int:
    if not CACHE.exists():
        recs, url = [], URL
        while url:
            d = get(url)
            recs += d["drug_indications"]
            url = BASE + d["page_meta"]["next"] if d["page_meta"]["next"] else None
            print(len(recs), "/", d["page_meta"]["total_count"], flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(recs))
    recs = json.loads(CACHE.read_text())
    df = pd.DataFrame([{k: r.get(k) for k in ("molecule_chembl_id", "parent_molecule_chembl_id", "mesh_id",
                                               "mesh_heading", "efo_id", "efo_term", "max_phase_for_ind")} for r in recs])
    df["max_phase_for_ind"] = df.max_phase_for_ind.astype(float)
    assert (df.max_phase_for_ind == 4).all()
    print(len(df), "records;", df.molecule_chembl_id.nunique(), "molecules")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input("chembl_drug_indication_phase4", asset_kind="api_response", local_path=CACHE,
                                 source_registry="ChEMBL web services", url=URL, genome_build="not_applicable",
                                 build_evidence="drug indication records, not genomic data")
        df["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM ref_chembl_indication")
        run.archive_table("ref_chembl_indication", df[["run_id", "molecule_chembl_id", "parent_molecule_chembl_id",
                                                       "mesh_id", "mesh_heading", "efo_id", "efo_term",
                                                       "max_phase_for_ind"]], inputs=[aid])
    return 0


if __name__ == "__main__":
    sys.exit(main())
