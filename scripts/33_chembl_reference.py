#!/usr/bin/env python3
"""Build the local drug dictionary from ChEMBL: every drug's synonyms, WHO-ATC codes and
mechanism of action.

Two paged passes over ChEMBL's /drug (16,083 records) and /mechanism (7,561) replace
what would otherwise be thousands of per-name lookups against the trial corpus. The ATC
level-4 code is the pre-registered class unit; the mechanism TARGET is carried for the
later genetic triangulation. Raw pages are cached so a re-run is offline and identical.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

CACHE = ROOT / "cache" / "chembl"
API = "https://www.ebi.ac.uk/chembl/api/data"


def pages(endpoint, key):
    CACHE.mkdir(parents=True, exist_ok=True)
    out, offset = [], 0
    while True:
        f = CACHE / f"{endpoint}_{offset:06d}.json"
        if not f.exists():
            for attempt in range(6):
                r = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "300",
                                    f"{API}/{endpoint}.json?limit=1000&offset={offset}"],
                                   capture_output=True)
                try:
                    json.loads(r.stdout); f.write_bytes(r.stdout); break
                except json.JSONDecodeError:
                    time.sleep(8 * (attempt + 1))
            else:
                raise RuntimeError(f"{endpoint} offset {offset} failed")
        d = json.loads(f.read_text())
        out += d[key]
        total = d["page_meta"]["total_count"]
        offset += 1000
        print(f"  {endpoint}: {len(out)}/{total}", flush=True)
        if offset >= total:
            return out, total


def main() -> int:
    drugs, nd = pages("drug", "drugs")
    mechs, nm = pages("mechanism", "mechanisms")
    drows = []
    for x in drugs:
        # /drug returns NO pref_name, and for many drugs the plain INN is only in the
        # separate `synonyms` field, tagged with its nomenclature source
        # ("Canagliflozin (INN, MI, USAN)"). Omitting it left canagliflozin and lisinopril
        # unfindable by name in this table, which silently dropped two rows from a figure.
        inn = {re.sub(r"\s*\([^)]*\)\s*$", "", t or "").strip() for t in (x.get("synonyms") or [])}
        syn = sorted(({(s.get("molecule_synonym") or "").strip() for s in
                       (x.get("molecule_synonyms") or []) if s.get("molecule_synonym")}
                      | inn | ({x["pref_name"]} if x.get("pref_name") else set())) - {""})
        atc = x.get("atc_classification") or []
        codes = sorted({a["code"] for a in atc if a.get("code")})
        drows.append(dict(chembl_id=x["molecule_chembl_id"], pref_name=x.get("pref_name"),
                          synonyms=syn, atc_codes=codes,
                          atc4=sorted({c[:5] for c in codes}),
                          atc4_desc=sorted({(a.get("description") or "").rsplit(": ", 1)[0]
                                            for a in atc}),
                          max_phase=float(x["max_phase"]) if x.get("max_phase") not in (None, "") else None,
                          first_approval=x.get("first_approval"),
                          drug_type=str(x.get("drug_type")),
                          biotherapeutic=bool(x.get("biotherapeutic"))))
    mrows = [dict(chembl_id=m.get("molecule_chembl_id"),
                  parent_chembl_id=m.get("parent_molecule_chembl_id"),
                  target_chembl_id=m.get("target_chembl_id"),
                  action_type=m.get("action_type"),
                  mechanism_of_action=m.get("mechanism_of_action"),
                  direct_interaction=bool(m.get("direct_interaction")))
             for m in mechs]
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input(
            "chembl:drug_mechanism", asset_kind="drug_reference", local_path=CACHE,
            accession="ChEMBL-REST", source_registry="EMBL-EBI ChEMBL",
            url=API, genome_build="not_applicable",
            build_evidence="drug reference data, no genomic coordinates",
            licence="ChEMBL data, CC BY-SA 3.0",
            licence_url="https://creativecommons.org/licenses/by-sa/3.0/",
            notes=f"{nd} drugs, {nm} mechanisms; pages cached under cache/chembl")
        run.archive_table("ref_chembl_drug", drows, inputs=[aid], mode="replace_all")
        run.archive_table("ref_chembl_mechanism", mrows, inputs=[aid], mode="replace_all")
    with_atc = sum(1 for r in drows if r["atc4"])
    print(f"\n{len(drows)} drugs ({with_atc} with an ATC-4 class), {len(mrows)} mechanisms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
