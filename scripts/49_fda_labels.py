#!/usr/bin/env python3
"""Extract one current FDA label per single-ingredient drug from the openFDA export.

Storage-aware (user instruction 2026-09-24): the 14 partitions (1.77 GB zipped) are handled
ONE AT A TIME -- download, parse, keep the best candidate per generic name in memory, delete
the zip -- so peak disk use stays near one partition. Only the ADVERSE REACTIONS text of the
chosen label is archived. Selection rule and matching rule: gate af_fda_labels.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

TMP = ROOT / "cache" / "openfda_tmp"


def rank(app, eff):
    """Innovator (NDA/BLA) before generic (ANDA/other); then the newest label."""
    a = (app or "").upper()
    return (0 if a.startswith(("NDA", "BLA")) else 1, -(int(eff) if (eff or "").isdigit() else 0))


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(subprocess.run(
        ["curl", "-s", "-A", "Mozilla/5.0", "https://api.fda.gov/download.json"],
        capture_output=True).stdout)
    lab = manifest["results"]["drug"]["label"]
    parts = lab["partitions"]
    best, counts, seen = {}, {}, 0
    for i, p in enumerate(parts):
        z = TMP / Path(p["file"]).name
        subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "1800", "-o", str(z),
                        p["file"]], check=True)
        with zipfile.ZipFile(z) as zf:
            name = zf.namelist()[0]
            recs = json.loads(zf.read(name))["results"]
        z.unlink()                                       # delete the zip immediately
        for r in recs:
            seen += 1
            o = r.get("openfda") or {}
            if "HUMAN PRESCRIPTION DRUG" not in (o.get("product_type") or []):
                continue
            gn = o.get("generic_name") or []
            ar = r.get("adverse_reactions") or []
            if len(gn) != 1 or not ar:
                continue
            g = gn[0].strip().lower()
            if " and " in g or "," in g:
                continue
            app = (o.get("application_number") or [""])[0]
            eff = r.get("effective_time") or ""
            counts[g] = counts.get(g, 0) + 1
            cand = dict(generic_name=g, set_id=r.get("set_id"), application_number=app,
                        effective_time=eff, route=";".join(o.get("route") or []),
                        adverse_reactions_text=" ".join(ar))
            if g not in best or rank(app, eff) < rank(best[g]["application_number"], best[g]["effective_time"]):
                best[g] = cand
        del recs
        print(f"  partition {i + 1:>2}/{len(parts)}: labels seen {seen:,}; generics kept {len(best):,}",
              flush=True)
    rows = [dict(b, n_candidate_labels=counts[g]) for g, b in best.items()]
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input(
            "openfda:drug_label_2026-09-24", asset_kind="drug_label",
            accession=f"openFDA-drug-label-{lab.get('export_date')}",
            source_registry="U.S. FDA openFDA (public domain)",
            url="https://open.fda.gov/apis/drug/label/download/",
            genome_build="not_applicable", build_evidence="drug label text",
            licence="public domain (U.S. government work), CC0 per openFDA terms",
            licence_url="https://open.fda.gov/terms/", status="purged_local_copy",
            notes=f"{len(parts)} partitions streamed one at a time and deleted after parsing; "
                  f"{seen:,} labels read; only the chosen label's ADVERSE REACTIONS text kept")
        for r in rows:
            r["run_id"] = run.run_id
        run.archive_table("fda_label", rows, inputs=[aid], mode="replace_all")
    TMP.rmdir()
    nda = sum(1 for r in rows if r["application_number"].upper().startswith(("NDA", "BLA")))
    print(f"\nkept {len(rows):,} single-ingredient generics ({nda:,} with an innovator NDA/BLA label)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
