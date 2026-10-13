#!/usr/bin/env python3
"""Route-level label pool (gate af_route): distinct ADVERSE REACTIONS texts per
(single-ingredient generic, route family), innovator labels first within each family.

Same streaming discipline as script 53: one openFDA partition at a time, deleted after use.
A label with several routes (kits) is filed under each of its families.
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402
from oajoint.routes import fda_family  # noqa: E402

TMP = ROOT / "cache" / "openfda_tmp"


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    lab = json.loads(subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "https://api.fda.gov/download.json"],
                                    capture_output=True).stdout)["results"]["drug"]["label"]
    texts = defaultdict(dict)            # (pool, generic, family) -> {sha1: text}
    nlab = Counter()
    unmapped = Counter()
    for i, p in enumerate(lab["partitions"]):
        z = TMP / Path(p["file"]).name
        subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "1800", "-o", str(z), p["file"]], check=True)
        with zipfile.ZipFile(z) as zf:
            recs = json.loads(zf.read(zf.namelist()[0]))["results"]
        z.unlink()
        for r in recs:
            o = r.get("openfda") or {}
            if "HUMAN PRESCRIPTION DRUG" not in (o.get("product_type") or []):
                continue
            gn, ar = o.get("generic_name") or [], r.get("adverse_reactions") or []
            if len(gn) != 1 or not ar:
                continue
            g = gn[0].strip().lower()
            if " and " in g or "," in g:
                continue
            app = (o.get("application_number") or [""])[0]
            pool = "innovator" if app.upper().startswith(("NDA", "BLA")) else "all_labels"
            fams = {fda_family(x) for x in (o.get("route") or [])}
            for x in (o.get("route") or []):
                if fda_family(x) is None:
                    unmapped[x.upper()] += 1
            fams.discard(None)
            txt = " ".join(ar)
            h = hashlib.sha1(txt.encode()).hexdigest()
            for f in fams:
                texts[(pool, g, f)][h] = txt
                nlab[(pool, g, f)] += 1
        del recs
        print(f"  partition {i + 1:>2}/{len(lab['partitions'])}: (generic, route) strata "
              f"{len({k[1:] for k in texts}):,}", flush=True)
    rows = []
    for g, f in sorted({k[1:] for k in texts}):
        pool = "innovator" if ("innovator", g, f) in texts else "all_labels"
        t = texts[(pool, g, f)]
        rows.append(dict(generic_name=g, route_family=f, pool=pool, n_labels=nlab[(pool, g, f)],
                         n_distinct_texts=len(t), adverse_reactions_text="\n\n".join(t.values())))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            aid = [x[0] for x in con.execute(
                "SELECT asset_id FROM prov_input_asset WHERE asset_id='openfda:drug_label_2026-09-24'").fetchall()]
        for r in rows:
            r["run_id"] = run.run_id
        run.archive_table("fda_label_route_pool", rows, inputs=aid, mode="replace_all")
    TMP.rmdir()
    print(f"\n{len(rows):,} (generic, route family) strata; by family: "
          f"{dict(Counter(r['route_family'] for r in rows).most_common())}")
    print(f"label routes outside the family map (not used): {dict(unmapped.most_common(12))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
