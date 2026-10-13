#!/usr/bin/env python3
"""v2 label pool (gate af_v2, change B): keep the DISTINCT adverse-reaction texts of every
current innovator label per single-ingredient generic, across formulations and routes.

Storage-aware: partitions streamed one at a time and deleted; identical texts (generic
copies of a brand label) are stored once, so the pool stays compact.
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

TMP = ROOT / "cache" / "openfda_tmp"


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    lab = json.loads(subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "https://api.fda.gov/download.json"],
                                    capture_output=True).stdout)["results"]["drug"]["label"]
    texts = {"inno": defaultdict(dict), "any": defaultdict(dict)}
    meta = {"inno": defaultdict(lambda: [0, set(), set()]), "any": defaultdict(lambda: [0, set(), set()])}
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
            txt = " ".join(ar)
            h = hashlib.sha1(txt.encode()).hexdigest()
            pool = "inno" if app.upper().startswith(("NDA", "BLA")) else "any"
            texts[pool][g][h] = txt
            m = meta[pool][g]; m[0] += 1; m[1].update(o.get("route") or []); m[2].add(app)
        del recs
        print(f"  partition {i + 1:>2}/{len(lab['partitions'])}: innovator generics {len(texts['inno']):,}, "
              f"other {len(texts['any']):,}", flush=True)
    rows = []
    for g in set(texts["inno"]) | set(texts["any"]):
        pool = "inno" if g in texts["inno"] else "any"
        t, m = texts[pool][g], meta[pool][g]
        rows.append(dict(generic_name=g, pool="innovator" if pool == "inno" else "all_labels",
                         n_labels=m[0], n_distinct_texts=len(t), routes=";".join(sorted(m[1])),
                         application_numbers=sorted(m[2])[:50],
                         adverse_reactions_text="\n\n".join(t.values())))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            aid = [x[0] for x in con.execute(
                "SELECT asset_id FROM prov_input_asset WHERE asset_id='openfda:drug_label_2026-09-24'").fetchall()]
        for r in rows:
            r["run_id"] = run.run_id
        run.archive_table("fda_label_pool", rows, inputs=aid, mode="replace_all")
    TMP.rmdir()
    ni = sum(r["pool"] == "innovator" for r in rows)
    print(f"\npooled {len(rows):,} generics ({ni:,} from innovator labels); "
          f"distinct texts {sum(r['n_distinct_texts'] for r in rows):,}; "
          f"text MB {sum(len(r['adverse_reactions_text']) for r in rows) / 1e6:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
