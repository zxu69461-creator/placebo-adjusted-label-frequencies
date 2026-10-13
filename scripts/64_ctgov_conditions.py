#!/usr/bin/env python3
"""Condition MeSH terms for every trial in the direction-C atlas (gate af_indication).
CT.gov API v2, conditionBrowseModule only; ~100 ids per request. Nothing is written to disk but the DB."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

HEALTHY = ("healthy", "healthy volunteers", "healthy subjects", "healthy volunteer", "healthy participants")


def main() -> int:
    with pv.connect(read_only=True) as con:
        tr = con.execute("""SELECT DISTINCT t.nct, t.conditions FROM atlas_contrast c JOIN ctgov_trial t USING (nct)
                            WHERE c.single_ingredient""").df()
    ids = tr.nct.tolist(); got = {}
    for i in range(0, len(ids), 100):
        url = ("https://clinicaltrials.gov/api/v2/studies?filter.ids=" + ",".join(ids[i:i + 100]) +
               "&fields=NCTId,ConditionBrowseModule&pageSize=100")
        for attempt in range(4):
            out = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "120", url], capture_output=True)
            try:
                js = json.loads(out.stdout); break
            except Exception:
                time.sleep(5)
        for s in js.get("studies", []):
            nct = s["protocolSection"]["identificationModule"]["nctId"]
            cb = (s.get("derivedSection") or {}).get("conditionBrowseModule") or {}
            got[nct] = ([m["term"] for m in cb.get("meshes") or []], [m["term"] for m in cb.get("ancestors") or []])
        if i % 2000 == 0:
            print(f"  {i + 100:>6,}/{len(ids):,}", flush=True)
    rows = []
    for nct, cond in zip(tr.nct, tr.conditions):
        m, a = got.get(nct, ([], []))
        cond = [] if cond is None else list(cond)
        healthy = len(cond) > 0 and all((c or "").strip().lower() in HEALTHY for c in cond)
        ind = "healthy volunteers" if healthy else (m[0] if m else None)
        rows.append(dict(nct=nct, indication=ind, meshes=m, ancestors=a))
    df = pd.DataFrame(rows)
    print(f"trials {len(df):,}; fetched {len(got):,}; with indication {df.indication.notna().sum():,}; "
          f"healthy {int((df.indication == 'healthy volunteers').sum()):,}; distinct indications {df.indication.nunique():,}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            aid = [x[0] for x in con.execute("SELECT asset_id FROM prov_input_asset WHERE asset_id LIKE 'ctgov%'").fetchall()]
        df["run_id"] = run.run_id
        run.archive_table("ctgov_condition", df, inputs=aid, mode="replace_all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
