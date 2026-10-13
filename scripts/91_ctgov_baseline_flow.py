#!/usr/bin/env python3
"""AF 3.0 branch, covariates: download the baseline-characteristics and participant-flow modules of every
placebo-controlled trial in the atlas (ClinicalTrials.gov API v2, curl transport as in 32_ctgov_download.py), register
the raw pages, and parse trial-level age and sex and group-level completion and adverse-event withdrawal."""
import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

BASE = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,BaselineCharacteristicsModule,ParticipantFlowModule"
OUT = ROOT / "cache" / "ctgov" / "baseline_flow"
BATCH = 100


def fetch(ids, dest):
    if dest.exists():
        return
    url = f"{BASE}?filter.ids={','.join(ids)}&fields={FIELDS}&pageSize={len(ids)}&format=json"
    for attempt in range(5):
        r = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "300", url], capture_output=True)
        try:
            d = json.loads(r.stdout)
            if "studies" in d:
                with gzip.open(dest, "wt") as f:
                    json.dump(d, f)
                return
        except json.JSONDecodeError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed batch starting {ids[0]}")


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def total_group(groups):
    """The baseline group that is the trial total: titled 'Total', else the last group (registry convention)."""
    for g in groups:
        if (g.get("title") or "").strip().lower() == "total":
            return g["id"]
    return groups[-1]["id"] if groups else None


def parse(study):
    nct = study["protocolSection"]["identificationModule"]["nctId"]
    r = study.get("resultsSection", {})
    b = r.get("baselineCharacteristicsModule") or {}
    groups, measures = b.get("groups") or [], b.get("measures") or []
    tot = total_group(groups)
    rec = dict(nct=nct, age_value=np.nan, age_param=None, age_unit=None, pct_female=np.nan, n_baseline=np.nan)
    for dn in b.get("denoms") or []:
        for c in dn.get("counts") or []:
            if c.get("groupId") == tot:
                rec["n_baseline"] = num(c.get("value"))
    for m in measures:
        t = (m.get("title") or "").lower()
        cls = m.get("classes") or []
        if t.startswith("age") and "continuous" in t and m.get("paramType") in ("MEAN", "MEDIAN") and np.isnan(rec["age_value"]):
            for c in cls:
                for cat in c.get("categories") or []:
                    for x in cat.get("measurements") or []:
                        if x.get("groupId") == tot:
                            rec.update(age_value=num(x.get("value")), age_param=m.get("paramType"),
                                       age_unit=(m.get("unitOfMeasure") or "").lower())
        if (t.startswith("sex") or t.startswith("gender")) and np.isnan(rec["pct_female"]):
            f = mm = np.nan
            for c in cls:
                for cat in c.get("categories") or []:
                    for x in cat.get("measurements") or []:
                        if x.get("groupId") == tot:
                            ct = (cat.get("title") or "").lower()
                            if ct.startswith("female"):
                                f = num(x.get("value"))
                            elif ct.startswith("male"):
                                mm = num(x.get("value"))
            if f == f and mm == mm and f + mm > 0:
                rec["pct_female"] = f / (f + mm)
    fl = r.get("participantFlowModule") or {}
    fg = []
    for g in fl.get("groups") or []:
        fg.append(dict(nct=nct, fg_id=g["id"], fg_title=g.get("title"), started=0.0, completed=0.0,
                       drop_total=0.0, drop_ae=0.0))
    idx = {g["fg_id"]: g for g in fg}
    periods = fl.get("periods") or []
    for pi, p in enumerate(periods):
        for ms in p.get("milestones") or []:
            typ = (ms.get("type") or "").upper()
            for a in ms.get("achievements") or []:
                g = idx.get(a.get("groupId"))
                if g is None:
                    continue
                if typ == "STARTED" and pi == 0:
                    g["started"] += np.nan_to_num(num(a.get("numSubjects")))
                if typ == "COMPLETED" and pi == len(periods) - 1:
                    g["completed"] += np.nan_to_num(num(a.get("numSubjects")))
        for dw in p.get("dropWithdraws") or []:
            ae = "adverse" in (dw.get("type") or "").lower()
            for a in dw.get("reasons") or []:
                g = idx.get(a.get("groupId"))
                if g is None:
                    continue
                v = np.nan_to_num(num(a.get("numSubjects")))
                g["drop_total"] += v
                if ae:
                    g["drop_ae"] += v
    return rec, fg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with pv.connect(read_only=True) as con:
        ncts = sorted(r[0] for r in con.execute("SELECT DISTINCT nct FROM atlas_contrast").fetchall())
    batches = [ncts[i:i + BATCH] for i in range(0, len(ncts), BATCH)]
    for i, b in enumerate(batches):
        fetch(b, OUT / f"batch_{i:04d}.json.gz")
        if i % 20 == 0:
            print(f"batch {i + 1}/{len(batches)}", flush=True)
    base, flow, seen = [], [], set()
    for f in sorted(OUT.glob("batch_*.json.gz")):
        for s in json.load(gzip.open(f))["studies"]:
            rec, fg = parse(s)
            seen.add(rec["nct"]); base.append(rec); flow += fg
    B, F = pd.DataFrame(base).drop_duplicates("nct"), pd.DataFrame(flow).drop_duplicates(["nct", "fg_id"])
    print(f"trials requested {len(ncts):,}, returned {len(seen):,}; age {B.age_value.notna().mean():.1%}, "
          f"sex {B.pct_female.notna().mean():.1%}; flow groups {len(F):,} in {F.nct.nunique():,} trials")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        aid = run.register_input("ctgov:baseline_flow_modules", asset_kind="api_snapshot", local_path=OUT,
                                 url=f"{BASE}?filter.ids=<atlas trials>&fields={FIELDS}", source_registry="ClinicalTrials.gov",
                                 genome_build="not_applicable", build_evidence="trial registry records; no genomic coordinates",
                                 licence="public domain (U.S. National Library of Medicine terms of use)",
                                 licence_url="https://clinicaltrials.gov/about-site/terms-conditions",
                                 notes=f"{len(batches)} batches of <= {BATCH} NCT IDs; retrieved {time.strftime('%Y-%m-%d')}")
        with pv.connect(read_only=True) as con:
            up = [r[0] for r in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name='atlas_contrast'").fetchall()]
        B["run_id"] = run.run_id; F["run_id"] = run.run_id
        run.archive_table("ctgov_baseline", B, inputs=[aid], upstream=up, mode="replace_all")
        run.archive_table("ctgov_flow_group", F, inputs=[aid], upstream=up, mode="replace_all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
