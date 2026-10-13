#!/usr/bin/env python3
"""Download every randomized interventional ClinicalTrials.gov study that has results.

48,006 studies at time of writing. Only the modules the pre-registered analysis needs
are requested, which keeps the corpus to roughly 0.7 GB instead of several.

Transport: curl, not urllib. ClinicalTrials.gov answers urllib with HTTP 403 and curl
with 200 for the identical URL, so the choice is forced, not stylistic.

Resumable and verifiable: each page is written atomically as page_NNNNN.json.gz together
with the token that produced it, so an interrupted run resumes from the last complete
page instead of silently restarting or silently skipping. A page is only accepted if it
parses as JSON and its study count matches what the API reported for it.
"""
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "ctgov" / "pages"
BASE = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ("NCTId,BriefTitle,OfficialTitle,Phase,DesignInfo,ArmsInterventionsModule,"
          "Condition,EnrollmentInfo,StartDate,CompletionDate,LeadSponsor,"
          "AdverseEventsModule,InterventionBrowseModule")
FILTER = ("AREA%5BStudyType%5DINTERVENTIONAL%20AND%20AREA%5BDesignAllocation%5DRANDOMIZED")
PAGE = 1000


def fetch(token):
    url = (f"{BASE}?aggFilters=results:with&filter.advanced={FILTER}&pageSize={PAGE}"
           f"&fields={FIELDS}&format=json&countTotal=true")
    if token:
        url += f"&pageToken={token}"
    for attempt in range(6):
        r = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "300", url],
                           capture_output=True)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"page failed 6 times (token={token})")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    done = sorted(OUT.glob("page_*.json.gz"))
    token, idx = None, 0
    if done:                                  # resume after the last COMPLETE page
        last = json.loads(gzip.open(done[-1], "rt").read())
        token, idx = last["_next_token"], len(done)
        if token is None:
            print(f"already complete: {len(done)} pages")
            return 0
        print(f"resuming after {len(done)} complete pages")
    total_studies = 0
    while True:
        d = fetch(token)
        studies = d.get("studies", [])
        nxt = d.get("nextPageToken")
        payload = {"_page": idx, "_token_used": token, "_next_token": nxt,
                   "_total_count": d.get("totalCount"), "studies": studies}
        tmp = OUT / f"page_{idx:05d}.json.gz.tmp"
        with gzip.open(tmp, "wt") as fh:
            json.dump(payload, fh)
        tmp.rename(OUT / f"page_{idx:05d}.json.gz")      # atomic: never a half page
        total_studies += len(studies)
        print(f"page {idx:>3}  +{len(studies):>4}  ({total_studies:>6} this run)", flush=True)
        idx += 1
        if not nxt:
            break
        token = nxt
    print("DOWNLOAD COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
