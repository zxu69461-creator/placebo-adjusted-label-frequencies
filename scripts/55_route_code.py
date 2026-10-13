#!/usr/bin/env python3
"""Route family per single-ingredient contrast (gate af_route_coding).

T1 interventions of the active arm that map to the drug of interest; T2 arm description +
event-group title/description; T3 the single route family of all current FDA labels of the
ingredient (needs fda_label_route_pool, script 56). Reads the cached CT.gov pages.
"""
import gzip
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402
from oajoint.routes import decide, families_in  # noqa: E402

spec = importlib.util.spec_from_file_location("m34", ROOT / "scripts" / "34_ctgov_parse.py")
M34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M34)
OK = ("exact", "research_code", "combination_parts", "token")


def main() -> int:
    with pv.connect(read_only=True) as con:
        ct = con.execute("""
            SELECT c.contrast_id, c.nct, c.drug_ids[1] AS drug, c.eg_active, e.arm_label
            FROM atlas_contrast c LEFT JOIN ctgov_event_group e ON e.nct = c.nct AND e.eg_id = c.eg_active
            WHERE c.single_ingredient AND c.n_act_other > 0 AND c.n_pl_other > 0""").df()
        pool = con.execute("SELECT generic_name, route_family FROM fda_label_route_pool").df()
    idx, _ = M34.build_index()
    need = set(ct.nct)
    study = {}
    for f in sorted((ROOT / "cache" / "ctgov" / "pages").glob("page_*.json.gz")):
        for s in json.load(gzip.open(f))["studies"]:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            if nct in need:
                ai = ps.get("armsInterventionsModule") or {}
                eg = ((s.get("resultsSection") or {}).get("adverseEventsModule") or {}).get("eventGroups") or []
                study[nct] = (ai.get("armGroups") or [], ai.get("interventions") or [],
                              {g.get("id"): g for g in eg})
    fam_by_drug = defaultdict(set)
    for g, fam in zip(pool.generic_name, pool.route_family):
        hit, how = M34.match(g, [], idx)
        if hit and how in ("exact", "research_code", "combination_parts"):
            for c in hit:
                fam_by_drug[c].add(fam)
    out = []
    for r in ct.itertuples():
        arms, ivs, egs = study.get(r.nct, ([], [], {}))
        arm = next((a for a in arms if a.get("label") == r.arm_label), None)
        mine = []
        for iv in ivs:
            if arm is not None and r.arm_label not in (iv.get("armGroupLabels") or []):
                continue
            names = [iv.get("name") or ""] + list(iv.get("otherNames") or [])
            if any(r.drug in (M34.match(n, [], idx)[0] or set()) and M34.match(n, [], idx)[1] in OK
                   for n in names):
                mine.append(" ".join(names + [iv.get("description") or ""]))
        fam, tier, ev = decide(families_in(" ".join(mine))), "T1", " | ".join(mine)[:300]
        if fam is None:
            g = egs.get(r.eg_active) or {}
            txt = " ".join([(arm or {}).get("description") or "", g.get("title") or "", g.get("description") or ""])
            fam, tier, ev = decide(families_in(txt)), "T2", txt[:300]
        if fam is None:
            lf = fam_by_drug.get(r.drug, set())
            fam, tier, ev = (next(iter(lf)), "T3", "label routes: " + ",".join(sorted(lf))) if len(lf) == 1 else \
                ("unresolved", "none", "label routes: " + ",".join(sorted(lf)))
        out.append(dict(contrast_id=r.contrast_id, nct=r.nct, drug=r.drug, route_family=fam, tier=tier, evidence=ev))
    df = pd.DataFrame(out)
    coded = ~df.route_family.isin(["mixed", "unresolved"])
    frac, frac_text = float(coded.mean()), float((coded & df.tier.isin(["T1", "T2"])).mean())
    print(f"contrasts {len(df):,}; trials found in cache {len(study):,}/{len(need):,}")
    print("tier x family:\n", pd.crosstab(df.route_family, df.tier, margins=True).to_string())
    print(f"\ncoded (T1-T3) {frac:.1%}; coded from trial text (T1-T2) {frac_text:.1%} "
          f"-> af_route_coding {'PASS' if frac >= .70 else 'FAIL'}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('atlas_contrast','ctgov_event_group','fda_label_route_pool')").fetchall()]
            aid = [x[0] for x in con.execute("SELECT asset_id FROM prov_input_asset WHERE asset_id LIKE 'ctgov%'").fetchall()]
        df["run_id"] = run.run_id
        run.archive_table("atlas_contrast_route", df, inputs=aid, upstream=up, mode="replace_all")
        run.evaluate_gate("af_route_coding", frac, frac >= .70, detail={
            "n_contrasts": int(len(df)), "frac_text_T1_T2": frac_text,
            "tier_counts": dict(Counter(df.tier)), "family_counts": dict(Counter(df.route_family))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
