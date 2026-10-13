#!/usr/bin/env python3
"""Within-trial randomized dose contrasts (gate af_dose_response): does a higher dose raise the risk of
label-listed side effects in proportion to their attributable fraction?"""
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af_common as af  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

spec = importlib.util.spec_from_file_location("v2", ROOT / "scripts" / "52_af_v2.py")
V2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V2)
DOSE_RX = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg/kg|mg/m2|mg/m²|mg|mcg|µg|μg|ug|g|iu|units?)\b", re.I)
TO_MG = {"mg": 1.0, "g": 1000.0, "mcg": 1e-3, "µg": 1e-3, "μg": 1e-3, "ug": 1e-3}


def dose(text):
    m = DOSE_RX.search(text if isinstance(text, str) else "")
    if not m:
        return None
    v, u = float(m.group(1).replace(",", ".")), m.group(2).lower()
    if u in TO_MG:
        return v * TO_MG[u], "mg"
    return v, {"mg/m²": "mg/m2", "unit": "units"}.get(u, u)


def main() -> int:
    rows = V2.canon_rows()
    with pv.connect(read_only=True) as con:
        eg = con.execute("""SELECT c.contrast_id, c.nct, c.drug_ids[1] AS drug, e.eg_title, e.arm_label
                            FROM atlas_contrast c JOIN ctgov_event_group e ON e.nct=c.nct AND e.eg_id=c.eg_active
                            WHERE c.single_ingredient""").df()
    multi = eg.groupby(["nct", "drug"]).contrast_id.transform("size") >= 2
    eg = eg[multi].copy()
    need = set(eg.nct)
    desc = {}
    for f in sorted((ROOT / "cache" / "ctgov" / "pages").glob("page_*.json.gz")):
        for s in json.load(gzip.open(f))["studies"]:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            if nct in need:
                for a in (ps.get("armsInterventionsModule") or {}).get("armGroups") or []:
                    desc[(nct, a.get("label"))] = a.get("description") or ""
    ds = []
    for r in eg.itertuples():
        d = dose(r.eg_title) or dose(r.arm_label) or dose(desc.get((r.nct, r.arm_label), ""))
        ds.append(d)
    eg["dose"] = [d[0] if d else np.nan for d in ds]; eg["unit"] = [d[1] if d else None for d in ds]
    eg = eg.dropna(subset=["dose"])
    pick = []
    for (nct, drug), g in eg.groupby(["nct", "drug"]):
        if g.unit.nunique() != 1 or g.dose.nunique() < 2:
            continue
        pick.append((nct, drug, g.loc[g.dose.idxmax(), "contrast_id"], g.loc[g.dose.idxmin(), "contrast_id"],
                     g.dose.max() / g.dose.min()))
    pk = pd.DataFrame(pick, columns=["nct", "drug", "hi", "lo", "ratio"])
    print(f"trials with >=2 dose arms of one ingredient (parsed, same unit): {len(pk):,}; ingredients {pk.drug.nunique():,}; "
          f"median dose ratio {pk.ratio.median():.1f}")
    r = rows.set_index("contrast_id")
    hi = r.loc[r.index.intersection(pk.hi)].reset_index()[["contrast_id", "term", "a", "n1"]]
    lo = r.loc[r.index.intersection(pk.lo)].reset_index()[["contrast_id", "term", "a", "n1"]]
    hi = hi.merge(pk[["hi", "nct", "drug", "lo"]], left_on="contrast_id", right_on="hi")
    m = hi.merge(lo, left_on=["lo", "term"], right_on=["contrast_id", "term"], suffixes=("", "_lo"))
    m = m.rename(columns={"a": "a", "n1": "n1", "a_lo": "c", "n1_lo": "n0"})
    cells = []
    for (d, t), g in m.groupby(["drug", "term"]):
        if g.nct.nunique() < 2 or (g.a.sum() + g.c.sum()) < 20:
            continue
        lr, se = af.mh_rr(g)
        if lr != lr or se != se:
            continue
        cells.append(dict(drug=d, term=t, n_trials=int(g.nct.nunique()), events=float(g.a.sum() + g.c.sum()),
                          rr_hl=float(np.exp(lr)), rr_hl_lo=float(np.exp(lr - 1.96 * se)),
                          rr_hl_hi=float(np.exp(lr + 1.96 * se)), p=float(2 * stats.norm.sf(abs(lr / se)))))
    dc = pd.DataFrame(cells)
    with pv.connect(read_only=True) as con:
        lab = con.execute("""SELECT c.drug, c.term, c.af, c.af_lo, c.af_hi FROM af_cell c
                             JOIN af_label_match m USING (drug, term) WHERE m.on_fda""").df()
    x = dc.merge(lab, on=["drug", "term"])
    x["lhl"] = np.log(x.rr_hl)
    rho = stats.spearmanr(x.af, x.lhl)
    p_one = rho.pvalue / 2 if rho.statistic > 0 else 1 - rho.pvalue / 2
    clear = x[x.af_lo > .5]; bg = x[x.af_hi < .5]
    k = int((clear.lhl > 0).sum()); n = len(clear)
    p_sign = stats.binomtest(k, n, .5, alternative="greater").pvalue if n else np.nan
    fail = []
    if not (rho.statistic > 0 and p_one < .01):
        fail.append("a_spearman")
    if not (n and k / n > .5 and p_sign < .01):
        fail.append("b_sign")
    det = dict(n_trials_dose=int(len(pk)), n_dose_cells=int(len(dc)), n_label_cells_with_dose=int(len(x)),
               spearman=float(rho.statistic), p_one_sided=float(p_one),
               clear_n=n, clear_pos=k, clear_median_log_rr_hl=float(clear.lhl.median()) if n else None, p_sign=float(p_sign),
               bg_n=int(len(bg)), bg_median_log_rr_hl=float(bg.lhl.median()), bg_share_pos=float((bg.lhl > 0).mean()),
               median_dose_ratio=float(pk.ratio.median()), failed=fail)
    print(json.dumps(det, indent=1))
    print(f"-> af_dose_response {'PASS' if not fail else 'FAIL'}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [q[0] for q in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('atlas_contrast','ctgov_event_group','af_cell','af_label_match')").fetchall()]
        dc["run_id"] = run.run_id
        run.archive_table("af_dose_cell", dc, upstream=up, mode="replace_all")
        run.evaluate_gate("af_dose_response", float(len(fail)), not fail, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
