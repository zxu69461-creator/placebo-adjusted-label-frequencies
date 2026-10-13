#!/usr/bin/env python3
"""Gate jno_discontinuation: discontinuation because of adverse events, drug versus placebo (rule in 115_jno_prereg.py).

Flow groups are linked to the active and placebo event groups of each single-ingredient contrast by exact title (or arm
label) after lower-casing and trimming; a contrast is evaluable when every linked arm has exactly one flow group with
participants started > 0 and adverse-event withdrawals recorded. Placebo counts shared by m evaluable contrasts of a
trial are divided by m. Pooled with the Mantel-Haenszel risk ratio (Greenland-Robins variance) and risk difference
(Sato variance, per 100 participants).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

ATC1 = {"A": "alimentary tract and metabolism", "B": "blood", "C": "cardiovascular", "D": "dermatological",
        "G": "genito-urinary and sex hormones", "H": "systemic hormones", "J": "anti-infectives",
        "L": "antineoplastic and immunomodulating", "M": "musculoskeletal", "N": "nervous system",
        "P": "antiparasitic", "R": "respiratory", "S": "sensory organs", "V": "various"}


def mh(d):
    """Mantel-Haenszel risk ratio (Greenland-Robins variance) and risk difference (Sato variance), 95% CIs."""
    a, n1, c, n0 = (d[k].values.astype(float) for k in ("x_drug", "n_drug", "x_pl", "n_pl"))
    N = n1 + n0
    R, S = a * n0 / N, c * n1 / N
    rr = R.sum() / S.sum()
    se = np.sqrt(((n1 * n0 * (a + c) - a * c * N) / N ** 2).sum() / (R.sum() * S.sum()))
    w = n1 * n0 / N
    rd = ((a * n0 - c * n1) / N).sum() / w.sum()
    P = (n1 ** 2 * c - n0 ** 2 * a + n1 * n0 * (n0 - n1) / 2) / N ** 2
    Q = (a * (n0 - c) + c * (n1 - a)) / (2 * N)
    se_rd = np.sqrt((rd * P.sum() + Q.sum()) / w.sum() ** 2)
    return dict(rr=float(rr), rr_lo=float(rr * np.exp(-1.96 * se)), rr_hi=float(rr * np.exp(1.96 * se)),
                rd=float(100 * rd), rd_lo=float(100 * (rd - 1.96 * se_rd)), rd_hi=float(100 * (rd + 1.96 * se_rd)),
                ef=float(1 - 1 / rr), ef_lo=float(1 - np.exp(1.96 * se) / rr), ef_hi=float(1 - np.exp(-1.96 * se) / rr),
                per100_drug=float(100 * a.sum() / n1.sum()), per100_pl=float(100 * c.sum() / n0.sum()),
                n_contrasts=int(len(d)), n_trials=int(d.nct.nunique()), n_drug=float(n1.sum()), n_pl=float(n0.sum()))


def main() -> int:
    with pv.connect(read_only=True) as con:
        ct = con.execute("""SELECT contrast_id, nct, eg_active, eg_placebo, list_extract(drug_ids, 1) AS drug,
                                   list_extract(atc4, 1) AS atc4 FROM atlas_contrast WHERE single_ingredient""").df()
        eg = con.execute("SELECT nct, eg_id, lower(trim(eg_title)) AS t, lower(trim(arm_label)) AS a "
                         "FROM ctgov_event_group").df()
        fg = con.execute("SELECT nct, fg_id, lower(trim(fg_title)) AS t, started, drop_ae FROM ctgov_flow_group").df()
    titles = {(n, e): {x for x in (t, a) if isinstance(x, str) and x} for n, e, t, a in zip(eg.nct, eg.eg_id, eg.t, eg.a)}
    fgs = defaultdict(list)
    for r in fg.itertuples():
        fgs[r.nct].append(r)

    def link(nct, eg_id):
        ts = titles.get((nct, eg_id), set())
        hit = [r for r in fgs.get(nct, []) if isinstance(r.t, str) and r.t in ts]
        if len(hit) != 1:
            return None
        r = hit[0]
        return r if (r.started or 0) > 0 and r.drop_ae == r.drop_ae else None

    ev = []
    for r in ct.itertuples():
        a = link(r.nct, r.eg_active)
        ps = [link(r.nct, p) for p in r.eg_placebo]
        if a is None or not ps or any(p is None for p in ps):
            continue
        ev.append(dict(contrast_id=r.contrast_id, nct=r.nct, drug=r.drug, atc1=(r.atc4 if isinstance(r.atc4, str) and r.atc4 else "?")[0],
                       x_drug=float(a.drop_ae), n_drug=float(a.started), pl_fgs=tuple(sorted(p.fg_id for p in ps)),
                       x_pl_raw=float(sum(p.drop_ae for p in ps)), n_pl_raw=float(sum(p.started for p in ps))))
    E = pd.DataFrame(ev)
    use = Counter((n, f) for n, fs in zip(E.nct, E.pl_fgs) for f in fs)
    E["m"] = [max(use[(n, f)] for f in fs) for n, fs in zip(E.nct, E.pl_fgs)]
    E["x_pl"], E["n_pl"] = E.x_pl_raw / E.m, E.n_pl_raw / E.m
    E = E[(E.x_drug + E.x_pl) >= 0]
    det = dict(n_contrasts_single=int(len(ct)), n_evaluable=int(len(E)), n_trials=int(E.nct.nunique()),
               n_drugs=int(E.drug.nunique()), n_with_any_event=int(((E.x_drug + E.x_pl) > 0).sum()))
    # all evaluable contrasts, as registered: strata without any withdrawal add nothing to the MH risk ratio but count
    # in the pooled rates and the MH risk difference (first run on 2026-10-12 used only strata with withdrawals; fixed)
    ok = E
    det["overall"] = mh(ok)
    det["overall"]["n_contrasts_all"] = int(len(E))
    by = {}
    for g, d in ok.groupby("atc1"):
        if d.nct.nunique() >= 10:
            by[g] = mh(d) | dict(label=ATC1.get(g, g))
    det["by_atc1"] = by
    drugs = {}
    for g, d in ok.groupby("drug"):
        if len(d) >= 5:
            drugs[g] = mh(d)
    det["n_drugs_ge5"] = len(drugs)
    efs = np.array([v["ef"] for v in drugs.values()])
    efs = efs[np.isfinite(efs)]                      # drugs without any withdrawal have no defined risk ratio
    det["n_drugs_ge5_defined"] = int(len(efs))
    det["drug_ef_median_iqr"] = [float(np.quantile(efs, q)) for q in (.25, .5, .75)] if len(efs) else []
    det["drug_share_ef_lt_half"] = float((efs < .5).mean()) if len(efs) else None
    print({k: v for k, v in det.items() if k not in ("by_atc1",)})
    for g, v in by.items():
        print(f"  {g} {v['label'][:28]:28s} trials {v['n_trials']:5d} RR {v['rr']:.2f} EF {v['ef']:.2f} "
              f"({v['ef_lo']:.2f}-{v['ef_hi']:.2f}) per100 {v['per100_drug']:.2f} vs {v['per100_pl']:.2f}")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('atlas_contrast','ctgov_event_group','ctgov_flow_group')").fetchall()]
        out = E.rename(columns={"m": "n_contrasts_in_trial"})[["contrast_id", "nct", "drug", "atc1",
                                                                "n_contrasts_in_trial", "x_drug", "n_drug", "x_pl",
                                                                "n_pl"]].copy()
        out["run_id"] = run.run_id
        run.archive_table("jno_discontinuation_contrast", out, upstream=up, mode="replace_all")
        rows = [dict(analysis="jno_discontinuation", set_name="overall", grp="all", n_pairs=det["overall"]["n_contrasts"],
                     n_drugs=det["n_drugs"], metric=k, value=det["overall"][k],
                     lo=det["overall"].get(k + "_lo", np.nan), hi=det["overall"].get(k + "_hi", np.nan))
                for k in ("rr", "rd", "ef", "per100_drug", "per100_pl")]
        for g, v in by.items():
            rows += [dict(analysis="jno_discontinuation", set_name="atc1", grp=g, n_pairs=v["n_contrasts"],
                          n_drugs=0, metric=k, value=v[k], lo=v.get(k + "_lo", np.nan), hi=v.get(k + "_hi", np.nan))
                     for k in ("rr", "ef", "per100_drug", "per100_pl")]
        sm = pd.DataFrame(rows); sm["run_id"] = run.run_id
        with pv.connect() as con:
            con.execute("DELETE FROM jno_summary WHERE analysis = 'jno_discontinuation'")
        run.archive_table("jno_summary", sm, upstream=up)
        run.evaluate_gate("jno_discontinuation", det["overall"]["ef"], None, detail=det)
    return 0


if __name__ == "__main__":
    sys.exit(main())
