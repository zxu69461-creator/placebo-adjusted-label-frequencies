#!/usr/bin/env python3
"""Robustness (af_robustness) and determinants (af_d1_subjective, af_d2_by_soc) for
the attributable fractions, exactly as pre-registered by script 45."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import af_common as af  # noqa: E402
from oajoint import provenance as pv  # noqa: E402

SETS = json.loads((ROOT / "config" / "af_symptom_sets.json").read_text())


def norm_soc(s):
    s = (s or "").strip().lower()
    return "general disorders" if s.startswith("general disorders") else s


def main() -> int:
    rows = af.load_rows()
    label, mapped = af.label_pairs()
    print(f"rows {len(rows):,}; SIDER label pairs {len(label):,}")
    blinded = rows.masking.isin(["DOUBLE", "TRIPLE", "QUADRUPLE"])
    complete = rows.thr.fillna(-1) == 0
    runs = {
        "MAIN (reference)": af.estimate(rows),
        "R1 blinded only": af.estimate(rows[blinded]),
        "R2 complete reporting (threshold 0)": af.estimate(rows[complete]),
        "R3 >=3 trials, >=30 events": af.estimate(rows, 3, 30),
        "R4 DerSimonian-Laird": af.estimate(rows, method="DL"),
    }
    res = {}
    print(f"\n{'analysis':<38}{'cells':>7}{'label':>7}{'AF_hi<.5':>10}{'AF<=0':>8}{'AF<.5':>8}{'validity P':>12}")
    for k, cells in runs.items():
        s = af.h1_summary(cells, label, mapped); s["n_cells"] = len(cells); res[k] = s
        print(f"{k:<38}{len(cells):>7,}{s['n_label']:>7,}{s['frac_afhi_lt_half']:>10.1%}"
              f"{s['frac_af_le_zero']:>8.1%}{s['frac_af_lt_half']:>8.1%}{s['validity_p']:>12.1e}")
    sens = {k: v for k, v in res.items() if not k.startswith("MAIN")}
    holds = {k: bool(v["frac_afhi_lt_half"] >= 0.35 and v["frac_af_le_zero"] >= 0.20
                     and v["validity_p"] < 0.001) for k, v in sens.items()}
    robust = all(holds.values())
    mn = min(v["frac_afhi_lt_half"] for v in sens.values())
    print(f"\n-> af_robustness: {'PASS' if robust else 'FAIL'}  (holds: {holds})")

    # ---- determinants on the MAIN analysis ------------------------------------------
    main_cells = runs["MAIN (reference)"]
    with pv.connect(read_only=True) as con:
        soc = con.execute("""
            SELECT t, organ_system FROM (
              SELECT lower(trim(regexp_replace(term,'\\s+',' ','g'))) AS t, organ_system,
                     row_number() OVER (PARTITION BY lower(trim(regexp_replace(term,'\\s+',' ','g')))
                                        ORDER BY count(*) DESC) AS rk
              FROM v_ctgov_ae WHERE kind='other' GROUP BY 1, 2) WHERE rk = 1""").df()
    socmap = {t: norm_soc(s) for t, s in zip(soc.t, soc.organ_system)}
    lab = main_cells[[(d, t) in label for d, t in zip(main_cells.drug, main_cells.term)]].copy()
    lab["soc"] = lab.term.map(socmap)
    subj = lab[lab.term.isin(SETS["subjective"])]
    obj = lab[lab.soc.isin(SETS["objective_soc"]) & ~lab.term.isin(SETS["subjective"])]
    mw = stats.mannwhitneyu(subj.af, obj.af, alternative="less")
    dd = float(subj.af.median() - obj.af.median())
    d1 = bool(mw.pvalue < 0.001 and dd < 0)
    print(f"\nD1: label-listed SUBJECTIVE symptoms median AF {subj.af.median():+.3f} (n={len(subj)}) vs "
          f"OBJECTIVE-SOC events {obj.af.median():+.3f} (n={len(obj)});  one-sided MW P = {mw.pvalue:.1e}"
          f"  -> {'PASS' if d1 else 'FAIL'}")
    print("\nD2: by system organ class (label-listed cells, >=30):")
    d2 = (lab.groupby("soc").agg(n=("af", "size"), median_af=("af", "median"),
                                 frac_afhi_lt_half=("af_hi", lambda x: float((x < .5).mean())))
             .query("n >= 30").sort_values("median_af"))
    print(d2.to_string(float_format=lambda v: f"{v:.3f}"))

    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.evaluate_gate("af_robustness", float(mn), robust,
                          detail={k: v for k, v in res.items()} | {"holds": holds})
        run.evaluate_gate("af_d1_subjective", float(mw.pvalue), d1,
                          detail={"median_subjective": float(subj.af.median()),
                                  "median_objective": float(obj.af.median()),
                                  "n_subjective": int(len(subj)), "n_objective": int(len(obj))})
        run.evaluate_gate("af_d2_by_soc", None, True,
                          detail=json.loads(d2.reset_index().to_json(orient="records")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
