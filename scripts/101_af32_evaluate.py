#!/usr/bin/env python3
"""Evaluate af32_confirm, af32_seen, af32_calibration, af32_adopt, gen_newdrug and gen_serious from the archived
BENCH-2.1 scores (hierarchical placebo scoring), exactly as registered by 99b_af32_prereg.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

V, PREV = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("AF 3.2", "AF 3.0")
PRE, GPRE = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("af32", "gen")
CBENCH = sys.argv[5] if len(sys.argv) > 5 else "bench21_confirm"
TAG = "" if V == "AF 3.2" else V.replace("AF ", "").replace(".", "")


def rows(con, bench, stratum):
    return con.execute("""SELECT method, n_records, mean_logscore, diff_vs_ref, diff_ci_lo, diff_ci_hi, pit_central90
                          FROM bench_score WHERE bench = ? AND win = 'pooled' AND stratum = ? AND scoring = 'hier'
                          AND method NOT LIKE '%no dose offsets'""", [bench, stratum]).df().set_index("method")


def comp(t, others):
    res = {m: dict(diff=float(t.loc[m, "diff_vs_ref"]), lo=float(t.loc[m, "diff_ci_lo"]), hi=float(t.loc[m, "diff_ci_hi"]),
                   passed=bool(t.loc[m, "diff_ci_lo"] > 0)) for m in others if m in t.index}
    missing = [m for m in others if m not in t.index]
    return res, missing


def main() -> int:
    with pv.connect(read_only=True) as con:
        c_sup = rows(con, CBENCH, "AF1-support"); c_seen = rows(con, CBENCH, "seen")
        g_new = rows(con, f"generality_newdrug{TAG}", "unseen"); g_ser = rows(con, f"generality_serious{TAG}", "AF1-support")
        sim = con.execute("SELECT passed FROM prov_decision WHERE gate_name=?", [f"{PRE}_sim"]).fetchone()[0]
    out = {}
    extra = ["B6 class-mean", "B7 class-Bayes"] if PRE != "af32" else []
    r, miss = comp(c_sup, ["B0 null", "B1 MH", "B2 PM-t", "B3 Bayes-NN", "B4 AF 2.0", "B5 term-only"] + extra + [PREV])
    nf = sum(not x["passed"] for x in r.values()) + len(miss)
    out[f"{PRE}_confirm"] = (float(nf), nf == 0, dict(comparisons=r, missing=miss, n_records=int(c_sup.loc[V, "n_records"])))
    r2, miss2 = comp(c_seen, ["B0 null", "B2 PM-t", "B3 Bayes-NN", "B5 term-only"] + extra + [PREV])
    nf2 = sum(not x["passed"] for x in r2.values()) + len(miss2)
    out[f"{PRE}_seen"] = (float(nf2), nf2 == 0, dict(comparisons=r2, missing=miss2))
    cal = float(c_sup.loc[V, "pit_central90"])
    out[f"{PRE}_calibration"] = (cal, .87 <= cal <= .93, dict(table=c_sup.pit_central90.to_dict()))
    flags = {f"{PRE}_sim": bool(sim), f"{PRE}_confirm": out[f"{PRE}_confirm"][1], f"{PRE}_calibration": out[f"{PRE}_calibration"][1]}
    na = sum(not v for v in flags.values())
    out[f"{PRE}_adopt"] = (float(na), na == 0, dict(flags=flags))
    r3, miss3 = comp(g_new, ["B6 class-mean", "B5 term-only", "B0 null"])
    cal3 = float(g_new.loc[V, "pit_central90"])
    nf3 = sum(not x["passed"] for x in r3.values()) + len(miss3) + int(not .85 <= cal3 <= .95)
    out[f"{GPRE}_newdrug"] = (float(nf3), nf3 == 0, dict(comparisons=r3, missing=miss3, calibration=cal3, n_records=int(g_new.loc[V, "n_records"])))
    r4, miss4 = comp(g_ser, ["B0 null", "B1 MH", "B2 PM-t", "B3 Bayes-NN", "B5 term-only"])
    cal4 = float(g_ser.loc[V, "pit_central90"])
    nf4 = sum(not x["passed"] for x in r4.values()) + len(miss4) + int(not .85 <= cal4 <= .95)
    out[f"{GPRE}_serious"] = (float(nf4), nf4 == 0, dict(comparisons=r4, missing=miss4, calibration=cal4, n_records=int(g_ser.loc[V, "n_records"])))
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for g, (v, p, det) in out.items():
            run.evaluate_gate(g, v, p, detail=det)
    for g, (v, p, det) in out.items():
        print(g, v, "PASS" if p else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
