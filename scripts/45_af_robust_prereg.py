#!/usr/bin/env python3
"""Pre-register robustness and determinant analyses of the attributable fractions, before computing them.

Registered after af_main / af_label_validity / af_h1_label were evaluated; none of the
analyses below has been run. The subsets were inspected for SIZE only (masking counts,
frequency-threshold counts, organ-system names), never for effect estimates.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

SUBJECTIVE = ["headache", "fatigue", "dizziness", "nausea", "insomnia", "somnolence",
              "asthenia", "malaise", "myalgia", "arthralgia", "back pain", "pain in extremity",
              "abdominal pain", "abdominal pain upper", "dyspepsia", "diarrhoea", "constipation",
              "anxiety", "paraesthesia", "dry mouth", "vertigo", "pruritus", "palpitations"]
OBJECTIVE_SOC = ["investigations", "infections and infestations",
                 "blood and lymphatic system disorders"]

GATES = [
 ("af_robustness", "pre_registration",
  "The H1 conclusion is ROBUST only if it holds in EVERY one of: R1 blinded trials only "
  "(masking DOUBLE/TRIPLE/QUADRUPLE); R2 trials that report non-serious AEs COMPLETELY "
  "(frequency threshold 0), which removes the threshold trap entirely; R3 stricter "
  "estimability (>=3 trials, >=30 events); R4 DerSimonian-Laird random-effects log-RR per "
  "cell (0.5 correction only for contrasts with a single zero cell; double zeros dropped). "
  "'Holds' = among label-listed estimable cells, fraction with AF 95% upper bound < 0.5 is "
  ">= 0.35 AND fraction with AF <= 0 is >= 0.20 AND label-listed log-RR exceeds non-listed "
  "log-RR (one-sided Mann-Whitney P < 0.001). Metric = the smallest AF_hi<0.5 fraction.",
  "min_frac_afhi_lt_half", ">=", 0.35),
 ("af_d1_subjective", "pre_registration",
  "D1: among label-listed estimable cells (main analysis), side effects in the pre-"
  f"specified SUBJECTIVE nonspecific-symptom set {SUBJECTIVE} have LOWER attributable "
  "fractions than side effects in OBJECTIVELY ascertained system organ classes "
  f"{OBJECTIVE_SOC} (each term assigned its modal SOC, case-normalised). One-sided "
  "Mann-Whitney P < 0.001 with a negative median difference.", "mw_p_one_sided", "<", 0.001),
 ("af_d2_by_soc", "pre_registration",
  "D2, descriptive: median AF and the AF_hi<0.5 fraction of label-listed cells for every "
  "system organ class with >=30 label-listed cells.", None, None, None),
]


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        for name, fam, rule, metric, cmp_, thr in GATES:
            run.register_gate(name, rule, gate_family=fam, metric_name=metric,
                              comparator=cmp_, threshold=thr,
                              notes="registered before any robustness or determinant estimate")
            print(f"  registered {name}")
    import json
    (ROOT / "config" / "af_symptom_sets.json").write_text(
        json.dumps({"subjective": SUBJECTIVE, "objective_soc": OBJECTIVE_SOC}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
