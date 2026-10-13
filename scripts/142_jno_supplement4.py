#!/usr/bin/env python3
"""Supplement 3 of the JAMA Network Open manuscript (numbered in the order of citation): estimates for every adverse reaction listed on a current FDA label
(the primary analysis set), with a README sheet defining each column. Output: manuscript/jama/supplement3_listed_reactions.xlsx.
Estimates for all 214 010 drug-adverse event pairs are kept in the study database for the public code and data
release."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

OUT = ROOT / "manuscript" / "jama" / "supplement3_listed_reactions.xlsx"
SECTION = {"trial": "trials", "postmarketing_only": "postmarketing", "other_only": "other", "unsegmented": "unsegmented"}
COLS = [
    ("drug", "Drug (international nonproprietary name)"),
    ("adverse_reaction", "Adverse event term as posted on ClinicalTrials.gov (mostly MedDRA preferred terms, US spelling)"),
    ("organ_system", "MedDRA system organ class"),
    ("label_subsection", "Subsection of the current FDA label in which the term was matched: trials (clinical trials "
                         "experience), postmarketing (postmarketing experience only), other (other text only), "
                         "unsegmented (section without subsections)"),
    ("trials", "Number of placebo-controlled trials contributing to the estimate"),
    ("excess_fraction", "Posterior mean excess fraction, 1 - 1/HR, in a reference trial (12 weeks, double-blind, "
                        "systematic assessment, industry sponsored, phase 2 to 4, in patients, at the median dose)"),
    ("excess_fraction_lo", "Lower limit of the 95% credible interval of the excess fraction"),
    ("excess_fraction_hi", "Upper limit of the 95% credible interval of the excess fraction"),
    ("prob_excess_fraction_lt_0.5", "Posterior probability that the excess fraction is below 0.5 (which implies a drug frequency "
                                    "less than twice the placebo frequency)"),
    ("placebo_per_100", "Frequency with placebo per 100 patients, pooled over the drug's placebo arms in trials that reported "
                        "the reaction"),
    ("excess_per_100", "Expected excess per 100 patients implied by the posterior (expected frequency with the drug "
                       "= placebo_per_100 + excess_per_100)"),
]


def main() -> int:
    with pv.connect(read_only=True) as con:
        d = con.execute("""
            SELECT coalesce(n.display_name, p.drug) AS drug, p.drug AS chembl_id, p.term AS adverse_reaction,
                   p.soc_h AS organ_system, s.section_class, p.n_trials AS trials, e.af_mean AS excess_fraction,
                   p.af_hz_lo AS excess_fraction_lo, p.af_hz_hi AS excess_fraction_hi,
                   p.p_af_lt_half AS "prob_excess_fraction_lt_0.5", p.p_no_excess AS prob_no_excess,
                   p.per100_pl AS placebo_per_100, e.per100_drug_exp AS drug_per_100, e.per100_added_exp AS excess_per_100,
                   p.per100_added_lo AS excess_per_100_lo, p.per100_added_hi AS excess_per_100_hi
            FROM afstd_pair p JOIN afstd_pair_exp e USING (drug, term)
            LEFT JOIN ref_drug_name n ON n.chembl_id = p.drug
            LEFT JOIN jno_label_section s ON s.drug = p.drug AND s.term = p.term
            WHERE p.on_fda ORDER BY 1, 3""").df()
    assert len(d) == 19249, len(d)
    d["label_subsection"] = d.pop("section_class").map(SECTION).fillna("Not classified")
    d = d[[c for c, _ in COLS]]
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(2)
    readme = pd.DataFrame({"Supplement 3": [
        "Placebo-Adjusted Frequencies of Nonserious Adverse Reactions Listed on US Prescription Drug Labels",
        f"Estimates for the {len(d):,} adverse reactions listed on current US Food and Drug Administration labels "
        "(primary analysis set), from the hierarchical model described in Supplement 1.",
        "Frequencies are conditional on the reaction being reported by the trial (see eTable 9 in Supplement 1).",
        ""] + [f"{c}: {t}" for c, t in COLS]})
    with pd.ExcelWriter(OUT) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        d.to_excel(xw, sheet_name="listed_reactions", index=False)
    print(len(d), "rows;", round(OUT.stat().st_size / 1024), "kB")
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        with pv.connect(read_only=True) as con:
            up = [x[0] for x in con.execute("SELECT out_id FROM prov_output_asset WHERE object_name IN "
                                            "('afstd_pair','afstd_pair_exp','jno_label_section','ref_drug_name')").fetchall()]
        run.archive_document("jama_supplement3_listed_reactions", OUT, kind="xlsx", upstream=up)
        old = OUT.parent / "supplement4_listed_reactions.xlsx"         # Supplement 4 before renumbering by citation order
        if old.exists():
            old.unlink()
            run.retire([str(old.relative_to(ROOT))], "renumbered as Supplement 3 (supplement3_listed_reactions.xlsx)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
