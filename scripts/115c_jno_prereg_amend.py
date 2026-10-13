#!/usr/bin/env python3
"""Amendment of jno_label_section BEFORE it is evaluated (2026-10-12). The registered rule recognised postmarketing text
only under a heading numbered 6.2; labels of biologics often number it 6.3 (after 6.2 Immunogenicity), which the rule
would have classed as 'other'. The amendment classifies subsection headings by title, not number. Nothing about label
sections has been computed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oajoint import provenance as pv  # noqa: E402

RULE = """Where on the label the listed side effects come from. Segment each current label text in the pool of
afstd_label_match at subsection headings '6.N <title>' (N = 1-9) and at section headings '6 ADVERSE REACTIONS': text
under a subsection whose title matches Clinical Trial(s) Experience or Clinical Studies Experience is a trial span; text
under a subsection whose title contains Post(-)marketing is a postmarketing span; all other text (section introduction
and cross-references to warnings, immunogenicity, other subsections) is other. Ingredients whose label text has no trial
subsection heading are unsegmented. Re-apply the afstd_label_match term rule within each kind of span. Classes of listed
pairs: 'trial' (matched in a trial span), 'postmarketing_only', 'other_only', 'unsegmented'. QC: for segmented
ingredients, matches in any span must equal on_fda. Report n per class and the expected share with AF < 0.5 (bootstrap
CI as in afstd_headline) per class. Secondary; the primary outcome stays the registered set of all listed pairs.
Amended before evaluation: subsection headings are classified by title rather than by number."""


def main() -> int:
    with pv.run_context(__file__, env_name="oajoint-py311") as run:
        run.register_gate("jno_label_section", " ".join(RULE.split()), gate_family="descriptive",
                          metric_name="n_trial_section", notes="amended before evaluation: headings by title")
        print("  amended jno_label_section")
    return 0


if __name__ == "__main__":
    sys.exit(main())
