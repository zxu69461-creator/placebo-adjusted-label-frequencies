"""PLOS Medicine manuscript (2026-10-17): terminology, figure lists and
display names that differ from the earlier version (oajoint.paper, frozen with the Nature Communications submission).

Everything not redefined here is taken from oajoint.paper unchanged, so that colours, method names, organ-system
names and the strata stay identical between the two versions.
"""
from oajoint.paper import *  # noqa: F401,F403
from oajoint import paper as _P

# ------------------------------------------------------------------ the four posterior zones (text, legends, figures)
# P(AF < 0.5) > 0.975 | 0.5-0.975 | 0.025-0.5 | < 0.025 ; AF < 0.5 is a relative risk below 2
ZONE_LABEL = {"confident_bg": "Confidently background-dominant", "probable_bg": "Likely background-dominant",
              "probable_drug": "Likely drug-dominant", "confident_drug": "Confidently drug-dominant"}
ZONE_RANGE = {"confident_bg": "P(AF < 0.5) > 0.975", "probable_bg": "0.5 < P(AF < 0.5) ≤ 0.975",
              "probable_drug": "0.025 ≤ P(AF < 0.5) ≤ 0.5", "confident_drug": "P(AF < 0.5) < 0.025"}
# two-line forms for narrow badges (Fig. 5d)
ZONE_LINES = {"confident_bg": ("Confidently", "background-dominant"), "probable_bg": ("Likely", "background-dominant"),
              "probable_drug": ("Likely", "drug-dominant"), "confident_drug": ("Confidently", "drug-dominant")}
# lower-case running-text forms
ZONE_TEXT_FORM = {k: v[0].lower() + v[1:] for k, v in ZONE_LABEL.items()}

# ------------------------------------------------------------------ figures of the PLOS Medicine manuscript
FIG_DIR = "figures/v3/main"
SUPP_DIR = "figures/v3/supplementary"
MAIN_FIGS = [
    ("fig1_model_data", "Figure 1"),
    ("fig2_validation", "Figure 2"),
    ("fig3_distribution", "Figure 3"),
    ("fig4_heterogeneity", "Figure 4"),
    ("fig5_translation", "Figure 5"),
]
SUPP_FIGS = [                       # numbered in order of first citation in the v3 main text
    ("sfig1_data_labels", "Supplementary Fig. 1"),
    ("sfig2_calibration", "Supplementary Fig. 2"),
    ("sfig3_serious", "Supplementary Fig. 3"),
    ("sfig4_conventional", "Supplementary Fig. 4"),
    ("sfig5_cards", "Supplementary Fig. 5"),
    ("sfig6_legacy", "Supplementary Fig. 6"),
    ("sfig7_development", "Supplementary Fig. 7"),
]

# ------------------------------------------------------------------ readable ATC level-1 names for axis labels
ATC1_AXIS = {"L": "Antineoplastic (L)", "N": "Nervous (N)", "A": "Alimentary (A)", "C": "Cardiovascular (C)",
             "D": "Dermatological (D)", "S": "Sensory (S)", "M": "Musculoskeletal (M)", "G": "Genitourinary (G)",
             "J": "Anti-infective (J)", "R": "Respiratory (R)", "H": "Hormones (H)", "B": "Blood (B)",
             "V": "Various (V)", "P": "Antiparasitic (P)"}

# method names on figure axes: unchanged, but 'MA' is spelled out where room allows
METHOD_SHORT = dict(_P.METHOD_SHORT, **{"B5 term-only": "Event average", "AF 3.2": "Without zero-excess part"})
METHOD_LABEL = dict(_P.METHOD_LABEL, **{"B5 term-only": "Event average", "AF 3.2": "Without zero-excess component"})

# words that must not appear in the v3 text (old zone names and internal process terms)
OLD_TERMS = ["mostly background", "probably the drug", "mostly drug-caused", "probably mostly", "confidently mostly",
             "mostly the drug", "P(background)", "confirmation set", "scored once", "time-stamped", "rule ledger",
             "study database", "provisional"]
