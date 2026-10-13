"""The single source of truth for figure lists, the rule ledger, descriptive method names, zones and colours.

Imported by 71_manuscript.py, 72_supplement.py, 77_review_copy.py, 111_fig_main.py and 112_fig_supp.py, so that the
figure list, the ledger and the wording of methods cannot drift between the text, the supplement and the figures.
"""
import re

# ------------------------------------------------------------------ descriptive names (no version names in the text)
METHOD_LABEL = {
    "AF 3.3": "Hierarchical estimator (this study)",
    "AF 3.2": "Without no-effect component",
    "AF 3.0": "Earlier hierarchical version",
    "B0 null": "No drug effect",
    "B1 MH": "Mantel–Haenszel pooling",
    "B2 PM-t": "Random-effects meta-analysis",
    "B3 Bayes-NN": "Bayesian random-effects meta-analysis",
    "B4 AF 2.0": "Empirical-Bayes shrinkage",
    "B5 term-only": "Side-effect average",
    "B6 class-mean": "Drug-class average",
    "B7 class-Bayes": "Class-informed Bayesian meta-analysis",
}
# the versions and benchmark names are allowed only in Code availability and the Supplementary Information
FORBIDDEN_RE = re.compile(r"\bAF[ -]?\d(\.\d)?\b|\bBENCH\b|BENCH-|AF-STD")

# ------------------------------------------------------------------ colours: all from the global figkit rules
# (~/.claude/skills/figkit/rules.toml, [palette] and [semantic.oajoint]); no hex literals here.
import figkit as fk  # noqa: E402

_SEM = fk.sem("oajoint")
VERM, BLUE, GREEN, PURP, SKY, BLACK = fk.c.vermilion, fk.c.blue, fk.c.green, fk.c.purple, fk.c.sky, fk.c.black
ORANGE = YELLOW = fk.c.amber      # fill colours only (contrast < 3:1 on white)
GREY, LGREY = fk.c.grey, fk.c.grey_light
METHOD_COLOR = _SEM["method"]
ZONES = ["confident_bg", "probable_bg", "probable_drug", "confident_drug"]
ZONE_LABEL = {"confident_bg": "confidently mostly background", "probable_bg": "probably mostly background",
              "probable_drug": "probably mostly drug-caused", "confident_drug": "confidently mostly drug-caused"}
ZONE_COLOR = _SEM["zone"]
STRATA = ["1", "2-4", "5-9", ">=10"]
STRATUM_LABEL = {"1": "1", "2-4": "2–4", "5-9": "5–9", ">=10": "≥10"}

# ------------------------------------------------------------------ figures
MAIN_FIGS = [
    ("fig1_question_model", "Figure 1"),
    ("fig2_prediction", "Figure 2"),
    ("fig3_background_share", "Figure 3"),
    ("fig4_where", "Figure 4"),
    ("fig5_label_patient", "Figure 5"),
]
SUPP_FIGS = [
    ("sfig1_data_labels", "Supplementary Fig. 1"),
    ("sfig2_development", "Supplementary Fig. 2"),
    ("sfig3_calibration", "Supplementary Fig. 3"),
    ("sfig4_conventional", "Supplementary Fig. 4"),
    ("sfig5_cards", "Supplementary Fig. 5"),
    ("sfig6_legacy", "Supplementary Fig. 6"),
]

# ------------------------------------------------------------------ rule ledger (Supplementary Table 1), in study order
GATES = {
    "Atlas and conventional estimator": [
        "af_design", "af_reproducibility", "af_reproducibility_temporal", "af_main",
        "af_label_validity", "af_h1_label", "af_robustness", "af_d1_subjective", "af_d2_by_soc", "af_fda_labels",
        "af_fda_validity", "af_fda_h1", "af_class", "af_probability_table", "af_v2", "af_route_coding", "af_route",
        "af_external_concordance", "af_indication", "af_deconvolution", "af_deconvolution_v2", "af_label_placebo_info",
        "af_dose_response", "af_excess_risk", "af_label_rules"],
    "Empirical-Bayes refinement": ["af2_design", "af2_sim", "af2_anchor", "af2_holdout", "af2_adopt"],
    "First hierarchical version": [
        "af3_design", "af3_sim", "af3_validity", "af3_prospective", "af3_random_holdout", "af3_calibration",
        "af3_calibration_v2", "af3_headline", "af3_adopt", "af3_adopt_v2"],
    "Prediction benchmark": ["bench2_protocol", "bench22_af32_vs_b7"],
    "Cell-specific heterogeneity": [
        "af32_design", "af32_sim", "af32_confirm", "af32_seen", "af32_calibration", "af32_adopt", "gen_newdrug",
        "gen_serious"],
    "No-effect component (adopted estimator)": [
        "af33_design", "af33_sim", "af33_confirm", "af33_seen", "af33_calibration", "af33_adopt", "gen33_newdrug",
        "gen33_serious"],
    "Label analyses": [
        "afstd_definitions", "afstd_label_match", "afstd_headline",
        "afstd_sensitivity", "afstd_robustness", "afstd_label_validity", "afstd_by_soc", "afstd_by_class",
        "afstd_absolute", "afstd_listing_rules", "afstd_examples", "afstd_definitions_v2", "afstd_absolute_v2",
        "afstd_listing_rules_v2", "afstd_point_v2", "afstd_no_effect_sensitivity", "afstd_future_prospective"],
    "Earlier analyses of the same registry (serious adverse events)": [
        "ct_unit_and_contrast", "ct_primary_outcome", "ct_meta_method", "ct_positive_controls", "ct_negative_control",
        "ct_positive_controls_v2", "atlas_design", "atlas_multiplicity", "atlas_benchmark_omop", "atlas_benchmark_euadr",
        "atlas_calibration", "atlas_h1_label_concordance", "atlas_h2_side_benefits"],
    "External benchmarks and secondary analyses": [
        "jno_ctt_reference", "jno_ext_ctt", "jno_ext_acei_cough", "jno_complete_reporting", "jno_label_section",
        "jno_discontinuation", "jno_trial_characteristics", "jno_ext_ctt_independent"],
    "Analyses added after internal review": [
        "jno_rev_label_aware", "jno_rev_refit_phase", "jno_rev_refit_shared", "jno_rev_bootstrap", "jno_rev_sim_nozero",
        "jno_rev_temporal_calibration", "jno_rev_classification", "jno_rev_positive_controls", "jno_rev_scope",
        "jno_rev_label_audit", "jno_rev_discontinuation_v2", "jno_rev_descriptives",
        "jno_rev_temporal_calibration_noise"],
    "Analyses added after second internal review": [
        "jno_r2_subsample", "jno_r2_sim_range", "jno_r2_absolute_complete", "jno_r2_route_single",
        "jno_r2_prior_vs_data"],
    "Labeled use and linkage recall": [
        "jno_r3_phase34", "jno_r3_approval", "jno_r3_indication", "jno_r3_highdose", "jno_r3_recall",
        "jno_r3_label_molecule"],
}
GATE_PREFIX_RE = re.compile(r"^(af|bench|gen|afstd|ms_|topic_selection_direction_c)")
# project-management records kept in the database but not analysis rules of the study (choice of research question,
# freezing of an earlier draft, choice of the primary estimator for the text): excluded from the published ledger
PROCESS_RECORDS = {"topic_selection_direction_c", "ms_v1_frozen", "ms_v2_primary_estimator"}
SUPERSEDED = {"af_deconvolution": "af_deconvolution_v2", "af3_calibration": "af3_calibration_v2",
              "af3_adopt": "af3_adopt_v2", "afstd_absolute": "afstd_absolute_v2",
              "afstd_listing_rules": "afstd_listing_rules_v2", "afstd_definitions": "afstd_definitions_v2 (point summaries)",
              "jno_discontinuation": "jno_rev_discontinuation_v2"}
# evaluations that touched the confirmation zone (trials started from February 2017), in order
CONFIRMATION_LOOKS = ["af3_prospective", "af3_calibration_v2", "af32_confirm", "bench22_af32_vs_b7", "af33_confirm"]


def all_gates():
    return [g for v in GATES.values() for g in v]


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def pfmt(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10^{int(e)}^"

# ------------------------------------------------------------------ display names
SOC_SHORT = {
    "blood and lymphatic system disorders": "Blood and lymphatic", "cardiac disorders": "Cardiac",
    "congenital, familial and genetic disorders": "Congenital", "ear and labyrinth disorders": "Ear and labyrinth",
    "endocrine disorders": "Endocrine", "eye disorders": "Eye", "gastrointestinal disorders": "Gastrointestinal",
    "general disorders": "General", "general disorders and administration site conditions": "General",
    "hepatobiliary disorders": "Hepatobiliary", "immune system disorders": "Immune system",
    "infections and infestations": "Infections", "injury, poisoning and procedural complications": "Injury and procedural",
    "investigations": "Investigations", "metabolism and nutrition disorders": "Metabolism and nutrition",
    "musculoskeletal and connective tissue disorders": "Musculoskeletal",
    "neoplasms benign, malignant and unspecified (incl cysts and polyps)": "Neoplasms",
    "nervous system disorders": "Nervous system", "pregnancy, puerperium and perinatal conditions": "Pregnancy",
    "product issues": "Product issues", "psychiatric disorders": "Psychiatric", "renal and urinary disorders": "Renal and urinary",
    "reproductive system and breast disorders": "Reproductive and breast",
    "respiratory, thoracic and mediastinal disorders": "Respiratory", "skin and subcutaneous tissue disorders": "Skin",
    "social circumstances": "Social circumstances", "surgical and medical procedures": "Procedures",
    "vascular disorders": "Vascular"}
ATC1_NAME = {"A": "Alimentary and metabolism", "B": "Blood", "C": "Cardiovascular", "D": "Dermatological",
             "G": "Genito-urinary and sex hormones", "H": "Systemic hormones", "J": "Anti-infectives",
             "L": "Antineoplastic and immunomodulating", "M": "Musculoskeletal", "N": "Nervous system",
             "P": "Antiparasitic", "R": "Respiratory", "S": "Sensory organs", "V": "Various"}
ATC1_COLOR = _SEM["atc1"]
COV_LABEL = {"open_label": "Open-label", "phase1": "Phase 1 only", "single_blind": "Single-blind",
             "nonsys": "Non-systematic assessment", "nonindustry": "Non-industry sponsor", "healthy": "Healthy volunteers",
             "year10": "Start year (per decade)", "lwk": "Follow-up (per e-fold)", "ldose_mean": "Mean dose (per e-fold)",
             "age10": "Age (per decade)", "female_c": "Female share (0 to 100%)", "lenroll": "Enrolment (per e-fold)"}
# short method names for figure axes (MA = meta-analysis, defined in every legend that uses them)
METHOD_SHORT = {"AF 3.3": "This estimator", "AF 3.2": "Without no-effect part", "B7 class-Bayes": "Class-informed Bayes MA",
                "B3 Bayes-NN": "Bayesian random-effects MA", "B4 AF 2.0": "Empirical-Bayes shrinkage",
                "B6 class-mean": "Drug-class average", "B2 PM-t": "Random-effects MA", "B1 MH": "Mantel–Haenszel",
                "B5 term-only": "Side-effect average", "B0 null": "No drug effect"}
# darker zone shades for TEXT (>= 3:1 contrast on white); fills use ZONE_COLOR
ZONE_TEXT = _SEM["zone_text"]


def abbrev_sentence(con, fig_id: str) -> str:
    """' Abbreviations: UTI, urinary tract infection; ...' for a figure legend, from fig_abbrev ('' if none)."""
    rows = con.execute("SELECT short, full_text FROM fig_abbrev WHERE fig_id = ? ORDER BY short", [fig_id]).fetchall()
    return (" Abbreviations: " + "; ".join(f"{a}, {f}" for a, f in rows) + ".") if rows else ""
