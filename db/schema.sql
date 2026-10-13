-- Database schema for the analysis code of this repository (DuckDB). Idempotent: safe to re-run.
-- Every result table carries run_id -> prov_script_run; every output is registered in
-- prov_output_asset and prov_lineage; every analysis rule is a prov_decision row written before it is
-- evaluated. Large matrices live in db/parquet/ and are exposed through views.

CREATE TABLE IF NOT EXISTS prov_script_run (
    run_id            VARCHAR PRIMARY KEY,
    script_path       VARCHAR NOT NULL,
    script_sha256     VARCHAR NOT NULL,
    git_commit        VARCHAR,
    cmdline           VARCHAR,
    env_name          VARCHAR,
    env_lock_sha256   VARCHAR,
    software_versions JSON,
    host              VARCHAR,
    gpu_model         VARCHAR,
    started_at        TIMESTAMP NOT NULL,
    finished_at       TIMESTAMP,
    exit_status       INTEGER,
    cpu_seconds       DOUBLE,
    peak_rss_gb       DOUBLE,
    notes             VARCHAR
);

CREATE TABLE IF NOT EXISTS prov_input_asset (
    asset_id          VARCHAR PRIMARY KEY,
    asset_kind        VARCHAR NOT NULL,

    source_registry   VARCHAR,

    accession         VARCHAR,
    url               VARCHAR,
    local_path        VARCHAR,
    file_bytes        BIGINT,
    file_sha256       VARCHAR,
    published_md5     VARCHAR,
    md5_match         BOOLEAN,
    downloaded_at     TIMESTAMP,
    genome_build      VARCHAR,
    build_evidence    VARCHAR,
    coordinate_system VARCHAR,
    licence           VARCHAR,
    licence_url       VARCHAR,
    status            VARCHAR DEFAULT 'present',
    registered_by     VARCHAR REFERENCES prov_script_run(run_id),
    notes             VARCHAR
);

CREATE TABLE IF NOT EXISTS prov_output_asset (
    out_id      VARCHAR PRIMARY KEY,
    run_id      VARCHAR NOT NULL REFERENCES prov_script_run(run_id),
    object_name VARCHAR NOT NULL,
    object_kind VARCHAR,
    storage     VARCHAR,
    relpath     VARCHAR,
    n_rows      BIGINT,
    n_cols      INTEGER,
    file_sha256 VARCHAR,
    created_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS prov_lineage (
    out_id    VARCHAR NOT NULL REFERENCES prov_output_asset(out_id),
    asset_id  VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    role      VARCHAR
);

CREATE TABLE IF NOT EXISTS prov_decision (
    decision_id   VARCHAR PRIMARY KEY,
    gate_name     VARCHAR NOT NULL,
    gate_family   VARCHAR,
    rule_text     VARCHAR NOT NULL,
    metric_name   VARCHAR,
    threshold     DOUBLE,
    comparator    VARCHAR,
    registered_at TIMESTAMP NOT NULL,
    registered_by VARCHAR REFERENCES prov_script_run(run_id),
    metric_value  DOUBLE,
    metric_detail JSON,
    passed        BOOLEAN,
    evaluated_at  TIMESTAMP,
    evaluated_by  VARCHAR REFERENCES prov_script_run(run_id),
    notes         VARCHAR
);

CREATE TABLE IF NOT EXISTS ref_phenotype (
    pheno_id            VARCHAR PRIMARY KEY,
    label               VARCHAR NOT NULL,
    joint_site          VARCHAR,
    pheno_class         VARCHAR,

    definition_text     VARCHAR,
    accession           VARCHAR,




    asset_id            VARCHAR,
    ancestry_setting    VARCHAR,
    n_case              BIGINT,
    n_ctrl              BIGINT,
    n_total             BIGINT,
    n_eff               DOUBLE,
    sample_prev         DOUBLE,
    pop_prev            DOUBLE,
    genome_build        VARCHAR,
    source_pmid         VARCHAR,
    n_reported_assoc    INTEGER,
    eur_frac            DOUBLE,
    n_ancestry_groups   INTEGER,
    eur_frac_reliable   BOOLEAN,
    resolution_evidence VARCHAR,
    is_nested           BOOLEAN,
    in_primary_block    BOOLEAN,
    eligible_indicator  BOOLEAN,
    notes               VARCHAR
);

CREATE TABLE IF NOT EXISTS ref_parquet_domain (
    domain        VARCHAR PRIMARY KEY,
    relpath       VARCHAR NOT NULL,
    view_name     VARCHAR NOT NULL,
    partition_by  VARCHAR[],
    description   VARCHAR
);

CREATE TABLE IF NOT EXISTS af_label_match (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, on_fda BOOLEAN, on_sider BOOLEAN,
    has_fda_label BOOLEAN, fda_generic VARCHAR, fda_set_id VARCHAR
);

CREATE TABLE IF NOT EXISTS af_class_summary (
    run_id VARCHAR, atc3 VARCHAR, class_label VARCHAR, n_label_cells INTEGER,
    frac_afhi_lt_half DOUBLE, frac_af_le_zero DOUBLE, median_af DOUBLE
);

CREATE TABLE IF NOT EXISTS af_probability (
    run_id VARCHAR, chembl_id VARCHAR, drug_name VARCHAR, n_trials_drug INTEGER,
    term VARCHAR, n_trials INTEGER, risk_drug DOUBLE, risk_pl DOUBLE,
    af DOUBLE, af_lo DOUBLE, af_hi DOUBLE
);

CREATE TABLE IF NOT EXISTS fda_label_pool (
    run_id VARCHAR, generic_name VARCHAR, pool VARCHAR, n_labels INTEGER,
    n_distinct_texts INTEGER, routes VARCHAR, application_numbers VARCHAR[],
    adverse_reactions_text VARCHAR
);

CREATE TABLE IF NOT EXISTS fda_label (
    run_id VARCHAR, generic_name VARCHAR, set_id VARCHAR, application_number VARCHAR,
    effective_time VARCHAR, route VARCHAR, n_candidate_labels INTEGER,
    adverse_reactions_text VARCHAR
);

CREATE TABLE IF NOT EXISTS af_cell (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, n_trials INTEGER,
    events_drug DOUBLE, n_drug DOUBLE, events_pl DOUBLE, n_pl DOUBLE,
    risk_drug DOUBLE, risk_pl DOUBLE, rr DOUBLE, rr_lo DOUBLE, rr_hi DOUBLE,
    af DOUBLE, af_lo DOUBLE, af_hi DOUBLE, p DOUBLE, fdr DOUBLE,
    sider_mapped BOOLEAN, on_label BOOLEAN
);

CREATE TABLE IF NOT EXISTS af_split_estimate (
    run_id VARCHAR, split VARCHAR, half VARCHAR, drug VARCHAR, term VARCHAR,
    n_trials INTEGER, events_drug DOUBLE, n_drug DOUBLE, events_pl DOUBLE, n_pl DOUBLE,
    log_rr DOUBLE, se DOUBLE, p DOUBLE
);

CREATE TABLE IF NOT EXISTS atlas_contrast (
    contrast_id VARCHAR, run_id VARCHAR, nct VARCHAR, eg_active VARCHAR,
    eg_placebo VARCHAR[], k INTEGER, drug_ids VARCHAR[], atc4 VARCHAR[],
    single_ingredient BOOLEAN,
    n_act_serious DOUBLE, n_act_other DOUBLE, n_pl_serious DOUBLE, n_pl_other DOUBLE
);

CREATE TABLE IF NOT EXISTS ctgov_trial (
    nct VARCHAR PRIMARY KEY, run_id VARCHAR, title VARCHAR, phase VARCHAR[],
    masking VARCHAR, primary_purpose VARCHAR, enrollment INTEGER, start_date VARCHAR,
    conditions VARCHAR[], sponsor VARCHAR, ae_time_frame VARCHAR,
    frequency_threshold DOUBLE, n_event_groups INTEGER, meshes VARCHAR[]
);

CREATE TABLE IF NOT EXISTS ctgov_event_group (
    nct VARCHAR, eg_id VARCHAR, run_id VARCHAR, eg_title VARCHAR,
    arm_label VARCHAR, arm_type VARCHAR, link_method VARCHAR, link_score DOUBLE,
    role VARCHAR,
    ingredients VARCHAR[], chembl_ids VARCHAR[], atc4 VARCHAR[], map_methods VARCHAR[],
    n_risk_serious INTEGER, n_risk_other INTEGER,
    oa_serious_max INTEGER, oa_serious_sum INTEGER, oa_serious_osteoarthritis INTEGER,
    arthralgia_serious INTEGER, cataract_serious INTEGER, arthroplasty_serious INTEGER,
    oa_other_max INTEGER, arthralgia_other INTEGER
);

CREATE TABLE IF NOT EXISTS ref_chembl_drug (
    chembl_id VARCHAR, pref_name VARCHAR, synonyms VARCHAR[],
    atc_codes VARCHAR[], atc4 VARCHAR[], atc4_desc VARCHAR[],
    max_phase DOUBLE, first_approval INTEGER, drug_type VARCHAR, biotherapeutic BOOLEAN
);

CREATE TABLE IF NOT EXISTS ref_chembl_mechanism (
    chembl_id VARCHAR, parent_chembl_id VARCHAR, target_chembl_id VARCHAR,
    action_type VARCHAR, mechanism_of_action VARCHAR, direct_interaction BOOLEAN
);

CREATE TABLE IF NOT EXISTS fig_panel (
    fig_id VARCHAR, panel_id VARCHAR, panel_title VARCHAR,
    source_objects VARCHAR[], script_path VARCHAR,
    needs_handmade_schematic BOOLEAN, imagegen_prompt_text VARCHAR,
    rendered_relpath VARCHAR, rendered_sha256 VARCHAR, rendered_at TIMESTAMP,





    PRIMARY KEY (fig_id, panel_id, rendered_relpath)
);

CREATE TABLE IF NOT EXISTS fig_qa (
    fig_id VARCHAR, check_name VARCHAR, status VARCHAR, detail VARCHAR,
    pdf_relpath VARCHAR, pdf_sha256 VARCHAR, rules_version VARCHAR, checked_at TIMESTAMP,
    PRIMARY KEY (fig_id, check_name)
);

CREATE TABLE IF NOT EXISTS fig_abbrev (
    fig_id VARCHAR, short VARCHAR, full_text VARCHAR, rendered_at TIMESTAMP,
    PRIMARY KEY (fig_id, short)
);

CREATE TABLE IF NOT EXISTS api_cache (
    query_sha256 VARCHAR PRIMARY KEY, api_name VARCHAR, api_version VARCHAR,
    query_text VARCHAR, response JSON, fetched_at TIMESTAMP
);

CREATE OR REPLACE VIEW v_manifest AS
SELECT o.object_name,
       o.object_kind,
       o.storage,
       o.relpath,
       o.n_rows,
       o.file_sha256,
       r.script_path,
       r.script_sha256,
       r.git_commit,
       r.env_name,
       r.software_versions,
       o.created_at,
       count(DISTINCT l.asset_id)    FILTER (l.edge_kind = 'input_asset')    AS n_input_edges,
       count(DISTINCT l.asset_id)    FILTER (l.edge_kind = 'upstream_output') AS n_upstream_edges,
       list(DISTINCT i.accession)    FILTER (i.accession    IS NOT NULL) AS input_accessions,
       list(DISTINCT i.file_sha256)  FILTER (i.file_sha256  IS NOT NULL) AS input_sha256,
       list(DISTINCT i.genome_build) FILTER (i.genome_build IS NOT NULL
                                        AND i.genome_build <> 'not_applicable') AS input_builds
FROM prov_output_asset o
JOIN prov_script_run  r USING (run_id)
LEFT JOIN prov_lineage l USING (out_id)
LEFT JOIN prov_input_asset i
       ON i.asset_id = l.asset_id AND l.edge_kind = 'input_asset'
GROUP BY o.object_name, o.object_kind, o.storage, o.relpath, o.n_rows, o.file_sha256,
         r.script_path, r.script_sha256, r.git_commit, r.env_name, r.software_versions,
         o.created_at;

CREATE OR REPLACE VIEW v_manifest_latest AS
SELECT * EXCLUDE (rn) FROM (
  SELECT *, row_number() OVER (PARTITION BY object_name ORDER BY created_at DESC) AS rn
  FROM v_manifest
) WHERE rn = 1;

CREATE OR REPLACE MACRO manifest() AS TABLE
  SELECT * FROM v_manifest_latest ORDER BY created_at;

CREATE OR REPLACE MACRO manifest_history() AS TABLE
  SELECT * FROM v_manifest ORDER BY created_at;

CREATE OR REPLACE MACRO manifest_gaps() AS TABLE
  SELECT m.object_name, m.object_kind, m.script_path, m.created_at
  FROM v_manifest_latest m
  WHERE m.n_input_edges = 0 AND m.n_upstream_edges = 0
    AND m.object_name NOT IN (
      SELECT fig_id || ':' || panel_id FROM fig_panel WHERE needs_handmade_schematic)
  ORDER BY m.created_at;

CREATE OR REPLACE MACRO manifest_schematics() AS TABLE
  SELECT fig_id, panel_id, panel_title, rendered_relpath, imagegen_prompt_text
  FROM fig_panel WHERE needs_handmade_schematic ORDER BY fig_id, panel_id;

CREATE OR REPLACE MACRO manifest_builds() AS TABLE
  SELECT object_name, input_builds, script_path
  FROM v_manifest_latest
  WHERE input_builds IS NOT NULL AND len(input_builds) > 1;

CREATE OR REPLACE MACRO gate_status() AS TABLE
  SELECT gate_name, gate_family, rule_text, metric_name, comparator, threshold,
         metric_value, passed, registered_at, evaluated_at,
         (evaluated_at IS NOT NULL AND registered_at < evaluated_at) AS prereg_intact
  FROM prov_decision
  ORDER BY gate_family, gate_name;

CREATE OR REPLACE MACRO prereg_violations() AS TABLE
  SELECT gate_name, registered_at, evaluated_at
  FROM prov_decision
  WHERE evaluated_at IS NOT NULL AND registered_at >= evaluated_at;

CREATE TABLE IF NOT EXISTS fda_label_route_pool (
    run_id VARCHAR, generic_name VARCHAR, route_family VARCHAR, pool VARCHAR, n_labels INTEGER,
    n_distinct_texts INTEGER, adverse_reactions_text VARCHAR
);

CREATE TABLE IF NOT EXISTS atlas_contrast_route (
    run_id VARCHAR, contrast_id VARCHAR, nct VARCHAR, drug VARCHAR, route_family VARCHAR,
    tier VARCHAR, evidence VARCHAR
);

CREATE TABLE IF NOT EXISTS af_route_cell (
    run_id VARCHAR, drug VARCHAR, route_family VARCHAR, term VARCHAR, n_trials INTEGER,
    events_drug DOUBLE, n_drug DOUBLE, events_pl DOUBLE, n_pl DOUBLE, risk_drug DOUBLE, risk_pl DOUBLE,
    rr DOUBLE, rr_lo DOUBLE, rr_hi DOUBLE, af DOUBLE, af_lo DOUBLE, af_hi DOUBLE, p DOUBLE, fdr DOUBLE,
    has_route_label BOOLEAN, on_route_label BOOLEAN
);

CREATE TABLE IF NOT EXISTS af_route_probability (
    run_id VARCHAR, chembl_id VARCHAR, drug_name VARCHAR, route_family VARCHAR, n_trials_stratum INTEGER,
    term VARCHAR, n_trials INTEGER, risk_drug DOUBLE, risk_pl DOUBLE, af DOUBLE, af_lo DOUBLE, af_hi DOUBLE
);

CREATE TABLE IF NOT EXISTS ref_drug_name (
    run_id VARCHAR, chembl_id VARCHAR, display_name VARCHAR, source VARCHAR
);

CREATE TABLE IF NOT EXISTS af_external_ref (
    run_id VARCHAR, drug VARCHAR, route_family VARCHAR, term VARCHAR, found BOOLEAN, pmid VARCHAR,
    pub_year INTEGER, measure VARCHAR, estimate DOUBLE, context VARCHAR, query VARCHAR,
    atlas_rr DOUBLE, atlas_rr_lo DOUBLE, atlas_rr_hi DOUBLE, atlas_n_trials INTEGER
);

CREATE TABLE IF NOT EXISTS ctgov_condition (
    run_id VARCHAR, nct VARCHAR, indication VARCHAR, meshes VARCHAR[], ancestors VARCHAR[]
);

CREATE TABLE IF NOT EXISTS af_indication_cell (
    run_id VARCHAR, drug VARCHAR, indication VARCHAR, term VARCHAR, n_trials INTEGER, events DOUBLE,
    rr DOUBLE, rr_lo DOUBLE, rr_hi DOUBLE, af DOUBLE, af_lo DOUBLE, af_hi DOUBLE
);

CREATE TABLE IF NOT EXISTS af_eb_cell (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, af_raw DOUBLE, af_post_mean DOUBLE,
    p_no_excess DOUBLE, p_af_lt_half DOUBLE
);

CREATE TABLE IF NOT EXISTS af_eb_prior (run_id VARCHAR, theta DOUBLE, weight DOUBLE);

CREATE TABLE IF NOT EXISTS af_dose_cell (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, n_trials INTEGER, events DOUBLE, rr_hl DOUBLE, rr_hl_lo DOUBLE,
    rr_hl_hi DOUBLE, p DOUBLE
);

CREATE TABLE IF NOT EXISTS af2_cell (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, k INTEGER, events DOUBLE, n_trunc INTEGER,
    theta_mh DOUBLE, se_mh DOUBLE, theta_s1 DOUBLE, se_s1 DOUBLE, mu DOUBLE, tau2 DOUBLE, shrink DOUBLE,
    theta2 DOUBLE, se2 DOUBLE, af2 DOUBLE, af2_lo DOUBLE, af2_hi DOUBLE
);

CREATE TABLE IF NOT EXISTS prov_retired (relpath VARCHAR, retired_at TIMESTAMP, run_id VARCHAR, reason VARCHAR);

CREATE TABLE IF NOT EXISTS ctgov_baseline (
    nct VARCHAR PRIMARY KEY, run_id VARCHAR,
    age_value DOUBLE, age_param VARCHAR, age_unit VARCHAR,
    pct_female DOUBLE, n_baseline DOUBLE
);

CREATE TABLE IF NOT EXISTS ctgov_flow_group (
    nct VARCHAR, fg_id VARCHAR, run_id VARCHAR, fg_title VARCHAR,
    started DOUBLE, completed DOUBLE, drop_total DOUBLE, drop_ae DOUBLE,
    PRIMARY KEY (nct, fg_id)
);

CREATE TABLE IF NOT EXISTS af3_sim (
    run_id VARCHAR, scenario VARCHAR, n_cells_af1 INTEGER, coverage_af3 DOUBLE, coverage_af1 DOUBLE,
    rmse_af3 DOUBLE, rmse_af1 DOUBLE, tau_true DOUBLE, tau_est_median DOUBLE, passed BOOLEAN, version VARCHAR
);

CREATE TABLE IF NOT EXISTS af3_cell (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, soc VARCHAR, n_trials INTEGER,
    psi_mean DOUBLE, psi_sd DOUBLE, psi_lo DOUBLE, psi_med DOUBLE, psi_hi DOUBLE,
    af_hz DOUBLE, af_hz_lo DOUBLE, af_hz_hi DOUBLE, af_risk12 DOUBLE, p0_12wk DOUBLE,
    p_af_lt_half DOUBLE, p_no_excess DOUBLE, pred_lo DOUBLE, pred_hi DOUBLE, tau_soc DOUBLE,
    prior_mean DOUBLE, lik_mean DOUBLE, lik_sd DOUBLE, af1_estimable BOOLEAN
);

CREATE TABLE IF NOT EXISTS af3_param (
    run_id VARCHAR, fit VARCHAR, kind VARCHAR, soc VARCHAR, name VARCHAR, value DOUBLE
);

CREATE TABLE IF NOT EXISTS af3_score (
    run_id VARCHAR, split VARCHAR, method VARCHAR, n_records INTEGER, n_trials INTEGER,
    mean_logscore DOUBLE, diff_vs_af3 DOUBLE, diff_ci_lo DOUBLE, diff_ci_hi DOUBLE, pit_central90 DOUBLE
);

CREATE TABLE IF NOT EXISTS bench_score (
    run_id VARCHAR, bench VARCHAR, win VARCHAR, stratum VARCHAR, method VARCHAR, reference VARCHAR,
    n_records INTEGER, n_trials INTEGER, mean_logscore DOUBLE, diff_vs_ref DOUBLE, diff_ci_lo DOUBLE, diff_ci_hi DOUBLE,
    pit_central90 DOUBLE, pit_low10 DOUBLE, pit_high10 DOUBLE, scoring VARCHAR
);

CREATE TABLE IF NOT EXISTS bench_record (
    run_id VARCHAR, bench VARCHAR, win VARCHAR, nct VARCHAR, drug VARCHAR, term VARCHAR, soc VARCHAR,
    k_train INTEGER, af1_support BOOLEAN, method VARCHAR, logscore DOUBLE, pit DOUBLE, scoring VARCHAR
);

CREATE TABLE IF NOT EXISTS af32_cell (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, soc VARCHAR, n_trials INTEGER,
    psi_mean DOUBLE, psi_sd DOUBLE, psi_lo DOUBLE, psi_med DOUBLE, psi_hi DOUBLE,
    af_hz DOUBLE, af_hz_lo DOUBLE, af_hz_hi DOUBLE, af_risk12 DOUBLE, p0_12wk DOUBLE,
    p_af_lt_half DOUBLE, p_no_excess DOUBLE, pred_lo DOUBLE, pred_hi DOUBLE, tau_cell DOUBLE,
    prior_mean DOUBLE, lik_mean DOUBLE, lik_sd DOUBLE, af1_estimable BOOLEAN
);

CREATE TABLE IF NOT EXISTS afstd_cell (
    run_id VARCHAR, estimator VARCHAR, drug VARCHAR, term VARCHAR, soc VARCHAR, n_trials INTEGER,
    psi_mean DOUBLE, psi_sd DOUBLE, psi_lo DOUBLE, psi_med DOUBLE, psi_hi DOUBLE,
    af_hz DOUBLE, af_hz_lo DOUBLE, af_hz_hi DOUBLE, af_risk12 DOUBLE, p0_12wk DOUBLE,
    p_af_lt_half DOUBLE, p_no_excess DOUBLE, p_no_effect DOUBLE, pred_lo DOUBLE, pred_hi DOUBLE, tau_cell DOUBLE,
    prior_mean DOUBLE, lik_mean DOUBLE, lik_sd DOUBLE, af1_estimable BOOLEAN
);

CREATE TABLE IF NOT EXISTS afstd_label_match (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, has_fda_label BOOLEAN, label_source VARCHAR,
    on_fda BOOLEAN, sider_mapped BOOLEAN, on_sider BOOLEAN, fda_generic VARCHAR, term_promiscuity DOUBLE
);

CREATE TABLE IF NOT EXISTS afstd_pair (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, soc_h VARCHAR, atc3 VARCHAR, atc1 VARCHAR,
    n_trials INTEGER, evidence VARCHAR, zone VARCHAR,
    has_fda_label BOOLEAN, on_fda BOOLEAN, sider_mapped BOOLEAN, on_sider BOOLEAN, label_source VARCHAR,
    promiscuous BOOLEAN, af1_estimable BOOLEAN,
    psi_mean DOUBLE, psi_med DOUBLE, af_hz DOUBLE, af_med DOUBLE, af_hz_lo DOUBLE, af_hz_hi DOUBLE, af_risk12 DOUBLE,
    p_af_lt_half DOUBLE, p_no_excess DOUBLE, p_no_effect DOUBLE,
    x0_own DOUBLE, n0_own DOUBLE, p0_own DOUBLE, p0_12wk DOUBLE,
    per100_pl DOUBLE, per100_drug_med DOUBLE, per100_drug_lo DOUBLE, per100_drug_hi DOUBLE,
    per100_added_med DOUBLE, per100_added_lo DOUBLE, per100_added_hi DOUBLE,
    per100_added_pred_lo DOUBLE, per100_added_pred_hi DOUBLE, per100_added_ref_med DOUBLE,
    rule_a BOOLEAN, rule_b BOOLEAN, rule_c BOOLEAN
);

CREATE TABLE IF NOT EXISTS afstd_summary (
    run_id VARCHAR, analysis VARCHAR, set_name VARCHAR, grp VARCHAR, n_pairs INTEGER, n_drugs INTEGER,
    metric VARCHAR, value DOUBLE, lo DOUBLE, hi DOUBLE
);

CREATE TABLE IF NOT EXISTS afstd_card (
    run_id VARCHAR, card_set VARCHAR, rank_drug INTEGER, rank_term INTEGER, drug VARCHAR, display_name VARCHAR,
    term VARCHAR, n_trials INTEGER, per100_pl DOUBLE, per100_drug_med DOUBLE,
    per100_added_med DOUBLE, per100_added_lo DOUBLE, per100_added_hi DOUBLE,
    per100_added_pred_lo DOUBLE, per100_added_pred_hi DOUBLE,
    af_med DOUBLE, af_lo DOUBLE, af_hi DOUBLE, p_af_lt_half DOUBLE, p_no_effect DOUBLE, zone VARCHAR
);

INSERT INTO ref_parquet_domain (domain, relpath, view_name, partition_by, description) VALUES
  ('afstd_posterior', 'db/parquet/afstd_posterior', 'v_afstd_posterior', [], 'posterior mass of every drug-side-effect pair on the log hazard-ratio grid (-5..5, step 0.05) of the adopted hierarchical estimator (AF 3.3)')
ON CONFLICT (domain) DO NOTHING;

CREATE TABLE IF NOT EXISTS afstd_pair_exp (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, af_mean DOUBLE,
    x_drug_own DOUBLE, n_drug_own DOUBLE, per100_drug_obs DOUBLE, per100_pl_obs DOUBLE,
    per100_drug_exp DOUBLE, per100_added_exp DOUBLE, rule_a_v2 BOOLEAN, rule_b_v2 BOOLEAN, rule_c_v2 BOOLEAN
);

CREATE TABLE IF NOT EXISTS ext_ctt_outcome (
    run_id VARCHAR, outcome VARCHAR, soc VARCHAR, pts VARCHAR[], n_statin INTEGER, n_placebo INTEGER,
    oe DOUBLE, var_oe DOUBLE, rr DOUBLE, rr_lo DOUBLE, rr_hi DOUBLE, fdr_sig BOOLEAN
);

CREATE TABLE IF NOT EXISTS jno_ext_concordance (
    run_id VARCHAR, benchmark VARCHAR, outcome VARCHAR, n_pairs INTEGER, n_drugs INTEGER, n_trials INTEGER,
    our_loghr DOUBLE, our_se DOUBLE, our_ef DOUBLE, our_ef_lo DOUBLE, our_ef_hi DOUBLE,
    ref_ratio DOUBLE, ref_lo DOUBLE, ref_hi DOUBLE, ref_ef DOUBLE, ref_sig BOOLEAN
);

CREATE TABLE IF NOT EXISTS jno_cell_complete (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, n_trials INTEGER, psi_mean DOUBLE, psi_sd DOUBLE, af_hz DOUBLE,
    p_af_lt_half DOUBLE, p_no_effect DOUBLE
);

CREATE TABLE IF NOT EXISTS jno_label_section (
    run_id VARCHAR, drug VARCHAR, term VARCHAR, segmented BOOLEAN, on_trial BOOLEAN, on_postmarketing BOOLEAN,
    on_other BOOLEAN, section_class VARCHAR
);

CREATE TABLE IF NOT EXISTS jno_discontinuation_contrast (
    run_id VARCHAR, contrast_id VARCHAR, nct VARCHAR, drug VARCHAR, atc1 VARCHAR, n_contrasts_in_trial INTEGER,
    x_drug DOUBLE, n_drug DOUBLE, x_pl DOUBLE, n_pl DOUBLE
);

CREATE TABLE IF NOT EXISTS jno_summary (
    run_id VARCHAR, analysis VARCHAR, set_name VARCHAR, grp VARCHAR, n_pairs INTEGER, n_drugs INTEGER,
    metric VARCHAR, value DOUBLE, lo DOUBLE, hi DOUBLE
);

CREATE TABLE IF NOT EXISTS jno_cell_variant (
    run_id VARCHAR, variant VARCHAR, drug VARCHAR, term VARCHAR, n_trials INTEGER, psi_mean DOUBLE, psi_sd DOUBLE,
    af_hz DOUBLE, p_af_lt_half DOUBLE, p_no_effect DOUBLE, p_no_excess DOUBLE
);

CREATE TABLE IF NOT EXISTS ref_chembl_indication (run_id VARCHAR, molecule_chembl_id VARCHAR, parent_molecule_chembl_id VARCHAR, mesh_id VARCHAR, mesh_heading VARCHAR, efo_id VARCHAR, efo_term VARCHAR, max_phase_for_ind DOUBLE);
