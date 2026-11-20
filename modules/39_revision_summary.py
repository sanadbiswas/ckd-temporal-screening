# Modular pipeline stage — REVISION ROUND 1.
#
# Collects the revision outputs into:
#   REVIEWER_COMMENT_OUTPUT_MAP.csv — every reviewer comment -> module and
#       output files (or "manuscript text" for presentation-only comments)
#   REVISION_KEY_NUMBERS.json       — headline numbers for the response letter
#   reproducibility_environment_revision.json — revision settings

# =========================================================
# 1. Reviewer comment -> analysis -> output map
# =========================================================
comment_map = [
    # Reviewer 1
    ("Reviewer 1", "Main (novelty)", "29, 30, 28",
     "R1c1_R4c1_*; R1c3_*; R3c1_R4c5_selection_under_calibration.csv",
     "Comparators/rules under identical procedure; two refinements tested against the original (weighted threshold; calibration-aware selection)"),
    ("Reviewer 1", "1", "29",
     "R1c1_R4c1_comparator_and_ablation_models.csv; R1c1_R4c1_paired_differences_temporal.csv; R4c1_testing_burden_per_1000.csv",
     "Simpler models, risk-factor rules, 12-predictor ablation"),
    ("Reviewer 1", "2", "28 (Part E)",
     "R1c2_fully_nested_procedure_folds.csv; R1c2_fully_nested_vs_reported.csv",
     "Tuning -> calibration -> threshold sequence; fully nested estimate"),
    ("Reviewer 1", "3", "30",
     "R1c3_weighted_vs_unweighted_threshold.csv; R1c3_case_mix_weighted_vs_unweighted.csv",
     "Survey-weighted 80% threshold, frozen and transported"),
    # Reviewer 2
    ("Reviewer 2", "1", "—", "manuscript text (Abstract)",
     "State single-visit CKD-compatible phenotype in Abstract"),
    ("Reviewer 2", "2", "27", "R2c2_included_vs_excluded.csv",
     "Included vs excluded (missing creatinine/ACR)"),
    ("Reviewer 2", "3", "35",
     "R2c3_alcohol_sensitivity_models.csv; R2c3_alcohol_missingness_profile.csv",
     "Alcohol removed / indicator / skip-pattern recode"),
    ("Reviewer 2", "4", "28 (Part A)",
     "R2c4_one_se_rule_details.csv; R2c4_best_model_fold_aucs.csv",
     "One-SE rule rationale and SE computation"),
    ("Reviewer 2", "5", "28 (Part C)", "R2c5_calibration_method_paired_differences.csv",
     "Sigmoid vs isotonic differences with CIs"),
    ("Reviewer 2", "6", "27", "R2c6_prevalence_n_over_N.csv",
     "Prevalence n/N and %, both cohorts"),
    ("Reviewer 2", "7", "30", "R1c3_*; R3c3_threshold_tradeoff_temporal.csv",
     "Population-level vs sample-level screening"),
    ("Reviewer 2", "8", "33", "R2c8_R3m5_subgroup_sens_spec_CI.csv",
     "Subgroup sensitivity/specificity CIs"),
    ("Reviewer 2", "9", "33", "R4c2_policy_comparison_net_benefit.csv + manuscript text",
     "Stratified thresholds framed as hypothesis-generating"),
    ("Reviewer 2", "10", "36", "R2c10_dca_threshold_interpretation.csv",
     "DCA range and frozen-threshold interpretation"),
    ("Reviewer 2", "11", "34",
     "R3m2_R4c3_component_and_age_performance.csv; R2c11_R4c3_within_age_discrimination.csv",
     "Age-only model; increment over age; within-age discrimination"),
    ("Reviewer 2", "12", "—", "manuscript text", "Condense Introduction/Discussion"),
    # Reviewer 3
    ("Reviewer 3", "1", "28 (Part D)",
     "R3c1_R4c5_calibrated_comparison_summary.csv; R3c1_R4c5_selection_under_calibration.csv",
     "All five families compared after calibration"),
    ("Reviewer 3", "2", "31", "R3c2_design_based_uncertainty.csv; R3c2_design_description.csv",
     "Rao-Wu survey-design bootstrap"),
    ("Reviewer 3", "3", "30", "R1c3_*; R3c3_threshold_tradeoff_temporal.csv",
     "Weighted threshold and trade-off"),
    ("Reviewer 3", "4", "32",
     "R3c4_calibration_with_CI.csv; R3c4_ece_binning_sensitivity.csv; R3c4_calibration_around_threshold.csv; R3c4_calibration_decision_impact.csv",
     "Calibration CIs, ECE binning, threshold-local calibration"),
    ("Reviewer 3", "5", "36", "R3c5_screening_scenarios_per_10000.csv",
     "Tests, missed cases, unnecessary referrals by threshold"),
    ("Reviewer 3", "minor 1", "28 (Part B)", "R3m1_table2_ci_method.json",
     "Table 2 CI construction"),
    ("Reviewer 3", "minor 2", "27, 34", "R2c6_prevalence_n_over_N.csv; R3m2_R4c3_component_and_age_performance.csv",
     "Composite vs components"),
    ("Reviewer 3", "minor 3", "37",
     "R3m3_predictor_spearman_correlation.csv; R3m3_grouped_permutation_importance.csv",
     "Correlation-aware importance"),
    ("Reviewer 3", "minor 4", "35", "R2c3_*", "Missing-data treatment of alcohol"),
    ("Reviewer 3", "minor 5", "33", "R2c8_R3m5_subgroup_sens_spec_CI.csv",
     "Subgroup CIs"),
    # Reviewer 4
    ("Reviewer 4", "1", "29, 36",
     "R1c1_R4c1_*; R4c1_testing_burden_per_1000.csv; R4c1_dca_models_vs_rules_temporal.csv",
     "Clinical rule + calibrated LR; DCA; testing burden; additional cases"),
    ("Reviewer 4", "2", "33",
     "R4c2_policy_comparison_net_benefit.csv; R4c2_policy_net_benefit_differences.csv; R4c2_dca_within_diabetes.csv",
     "Global vs stratified vs test-all-high-risk policies"),
    ("Reviewer 4", "3", "34", "R3m2_R4c3_*; R2c11_R4c3_*",
     "Age in predictor and CKD-EPI; component AUCs"),
    ("Reviewer 4", "4", "30 + manuscript text", "R1c3_*",
     "Temporal != external; weighted sensitivity emphasized"),
    ("Reviewer 4", "5", "28 (Part D)", "R3c1_R4c5_*", "Post-calibration comparison"),
    ("Reviewer 4", "minor (alcohol)", "35", "R2c3_*", "Alcohol sensitivity"),
    ("Reviewer 4", "minor (figures)", "16, 26", "all revision figures",
     "Labelled axes and legends"),
    ("Reviewer 4", "minor (BP protocol)", "38", "R4m_bp_protocol_*",
     "Oscillometric BP sensitivity"),
]
comment_output_map = pd.DataFrame(
    comment_map,
    columns=["reviewer", "comment", "module", "output_files", "analysis"]
)
display(comment_output_map)
comment_output_map.to_csv(
    REVISION_DIR / "REVIEWER_COMMENT_OUTPUT_MAP.csv", index=False
)


# =========================================================
# 2. Headline numbers for the response letter
# =========================================================
def _get(table, **conds):
    t = REVISION_TABLES.get(table)
    if t is None:
        return None
    m = np.ones(len(t), dtype=bool)
    for k, v in conds.items():
        m &= (t[k] == v).to_numpy()
    return t[m]

key_numbers = {
    "primary_frozen_threshold": SCREENING_THRESHOLD,
    "survey_weighted_threshold": WEIGHTED_SCREENING_THRESHOLD,
    "temporal_auc_primary": float(roc_auc_score(y_ext, p_ext)),
}

rows = _get("weighted_threshold", cohort="2021-2023 temporal")
if rows is not None:
    key_numbers["weighted_threshold_temporal"] = (
        rows.drop(columns="cohort").to_dict(orient="records")
    )

if "selection_under_calibration" in REVISION_TABLES:
    key_numbers["selection_under_calibration"] = (
        REVISION_TABLES["selection_under_calibration"][
            ["calibration_state", "selected_by_prespecified_rule",
             "best_auc_model", "max_auc_spread"]
        ].to_dict(orient="records")
    )

if "fully_nested_procedure" in REVISION_TABLES:
    key_numbers["fully_nested_vs_reported"] = (
        REVISION_TABLES["fully_nested_procedure"].to_dict(orient="records")
    )

if "comparators" in REVISION_TABLES:
    key_numbers["comparators_temporal"] = (
        REVISION_TABLES["comparators"][[
            "model", "temporal_roc_auc", "temporal_brier",
            "temporal_sensitivity", "temporal_specificity",
            "temporal_weighted_sensitivity"
        ]].to_dict(orient="records")
    )

if "comparator_differences" in REVISION_TABLES:
    key_numbers["comparator_differences"] = (
        REVISION_TABLES["comparator_differences"].to_dict(orient="records")
    )

if "alcohol_sensitivity" in REVISION_TABLES:
    key_numbers["alcohol_sensitivity"] = (
        REVISION_TABLES["alcohol_sensitivity"][[
            "model", "temporal_roc_auc", "temporal_auc_minus_V0",
            "temporal_auc_minus_V0_ci_low", "temporal_auc_minus_V0_ci_high"
        ]].to_dict(orient="records")
    )

if "component_age_performance" in REVISION_TABLES:
    key_numbers["component_and_age"] = (
        REVISION_TABLES["component_age_performance"][[
            "cohort", "outcome", "auc_primary", "auc_age_only",
            "auc_increment_over_age_only", "increment_ci_low",
            "increment_ci_high"
        ]].to_dict(orient="records")
    )

with open(REVISION_DIR / "REVISION_KEY_NUMBERS.json", "w") as f:
    json.dump(key_numbers, f, indent=2, default=float)

print(json.dumps(
    {k: v for k, v in key_numbers.items()
     if k in ["primary_frozen_threshold", "survey_weighted_threshold",
              "temporal_auc_primary"]},
    indent=2
))


# =========================================================
# 3. Revision reproducibility record
# =========================================================
revision_environment = {
    "manuscript": "MLWA-D-26-01812 (Revision 1)",
    "primary_analysis_unchanged": True,
    "selected_model": SELECTED_MODEL_NAME,
    "best_calibration": BEST_CALIBRATION,
    "screening_threshold_unweighted": SCREENING_THRESHOLD,
    "screening_threshold_survey_weighted": WEIGHTED_SCREENING_THRESHOLD,
    "n_boot_participant": N_BOOT,
    "n_boot_design_rao_wu": N_BOOT_DESIGN,
    "n_boot_subgroup": N_BOOT_SUBGROUP,
    "dca_threshold_range": [float(DCA_THRESHOLDS.min()), float(DCA_THRESHOLDS.max())],
    "development_n": int(len(y_dev)),
    "temporal_n": int(len(y_ext)),
    "revision_outputs": sorted(p.name for p in REVISION_DIR.glob("*")),
}
with open(REVISION_DIR / "reproducibility_environment_revision.json", "w") as f:
    json.dump(revision_environment, f, indent=2, default=str)

print("\nRevision outputs saved to:", REVISION_DIR)
print("Files:", len(revision_environment["revision_outputs"]))
