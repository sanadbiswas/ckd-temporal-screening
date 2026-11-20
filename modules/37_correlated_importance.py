# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 3, minor 3 — permutation importance can be diluted when
#       predictors are correlated (e.g. systolic/diastolic BP, BMI/waist,
#       hypertension diagnosis/BP), because permuting one member leaves its
#       information available through the others.
#
# (1) Spearman correlation matrix of the 12 predictors (development data).
# (2) GROUPED permutation importance on the temporal cohort: correlated
#     predictors are permuted jointly (same row permutation), so their
#     shared information is removed together.

# =========================================================
# 1. Correlation among predictors
# =========================================================
EDU_ORDER = {
    "less_than_9th_grade": 1, "9th_to_11th_grade": 2,
    "high_school_grad_or_GED": 3, "some_college_or_AA_degree": 4,
    "college_grad_or_above": 5,
}

def numeric_predictors(frame):
    out = frame[PRIMARY_FEATURES].copy()
    out["RIAGENDR"] = (out["RIAGENDR"] == "female").astype(float)
    out["DMDEDUC2"] = out["DMDEDUC2"].map(EDU_ORDER)
    for col in ["has_htn_dx", "has_diabetes_dx", "smoked_100_cigs"]:
        out[col] = out[col].map({"yes": 1.0, "no": 0.0})
    return out.rename(columns=FEATURE_LABELS)

predictor_correlation = numeric_predictors(df).corr(method="spearman")
display(predictor_correlation.round(2))
predictor_correlation.to_csv(
    REVISION_DIR / "R3m3_predictor_spearman_correlation.csv"
)

corr_pairs = (
    predictor_correlation.where(
        np.triu(np.ones(predictor_correlation.shape, dtype=bool), k=1)
    )
    .stack()
    .rename("spearman_rho")
    .reset_index()
    .rename(columns={"level_0": "predictor_1", "level_1": "predictor_2"})
)
corr_pairs["abs_rho"] = corr_pairs["spearman_rho"].abs()
corr_pairs = corr_pairs.sort_values("abs_rho", ascending=False)
display(corr_pairs.head(10).round(3))
corr_pairs.to_csv(REVISION_DIR / "R3m3_predictor_correlation_pairs.csv",
                  index=False)


# =========================================================
# 2. Grouped permutation importance (temporal cohort)
# =========================================================
PREDICTOR_GROUPS = {
    "Age": ["RIDAGEYR"],
    "Sex": ["RIAGENDR"],
    "Blood pressure (SBP + DBP)": ["mean_sbp", "mean_dbp"],
    "Hypertension diagnosis + BP": ["has_htn_dx", "mean_sbp", "mean_dbp"],
    "Adiposity (BMI + waist)": ["BMXBMI", "BMXWAIST"],
    "Diagnosed diabetes": ["has_diabetes_dx"],
    "Cardiometabolic (diabetes + hypertension + BP + adiposity)": [
        "has_diabetes_dx", "has_htn_dx", "mean_sbp", "mean_dbp",
        "BMXBMI", "BMXWAIST"
    ],
    "Socioeconomic (education + income-to-poverty)": ["DMDEDUC2", "INDFMPIR"],
    "Lifestyle (smoking + alcohol)": ["smoked_100_cigs", "ALQ130"],
    "All non-age predictors": [c for c in PRIMARY_FEATURES if c != "RIDAGEYR"],
}

N_PERM_REPEATS = 30
base_auc = roc_auc_score(y_ext, p_ext)
rng_perm = np.random.default_rng(RANDOM_STATE + 3700)

grouped_rows = []
for group, cols in PREDICTOR_GROUPS.items():
    drops = []
    for _ in range(N_PERM_REPEATS):
        X_perm = X_ext.copy()
        order = rng_perm.permutation(len(X_perm))
        X_perm[cols] = X_ext[cols].to_numpy()[order]
        drops.append(
            base_auc - roc_auc_score(
                y_ext, FINAL_MODEL.predict_proba(X_perm)[:, 1]
            )
        )
    single_sum = perm_df.set_index("feature").loc[
        cols, "importance_mean"
    ].sum()
    grouped_rows.append({
        "group": group,
        "predictors": ", ".join(FEATURE_LABELS.get(c, c) for c in cols),
        "n_predictors": len(cols),
        "grouped_auc_drop_mean": float(np.mean(drops)),
        "grouped_auc_drop_sd": float(np.std(drops, ddof=1)),
        "sum_of_individual_drops": float(single_sum),
        "grouped_minus_sum_of_individual": float(np.mean(drops) - single_sum),
    })

grouped_importance = pd.DataFrame(grouped_rows).sort_values(
    "grouped_auc_drop_mean", ascending=False
)
display(grouped_importance.round(4))
grouped_importance.to_csv(
    REVISION_DIR / "R3m3_grouped_permutation_importance.csv", index=False
)
REVISION_TABLES["grouped_importance"] = grouped_importance
print(
    "A grouped drop larger than the sum of individual drops indicates "
    "shared (redundant) information that single-feature permutation hides."
)
