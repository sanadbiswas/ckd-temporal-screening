# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 4, minor — differences in blood-pressure protocols between
#       cycles: 2017-2018 development used manual auscultatory readings
#       (BPX_J), whereas 2021-2023 provides oscillometric readings only
#       (BPXO_L). NHANES 2017-2018 ALSO collected oscillometric readings
#       (BPXO_J), allowing a like-for-like sensitivity analysis.
#
# (1) Agreement between auscultatory and oscillometric BP in 2017-2018.
# (2) Input-shift check: change in frozen-model predictions on development
#     participants when oscillometric BP replaces auscultatory BP.
# (3) Like-for-like model: development refit with oscillometric BP using
#     the primary procedure, then evaluated on 2021-2023 (oscillometric).

# =========================================================
# 1. Oscillometric readings for the development cohort
# =========================================================
bpxo_j = read_j("BPXO_J.XPT")
bpxo_cols_sy = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in bpxo_j.columns]
bpxo_cols_di = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in bpxo_j.columns]

bpxo_dev = (
    pd.DataFrame({"SEQN": seqn_dev.to_numpy()})
    .merge(bpxo_j[["SEQN"] + bpxo_cols_sy + bpxo_cols_di],
           on="SEQN", how="left", validate="one_to_one")
)
osc_sbp_dev = bpxo_dev[bpxo_cols_sy].mean(axis=1).to_numpy()
osc_dbp_dev = bpxo_dev[bpxo_cols_di].mean(axis=1).to_numpy()

print("Development participants with oscillometric SBP:",
      int(np.isfinite(osc_sbp_dev).sum()), "of", len(df))
print("Development participants with auscultatory SBP:",
      int(df["mean_sbp"].notna().sum()), "of", len(df))


# =========================================================
# 2. Agreement (Bland-Altman summary)
# =========================================================
agreement_rows = []
for label, osc, aus in [
    ("Systolic BP", osc_sbp_dev, df["mean_sbp"].to_numpy()),
    ("Diastolic BP", osc_dbp_dev, df["mean_dbp"].to_numpy()),
]:
    both = np.isfinite(osc) & np.isfinite(aus)
    diff = osc[both] - aus[both]
    agreement_rows.append({
        "measure": label,
        "n_paired": int(both.sum()),
        "mean_auscultatory": aus[both].mean(),
        "mean_oscillometric": osc[both].mean(),
        "mean_difference_osc_minus_ausc": diff.mean(),
        "sd_difference": diff.std(ddof=1),
        "lower_limit_of_agreement": diff.mean() - 1.96 * diff.std(ddof=1),
        "upper_limit_of_agreement": diff.mean() + 1.96 * diff.std(ddof=1),
        "pearson_r": np.corrcoef(osc[both], aus[both])[0, 1],
    })
bp_agreement = pd.DataFrame(agreement_rows)
display(bp_agreement.round(2))
bp_agreement.to_csv(REVISION_DIR / "R4m_bp_protocol_agreement.csv", index=False)
REVISION_TABLES["bp_agreement"] = bp_agreement


# =========================================================
# 3. Input-shift check with the frozen model
# =========================================================
X_dev_osc = X_dev.copy()
X_dev_osc["mean_sbp"] = osc_sbp_dev
X_dev_osc["mean_dbp"] = osc_dbp_dev

p_dev_aus = FINAL_MODEL.predict_proba(X_dev)[:, 1]
p_dev_osc = FINAL_MODEL.predict_proba(X_dev_osc)[:, 1]
shift = p_dev_osc - p_dev_aus
reclassified = (
    (p_dev_osc >= SCREENING_THRESHOLD) != (p_dev_aus >= SCREENING_THRESHOLD)
)
bp_input_shift = pd.DataFrame([{
    "mean_prediction_change": shift.mean(),
    "mean_absolute_prediction_change": np.abs(shift).mean(),
    "p95_absolute_prediction_change": np.quantile(np.abs(shift), 0.95),
    "pct_reclassified_at_frozen_threshold": 100 * reclassified.mean(),
    "note": "Frozen model applied to development participants (in-sample; "
            "isolates the effect of the BP protocol on model inputs)",
}])
display(bp_input_shift.round(4))
bp_input_shift.to_csv(REVISION_DIR / "R4m_bp_protocol_input_shift.csv",
                      index=False)


# =========================================================
# 4. Like-for-like model: oscillometric BP in both cohorts
# =========================================================
df_osc = df.copy()
df_osc["mean_sbp"] = osc_sbp_dev
df_osc["mean_dbp"] = osc_dbp_dev

osc_variant = fit_frozen_variant(
    f"{SELECTED_MODEL_NAME} with oscillometric BP in development",
    df_osc[PRIMARY_FEATURES], ext[PRIMARY_FEATURES],
    selected_template, selected_params, seed_offset=3801
)

bp_protocol_models = pd.DataFrame([
    summarize_variant(comparator_results[f"{SELECTED_MODEL_NAME} (primary, 12 predictors)"],
                      extra={"development_bp_protocol": "auscultatory (BPX_J)"}),
    summarize_variant(osc_variant,
                      extra={"development_bp_protocol": "oscillometric (BPXO_J)"}),
])

bp_boot = []
for idx in iid_bootstrap_indices(len(y_ext_arr), N_BOOT, RANDOM_STATE + 3802):
    yy = y_ext_arr[idx]
    if np.unique(yy).size < 2:
        continue
    bp_boot.append(roc_auc_score(yy, osc_variant["p_ext"][idx])
                   - roc_auc_score(yy, p_ext_arr[idx]))
lo, hi = percentile_ci(bp_boot)
bp_protocol_models["temporal_auc_minus_primary"] = [
    0.0, roc_auc_score(y_ext, osc_variant["p_ext"]) - roc_auc_score(y_ext, p_ext)
]
bp_protocol_models["temporal_auc_minus_primary_ci_low"] = [np.nan, lo]
bp_protocol_models["temporal_auc_minus_primary_ci_high"] = [np.nan, hi]

display(bp_protocol_models.round(4))
bp_protocol_models.to_csv(
    REVISION_DIR / "R4m_bp_protocol_like_for_like_model.csv", index=False
)
REVISION_TABLES["bp_protocol_models"] = bp_protocol_models
