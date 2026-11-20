# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 2, comment 4  — why and how the one-standard-error rule was used
#   Reviewer 2, comment 5  — sigmoid vs isotonic: pragmatic choice, not superiority
#   Reviewer 3, minor 1    — how the Table 2 confidence intervals were built
#   Reviewer 3, comment 1  — would HGB still be chosen if all candidates were
#   Reviewer 4, comment 5     calibrated first? (common post-calibration comparison)
#   Reviewer 1, comment 2  — are the internal estimates independent of tuning?
#                            (fully nested evaluation of the whole procedure)
#
# The calibration-aware comparison in Part D is also one of the two
# "refinement vs original under identical conditions" analyses offered in
# response to Reviewer 1's main novelty concern.

from scipy.stats import t as t_dist

# =========================================================
# Part A — One-standard-error rule, made fully explicit
# =========================================================
# SE used for the cutoff: the sample SD (ddof = 1) of the BEST model's five
# outer-fold ROC-AUC values divided by sqrt(5). The cutoff is
#     best mean AUC - 1 x SE(best model).
# Models whose mean outer-fold AUC is at or above the cutoff form the
# competitive set; the lowest mean Brier score within that set is selected.

one_se_table = model_comparison[[
    "model", "mean_roc_auc", "sd_roc_auc", "se_roc_auc",
    "mean_brier", "mean_log_loss"
]].copy()

one_se_table["best_model_mean_auc"] = best_mean_auc
one_se_table["best_model_se_auc"] = best_auc_se
one_se_table["one_se_cutoff"] = auc_cutoff
one_se_table["in_competitive_set"] = one_se_table["mean_roc_auc"] >= auc_cutoff
one_se_table["selected"] = one_se_table["model"] == SELECTED_MODEL_NAME

# Fold-level AUCs of the best model, so the SE can be recomputed by hand.
best_model_name = model_comparison.loc[0, "model"]
best_fold_aucs = nested_results[best_model_name]["fold_metrics"]["roc_auc"]
print("Best-AUC model:", best_model_name)
print("Outer-fold AUCs:", np.round(best_fold_aucs.to_numpy(), 4))
print(
    f"SD = {best_fold_aucs.std(ddof=1):.4f}; "
    f"SE = SD / sqrt({len(best_fold_aucs)}) = {best_auc_se:.4f}; "
    f"cutoff = {best_mean_auc:.4f} - {best_auc_se:.4f} = {auc_cutoff:.4f}"
)

display(one_se_table.round(4))
one_se_table.to_csv(
    REVISION_DIR / "R2c4_one_se_rule_details.csv", index=False
)
pd.DataFrame({
    "outer_fold": np.arange(1, len(best_fold_aucs) + 1),
    "model": best_model_name,
    "roc_auc": best_fold_aucs.to_numpy()
}).to_csv(REVISION_DIR / "R2c4_best_model_fold_aucs.csv", index=False)
REVISION_TABLES["one_se_rule"] = one_se_table


# =========================================================
# Part B — Record of the Table 2 confidence-interval method
# =========================================================
ci_method_note = {
    "table": "Table 2 (nested cross-validation model comparison)",
    "unit_of_resampling": "outer cross-validation fold (k = 5)",
    "interval": "mean ± t(0.975, df = 4) × SD / sqrt(5)",
    "t_critical_value": float(t_dist.ppf(0.975, df=OUTER_FOLDS - 1)),
    "interpretation": (
        "Describes variability of performance across the five outer test "
        "folds; folds share training data, so the interval is approximate "
        "and likely somewhat narrow (Bengio & Grandvalet, 2004)."
    ),
    "source_file": "nested_cv_outer_fold_summary_CI.csv (module 07)",
}
with open(REVISION_DIR / "R3m1_table2_ci_method.json", "w") as f:
    json.dump(ci_method_note, f, indent=2)
print(json.dumps(ci_method_note, indent=2))


# =========================================================
# Part C — Sigmoid vs isotonic vs raw: paired bootstrap differences
# =========================================================
def paired_metric_differences(y, p_a, p_b, n_boot, seed):
    y = np.asarray(y)
    rows = []
    for idx in iid_bootstrap_indices(len(y), n_boot, seed):
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        a = p_a[idx]
        b = p_b[idx]
        rows.append({
            "roc_auc": roc_auc_score(yy, a) - roc_auc_score(yy, b),
            "brier": brier_score_loss(yy, a) - brier_score_loss(yy, b),
            "log_loss": log_loss(yy, a, labels=[0, 1])
                        - log_loss(yy, b, labels=[0, 1]),
            "ece": weighted_ece(yy, a, 10) - weighted_ece(yy, b, 10),
        })
    return pd.DataFrame(rows)

calibration_difference_rows = []
for a_name, b_name in [("Sigmoid", "Isotonic"), ("Sigmoid", "Raw"),
                       ("Isotonic", "Raw")]:
    p_a = np.asarray(calibration_candidates[a_name])
    p_b = np.asarray(calibration_candidates[b_name])
    boot = paired_metric_differences(
        y_dev_arr, p_a, p_b, N_BOOT, RANDOM_STATE + 2800
    )
    for metric in ["roc_auc", "brier", "log_loss", "ece"]:
        if metric == "roc_auc":
            point = roc_auc_score(y_dev, p_a) - roc_auc_score(y_dev, p_b)
        elif metric == "brier":
            point = brier_score_loss(y_dev, p_a) - brier_score_loss(y_dev, p_b)
        elif metric == "log_loss":
            point = log_loss(y_dev, p_a) - log_loss(y_dev, p_b)
        else:
            point = weighted_ece(y_dev, p_a, 10) - weighted_ece(y_dev, p_b, 10)
        lo, hi = percentile_ci(boot[metric])
        calibration_difference_rows.append({
            "comparison": f"{a_name} - {b_name}",
            "metric": metric,
            "difference": point,
            "ci_low_95": lo,
            "ci_high_95": hi,
            "ci_includes_zero": lo <= 0 <= hi,
        })

calibration_differences = pd.DataFrame(calibration_difference_rows)
display(calibration_differences.round(5))
calibration_differences.to_csv(
    REVISION_DIR / "R2c5_calibration_method_paired_differences.csv",
    index=False
)
REVISION_TABLES["calibration_method_differences"] = calibration_differences


# =========================================================
# Part D — Common post-calibration comparison of all five families
# =========================================================
# Same outer folds and the same tuned hyperparameters as module 06
# (best_params_by_model[model][fold - 1]); within each outer TRAINING fold
# the tuned pipeline is wrapped in CalibratedClassifierCV (sigmoid and
# isotonic, internal 5-fold) and evaluated on the untouched outer TEST fold.
# The prespecified selection rule (one-SE on AUC, then lowest Brier) is
# then re-applied under each calibration state.

calibrated_fold_rows = []
calibrated_oof = {
    state: {m: np.full(len(X_dev), np.nan) for m in MODEL_SPECS}
    for state in ["Raw", "Sigmoid", "Isotonic"]
}

outer_splits = list(outer_cv.split(X_dev, y_dev))

for model_name, spec in MODEL_SPECS.items():
    print(f"\n=== Calibrated comparison: {model_name} ===")
    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_te, y_te = X_dev.iloc[test_idx], y_dev.iloc[test_idx]

        tuned = clone(spec["pipeline"]).set_params(
            **best_params_by_model[model_name][fold - 1]
        )

        fitted = {
            "Raw": clone(tuned).fit(X_tr, y_tr),
            "Sigmoid": CalibratedClassifierCV(
                estimator=clone(tuned), method="sigmoid", cv=5, n_jobs=-1
            ).fit(X_tr, y_tr),
            "Isotonic": CalibratedClassifierCV(
                estimator=clone(tuned), method="isotonic", cv=5, n_jobs=-1
            ).fit(X_tr, y_tr),
        }

        for state, est in fitted.items():
            p = est.predict_proba(X_te)[:, 1]
            calibrated_oof[state][model_name][test_idx] = p
            intercept, slope = calibration_intercept_slope(y_te, p)
            calibrated_fold_rows.append({
                "model": model_name,
                "calibration": state,
                "fold": fold,
                **metrics_from_prob(y_te, p),
                "ece": weighted_ece(y_te, p, 10),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            })

        last = calibrated_fold_rows[-3:]
        print(
            f"fold {fold}: "
            + " | ".join(
                f"{r['calibration']} AUC={r['roc_auc']:.3f} "
                f"Brier={r['brier']:.4f}"
                for r in last
            )
        )

calibrated_folds = pd.DataFrame(calibrated_fold_rows)
calibrated_folds.to_csv(
    REVISION_DIR / "R3c1_R4c5_calibrated_comparison_fold_metrics.csv",
    index=False
)

calibrated_summary = (
    calibrated_folds
    .groupby(["calibration", "model"])
    .agg(
        mean_roc_auc=("roc_auc", "mean"),
        sd_roc_auc=("roc_auc", lambda v: v.std(ddof=1)),
        mean_pr_auc=("pr_auc", "mean"),
        mean_brier=("brier", "mean"),
        sd_brier=("brier", lambda v: v.std(ddof=1)),
        mean_log_loss=("log_loss", "mean"),
        mean_ece=("ece", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    .reset_index()
)
calibrated_summary["se_roc_auc"] = (
    calibrated_summary["sd_roc_auc"] / np.sqrt(OUTER_FOLDS)
)

selection_rows = []
for state, g in calibrated_summary.groupby("calibration"):
    g = g.sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    cutoff = g.loc[0, "mean_roc_auc"] - g.loc[0, "se_roc_auc"]
    competitive = g[g["mean_roc_auc"] >= cutoff].sort_values(
        ["mean_brier", "mean_roc_auc"], ascending=[True, False]
    )
    selection_rows.append({
        "calibration_state": state,
        "best_auc_model": g.loc[0, "model"],
        "best_mean_auc": g.loc[0, "mean_roc_auc"],
        "one_se_cutoff": cutoff,
        "competitive_set": ", ".join(competitive["model"]),
        "selected_by_prespecified_rule": competitive.iloc[0]["model"],
        "lowest_brier_overall": g.sort_values("mean_brier").iloc[0]["model"],
        "max_auc_spread": g["mean_roc_auc"].max() - g["mean_roc_auc"].min(),
        "max_brier_spread": g["mean_brier"].max() - g["mean_brier"].min(),
    })

selection_under_calibration = pd.DataFrame(selection_rows)

display(calibrated_summary.round(4))
display(selection_under_calibration.round(4))
calibrated_summary.to_csv(
    REVISION_DIR / "R3c1_R4c5_calibrated_comparison_summary.csv", index=False
)
selection_under_calibration.to_csv(
    REVISION_DIR / "R3c1_R4c5_selection_under_calibration.csv", index=False
)
REVISION_TABLES["calibrated_comparison"] = calibrated_summary
REVISION_TABLES["selection_under_calibration"] = selection_under_calibration

# Paired fold-level differences after calibration (selected method).
paired_cal_rows = []
cal_state = BEST_CALIBRATION if BEST_CALIBRATION != "Raw" else "Sigmoid"
sel = calibrated_folds[
    (calibrated_folds["model"] == SELECTED_MODEL_NAME)
    & (calibrated_folds["calibration"] == cal_state)
].sort_values("fold")
for other in MODEL_SPECS:
    if other == SELECTED_MODEL_NAME:
        continue
    oth = calibrated_folds[
        (calibrated_folds["model"] == other)
        & (calibrated_folds["calibration"] == cal_state)
    ].sort_values("fold")
    for metric in ["roc_auc", "brier", "log_loss", "ece"]:
        diff = sel[metric].to_numpy() - oth[metric].to_numpy()
        tcrit = t_dist.ppf(0.975, df=len(diff) - 1)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        paired_cal_rows.append({
            "comparison": f"{SELECTED_MODEL_NAME} vs {other}",
            "calibration": cal_state,
            "metric": metric,
            "mean_difference": diff.mean(),
            "ci_low_95": diff.mean() - tcrit * se,
            "ci_high_95": diff.mean() + tcrit * se,
        })
paired_calibrated_differences = pd.DataFrame(paired_cal_rows)
display(paired_calibrated_differences.round(4))
paired_calibrated_differences.to_csv(
    REVISION_DIR / "R3c1_R4c5_paired_differences_after_calibration.csv",
    index=False
)


# =========================================================
# Part E — Fully nested evaluation of the complete procedure (R1.2)
# =========================================================
# In the original pipeline the internal operating-point estimates
# (module 10/11) use hyperparameters tuned on ALL development data
# (module 09), so they are not strictly independent of tuning.
# Here every step is repeated inside each outer training fold:
#   tuning (the fold's own inner search from module 06)
#   -> calibration (BEST_CALIBRATION, inside the training fold)
#   -> 80%-sensitivity threshold (from training-fold OOF predictions)
# and the outer test fold is touched only once, for evaluation.

nested_procedure_rows = []
nested_procedure_oof = np.full(len(X_dev), np.nan)
nested_procedure_pred = np.full(len(X_dev), -1)

def calibrated_estimator(tuned_pipeline):
    if BEST_CALIBRATION == "Sigmoid":
        return CalibratedClassifierCV(
            estimator=clone(tuned_pipeline), method="sigmoid", cv=5, n_jobs=-1
        )
    if BEST_CALIBRATION == "Isotonic":
        return CalibratedClassifierCV(
            estimator=clone(tuned_pipeline), method="isotonic", cv=5, n_jobs=-1
        )
    return clone(tuned_pipeline)

for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
    X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
    X_te, y_te = X_dev.iloc[test_idx], y_dev.iloc[test_idx]

    tuned = clone(MODEL_SPECS[SELECTED_MODEL_NAME]["pipeline"]).set_params(
        **best_params_by_model[SELECTED_MODEL_NAME][fold - 1]
    )
    est = calibrated_estimator(tuned)

    inner_oof = cross_val_predict(
        clone(est), X_tr, y_tr,
        cv=StratifiedKFold(n_splits=5, shuffle=True,
                           random_state=RANDOM_STATE + 2900 + fold),
        method="predict_proba", n_jobs=-1
    )[:, 1]
    fold_threshold = threshold_for_target_sensitivity(
        y_tr, inner_oof, target=TARGET_SENSITIVITY
    )

    p_te = clone(est).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    nested_procedure_oof[test_idx] = p_te
    nested_procedure_pred[test_idx] = (p_te >= fold_threshold).astype(int)

    intercept, slope = calibration_intercept_slope(y_te, p_te)
    nested_procedure_rows.append({
        "fold": fold,
        "fold_threshold": fold_threshold,
        **metrics_from_prob(y_te, p_te, threshold=fold_threshold),
        "ece": weighted_ece(y_te, p_te, 10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    })

nested_procedure_folds = pd.DataFrame(nested_procedure_rows)

tp_, fn_, tn_, fp_ = sens_spec_counts(y_dev_arr, nested_procedure_pred)
nested_vs_reported = pd.DataFrame([
    {
        "estimate": "Reported internal OOF (hyperparameters tuned on all development data)",
        "roc_auc": internal_operating["roc_auc"],
        "brier": internal_operating["brier"],
        "sensitivity": internal_operating["sensitivity"],
        "specificity": internal_operating["specificity"],
        "threshold": SCREENING_THRESHOLD,
    },
    {
        "estimate": "Fully nested (tuning, calibration and threshold inside each outer fold) — mean of folds",
        "roc_auc": nested_procedure_folds["roc_auc"].mean(),
        "brier": nested_procedure_folds["brier"].mean(),
        "sensitivity": nested_procedure_folds["sensitivity"].mean(),
        "specificity": nested_procedure_folds["specificity"].mean(),
        "threshold": nested_procedure_folds["fold_threshold"].mean(),
    },
    {
        "estimate": "Fully nested — pooled outer-fold predictions",
        "roc_auc": roc_auc_score(y_dev, nested_procedure_oof),
        "brier": brier_score_loss(y_dev, nested_procedure_oof),
        "sensitivity": tp_ / (tp_ + fn_),
        "specificity": tn_ / (tn_ + fp_),
        "threshold": np.nan,
    },
])

display(nested_procedure_folds.round(4))
display(nested_vs_reported.round(4))
nested_procedure_folds.to_csv(
    REVISION_DIR / "R1c2_fully_nested_procedure_folds.csv", index=False
)
nested_vs_reported.to_csv(
    REVISION_DIR / "R1c2_fully_nested_vs_reported.csv", index=False
)
REVISION_TABLES["fully_nested_procedure"] = nested_vs_reported

print(
    "Optimism of reported internal AUC relative to the fully nested "
    f"estimate: {internal_operating['roc_auc'] - nested_vs_reported.loc[1, 'roc_auc']:+.4f}"
)
