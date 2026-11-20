# Modular pipeline stage.

# Source: final notebook cell 54
# ---------------------------------------------------------------
# Age-stratified thresholds, derived from development OOF only
# ---------------------------------------------------------------

dev_age = df["RIDAGEYR"].to_numpy()
dev_age_group = pd.cut(
    dev_age,
    bins=[17, 39, 59, np.inf],
    labels=["18-39", "40-59", "60+"]
)

age_thresholds = {}
for level in ["18-39", "40-59", "60+"]:
    mask = (dev_age_group == level)
    y_g = y_dev.to_numpy()[mask]
    p_g = np.asarray(selected_internal_prob)[mask]
    t_g = threshold_for_target_sensitivity(y_g, p_g, target=0.80)
    age_thresholds[level] = t_g

    dev_op = metrics_from_prob(y_g, p_g, threshold=t_g)
    print(
        f"[{level}] development n={mask.sum()}, "
        f"frozen threshold={t_g:.6f}, "
        f"dev sensitivity={dev_op['sensitivity']:.4f}, "
        f"dev specificity={dev_op['specificity']:.4f}"
    )

print("\nFrozen age-stratified thresholds (development OOF only):")
print(age_thresholds)


# Source: final notebook cell 56
# ---------------------------------------------------------------
# Apply frozen age-stratified thresholds to the 2021-2023 cohort
# ---------------------------------------------------------------

ext_eval["age_stratified_pred"] = np.nan

rows = []
for level in ["18-39", "40-59", "60+"]:
    mask = (ext_eval["age_group"] == level)
    y_g = ext_eval.loc[mask, "ckd"].to_numpy(dtype=int)
    p_g = ext_eval.loc[mask, "prob"].to_numpy(dtype=float)
    t_g = age_thresholds[level]

    global_metrics = metrics_from_prob(y_g, p_g, threshold=SCREENING_THRESHOLD)
    strat_metrics = metrics_from_prob(y_g, p_g, threshold=t_g)

    rows.append({
        "age_group": level,
        "n": int(mask.sum()),
        "cases": int(y_g.sum()),
        "global_threshold": SCREENING_THRESHOLD,
        "global_sensitivity": global_metrics["sensitivity"],
        "global_specificity": global_metrics["specificity"],
        "stratified_threshold": t_g,
        "stratified_sensitivity": strat_metrics["sensitivity"],
        "stratified_specificity": strat_metrics["specificity"],
    })

age_strat_comparison = pd.DataFrame(rows)
display(age_strat_comparison.round(4))

age_strat_comparison.to_csv(
    RESULTS_DIR / "temporal_age_stratified_threshold_comparison.csv",
    index=False
)

# Overall pooled performance if age-stratified thresholds were used
# instead of the single global threshold, computed across the whole
# temporal cohort (pooling predictions made under each participant's
# own age-specific threshold).
pooled_pred = np.zeros(len(ext_eval), dtype=int)
for level in ["18-39", "40-59", "60+"]:
    mask = (ext_eval["age_group"] == level).to_numpy()
    pooled_pred[mask] = (
        ext_eval.loc[mask, "prob"].to_numpy() >= age_thresholds[level]
    ).astype(int)

y_all = ext_eval["ckd"].to_numpy(dtype=int)
pooled_sensitivity = recall_score(y_all, pooled_pred, zero_division=0)
pooled_specificity = safe_specificity(y_all, pooled_pred)

print(
    f"\nPooled temporal performance under age-stratified thresholds: "
    f"sensitivity={pooled_sensitivity:.4f}, specificity={pooled_specificity:.4f}"
)
print(
    f"Pooled temporal performance under the single global threshold: "
    f"sensitivity={temporal_unweighted['sensitivity']:.4f}, "
    f"specificity={temporal_unweighted['specificity']:.4f}"
)


# Source: final notebook cell 58
# ---------------------------------------------------------------
# Diabetes-tier thresholds, derived from development OOF only
# ---------------------------------------------------------------

dev_diabetes = df["has_diabetes_dx"].to_numpy()

diabetes_thresholds = {}
for level in ["no", "yes"]:
    mask = (dev_diabetes == level)
    y_g = y_dev.to_numpy()[mask]
    p_g = np.asarray(selected_internal_prob)[mask]
    t_g = threshold_for_target_sensitivity(y_g, p_g, target=0.80)
    diabetes_thresholds[level] = t_g

    dev_op = metrics_from_prob(y_g, p_g, threshold=t_g)
    print(
        f"[diabetes={level}] development n={mask.sum()}, "
        f"frozen threshold={t_g:.6f}, "
        f"dev sensitivity={dev_op['sensitivity']:.4f}, "
        f"dev specificity={dev_op['specificity']:.4f}"
    )

rows = []
for level in ["no", "yes"]:
    mask = (ext_eval["has_diabetes_dx"] == level)
    y_g = ext_eval.loc[mask, "ckd"].to_numpy(dtype=int)
    p_g = ext_eval.loc[mask, "prob"].to_numpy(dtype=float)
    t_g = diabetes_thresholds[level]

    global_metrics = metrics_from_prob(y_g, p_g, threshold=SCREENING_THRESHOLD)
    strat_metrics = metrics_from_prob(y_g, p_g, threshold=t_g)

    rows.append({
        "diabetes_dx": level,
        "n": int(mask.sum()),
        "cases": int(y_g.sum()),
        "global_threshold": SCREENING_THRESHOLD,
        "global_sensitivity": global_metrics["sensitivity"],
        "global_specificity": global_metrics["specificity"],
        "stratified_threshold": t_g,
        "stratified_sensitivity": strat_metrics["sensitivity"],
        "stratified_specificity": strat_metrics["specificity"],
    })

diabetes_strat_comparison = pd.DataFrame(rows)
display(diabetes_strat_comparison.round(4))

diabetes_strat_comparison.to_csv(
    RESULTS_DIR / "temporal_diabetes_stratified_threshold_comparison.csv",
    index=False
)

