# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 2, comment 8 / Reviewer 3, minor 5 — 95% CIs for subgroup
#       sensitivity and specificity, especially in small strata.
#   Reviewer 2, comment 9 — stratified thresholds are exploratory; this
#       module quantifies them as hypothesis-generating comparisons.
#   Reviewer 4, comment 2 — net benefit of global vs stratified thresholds
#       vs a simple "test all high-risk patients" policy, and the model's
#       incremental triage value among adults with diagnosed diabetes.
#
# Intervals: Wilson score (sample level; valid at 0% and 100%, where the
# original percentile bootstrap collapses to zero width) and Rao-Wu design
# bootstrap for population-level (weighted) subgroup estimates, computed as
# domain estimates with the full-sample replicate weights.

# =========================================================
# 1. Replicate weights for the temporal cohort (reused below)
# =========================================================
ext_replicate_weights = np.vstack(list(
    design_bootstrap_weights(w_ext_arr, ext_strata, ext_psu,
                             N_BOOT_DESIGN, RANDOM_STATE + 3300)
)).astype(np.float32)
print("Temporal replicate weights:", ext_replicate_weights.shape)


def weighted_sens_spec(y, pred, w):
    pos = y == 1
    neg = ~pos
    sens = w[pos & (pred == 1)].sum() / w[pos].sum() if w[pos].sum() > 0 else np.nan
    spec = w[neg & (pred == 0)].sum() / w[neg].sum() if w[neg].sum() > 0 else np.nan
    return sens, spec


def subgroup_operating_rows(group_var, levels_series, pred, policy):
    rows = []
    levels = pd.Series(levels_series).astype(object).where(
        pd.Series(levels_series).notna(), "missing"
    ).to_numpy()
    for level in pd.unique(levels):
        m = levels == level
        y_g = y_ext_arr[m]
        pred_g = pred[m]
        tp, fn, tn, fp = sens_spec_counts(y_g, pred_g)
        s_lo, s_hi = wilson_ci(tp, tp + fn)
        c_lo, c_hi = wilson_ci(tn, tn + fp)

        w_sens, w_spec = weighted_sens_spec(y_g, pred_g, w_ext_arr[m])
        rep_sens, rep_spec = [], []
        for rep_w in ext_replicate_weights:
            a, b = weighted_sens_spec(y_g, pred_g, rep_w[m].astype(float))
            rep_sens.append(a)
            rep_spec.append(b)
        ws_lo, ws_hi = percentile_ci(rep_sens)
        wc_lo, wc_hi = percentile_ci(rep_spec)

        rows.append({
            "policy": policy,
            "variable": group_var,
            "subgroup": str(level),
            "n": int(m.sum()),
            "cases": int(tp + fn),
            "controls": int(tn + fp),
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "sensitivity_wilson_low": s_lo,
            "sensitivity_wilson_high": s_hi,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "specificity_wilson_low": c_lo,
            "specificity_wilson_high": c_hi,
            "weighted_sensitivity": w_sens,
            "weighted_sensitivity_design_low": ws_lo,
            "weighted_sensitivity_design_high": ws_hi,
            "weighted_specificity": w_spec,
            "weighted_specificity_design_low": wc_lo,
            "weighted_specificity_design_high": wc_hi,
            "small_stratum_flag": (tp + fn) < 50 or (tn + fp) < 50,
        })
    return rows


# =========================================================
# 2. Screening policies (all frozen on development data)
# =========================================================
ext_age_group = ext_eval["age_group"].astype(str).to_numpy()
ext_dm = ext_eval["has_diabetes_dx"].to_numpy()
prob_ext = ext_eval["prob"].to_numpy()

policy_preds = {}
policy_preds["Global threshold (primary)"] = (
    prob_ext >= SCREENING_THRESHOLD
).astype(int)

age_thr_vec = np.array([age_thresholds.get(g, SCREENING_THRESHOLD)
                        for g in ext_age_group])
policy_preds["Age-stratified thresholds (exploratory)"] = (
    prob_ext >= age_thr_vec
).astype(int)

dm_thr_vec = np.array([
    diabetes_thresholds.get(v, SCREENING_THRESHOLD)
    if isinstance(v, str) else SCREENING_THRESHOLD
    for v in ext_dm
])
policy_preds["Diabetes-stratified thresholds (exploratory)"] = (
    prob_ext >= dm_thr_vec
).astype(int)

policy_preds["Test all with diabetes; global threshold otherwise"] = np.where(
    ext_dm == "yes", 1, (prob_ext >= SCREENING_THRESHOLD).astype(int)
)
policy_preds["Rule: diabetes or hypertension"] = rule_flags_ext[
    "Rule: diabetes or hypertension"
]
policy_preds["Test all"] = np.ones(len(prob_ext), dtype=int)


# =========================================================
# 3. Subgroup sensitivity/specificity with CIs for every policy
# =========================================================
subgroup_ci_rows = []
for policy in [
    "Global threshold (primary)",
    "Age-stratified thresholds (exploratory)",
    "Diabetes-stratified thresholds (exploratory)",
]:
    pred = policy_preds[policy]
    for var in ["RIAGENDR", "RIDRETH1", "age_group",
                "has_diabetes_dx", "has_htn_dx"]:
        subgroup_ci_rows.extend(
            subgroup_operating_rows(var, ext_eval[var].to_numpy(), pred, policy)
        )

subgroup_ci = pd.DataFrame(subgroup_ci_rows)
display(subgroup_ci[subgroup_ci["policy"] == "Global threshold (primary)"].round(4))
subgroup_ci.to_csv(
    REVISION_DIR / "R2c8_R3m5_subgroup_sens_spec_CI.csv", index=False
)
REVISION_TABLES["subgroup_ci"] = subgroup_ci


# =========================================================
# 4. Pooled operating characteristics and net benefit by policy
# =========================================================
NB_POINTS = sorted(set([0.05, 0.10, round(SCREENING_THRESHOLD, 4), 0.15, 0.20]))

policy_rows = []
for policy, pred in policy_preds.items():
    tp, fn, tn, fp = sens_spec_counts(y_ext_arr, pred)
    s_lo, s_hi = wilson_ci(tp, tp + fn)
    c_lo, c_hi = wilson_ci(tn, tn + fp)
    row = {
        "policy": policy,
        "sensitivity": tp / (tp + fn),
        "sensitivity_wilson_low": s_lo,
        "sensitivity_wilson_high": s_hi,
        "specificity": tn / (tn + fp),
        "specificity_wilson_low": c_lo,
        "specificity_wilson_high": c_hi,
        "positive_rate": pred.mean(),
        "tests_per_1000": 1000 * pred.mean(),
        "cases_found_per_1000": 1000 * tp / len(pred),
    }
    for pt in NB_POINTS:
        row[f"net_benefit_pt_{pt}"] = net_benefit_binary(y_ext_arr, pred, pt)
        row[f"weighted_net_benefit_pt_{pt}"] = net_benefit_binary(
            y_ext_arr, pred, pt, sample_weight=w_ext_arr
        )
    policy_rows.append(row)

policy_comparison = pd.DataFrame(policy_rows)
display(policy_comparison.round(4))
policy_comparison.to_csv(
    REVISION_DIR / "R4c2_policy_comparison_net_benefit.csv", index=False
)
REVISION_TABLES["policy_comparison"] = policy_comparison

# Bootstrap CIs for net-benefit differences vs the global threshold
nb_diff_store = defaultdict(list)
for idx in iid_bootstrap_indices(len(y_ext_arr), N_BOOT, RANDOM_STATE + 3301):
    yy = y_ext_arr[idx]
    for policy, pred in policy_preds.items():
        if policy == "Global threshold (primary)":
            continue
        for pt in NB_POINTS:
            nb_diff_store[(policy, pt)].append(
                net_benefit_binary(yy, pred[idx], pt)
                - net_benefit_binary(
                    yy, policy_preds["Global threshold (primary)"][idx], pt)
            )

nb_diff_rows = []
for (policy, pt), vals in nb_diff_store.items():
    point = (net_benefit_binary(y_ext_arr, policy_preds[policy], pt)
             - net_benefit_binary(y_ext_arr,
                                  policy_preds["Global threshold (primary)"], pt))
    lo, hi = percentile_ci(vals)
    nb_diff_rows.append({
        "policy": policy,
        "threshold_probability": pt,
        "net_benefit_minus_global": point,
        "ci_low_95": lo,
        "ci_high_95": hi,
    })
policy_nb_differences = pd.DataFrame(nb_diff_rows)
display(policy_nb_differences.round(5))
policy_nb_differences.to_csv(
    REVISION_DIR / "R4c2_policy_net_benefit_differences.csv", index=False
)


# =========================================================
# 5. Incremental triage value WITHIN diagnosed diabetes
# =========================================================
# Among adults with diagnosed diabetes, is ranking by the model better
# than testing everyone (the guideline default)?
dm_mask = ext_dm == "yes"
dca_dm = decision_curve(y_ext_arr[dm_mask], prob_ext[dm_mask],
                        thresholds=DCA_THRESHOLDS)
dca_dm.to_csv(REVISION_DIR / "R4c2_dca_within_diabetes.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot(dca_dm["threshold"], dca_dm["model"], linewidth=2.0,
        label="Model (continuous risk)")
ax.plot(dca_dm["threshold"], dca_dm["treat_all"], linestyle="--",
        linewidth=2.0, label="Test all with diabetes")
ax.plot(dca_dm["threshold"], dca_dm["treat_none"], linestyle=":",
        linewidth=2.0, label="Test none")
style_axes(ax, "Threshold probability", "Net benefit",
           "Decision Curve Within Diagnosed Diabetes (2021–2023)")
save_figure(fig, "decision_curve_within_diabetes_temporal")

first_model_advantage = dca_dm.loc[
    dca_dm["model"] > dca_dm["treat_all"] + 1e-9, "threshold"
]
print(
    "Within diagnosed diabetes, model net benefit first exceeds 'test all' "
    "at threshold probability: "
    + (f"{first_model_advantage.iloc[0]:.3f}" if len(first_model_advantage)
       else "never within 0.03–0.40")
)
