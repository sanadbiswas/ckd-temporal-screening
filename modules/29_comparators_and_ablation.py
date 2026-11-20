# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 1, main concern + comment 1 — added value over simpler models
#       and risk-based screening rules under comparable calibration and
#       evaluation; ablation of the 12 predictors vs major risk factors.
#   Reviewer 4, comment 1 — comparison with a parsimonious clinical rule and
#       calibrated logistic regression using decision-curve analysis,
#       testing burden, and number of additional cases identified.
#   Reviewer 4, comment 2 (part) — "test all high-risk patients" policy.
#
# Every model below is fitted with fit_frozen_variant() (module 26), i.e.
# exactly the primary procedure: development-only tuning, the selected
# calibration method, an 80%-sensitivity threshold from development OOF
# predictions, and unchanged application to 2021-2023.

# =========================================================
# 1. Comparator and ablation models
# =========================================================
selected_template = clone(MODEL_SPECS[SELECTED_MODEL_NAME]["pipeline"]
                          .named_steps["model"])
selected_params = MODEL_SPECS[SELECTED_MODEL_NAME]["params"]

lr_template = clone(MODEL_SPECS["Logistic Regression"]["pipeline"]
                    .named_steps["model"])
lr_params = MODEL_SPECS["Logistic Regression"]["params"]

MAJOR_RISK_FACTORS = ["RIDAGEYR", "has_diabetes_dx", "has_htn_dx"]

ABLATION_SETS = {
    "A1: age + diabetes + hypertension": MAJOR_RISK_FACTORS,
    "A2: A1 + systolic/diastolic BP": MAJOR_RISK_FACTORS + [
        "mean_sbp", "mean_dbp"
    ],
    "A3: A2 + BMI + waist": MAJOR_RISK_FACTORS + [
        "mean_sbp", "mean_dbp", "BMXBMI", "BMXWAIST"
    ],
    "A4: A3 + sex": MAJOR_RISK_FACTORS + [
        "mean_sbp", "mean_dbp", "BMXBMI", "BMXWAIST", "RIAGENDR"
    ],
}

comparator_results = {}

# Primary model: reuse the frozen objects, do NOT refit.
comparator_results[f"{SELECTED_MODEL_NAME} (primary, 12 predictors)"] = {
    "label": f"{SELECTED_MODEL_NAME} (primary, 12 predictors)",
    "oof": np.asarray(selected_internal_prob),
    "p_ext": np.asarray(p_ext),
    "threshold": SCREENING_THRESHOLD,
    "best_params": final_search.best_params_,
}

comparator_results["Calibrated logistic regression (12 predictors)"] = (
    fit_frozen_variant(
        "Calibrated logistic regression (12 predictors)",
        df[PRIMARY_FEATURES], ext[PRIMARY_FEATURES],
        lr_template, lr_params, seed_offset=2901
    )
)

comparator_results["Calibrated logistic regression (age, diabetes, hypertension)"] = (
    fit_frozen_variant(
        "Calibrated logistic regression (age, diabetes, hypertension)",
        df[MAJOR_RISK_FACTORS], ext[MAJOR_RISK_FACTORS],
        lr_template, lr_params, seed_offset=2902
    )
)

for i, (label, cols) in enumerate(ABLATION_SETS.items()):
    full_label = f"{SELECTED_MODEL_NAME} {label}"
    comparator_results[full_label] = fit_frozen_variant(
        full_label, df[cols], ext[cols],
        selected_template, selected_params, seed_offset=2910 + i
    )

comparator_table = pd.DataFrame([
    summarize_variant(
        r,
        extra={"n_predictors": (
            12 if "12 predictors" in name
            else 3 if "age, diabetes, hypertension" in name
            else len(next(v for k, v in ABLATION_SETS.items() if k in name))
        )}
    )
    for name, r in comparator_results.items()
])

display(comparator_table.round(4))
comparator_table.to_csv(
    REVISION_DIR / "R1c1_R4c1_comparator_and_ablation_models.csv", index=False
)
REVISION_TABLES["comparators"] = comparator_table


# =========================================================
# 2. Guideline-style risk-factor screening rules
# =========================================================
# Missing diagnosis fields cannot trigger a rule (treated as not flagged).
def risk_rule_flags(frame, include_age60):
    dm = (frame["has_diabetes_dx"] == "yes").to_numpy()
    htn = (frame["has_htn_dx"] == "yes").to_numpy()
    flag = dm | htn
    if include_age60:
        flag = flag | (frame["RIDAGEYR"].to_numpy() >= 60)
    return flag.astype(int)

RULES = {
    "Rule: diabetes or hypertension": False,
    "Rule: diabetes, hypertension, or age >=60": True,
}

rule_flags_dev = {name: risk_rule_flags(df, age60) for name, age60 in RULES.items()}
rule_flags_ext = {name: risk_rule_flags(ext, age60) for name, age60 in RULES.items()}
rule_flags_dev["Test all"] = np.ones(len(df), dtype=int)
rule_flags_ext["Test all"] = np.ones(len(ext), dtype=int)


# =========================================================
# 3. Burden-matched model thresholds (frozen on development data)
# =========================================================
# For each rule, choose the primary-model threshold that flags the SAME
# fraction of the development cohort as the rule, freeze it, and apply it
# to 2021-2023. This asks: for the same testing burden, does the model
# find more phenotype-positive adults than the rule?
burden_matched = {}
for name in RULES:
    q = rule_flags_dev[name].mean()
    thr = float(np.quantile(np.asarray(selected_internal_prob), 1 - q))
    burden_matched[name] = thr
    print(f"Burden matched to '{name}': dev positive rate={q:.3f}, "
          f"frozen model threshold={thr:.4f}")


# =========================================================
# 4. Temporal operating points and testing burden per 1,000 screened
# =========================================================
def burden_row(label, pred, y=y_ext_arr, w=None):
    op = operating_point_row(y, pred, sample_weight=w)
    tp, fn, tn, fp = sens_spec_counts(y, pred)
    n = len(y)
    if w is None:
        per_k = 1000 / n
        tests = (tp + fp) * per_k
        found = tp * per_k
        missed = fn * per_k
    else:
        w = np.asarray(w, dtype=float)
        tot = w.sum()
        tests = 1000 * w[pred == 1].sum() / tot
        found = 1000 * w[(pred == 1) & (y == 1)].sum() / tot
        missed = 1000 * w[(pred == 0) & (y == 1)].sum() / tot
    return {
        "strategy": label,
        **op,
        "tests_per_1000": tests,
        "cases_found_per_1000": found,
        "cases_missed_per_1000": missed,
        "tests_per_case_found": tests / found if found > 0 else np.nan,
    }

strategy_preds_ext = {}
for name, r in comparator_results.items():
    strategy_preds_ext[name] = (r["p_ext"] >= r["threshold"]).astype(int)
for name, flags in rule_flags_ext.items():
    strategy_preds_ext[name] = flags
for name, thr in burden_matched.items():
    strategy_preds_ext[f"Primary model, burden-matched to '{name}'"] = (
        np.asarray(p_ext) >= thr
    ).astype(int)

burden_rows = []
for label, pred in strategy_preds_ext.items():
    burden_rows.append({"weighting": "unweighted", **burden_row(label, pred)})
    burden_rows.append({
        "weighting": "survey-weighted (WTPH2YR)",
        **burden_row(label, pred, w=w_ext_arr)
    })

testing_burden = pd.DataFrame(burden_rows)
reference = testing_burden[
    testing_burden["strategy"] == "Rule: diabetes or hypertension"
].set_index("weighting")
testing_burden["additional_cases_vs_dm_htn_rule_per_1000"] = [
    r["cases_found_per_1000"] - reference.loc[r["weighting"], "cases_found_per_1000"]
    for _, r in testing_burden.iterrows()
]
testing_burden["additional_tests_vs_dm_htn_rule_per_1000"] = [
    r["tests_per_1000"] - reference.loc[r["weighting"], "tests_per_1000"]
    for _, r in testing_burden.iterrows()
]

display(testing_burden.round(3))
testing_burden.to_csv(
    REVISION_DIR / "R4c1_testing_burden_per_1000.csv", index=False
)
REVISION_TABLES["testing_burden"] = testing_burden


# =========================================================
# 5. Decision curves: models vs rules (temporal cohort)
# =========================================================
dca_comparison = {"threshold": DCA_THRESHOLDS}
for name in [
    f"{SELECTED_MODEL_NAME} (primary, 12 predictors)",
    "Calibrated logistic regression (12 predictors)",
    "Calibrated logistic regression (age, diabetes, hypertension)",
]:
    dca_comparison[name] = decision_curve(
        y_ext, comparator_results[name]["p_ext"], thresholds=DCA_THRESHOLDS
    )["model"].to_numpy()
for name in ["Rule: diabetes or hypertension",
             "Rule: diabetes, hypertension, or age >=60", "Test all"]:
    dca_comparison[name] = policy_decision_curve(
        y_ext, rule_flags_ext[name]
    )["net_benefit"].to_numpy()
dca_comparison["Test none"] = np.zeros(len(DCA_THRESHOLDS))
dca_comparison = pd.DataFrame(dca_comparison)
dca_comparison.to_csv(
    REVISION_DIR / "R4c1_dca_models_vs_rules_temporal.csv", index=False
)

fig, ax = plt.subplots(figsize=(8, 6.5))
line_styles = {
    f"{SELECTED_MODEL_NAME} (primary, 12 predictors)": "-",
    "Calibrated logistic regression (12 predictors)": "-",
    "Calibrated logistic regression (age, diabetes, hypertension)": "-.",
    "Rule: diabetes or hypertension": "--",
    "Rule: diabetes, hypertension, or age >=60": "--",
    "Test all": ":",
    "Test none": ":",
}
for name, ls in line_styles.items():
    ax.plot(dca_comparison["threshold"], dca_comparison[name],
            linestyle=ls, linewidth=2.0, label=name)
ax.axvline(SCREENING_THRESHOLD, color="grey", linewidth=1.0, linestyle="--")
# Conventional DCA display: clip far-negative "test all" values.
ax.set_ylim(-0.05, dca_comparison.drop(columns="threshold").max().max() + 0.02)
style_axes(ax, "Threshold probability", "Net benefit",
           "Decision Curves — Models vs Risk-Factor Rules (2021–2023)")
ax.legend(fontsize=8.5, frameon=True, loc="upper right").get_frame().set_edgecolor("black")
save_figure(fig, "decision_curve_models_vs_rules_temporal")


# =========================================================
# 6. Paired bootstrap of key differences (temporal cohort)
# =========================================================
primary_name = f"{SELECTED_MODEL_NAME} (primary, 12 predictors)"
p_primary = comparator_results[primary_name]["p_ext"]
pred_primary = strategy_preds_ext[primary_name]

difference_specs = []
for other in [
    "Calibrated logistic regression (12 predictors)",
    "Calibrated logistic regression (age, diabetes, hypertension)",
    f"{SELECTED_MODEL_NAME} A1: age + diabetes + hypertension",
    f"{SELECTED_MODEL_NAME} A4: A3 + sex",
]:
    difference_specs.append(("auc", other))
    difference_specs.append(("brier", other))
for rule in ["Rule: diabetes or hypertension",
             "Rule: diabetes, hypertension, or age >=60"]:
    difference_specs.append(("net_benefit_at_frozen_threshold", rule))
    difference_specs.append(("burden_matched_sensitivity", rule))

boot_store = defaultdict(list)
for idx in iid_bootstrap_indices(len(y_ext_arr), N_BOOT, RANDOM_STATE + 2950):
    yy = y_ext_arr[idx]
    if np.unique(yy).size < 2:
        continue
    for kind, other in difference_specs:
        if kind == "auc":
            d = (roc_auc_score(yy, p_primary[idx])
                 - roc_auc_score(yy, comparator_results[other]["p_ext"][idx]))
        elif kind == "brier":
            d = (brier_score_loss(yy, p_primary[idx])
                 - brier_score_loss(yy, comparator_results[other]["p_ext"][idx]))
        elif kind == "net_benefit_at_frozen_threshold":
            d = (net_benefit_binary(yy, pred_primary[idx], SCREENING_THRESHOLD)
                 - net_benefit_binary(yy, rule_flags_ext[other][idx],
                                      SCREENING_THRESHOLD))
        else:
            matched = strategy_preds_ext[
                f"Primary model, burden-matched to '{other}'"
            ][idx]
            rule = rule_flags_ext[other][idx]
            d = (matched[yy == 1].mean() - rule[yy == 1].mean())
        boot_store[(kind, other)].append(d)

difference_rows = []
for kind, other in difference_specs:
    if kind == "auc":
        point = roc_auc_score(y_ext, p_primary) - roc_auc_score(
            y_ext, comparator_results[other]["p_ext"])
    elif kind == "brier":
        point = brier_score_loss(y_ext, p_primary) - brier_score_loss(
            y_ext, comparator_results[other]["p_ext"])
    elif kind == "net_benefit_at_frozen_threshold":
        point = (net_benefit_binary(y_ext_arr, pred_primary, SCREENING_THRESHOLD)
                 - net_benefit_binary(y_ext_arr, rule_flags_ext[other],
                                      SCREENING_THRESHOLD))
    else:
        matched = strategy_preds_ext[f"Primary model, burden-matched to '{other}'"]
        point = (matched[y_ext_arr == 1].mean()
                 - rule_flags_ext[other][y_ext_arr == 1].mean())
    lo, hi = percentile_ci(boot_store[(kind, other)])
    difference_rows.append({
        "contrast": f"Primary minus {other}",
        "quantity": kind,
        "difference": point,
        "ci_low_95": lo,
        "ci_high_95": hi,
    })

comparator_differences = pd.DataFrame(difference_rows)
display(comparator_differences.round(4))
comparator_differences.to_csv(
    REVISION_DIR / "R1c1_R4c1_paired_differences_temporal.csv", index=False
)
REVISION_TABLES["comparator_differences"] = comparator_differences
