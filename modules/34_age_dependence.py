# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 2, comment 11 — is the model mainly an age-driven risk
#       stratifier supplemented by cardiometabolic characteristics?
#   Reviewer 4, comment 3  — age is both a predictor and a term in the
#       CKD-EPI equation defining low eGFR; interpret age importance and
#       the different component AUCs (low eGFR vs albuminuria).
#   Reviewer 3, minor 2    — performance for the composite phenotype vs
#       its individual components, in both cohorts.
#
# Three complementary views:
#   (1) an AGE-ONLY model fitted with the primary procedure: what age alone
#       achieves, including through its mechanical role in eGFR;
#   (2) the primary model's increment over age alone, per outcome;
#   (3) discrimination WITHIN age bands, where age contributes little,
#       so remaining discrimination reflects the non-age predictors.

# =========================================================
# 1. Age-only model (same family and procedure as the primary model)
# =========================================================
age_only = fit_frozen_variant(
    f"{SELECTED_MODEL_NAME} age only",
    df[["RIDAGEYR"]], ext[["RIDAGEYR"]],
    selected_template, selected_params, seed_offset=3401
)

p_age_only_ext = age_only["p_ext"]
p_age_only_dev = age_only["oof"]


# =========================================================
# 2. Composite vs component discrimination, both cohorts
# =========================================================
outcome_sets = {
    "2017-2018 development OOF": {
        "CKD screening phenotype": y_dev_arr,
        "Reduced eGFR (<60)": dev_components["low_egfr"].to_numpy(dtype=int),
        "Albuminuria (ACR >=30)": dev_components["albuminuria"].to_numpy(dtype=int),
    },
    "2021-2023 temporal": {
        "CKD screening phenotype": y_ext_arr,
        "Reduced eGFR (<60)": ext["low_egfr"].to_numpy(dtype=int),
        "Albuminuria (ACR >=30)": ext["albuminuria"].to_numpy(dtype=int),
    },
}
cohort_predictions = {
    "2017-2018 development OOF": {
        "primary": np.asarray(selected_internal_prob),
        "age_only": p_age_only_dev,
        "age_removed": np.asarray(agefree_oof),
    },
    "2021-2023 temporal": {
        "primary": p_ext_arr,
        "age_only": p_age_only_ext,
        "age_removed": np.asarray(p_ext_agefree),
    },
}

component_rows = []
for cohort, outcomes in outcome_sets.items():
    preds = cohort_predictions[cohort]
    thr_pred = (preds["primary"] >= SCREENING_THRESHOLD).astype(int)
    for outcome, yy in outcomes.items():
        boot = defaultdict(list)
        for idx in iid_bootstrap_indices(len(yy), N_BOOT, RANDOM_STATE + 3402):
            y_b = yy[idx]
            if np.unique(y_b).size < 2:
                continue
            a_p = roc_auc_score(y_b, preds["primary"][idx])
            a_a = roc_auc_score(y_b, preds["age_only"][idx])
            a_r = roc_auc_score(y_b, preds["age_removed"][idx])
            boot["primary"].append(a_p)
            boot["age_only"].append(a_a)
            boot["age_removed"].append(a_r)
            boot["increment"].append(a_p - a_a)

        auc_p = roc_auc_score(yy, preds["primary"])
        auc_a = roc_auc_score(yy, preds["age_only"])
        auc_r = roc_auc_score(yy, preds["age_removed"])
        tp, fn, _, _ = sens_spec_counts(yy, thr_pred)
        s_lo, s_hi = wilson_ci(tp, tp + fn)

        row = {
            "cohort": cohort,
            "outcome": outcome,
            "n_positive": int(yy.sum()),
            "prevalence": yy.mean(),
            "auc_primary": auc_p,
            "auc_age_only": auc_a,
            "auc_age_removed": auc_r,
            "auc_increment_over_age_only": auc_p - auc_a,
            "pr_auc_primary": average_precision_score(yy, preds["primary"]),
            "sensitivity_at_frozen_threshold": tp / (tp + fn),
            "sensitivity_wilson_low": s_lo,
            "sensitivity_wilson_high": s_hi,
        }
        for key in ["primary", "age_only", "age_removed", "increment"]:
            lo, hi = percentile_ci(boot[key])
            row[f"{key}_ci_low"] = lo
            row[f"{key}_ci_high"] = hi
        component_rows.append(row)

component_performance = pd.DataFrame(component_rows)
display(component_performance.round(4))
component_performance.to_csv(
    REVISION_DIR / "R3m2_R4c3_component_and_age_performance.csv", index=False
)
REVISION_TABLES["component_age_performance"] = component_performance


# =========================================================
# 3. Discrimination within age bands (temporal), per outcome
# =========================================================
within_age_rows = []
age_bands = pd.cut(ext["RIDAGEYR"], bins=[17, 39, 59, np.inf],
                   labels=["18-39", "40-59", "60+"]).astype(str).to_numpy()
for outcome, yy in outcome_sets["2021-2023 temporal"].items():
    for band in ["18-39", "40-59", "60+"]:
        m = age_bands == band
        y_g = yy[m]
        if y_g.sum() < 10 or (len(y_g) - y_g.sum()) < 10:
            within_age_rows.append({"outcome": outcome, "age_group": band,
                                    "n": int(m.sum()),
                                    "cases": int(y_g.sum()),
                                    "status": "insufficient_events"})
            continue
        p_g = p_ext_arr[m]
        pa_g = p_age_only_ext[m]
        boot_p, boot_a = [], []
        for idx in iid_bootstrap_indices(len(y_g), N_BOOT_SUBGROUP,
                                         RANDOM_STATE + 3403):
            if np.unique(y_g[idx]).size < 2:
                continue
            boot_p.append(roc_auc_score(y_g[idx], p_g[idx]))
            boot_a.append(roc_auc_score(y_g[idx], pa_g[idx]))
        lo_p, hi_p = percentile_ci(boot_p)
        lo_a, hi_a = percentile_ci(boot_a)
        within_age_rows.append({
            "outcome": outcome,
            "age_group": band,
            "n": int(m.sum()),
            "cases": int(y_g.sum()),
            "status": "evaluated",
            "auc_primary": roc_auc_score(y_g, p_g),
            "auc_primary_ci_low": lo_p,
            "auc_primary_ci_high": hi_p,
            "auc_age_only": roc_auc_score(y_g, pa_g),
            "auc_age_only_ci_low": lo_a,
            "auc_age_only_ci_high": hi_a,
        })

within_age_discrimination = pd.DataFrame(within_age_rows)
display(within_age_discrimination.round(4))
within_age_discrimination.to_csv(
    REVISION_DIR / "R2c11_R4c3_within_age_discrimination.csv", index=False
)
REVISION_TABLES["within_age_discrimination"] = within_age_discrimination
