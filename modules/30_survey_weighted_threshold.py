# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 1, comment 3 — select an 80%-sensitivity threshold using
#       survey-weighted development predictions, apply it unchanged to the
#       temporal cohort, and compare with the current threshold.
#   Reviewer 2, comment 7 — population-level vs sample-level screening.
#   Reviewer 3, comment 3 — justification of the unweighted criterion and
#       the sensitivity/specificity trade-off of a weighted threshold.
#   Reviewer 4, comment 4 — population-level sensitivity below target.
#
# This is the second "refinement vs original under identical conditions"
# analysis offered for Reviewer 1's novelty concern: same frozen model,
# same development OOF predictions, only the threshold-selection rule
# changes (sample-level vs population-level sensitivity target).

# =========================================================
# 1. Derive the population-level threshold on development data only
# =========================================================
dev_prob = np.asarray(selected_internal_prob)
p_ext_arr = np.asarray(p_ext)

WEIGHTED_SCREENING_THRESHOLD = weighted_threshold_for_target_sensitivity(
    y_dev_arr, dev_prob, w_dev, target=TARGET_SENSITIVITY
)

# Sanity check: the weighted routine with unit weights must reproduce the
# original unweighted threshold from module 11.
check_threshold = weighted_threshold_for_target_sensitivity(
    y_dev_arr, dev_prob, np.ones(len(y_dev_arr)), target=TARGET_SENSITIVITY
)
print(f"Unweighted threshold (module 11):          {SCREENING_THRESHOLD:.6f}")
print(f"Same rule re-derived with unit weights:   {check_threshold:.6f}")
print(f"Survey-weighted 80%-sensitivity threshold: {WEIGHTED_SCREENING_THRESHOLD:.6f}")

# roc_curve() (used in module 11) drops collinear ROC points, so the two
# routines can return thresholds that differ in the 5th-6th decimal while
# classifying participants identically. Compare the resulting operating
# points rather than the raw threshold values.
_op_module11 = metrics_from_prob(y_dev, dev_prob, threshold=SCREENING_THRESHOLD)
_op_check = metrics_from_prob(y_dev, dev_prob, threshold=check_threshold)
if (round(_op_module11["sensitivity"], 3) != round(_op_check["sensitivity"], 3)
        or round(_op_module11["specificity"], 3) != round(_op_check["specificity"], 3)):
    print("WARNING: unit-weight check gives a different operating point than module 11.")
else:
    print("Unit-weight check: same operating point as module 11 (OK).")


# =========================================================
# 2. Compare both frozen thresholds in both cohorts, with design-based CIs
# =========================================================
THRESHOLD_RULES = {
    "Current (unweighted 80% target)": SCREENING_THRESHOLD,
    "Revised (survey-weighted 80% target)": WEIGHTED_SCREENING_THRESHOLD,
}

def threshold_metrics(y, p, thr, w):
    pred = (p >= thr).astype(int)
    out = {}
    for prefix, ww in [("unweighted", None), ("weighted", w)]:
        op = operating_point_row(y, pred, sample_weight=ww)
        for k, v in op.items():
            out[f"{prefix}_{k}"] = v
    return out

cohort_inputs = {
    "2017-2018 development OOF": (y_dev_arr, dev_prob, w_dev,
                                  dev_strata, dev_psu),
    "2021-2023 temporal": (y_ext_arr, p_ext_arr, w_ext_arr,
                           ext_strata, ext_psu),
}

threshold_rows = []
for cohort, (yy, pp, ww, st, ps) in cohort_inputs.items():
    # Point estimates
    points = {name: threshold_metrics(yy, pp, thr, ww)
              for name, thr in THRESHOLD_RULES.items()}

    # Design-based bootstrap: the same replicate weights for both rules, so
    # the difference between rules is also a paired, design-aware estimate.
    boot = defaultdict(list)
    for rep_w in design_bootstrap_weights(ww, st, ps, N_BOOT_DESIGN,
                                          RANDOM_STATE + 3000):
        rep = {}
        for name, thr in THRESHOLD_RULES.items():
            op = operating_point_row(yy, (pp >= thr).astype(int),
                                     sample_weight=rep_w)
            rep[name] = op
            for k, v in op.items():
                boot[(name, k)].append(v)
        for k in ["sensitivity", "specificity", "positive_rate", "ppv"]:
            boot[("difference", k)].append(
                rep["Revised (survey-weighted 80% target)"][k]
                - rep["Current (unweighted 80% target)"][k]
            )

    for name, thr in THRESHOLD_RULES.items():
        row = {"cohort": cohort, "threshold_rule": name, "threshold": thr,
               **points[name]}
        for k in ["sensitivity", "specificity", "ppv", "positive_rate"]:
            lo, hi = percentile_ci(boot[(name, k)])
            row[f"weighted_{k}_design_ci_low"] = lo
            row[f"weighted_{k}_design_ci_high"] = hi
        threshold_rows.append(row)

    diff_row = {"cohort": cohort,
                "threshold_rule": "Difference (revised - current), weighted",
                "threshold": WEIGHTED_SCREENING_THRESHOLD - SCREENING_THRESHOLD}
    for k in ["sensitivity", "specificity", "positive_rate", "ppv"]:
        diff_row[f"weighted_{k}"] = (
            points["Revised (survey-weighted 80% target)"][f"weighted_{k}"]
            - points["Current (unweighted 80% target)"][f"weighted_{k}"]
        )
        lo, hi = percentile_ci(boot[("difference", k)])
        diff_row[f"weighted_{k}_design_ci_low"] = lo
        diff_row[f"weighted_{k}_design_ci_high"] = hi
    threshold_rows.append(diff_row)

weighted_threshold_comparison = pd.DataFrame(threshold_rows)
display(weighted_threshold_comparison.round(4))
weighted_threshold_comparison.to_csv(
    REVISION_DIR / "R1c3_weighted_vs_unweighted_threshold.csv", index=False
)
REVISION_TABLES["weighted_threshold"] = weighted_threshold_comparison

temporal_rev = weighted_threshold_comparison[
    (weighted_threshold_comparison["cohort"] == "2021-2023 temporal")
    & (weighted_threshold_comparison["threshold_rule"]
       == "Revised (survey-weighted 80% target)")
].iloc[0]
print(
    "\nTemporal weighted sensitivity at the survey-weighted threshold: "
    f"{temporal_rev['weighted_sensitivity']:.4f} "
    f"(design-based 95% CI {temporal_rev['weighted_sensitivity_design_ci_low']:.4f}"
    f"–{temporal_rev['weighted_sensitivity_design_ci_high']:.4f})"
)


# =========================================================
# 3. Why do weighted and unweighted sensitivity differ?
# =========================================================
# The weighted sensitivity is a weighted average of age-group sensitivities.
# NHANES oversamples older adults, whose cases are easiest to detect, so
# the population (weighted) case mix contains relatively more younger
# cases than the sample does.
case_mix_rows = []
for cohort, (yy, pp, ww, _, _) in cohort_inputs.items():
    age = (df["RIDAGEYR"] if cohort.startswith("2017") else ext["RIDAGEYR"]).to_numpy()
    groups = pd.cut(age, bins=[17, 39, 59, np.inf], labels=["18-39", "40-59", "60+"])
    cases = yy == 1
    for level in ["18-39", "40-59", "60+"]:
        m = (groups == level) & cases
        case_mix_rows.append({
            "cohort": cohort,
            "age_group": level,
            "share_of_cases_unweighted": m.sum() / cases.sum(),
            "share_of_cases_weighted": ww[m].sum() / ww[cases].sum(),
            "sensitivity_at_current_threshold": (
                (pp[m] >= SCREENING_THRESHOLD).mean() if m.sum() else np.nan
            ),
        })
case_mix = pd.DataFrame(case_mix_rows)
display(case_mix.round(4))
case_mix.to_csv(REVISION_DIR / "R1c3_case_mix_weighted_vs_unweighted.csv",
                index=False)


# =========================================================
# 4. Population-level trade-off curve across thresholds (temporal)
# =========================================================
tradeoff_grid = np.round(np.arange(0.03, 0.4001, 0.005), 4)
tradeoff_rows = []
for thr in tradeoff_grid:
    pred = (p_ext_arr >= thr).astype(int)
    uw = operating_point_row(y_ext_arr, pred)
    wt = operating_point_row(y_ext_arr, pred, sample_weight=w_ext_arr)
    tradeoff_rows.append({
        "threshold": thr,
        "unweighted_sensitivity": uw["sensitivity"],
        "unweighted_specificity": uw["specificity"],
        "weighted_sensitivity": wt["sensitivity"],
        "weighted_specificity": wt["specificity"],
        "weighted_positive_rate": wt["positive_rate"],
    })
tradeoff = pd.DataFrame(tradeoff_rows)
tradeoff.to_csv(
    REVISION_DIR / "R3c3_threshold_tradeoff_temporal.csv", index=False
)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot(tradeoff["threshold"], tradeoff["weighted_sensitivity"],
        linewidth=2.0, label="Sensitivity (survey-weighted)")
ax.plot(tradeoff["threshold"], tradeoff["weighted_specificity"],
        linewidth=2.0, label="Specificity (survey-weighted)")
ax.plot(tradeoff["threshold"], tradeoff["unweighted_sensitivity"],
        linewidth=2.0, linestyle="--", label="Sensitivity (unweighted)")
ax.plot(tradeoff["threshold"], tradeoff["unweighted_specificity"],
        linewidth=2.0, linestyle="--", label="Specificity (unweighted)")
ax.axvline(SCREENING_THRESHOLD, color="black", linestyle=":", linewidth=1.5,
           label=f"Current threshold ({SCREENING_THRESHOLD:.3f})")
ax.axvline(WEIGHTED_SCREENING_THRESHOLD, color="red", linestyle=":",
           linewidth=1.5,
           label=f"Weighted threshold ({WEIGHTED_SCREENING_THRESHOLD:.3f})")
ax.axhline(TARGET_SENSITIVITY, color="grey", linewidth=1.0)
ax.set_ylim(0, 1.02)
style_axes(ax, "Classification threshold (predicted probability)",
           "Proportion", "Sensitivity–Specificity Trade-off (2021–2023)")
save_figure(fig, "threshold_tradeoff_weighted_temporal")
