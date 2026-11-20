# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 3, comment 5  — expected confirmatory tests, missed screening-
#       positive cases and unnecessary referrals under different thresholds.
#   Reviewer 2, comment 10 — justify the 0.03–0.40 decision-curve range and
#       relate the frozen threshold to a willingness to order confirmatory
#       kidney testing.
#   Reviewer 4, comment 1  — testing burden and additional cases identified.

# =========================================================
# 1. What a threshold probability means for confirmatory testing
# =========================================================
# In decision-curve analysis, choosing threshold probability pt means one
# true positive is valued as much as (1 - pt) / pt false positives.
# Equivalently, testing is worthwhile when no more than 1 / pt confirmatory
# tests are needed per screening-positive adult found.
pt_grid = sorted(set([0.03, 0.05, 0.08, 0.10, round(SCREENING_THRESHOLD, 4),
                      0.15, 0.20, 0.30, 0.40]))
dca_interpretation = pd.DataFrame({
    "threshold_probability": pt_grid,
    "false_positive_tests_accepted_per_case_found": [(1 - p) / p for p in pt_grid],
    "max_total_tests_per_case_found": [1 / p for p in pt_grid],
})
dca_interpretation["is_frozen_threshold"] = np.isclose(
    dca_interpretation["threshold_probability"], round(SCREENING_THRESHOLD, 4)
)
display(dca_interpretation.round(2))
dca_interpretation.to_csv(
    REVISION_DIR / "R2c10_dca_threshold_interpretation.csv", index=False
)
REVISION_TABLES["dca_interpretation"] = dca_interpretation


# =========================================================
# 2. Scenario table per 10,000 adults screened (temporal cohort)
# =========================================================
SCENARIO_THRESHOLDS = sorted(set(
    [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    + [SCREENING_THRESHOLD, WEIGHTED_SCREENING_THRESHOLD]
))

def scenario_row(label, pred, weights, scale_name):
    y = y_ext_arr
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    tot = w.sum()
    per = 10000 / tot
    tp = w[(pred == 1) & (y == 1)].sum() * per
    fp = w[(pred == 1) & (y == 0)].sum() * per
    fn = w[(pred == 0) & (y == 1)].sum() * per
    referred = tp + fp
    return {
        "strategy": label,
        "scale": scale_name,
        "referred_for_confirmatory_testing_per_10000": referred,
        "screening_positive_cases_found_per_10000": tp,
        "screening_positive_cases_missed_per_10000": fn,
        "unnecessary_referrals_per_10000": fp,
        "tests_per_case_found": referred / tp if tp > 0 else np.nan,
        "share_of_cases_found": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    }

scenario_rows = []
for weights, scale_name in [(None, "NHANES sample"),
                            (w_ext_arr, "U.S. adults (WTPH2YR-weighted)")]:
    for thr in SCENARIO_THRESHOLDS:
        tag = ""
        if np.isclose(thr, SCREENING_THRESHOLD):
            tag = " [frozen primary threshold]"
        elif np.isclose(thr, WEIGHTED_SCREENING_THRESHOLD):
            tag = " [survey-weighted threshold]"
        pred = (p_ext_arr >= thr).astype(int)
        scenario_rows.append(
            scenario_row(f"Model, threshold {thr:.3f}{tag}", pred, weights,
                         scale_name)
        )
    for rule_name in ["Rule: diabetes or hypertension",
                      "Rule: diabetes, hypertension, or age >=60", "Test all"]:
        scenario_rows.append(
            scenario_row(rule_name, rule_flags_ext[rule_name], weights,
                         scale_name)
        )

screening_scenarios = pd.DataFrame(scenario_rows)
display(screening_scenarios.round(1))
screening_scenarios.to_csv(
    REVISION_DIR / "R3c5_screening_scenarios_per_10000.csv", index=False
)
REVISION_TABLES["screening_scenarios"] = screening_scenarios


# =========================================================
# 3. Testing-efficiency figure: cases found vs referrals
# =========================================================
curve_thr = np.round(np.arange(0.02, 0.6001, 0.005), 4)
curve = []
for thr in curve_thr:
    pred = (p_ext_arr >= thr).astype(int)
    r = scenario_row("", pred, w_ext_arr, "")
    curve.append((r["referred_for_confirmatory_testing_per_10000"],
                  r["screening_positive_cases_found_per_10000"]))
curve = np.array(curve)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot(curve[:, 0], curve[:, 1], linewidth=2.0,
        label="Model (varying threshold)")
markers = {"Rule: diabetes or hypertension": "s",
           "Rule: diabetes, hypertension, or age >=60": "^",
           "Test all": "D"}
for rule_name, mk in markers.items():
    r = scenario_row(rule_name, rule_flags_ext[rule_name], w_ext_arr, "")
    ax.plot(r["referred_for_confirmatory_testing_per_10000"],
            r["screening_positive_cases_found_per_10000"],
            marker=mk, markersize=9, linestyle="none", label=rule_name)
r = scenario_row("", (p_ext_arr >= SCREENING_THRESHOLD).astype(int),
                 w_ext_arr, "")
ax.plot(r["referred_for_confirmatory_testing_per_10000"],
        r["screening_positive_cases_found_per_10000"],
        marker="o", markersize=9, color="black", linestyle="none",
        label=f"Model at frozen threshold ({SCREENING_THRESHOLD:.3f})")
style_axes(ax, "Adults referred for confirmatory testing per 10,000",
           "Screening-positive adults found per 10,000",
           "Testing Burden vs Cases Found (U.S. adults, 2021–2023)")
save_figure(fig, "testing_burden_vs_cases_found_temporal")
