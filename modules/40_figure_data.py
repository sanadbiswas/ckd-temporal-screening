# Modular pipeline stage — REVISION ROUND 1.
#
# Saves the numbers behind every manuscript figure to
#   mlwa_results/figure_data/
# so that the figures can be redrawn (module 41 or make_figures.py)
# without re-running any model. Nothing here changes a result: the
# stage only reads objects already in memory and CSVs already written.

import json
import shutil

FIGURE_DATA_DIR = RESULTS_DIR / "figure_data"
FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def calibration_points(y, p, n_bins=10):
    """Mean predicted probability and observed rate in quantile bins
    (the same binning used by modules 10 and 15)."""
    temp = pd.DataFrame({
        "p": np.asarray(p, dtype=float),
        "y": np.asarray(y, dtype=float),
    })
    temp["bin"] = pd.qcut(temp["p"], q=n_bins, duplicates="drop")
    grouped = temp.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"),
        observed=("y", "mean"),
        n=("y", "size"),
    ).reset_index(drop=True)
    grouped.insert(0, "bin", np.arange(1, len(grouped) + 1))
    return grouped


# ---------------------------------------------------------
# Figure 1. Internal calibration: raw, sigmoid, isotonic
# ---------------------------------------------------------
fig1_rows = []
for name, prob in calibration_candidates.items():
    pts = calibration_points(y_dev.to_numpy(), prob)
    pts.insert(0, "series", name)
    fig1_rows.append(pts)
fig1_points = pd.concat(fig1_rows, ignore_index=True)
fig1_points.to_csv(
    FIGURE_DATA_DIR / "fig1_internal_calibration_points.csv", index=False
)

# ---------------------------------------------------------
# Figure 2. Calibration transport: development OOF vs temporal
# ---------------------------------------------------------
fig2_rows = []
for name, yy, pp in [
    ("Development OOF", y_dev.to_numpy(), selected_internal_prob),
    ("2021-2023 temporal", y_ext.to_numpy(), p_ext),
]:
    pts = calibration_points(yy, pp)
    pts.insert(0, "series", name)
    fig2_rows.append(pts)
fig2_points = pd.concat(fig2_rows, ignore_index=True)
fig2_points.to_csv(
    FIGURE_DATA_DIR / "fig2_calibration_transport_points.csv", index=False
)

# ---------------------------------------------------------
# Figure 5. Model curve: referrals vs cases found per 10,000
# U.S. adults (WTPH2YR-weighted), same grid as module 36.
# ---------------------------------------------------------
fig5_rows = []
for thr in np.round(np.arange(0.02, 0.6001, 0.005), 4):
    r = scenario_row("", (p_ext_arr >= thr).astype(int), w_ext_arr, "")
    fig5_rows.append({
        "threshold": thr,
        "referred_per_10000": r["referred_for_confirmatory_testing_per_10000"],
        "cases_found_per_10000": r["screening_positive_cases_found_per_10000"],
    })
fig5_curve = pd.DataFrame(fig5_rows)
fig5_curve.to_csv(FIGURE_DATA_DIR / "fig5_model_curve.csv", index=False)

# ---------------------------------------------------------
# Figures 3, 4, 5 (markers), 6 and S1: copies of result CSVs
# already written by earlier stages, under figure-specific names.
# ---------------------------------------------------------
figure_csv_sources = {
    "fig3_fig5_threshold_tradeoff.csv":
        REVISION_DIR / "R3c3_threshold_tradeoff_temporal.csv",
    "fig4_decision_curves.csv":
        REVISION_DIR / "R4c1_dca_models_vs_rules_temporal.csv",
    "fig5_screening_scenarios.csv":
        REVISION_DIR / "R3c5_screening_scenarios_per_10000.csv",
    "fig6_permutation_importance.csv":
        RESULTS_DIR / "temporal_permutation_importance.csv",
    "figS1_dca_within_diabetes.csv":
        REVISION_DIR / "R4c2_dca_within_diabetes.csv",
}
for target_name, source_path in figure_csv_sources.items():
    shutil.copyfile(source_path, FIGURE_DATA_DIR / target_name)

# ---------------------------------------------------------
# Constants drawn on the figures (threshold lines)
# ---------------------------------------------------------
figure_constants = {
    "frozen_threshold": float(SCREENING_THRESHOLD),
    "weighted_threshold": float(WEIGHTED_SCREENING_THRESHOLD),
}
with open(FIGURE_DATA_DIR / "figure_constants.json", "w") as f:
    json.dump(figure_constants, f, indent=2)

print("Figure data written to:", FIGURE_DATA_DIR)
for path in sorted(FIGURE_DATA_DIR.iterdir()):
    print("  ", path.name)
