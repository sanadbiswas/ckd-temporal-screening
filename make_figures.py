"""
Draw the manuscript figures from saved results (no model re-run needed).

The pipeline writes the numbers behind each figure to
    mlwa_results/figure_data/        (module 40)
and this script turns them into
    mlwa_results/figures/            as .eps, .pdf and .png,
using the file names that main.tex expects.

Usage
-----
    python make_figures.py            # all figures
    python make_figures.py 1 4        # only Figure 1 and Figure 4
    python make_figures.py S1         # only Supplementary Figure S1

In Colab, after the pipeline has run:
    %run make_figures.py 1
or copy one fig_*() function into a cell, edit it, and call it.

To change a figure, edit its function below; settings shared by every
figure are in the SETTINGS block. The pipeline (module 41) runs this same
file, so a change made here also applies to the next full run.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================== SETTINGS ==============================
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# Module 41 passes the two folders in; run on its own, the script uses the
# project's mlwa_results folder next to this file.
DATA = Path(globals().get("DATA_DIR_OVERRIDE", HERE / "mlwa_results" / "figure_data"))
OUT = Path(globals().get("OUT_DIR_OVERRIDE", HERE / "mlwa_results" / "figures"))
SHOW = bool(globals().get("SHOW_FIGURES", False))   # display each figure (notebook)

SHOW_TITLE = False     # journal guide: put the title in the caption, not on the figure
FIGSIZE = (8, 6.5)     # inches
DPI = 600              # resolution of the PNG
FONT = "DejaVu Sans"   # e.g. "Arial", "Times New Roman"
BASE_FONTSIZE = 12
LINEWIDTH = 2.0
MARKERSIZE = 6.5
GRID = True

# Threshold lines: read from the run when available, else the published values.
FROZEN_THRESHOLD = 0.123898      # primary threshold
WEIGHTED_THRESHOLD = 0.094830    # survey-weighted threshold
_constants_file = DATA / "figure_constants.json"
if _constants_file.exists():
    _constants = json.loads(_constants_file.read_text())
    FROZEN_THRESHOLD = _constants.get("frozen_threshold", FROZEN_THRESHOLD)
    WEIGHTED_THRESHOLD = _constants.get("weighted_threshold", WEIGHTED_THRESHOLD)

# One colour per curve; change here to restyle every figure consistently.
C = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c", "red": "#d62728",
    "purple": "#9467bd", "grey": "#7f7f7f", "black": "#000000",
}

plt.rcParams.update({
    "font.family": FONT,
    "font.size": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE + 2,
    "axes.titlesize": BASE_FONTSIZE + 4,
    "legend.fontsize": BASE_FONTSIZE - 1,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,   # embed fonts as text (journal requirement)
    "ps.fonttype": 42,
})


# ============================== HELPERS ===============================
def new_axes(figsize=FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    if GRID:
        ax.grid(True, linestyle="--", linewidth=0.6, color="#cccccc")   # no transparency: EPS cannot store it
        ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, name, title=None, legend_loc="best"):
    if SHOW_TITLE and title:
        ax.set_title(title, pad=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, frameon=True, edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"saved  {OUT / name}.[png|pdf|eps]")


# ============================== FIGURE 1 ==============================
def fig_1():
    """Internal calibration (2017-2018 out-of-fold): raw, sigmoid, isotonic."""
    d = pd.read_csv(DATA / "fig1_internal_calibration_points.csv")
    fig, ax = new_axes()
    for series, colour in [("Raw", C["blue"]), ("Sigmoid", C["orange"]), ("Isotonic", C["green"])]:
        s = d[d.series == series]
        ax.plot(s.mean_predicted, s.observed, marker="o", color=colour,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=series)
    ax.plot([0, 1], [0, 1], linestyle="--", color=C["red"], linewidth=LINEWIDTH, label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed outcome frequency")
    ax.set_xlim(-0.05, 1.05)      # try (0, 0.65) to zoom in on the data
    ax.set_ylim(-0.05, 1.05)
    finish(fig, ax, "internal_calibration_oof",
           "Internal Calibration — Out-of-Fold Predictions", legend_loc="upper left")


# ============================== FIGURE 2 ==============================
def fig_2():
    """Calibration transport: development OOF vs 2021-2023 temporal cohort."""
    d = pd.read_csv(DATA / "fig2_calibration_transport_points.csv")
    fig, ax = new_axes()
    for series, colour in [("Development OOF", C["blue"]), ("2021-2023 temporal", C["orange"])]:
        s = d[d.series == series]
        ax.plot(s.mean_predicted, s.observed, marker="o", color=colour,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=series.replace("-", "–"))
    ax.plot([0, 1], [0, 1], linestyle="--", color=C["green"], linewidth=LINEWIDTH, label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed outcome frequency")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    finish(fig, ax, "calibration_transport", "Calibration Transport", legend_loc="upper left")


# ============================== FIGURE 3 ==============================
def fig_3():
    """Sensitivity-specificity trade-off across thresholds, weighted and unweighted."""
    d = pd.read_csv(DATA / "fig3_fig5_threshold_tradeoff.csv")
    fig, ax = new_axes()
    ax.plot(d.threshold, d.weighted_sensitivity, color=C["blue"], linewidth=LINEWIDTH,
            label="Sensitivity (survey-weighted)")
    ax.plot(d.threshold, d.weighted_specificity, color=C["orange"], linewidth=LINEWIDTH,
            label="Specificity (survey-weighted)")
    ax.plot(d.threshold, d.unweighted_sensitivity, color=C["green"], linewidth=LINEWIDTH - 0.5,
            linestyle="--", label="Sensitivity (unweighted)")
    ax.plot(d.threshold, d.unweighted_specificity, color=C["red"], linewidth=LINEWIDTH - 0.5,
            linestyle="--", label="Specificity (unweighted)")
    ax.axvline(FROZEN_THRESHOLD, color=C["black"], linestyle=":", linewidth=1.5,
               label=f"Current threshold ({FROZEN_THRESHOLD:.3f})")
    ax.axvline(WEIGHTED_THRESHOLD, color=C["purple"], linestyle=":", linewidth=1.5,
               label=f"Weighted threshold ({WEIGHTED_THRESHOLD:.3f})")
    ax.axhline(0.80, color=C["grey"], linewidth=1.0)          # 80% sensitivity target
    ax.set_xlabel("Classification threshold (predicted probability)")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.03)
    finish(fig, ax, "threshold_tradeoff_weighted_temporal",
           "Sensitivity–Specificity Trade-off (2021–2023)", legend_loc="lower right")


# ============================== FIGURE 4 ==============================
def fig_4():
    """Decision curves: primary model, logistic regressions, risk-factor rules."""
    d = pd.read_csv(DATA / "fig4_decision_curves.csv")
    curves = [  # (column in the CSV, legend label, colour, line style)
        ("HistGradientBoosting (primary, 12 predictors)", "HistGradientBoosting (primary, 12 predictors)", C["blue"], "-"),
        ("Calibrated logistic regression (12 predictors)", "Calibrated logistic regression (12 predictors)", C["orange"], "-"),
        ("Calibrated logistic regression (age, diabetes, hypertension)", "Calibrated logistic regression (age, diabetes, hypertension)", C["green"], "-"),
        ("Rule: diabetes or hypertension", "Rule: diabetes or hypertension", C["red"], "--"),
        ("Rule: diabetes, hypertension, or age >=60", "Rule: diabetes, hypertension, or age ≥60", C["purple"], "--"),
        ("Test all", "Test all", C["grey"], ":"),
        ("Test none", "Test none", C["black"], ":"),
    ]
    fig, ax = new_axes()
    for column, label, colour, style in curves:
        ax.plot(d.threshold, d[column], color=colour, linestyle=style, linewidth=LINEWIDTH, label=label)
    ax.axvline(FROZEN_THRESHOLD, color=C["grey"], linestyle=":", linewidth=1.2)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.05, 0.16)      # values below -0.05 are cut off
    finish(fig, ax, "decision_curve_models_vs_rules_temporal",
           "Decision Curves — Models vs Risk-Factor Rules (2021–2023)", legend_loc="upper right")


# ============================== FIGURE 5 ==============================
def fig_5():
    """Testing burden vs cases found per 10,000 U.S. adults (survey-weighted)."""
    curve = pd.read_csv(DATA / "fig5_model_curve.csv")
    sc = pd.read_csv(DATA / "fig5_screening_scenarios.csv")
    sc = sc[sc.scale.str.startswith("U.S.")]
    x = "referred_for_confirmatory_testing_per_10000"
    y = "screening_positive_cases_found_per_10000"

    fig, ax = new_axes()
    ax.plot(curve.referred_per_10000, curve.cases_found_per_10000,
            color=C["blue"], linewidth=LINEWIDTH, label="Model (varying threshold)")
    points = [  # (start of the strategy name in the CSV, legend label, marker, colour)
        ("Rule: diabetes or hypertension", "Rule: diabetes or hypertension", "s", C["orange"]),
        ("Rule: diabetes, hypertension, or age", "Rule: diabetes, hypertension, or age ≥60", "^", C["green"]),
        ("Test all", "Test all", "D", C["red"]),
        ("Model, threshold 0.124", "Model at frozen threshold (0.124)", "o", C["black"]),
    ]
    for start, label, marker, colour in points:
        r = sc[sc.strategy.str.startswith(start)].iloc[0]
        ax.scatter(r[x], r[y], marker=marker, color=colour, s=70, zorder=3, label=label)
    ax.set_xlabel("Adults referred for confirmatory testing per 10,000")
    ax.set_ylabel("Screening-positive adults found per 10,000")
    finish(fig, ax, "testing_burden_vs_cases_found_temporal",
           "Testing Burden vs Cases Found (U.S. adults, 2021–2023)", legend_loc="lower right")


# ============================== FIGURE 6 ==============================
FEATURE_LABELS = {
    "RIDAGEYR": "Age", "has_diabetes_dx": "Diagnosed diabetes", "has_htn_dx": "Diagnosed hypertension",
    "mean_sbp": "Mean systolic BP", "INDFMPIR": "Income-to-poverty ratio", "BMXBMI": "BMI",
    "mean_dbp": "Mean diastolic BP", "DMDEDUC2": "Education", "RIAGENDR": "Sex",
    "BMXWAIST": "Waist circumference", "smoked_100_cigs": "Smoking history", "ALQ130": "Alcohol intake",
}


def fig_6():
    """Temporal permutation importance (mean decrease in ROC-AUC, with SD)."""
    d = pd.read_csv(DATA / "fig6_permutation_importance.csv").sort_values("importance_mean")
    labels = [FEATURE_LABELS.get(f, f) for f in d.feature]
    fig, ax = new_axes(figsize=(8, 7))
    ax.barh(labels, d.importance_mean, xerr=d.importance_sd, color=C["blue"],
            edgecolor="black", linewidth=0.8, capsize=3)
    ax.axvline(0, color="black", linewidth=1.0)
    for yy, (m, sd) in enumerate(zip(d.importance_mean, d.importance_sd)):   # value labels
        ax.text(max(m, 0) + sd + 0.002, yy, f"{m + 0.0:.4f}".replace("-0.0000", "0.0000"), va="center", fontsize=BASE_FONTSIZE - 2)
    ax.set_xlabel("Decrease in ROC-AUC after permutation")
    ax.set_ylabel("Predictor")
    ax.set_xlim(right=d.importance_mean.max() * 1.22)
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "temporal_permutation_importance", "Temporal Permutation Feature Importance")


# ====================== SUPPLEMENTARY FIGURE S1 =======================
def fig_S1():
    """Decision curves within adults with diagnosed diabetes (2021-2023)."""
    d = pd.read_csv(DATA / "figS1_dca_within_diabetes.csv")
    fig, ax = new_axes()
    ax.plot(d.threshold, d.model, color=C["blue"], linewidth=LINEWIDTH, label="Model (continuous risk)")
    ax.plot(d.threshold, d.treat_all, color=C["orange"], linestyle="--", linewidth=LINEWIDTH,
            label="Test all with diabetes")
    ax.plot(d.threshold, d.treat_none, color=C["green"], linestyle=":", linewidth=LINEWIDTH, label="Test none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    finish(fig, ax, "decision_curve_within_diabetes_temporal",
           "Decision Curve Within Diagnosed Diabetes (2021–2023)", legend_loc="upper right")


# ================================ RUN =================================
FIGURES = {"1": fig_1, "2": fig_2, "3": fig_3, "4": fig_4, "5": fig_5, "6": fig_6, "S1": fig_S1}

def make_all():
    """Draw every figure (used by pipeline module 41)."""
    for draw in FIGURES.values():
        draw()


if __name__ == "__main__":
    wanted = [a.upper().replace("FIG", "") for a in sys.argv[1:] if not a.startswith("-")] or list(FIGURES)
    for key in wanted:
        if key not in FIGURES:
            sys.exit(f"Unknown figure '{key}'. Choose from: {', '.join(FIGURES)}")
        FIGURES[key]()
