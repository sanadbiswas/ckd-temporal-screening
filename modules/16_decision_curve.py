# Modular pipeline stage.

# Source: final notebook cell 42
# ============================================================
# Decision-curve analysis — publication-ready
# Same analysis as original figure
# - no frozen-threshold line
# - moderately thicker borders and lines
# - non-bold labels/title
# - EPS + high-resolution PNG export
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def decision_curve(y, p, thresholds=None, sample_weight=None):
    y = np.asarray(y)
    p = np.asarray(p)

    if thresholds is None:
        thresholds = np.linspace(0.03, 0.40, 75)

    w = (
        np.ones(len(y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    total_w = w.sum()
    prevalence = np.average(y, weights=w)

    rows = []

    for pt in thresholds:
        pred = p >= pt

        tp_w = w[(pred == 1) & (y == 1)].sum()
        fp_w = w[(pred == 1) & (y == 0)].sum()

        nb_model = (
            tp_w / total_w
            - fp_w / total_w * pt / (1 - pt)
        )

        nb_all = (
            prevalence
            - (1 - prevalence) * pt / (1 - pt)
        )

        rows.append({
            "threshold": pt,
            "model": nb_model,
            "treat_all": nb_all,
            "treat_none": 0.0
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Calculate decision curves
# ------------------------------------------------------------

dca_dev = decision_curve(
    y_dev,
    selected_internal_prob
)

dca_ext = decision_curve(
    y_ext,
    p_ext
)

dca_ext_weighted = decision_curve(
    y_ext,
    p_ext,
    sample_weight=w_ext
)


# ------------------------------------------------------------
# Save numerical results
# ------------------------------------------------------------

dca_dev.to_csv(
    RESULTS_DIR / "dca_development_oof.csv",
    index=False
)

dca_ext.to_csv(
    RESULTS_DIR / "dca_temporal.csv",
    index=False
)

dca_ext_weighted.to_csv(
    RESULTS_DIR / "dca_temporal_weighted.csv",
    index=False
)


# ------------------------------------------------------------
# Publication-ready temporal decision curve
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6.5))

ax.plot(
    dca_ext["threshold"],
    dca_ext["model"],
    linewidth=2.0,
    label="Model"
)

ax.plot(
    dca_ext["threshold"],
    dca_ext["treat_all"],
    linestyle="--",
    linewidth=2.0,
    label="Treat all"
)

ax.plot(
    dca_ext["threshold"],
    dca_ext["treat_none"],
    linestyle=":",
    linewidth=2.0,
    label="Treat none"
)


# ------------------------------------------------------------
# Axis labels and title
# ------------------------------------------------------------

ax.set_xlabel(
    "Threshold probability",
    fontsize=13
)

ax.set_ylabel(
    "Net benefit",
    fontsize=13
)

ax.set_title(
    "Decision Curve — 2021–2023 Temporal Validation",
    fontsize=15,
    pad=12
)


# ------------------------------------------------------------
# Border styling
# ------------------------------------------------------------

for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color("black")


# ------------------------------------------------------------
# Tick styling
# ------------------------------------------------------------

ax.tick_params(
    axis="both",
    which="major",
    labelsize=11,
    width=1.4,
    length=5
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.45
)

ax.set_axisbelow(True)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend = ax.legend(
    fontsize=10.5,
    frameon=True,
    loc="best"
)

legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_edgecolor("black")


# ------------------------------------------------------------
# Layout and save
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "decision_curve_temporal.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    RESULTS_DIR / "decision_curve_temporal.eps",
    format="eps",
    bbox_inches="tight"
)

plt.show()

