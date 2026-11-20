# Modular pipeline stage.

# Source: final notebook cell 38
ext_intercept, ext_slope = calibration_intercept_slope(y_ext, p_ext)

temporal_calibration = {
    "brier": brier_score_loss(y_ext, p_ext),
    "calibration_intercept": ext_intercept,
    "calibration_slope": ext_slope,
    "ece": weighted_ece(y_ext, p_ext, 10)
}

temporal_calibration_weighted = {
    "brier": brier_score_loss(
        y_ext, p_ext, sample_weight=w_ext
    ),
    "ece": weighted_ece(
        y_ext, p_ext, 10, sample_weight=w_ext
    )
}

print("Temporal calibration")
print(temporal_calibration)
print("Survey-weighted calibration sensitivity")
print(temporal_calibration_weighted)

# Persist both dicts to CSV. Previously these were only printed to stdout,
# which meant the exact overall temporal calibration numbers (as opposed
# to the per-subgroup breakdown in temporal_subgroup_calibration.csv) were
# not recoverable from the results folder after the run ended.
pd.DataFrame([
    {"analysis": "temporal_unweighted", **temporal_calibration},
    {"analysis": "temporal_survey_weighted", **temporal_calibration_weighted},
]).to_csv(RESULTS_DIR / "temporal_calibration_summary.csv", index=False)


# Source: final notebook cell 39
# ============================================================
# Calibration transport plot — balanced publication style
# - moderately thicker borders
# - moderately thicker lines
# - clear but not oversized text
# - EPS + high-resolution PNG export
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6.5))

# Cleaner display names
label_map = {
    "Development OOF": "Development OOF",
    "2021–2023 temporal": "2021–2023 temporal"
}

# Balanced line/marker thickness
style_map = {
    "Development OOF": {
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 6.5
    },
    "2021–2023 temporal": {
        "marker": "o",
        "linewidth": 2.0,
        "markersize": 6.5
    }
}

for label, yy, pp in [
    ("Development OOF", y_dev.to_numpy(), selected_internal_prob),
    ("2021–2023 temporal", y_ext.to_numpy(), p_ext)
]:

    bins = pd.qcut(
        pd.Series(pp),
        q=10,
        duplicates="drop"
    )

    temp = pd.DataFrame({
        "p": pp,
        "y": yy,
        "bin": bins
    })

    grouped = temp.groupby(
        "bin",
        observed=True
    ).agg(
        mean_pred=("p", "mean"),
        observed=("y", "mean")
    )

    style = style_map.get(
        label,
        {
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 6.5
        }
    )

    ax.plot(
        grouped["mean_pred"],
        grouped["observed"],
        marker=style["marker"],
        linewidth=style["linewidth"],
        markersize=style["markersize"],
        label=label_map.get(label, label)
    )


# ------------------------------------------------------------
# Ideal calibration line
# ------------------------------------------------------------

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2.0,
    color="green",
    label="Ideal"
)


# ------------------------------------------------------------
# Axis labels and title
# ------------------------------------------------------------

ax.set_xlabel(
    "Predicted probability",
    fontsize=13,
    #fontweight="bold"
)

ax.set_ylabel(
    "Observed outcome frequency",
    fontsize=13,
    #fontweight="bold"
)

ax.set_title(
    "Calibration Transport",
    fontsize=15,
    #fontweight="bold",
    pad=12
)


# ------------------------------------------------------------
# Thicker figure border
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
    RESULTS_DIR / "calibration_transport.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    RESULTS_DIR / "calibration_transport.eps",
    format="eps",
    bbox_inches="tight"
)

plt.show()

