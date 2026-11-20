# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 3, comment 4 — confidence intervals for calibration intercept
#       and slope; ECE depends on binning; calibration around the selected
#       screening threshold, where errors change classification decisions.

from scipy.optimize import brentq
from scipy.special import expit, logit

EPS = 1e-6

def calibration_in_the_large(y, p):
    """Intercept a with the slope fixed at 1: logit(P(y=1)) = a + logit(p)."""
    y = np.asarray(y, dtype=float)
    lp = logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))
    f = lambda a: np.sum(y - expit(a + lp))
    return float(brentq(f, -10, 10))

def ece_uniform_bins(y, p, n_bins):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in np.unique(ids):
        m = ids == b
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


# =========================================================
# 1. Calibration intercept, slope, CITL and O/E with bootstrap CIs
# =========================================================
def calibration_summary(y, p):
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_in_the_large": calibration_in_the_large(y, p),
        "observed_expected_ratio": float(np.mean(y) / np.mean(p)),
        "ece_quantile_10": weighted_ece(y, p, 10),
    }

calibration_ci_rows = []
for cohort, (yy, pp, ww, st, ps) in cohort_inputs.items():
    point = calibration_summary(yy, pp)
    boot = defaultdict(list)
    for idx in iid_bootstrap_indices(len(yy), N_BOOT, RANDOM_STATE + 3200):
        if np.unique(yy[idx]).size < 2:
            continue
        for k, v in calibration_summary(yy[idx], pp[idx]).items():
            boot[k].append(v)
    for k, v in point.items():
        lo, hi = percentile_ci(boot[k])
        calibration_ci_rows.append({
            "cohort": cohort, "metric": k, "estimate": v,
            "ci_low_95": lo, "ci_high_95": hi,
            "ideal_value": {"calibration_intercept": 0.0,
                            "calibration_slope": 1.0,
                            "calibration_in_the_large": 0.0,
                            "observed_expected_ratio": 1.0,
                            "ece_quantile_10": 0.0}[k],
        })

calibration_with_ci = pd.DataFrame(calibration_ci_rows)
calibration_with_ci["ci_method"] = f"participant-level bootstrap, {N_BOOT} replicates"
calibration_with_ci["ci_contains_ideal"] = (
    (calibration_with_ci["ci_low_95"] <= calibration_with_ci["ideal_value"])
    & (calibration_with_ci["ideal_value"] <= calibration_with_ci["ci_high_95"])
)
display(calibration_with_ci.round(4))
calibration_with_ci.to_csv(
    REVISION_DIR / "R3c4_calibration_with_CI.csv", index=False
)
REVISION_TABLES["calibration_ci"] = calibration_with_ci


# =========================================================
# 2. ECE sensitivity to the number and type of bins
# =========================================================
ece_rows = []
for cohort, (yy, pp, ww, _, _) in cohort_inputs.items():
    for n_bins in [5, 10, 15, 20]:
        ece_rows.append({
            "cohort": cohort,
            "n_bins": n_bins,
            "ece_equal_frequency_bins": weighted_ece(yy, pp, n_bins),
            "ece_equal_width_bins": ece_uniform_bins(yy, pp, n_bins),
        })
ece_sensitivity = pd.DataFrame(ece_rows)
display(ece_sensitivity.round(4))
ece_sensitivity.to_csv(
    REVISION_DIR / "R3c4_ece_binning_sensitivity.csv", index=False
)


# =========================================================
# 3. Calibration in predicted-risk bands around the threshold
# =========================================================
RISK_BANDS = [0.0, 0.06, 0.10, 0.15, 0.20, 0.30, 1.0]
band_rows = []
for cohort, (yy, pp, ww, _, _) in cohort_inputs.items():
    bands = pd.cut(pp, bins=RISK_BANDS, right=False, include_lowest=True)
    for band in bands.categories:
        m = np.asarray(bands == band)
        n = int(m.sum())
        if n == 0:
            continue
        events = int(yy[m].sum())
        lo, hi = wilson_ci(events, n)
        band_rows.append({
            "cohort": cohort,
            "predicted_risk_band": str(band),
            "contains_threshold": band.left <= SCREENING_THRESHOLD < band.right,
            "n": n,
            "mean_predicted": pp[m].mean(),
            "observed_rate": events / n,
            "observed_ci_low": lo,
            "observed_ci_high": hi,
            "observed_minus_predicted": events / n - pp[m].mean(),
        })

    # Narrow window: +/- 25% around the frozen threshold
    lo_t, hi_t = 0.75 * SCREENING_THRESHOLD, 1.25 * SCREENING_THRESHOLD
    m = (pp >= lo_t) & (pp < hi_t)
    n = int(m.sum())
    if n:
        events = int(yy[m].sum())
        lo, hi = wilson_ci(events, n)
        band_rows.append({
            "cohort": cohort,
            "predicted_risk_band": f"[{lo_t:.3f}, {hi_t:.3f}) threshold window",
            "contains_threshold": True,
            "n": n,
            "mean_predicted": pp[m].mean(),
            "observed_rate": events / n,
            "observed_ci_low": lo,
            "observed_ci_high": hi,
            "observed_minus_predicted": events / n - pp[m].mean(),
        })

threshold_band_calibration = pd.DataFrame(band_rows)
display(threshold_band_calibration.round(4))
threshold_band_calibration.to_csv(
    REVISION_DIR / "R3c4_calibration_around_threshold.csv", index=False
)
REVISION_TABLES["threshold_band_calibration"] = threshold_band_calibration


# =========================================================
# 4. Do the observed calibration errors change screening decisions?
# =========================================================
# Re-calibrate temporal predictions in the large (and with intercept+slope),
# then count how many participants would cross the frozen threshold.
ext_citl = calibration_in_the_large(y_ext_arr, p_ext_arr)
ext_int, ext_slp = calibration_intercept_slope(y_ext_arr, p_ext_arr)
lp_ext = logit(np.clip(p_ext_arr, EPS, 1 - EPS))

recalibrated = {
    "Original frozen predictions": p_ext_arr,
    "Recalibrated in the large (intercept only)": expit(ext_citl + lp_ext),
    "Recalibrated (intercept and slope)": expit(ext_int + ext_slp * lp_ext),
}
base_pred = (p_ext_arr >= SCREENING_THRESHOLD).astype(int)

decision_rows = []
for name, pr in recalibrated.items():
    pred = (pr >= SCREENING_THRESHOLD).astype(int)
    op = operating_point_row(y_ext_arr, pred)
    decision_rows.append({
        "predictions": name,
        "n_reclassified_vs_original": int((pred != base_pred).sum()),
        "pct_reclassified_vs_original": 100 * (pred != base_pred).mean(),
        "sensitivity": op["sensitivity"],
        "specificity": op["specificity"],
        "positive_rate": op["positive_rate"],
    })
decision_impact = pd.DataFrame(decision_rows)
display(decision_impact.round(4))
decision_impact.to_csv(
    REVISION_DIR / "R3c4_calibration_decision_impact.csv", index=False
)
REVISION_TABLES["calibration_decision_impact"] = decision_impact
