# Modular pipeline stage — REVISION ROUND 1 (MLWA-D-26-01812).
#
# Shared helpers for modules 26-39. Nothing in this module changes the
# primary analysis (modules 00-25): the frozen model, the frozen threshold,
# and every result reported in the original submission are untouched.
# The revision modules ADD analyses requested by the reviewers and are all
# development-only in the same sense as the primary pipeline: any model,
# threshold, or policy is chosen on 2017-2018 data and then applied
# unchanged to the 2021-2023 temporal cohort.

REVISION_DIR = RESULTS_DIR / "revision_round1"
REVISION_DIR.mkdir(parents=True, exist_ok=True)

print("Revision outputs directory:", REVISION_DIR)


# =========================================================
# 1. Survey-design objects for both cohorts
# =========================================================
# Development: WTMEC2YR with SDMVSTRA / SDMVPSU (merged in module 02).
# Temporal:    WTPH2YR  with SDMVSTRA / SDMVPSU (merged in module 13).
dev_strata = survey_dev["SDMVSTRA"].to_numpy()
dev_psu = survey_dev["SDMVPSU"].to_numpy()
w_dev = survey_dev["WTMEC2YR"].to_numpy(dtype=float)

ext_strata = ext["SDMVSTRA"].to_numpy()
ext_psu = ext["SDMVPSU"].to_numpy()
w_ext_arr = np.asarray(w_ext, dtype=float)

y_dev_arr = y_dev.to_numpy(dtype=int)
y_ext_arr = y_ext.to_numpy(dtype=int)

print(
    "Development design: strata =", len(np.unique(dev_strata)),
    "| PSUs =", len(np.unique(list(zip(dev_strata, dev_psu))))
)
print(
    "Temporal design:    strata =", len(np.unique(ext_strata)),
    "| PSUs =", len(np.unique(list(zip(ext_strata, ext_psu))))
)


# =========================================================
# 2. Survey-design (Rao-Wu rescaling) bootstrap
# =========================================================
# Reviewer 3, comment 2: the original bootstrap resampled individual
# participants, ignoring NHANES stratification and clustering.
# The Rao-Wu rescaling bootstrap resamples PSUs WITHIN strata
# (m_h = n_h - 1 PSUs drawn with replacement in each stratum) and
# rescales the base weight of each participant by
#       (n_h / m_h) * (number of times the participant's PSU was drawn).
# Every replicate therefore preserves the stratified multistage design,
# and metrics are recomputed with the replicate weights.

def design_bootstrap_weights(base_w, strata, psu, n_boot, seed):
    """Yield n_boot Rao-Wu replicate weight vectors."""
    base_w = np.asarray(base_w, dtype=float)
    strata = np.asarray(strata)
    psu = np.asarray(psu)
    rng = np.random.default_rng(seed)

    stratum_info = []
    for h in np.unique(strata):
        in_h = strata == h
        psus_h = np.unique(psu[in_h])
        stratum_info.append((in_h, psus_h))

    for _ in range(n_boot):
        mult = np.ones(len(base_w), dtype=float)
        for in_h, psus_h in stratum_info:
            n_h = len(psus_h)
            if n_h < 2:
                continue
            m_h = n_h - 1
            drawn = rng.choice(psus_h, size=m_h, replace=True)
            counts = {k: 0 for k in psus_h}
            for d in drawn:
                counts[d] += 1
            for k in psus_h:
                mult[in_h & (psu == k)] = (n_h / m_h) * counts[k]
        yield base_w * mult


def iid_bootstrap_indices(n, n_boot, seed):
    """Yield participant-level bootstrap index vectors (original method)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, size=n)


def percentile_ci(values, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(v, alpha / 2)),
        float(np.quantile(v, 1 - alpha / 2))
    )


# =========================================================
# 3. Binomial confidence intervals for subgroup sensitivity/specificity
# =========================================================
# Reviewers 2 (comment 8) and 3 (minor 5): bootstrap percentile intervals
# collapse to zero width when a proportion is exactly 0 or 1 (e.g.
# specificity = 0.000 among adults with diagnosed diabetes). Wilson score
# intervals remain informative at the boundaries and in small strata.

from scipy.stats import norm as _norm

def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    z = _norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def sens_spec_counts(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


# =========================================================
# 4. Survey-weighted threshold selection
# =========================================================
# Same rule as threshold_for_target_sensitivity() in module 05 (highest
# threshold whose sensitivity reaches the target, i.e. the best specificity
# among thresholds meeting the target), but sensitivity is computed with
# survey weights, so the target refers to the U.S. adult population rather
# than to the NHANES sample.

def weighted_threshold_for_target_sensitivity(y_true, prob, sample_weight,
                                              target=0.80):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    w = np.asarray(sample_weight, dtype=float)

    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    tp_cum = np.cumsum((w * y)[order])
    total_pos = (w * y).sum()
    if total_pos <= 0:
        return 0.0

    # Evaluate only at the last index of each tied block of probabilities.
    last_of_block = np.r_[np.diff(p_sorted) != 0, True]
    sens_at_block = tp_cum[last_of_block] / total_pos
    thr_at_block = p_sorted[last_of_block]

    eligible = np.where(sens_at_block >= target)[0]
    if len(eligible) == 0:
        return 0.0
    return float(thr_at_block[eligible[0]])


# =========================================================
# 5. Operating-point summary used across revision modules
# =========================================================
def operating_point_row(y, pred, sample_weight=None):
    """Sensitivity, specificity, PPV, NPV and positive rate for a binary rule."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    w = (
        np.ones(len(y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    tp = w[(pred == 1) & (y == 1)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    tn = w[(pred == 0) & (y == 0)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    total = w.sum()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "positive_rate": (tp + fp) / total if total > 0 else np.nan,
    }


# =========================================================
# 6. Net benefit for binary screening policies
# =========================================================
# decision_curve() in module 16 handles probability models. A fixed policy
# (a rule, or a model with frozen subgroup thresholds) produces one binary
# classification, whose net benefit still depends on the threshold
# probability pt through the false-positive weight pt / (1 - pt).

def net_benefit_binary(y, pred, pt, sample_weight=None):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    w = (
        np.ones(len(y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    total = w.sum()
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    return tp / total - fp / total * pt / (1 - pt)


def policy_decision_curve(y, pred, thresholds=None, sample_weight=None):
    thresholds = DCA_THRESHOLDS if thresholds is None else thresholds
    return pd.DataFrame({
        "threshold": thresholds,
        "net_benefit": [
            net_benefit_binary(y, pred, pt, sample_weight) for pt in thresholds
        ]
    })


# =========================================================
# 7. Fit a variant model with EXACTLY the primary procedure
# =========================================================
# Used by the comparator, ablation, alcohol and blood-pressure analyses so
# that every alternative is tuned, calibrated, thresholded and transported
# in the same way as the primary model (modules 09-12):
#   (1) RandomizedSearchCV on the full development cohort (ROC-AUC),
#   (2) the selected calibration method (BEST_CALIBRATION),
#   (3) out-of-fold predictions on the SAME calibration folds as module 10,
#   (4) threshold for 80% development sensitivity on those OOF predictions,
#   (5) refit on all development data and apply unchanged to 2021-2023.

def fit_frozen_variant(
    label,
    X_dev_variant,
    X_ext_variant,
    model_template,
    param_distributions,
    seed_offset,
    calibration=None,
    preprocessor=None,
    n_iter=None,
):
    calibration = BEST_CALIBRATION if calibration is None else calibration
    n_iter = N_ITER if n_iter is None else n_iter

    pipe = Pipeline([
        ("prep", make_preprocessor(X_dev_variant) if preprocessor is None
         else preprocessor),
        ("model", clone(model_template))
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + seed_offset
        ),
        random_state=RANDOM_STATE + seed_offset,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_dev_variant, y_dev)
    tuned = search.best_estimator_

    if calibration == "Sigmoid":
        estimator = CalibratedClassifierCV(
            estimator=clone(tuned), method="sigmoid", cv=5, n_jobs=-1
        )
    elif calibration == "Isotonic":
        estimator = CalibratedClassifierCV(
            estimator=clone(tuned), method="isotonic", cv=5, n_jobs=-1
        )
    else:
        estimator = clone(tuned)

    oof = cross_val_predict(
        clone(estimator),
        X_dev_variant,
        y_dev,
        cv=cv_for_calibration,          # same folds as module 10
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    threshold = threshold_for_target_sensitivity(
        y_dev, oof, target=TARGET_SENSITIVITY
    )

    final = clone(estimator).fit(X_dev_variant, y_dev)
    p_ext_variant = final.predict_proba(X_ext_variant)[:, 1]

    print(
        f"[{label}] dev OOF AUC={roc_auc_score(y_dev, oof):.4f} | "
        f"temporal AUC={roc_auc_score(y_ext, p_ext_variant):.4f} | "
        f"frozen threshold={threshold:.4f}"
    )

    return {
        "label": label,
        "oof": oof,
        "p_ext": p_ext_variant,
        "threshold": threshold,
        "best_params": search.best_params_,
        "final_model": final,
    }


def summarize_variant(result, extra=None):
    """One-row summary: development OOF and temporal metrics at the variant's own frozen threshold."""
    thr = result["threshold"]
    dev_m = metrics_from_prob(y_dev, result["oof"], threshold=thr)
    ext_m = metrics_from_prob(y_ext, result["p_ext"], threshold=thr)
    ext_w = metrics_from_prob(
        y_ext, result["p_ext"], threshold=thr, sample_weight=w_ext_arr
    )
    ext_intercept, ext_slope = calibration_intercept_slope(y_ext, result["p_ext"])
    row = {
        "model": result["label"],
        "frozen_threshold": thr,
        "dev_roc_auc": dev_m["roc_auc"],
        "dev_brier": dev_m["brier"],
        "temporal_roc_auc": ext_m["roc_auc"],
        "temporal_pr_auc": ext_m["pr_auc"],
        "temporal_brier": ext_m["brier"],
        "temporal_cal_intercept": ext_intercept,
        "temporal_cal_slope": ext_slope,
        "temporal_ece": weighted_ece(y_ext, result["p_ext"], 10),
        "temporal_sensitivity": ext_m["sensitivity"],
        "temporal_specificity": ext_m["specificity"],
        "temporal_weighted_sensitivity": ext_w["sensitivity"],
        "temporal_weighted_specificity": ext_w["specificity"],
    }
    if extra:
        row.update(extra)
    return row


# =========================================================
# 8. Shared figure styling (matches the existing publication figures)
# =========================================================
def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", labelsize=11, width=1.4, length=5)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=10.5, frameon=True, loc="best")
    if legend is not None:
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_edgecolor("black")


def save_figure(fig, stem):
    plt.tight_layout()
    fig.savefig(REVISION_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(REVISION_DIR / f"{stem}.eps", format="eps", bbox_inches="tight")
    plt.show()


# Readable predictor names (same mapping as module 24).
FEATURE_LABELS = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "DMDEDUC2": "Education",
    "INDFMPIR": "Income-to-poverty ratio",
    "has_diabetes_dx": "Diagnosed diabetes",
    "has_htn_dx": "Diagnosed hypertension",
    "BMXBMI": "BMI",
    "BMXWAIST": "Waist circumference",
    "smoked_100_cigs": "Smoking history",
    "ALQ130": "Alcohol intake",
    "mean_sbp": "Mean systolic BP",
    "mean_dbp": "Mean diastolic BP",
}

# Every revision table is collected here and summarized by module 39.
REVISION_TABLES = {}

print("Revision helpers loaded.")
