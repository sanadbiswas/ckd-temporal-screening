# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 2, comment 3 — sensitivity analysis excluding alcohol intake
#       (~35.6% missing; negligible temporal permutation importance).
#   Reviewer 3, minor 4   — expanded treatment of missing data, especially
#       alcohol intake.
#   Reviewer 4, minor     — exclude the predictor or explicitly account for
#       its missingness (partly questionnaire skip patterns).
#
# ALQ130 (average drinks per drinking day, past 12 months) is only asked of
# participants who report drinking in the past year. Participants who never
# drank (ALQ111 = 2) or did not drink in the past 12 months (ALQ121 = 0)
# are skipped, so much of the missingness is structural: their true
# intake is zero, but median imputation assigns them a typical drinker's
# value. Four specifications are compared, each fitted with the primary
# procedure (module 26, fit_frozen_variant):
#   V0  primary: median imputation (as published)
#   V1  alcohol removed (11 predictors)
#   V2  median imputation + explicit missingness indicator
#   V3  skip-pattern recode (structural non-drinkers -> 0), then median
#       imputation for the remaining item non-response

# =========================================================
# 1. Missingness profile
# =========================================================
dev_alc_missing = df["ALQ130"].isna().to_numpy()
ext_alc_missing = ext["ALQ130"].isna().to_numpy()

dev_age_groups = pd.cut(df["RIDAGEYR"], bins=[17, 39, 59, np.inf],
                        labels=["18-39", "40-59", "60+"]).astype(str).to_numpy()

missing_profile_rows = [
    {"cohort": "2017-2018 development", "group": "overall",
     "n": len(df), "pct_alcohol_missing": 100 * dev_alc_missing.mean()},
    {"cohort": "2021-2023 temporal", "group": "overall",
     "n": len(ext), "pct_alcohol_missing": 100 * ext_alc_missing.mean()},
]
for label, mask in [("phenotype positive", y_dev_arr == 1),
                    ("phenotype negative", y_dev_arr == 0)]:
    missing_profile_rows.append({
        "cohort": "2017-2018 development", "group": label,
        "n": int(mask.sum()),
        "pct_alcohol_missing": 100 * dev_alc_missing[mask].mean()
    })
for band in ["18-39", "40-59", "60+"]:
    mask = dev_age_groups == band
    missing_profile_rows.append({
        "cohort": "2017-2018 development", "group": f"age {band}",
        "n": int(mask.sum()),
        "pct_alcohol_missing": 100 * dev_alc_missing[mask].mean()
    })

# Phenotype prevalence by missingness status (is missingness informative?)
for label, mask in [("alcohol missing", dev_alc_missing),
                    ("alcohol observed", ~dev_alc_missing)]:
    missing_profile_rows.append({
        "cohort": "2017-2018 development",
        "group": f"phenotype prevalence | {label}",
        "n": int(mask.sum()),
        "pct_alcohol_missing": np.nan,
        "phenotype_prevalence_pct": 100 * y_dev_arr[mask].mean(),
    })


# =========================================================
# 2. Skip-pattern information from the raw alcohol questionnaires
# =========================================================
def skip_pattern_frame(raw):
    out = raw[["SEQN"]].copy()
    for c in ["ALQ111", "ALQ121"]:
        out[c] = raw[c] if c in raw.columns else np.nan
    out["structural_nondrinker"] = (
        (out["ALQ111"] == 2) | (out["ALQ121"] == 0)
    ).astype(int)
    return out

try:
    alq_j_skip = skip_pattern_frame(read_j("ALQ_J.XPT"))
    alq_l_skip = skip_pattern_frame(read_l("ALQ_L.XPT"))
    SKIP_PATTERN_AVAILABLE = True
except Exception as exc:
    print("Skip-pattern variables could not be downloaded:", exc)
    SKIP_PATTERN_AVAILABLE = False

if SKIP_PATTERN_AVAILABLE:
    dev_skip = (pd.DataFrame({"SEQN": seqn_dev.to_numpy()})
                .merge(alq_j_skip, on="SEQN", how="left"))
    ext_skip = (ext[["SEQN"]].merge(alq_l_skip, on="SEQN", how="left"))

    for cohort, frame, miss in [
        ("2017-2018 development", dev_skip, dev_alc_missing),
        ("2021-2023 temporal", ext_skip, ext_alc_missing),
    ]:
        structural = frame["structural_nondrinker"].to_numpy() == 1
        missing_profile_rows.append({
            "cohort": cohort,
            "group": "missing ALQ130 explained by skip pattern (non-drinker)",
            "n": int(miss.sum()),
            "pct_alcohol_missing": 100 * (structural & miss).sum() / max(miss.sum(), 1),
        })

alcohol_missing_profile = pd.DataFrame(missing_profile_rows)
display(alcohol_missing_profile.round(2))
alcohol_missing_profile.to_csv(
    REVISION_DIR / "R2c3_alcohol_missingness_profile.csv", index=False
)
REVISION_TABLES["alcohol_missing_profile"] = alcohol_missing_profile


# =========================================================
# 3. Alternative specifications, fitted with the primary procedure
# =========================================================
def make_preprocessor_with_indicator(X):
    cat_cols = list(X.select_dtypes(exclude=np.number).columns)
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler())
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])

NO_ALCOHOL_FEATURES = [c for c in PRIMARY_FEATURES if c != "ALQ130"]

alcohol_variants = {}
alcohol_variants["V0 primary: median imputation"] = (
    comparator_results[f"{SELECTED_MODEL_NAME} (primary, 12 predictors)"]
)
alcohol_variants["V1 alcohol removed (11 predictors)"] = fit_frozen_variant(
    "V1 alcohol removed (11 predictors)",
    df[NO_ALCOHOL_FEATURES], ext[NO_ALCOHOL_FEATURES],
    selected_template, selected_params, seed_offset=3501
)
alcohol_variants["V2 median imputation + missingness indicator"] = fit_frozen_variant(
    "V2 median imputation + missingness indicator",
    df[PRIMARY_FEATURES], ext[PRIMARY_FEATURES],
    selected_template, selected_params, seed_offset=3502,
    preprocessor=make_preprocessor_with_indicator(df[PRIMARY_FEATURES])
)

if SKIP_PATTERN_AVAILABLE:
    X_dev_skip = df[PRIMARY_FEATURES].copy()
    X_ext_skip = ext[PRIMARY_FEATURES].copy()
    X_dev_skip.loc[
        X_dev_skip["ALQ130"].isna().to_numpy()
        & (dev_skip["structural_nondrinker"].to_numpy() == 1), "ALQ130"
    ] = 0.0
    X_ext_skip.loc[
        X_ext_skip["ALQ130"].isna().to_numpy()
        & (ext_skip["structural_nondrinker"].to_numpy() == 1), "ALQ130"
    ] = 0.0
    print("Remaining ALQ130 missingness after skip-pattern recode: "
          f"dev {100 * X_dev_skip['ALQ130'].isna().mean():.1f}%, "
          f"temporal {100 * X_ext_skip['ALQ130'].isna().mean():.1f}%")
    alcohol_variants["V3 skip-pattern recode (non-drinkers = 0)"] = fit_frozen_variant(
        "V3 skip-pattern recode (non-drinkers = 0)",
        X_dev_skip, X_ext_skip,
        selected_template, selected_params, seed_offset=3503
    )

alcohol_table = pd.DataFrame([summarize_variant(r) for r in alcohol_variants.values()])

# Paired bootstrap of temporal AUC and Brier differences vs V0
v0_p = alcohol_variants["V0 primary: median imputation"]["p_ext"]
alc_boot = defaultdict(list)
for idx in iid_bootstrap_indices(len(y_ext_arr), N_BOOT, RANDOM_STATE + 3504):
    yy = y_ext_arr[idx]
    if np.unique(yy).size < 2:
        continue
    for name, r in alcohol_variants.items():
        alc_boot[(name, "auc")].append(
            roc_auc_score(yy, r["p_ext"][idx]) - roc_auc_score(yy, v0_p[idx]))
        alc_boot[(name, "brier")].append(
            brier_score_loss(yy, r["p_ext"][idx]) - brier_score_loss(yy, v0_p[idx]))

for metric in ["auc", "brier"]:
    lows, highs, points = [], [], []
    for name, r in alcohol_variants.items():
        if metric == "auc":
            points.append(roc_auc_score(y_ext, r["p_ext"]) - roc_auc_score(y_ext, v0_p))
        else:
            points.append(brier_score_loss(y_ext, r["p_ext"]) - brier_score_loss(y_ext, v0_p))
        lo, hi = percentile_ci(alc_boot[(name, metric)])
        lows.append(lo)
        highs.append(hi)
    alcohol_table[f"temporal_{metric}_minus_V0"] = points
    alcohol_table[f"temporal_{metric}_minus_V0_ci_low"] = lows
    alcohol_table[f"temporal_{metric}_minus_V0_ci_high"] = highs

display(alcohol_table.round(4))
alcohol_table.to_csv(
    REVISION_DIR / "R2c3_alcohol_sensitivity_models.csv", index=False
)
REVISION_TABLES["alcohol_sensitivity"] = alcohol_table
