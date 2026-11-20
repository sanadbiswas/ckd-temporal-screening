# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 2, comment 2  — included vs excluded participants
#                            (missing serum creatinine or urinary ACR)
#   Reviewer 2, comment 6  — phenotype prevalence as n/N and %, both cohorts
#   Reviewer 3, minor 2    — composite phenotype vs its components
#                            (reduced eGFR, albuminuria) in both cohorts
#
# The development CSV deliberately drops the outcome-source laboratory
# values (01_data_preparation.py), so the development components and the
# excluded participants are rebuilt here from the same CDC/NCHS files.
# Requires internet access (Colab).

# =========================================================
# 1. Rebuild the 2017-2018 adult source cohort with laboratory values
# =========================================================
BASE_J = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

_XPT_CACHE = {}

def read_j(filename):
    if filename not in _XPT_CACHE:
        _XPT_CACHE[filename] = read_xpt_url(f"{BASE_J}/{filename}")
    return _XPT_CACHE[filename].copy()

YES_NO = {1: "yes", 2: "no", 7: np.nan, 9: np.nan}
SEX_MAP = {1: "male", 2: "female"}
RACE_MAP = {
    1: "mexican_american",
    2: "other_hispanic",
    3: "non_hispanic_white",
    4: "non_hispanic_black",
    5: "other_or_multiracial",
}
EDU_MAP = {
    1: "less_than_9th_grade",
    2: "9th_to_11th_grade",
    3: "high_school_grad_or_GED",
    4: "some_college_or_AA_degree",
    5: "college_grad_or_above",
    7: np.nan,
    9: np.nan,
}

def recode_predictors(frame, sbp_cols, dbp_cols):
    """Apply the same questionnaire recoding as modules 01 and 13."""
    out = frame.copy()
    out["sex_code"] = out["RIAGENDR"]
    for col in ["has_htn_dx", "has_diabetes_dx", "smoked_100_cigs"]:
        out[col] = out[col].map(YES_NO)
    out["RIAGENDR"] = out["RIAGENDR"].map(SEX_MAP)
    out["RIDRETH1"] = out["RIDRETH1"].map(RACE_MAP)
    out["DMDEDUC2"] = out["DMDEDUC2"].map(EDU_MAP)
    out["ALQ130"] = out["ALQ130"].replace({777: np.nan, 999: np.nan})
    out["mean_sbp"] = out[sbp_cols].mean(axis=1)
    out["mean_dbp"] = out[dbp_cols].mean(axis=1)
    return out

demo_raw_j = read_j("DEMO_J.XPT")[[
    "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "INDFMPIR",
    "WTMEC2YR", "SDMVPSU", "SDMVSTRA"
]]
bio_raw_j = read_j("BIOPRO_J.XPT")[["SEQN", "LBXSCR"]]
acr_raw_j = read_j("ALB_CR_J.XPT")[["SEQN", "URDACT"]]

dev_source = (
    demo_raw_j
    .merge(read_j("BPQ_J.XPT")[["SEQN", "BPQ020"]]
           .rename(columns={"BPQ020": "has_htn_dx"}), on="SEQN", how="left")
    .merge(read_j("DIQ_J.XPT")[["SEQN", "DIQ010"]]
           .rename(columns={"DIQ010": "has_diabetes_dx"}), on="SEQN", how="left")
    .merge(read_j("BMX_J.XPT")[["SEQN", "BMXBMI", "BMXWAIST"]],
           on="SEQN", how="left")
    .merge(read_j("SMQ_J.XPT")[["SEQN", "SMQ020"]]
           .rename(columns={"SMQ020": "smoked_100_cigs"}), on="SEQN", how="left")
    .merge(read_j("ALQ_J.XPT")[["SEQN", "ALQ130"]], on="SEQN", how="left")
    .merge(read_j("BPX_J.XPT")[[
        "SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"
    ]], on="SEQN", how="left")
    # Same inner joins on the laboratory files as 01_data_preparation.py
    .merge(bio_raw_j.merge(acr_raw_j, on="SEQN", how="inner"),
           on="SEQN", how="inner")
)

dev_adults = dev_source[dev_source["RIDAGEYR"] >= 18].copy()
dev_adults = recode_predictors(
    dev_adults,
    ["BPXSY1", "BPXSY2", "BPXSY3"],
    ["BPXDI1", "BPXDI2", "BPXDI3"]
)
dev_adults["excluded_missing_labs"] = (
    dev_adults[["LBXSCR", "URDACT"]].isna().any(axis=1)
).astype(int)

print("2017-2018 adults with laboratory records:", len(dev_adults))
print("  excluded for missing creatinine/ACR:",
      int(dev_adults["excluded_missing_labs"].sum()))
print("  retained:", int((dev_adults["excluded_missing_labs"] == 0).sum()))

# Integrity check: the rebuilt retained set must match the analytic cohort.
retained_seqn = set(
    dev_adults.loc[dev_adults["excluded_missing_labs"] == 0, "SEQN"]
)
if retained_seqn != set(seqn_dev):
    print(
        "WARNING: rebuilt retained development participants do not match "
        "the analytic cohort exactly. "
        f"Only in rebuild: {len(retained_seqn - set(seqn_dev))}, "
        f"only in analytic file: {len(set(seqn_dev) - retained_seqn)}"
    )
else:
    print("Integrity check passed: rebuilt retained set = analytic cohort.")


# =========================================================
# 2. Development outcome components (reduced eGFR, albuminuria)
# =========================================================
dev_labs = dev_adults.loc[
    dev_adults["excluded_missing_labs"] == 0,
    ["SEQN", "LBXSCR", "URDACT", "RIDAGEYR", "sex_code"]
].copy()

dev_labs["egfr"] = ckd_epi_2021(
    dev_labs["LBXSCR"], dev_labs["RIDAGEYR"], dev_labs["sex_code"]
)
dev_labs["low_egfr"] = (dev_labs["egfr"] < 60).astype(int)
dev_labs["albuminuria"] = (dev_labs["URDACT"] >= 30).astype(int)
dev_labs["ckd_rebuilt"] = (
    (dev_labs["low_egfr"] == 1) | (dev_labs["albuminuria"] == 1)
).astype(int)

# Align to the analytic cohort order (seqn_dev).
dev_components = (
    pd.DataFrame({"SEQN": seqn_dev.to_numpy()})
    .merge(dev_labs[["SEQN", "low_egfr", "albuminuria", "ckd_rebuilt", "egfr"]],
           on="SEQN", how="left", validate="one_to_one")
)

mismatch = int(
    (dev_components["ckd_rebuilt"].to_numpy() != y_dev_arr).sum()
)
print("Development phenotype mismatches after rebuild:", mismatch)


# =========================================================
# 3. Prevalence table: n/N, unweighted %, survey-weighted %
# =========================================================
prevalence_rows = []

def add_prevalence(cohort, outcome, y, weights, weight_name):
    y = np.asarray(y, dtype=int)
    prevalence_rows.append({
        "cohort": cohort,
        "outcome": outcome,
        "n_positive": int(y.sum()),
        "N": int(len(y)),
        "n_over_N": f"{int(y.sum()):,}/{len(y):,}",
        "unweighted_pct": 100 * y.mean(),
        "survey_weighted_pct": 100 * np.average(y, weights=weights),
        "survey_weight": weight_name,
    })

for outcome, yy in [
    ("CKD screening phenotype", y_dev_arr),
    ("Reduced eGFR (<60)", dev_components["low_egfr"].to_numpy()),
    ("Albuminuria (ACR >=30 mg/g)", dev_components["albuminuria"].to_numpy()),
    ("Both components", (dev_components["low_egfr"]
                         & dev_components["albuminuria"]).to_numpy()),
]:
    add_prevalence("2017-2018 development", outcome, yy, w_dev, "WTMEC2YR")

for outcome, yy in [
    ("CKD screening phenotype", y_ext_arr),
    ("Reduced eGFR (<60)", ext["low_egfr"].to_numpy()),
    ("Albuminuria (ACR >=30 mg/g)", ext["albuminuria"].to_numpy()),
    ("Both components", (ext["low_egfr"] & ext["albuminuria"]).to_numpy()),
]:
    add_prevalence("2021-2023 temporal", outcome, yy, w_ext_arr, "WTPH2YR")

prevalence_table = pd.DataFrame(prevalence_rows)
display(prevalence_table.round(2))
prevalence_table.to_csv(
    REVISION_DIR / "R2c6_prevalence_n_over_N.csv", index=False
)
REVISION_TABLES["prevalence"] = prevalence_table


# =========================================================
# 4. Included vs excluded participants (both cohorts)
# =========================================================
# 13_temporal_data.py keeps the raw NHANES questionnaire names until after
# the laboratory exclusion, so rename them before applying the shared recoding.
ext_adults = recode_predictors(
    ext_adults_before_lab_exclusion.rename(columns={
        "BPQ020": "has_htn_dx",
        "DIQ010": "has_diabetes_dx",
        "SMQ020": "smoked_100_cigs",
    }),
    ["BPXOSY1", "BPXOSY2", "BPXOSY3"],
    ["BPXODI1", "BPXODI2", "BPXODI3"]
)

CONTINUOUS_VARS = [
    ("RIDAGEYR", "Age, years"),
    ("INDFMPIR", "Income-to-poverty ratio"),
    ("BMXBMI", "BMI, kg/m2"),
    ("BMXWAIST", "Waist circumference, cm"),
    ("mean_sbp", "Mean systolic BP, mmHg"),
    ("mean_dbp", "Mean diastolic BP, mmHg"),
]

BINARY_VARS = [
    ("female", "Female, %"),
    ("htn_yes", "Diagnosed hypertension, %"),
    ("dm_yes", "Diagnosed diabetes, %"),
    ("smoke_yes", "Smoked >=100 cigarettes, %"),
    ("college", "College graduate or above, %"),
    ("nh_white", "Non-Hispanic White, %"),
    ("nh_black", "Non-Hispanic Black, %"),
    ("hispanic", "Mexican American or other Hispanic, %"),
    ("alcohol_missing", "Alcohol intake missing, %"),
]

def add_binary_indicators(frame):
    out = frame.copy()
    out["female"] = (out["RIAGENDR"] == "female").astype(float)
    out["htn_yes"] = np.where(out["has_htn_dx"].isna(), np.nan,
                              (out["has_htn_dx"] == "yes").astype(float))
    out["dm_yes"] = np.where(out["has_diabetes_dx"].isna(), np.nan,
                             (out["has_diabetes_dx"] == "yes").astype(float))
    out["smoke_yes"] = np.where(out["smoked_100_cigs"].isna(), np.nan,
                                (out["smoked_100_cigs"] == "yes").astype(float))
    out["college"] = np.where(out["DMDEDUC2"].isna(), np.nan,
                              (out["DMDEDUC2"] == "college_grad_or_above")
                              .astype(float))
    out["nh_white"] = (out["RIDRETH1"] == "non_hispanic_white").astype(float)
    out["nh_black"] = (out["RIDRETH1"] == "non_hispanic_black").astype(float)
    out["hispanic"] = out["RIDRETH1"].isin(
        ["mexican_american", "other_hispanic"]
    ).astype(float)
    out["alcohol_missing"] = out["ALQ130"].isna().astype(float)
    return out

def standardized_difference(a, b, binary):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if binary:
        pa, pb = a.mean(), b.mean()
        denom = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        return (pa - pb) / denom if denom > 0 else np.nan
    denom = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / denom if denom > 0 else np.nan

def included_excluded_table(frame, cohort):
    frame = add_binary_indicators(frame)
    inc = frame[frame["excluded_missing_labs"] == 0]
    exc = frame[frame["excluded_missing_labs"] == 1]

    rows = [{
        "cohort": cohort,
        "characteristic": "N",
        "included": f"{len(inc):,}",
        "excluded": f"{len(exc):,}",
        "smd": np.nan,
    }]

    for col, label in CONTINUOUS_VARS:
        rows.append({
            "cohort": cohort,
            "characteristic": label + " (mean ± SD)",
            "included": f"{inc[col].mean():.1f} ± {inc[col].std():.1f}",
            "excluded": f"{exc[col].mean():.1f} ± {exc[col].std():.1f}",
            "smd": standardized_difference(inc[col], exc[col], binary=False),
        })

    for col, label in BINARY_VARS:
        rows.append({
            "cohort": cohort,
            "characteristic": label,
            "included": f"{100 * inc[col].mean():.1f}",
            "excluded": f"{100 * exc[col].mean():.1f}",
            "smd": standardized_difference(inc[col], exc[col], binary=True),
        })

    # Which laboratory value was missing among the excluded.
    miss_scr = exc["LBXSCR"].isna()
    miss_acr = exc["URDACT"].isna()
    for label, mask in [
        ("Missing creatinine only", miss_scr & ~miss_acr),
        ("Missing ACR only", ~miss_scr & miss_acr),
        ("Missing both", miss_scr & miss_acr),
    ]:
        rows.append({
            "cohort": cohort,
            "characteristic": label + ", n",
            "included": "",
            "excluded": f"{int(mask.sum()):,}",
            "smd": np.nan,
        })

    return pd.DataFrame(rows)

included_excluded = pd.concat([
    included_excluded_table(dev_adults, "2017-2018 development"),
    included_excluded_table(ext_adults, "2021-2023 temporal"),
], ignore_index=True)

display(included_excluded.round(3))
included_excluded.to_csv(
    REVISION_DIR / "R2c2_included_vs_excluded.csv", index=False
)
REVISION_TABLES["included_vs_excluded"] = included_excluded

max_abs_smd = included_excluded["smd"].abs().max()
print(
    f"Largest absolute standardized difference: {max_abs_smd:.3f} "
    "(|SMD| < 0.10 is conventionally regarded as negligible)."
)
