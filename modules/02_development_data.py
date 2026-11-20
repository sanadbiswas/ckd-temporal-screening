# Modular pipeline stage.

# Source: final notebook cell 4
df_raw = pd.read_csv(DATA_PATH)

required = [
    "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "INDFMPIR",
    "has_htn_dx", "has_diabetes_dx", "BMXBMI", "BMXWAIST",
    "smoked_100_cigs", "ALQ130",
    "BPXSY1", "BPXSY2", "BPXSY3",
    "BPXDI1", "BPXDI2", "BPXDI3", "ckd"
]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    raise ValueError(f"Required columns missing from cleaned file: {missing}")

df = df_raw.copy()
df["mean_sbp"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
df["mean_dbp"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)

PRIMARY_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "DMDEDUC2",
    "INDFMPIR",
    "has_htn_dx",
    "has_diabetes_dx",
    "BMXBMI",
    "BMXWAIST",
    "smoked_100_cigs",
    "ALQ130",
    "mean_sbp",
    "mean_dbp",
]

X_dev = df[PRIMARY_FEATURES].copy()
y_dev = df["ckd"].astype(int).copy()
seqn_dev = df["SEQN"].copy()

# Revision guard: the corrected development cohort is n = 5,063 with
# 939 phenotype-positive participants. A different size almost always
# means the pre-fix file (n = 5,533) or a partial download was loaded.
if len(df) != EXPECTED_DEV_N or int(df["ckd"].sum()) != EXPECTED_DEV_CASES:
    print(
        f"WARNING: development cohort is n={len(df):,} with "
        f"{int(df['ckd'].sum()):,} cases; expected n={EXPECTED_DEV_N:,} "
        f"with {EXPECTED_DEV_CASES:,} cases. Check DATA_PATH before "
        f"interpreting any result."
    )

print("Development n:", len(df))
print("Unweighted CKD phenotype prevalence:", round(y_dev.mean(), 4))
print("Primary predictors:", PRIMARY_FEATURES)


# Source: final notebook cell 6
def read_xpt_url(url, timeout=120):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_sas(io.BytesIO(r.content), format="xport")

DEMO_J_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT"

demo_j = read_xpt_url(DEMO_J_URL)[
    ["SEQN", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
].copy()

survey_dev = pd.DataFrame({"SEQN": seqn_dev}).merge(
    demo_j, on="SEQN", how="left", validate="one_to_one"
)

weighted_prev = np.average(
    y_dev,
    weights=survey_dev["WTMEC2YR"]
)

print("Survey-weighted CKD phenotype prevalence:", round(weighted_prev, 4))

