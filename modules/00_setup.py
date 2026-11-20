# Modular setup: paths, imports, and reproducibility constants.
from pathlib import Path
import urllib.request

ROOT = Path(globals().get("PROJECT_ROOT", Path.cwd())).resolve()
RESULTS_DIR = ROOT / "mlwa_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Development data: ONLY the corrected cleaned CSV is accepted.
# ---------------------------------------------------------------
# REVISION NOTE (MLWA-D-26-01812, Round 1):
# Earlier versions also accepted ``nhanes_ckd_risk_data.csv`` as a local
# fallback and downloaded that same file from GitHub when nothing was found
# locally. That file was produced BEFORE the missing-laboratory fix
# (n = 5,533; see CHANGELOG_MISSING_LABS_FIX.md) and must never be used.
# The only accepted development file is now ``nhanes_ckd_risk_clean.csv``
# (n = 5,063), produced by the corrected 01_data_preparation.py.
# The GitHub fallback points to the repository cited in the manuscript's
# data availability statement and to the corrected file name.
DATA_PATH = ROOT / "nhanes_ckd_risk_clean.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/sanadbiswas/"
    "ckd-temporal-screening/main/nhanes_ckd_risk_clean.csv"
)

STALE_PRE_FIX_FILE = ROOT / "nhanes_ckd_risk_data.csv"
if STALE_PRE_FIX_FILE.exists():
    print(
        "WARNING: found the pre-fix file nhanes_ckd_risk_data.csv in the "
        "project folder. It is ignored by this pipeline and should be deleted."
    )

if not DATA_PATH.exists():
    print("Local nhanes_ckd_risk_clean.csv not found; downloading from GitHub...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    except Exception as exc:
        print(
            "GitHub download failed (", exc, "). Set RUN_DATA_PREPARATION = True "
            "in the runner so 01_data_preparation.py rebuilds the file from CDC/NCHS."
        )

EXPECTED_DEV_N = 5063        # corrected development cohort size
EXPECTED_DEV_CASES = 939     # corrected phenotype-positive cases

print("Project root:", ROOT)
print("Dataset:", DATA_PATH)
print("Results directory:", RESULTS_DIR)

# =========================================================
# 1. Imports and reproducibility
# =========================================================
import io
import json
import math
import warnings
import platform
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from scipy.stats import randint, uniform, loguniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_ITER = 20
N_BOOT = 2000

# Revision-round settings (used only by modules 26-39).
N_BOOT_DESIGN = 1000          # survey-design (Rao-Wu) bootstrap replicates
N_BOOT_SUBGROUP = 1000        # participant-level bootstrap within subgroups
TARGET_SENSITIVITY = 0.80     # prespecified screening target (unchanged)
DCA_THRESHOLDS = np.linspace(0.03, 0.40, 75)   # same grid as module 16

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")

print("Python:", platform.python_version())

from IPython.display import display
