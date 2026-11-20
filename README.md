# CKD Temporal Screening — MLWA Revision 1

Reproducibility repository for:

**Temporal Validation of a Non-Laboratory Machine Learning Framework for Chronic Kidney Disease Screening in U.S. Adults**

Manuscript: **MLWA-D-26-01812**, *Machine Learning with Applications*.

## Final Revision 1 code

The final reviewer-revision pipeline is distributed as:

- **`ckd-project.zip`** — the complete modular project used for Revision 1.
- **`ckd-runner-colab.ipynb`** — Google Colab runner that pins the manuscript environment and executes the complete pipeline.

The archive contains **42 ordered modules (00–41)**, `pipeline_runner.py`, the corrected development dataset (`nhanes_ckd_risk_clean.csv`), requirements, changelogs, and figure-generation code.

Modules 00–25 reproduce the primary analysis. Modules 26–39 implement reviewer-requested analyses. Modules 40–41 export the figure data and generate the manuscript figures.

The older pre-revision code remains available in Git history; the files on `main` are the Revision 1 reproducibility release.

## Reproduce in Google Colab

1. Download `ckd-project.zip` and `ckd-runner-colab.ipynb`.
2. Put `ckd-project.zip` in:
   `My Drive/CKD-Paper-Final/`
3. Open `ckd-runner-colab.ipynb` in Google Colab.
4. Choose **Runtime → Run all**.
5. If the first environment cell changes an already-loaded package, restart the runtime and **Run all** once more.

The runner checks that all 42 modules are present, executes them in order, and prints reproduction checks against the manuscript's anchor results.

## Reproduction environment

The manuscript results were reproduced on Linux with two CPU cores using:

```text
Python              3.13.18
numpy               2.4.6
pandas              2.2.3
scipy               1.16.3
scikit-learn        1.6.1
xgboost             3.4.1
lightgbm            4.7.0
imbalanced-learn    0.14.2
statsmodels         0.15.0
joblib              1.6.0
threadpoolctl       3.7.0
```

The runner pins the Python packages above. The manuscript documents a platform-specific XGBoost near-tie observed on Windows; the published values were reproduced with the Linux environment above.

## Corrected cohorts

The outcome-lab correction excludes participants with missing serum creatinine or urinary ACR **before** constructing the single-visit CKD-compatible phenotype.

- Development (NHANES 2017–2018): **5,063 adults; 939 phenotype-positive**
- Temporal validation (NHANES 2021–2023): **5,552 adults; 949 phenotype-positive**

The obsolete pre-fix 5,533-row development dataset is not used.

## Anchor results

- Internal calibrated ROC-AUC: **0.7864**
- Temporal ROC-AUC: **0.7949**
- Frozen primary threshold: **0.1239**
- Temporal sensitivity / specificity: **0.8251 / 0.6037**
- Survey-weighted temporal sensitivity at primary threshold: **0.7671**
- Survey-weighted development threshold: **0.0948**
- Survey-weighted temporal sensitivity at refined threshold: **0.8012**

## Revision 1 analyses

The revision code adds, among other analyses:

- calibrated comparison of all model families and fully nested evaluation;
- calibrated logistic-regression and reduced-predictor benchmarks;
- risk-factor screening rules and burden-matched comparisons;
- survey-weighted threshold selection;
- Rao–Wu design-based uncertainty using NHANES strata and PSUs;
- calibration confidence intervals and threshold-local calibration;
- subgroup confidence intervals and screening-policy comparisons;
- age-only, age-removed, and outcome-component analyses;
- alcohol-missingness sensitivity analyses;
- screening scenarios and decision-curve interpretation;
- grouped permutation importance for correlated predictors;
- blood-pressure protocol sensitivity analysis.

## Data

NHANES source data are publicly available from the U.S. CDC/NCHS. The project archive includes the corrected cleaned development dataset and can rebuild it from the public raw NHANES files.

## Archive integrity

SHA-256 for `ckd-project.zip`:

`b8918a8b1b7f53abecef065edaa1010996c8c0bb6aeae4391bd081c44389a1eb`

## License

MIT License.
