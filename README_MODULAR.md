# CKD Project — Modular Code (Revision 1, MLWA-D-26-01812)

Temporal validation of a non-laboratory machine-learning framework for
CKD screening in U.S. adults (NHANES 2017–2018 development, NHANES
2021–2023 temporal validation).

## How to reproduce the results (Google Colab)

All published values were produced, and are verified, on Google Colab
(Linux) with the package versions pinned in `ckd-runner-colab.ipynb`.

1. Download this repository and place its folder in Google Drive under
   `My Drive/CKD-Paper-Final/` (any folder name is fine; the runner looks
   for the folder that contains `pipeline_runner.py` and `modules/`).
2. Put `ckd-runner-colab.ipynb` in `My Drive/CKD-Paper-Final/` and open it
   in Colab.
3. Runtime -> Run all. If the runtime restarts while package versions are
   being pinned, choose Run all again.

The runner executes every stage in order through `pipeline_runner.py`
(modules 00-41), stops if either reproduction check fails, and writes

```text
mlwa_results/                  primary results (original submission)
mlwa_results/revision_round1/  additional analyses (revision 1)
mlwa_results/figure_data/      the numbers behind each figure
mlwa_results/figures/          Figures 1-6 and Supplementary Figure S1
```

The results of the run reported in the manuscript are archived in
`mlwa_results_revision1.zip`.

To redraw or restyle a figure without re-running the pipeline:
`python make_figures.py` (all figures) or `python make_figures.py 1 4`.

Note on platforms: results reproduce exactly on Linux with the pinned
versions. On Windows, the XGBoost build selects a different
hyperparameter in one cross-validation fold, so XGBoost values in the
model comparison differ slightly; the selected model and all of its
results are unaffected.

## Data

- `nhanes_ckd_risk_clean.csv` — corrected development cohort
  (n = 5,063; 939 phenotype-positive). Produced by `01_data_preparation.py`,
  which excludes adults with missing serum creatinine or urinary ACR before
  constructing the outcome (see `CHANGELOG_MISSING_LABS_FIX.md`).
- `RUN_DATA_PREPARATION = True` in the runner rebuilds this file from
  CDC/NCHS raw XPT files; `False` uses the shipped copy. Both give identical
  results.
- The pre-fix file `nhanes_ckd_risk_data.csv` (n = 5,533) is no longer
  shipped, is never read by the pipeline, and is deleted by the runner if
  found.
- The 2021–2023 temporal cohort and several revision analyses download
  public NHANES files at run time (internet access required).

## Module order

Primary analysis — reproduces the submitted manuscript (unchanged):

- `00_setup.py` — paths, imports, reproducibility constants
- `01_data_preparation.py` — raw NHANES 2017–2018 acquisition and preparation
- `02_development_data.py` — development cohort and survey-design merge
- `03_preprocessing.py` — leakage-safe imputation/encoding/scaling
- `04_candidate_models.py` — candidate models and search spaces
- `05_metrics.py` — discrimination, probability and operating-point metrics
- `06_nested_cv.py` — nested cross-validation and model selection
- `07_bootstrap_uncertainty.py` — outer-fold uncertainty (Table 2 CIs)
- `08_paired_model_comparison.py` — paired outer-fold comparisons
- `09_final_tuning.py` — selected-model tuning on the development cohort
- `10_calibration.py` — raw/sigmoid/isotonic OOF calibration
- `11_screening_threshold.py` — frozen 80%-sensitivity operating point
- `12_final_model.py` — final frozen probability model
- `13_temporal_data.py` — NHANES 2021–2023 temporal cohort
- `14_temporal_validation.py` — frozen temporal evaluation and bootstrap CIs
- `15_temporal_calibration.py` — temporal calibration and figure
- `16_decision_curve.py` — decision-curve analysis
- `17_age_coupling_sensitivity.py` — age-removal analysis
- `18_outcome_component_sensitivity.py` — low-eGFR and albuminuria analyses
- `19_feature_set_sensitivity.py` — race/activity feature-set analysis
- `20_subgroup_analysis.py` — subgroup performance and calibration
- `21_stratified_thresholds.py` — age/diabetes threshold sensitivity
- `22_imbalance_sensitivity.py` — SMOTE/SMOTEENN/class-weight analysis
- `23_survey_weighted_evaluation.py` — population-weighted evaluation
- `24_permutation_importance.py` — temporal permutation importance
- `25_reproducibility_and_master_table.py` — environment record and master table

Revision 1 — reviewer-requested analyses (outputs in
`mlwa_results/revision_round1/`, each file prefixed by the comment it
answers, e.g. `R1c3_` = Reviewer 1, comment 3):

| Module | Analysis | Reviewer comments |
|---|---|---|
| `26_revision_setup.py` | Shared helpers: Rao-Wu design bootstrap, Wilson CIs, weighted threshold, policy net benefit, `fit_frozen_variant()` | — |
| `27_cohort_description.py` | Prevalence as n/N; included vs excluded; eGFR/albuminuria components | R2.2, R2.6, R3 minor 2 |
| `28_model_selection_revision.py` | One-SE rule details; Table 2 CI method; sigmoid vs isotonic differences; all five families compared after calibration; fully nested evaluation of the whole procedure | R1.2, R2.4, R2.5, R3.1, R3 minor 1, R4.5 |
| `29_comparators_and_ablation.py` | Calibrated logistic regression; predictor ablation; risk-factor rules; burden-matched comparison; DCA vs rules; testing burden | R1 main, R1.1, R4.1, R4.2 |
| `30_survey_weighted_threshold.py` | 80% sensitivity threshold chosen with survey weights, frozen and transported | R1.3, R2.7, R3.3, R4.4 |
| `31_design_based_uncertainty.py` | Weighted metrics with strata/PSU-aware intervals | R3.2 |
| `32_calibration_uncertainty.py` | CIs for calibration intercept/slope; ECE binning; calibration near the threshold; decision impact of recalibration | R3.4 |
| `33_subgroup_uncertainty_and_policies.py` | Wilson and design CIs for subgroup sensitivity/specificity; global vs stratified vs test-all policies; DCA within diabetes | R2.8, R2.9, R3 minor 5, R4.2 |
| `34_age_dependence.py` | Age-only model; increment over age by outcome; within-age discrimination | R2.11, R3 minor 2, R4.3 |
| `35_alcohol_missingness.py` | Alcohol removed; missing indicator; skip-pattern recode | R2.3, R3 minor 4, R4 minor |
| `36_screening_scenarios.py` | Tests, cases found/missed per 10,000; DCA threshold interpretation | R2.10, R3.5, R4.1 |
| `37_correlated_importance.py` | Predictor correlation; grouped permutation importance | R3 minor 3 |
| `38_blood_pressure_protocol.py` | Auscultatory vs oscillometric BP (BPXO_J) | R4 minor |
| `39_revision_summary.py` | `REVIEWER_COMMENT_OUTPUT_MAP.csv`, `REVISION_KEY_NUMBERS.json`, revision reproducibility record | all |

## Design principles of the revision

- The primary analysis (modules 00–25) is unchanged and is not refitted.
- Every alternative model is fitted with `fit_frozen_variant()`, which
  repeats the primary procedure exactly: development-only tuning, the
  selected calibration method, out-of-fold predictions on the same
  calibration folds, an 80%-sensitivity threshold from those predictions,
  and unchanged application to 2021–2023.
- Every alternative threshold or policy is fixed on 2017–2018 data before
  it is applied to 2021–2023.

## Expected run time (CPU)

Primary analysis: about the same as before (dominated by nested CV and the
subgroup bootstrap). Revision modules add roughly 20–40 minutes, most of it
in module 28 (five model families × three calibration states × five folds).
To make a quick check first, lower `N_BOOT`, `N_BOOT_DESIGN` and `N_ITER`
in `00_setup.py`, then restore them for the final run.
