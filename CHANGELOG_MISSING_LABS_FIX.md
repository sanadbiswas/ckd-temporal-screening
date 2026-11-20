# Changelog — Missing-Lab-Value Outcome Coding Fix

## Summary

Both cohort-construction scripts defined the CKD screening phenotype as:

```python
df["ckd"] = ((df["egfr"] < 60) | (df["URDACT"] >= 30)).astype(int)
```

without first confirming that `LBXSCR` (serum creatinine, used to compute
`egfr`) and `URDACT` (urine albumin-to-creatinine ratio) were individually
non-missing for every participant. The inner joins used to build each
cohort (`bio.merge(albcr, on="SEQN", how="inner")`, and the equivalent in
the temporal script) only guarantee that a participant has a *row* in the
relevant NHANES lab file — not that the specific analyte value in that row
is non-missing. NHANES lab files routinely contain rows with a missing
value for an individual analyte (insufficient specimen, assay failure,
etc.), even for participants who otherwise completed that lab file.

Because pandas evaluates `NaN < 60` and `NaN >= 30` as `False`, any
participant with both values missing — or with one value missing while the
other happened to read below threshold — was silently coded
`ckd = 0` (phenotype-negative) instead of being excluded from the analytic
cohort for insufficient information to construct the outcome.

This affected **both** cohorts identically:
- `modules/01_data_preparation.py` (development cohort, NHANES 2017–2018)
- `modules/13_temporal_data.py` (temporal validation cohort, NHANES
  August 2021–August 2023)

## What was fixed

An explicit exclusion step was added to both scripts, placed after the age
filter and before any outcome construction:

```python
n_before_labs = len(df)
df = df.dropna(subset=["LBXSCR", "URDACT"]).reset_index(drop=True)
n_removed_labs = n_before_labs - len(df)
print(f"Excluded for missing serum creatinine or urinary ACR: {n_removed_labs:,}")
```

An assertion immediately before outcome construction now verifies the
exclusion actually worked (`LBXSCR` and `URDACT` are both fully non-missing
by that point), so a future change that accidentally reintroduces this bug
will fail loudly instead of silently miscoding participants again.

Two hardcoded regression-style assertions in `01_data_preparation.py` that
locked in the old (bug-affected) cohort size and case count —
`assert len(df) == 5533` and `assert int(df["ckd"].sum()) == 1020` — were
removed, since they encoded the wrong answer as ground truth. They are
replaced with prints reporting whatever the corrected pipeline actually
produces, alongside the old values for reference. A dead notebook-artifact
line (a bare `missing_summary` expression that printed nothing when run as
a plain script) was also fixed to actually print.

## What this means for every existing result

This is an outcome-label bug, not a downstream statistical or reporting
bug. It changes:
- The final size of both cohorts (currently reported as 5,533 development /
  6,337 temporal — both numbers are pre-fix and will change once the
  corrected scripts are re-run against live NHANES data)
- Phenotype prevalence in both cohorts
- Every single performance metric derived from those outcomes: nested-CV
  comparison, calibration, the frozen screening threshold, temporal
  validation, survey-weighted evaluation, subgroup performance, permutation
  importance, decision-curve analysis, and every sensitivity analysis

**No previously reported number can be assumed to still hold.** A full,
fresh run of the entire pipeline (`ckd-runner.ipynb` end to end) is required
before any manuscript numbers can be trusted again.

## Action required before re-running

1. `nhanes_ckd_risk_clean.csv` is **not included** in this delivery, and the
   `RUN_DATA_PREPARATION` flag in `ckd-runner.ipynb` now defaults to `True`
   (previously `False`), specifically so the corrected
   `01_data_preparation.py` actually regenerates this file from CDC/NCHS
   rather than silently reusing the stale, pre-fix copy.
2. `modules/00_setup.py` falls back to downloading a development CSV from a
   GitHub-hosted copy if no local file is present
   (`raw.githubusercontent.com/sanadbiswas/ckd-mlwa-temporal-screening/...`).
   That hosted file may **also** be the pre-fix version — check it, and
   replace it with a corrected copy once you have one, or you risk silently
   pulling stale data even with the local flag set correctly.
3. All previously-generated result folders (`mlwa_results/`,
   `mlwa_research_grade_results/`, plus any version-suffixed variants) were
   built from the pre-fix phenotype and are not included in this delivery.
   Do not reuse any CSV, figure, or number from those folders. The results
   folder produced by this corrected pipeline is named plainly
   (`mlwa_results/`, set in `modules/00_setup.py`), with no version suffix.
4. Once a fresh end-to-end run completes, every number in the manuscript
   (all tables, both calibration figures, the decision-curve figure, the
   permutation-importance figure, and the participant-flow diagram's
   reported cohort sizes) needs to be re-verified against the new output,
   not assumed to match the previous draft.

## Not changed

No other part of the modeling pipeline was touched: preprocessing,
candidate models, nested cross-validation mechanics, calibration procedure,
threshold-selection logic, and every sensitivity-analysis module are
unchanged from the previously delivered version. This fix is scoped
entirely to how the outcome label itself is constructed in the two cohort
scripts.

## Addendum: a second, independent bug found during verification

After the missing-lab-value fix above was applied and run, results still came
back byte-identical to the pre-fix numbers on every development-side metric
(nested-CV comparison, calibration comparison, paired fold comparison),
despite `nhanes_ckd_risk_clean.csv` correctly showing the smaller, corrected
cohort (5,063 rows) on disk. This is a second, independent bug, not a
recurrence of the first one.

**Root cause:** `00_setup.py` resolves `DATA_PATH` exactly once, at the very
start of the run, by checking whether `nhanes_ckd_risk_clean.csv` already
exists on disk. If it did not exist yet at that exact moment (for example, on
a fresh project checkout, or after deleting a previous copy), `00_setup.py`
fell back to a GitHub-hosted copy (`nhanes_ckd_risk_data.csv`) -- which is
itself a stale, pre-fix file with the old 5,533-row cohort. `01_data_preparation.py`
then correctly writes a fresh, corrected `nhanes_ckd_risk_clean.csv` -- but
`DATA_PATH` had already been assigned earlier and is never re-checked, so
every downstream module (`02_development_data.py` onward) kept reading the
old, stale file for the rest of the run.

**Fix:** `01_data_preparation.py` now explicitly reassigns `DATA_PATH =
OUTPUT_FILE` immediately after writing the corrected CSV, forcing every
downstream module to use the file that was actually just written, regardless
of what `00_setup.py` resolved at the start of the run.

**Action required:** none beyond re-running the pipeline with this updated
module. If you want to remove the ambiguity entirely, you can also delete
the stale `nhanes_ckd_risk_data.csv` file from your project folder -- it is
a leftover from an earlier fallback download and is no longer needed once
`RUN_DATA_PREPARATION = True` (the current default) is used, but its
presence is now harmless either way because of the fix above.
