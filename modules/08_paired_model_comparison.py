# Modular pipeline stage.

# Source: final notebook cell 19
# =========================================================
# Paired OUTER-FOLD model comparison
# =========================================================
# Because the same outer folds are used for every model, compare the
# fold-level test metrics directly. With only 5 paired folds, these
# tests are descriptive/sensitivity analyses rather than definitive
# hypothesis tests.

from scipy.stats import ttest_rel, wilcoxon

selected_fold_metrics = nested_results[SELECTED_MODEL_NAME]["fold_metrics"]

paired_rows = []

for other_name, result in nested_results.items():
    if other_name == SELECTED_MODEL_NAME:
        continue

    other_fm = result["fold_metrics"]

    row = {"comparison": f"{SELECTED_MODEL_NAME} vs {other_name}"}

    for metric in ["roc_auc", "pr_auc", "brier", "log_loss"]:
        a = selected_fold_metrics[metric].to_numpy(dtype=float)
        b = other_fm[metric].to_numpy(dtype=float)
        diff = a - b

        # Paired t-test (reported cautiously due to n=5 folds)
        t_stat, t_p = ttest_rel(a, b)

        # Wilcoxon signed-rank as a nonparametric sensitivity check.
        try:
            w_stat, w_p = wilcoxon(diff)
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        row[f"{metric}_mean_diff"] = diff.mean()
        row[f"{metric}_sd_diff"] = diff.std(ddof=1)
        row[f"{metric}_paired_t_p"] = t_p
        row[f"{metric}_wilcoxon_p"] = w_p

    paired_rows.append(row)

paired_fold_results = pd.DataFrame(paired_rows)

display(paired_fold_results.round(4))
paired_fold_results.to_csv(
    RESULTS_DIR / "paired_outer_fold_model_comparison.csv",
    index=False
)

print(
    "Interpretation note: with only five outer folds, treat p-values as "
    "supporting/descriptive evidence, not as strong proof of superiority."
)

