# Modular pipeline stage — REVISION ROUND 1.
#
# Addresses:
#   Reviewer 3, comment 2 — survey weights, strata and PSUs in the
#       uncertainty analysis.
#
# Population-level (survey-weighted) performance is re-estimated with a
# Rao-Wu rescaling bootstrap that resamples PSUs within strata (module 26),
# and its intervals are set beside (a) the original participant-level
# bootstrap and (b) a participant-level bootstrap of the weighted metrics,
# so the effect of the design on interval width is visible.
#
# Model ESTIMATION remains unweighted by design: the model predicts an
# individual's probability, and survey weights describe how the sample
# represents the population, which matters for population summaries
# (prevalence, weighted sensitivity, weighted net benefit). This rationale
# is reported in the response; the analysis below quantifies the
# population-level uncertainty correctly.

def weighted_performance(y, p, thr, w):
    m = metrics_from_prob(y, p, threshold=thr, sample_weight=w)
    return {
        "prevalence": float(np.average(y, weights=w)),
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "brier": m["brier"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "ppv": m["precision"],
    }

design_rows = []
for cohort, (yy, pp, ww, st, ps) in cohort_inputs.items():
    point = weighted_performance(yy, pp, SCREENING_THRESHOLD, ww)

    # (a) Design-based (Rao-Wu) bootstrap of the weighted metrics
    design_boot = defaultdict(list)
    # Same seed as module 30 (RANDOM_STATE + 3000): both modules then draw
    # identical replicate weights, so the weighted sensitivity/specificity
    # intervals here equal those of the threshold comparison (Table 6).
    for rep_w in design_bootstrap_weights(ww, st, ps, N_BOOT_DESIGN,
                                          RANDOM_STATE + 3000):
        try:
            rep = weighted_performance(yy, pp, SCREENING_THRESHOLD, rep_w)
        except ValueError:
            continue
        for k, v in rep.items():
            design_boot[k].append(v)

    # (b) Participant-level bootstrap of the SAME weighted metrics
    iid_boot = defaultdict(list)
    for idx in iid_bootstrap_indices(len(yy), N_BOOT_DESIGN,
                                     RANDOM_STATE + 3101):
        if np.unique(yy[idx]).size < 2:
            continue
        rep = weighted_performance(yy[idx], pp[idx], SCREENING_THRESHOLD,
                                   ww[idx])
        for k, v in rep.items():
            iid_boot[k].append(v)

    for metric, est in point.items():
        d_lo, d_hi = percentile_ci(design_boot[metric])
        i_lo, i_hi = percentile_ci(iid_boot[metric])
        design_rows.append({
            "cohort": cohort,
            "metric": f"weighted_{metric}",
            "estimate": est,
            "design_ci_low": d_lo,
            "design_ci_high": d_hi,
            "design_ci_width": d_hi - d_lo,
            "participant_ci_low": i_lo,
            "participant_ci_high": i_hi,
            "participant_ci_width": i_hi - i_lo,
            "width_ratio_design_vs_participant": (
                (d_hi - d_lo) / (i_hi - i_lo) if (i_hi - i_lo) > 0 else np.nan
            ),
        })

design_uncertainty = pd.DataFrame(design_rows)
display(design_uncertainty.round(4))
design_uncertainty.to_csv(
    REVISION_DIR / "R3c2_design_based_uncertainty.csv", index=False
)
REVISION_TABLES["design_uncertainty"] = design_uncertainty

print(
    "Median design-effect on CI width (design / participant-level): "
    f"{design_uncertainty['width_ratio_design_vs_participant'].median():.2f}"
)

# Number of strata/PSUs actually used (for the Methods text).
design_description = pd.DataFrame([
    {"cohort": "2017-2018 development", "weight": "WTMEC2YR",
     "strata": len(np.unique(dev_strata)),
     "psus": len(set(zip(dev_strata, dev_psu))),
     "replicates": N_BOOT_DESIGN},
    {"cohort": "2021-2023 temporal", "weight": "WTPH2YR",
     "strata": len(np.unique(ext_strata)),
     "psus": len(set(zip(ext_strata, ext_psu))),
     "replicates": N_BOOT_DESIGN},
])
display(design_description)
design_description.to_csv(
    REVISION_DIR / "R3c2_design_description.csv", index=False
)
