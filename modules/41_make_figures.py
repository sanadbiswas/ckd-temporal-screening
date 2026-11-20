# Modular pipeline stage — REVISION ROUND 1.
#
# Draws the six manuscript figures and Supplementary Figure S1 from
# mlwa_results/figure_data/ (written by module 40) into
#   mlwa_results/figures/   as .eps, .pdf and .png.
#
# The drawing code lives in make_figures.py at the project root, so the
# same figures can be restyled and redrawn in seconds without the
# pipeline:   python make_figures.py        (all figures)
#             python make_figures.py 1 4    (only Figures 1 and 4)

import runpy

figure_namespace = runpy.run_path(
    str(ROOT / "make_figures.py"),
    init_globals={
        "DATA_DIR_OVERRIDE": RESULTS_DIR / "figure_data",
        "OUT_DIR_OVERRIDE": RESULTS_DIR / "figures",
        "SHOW_FIGURES": True,
    },
    run_name="pipeline",
)
figure_namespace["make_all"]()
print("Figures written to:", RESULTS_DIR / "figures")
