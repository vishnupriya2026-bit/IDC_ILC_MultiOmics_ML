import os
import glob
import shutil

cwd = os.getcwd()
figures_dir = os.path.join(cwd, "results", "figures")
os.makedirs(figures_dir, exist_ok=True)

# 1. Ensure all root/results PNGs are moved to results/figures/
for png in glob.glob("results/*.png"):
    shutil.move(png, os.path.join(figures_dir, os.path.basename(png)))

# 2. Complete Figure Inventory Mapping for Manuscript
figure_mapping = {
    "Figure 1": "preprocessing_qc_before_after.png",
    "Figure 2": "diablo_circos_plot.png",
    "Figure 3": "cv_roc_curves.png",
    "Figure 4": "threshold_calibration_curve.png",
    "Figure 5": "biomarker_parsimony_curve.png",
    "Figure 6": "shap_summary_beeswarm.png",
    "Figure 7": "shap_vs_cytoscape_concordance.png",
    "Figure 8": "string_ppi_network.png",
    "Figure 9": "kegg_reactome_enrichment_dotplot.png"
}

print("=" * 70)
print("             MANUSCRIPT SUBMISSION FIGURE INVENTORY             ")
print("=" * 70)

for fig_num, fig_name in figure_mapping.items():
    fig_path = os.path.join(figures_dir, fig_name)
    exists = os.path.exists(fig_path)
    status = "FOUND" if exists else "MISSING"
    size_kb = (os.path.getsize(fig_path) / 1024) if exists else 0
    print(f"[{status}] {fig_num:<10} -> {fig_name:<40} ({size_kb:.1f} KB)")

print("=" * 70)
