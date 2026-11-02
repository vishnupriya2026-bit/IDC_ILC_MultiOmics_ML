import os
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

os.makedirs("results/figures", exist_ok=True)
out_path = "results/figures/shap_beeswarm_mapped.png"

# 1. Gene & Protein Symbol Formatter
def format_feature(col):
    gene_map = {
        "ENSG00000111404.7": "RERGL (RNA)", "ENSG00000154330.13": "PGM5 (RNA)",
        "ENSG00000241644.2": "INMT (RNA)", "ENSG00000149451.18": "ADAM33 (RNA)",
        "ENSG00000161544.10": "CYGB (RNA)", "ENSG00000124664.11": "SPDEF (RNA)",
        "ENSG00000187260.16": "WDR86 (RNA)", "ENSG00000064300.9": "NGFR (RNA)",
        "ENSG00000072163.20": "LIMS2 (RNA)", "ENSG00000126218.12": "F10 (RNA)",
        "ENSG00000137225.13": "CAPN11 (RNA)", "ENSG00000165349.12": "SLC7A3 (RNA)",
        "ENSG00000204065.3": "TCEAL5 (RNA)", "ENSG00000187486.5": "KCNJ11 (RNA)",
        "ENSG00000129514.8": "FOXA1 (RNA)",
        "ECADHERIN": "E-cadherin (Protein)", "BETACATENIN": "beta-catenin (Protein)",
        "EGFR": "EGFR (Protein)", "AR": "AR (Protein)"
    }
    col_str = str(col)
    if col_str in gene_map:
        return gene_map[col_str]
    if col_str.startswith("RNA_"):
        return col_str.replace("RNA_", "") + " (RNA)"
    if col_str.startswith("PROT_"):
        return col_str.replace("PROT_", "") + " (Protein)"
    return col_str

# 2. Run base task script to obtain raw SHAP matrix in memory
print("[1/3] Calculating sample-level SHAP values...")
with open("src/run_task4_shap.py") as f:
    script_code = f.read()

# Replace any Seaborn barplot calls with variable capture
scope = {}
exec(script_code, scope)

# 3. Extract sample-level SHAP matrix & feature DataFrame
shap_vals = scope.get("shap_values", None)
X_df = scope.get("X_trans_df", scope.get("X_test", None))

if shap_vals is None or X_df is None:
    raise ValueError("Could not extract shap_values array or feature DataFrame from run_task4_shap.py")

# Ensure shap_vals is 2D matrix (n_samples, n_features)
if hasattr(shap_vals, "values"):
    shap_vals_matrix = shap_vals.values
else:
    shap_vals_matrix = np.array(shap_vals)

# Rename columns to user-friendly HGNC symbols + modality
X_mapped = X_df.copy()
X_mapped.columns = [format_feature(c) for c in X_mapped.columns]

# 4. Render native SHAP beeswarm plot
print("[2/3] Generating SHAP Beeswarm Dot Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals_matrix,
    X_mapped,
    plot_type="dot",
    show=False
)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[3/3] Successfully generated beeswarm plot at: {out_path}")
