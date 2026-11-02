import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Output directory
os.makedirs("results/figures", exist_ok=True)
output_path = "results/figures/shap_beeswarm_mapped.png"

# 2. Comprehensive Gene Symbol & Modality Mapping
gene_mapping = {
    "ENSG00000111404.7": "RERGL (RNA)", "ENSG00000154330.13": "PGM5 (RNA)",
    "ENSG00000241644.2": "INMT (RNA)", "ENSG00000149451.18": "ADAM33 (RNA)",
    "ENSG00000161544.10": "CYGB (RNA)", "ENSG00000124664.11": "SPDEF (RNA)",
    "ENSG00000187260.16": "WDR86 (RNA)", "ENSG00000064300.9": "NGFR (RNA)",
    "ENSG00000072163.20": "LIMS2 (RNA)", "ENSG00000126218.12": "F10 (RNA)",
    "ENSG00000137225.13": "CAPN11 (RNA)", "ENSG00000165349.12": "SLC7A3 (RNA)",
    "ENSG00000204065.3": "TCEAL5 (RNA)", "ENSG00000187486.5": "KCNJ11 (RNA)",
    "ENSG00000129514.8": "FOXA1 (RNA)",
    "ENSG00000111404": "RERGL (RNA)", "ENSG00000154330": "PGM5 (RNA)",
    "ENSG00000241644": "INMT (RNA)", "ENSG00000149451": "ADAM33 (RNA)",
    "ENSG00000161544": "CYGB (RNA)", "ENSG00000124664": "SPDEF (RNA)",
    "ENSG00000187260": "WDR86 (RNA)", "ENSG00000064300": "NGFR (RNA)",
    "ENSG00000072163": "LIMS2 (RNA)", "ENSG00000126218": "F10 (RNA)",
    "ENSG00000137225": "CAPN11 (RNA)", "ENSG00000165349": "SLC7A3 (RNA)",
    "ENSG00000204065": "TCEAL5 (RNA)", "ENSG00000187486": "KCNJ11 (RNA)",
    "ENSG00000129514": "FOXA1 (RNA)",
    "ECADHERIN": "E-cadherin (Protein)", "BETACATENIN": "beta-catenin (Protein)",
    "EGFR": "EGFR (Protein)", "AR": "AR (Protein)"
}

print("[1/3] Executing base SHAP task...")
exec(open("src/run_task4_shap.py").read())

print("[2/3] Mapping feature names and generating beeswarm plot...")
if 'X_trans_df' in locals():
    df_plot = X_trans_df.rename(columns=gene_mapping)
elif 'X_test' in locals():
    df_plot = X_test.rename(columns=gene_mapping)
else:
    raise ValueError("Feature DataFrame (X_trans_df or X_test) not found in scope.")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, df_plot, show=False)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[3/3] Successfully saved plot to: {output_path}")
