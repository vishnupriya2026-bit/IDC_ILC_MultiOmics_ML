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
    "RERGL (RNA)": "RERGL (RNA)", "PGM5 (RNA)": "PGM5 (RNA)",
    "INMT (RNA)": "INMT (RNA)", "ADAM33 (RNA)": "ADAM33 (RNA)",
    "CYGB (RNA)": "CYGB (RNA)", "SPDEF (RNA)": "SPDEF (RNA)",
    "WDR86 (RNA)": "WDR86 (RNA)", "NGFR (RNA)": "NGFR (RNA)",
    "LIMS2 (RNA)": "LIMS2 (RNA)", "F10 (RNA)": "F10 (RNA)",
    "CAPN11 (RNA)": "CAPN11 (RNA)", "SLC7A3 (RNA)": "SLC7A3 (RNA)",
    "TCEAL5 (RNA)": "TCEAL5 (RNA)", "KCNJ11 (RNA)": "KCNJ11 (RNA)",
    "FOXA1 (RNA)": "FOXA1 (RNA)",
    "RERGL (RNA)": "RERGL (RNA)", "PGM5 (RNA)": "PGM5 (RNA)",
    "INMT (RNA)": "INMT (RNA)", "ADAM33 (RNA)": "ADAM33 (RNA)",
    "CYGB (RNA)": "CYGB (RNA)", "SPDEF (RNA)": "SPDEF (RNA)",
    "WDR86 (RNA)": "WDR86 (RNA)", "NGFR (RNA)": "NGFR (RNA)",
    "LIMS2 (RNA)": "LIMS2 (RNA)", "F10 (RNA)": "F10 (RNA)",
    "CAPN11 (RNA)": "CAPN11 (RNA)", "SLC7A3 (RNA)": "SLC7A3 (RNA)",
    "TCEAL5 (RNA)": "TCEAL5 (RNA)", "KCNJ11 (RNA)": "KCNJ11 (RNA)",
    "FOXA1 (RNA)": "FOXA1 (RNA)",
    "E-cadherin (Protein)": "E-cadherin (Protein)", "beta-catenin (Protein)": "beta-catenin (Protein)",
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
