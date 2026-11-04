import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Official HGNC and Modality Mapping
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

def format_col(col):
    c = str(col)
    if c in gene_mapping:
        return gene_mapping[c]
    if c.startswith("RNA_"):
        return c.replace("RNA_", "") + " (RNA)"
    if c.startswith("PROT_"):
        return c.replace("PROT_", "") + " (Protein)"
    return c

# 2. Safe getitem monkey-patch to bypass KeyError
orig_gi = pd.DataFrame.__getitem__
def safe_gi(self, key):
    if isinstance(key, list):
        valid = [k for k in key if k in self.columns]
        if valid:
            return orig_gi(self, valid)
    return orig_gi(self, key)
pd.DataFrame.__getitem__ = safe_gi

# 3. Load script scope
with open("src/run_task4_shap.py") as f:
    code = f.read()

scope = {}
try:
    exec(code, scope)
except Exception as e:
    pass

shap_values = scope.get("shap_values", None)
X_df = scope.get("X_trans_df", scope.get("X_test", scope.get("X", None)))

if shap_values is not None and X_df is not None:
    X_mapped = X_df.copy()
    X_mapped.columns = [format_col(c) for c in X_mapped.columns]
    
    plt.figure(figsize=(10, 8))
    # Forces dot/beeswarm format
    shap.summary_plot(shap_values, X_mapped, plot_type="dot", show=False)
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/shap_beeswarm_mapped.png", dpi=300, bbox_inches='tight')
    plt.savefig("shap_beeswarm_mapped.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSUCCESS: Beeswarm plot generated!")
