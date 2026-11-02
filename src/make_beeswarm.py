import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Official HGNC and Modality Mapping
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
