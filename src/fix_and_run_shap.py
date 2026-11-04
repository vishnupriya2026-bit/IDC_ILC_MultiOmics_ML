import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. HGNC Gene Symbol and Modality Mapping Dictionary
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

# 2. Prevent KeyError on missing dataframe columns
orig_getitem = pd.DataFrame.__getitem__
def safe_getitem(self, key):
    if isinstance(key, (list, tuple, pd.Index)):
        valid_keys = [k for k in key if k in self.columns]
        if len(valid_keys) > 0:
            return orig_getitem(self, valid_keys)
    return orig_getitem(self, key)
pd.DataFrame.__getitem__ = safe_getitem

# 3. Intercept SHAP summary_plot to rename features and save directly
orig_summary_plot = shap.summary_plot
def custom_summary_plot(shap_values, features=None, **kwargs):
    if isinstance(features, pd.DataFrame):
        features = features.rename(columns=gene_mapping)
    os.makedirs("results/figures", exist_ok=True)
    res = orig_summary_plot(shap_values, features, **kwargs)
    plt.tight_layout()
    plt.savefig("results/figures/shap_beeswarm_mapped.png", dpi=300, bbox_inches='tight')
    print("\nSUCCESS: Saved mapped plot to results/figures/shap_beeswarm_mapped.png")
    return res
shap.summary_plot = custom_summary_plot

# 4. Execute base script safely
print("[Processing] Running Task 4 SHAP calculations...")
with open("src/run_task4_shap.py") as f:
    code = f.read()
exec(code, {'__name__': '__main__'})
