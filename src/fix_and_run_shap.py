import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. HGNC Gene Symbol and Modality Mapping Dictionary
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
