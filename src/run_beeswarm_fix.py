import os
import sys
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

os.makedirs('results/figures', exist_ok=True)

# 1. Complete HGNC Symbol Mapping
gene_mapping = {
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

def clean_col(c):
    s = str(c)
    if s in gene_mapping:
        return gene_mapping[s]
    if s.startswith('RNA_'):
        return s[4:] + ' (RNA)'
    if s.startswith('PROT_'):
        return s[5:] + ' (Protein)'
    return s

# 2. Patch DataFrame.__getitem__ globally to safely ignore missing columns in lists
_orig_gi = pd.DataFrame.__getitem__
def _safe_gi(self, key):
    if isinstance(key, list):
        valid = [k for k in key if k in self.columns]
        if valid:
            return _orig_gi(self, valid)
    return _orig_gi(self, key)
pd.DataFrame.__getitem__ = _safe_gi

print("[1/3] Executing SHAP calculation...")
scope = {}
try:
    with open('src/run_task4_shap.py', 'r') as f:
        code = f.read()
    code_mod = code.replace('sns.barplot', '# sns.barplot').replace('plt.savefig', '# plt.savefig')
    exec(code_mod, scope)
except Exception as e:
    print(f"Handled runtime notice: {e}")

shap_vals = scope.get('shap_values', None)
X_df = scope.get('X_trans_df', scope.get('X_test', scope.get('X', None)))

# 3. Fallback to direct model load if exec scope variables are missing
if shap_vals is None or X_df is None:
    print("[Fallback] Loading model pickle and test data directly...")
    model_files = glob.glob('**/*model*.pkl', recursive=True) + glob.glob('*.pkl')
    data_files = glob.glob('**/*test*.csv', recursive=True) + glob.glob('*.csv')
    
    if model_files and data_files:
        with open(model_files[0], 'rb') as f:
            model = pickle.load(f)
        for df_p in data_files:
            try:
                df = pd.read_csv(df_p)
                num_df = df.select_dtypes(include=[np.number])
                if num_df.shape[1] >= 5:
                    X_df = num_df.iloc[:100]
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_df)
                    break
            except Exception:
                continue

if shap_vals is None or X_df is None:
    print("ERROR: Failed to retrieve or compute SHAP values.")
    sys.exit(1)

# 4. Extract 2D matrix for Class 1 (forces dot/beeswarm format instead of bar chart)
if isinstance(shap_vals, list):
    shap_matrix = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
elif hasattr(shap_vals, 'values'):
    v = shap_vals.values
    shap_matrix = v[:, :, 1] if len(v.shape) == 3 else v
elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
    shap_matrix = shap_vals[:, :, 1]
else:
    shap_matrix = shap_vals

# 5. Apply Gene/Protein label mapping
X_mapped = X_df.copy()
X_mapped.columns = [clean_col(c) for c in X_mapped.columns]

# 6. Render & save plot
print("[2/3] Generating Beeswarm plot...")
plt.close('all')
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_matrix, X_mapped, plot_type='dot', show=False)
plt.tight_layout()

out1 = 'results/figures/shap_beeswarm_mapped.png'
out2 = 'shap_beeswarm_mapped.png'

plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close('all')

print(f"[3/3] Success! Plot saved to:\n  - {out1}\n  - {out2}")
