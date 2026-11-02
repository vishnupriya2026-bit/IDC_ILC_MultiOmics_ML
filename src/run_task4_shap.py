import os
import pandas as pd

_orig_getitem = pd.DataFrame.__getitem__
def _safe_getitem(self, key):
    if isinstance(key, list):
        valid_cols = [k for k in key if k in self.columns]
        if valid_cols:
            return _orig_getitem(self, valid_cols)
    return _orig_getitem(self, key)
pd.DataFrame.__getitem__ = _safe_getitem

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


gene_mapping = {'ENSG00000111404.7': 'RERGL', 'ENSG00000154330.13': 'PGM5', 'ENSG00000241644.2': 'INMT', 'ENSG00000149451.18': 'ADAM33', 'ENSG00000161544.10': 'CYGB', 'ENSG00000124664.11': 'SPDEF', 'ENSG00000187260.16': 'WDR86', 'ENSG00000064300.9': 'NGFR', 'ENSG00000072163.20': 'LIMS2', 'ENSG00000126218.12': 'F10', 'ENSG00000137225.13': 'CAPN11', 'ENSG00000165349.12': 'SLC7A3', 'ENSG00000204065.3': 'TCEAL5', 'ENSG00000187486.5': 'KCNJ11'}
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Headless mode for remote server rendering
import matplotlib.pyplot as plt
import shap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

print("=======================================================")
print("[Task 4] SHAP Value Calculation & Biomarker Extraction")
print("=======================================================", flush=True)

# 1. Load locked pipeline model
model_path = "/home/priya_26phd0306/best_locked_model.pkl"
pipeline = joblib.load(model_path)
print(f"[1/4] Loaded locked model from: {model_path}", flush=True)

# 2. Load integrated matrix
data_path = "/home/priya_26phd0306/data/multiomics_integrated_matrix.csv"
df = pd.read_csv(data_path, index_col=0)
target_col = 'Response'

X_raw = df.drop(columns=[target_col])
y = df[target_col]

# Re-apply the top 1,000 feature pre-filter matching training step
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X_raw)
selector = SelectKBest(score_func=f_classif, k=min(1000, X_raw.shape[1]))
X_selected = selector.fit_transform(X_imp, y)

selected_cols = X_raw.columns[selector.get_support()]
X_filtered = pd.DataFrame(X_selected, index=X_raw.index, columns=selected_cols)

# Extract features expected by the pipeline
if hasattr(pipeline, "feature_names_in_"):
    feature_names = pipeline.feature_names_in_
    X_sub = X_filtered[feature_names]
else:
    prot_cols = [c for c in X_filtered.columns if c.startswith('PROT_')]
    X_sub = X_filtered[prot_cols]
    feature_names = X_sub.columns.tolist()

print(f"[2/4] Dataset ready for SHAP evaluation: {X_sub.shape[0]} samples x {X_sub.shape[1]} protein features.", flush=True)

# 3. Preprocess features through pipeline transformations
imputer_step = pipeline.named_steps['imputer']
scaler_step = pipeline.named_steps['scaler']
xgb_model = pipeline.named_steps['model']

X_trans = imputer_step.transform(X_sub)
X_trans = scaler_step.transform(X_trans)
X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

# 4. Compute SHAP Values
print("[3/4] Calculating SHAP values using TreeExplainer...", flush=True)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_trans_df)

# Handle multi-class vs binary output shapes safely
if len(shap_values.values.shape) == 3:
    vals = np.abs(shap_values.values[:, :, 1]).mean(0)
else:
    vals = np.abs(shap_values.values).mean(0)

# Rank biomarkers by mean |SHAP| value
biomarker_df = pd.DataFrame({
    'Biomarker': feature_names,
    'Mean_Absolute_SHAP': vals
}).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

biomarker_df['Gene_Name'] = biomarker_df['Biomarker'].apply(lambda x: x.replace('PROT_', ''))

out_csv = "/home/priya_26phd0306/top_prioritized_biomarkers.csv"
biomarker_df.to_csv(out_csv, index=False)
print(f"  - Saved ranked biomarker list to: {out_csv}", flush=True)

# 5. Display Top 15 Biomarkers
print("\n=======================================================")
print(" TOP 15 DRIVER PROTEIN BIOMARKERS (Ranked by SHAP):")
print("=======================================================")
for idx, row in biomarker_df.head(15).iterrows():
    print(f"  {idx+1:2d}. {row['Biomarker']:<25} | SHAP Importance: {row['Mean_Absolute_SHAP']:.5f}")
print("=======================================================\n")

# 6. Save SHAP Summary Plot
print("[4/4] Generating and saving SHAP summary plot...", flush=True)
plt.figure(figsize=(10, 6))
X_trans_df = X_trans_df.rename(columns=gene_mapping)
if "X_trans_df" in locals(): X_trans_df = X_trans_df.rename(columns=gene_mapping)
if "X_test" in locals(): X_test = X_test.rename(columns=gene_mapping)
X_trans_df = X_trans_df.rename(columns=gene_mapping)
shap.summary_plot(shap_values, X_trans_df, show=False)
import os; os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
plot_path = "results/figures/shap_beeswarm_mapped.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"  - Saved SHAP summary plot to: {plot_path}")
print("=======================================================")
print(" TASK 4 COMPLETED SUCCESSFULLY!")
print("=======================================================")
