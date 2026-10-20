import os
import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed"

rna_df = pd.read_csv(os.path.join(DATA_DIR, "CPTAC_RNA.csv")).set_index("Patient_ID")
prot_df = pd.read_csv(os.path.join(DATA_DIR, "CPTAC_Proteomics.csv")).set_index("Patient_ID")

biomarkers = [
    'AURKA', 'BUB1B', 'MKI67', 'TOP2A', 'CDK1', 'CCNB1', 'BIRC5', 'MYBL2', 'MCM2',
    'ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1'
]

rna_found = [g for g in biomarkers if g in rna_df.columns]
prot_found = [g for g in biomarkers if g in prot_df.columns]
common_patients = rna_df.index.intersection(prot_df.index)

X_rna = rna_df.loc[common_patients, rna_found]
X_prot = prot_df.loc[common_patients, prot_found]
X_prot.columns = [f"{c}_prot" for c in X_prot.columns]

X_integrated = pd.concat([X_rna, X_prot], axis=1)
y = (X_rna['AURKA'] > X_rna['AURKA'].median()).astype(int)
X_integrated = X_integrated.drop(columns=['AURKA'], errors='ignore')

print("=" * 65)
print("PHASE 5: LEAK-FREE SHAP & LAYER CONTRIBUTION ANALYSIS")
print("=" * 65)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
shap_values_list = []
feature_names = X_integrated.columns

for train_idx, val_idx in skf.split(X_integrated, y):
    X_train, X_val = X_integrated.iloc[train_idx], X_integrated.iloc[val_idx]
    y_train = y.iloc[train_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_names)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_val_scaled)

    # Standardize output shape to 2D matrix (samples x features)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    shap_values_list.append(sv)

all_shap = np.vstack(shap_values_list)

# Mean Absolute SHAP Importance (1D array)
mean_abs_shap = np.abs(all_shap).mean(axis=0).flatten()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Layer': ['Proteomics' if f.endswith('_prot') else 'RNA' for f in feature_names]
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

print("\nTOP 10 FEATURE RANKING (SHAP):")
print(importance_df.head(10).to_string(index=False))

# Calculate Omics Layer Contribution Percentage
layer_summary = importance_df.groupby('Layer')['Mean_Abs_SHAP'].sum()
layer_pct = (layer_summary / layer_summary.sum()) * 100

print("\n" + "-" * 65)
print("OMICS LAYER IMPORTANCE SHARE (%)")
print("-" * 65)
for layer, pct in layer_pct.items():
    bar = "█" * int(pct / 5)
    print(f"• {layer:<12} [{pct:.1f}%] {bar}")

importance_df.to_csv(os.path.join(DATA_DIR, "shap_feature_importance.csv"), index=False)
print(f"\n[+] SHAP feature rankings saved to {DATA_DIR}/shap_feature_importance.csv")
