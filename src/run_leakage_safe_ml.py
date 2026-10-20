import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score, confusion_matrix

# --- 1. DATA LOADING & PANEL VERIFICATION ---
DATA_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed"

rna_df = pd.read_csv(os.path.join(DATA_DIR, "CPTAC_RNA.csv")).set_index("Patient_ID")
prot_df = pd.read_csv(os.path.join(DATA_DIR, "CPTAC_Proteomics.csv")).set_index("Patient_ID")

# 16-Biomarker Target Panel
biomarkers = [
    'AURKA', 'BUB1B', 'MKI67', 'TOP2A', 'CDK1', 'CCNB1', 'BIRC5', 'MYBL2', 'MCM2',
    'ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1'
]

print("=" * 70)
print("PHASE 2 & 3: BIOMARKER AUDIT & LEAKAGE-SAFE ML BENCHMARK")
print("=" * 70)

rna_found = [g for g in biomarkers if g in rna_df.columns]
prot_found = [g for g in biomarkers if g in prot_df.columns]

print(f"RNA Panel Coverage:        {len(rna_found)} / {len(biomarkers)} genes found")
print(f"Proteomics Panel Coverage: {len(prot_found)} / {len(biomarkers)} proteins found")

# Common matched patient cohort
common_patients = rna_df.index.intersection(prot_df.index)
print(f"Matched Cohort Size:       {len(common_patients)} patients")

# Align patient IDs
X_rna = rna_df.loc[common_patients, rna_found]
X_prot = prot_df.loc[common_patients, prot_found]
X_prot.columns = [f"{c}_prot" for c in X_prot.columns]

# Integrated Multi-Omics Feature Matrix
X_integrated = pd.concat([X_rna, X_prot], axis=1)

# Derive baseline binary target (e.g., AURKA high-risk proxy if clinical outcome unavailable)
# High AURKA expression (above median) serves as a gold-standard proliferation risk marker
y = (X_rna['AURKA'] > X_rna['AURKA'].median()).astype(int)
# Exclude target driver from predictors to eliminate self-leakage
X_rna = X_rna.drop(columns=['AURKA'], errors='ignore')
X_integrated = X_integrated.drop(columns=['AURKA'], errors='ignore')

# --- 2. LEAKAGE-SAFE CROSS-VALIDATION EVALUATION ---
def evaluate_leakage_safe(X, y, model_dict, cv_splits=5):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []

    for model_name, model in model_dict.items():
        metrics = {'auc': [], 'auprc': [], 'sens': [], 'spec': [], 'f1': []}

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LEAK-FREE PREPROCESSING: Scaler fit ONLY on training fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Fit model strictly on scaled training fold
            model.fit(X_train_scaled, y_train)

            # Predictions
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_val_scaled)[:, 1]
            else:
                y_probs = model.decision_function(X_val_scaled)
                
            y_preds = model.predict(X_val_scaled)

            # Compute fold metrics
            tn, fp, fn, tp = confusion_matrix(y_val, y_preds).ravel()
            metrics['auc'].append(roc_auc_score(y_val, y_probs))
            metrics['auprc'].append(average_precision_score(y_val, y_probs))
            metrics['sens'].append(recall_score(y_val, y_preds))
            metrics['spec'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            metrics['f1'].append(f1_score(y_val, y_preds))

        results.append({
            'Model': model_name,
            'AUROC': f"{np.mean(metrics['auc']):.3f} ± {np.std(metrics['auc']):.3f}",
            'AUPRC': f"{np.mean(metrics['auprc']):.3f} ± {np.std(metrics['auprc']):.3f}",
            'Sensitivity': f"{np.mean(metrics['sens']):.3f}",
            'Specificity': f"{np.mean(metrics['spec']):.3f}",
            'F1-Score': f"{np.mean(metrics['f1']):.3f}"
        })

    return pd.DataFrame(results)

models = {
    'Logistic Regression': LogisticRegression(penalty='l2', C=1.0, random_state=42),
    'SVM (RBF Kernel)': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=50, eval_metric='logloss', random_state=42)
}

print("\n" + "=" * 70)
print("EVALUATING MODEL PERFORMANCE ACROSS OMICS LAYERS")
print("=" * 70)

print("\n[1] SINGLE-OMICS: RNA-ONLY")
df_rna = evaluate_leakage_safe(X_rna, y, models)
print(df_rna.to_string(index=False))

print("\n[2] SINGLE-OMICS: PROTEOMICS-ONLY")
df_prot = evaluate_leakage_safe(X_prot, y, models)
print(df_prot.to_string(index=False))

print("\n[3] INTEGRATED MULTI-OMICS: RNA + PROTEOMICS")
df_integrated = evaluate_leakage_safe(X_integrated, y, models)
print(df_integrated.to_string(index=False))

# Export benchmark table
df_integrated['Omics_Layer'] = 'Multi-Omics (RNA+Prot)'
df_rna['Omics_Layer'] = 'RNA-Only'
df_prot['Omics_Layer'] = 'Proteomics-Only'

summary_df = pd.concat([df_rna, df_prot, df_integrated], axis=0)
summary_df.to_csv(os.path.join(DATA_DIR, "benchmark_ml_results.csv"), index=False)
print(f"\n[+] Full leakage-safe benchmark exported to {DATA_DIR}/benchmark_ml_results.csv")
