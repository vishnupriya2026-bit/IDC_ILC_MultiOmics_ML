import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif, SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

print("==========================================================")
print("  MINIMUM VIABLE PIPELINE (MVP) STATISTICAL VALIDATION")
print("==========================================================")

# 1. Load Clinical & Omics Data
clinical = pd.read_csv("cptac_119_clinical_final.csv", index_col=0)
labeled_clinical = clinical.dropna(subset=['Target_Stage_Binary']).copy()
labeled_clinical['Target_Stage_Binary'] = labeled_clinical['Target_Stage_Binary'].astype(int)
labeled_clinical = labeled_clinical[~labeled_clinical.index.duplicated(keep='first')]

y = labeled_clinical['Target_Stage_Binary']
patient_ids = set(labeled_clinical.index.tolist())

rna_path = "BreastCancer_MultiOmics/data/processed/CPTAC_RNA.csv"
prot_path = "BreastCancer_MultiOmics/data/processed/CPTAC_Proteomics.csv"

def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    if len(patient_ids.intersection(set(df.columns.astype(str)))) > len(patient_ids.intersection(set(df.index.astype(str)))):
        df = df.T
    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.duplicated(keep='first')]
    return df

df_rna = load_matrix(rna_path)
df_prot = load_matrix(prot_path)

matched_patients = [p for p in labeled_clinical.index if p in df_rna.index and p in df_prot.index]
y_matched = y.loc[matched_patients].values

df_rna_m = df_rna.loc[matched_patients]
df_prot_m = df_prot.loc[matched_patients]

print(f"Matched Cohort Size: {len(matched_patients)} samples")

# Impute omics matrices
imp = SimpleImputer(strategy='median')
X_rna_imp = imp.fit_transform(df_rna_m.values)
X_prot_imp = imp.fit_transform(df_prot_m.values)
X_multiomics = np.hstack([X_rna_imp, X_prot_imp])

# CV Evaluation Function
def evaluate_cv(X, y_labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_trues, y_probs = [], []
    
    for train_idx, test_idx in skf.split(X, y_labels):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_labels[train_idx], y_labels[test_idx]
        
        skb = SelectKBest(f_classif, k=200)
        X_tr_k = skb.fit_transform(X_tr, y_tr)
        X_te_k = skb.transform(X_te)
        
        sfm = SelectFromModel(LogisticRegression(penalty='l1', C=0.3, solver='liblinear', random_state=42), max_features=30)
        X_tr_s = sfm.fit_transform(X_tr_k, y_tr)
        X_te_s = sfm.transform(X_te_k)
        
        clf = SVC(probability=True, kernel='rbf', random_state=42)
        clf.fit(X_tr_s, y_tr)
        
        probs = clf.predict_proba(X_te_s)[:, 1]
        y_trues.extend(y_te)
        y_probs.extend(probs)
        
    return roc_auc_score(y_trues, y_probs), np.array(y_trues), np.array(y_probs)

# Step 1: Real Model Evaluation & Bootstrap CIs
actual_auc, y_true_all, y_prob_all = evaluate_cv(X_multiomics, y_matched)
print(f"\n[1/3] Real Model Cross-Validated ROC-AUC: {actual_auc:.4f}")

n_bootstraps = 500
boot_aucs = []
rng = np.random.RandomState(42)

for _ in range(n_bootstraps):
    idx = rng.randint(0, len(y_true_all), len(y_true_all))
    if len(np.unique(y_true_all[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_true_all[idx], y_prob_all[idx]))

ci_lower = np.percentile(boot_aucs, 2.5)
ci_upper = np.percentile(boot_aucs, 97.5)
print(f"      95% Confidence Interval for AUC: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Step 2: Permutation Null Distribution Test
print("\n[2/3] Running 50 Label-Permutation Null Distribution Tests...")
null_aucs = []
for i in range(50):
    y_perm = np.random.permutation(y_matched)
    null_auc, _, _ = evaluate_cv(X_multiomics, y_perm)
    null_aucs.append(null_auc)

p_value = (np.sum(np.array(null_aucs) >= actual_auc) + 1) / (len(null_aucs) + 1)
print(f"      Mean Null Distribution AUC: {np.mean(null_aucs):.4f}")
print(f"      Permutation Test Empirical p-value: {p_value:.4f}")

# Step 3: Cross-Omics Target Concordance (RNA vs Protein)
print("\n[3/3] Evaluating Cross-Omics Concordance (RNA vs Proteomics Spearman Correlation)...")
key_genes = ['ESR1', 'EGFR', 'PIK3CA', 'ERBB2', 'ACP5']
concordance_results = []

rna_cols_upper = {str(col).strip().upper(): col for col in df_rna_m.columns}
prot_cols_upper = {str(col).strip().upper(): col for col in df_prot_m.columns}

for g in key_genes:
    if g in rna_cols_upper and g in prot_cols_upper:
        rna_vals = df_rna_m[rna_cols_upper[g]].values
        prot_vals = df_prot_m[prot_cols_upper[g]].values
        
        valid_mask = ~np.isnan(rna_vals) & ~np.isnan(prot_vals)
        if np.sum(valid_mask) > 10:
            rho, p_val = spearmanr(rna_vals[valid_mask], prot_vals[valid_mask])
            concordance_results.append({'Gene': g, 'Spearman_Rho': round(rho, 4), 'p_value': round(p_val, 6)})

df_conc = pd.DataFrame(concordance_results)
print("\n--- Target Gene Concordance ---")
print(df_conc.to_string(index=False))

# Step 4: Plot Figure
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(null_aucs, kde=True, ax=axes[0], color='gray', stat='density', alpha=0.6)
axes[0].axvline(actual_auc, color='red', linestyle='--', linewidth=2.5, label=f'Model AUC ({actual_auc:.4f})')
axes[0].set_title(f"Permutation Null Distribution\nEmpirical p-value = {p_value:.4f}", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Cross-Validated ROC-AUC", fontsize=10)
axes[0].set_ylabel("Density", fontsize=10)
axes[0].legend(loc='upper right')

if not df_conc.empty:
    sns.barplot(data=df_conc, x='Spearman_Rho', y='Gene', color='teal', ax=axes[1])
    axes[1].set_title("Cross-Omics Concordance (RNA vs Protein)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Spearman Correlation (ρ)", fontsize=10)
    axes[1].set_ylabel("Target Gene", fontsize=10)
    axes[1].set_xlim(-0.2, 1.0)
    for i, row in df_conc.iterrows():
        axes[1].text(row['Spearman_Rho'] + 0.02, i, f"ρ={row['Spearman_Rho']:.2f}", va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("fig_mvp_validation_metrics.png", dpi=300)
plt.close()
print("\nSaved MVP validation figure to 'fig_mvp_validation_metrics.png'")
