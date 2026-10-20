import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

print("--- Step 1: Loading & Aligning Matched Multi-Omic (RNA + Proteomics) Data ---")
rna_path = "data/processed/CPTAC_RNA.csv"
prot_path = "data/processed/CPTAC_Proteomics.csv" if os.path.exists("data/processed/CPTAC_Proteomics.csv") else "data/processed/CPTAC_Prot.csv"
clinical_path = "cptac_119_clinical_final.csv"

rna_df = pd.read_csv(rna_path)
clinical_df = pd.read_csv(clinical_path)

rna_id = next(c for c in rna_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])
clinical_id = next(c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])

rna_df['merge_id'] = rna_df[rna_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()

if os.path.exists(prot_path):
    prot_df = pd.read_csv(prot_path)
    prot_id = next(c for c in prot_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id'])
    prot_df['merge_id'] = prot_df[prot_id].astype(str).str.strip()
    merged_df = pd.merge(rna_df, prot_df, on='merge_id', suffixes=('_rna', '_prot'))
    merged_df = pd.merge(merged_df, clinical_df, on='merge_id')
    print(f"Successfully paired RNA and Proteomics for {len(merged_df)} samples.")
else:
    merged_df = pd.merge(rna_df, clinical_df, on='merge_id')
    print("Proteomics file not found; using RNA dataset.")

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']
target_genes = prolif_genes + luminal_genes

rna_cols = [c for c in merged_df.columns if any(c == g or c == f"{g}_rna" for g in target_genes)]
prot_cols = [c for c in merged_df.columns if any(c == f"{g}_prot" for g in target_genes)]

print(f"Features mapped -> RNA: {len(rna_cols)} | Proteomics: {len(prot_cols)}")

# Target selection avoiding false substring matches
target_col = None
for col in clinical_df.columns:
    c_low = col.lower().strip()
    if c_low in ['merge_id', 'patient_id', 'sample_id']:
        continue
    words = c_low.replace("_", " ").split()
    if any(k in c_low for k in ['er status', 'er_status', 'pam50', 'subtype']) or 'er' in words:
        if "ancestry" not in c_low:
            target_col = col
            break

if target_col and target_col in merged_df.columns:
    print(f"Selected clinical target: '{target_col}'")
    val_series = merged_df[target_col].astype(str).str.lower().str.strip()
    # Binarize Luminal/ER+ vs Non-Luminal/ER-
    y = np.where(val_series.str.contains('pos|1|yes|luminal|luma|lumb'), 1, 0)
else:
    print("Clinical target column not found; using transcriptomic ESR1 status.")
    esr1_col = next(c for c in rna_cols if 'ESR1' in c)
    y = (merged_df[esr1_col] >= merged_df[esr1_col].median()).astype(int).values

print(f"Target Label Distribution -> Class 0: {(y==0).sum()} | Class 1: {(y==1).sum()}")

scaler_rna = StandardScaler()
X_rna = pd.DataFrame(scaler_rna.fit_transform(merged_df[rna_cols].select_dtypes(include=[np.number])), columns=rna_cols)

if len(prot_cols) > 0:
    scaler_prot = StandardScaler()
    X_prot = pd.DataFrame(scaler_prot.fit_transform(merged_df[prot_cols].select_dtypes(include=[np.number])), columns=prot_cols)
    X_multi = pd.concat([X_rna, X_prot], axis=1)
    modality_dict = {
        "RNA Only": X_rna,
        "Proteomics Only": X_prot,
        "Integrated Multi-Omics (RNA+Prot)": X_multi
    }
else:
    modality_dict = {"RNA Only": X_rna}

print("--- Step 2: Stratified 5-Fold Cross-Validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary_results = []

for mod_name, X_data in modality_dict.items():
    auc_list, pr_auc_list = [], []
    for train_idx, test_idx in skf.split(X_data, y):
        X_tr, X_te = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(C=0.5, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        
        auc_list.append(roc_auc_score(y_te, probs))
        pr_auc_list.append(average_precision_score(y_te, probs))
        
    summary_results.append({
        "Modality": mod_name,
        "N_Features": X_data.shape[1],
        "ROC_AUC_Mean": round(np.mean(auc_list), 4),
        "ROC_AUC_Std": round(np.std(auc_list), 4),
        "PR_AUC_Mean": round(np.mean(pr_auc_list), 4),
        "PR_AUC_Std": round(np.std(pr_auc_list), 4)
    })

res_df = pd.DataFrame(summary_results)
os.makedirs("results", exist_ok=True)
res_df.to_csv("results/task12_multiomics_validation.csv", index=False)

print("\n=== Task 12: Integrated CPTAC Multi-Omics Benchmarking ===")
print(res_df.to_string(index=False))
