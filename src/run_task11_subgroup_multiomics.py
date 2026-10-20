import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

print("--- Step 1: Parsing GSE2034 Relapse Outcome & ER Status ---")
series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
sample_ids = []
overall_relapse_line = None

with open(series_matrix_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        break

for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        if parts:
            tag = parts[0].lower()
            if "relapse" in tag and "time" not in tag and not any(s in tag for s in ["bone", "brain", "lung", "loco"]):
                overall_relapse_line = parts
                break

if overall_relapse_line is None:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
            if parts and "relapse" in parts[0].lower() and "time" not in parts[0].lower():
                overall_relapse_line = parts
                break

if overall_relapse_line is None:
    raise ValueError("Could not find overall relapse outcome line in GSE2034 series matrix.")

y_ext_list = [1 if x.split(":")[-1].strip().lower() in ["1", "yes", "relapse"] else 0 for x in overall_relapse_line[:len(sample_ids)]]
meta_df = pd.DataFrame({"sample_id": sample_ids[:len(y_ext_list)], "target": y_ext_list}).set_index("sample_id")

print(f"Successfully loaded metadata for {len(meta_df)} samples | Total Relapse Events: {meta_df['target'].sum()}")

print("--- Step 2: Loading Microarray Data & Model Scoring ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
meta_df = meta_df.loc[gse_clean.index]

if "ESR1" in gse_clean.columns:
    meta_df['er_status'] = (gse_clean['ESR1'] >= gse_clean['ESR1'].median()).astype(int)
    print(f"Transcriptomic ER Stratification (ESR1) -> ER+: {(meta_df['er_status'] == 1).sum()} | ER-: {(meta_df['er_status'] == 0).sum()}")
else:
    meta_df['er_status'] = 1

cptac_df = pd.read_csv("data/processed/CPTAC_RNA.csv")
clinical_df = pd.read_csv("cptac_119_clinical_final.csv")

cptac_id = next((c for c in cptac_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), cptac_df.columns[0])
clinical_id = next((c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), clinical_df.columns[0])

cptac_df['merge_id'] = cptac_df[cptac_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()
merged_cptac = pd.merge(cptac_df, clinical_df, on='merge_id', how='inner')

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']

X_cptac_raw = merged_cptac.drop(columns=clinical_df.columns.tolist() + [cptac_id, 'merge_id'], errors="ignore").select_dtypes(include=[np.number])
shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))

p_genes = [g for g in prolif_genes if g in shared_genes]
l_genes = [g for g in luminal_genes if g in shared_genes]

scaler_cptac = StandardScaler()
scaler_gse = StandardScaler()

X_cptac_std = pd.DataFrame(scaler_cptac.fit_transform(X_cptac_raw[shared_genes]), columns=shared_genes)
X_gse_std = pd.DataFrame(scaler_gse.fit_transform(gse_clean[shared_genes]), columns=shared_genes)

risk_index_cptac = X_cptac_std[p_genes].mean(axis=1) - X_cptac_std[l_genes].mean(axis=1)
y_train = (risk_index_cptac >= risk_index_cptac.median()).astype(int).values

feature_panel = p_genes + l_genes
clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
clf.fit(X_cptac_std[feature_panel], y_train)

y_probs = clf.predict_proba(X_gse_std[feature_panel])[:, 1]
meta_df['prob'] = y_probs

print("--- Step 3: ER Sub-group Stratification Analysis ---")
subgroup_results = []
for er_val, er_label in [(1, "ER Positive (ER+)"), (0, "ER Negative (ER-)")]:
    sub_df = meta_df[meta_df['er_status'] == er_val]
    if len(sub_df) > 10 and sub_df['target'].nunique() > 1:
        auc = roc_auc_score(sub_df['target'], sub_df['prob'])
        pr_auc = average_precision_score(sub_df['target'], sub_df['prob'])
        subgroup_results.append({
            "Subgroup": er_label,
            "N_Samples": len(sub_df),
            "Relapse_Events": int(sub_df['target'].sum()),
            "ROC_AUC": round(auc, 4),
            "PR_AUC": round(pr_auc, 4)
        })

subgroup_df = pd.DataFrame(subgroup_results)
os.makedirs("results", exist_ok=True)
subgroup_df.to_csv("results/task11_er_subgroup_stratification.csv", index=False)

print("\n=== Task 11: ER Status Sub-group Performance ===")
print(subgroup_df.to_string(index=False))
