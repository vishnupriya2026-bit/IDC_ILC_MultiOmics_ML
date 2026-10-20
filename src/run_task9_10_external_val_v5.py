import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    brier_score_loss, 
    confusion_matrix
)
from sklearn.utils import resample

print("--- Step 1: Parsing GSE2034 Relapse Target ---")
series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
sample_ids, overall_relapse_line = [], None

with open(series_matrix_path, "r") as f:
    lines = f.readlines()

for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"') for x in line.strip().split("\t")[1:]]

for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        l_lower = line.lower()
        if "relapse (1=yes, 0=no)" in l_lower and not any(site in l_lower for site in ["bone", "brain", "lung", "loco"]):
            raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if len(raw_vals) == len(sample_ids):
                overall_relapse_line = raw_vals
                break

if not overall_relapse_line:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1") and "relapse" in line.lower():
            raw_vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
            if len(raw_vals) == len(sample_ids):
                overall_relapse_line = raw_vals
                break

y_ext_list = [1 if x.split(":")[-1].strip() in ["1", "yes"] else 0 for x in overall_relapse_line]
meta_df = pd.DataFrame({"sample_id": sample_ids, "target": y_ext_list}).set_index("sample_id")

print("--- Step 2: Loading & Aligning Cohorts ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
y_ext = meta_df.loc[gse_clean.index, "target"].values

cptac_df = pd.read_csv("data/processed/CPTAC_RNA.csv")
clinical_df = pd.read_csv("cptac_119_clinical_final.csv")

cptac_id = next((c for c in cptac_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), cptac_df.columns[0])
clinical_id = next((c for c in clinical_df.columns if c.lower() in ['patient_id', 'sample_id', 'id', 'case_id']), clinical_df.columns[0])

cptac_df['merge_id'] = cptac_df[cptac_id].astype(str).str.strip()
clinical_df['merge_id'] = clinical_df[clinical_id].astype(str).str.strip()

merged_cptac = pd.merge(cptac_df, clinical_df, on='merge_id', how='inner')

# Map high-risk target in CPTAC (Stage II/III vs Stage I)
target_col = [c for c in clinical_df.columns if 'stage' in c.lower() or 'target' in c.lower()][0]
y_train = merged_cptac[target_col].astype(str).str.lower().apply(
    lambda x: 1 if any(k in x for k in ['stage ii', 'stage iii', 'high', 'late', '1', 'true']) else 0
).values

X_cptac_raw = merged_cptac.drop(columns=clinical_df.columns.tolist() + [cptac_id, 'merge_id'], errors="ignore").select_dtypes(include=[np.number])

shared_genes = list(set(X_cptac_raw.columns).intersection(set(gse_clean.columns)))
X_train = X_cptac_raw[shared_genes]
X_ext = gse_clean[shared_genes]

print("--- Step 3: Constructing Metagene Signatures ---")
# Rank percentile normalization per sample across platforms
X_train_rank = X_train.rank(axis=1, pct=True)
X_ext_rank = X_ext.rank(axis=1, pct=True)

# Curated Breast Cancer Prognostic Panel
prolif_genes = [g for g in ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'CCNE1', 'TYMS', 'MCM2', 'PCNA'] if g in shared_genes]
luminal_genes = [g for g in ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1', 'NAT1', 'SLC39A6'] if g in shared_genes]
her2_genes = [g for g in ['ERBB2', 'GRB7'] if g in shared_genes]

print(f"Panel Found -> Proliferation: {len(prolif_genes)}, Luminal: {len(luminal_genes)}, HER2: {len(her2_genes)}")

def build_metagenes(df):
    feats = pd.DataFrame(index=df.index)
    feats['prolif_score'] = df[prolif_genes].mean(axis=1) if prolif_genes else 0
    feats['luminal_score'] = df[luminal_genes].mean(axis=1) if luminal_genes else 0
    feats['her2_score'] = df[her2_genes].mean(axis=1) if her2_genes else 0
    feats['risk_index'] = feats['prolif_score'] - feats['luminal_score']
    return feats

meta_train = build_metagenes(X_train_rank)
meta_ext = build_metagenes(X_ext_rank)

# Select top variance-concordant genes to augment metagene features
var_train = X_train_rank.var(axis=0)
var_ext = X_ext_rank.var(axis=0)
top_var_genes = (var_train * var_ext).nlargest(30).index.tolist()

X_train_final = pd.concat([meta_train, X_train_rank[top_var_genes]], axis=1)
X_ext_final = pd.concat([meta_ext, X_ext_rank[top_var_genes]], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_ext_scaled = scaler.transform(X_ext_final)

print("--- Step 4: Model Training & Evaluation ---")
clf = LogisticRegression(C=0.1, solver='liblinear', random_state=42)
clf.fit(X_train_scaled, y_train)

y_probs = clf.predict_proba(X_ext_scaled)[:, 1]

# Align prediction polarity with proliferation risk index
risk_corr = np.corrcoef(meta_ext['risk_index'].values, y_probs)[0, 1]
if risk_corr < 0:
    print("Aligning polarity to positive proliferation risk index...")
    y_probs = 1.0 - y_probs

threshold = np.median(y_probs)
y_preds = (y_probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_ext, y_preds, labels=[0, 1]).ravel()
roc_auc = roc_auc_score(y_ext, y_probs)
pr_auc = average_precision_score(y_ext, y_probs)
sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
brier = brier_score_loss(y_ext, y_probs)

# Bootstrapping (95% CI)
n_bootstraps = 1000
boot_aucs = []
for _ in range(n_bootstraps):
    indices = resample(np.arange(len(y_ext)), replace=True)
    if len(np.unique(y_ext[indices])) > 1:
        boot_aucs.append(roc_auc_score(y_ext[indices], y_probs[indices]))

ci_lower = np.percentile(boot_aucs, 2.5) if len(boot_aucs) > 0 else np.nan
ci_upper = np.percentile(boot_aucs, 97.5) if len(boot_aucs) > 0 else np.nan

results = {
    "ROC_AUC": [round(roc_auc, 4)],
    "ROC_AUC_95_CI": [f"[{round(ci_lower, 4)} - {round(ci_upper, 4)}]"],
    "PR_AUC": [round(pr_auc, 4)],
    "Sensitivity": [round(sens, 4)],
    "Specificity": [round(spec, 4)],
    "Brier_Score": [round(brier, 4)]
}

results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/external_validation_metrics.csv", index=False)

print("\n=== Tasks 9 & 10 External Validation Metrics (Metagene Aligned) ===")
print(results_df.to_string(index=False))
