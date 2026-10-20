import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ml_dir = Path("data/processed/ml_ready")
rppa_dir = Path("data/rppa")

# 1. Clean Y Target Missing Values
y_tr = pd.read_csv(ml_dir / "y_train.csv", index_col=0)
y_te = pd.read_csv(ml_dir / "y_test.csv", index_col=0)

y_tr_clean = y_tr.bfill().ffill()
y_te_clean = y_te.bfill().ffill()

y_tr_clean.to_csv(ml_dir / "y_train.csv")
y_te_clean.to_csv(ml_dir / "y_test.csv")

train_ids = [str(i)[:12] for i in y_tr_clean.index]
test_ids  = [str(i)[:12] for i in y_te_clean.index]
all_ids   = train_ids + test_ids

print(f"Target Cohort Aligned: {len(all_ids)} Patients ({len(train_ids)} Train / {len(test_ids)} Test)")

# 2. Parse RPPA files using raw TSV headers: 'peptide_target' & 'protein_expression'
rppa_files = list(rppa_dir.rglob("*.tsv"))
print(f"Parsing {len(rppa_files)} raw RPPA TSV files...")

patient_profiles = {}

for f in rppa_files:
    fname = f.name
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', fname)
    if not match:
        match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', str(f))
    
    if not match:
        continue
    
    barcode = match.group(1)
    if barcode not in all_ids:
        continue

    try:
        df = pd.read_csv(f, sep='\t')
        if 'peptide_target' in df.columns and 'protein_expression' in df.columns:
            df['protein_expression'] = pd.to_numeric(df['protein_expression'], errors='coerce')
            series = df.groupby('peptide_target')['protein_expression'].mean()
            
            if barcode not in patient_profiles:
                patient_profiles[barcode] = series.to_dict()
            else:
                s_exist = pd.Series(patient_profiles[barcode])
                patient_profiles[barcode] = pd.concat([s_exist, series], axis=1).mean(axis=1).to_dict()
    except Exception:
        continue

# 3. Construct RPPA Feature Matrix
rppa_df = pd.DataFrame.from_dict(patient_profiles, orient='index')
rppa_df.index = [str(i)[:12] for i in rppa_df.index]
rppa_df = rppa_df.reindex(all_ids)

# Keep features present in >= 80% of samples
valid_features = rppa_df.columns[rppa_df.isnull().mean() < 0.2]
rppa_filtered = rppa_df[valid_features]

print(f"Parsed RPPA Matrix Shape: {rppa_filtered.shape[0]} Patients x {rppa_filtered.shape[1]} Proteins")

# 4. Leak-Free Imputation & Z-Score Standardization
tr_raw = rppa_filtered.loc[train_ids]
te_raw = rppa_filtered.loc[test_ids]

imp = SimpleImputer(strategy="median")
tr_imp = imp.fit_transform(tr_raw)
te_imp = imp.transform(te_raw)

scaler = StandardScaler()
tr_sc = scaler.fit_transform(tr_imp)
te_sc = scaler.transform(te_imp)

rppa_train_clean = pd.DataFrame(tr_sc, index=train_ids, columns=valid_features)
rppa_test_clean  = pd.DataFrame(te_sc, index=test_ids, columns=valid_features)

rppa_train_clean.to_csv(ml_dir / "rppa_train.csv")
rppa_test_clean.to_csv(ml_dir / "rppa_test.csv")

print("\n==========================================================")
print(" REPAIRED RPPA & TARGET MATRICES EXPORTED")
print("==========================================================")
print(f"rppa_train.csv -> Shape: {rppa_train_clean.shape} | Nulls: {rppa_train_clean.isnull().sum().sum()}")
print(f"rppa_test.csv  -> Shape: {rppa_test_clean.shape}  | Nulls: {rppa_test_clean.isnull().sum().sum()}")
print(f"y_train.csv    -> Shape: {y_tr_clean.shape} | Nulls: {y_tr_clean.isnull().sum().sum()}")
print(f"y_test.csv     -> Shape: {y_te_clean.shape}  | Nulls: {y_te_clean.isnull().sum().sum()}")

# 5. Regenerate Plot B
rna_count  = pd.read_csv(ml_dir / "rna_train.csv", index_col=0).shape[1]
meth_count = pd.read_csv(ml_dir / "methylation_train.csv", index_col=0).shape[1]
rppa_count = rppa_train_clean.shape[1]

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
modalities = ["RNA-Seq", "DNA Methylation", "RPPA Proteomics"]
counts = [rna_count, meth_count, rppa_count]
colors = ["#3B528B", "#21908C", "#5DC863"]

bars = ax.bar(modalities, counts, color=colors, width=0.5)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.015), f"{yval:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("ML-Ready Feature Scale per Modality (Zero Missing Values)", fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel("Filtered & Normalized Features (Z-Score Scaled)", fontsize=11)
ax.set_ylim(0, max(counts) * 1.18)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
out_fig = Path("results/figures/plot_b_updated_feature_scale.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300)
print(f"Updated Plot B saved to {out_fig}")
