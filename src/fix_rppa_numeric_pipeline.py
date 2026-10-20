import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Paths & Cohort Setup
ml_dir = Path("data/processed/ml_ready")
rppa_dir = Path("data/rppa")
meta_path = Path("data/rppa_metadata.csv")

y_train = pd.read_csv(ml_dir / "y_train.csv", index_col=0)
y_test  = pd.read_csv(ml_dir / "y_test.csv", index_col=0)

train_ids = [str(i)[:12] for i in y_train.index]
test_ids  = [str(i)[:12] for i in y_test.index]
all_ids   = train_ids + test_ids

print(f"Target Cohort: {len(all_ids)} Patients ({len(train_ids)} Train / {len(test_ids)} Test)")

# 2. Build File-to-Barcode Map from Metadata
file_to_barcode = {}
if meta_path.exists():
    meta_df = pd.read_csv(meta_path)
    for _, row in meta_df.iterrows():
        fname = str(row.get('file_name', ''))
        # Look across potential metadata columns for TCGA patient barcode
        barcode = ""
        for col in meta_df.columns:
            val = str(row[col])
            if "TCGA-" in val:
                idx = val.find("TCGA-")
                barcode = val[idx:idx+12]
                break
        if fname and barcode:
            file_to_barcode[fname] = barcode

# 3. Inspect Raw File Structure & Parse Antibodies
raw_files = list(rppa_dir.rglob("*.tsv")) + list(rppa_dir.rglob("*.txt")) + list(rppa_dir.rglob("*.csv"))
print(f"Found {len(raw_files)} RPPA files in {rppa_dir}")

patient_data = {}

for fpath in raw_files:
    fname = fpath.name
    barcode = file_to_barcode.get(fname, None)
    
    # Fallback search inside file path string
    if not barcode:
        for part in fpath.parts + (fname,):
            if "TCGA-" in part:
                idx = part.find("TCGA-")
                barcode = part[idx:idx+12]
                break

    if not barcode or barcode not in all_ids:
        continue

    try:
        # Read file with delimiter auto-detection
        df = pd.read_csv(fpath, sep=None, engine='python', comment='#')
        
        # Identify Protein Identifier column and Numeric Value column
        prot_col = None
        val_col = None
        
        for c in df.columns:
            c_lower = str(c).lower()
            if any(k in c_lower for k in ['composite', 'antibody', 'protein', 'gene', 'ref', 'id']):
                prot_col = c
            elif any(k in c_lower for k in ['value', 'expression', 'intensity', 'log2', 'abundance', 'signal']):
                val_col = c
                
        if prot_col is None:
            prot_col = df.columns[0]
        if val_col is None and len(df.columns) > 1:
            val_col = df.columns[1]

        if prot_col and val_col:
            # Force numeric conversion for protein expression values
            s = df.set_index(prot_col)[val_col]
            s_numeric = pd.to_numeric(s, errors='coerce')
            patient_data[barcode] = s_numeric.to_dict()

    except Exception:
        continue

# 4. Construct Numeric Feature Matrix
rppa_matrix = pd.DataFrame.from_dict(patient_data, orient='index')
rppa_matrix.index = [str(i)[:12] for i in rppa_matrix.index]
rppa_matrix = rppa_matrix.reindex(all_ids)

# Convert all matrix elements explicitly to float numeric values
rppa_matrix = rppa_matrix.apply(pd.to_numeric, errors='coerce')

# Drop columns/antibodies with more than 30% missing values across the cohort
valid_antibodies = rppa_matrix.columns[rppa_matrix.isnull().mean() < 0.3]
rppa_filtered = rppa_matrix[valid_antibodies]

print(f"Parsed Numeric RPPA Matrix Shape (Matched Cohort): {rppa_filtered.shape}")

# 5. Leak-Free Imputation & Standardization (Fitted on Train set only)
rppa_train_raw = rppa_filtered.loc[train_ids]
rppa_test_raw  = rppa_filtered.loc[test_ids]

imputer = SimpleImputer(strategy="median")
train_imp = imputer.fit_transform(rppa_train_raw)
test_imp  = imputer.transform(rppa_test_raw)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imp)
test_scaled  = scaler.transform(test_imp)

# 6. Save Clean Numeric Datasets
rppa_train_clean = pd.DataFrame(train_scaled, index=train_ids, columns=valid_antibodies, dtype=np.float64)
rppa_test_clean  = pd.DataFrame(test_scaled, index=test_ids, columns=valid_antibodies, dtype=np.float64)

rppa_train_clean.to_csv(ml_dir / "rppa_train.csv")
rppa_test_clean.to_csv(ml_dir / "rppa_test.csv")

print("\n==========================================================")
print(" NUMERIC RPPA MATRICES PROCESSED & EXPORTED")
print("==========================================================")
print(f"rppa_train.csv -> Shape: {rppa_train_clean.shape} | Missing Values: {rppa_train_clean.isnull().sum().sum()} | Numeric Dtype: {rppa_train_clean.dtypes.iloc[0]}")
print(f"rppa_test.csv  -> Shape: {rppa_test_clean.shape}  | Missing Values: {rppa_test_clean.isnull().sum().sum()} | Numeric Dtype: {rppa_test_clean.dtypes.iloc[0]}")

# 7. Regenerate Plot B
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
