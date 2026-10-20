import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Setup Paths
ml_dir = Path("data/processed/ml_ready")
rppa_dir = Path("data/rppa")
meta_path = Path("data/rppa_metadata.csv")

# Load Target Splits (N=410: 328 Train / 82 Test)
y_train = pd.read_csv(ml_dir / "y_train.csv", index_col=0)
y_test  = pd.read_csv(ml_dir / "y_test.csv", index_col=0)

train_ids = y_train.index.tolist()
test_ids  = y_test.index.tolist()
all_ids   = train_ids + test_ids

print(f"Cohort Loaded: {len(all_ids)} Patients ({len(train_ids)} Train / {len(test_ids)} Test)")

# 2. Parse Raw GDC RPPA Files
file_to_barcode = {}
if meta_path.exists():
    meta_df = pd.read_csv(meta_path)
    for _, row in meta_df.iterrows():
        fname = str(row.get('file_name', ''))
        barcode = str(row.get('cases.0.submitter_id', row.get('submitter_id', '')))[:12]
        if fname and barcode.startswith("TCGA"):
            file_to_barcode[fname] = barcode

# Locate all TSV/TXT/CSV files inside data/rppa/
raw_files = list(rppa_dir.rglob("*.tsv")) + list(rppa_dir.rglob("*.txt")) + list(rppa_dir.rglob("*.csv"))
print(f"Located {len(raw_files)} potential RPPA files in {rppa_dir}")

patient_rppa = {}

for fpath in raw_files:
    fname = fpath.name
    # Identify sample barcode from metadata or path
    barcode = file_to_barcode.get(fname, None)
    if not barcode:
        for part in fpath.parts + (fname,):
            if part.startswith("TCGA-"):
                barcode = part[:12]
                break

    if not barcode or not barcode.startswith("TCGA"):
        continue

    if barcode not in all_ids:
        continue

    try:
        df = pd.read_csv(fpath, sep='\t')
        if df.shape[1] < 2:
            df = pd.read_csv(fpath, sep=',')

        # Standard GDC RPPA headers: Composite Element REF / Abundance
        protein_col = [c for c in df.columns if 'Composite' in c or 'Antibody' in c or 'Gene' in c or 'Protein' in c]
        val_col     = [c for c in df.columns if 'Value' in c or 'Log2' in c or 'Intensity' in c or 'Expression' in c]

        if protein_col and val_col:
            series = df.set_index(protein_col[0])[val_col[0]]
            patient_rppa[barcode] = series.to_dict()
    except Exception:
        continue

# 3. Construct Unified RPPA Matrix
if len(patient_rppa) == 0:
    print("Warning: Direct file parsing returned 0 matches. Generating synthetic matrix aligned with cohort...")
    # Alternative fallback if files are pre-merged in data/
    alt_files = list(Path("data").glob("*rppa*.csv")) + list(Path("data").glob("*rppa*.tsv"))
    for af in alt_files:
        try:
            df_alt = pd.read_csv(af, index_col=0)
            df_alt.index = df_alt.index.str.slice(0, 12)
            matched = df_alt.reindex(all_ids)
            if matched.dropna(how='all').shape[0] > 0:
                patient_rppa = matched.T.to_dict()
                break
        except Exception:
            continue

rppa_matrix = pd.DataFrame.from_dict(patient_rppa, orient='index')
rppa_matrix = rppa_matrix.reindex(all_ids)

print(f"Raw Constructed RPPA Matrix Shape: {rppa_matrix.shape}")

# 4. Missing Value Filtering & Leak-Free Standardization
# Keep antibodies present in > 70% of cohort samples
valid_cols = rppa_matrix.columns[rppa_matrix.isnull().mean() < 0.3]
rppa_filtered = rppa_matrix[valid_cols]

# Split Train / Test
rppa_train_raw = rppa_filtered.loc[train_ids]
rppa_test_raw  = rppa_filtered.loc[test_ids]

# Leak-Free Imputation (Median fitted ONLY on Train)
imputer = SimpleImputer(strategy="median")
train_imp = imputer.fit_transform(rppa_train_raw)
test_imp  = imputer.transform(rppa_test_raw)

# Leak-Free Z-Score Normalization (Mean=0, Std=1 fitted ONLY on Train)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imp)
test_scaled  = scaler.transform(test_imp)

# Save Clean Matrices
rppa_train_clean = pd.DataFrame(train_scaled, index=train_ids, columns=valid_cols)
rppa_test_clean  = pd.DataFrame(test_scaled, index=test_ids, columns=valid_cols)

rppa_train_clean.to_csv(ml_dir / "rppa_train.csv")
rppa_test_clean.to_csv(ml_dir / "rppa_test.csv")

print("\n==========================================================")
print(" CLEAN RPPA MATRICES EXPORTED SUCCESSFULLY")
print("==========================================================")
print(f"rppa_train.csv -> Shape: {rppa_train_clean.shape} | Missing Values: {rppa_train_clean.isnull().sum().sum()}")
print(f"rppa_test.csv  -> Shape: {rppa_test_clean.shape}  | Missing Values: {rppa_test_clean.isnull().sum().sum()}")

# 5. Regenerate Plot B Visual
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
