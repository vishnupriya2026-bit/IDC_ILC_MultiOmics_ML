import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Define Paths
processed_dir = Path("data/processed/ml_ready")
rppa_raw_path = Path("data/master_cohort_metadata.csv") # Or your raw RPPA combined CSV

# Load sample splits to maintain strict 80/20 train/test alignment
y_train = pd.read_csv(processed_dir / "y_train.csv", index_col=0)
y_test  = pd.read_csv(processed_dir / "y_test.csv", index_col=0)

train_ids = y_train.index.tolist()
test_ids  = y_test.index.tolist()
all_ids   = train_ids + test_ids

print(f"Target Cohort: {len(all_ids)} Patients ({len(train_ids)} Train / {len(test_ids)} Test)")

# 2. Extract & Harmonize RPPA Matrix
# Load processed RPPA dataset or build from raw antibody files
try:
    rppa_df = pd.read_csv("data/rppa/rppa_combined.csv", index_col=0)
except FileNotFoundError:
    # If combined file doesn't exist, search for existing preprocessed RPPA matrix
    rppa_df = pd.read_csv("data/processed/rppa_matrix.csv", index_col=0)

# Harmonize TCGA barcodes to 12-character format (e.g., TCGA-A1-A0SF)
rppa_df.index = rppa_df.index.str.slice(0, 12)
rppa_df = rppa_df[~rppa_df.index.duplicated(keep='first')]

# Reindex to match exact 410 patient cohort
rppa_harmonized = rppa_df.reindex(all_ids)

print(f"RPPA Raw Dimensions (Matched Cohort): {rppa_harmonized.shape}")

# 3. Handle Missing Values (Zero Missingness Guarantee)
# Drop antibodies missing in > 20% of samples
rppa_filtered = rppa_harmonized.dropna(thresh=int(0.8 * len(all_ids)), axis=1)

# Split into train/test before fitting imputer/scaler to PREVENT DATA LEAKAGE
rppa_train_raw = rppa_filtered.loc[train_ids]
rppa_test_raw  = rppa_filtered.loc[test_ids]

# Impute remaining missing values using Median Imputation
imputer = SimpleImputer(strategy="median")
rppa_train_imp = imputer.fit_transform(rppa_train_raw)
rppa_test_imp  = imputer.transform(rppa_test_raw)

# 4. Standardize / Normalize (Z-Score Scaling: Mean=0, Std=1)
scaler = StandardScaler()
rppa_train_scaled = scaler.fit_transform(rppa_train_imp)
rppa_test_scaled  = scaler.transform(rppa_test_imp)

# 5. Reconstruct DataFrames & Verify Clean Matrix
rppa_train_clean = pd.DataFrame(rppa_train_scaled, index=train_ids, columns=rppa_filtered.columns)
rppa_test_clean  = pd.DataFrame(rppa_test_scaled, index=test_ids, columns=rppa_filtered.columns)

assert rppa_train_clean.isnull().sum().sum() == 0, "Error: Missing values found in Train!"
assert rppa_test_clean.isnull().sum().sum() == 0, "Error: Missing values found in Test!"

# Save Clean ML-Ready RPPA Matrices
rppa_train_clean.to_csv(processed_dir / "rppa_train.csv")
rppa_test_clean.to_csv(processed_dir / "rppa_test.csv")

print("\nSUCCESS: Clean RPPA Matrices Exported Successfully!")
print(f"RPPA Train Shape: {rppa_train_clean.shape} | Missing Values: {rppa_train_clean.isnull().sum().sum()}")
print(f"RPPA Test Shape:  {rppa_test_clean.shape}  | Missing Values: {rppa_test_clean.isnull().sum().sum()}")
