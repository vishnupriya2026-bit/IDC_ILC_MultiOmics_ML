import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

exp_dir = Path("results/diablo_exports")
ml_dir  = Path("data/processed/ml_ready")

print("==========================================================")
print(" PREPROCESSING INTEGRATED DIABLO DATASET FOR ML TRAINING")
print("==========================================================")

# 1. Load Raw DIABLO Features & Clinical Target Matrices
X_train_raw = pd.read_csv(exp_dir / "diablo_integrated_features_train.csv", index_col=0)
X_test_raw  = pd.read_csv(exp_dir / "diablo_integrated_features_test.csv", index_col=0)

y_train_raw = pd.read_csv(ml_dir / "y_train.csv", index_col=0)
y_test_raw  = pd.read_csv(ml_dir / "y_test.csv", index_col=0)

# 2. Synchronize Patient Index Alignment
common_tr = X_train_raw.index.intersection(y_train_raw.index)
common_te = X_test_raw.index.intersection(y_test_raw.index)

X_tr_aligned = X_train_raw.loc[common_tr]
y_tr_aligned = y_train_raw.loc[common_tr]
X_te_aligned = X_test_raw.loc[common_te]
y_te_aligned = y_test_raw.loc[common_te]

print(f"• Aligned Cohort Size  : {len(common_tr)} Train Samples | {len(common_te)} Test Samples")
print(f"• Initial Feature Count: {X_tr_aligned.shape[1]} Multi-Omics Biomarkers")

# 3. Variance Thresholding (Remove near-constant features)
var_thresh = VarianceThreshold(threshold=1e-4)
X_tr_var = var_thresh.fit_transform(X_tr_aligned)
X_te_var = var_thresh.transform(X_te_aligned)

retained_cols = X_tr_aligned.columns[var_thresh.get_support()]
print(f"• Retained Features    : {len(retained_cols)} Features after variance filtering")

# 4. Leak-Free Imputation (Median strategy fit on train)
imputer = SimpleImputer(strategy="median")
X_tr_imp = imputer.fit_transform(X_tr_var)
X_te_imp = imputer.transform(X_te_var)

# 5. Leak-Free Z-Score Standardization (Fit on train, transform on test)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_imp)
X_te_sc = scaler.transform(X_te_imp)

# Reconstruct DataFrames with original column headers & patient IDs
X_tr_final = pd.DataFrame(X_tr_sc, index=common_tr, columns=retained_cols)
X_te_final = pd.DataFrame(X_te_sc, index=common_te, columns=retained_cols)

# 6. Encode Target Labels
le = LabelEncoder()
y_tr_clean = y_tr_aligned.copy()
y_te_clean = y_te_aligned.copy()

y_tr_clean['Y_Stage_Encoded'] = le.fit_transform(y_tr_clean['Y_Stage'])
y_te_clean['Y_Stage_Encoded']  = le.transform(y_te_clean['Y_Stage'])

# Map target distribution
class_map = dict(zip(le.classes_, le.transform(le.classes_)))

# 7. Export Preprocessed Matrices
X_tr_final.to_csv(exp_dir / "X_diablo_train_scaled.csv")
X_te_final.to_csv(exp_dir / "X_diablo_test_scaled.csv")
y_tr_clean.to_csv(exp_dir / "y_diablo_train_encoded.csv")
y_te_clean.to_csv(exp_dir / "y_diablo_test_encoded.csv")

print("\n==========================================================")
print(" PREPROCESSING COMPLETE - EXPORT SUMMARY")
print("==========================================================")
print(f"• X_diablo_train_scaled.csv -> Shape: {X_tr_final.shape} | Nulls: {X_tr_final.isnull().sum().sum()}")
print(f"• X_diablo_test_scaled.csv  -> Shape: {X_te_final.shape}  | Nulls: {X_te_final.isnull().sum().sum()}")
print(f"• Mean across features      -> Train: {X_tr_final.mean().mean():.4f} | Test: {X_te_final.mean().mean():.4f}")
print(f"• Std across features       -> Train: {X_tr_final.std().mean():.4f} | Test: {X_te_final.std().mean():.4f}")
print(f"• Target Label Encoding     -> {class_map}")
print("==========================================================")
