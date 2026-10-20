import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RNA_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/TCGA/raw/cptac_brca_transcriptomics.csv"
PROT_PATH = "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/CPTAC/raw/cptac_brca_proteomics.csv"

def make_columns_unique(cols):
    seen = {}
    new_cols = []
    for col in cols:
        col_str = str(col)
        if col_str in seen:
            seen[col_str] += 1
            new_cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            new_cols.append(col_str)
    return new_cols

def load_cptac_matrix(path):
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        df.columns = [col[0] for col in df.columns]
    except Exception:
        df = pd.read_csv(path, index_col=0, on_bad_lines="skip")
    
    df.columns = make_columns_unique(df.columns)
    return df

print("--- Step 1: Loading Raw Matrices ---")
rna_df = load_cptac_matrix(RNA_PATH)
prot_df = load_cptac_matrix(PROT_PATH)

print(f"Raw RNA Matrix: {rna_df.shape[0]} samples x {rna_df.shape[1]} genes")
print(f"Raw Proteomics Matrix: {prot_df.shape[0]} samples x {prot_df.shape[1]} proteins")

print("\n--- Step 2: Aligning Samples (Intersection) ---")
common_samples = sorted(list(set(rna_df.index).intersection(set(prot_df.index))))
print(f"Found {len(common_samples)} overlapping samples with complete multi-omics data.")

rna_aligned = rna_df.loc[common_samples].copy()
prot_aligned = prot_df.loc[common_samples].copy()

print("\n--- Step 3: Filtering Low-Quality Features (>30% Missing) ---")
max_missing_thresh = 0.30 * len(common_samples)

rna_clean = rna_aligned.dropna(thresh=len(common_samples) - max_missing_thresh, axis=1)
prot_clean = prot_aligned.dropna(thresh=len(common_samples) - max_missing_thresh, axis=1)

print(f"RNA features after filtering: {rna_clean.shape[1]} (Dropped {rna_df.shape[1] - rna_clean.shape[1]} sparse genes)")
print(f"Proteomics features after filtering: {prot_clean.shape[1]} (Dropped {prot_df.shape[1] - prot_clean.shape[1]} sparse proteins)")

print("\n--- Step 4: Imputing Missing Values (k-NN Imputer) ---")
prot_numeric = prot_clean.apply(pd.to_numeric, errors='coerce')

imputer = KNNImputer(n_neighbors=5)
prot_imputed_array = imputer.fit_transform(prot_numeric.values)

prot_imputed = pd.DataFrame(
    prot_imputed_array, 
    index=prot_clean.index, 
    columns=prot_clean.columns
)

rna_imputed = rna_clean.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"RNA Remaining Missing Values: {rna_imputed.isnull().sum().sum()}")
print(f"Proteomics Remaining Missing Values: {prot_imputed.isnull().sum().sum()}")

print("\n--- Step 5: Normalization & Log Transformation ---")
rna_log = np.log2(rna_imputed.clip(lower=0) + 1)
prot_log = np.log2(prot_imputed.clip(lower=0) + 1) if (prot_imputed < 0).sum().sum() == 0 else prot_imputed

rna_log.to_csv(os.path.join(PROCESSED_DIR, "cptac_rna_clean.csv"))
prot_log.to_csv(os.path.join(PROCESSED_DIR, "cptac_proteomics_clean.csv"))

print("\n================ PREPROCESSING COMPLETE ================")
print(f"Processed datasets saved to: {PROCESSED_DIR}")
print(f"Final Matched Multi-Omics Cohort: {len(common_samples)} patients")
print(f"Clean RNA Matrix: {rna_log.shape}")
print(f"Clean Proteomics Matrix: {prot_log.shape}")
