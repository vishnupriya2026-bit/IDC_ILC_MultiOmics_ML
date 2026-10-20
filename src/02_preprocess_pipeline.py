import os
import pandas as pd
from pathlib import Path

base_dir = Path.home() / "phd_research" / "data"
out_dir = base_dir / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Clean Target File
print("[1/5] Loading Target Labels...")
target_df = pd.read_csv(base_dir / "target_y.csv")
patient_col = [c for c in target_df.columns if any(k in c.lower() for k in ['patient', 'cases', 'submitter'])][0]

target_df['patient_id'] = target_df[patient_col].astype(str).str.slice(0, 12)
target_clean = target_df.dropna(subset=['Y_Stage']).drop_duplicates(subset=['patient_id'])
target_clean.set_index('patient_id', inplace=True)
print(f"      Valid patients with Y_Stage labels: {len(target_clean)}")

# Helper to find matrix files recursively
def find_matrix_file(folder_name):
    valid_exts = ('.tsv', '.csv', '.parquet', '.txt', '.tsv.gz', '.csv.gz')
    search_path = base_dir / folder_name
    if not search_path.exists():
        return None
    candidates = [
        f for f in search_path.rglob("*") 
        if f.is_file() and 'metadata' not in f.name.lower() and str(f).endswith(valid_exts)
    ]
    return candidates[0] if candidates else None

# 2. Locate Data Files
print("[2/5] Locating Omics Matrices...")
rna_matrix_file = find_matrix_file("rna_seq") or find_matrix_file("rnaseq")
meth_matrix_file = find_matrix_file("methylation")
rppa_matrix_file = find_matrix_file("rppa") or find_matrix_file("proteomics")

print(f"      RNA-Seq File    : {rna_matrix_file}")
print(f"      Methylation File: {meth_matrix_file}")
print(f"      RPPA File       : {rppa_matrix_file}")

if not all([rna_matrix_file, meth_matrix_file, rppa_matrix_file]):
    print("\n[!] Could not locate all required omics files.")
    print("    Run: find ~/phd_research/data -maxdepth 3 -type f")
    exit(1)

# Helper to load and standardize matrices
def load_and_standardize_matrix(file_path):
    file_str = str(file_path)
    sep = '\t' if any(file_str.endswith(ext) for ext in ['.tsv', '.tsv.gz', '.txt']) else ','
    compression = 'gzip' if file_str.endswith('.gz') else 'infer'
    
    if file_str.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, sep=sep, index_col=0, compression=compression)
    
    # Check if samples are in columns
    col_barcodes = [c[:12] for c in df.columns if str(c).startswith('TCGA')]
    if len(col_barcodes) > 0:
        new_cols = {c: str(c)[:12] for c in df.columns if str(c).startswith('TCGA')}
        df = df.rename(columns=new_cols)
        df = df.T
        df = df.loc[~df.index.duplicated(keep='first')]
        return df
    
    # Check if samples are in index
    df.index = df.index.astype(str).str.slice(0, 12)
    df = df.loc[~df.index.duplicated(keep='first')]
    return df

# 3. Load Datasets
print("[3/5] Loading and Barcode-Standardizing Matrices...")
df_rna = load_and_standardize_matrix(rna_matrix_file)
df_meth = load_and_standardize_matrix(meth_matrix_file)
df_rppa = load_and_standardize_matrix(rppa_matrix_file)

# 4. Compute Common Cohort Intersection
print("[4/5] Calculating Patient Overlap...")
target_patients = set(target_clean.index)
rna_patients = set(df_rna.index)
meth_patients = set(df_meth.index)
rppa_patients = set(df_rppa.index)

common_patients = sorted(list(target_patients & rna_patients & meth_patients & rppa_patients))

print(f"      Target Patients     : {len(target_patients)}")
print(f"      RNA-Seq Samples     : {len(rna_patients)}")
print(f"      Methylation Samples : {len(meth_patients)}")
print(f"      RPPA Samples        : {len(rppa_patients)}")
print(f"      --> COMMON OVERLAP  : {len(common_patients)} patients")

# 5. Subset and Save Aligned Matrices
print("[5/5] Exporting Aligned Datasets to processed/...")
target_clean.loc[common_patients].to_csv(out_dir / "target_y_aligned.csv")
df_rna.loc[common_patients].to_csv(out_dir / "rna_seq_aligned.csv")
df_meth.loc[common_patients].to_csv(out_dir / "methylation_aligned.csv")
df_rppa.loc[common_patients].to_csv(out_dir / "rppa_aligned.csv")

print("\n=== PREPROCESSING & ALIGNMENT COMPLETE ===")
print(f"All files successfully saved to: {out_dir}")
