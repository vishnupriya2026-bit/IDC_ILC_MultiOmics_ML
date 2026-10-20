import pandas as pd
import glob
import os

print("=== Aligning Multi-Omics Patient IDs with Clinical Data ===")

# 1. Search for files containing patient IDs or expression matrices
possible_files = glob.glob("**/*.csv", recursive=True) + glob.glob("**/*.tsv", recursive=True)

target_id_file = None
for f in possible_files:
    if "119" in f and "clinical" not in f.lower():
        target_id_file = f
        break

if not target_id_file:
    # Look for RNA or Proteomics matrix files
    for f in possible_files:
        if any(term in f.lower() for term in ["rna", "prot", "expression", "matrix"]):
            target_id_file = f
            break

print(f"Using reference matrix/ID file: '{target_id_file}'")

# Load 119 Patient IDs
if target_id_file:
    df_ref = pd.read_csv(target_id_file, nrows=2)
    # Check if IDs are in rows or columns
    if len(df_ref.columns) > 50:
        ref_ids = set(df_ref.columns.astype(str).str.strip())
    else:
        ref_ids = set(pd.read_csv(target_id_file).iloc[:, 0].astype(str).str.strip())
else:
    print("Could not find expression/ID file automatically. Please provide path.")
    exit()

# Filter to IDs starting with '01BR', '11BR', etc.
my_119_ids = {i for i in ref_ids if "BR" in i or "TCGA" in i or len(i) > 5}
print(f"Identified {len(my_119_ids)} patient IDs from multi-omics file.")

# 2. Load Clinical Data
clinical_df = pd.read_csv("cptac_119_clinical_matched.csv", index_col=0)
clinical_df.index = clinical_df.index.astype(str).str.strip()

# Intersect
aligned_clinical = clinical_df[clinical_df.index.isin(my_119_ids)].copy()
print(f"\nSuccessfully aligned: {len(aligned_clinical)} clinical records to multi-omics cohort.")

# 3. Create Machine Learning Target Labels
# Binary Target 1: Early (Stage I/II) vs Advanced (Stage III)
def map_stage(stage):
    if str(stage) in ['Stage I', 'Stage II']:
        return 0  # Early Stage
    elif str(stage) in ['Stage III']:
        return 1  # Advanced Stage
    return None

aligned_clinical['Target_Stage_Binary'] = aligned_clinical['tumor_stage_pathological'].apply(map_stage)

# Save final aligned metadata
output_file = "cptac_119_clinical_final.csv"
aligned_clinical.to_csv(output_file)

print(f"\nFinal Dataset Target Distribution (saved to '{output_file}'):")
print(aligned_clinical['Target_Stage_Binary'].value_counts(dropna=False).rename({0.0: 'Early (Stage I/II)', 1.0: 'Advanced (Stage III)'}))

