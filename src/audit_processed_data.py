import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics/data/processed"

files = {
    "GSE2034_RNA": "GSE2034_harmonized.csv",
    "CPTAC_RNA": "CPTAC_RNA.csv",
    "CPTAC_Proteomics": "CPTAC_Proteomics.csv",
    "Metabolomics": "Metabolomics_harmonized.csv",
    "Clinical_Survival": "synthetic_clinical_survival.csv",
    "Matched_IDs": "CPTAC_119_matched_patient_ids.csv"
}

datasets = {}
print("=" * 65)
print("FEATURE MATRIX & COHORT OVERLAP AUDIT")
print("=" * 65)

for key, filename in files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        datasets[key] = df
        print(f"\n[+] {key} ({filename}):")
        print(f"    - Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"    - First 3 Columns: {list(df.columns[:3])}")
        
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            missing = num_df.isna().sum().sum()
            total = num_df.size
            print(f"    - Missing Values: {missing} / {total} ({missing/total*100:.2f}%)")
    else:
        print(f"\n[-] {key} ({filename}): File not found.")

print("\n" + "-" * 65)
print("SAMPLE ID INTERSECTIONS")
print("-" * 65)

id_map = {}
for key, df in datasets.items():
    id_col = df.columns[0]
    ids = set(df[id_col].dropna().astype(str).str.strip())
    id_map[key] = ids
    print(f"• {key}: {len(ids)} unique IDs in column '{id_col}'")

omics_keys = [k for k in ['GSE2034_RNA', 'CPTAC_RNA', 'CPTAC_Proteomics', 'Metabolomics'] if k in id_map]
if len(omics_keys) >= 2:
    for i in range(len(omics_keys)):
        for j in range(i+1, len(omics_keys)):
            k1, k2 = omics_keys[i], omics_keys[j]
            overlap = id_map[k1].intersection(id_map[k2])
            print(f"• Overlap ({k1} ∩ {k2}): {len(overlap)} samples")

if 'CPTAC_RNA' in id_map and 'CPTAC_Proteomics' in id_map and 'Metabolomics' in id_map:
    cptac_3way = id_map['CPTAC_RNA'].intersection(id_map['CPTAC_Proteomics']).intersection(id_map['Metabolomics'])
    print(f"\n★ CPTAC 3-Way Omics Overlap (RNA + Protein + Metabolite): {len(cptac_3way)} samples")

if 'Clinical_Survival' in datasets:
    print("\n" + "-" * 65)
    print("CLINICAL / SURVIVAL DATA STRUCTURE")
    print("-" * 65)
    df_clin = datasets['Clinical_Survival']
    print(df_clin.head(3))
    print("\nSummary statistics:")
    print(df_clin.describe(include='all'))

print("=" * 65)
