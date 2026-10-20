import os
import re
import pandas as pd

base_dir = os.path.expanduser("~/phd_research/data")

# 1. Inspect Omics Barcodes
rna_df = pd.read_csv(os.path.join(base_dir, "rna_seq_metadata.csv"), dtype=str)
print("--- Omics Barcode Samples ---")
print("RNA-seq patient_id sample values:", rna_df['patient_id'].dropna().head().tolist())

# 2. Inspect Clinical TSV
clin_path = os.path.join(base_dir, "clinical/extracted/clinical.tsv")
clin_df = pd.read_csv(clin_path, sep="\t", low_memory=False, dtype=str)

print("\n--- Clinical TSV Summary ---")
print(f"Total Rows: {len(clin_df)}, Total Columns: {len(clin_df.columns)}")

print("\nSearching for columns containing TCGA patient barcodes (e.g. TCGA-XX-XXXX)...")
barcode_pattern = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}')

matching_cols = []
for col in clin_df.columns:
    sample_vals = clin_df[col].dropna().astype(str)
    matches = sample_vals.apply(lambda x: bool(barcode_pattern.search(x)))
    if matches.sum() > 0:
        unique_matches = sample_vals[matches].apply(lambda x: barcode_pattern.search(x).group(0)).nunique()
        matching_cols.append((col, matches.sum(), unique_matches))

print("\nCandidate Barcode Columns Found:")
for col, total_matches, unique_matches in matching_cols:
    print(f"  • Column: '{col}' | Matching Rows: {total_matches} | Unique Patients: {unique_matches}")
