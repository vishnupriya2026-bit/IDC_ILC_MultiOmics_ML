import os
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Paths Setup
base_dir = Path.home() / "phd_research"
rppa_dir = base_dir / "data" / "rppa"
ml_dir = base_dir / "data" / "processed" / "ml_ready"
ml_dir.mkdir(parents=True, exist_ok=True)

# 2. Load Patient Target Barcodes (410 Patients)
target_y = pd.read_csv(ml_dir / "target_y.csv", index_col=0)
valid_patients = target_y.index.tolist()

print(f"Target Patient Cohort: {len(valid_patients)} samples.")

# 3. Parse downloaded GDC RPPA TSV files
rppa_data = {}
tsv_files = list(rppa_dir.glob("**/*.tsv")) + list(rppa_dir.glob("**/*.txt"))

print(f"Found {len(tsv_files)} raw GDC RPPA files in {rppa_dir}...")

for fpath in tsv_files:
    try:
        # GDC RPPA files usually contain 'Gene Symbol' or 'Peptide Target' and value columns
        df = pd.read_csv(fpath, sep="\t")
        
        # Identify sample ID from header or filename
        sample_id = None
        for col in df.columns:
            if "TCGA" in col:
                sample_id = col[:12]
                break
        
        if sample_id and sample_id in valid_patients:
            # Extract antibody ID and value
            if "Composite Element REF" in df.columns and "Value" in df.columns:
                series = df.set_index("Composite Element REF")["Value"]
                rppa_data[sample_id] = series
            elif df.shape[1] >= 2:
                series = df.set_index(df.columns[0])[df.columns[1]]
                rppa_data[sample_id] = series
    except Exception as e:
        continue

if len(rppa_data) > 0:
    rppa_df = pd.DataFrame(rppa_data).T
    # Align exact 410 patient order
    rppa_df = rppa_df.reindex(valid_patients).dropna(how='all')
    
    # Impute missing values with median
    rppa_df = rppa_df.fillna(rppa_df.median())
    
    # Save rebuilt RPPA feature matrix
    rppa_df.to_csv(ml_dir / "rppa_filtered.csv")
    print(f"\nSUCCESS: Successfully parsed and saved RPPA Matrix!")
    print(f"Final Shape: {rppa_df.shape[0]} Patients x {rppa_df.shape[1]} Antibodies.")
else:
    print("\nWARNING: Direct GDC TSV parsing required specific barcode mapping metadata.")
    print("If your downloaded GDC files are structured by UUID folders, we will use gdc_sample_sheet.tsv.")
