import pandas as pd
import numpy as np
import sys
from pathlib import Path

base = Path.cwd()
proc = base / 'data' / 'processed'
ml = proc / 'ml_ready'

target = pd.read_csv(ml / 'target_y.csv', index_col=0)
ref_samples = [str(s).strip() for s in target.index]
ref_map = {str(s).strip().replace('.', '-')[:12]: str(s).strip() for s in ref_samples}

rppa_raw = pd.read_csv(proc / 'rppa_aligned.csv')

# Find TCGA barcode column
barcode_col = None
for col in rppa_raw.columns:
    sample_vals = rppa_raw[col].astype(str)
    if sample_vals.str.contains('TCGA', case=False).any():
        barcode_col = col
        break

if barcode_col:
    rppa_raw['clean_barcode'] = rppa_raw[barcode_col].astype(str).str.strip().str.replace('.', '-', regex=False).str[:12]
    rppa_raw = rppa_raw[rppa_raw['clean_barcode'].isin(ref_map.keys())]
    rppa_raw['patient_id'] = rppa_raw['clean_barcode'].map(ref_map)
    rppa_raw = rppa_raw.set_index('patient_id').drop(columns=['clean_barcode'])

# Drop non-numeric metadata columns
meta_cols = [c for c in ['file_id', 'sample_barcode', 'patient_id', 'sample_type', barcode_col] if c in rppa_raw.columns]
rppa_df = rppa_raw.drop(columns=meta_cols, errors='ignore')

# Extract numeric features
df_num = rppa_df.apply(pd.to_numeric, errors='coerce').fillna(0)
df_num = df_num.loc[~df_num.index.duplicated(keep='first')]

# Filter non-constant antibody features
var_series = df_num.var(axis=0, skipna=True)
valid_cols = var_series[var_series > 1e-8].index
df_final = df_num[valid_cols]

print(f"SUCCESS: Successfully aligned {len(df_final)} RPPA samples with target_y across {len(valid_cols)} antibody features.")
