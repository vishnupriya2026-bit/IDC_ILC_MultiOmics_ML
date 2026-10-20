import os
import glob
import pandas as pd
from pathlib import Path

base_dir = Path.home() / "phd_research" / "data"
out_dir = base_dir / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Load Clean Target File
# ---------------------------------------------------------
print("[1/5] Loading Target Labels...")
target_df = pd.read_csv(base_dir / "target_y.csv")
patient_col = [c for c in target_df.columns if any(k in c.lower() for k in ['patient', 'cases', 'submitter'])][0]

target_df['patient_id'] = target_df[patient_col].astype(str).str.slice(0, 12)
target_clean = target_df.dropna(subset=['Y_Stage']).drop_duplicates(subset=['patient_id'])
target_clean.set_index('patient_id', inplace=True)
print(f"      Valid target patients: {len(target_clean)}")

# Helper to find patient mapping column in metadata CSVs
def get_patient_id_from_meta(meta_df):
    possible_cols = [c for c in meta_df.columns if any(k in c.lower() for k in ['cases', 'submitter', 'patient', 'sample'])]
    for col in possible_cols:
        sample_val = str(meta_df[col].dropna().iloc[0])
        if 'TCGA' in sample_val:
            return col
    return possible_cols[0] if possible_cols else meta_df.columns[0]

def get_file_col_from_meta(meta_df):
    possible_cols = [c for c in meta_df.columns if any(k in c.lower() for k in ['file_name', 'filename', 'file_id', 'file'])]
    return possible_cols[0] if possible_cols else meta_df.columns[0]

# ---------------------------------------------------------
# 2. Build RNA-Seq Matrix (tpm_unstranded)
# ---------------------------------------------------------
print("\n[2/5] Building RNA-Seq Matrix...")
rna_meta = pd.read_csv(base_dir / "rna_seq_metadata.csv")
rna_pat_col = get_patient_id_from_meta(rna_meta)
rna_file_col = get_file_col_from_meta(rna_meta)

rna_meta['patient_id'] = rna_meta[rna_pat_col].astype(str).str.slice(0, 12)
rna_files = list((base_dir / "rna_seq").rglob("*.rna_seq.augmented_star_gene_counts.tsv"))

rna_dict = {}
for f in rna_files:
    fname = f.name
    match = rna_meta[rna_meta[rna_file_col] == fname]
    if match.empty:
        # Match by file_id if filename match fails
        match = rna_meta[rna_meta[rna_file_col] == f.parent.name]
    
    if not match.empty:
        pat_id = match['patient_id'].values[0]
        # Read GDC STAR counts skipping comment header
        df = pd.read_csv(f, sep='\t', comment='#')
        # Skip STAR summary statistics rows (N_unmapped, N_multimapping, etc.)
        df = df[~df['gene_id'].str.startswith('N_')]
        val_col = 'tpm_unstranded' if 'tpm_unstranded' in df.columns else df.columns[3]
        rna_dict[pat_id] = df.set_index('gene_id')[val_col]

df_rna = pd.DataFrame(rna_dict).T  # Samples x Genes
df_rna = df_rna.loc[~df_rna.index.duplicated(keep='first')]
print(f"      RNA-Seq Matrix Shape: {df_rna.shape}")

# ---------------------------------------------------------
# 3. Build Methylation Matrix
# ---------------------------------------------------------
print("\n[3/5] Building Methylation Matrix...")
meth_meta = pd.read_csv(base_dir / "methylation_metadata.csv")
meth_pat_col = get_patient_id_from_meta(meth_meta)
meth_file_col = get_file_col_from_meta(meth_meta)

meth_meta['patient_id'] = meth_meta[meth_pat_col].astype(str).str.slice(0, 12)
meth_files = list((base_dir / "methylation").rglob("*.sesame.level3betas.txt"))

meth_dict = {}
for f in meth_files:
    fname = f.name
    match = meth_meta[meth_meta[meth_file_col] == fname]
    if match.empty:
        match = meth_meta[meth_meta[meth_file_col] == f.parent.name]
    
    if not match.empty:
        pat_id = match['patient_id'].values[0]
        df = pd.read_csv(f, sep='\t', header=None, names=['probe_id', 'beta'])
        meth_dict[pat_id] = df.set_index('probe_id')['beta']

df_meth = pd.DataFrame(meth_dict).T  # Samples x Probes
df_meth = df_meth.loc[~df_meth.index.duplicated(keep='first')]
print(f"      Methylation Matrix Shape: {df_meth.shape}")

# ---------------------------------------------------------
# 4. Build RPPA Matrix
# ---------------------------------------------------------
print("\n[4/5] Building RPPA Matrix...")
rppa_meta = pd.read_csv(base_dir / "rppa_metadata.csv")
rppa_pat_col = get_patient_id_from_meta(rppa_meta)
rppa_file_col = get_file_col_from_meta(rppa_meta)

rppa_meta['patient_id'] = rppa_meta[rppa_pat_col].astype(str).str.slice(0, 12)
rppa_files = list((base_dir / "rppa").rglob("*_RPPA_data.tsv")) + list((base_dir / "rppa").rglob("*.tsv"))

rppa_dict = {}
for f in rppa_files:
    if 'metadata' in f.name:
        continue
    fname = f.name
    match = rppa_meta[rppa_meta[rppa_file_col] == fname]
    if match.empty:
        match = rppa_meta[rppa_meta[rppa_file_col] == f.parent.name]
    
    if not match.empty:
        pat_id = match['patient_id'].values[0]
        df = pd.read_csv(f, sep='\t')
        # Use antibody or gene target as index
        feature_col = df.columns[0]
        val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        rppa_dict[pat_id] = df.set_index(feature_col)[val_col]

df_rppa = pd.DataFrame(rppa_dict).T  # Samples x Antibodies
df_rppa = df_rppa.loc[~df_rppa.index.duplicated(keep='first')]
print(f"      RPPA Matrix Shape: {df_rppa.shape}")

# ---------------------------------------------------------
# 5. Cohort Alignment & Saving
# ---------------------------------------------------------
print("\n[5/5] Aligning Cohort Intersection...")
target_pats = set(target_clean.index)
rna_pats = set(df_rna.index)
meth_pats = set(df_meth.index)
rppa_pats = set(df_rppa.index)

common_patients = sorted(list(target_pats & rna_pats & meth_pats & rppa_pats))

print(f"      Target Patients     : {len(target_pats)}")
print(f"      RNA-Seq Patients    : {len(rna_pats)}")
print(f"      Methylation Patients: {len(meth_pats)}")
print(f"      RPPA Patients       : {len(rppa_pats)}")
print(f"      ====================================")
print(f"      --> COMMON OVERLAP  : {len(common_patients)} patients")

target_clean.loc[common_patients].to_csv(out_dir / "target_y_aligned.csv")
df_rna.loc[common_patients].to_csv(out_dir / "rna_seq_aligned.csv")
df_meth.loc[common_patients].to_csv(out_dir / "methylation_aligned.csv")
df_rppa.loc[common_patients].to_csv(out_dir / "rppa_aligned.csv")

print(f"\nAll aligned matrices exported successfully to: {out_dir}")
