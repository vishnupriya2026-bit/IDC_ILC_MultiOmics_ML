import os
import pandas as pd

base_dir = os.path.expanduser("~/phd_research/data")

# 1. Load Metadata CSVs
rna_df = pd.read_csv(os.path.join(base_dir, "rna_seq_metadata.csv"), dtype=str)
rppa_df = pd.read_csv(os.path.join(base_dir, "rppa_metadata.csv"), dtype=str)
meth_df = pd.read_csv(os.path.join(base_dir, "methylation_metadata.csv"), dtype=str)

# 2. Load Clinical TSV
clin_path = os.path.join(base_dir, "clinical/extracted/clinical.tsv")
clin_df = pd.read_csv(clin_path, sep="\t", low_memory=False, dtype=str)

# 3. Filter Primary Tumors
rna_prim = rna_df[rna_df['sample_type'].str.contains('Primary', case=False, na=False)].drop_duplicates(subset=['patient_id'])
rppa_prim = rppa_df[rppa_df['sample_type'].str.contains('Primary', case=False, na=False)].drop_duplicates(subset=['patient_id'])
meth_prim = meth_df[meth_df['sample_type'].str.contains('Primary', case=False, na=False)].drop_duplicates(subset=['patient_id'])

# 4. Set Verified Clinical Patient Barcode Column
clin_id_col = 'cases.submitter_id'
clin_pts = set(clin_df[clin_id_col].dropna().unique())

# 5. Extract Patient ID Sets
rna_pts = set(rna_prim['patient_id'].dropna().unique())
rppa_pts = set(rppa_prim['patient_id'].dropna().unique())
meth_pts = set(meth_prim['patient_id'].dropna().unique())

# 6. Compute 4-Way Intersection
common_pts = sorted(list(rna_pts & rppa_pts & meth_pts & clin_pts))

print("\n--- Cohort Summary ---")
print(f"Primary RNA-seq Patients: {len(rna_pts)}")
print(f"Primary RPPA Patients: {len(rppa_pts)}")
print(f"Primary Methylation Patients: {len(meth_pts)}")
print(f"Clinical Patients: {len(clin_pts)}")
print(f"-> Overlapping Multi-Omics Cohort Size: {len(common_pts)}")

# 7. Construct Master Mapping Table
master_df = pd.DataFrame({'patient_id': common_pts})
master_df = master_df.merge(rna_prim[['patient_id', 'file_id', 'sample_barcode']], on='patient_id', how='left').rename(columns={'file_id': 'rna_uuid', 'sample_barcode': 'rna_barcode'})
master_df = master_df.merge(rppa_prim[['patient_id', 'file_id', 'sample_barcode']], on='patient_id', how='left').rename(columns={'file_id': 'rppa_uuid', 'sample_barcode': 'rppa_barcode'})
master_df = master_df.merge(meth_prim[['patient_id', 'file_id', 'sample_barcode']], on='patient_id', how='left').rename(columns={'file_id': 'meth_uuid', 'sample_barcode': 'meth_barcode'})

out_path = os.path.join(base_dir, "master_cohort_metadata.csv")
master_df.to_csv(out_path, index=False)
print(f"\nSaved master metadata mapping to: {out_path}\n")
