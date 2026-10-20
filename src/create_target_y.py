import os
import pandas as pd

base_dir = os.path.expanduser("~/phd_research/data")

# 1. Load Master Cohort and Clinical Data
master_df = pd.read_csv(os.path.join(base_dir, "master_cohort_metadata.csv"))
clin_df = pd.read_csv(os.path.join(base_dir, "clinical/extracted/clinical.tsv"), sep="\t", low_memory=False, dtype=str)

clin_matched = clin_df[clin_df['cases.submitter_id'].isin(master_df['patient_id'])].drop_duplicates(subset=['cases.submitter_id'])

# 2. Map Histological Target (IDC vs ILC)
def map_histology(val):
    val = str(val).lower()
    if 'infiltrating duct carcinoma' in val and 'lobular' not in val:
        return 'IDC'
    elif 'lobular carcinoma' in val and 'duct' not in val:
        return 'ILC'
    return None

# 3. Map Pathologic Stage Target (Early vs Advanced)
def map_stage(val):
    val = str(val).strip().upper()
    if val in ['STAGE I', 'STAGE IA', 'STAGE IB', 'STAGE II', 'STAGE IIA', 'STAGE IIB']:
        return 'Early_Stage'
    elif val in ['STAGE III', 'STAGE IIIA', 'STAGE IIIB', 'STAGE IIIC', 'STAGE IV']:
        return 'Advanced_Stage'
    return None

target_df = pd.DataFrame({'patient_id': clin_matched['cases.submitter_id']})
target_df['Y_Histology'] = clin_matched['diagnoses.primary_diagnosis'].apply(map_histology)
target_df['Y_Stage'] = clin_matched['diagnoses.ajcc_pathologic_stage'].apply(map_stage)

# Save Master Y File
out_path = os.path.join(base_dir, "target_y.csv")
target_df.to_csv(out_path, index=False)

print("--- TARGET MAPPING SUMMARY ---")
print(f"Saved Target file to: {out_path}\n")
print("Histology (Y_Histology) Class Counts:")
print(target_df['Y_Histology'].value_counts(dropna=False))
print("\nPathologic Stage (Y_Stage) Class Counts:")
print(target_df['Y_Stage'].value_counts(dropna=False))
