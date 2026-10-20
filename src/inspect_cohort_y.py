import os
import pandas as pd

base_dir = os.path.expanduser("~/phd_research/data")

# 1. Load Master Cohort and Clinical Data
master_df = pd.read_csv(os.path.join(base_dir, "master_cohort_metadata.csv"))
clin_df = pd.read_csv(os.path.join(base_dir, "clinical/extracted/clinical.tsv"), sep="\t", low_memory=False, dtype=str)

cohort_pts = set(master_df['patient_id'])

# 2. Filter Clinical Data to Matched 460 Patients
clin_matched = clin_df[clin_df['cases.submitter_id'].isin(cohort_pts)].copy()
# Deduplicate clinical rows per patient if needed
clin_matched = clin_matched.drop_duplicates(subset=['cases.submitter_id'])

# 3. Select Target Candidates
target_cols = [
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.primary_diagnosis',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.ajcc_pathologic_n',
    'demographic.vital_status'
]

print(f"--- CANDIDATE TARGET EVALUATION (N = {len(clin_matched)} Patients) ---\n")

for col in target_cols:
    if col in clin_matched.columns:
        s = clin_matched[col].fillna('Missing').astype(str).str.strip()
        s = s.replace({'nan': 'Missing', "'--": 'Missing', '--': 'Missing', 'not reported': 'Missing'})
        
        valid_cnt = len(s[s != 'Missing'])
        print(f"Variable: '{col}'")
        print(f"Coverage: {valid_cnt}/{len(s)} patients ({valid_cnt/len(s)*100:.1f}%)")
        print("Class Breakdown:")
        for val, cnt in s.value_counts().items():
            print(f"  • {val}: {cnt}")
        print("-" * 50)
