import os
import shutil
import pandas as pd

print("="*70)
print(" 1. LOCATING LOCAL METABRIC DATASET")
print("="*70)

source_dir = os.path.expanduser("~/brca_metabric")
target_dir = "./brca_metabric"

if os.path.exists(source_dir):
    print(f"Found existing local dataset at: {source_dir}")
    if not os.path.exists(target_dir):
        print("Copying local dataset to current workspace...")
        shutil.copytree(source_dir, target_dir)
        print("Local dataset copied successfully!")
    else:
        print("Dataset folder already present in workspace.")
else:
    print("Local copy not found. Attempting web download with custom User-Agent...")
    archive_url = "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
    os.system(f'wget --user-agent="Mozilla/5.0" -O brca_metabric.tar.gz "{archive_url}"')
    os.system("tar -xzf brca_metabric.tar.gz")

data_dir = "./brca_metabric"

print("\n" + "="*70)
print(" 2. FILTERING IDC vs. ILC COHORT (N = 1,894 Target)")
print("="*70)

sample_file = os.path.join(data_dir, "data_clinical_sample.txt")

if os.path.exists(sample_file):
    clinical_df = pd.read_csv(sample_file, sep="\t", skiprows=4)
    print(f"Raw Clinical Samples Loaded: {len(clinical_df)} rows")

    hist_col = None
    for col in ['CANCER_TYPE_DETAILED', 'HISTOLOGICAL_SUBTYPE', 'CANCER_TYPE']:
        if col in clinical_df.columns:
            hist_col = col
            break

    if hist_col:
        print(f"Using Histology Column: '{hist_col}'")
        
        idc_mask = clinical_df[hist_col].astype(str).str.contains("Ductal", case=False, na=False)
        ilc_mask = clinical_df[hist_col].astype(str).str.contains("Lobular", case=False, na=False) & ~clinical_df[hist_col].astype(str).str.contains("Mixed|Ductal", case=False, na=False)
        
        idc_df = clinical_df[idc_mask]
        ilc_df = clinical_df[ilc_mask]
        
        print(f"  • IDC (Invasive Ductal Carcinoma) Cases : {len(idc_df)}")
        print(f"  • ILC (Invasive Lobular Carcinoma) Cases: {len(ilc_df)}")
        
        filtered_clinical = pd.concat([idc_df, ilc_df]).drop_duplicates(subset=['SAMPLE_ID'])
        print(f"  • Total Filtered Validation Cohort      : {len(filtered_clinical)} patients")

        filtered_clinical.to_csv("filtered_metabric_validation_cohort.csv", index=False)
        print("\nSaved preprocessed metadata to: 'filtered_metabric_validation_cohort.csv'")
else:
    print(f"Clinical file not found in {data_dir}. Checking directory contents...")
    print(os.listdir(data_dir) if os.path.exists(data_dir) else "Directory missing.")

print("\n" + "="*70)
print(" 3. EXECUTING VALIDATION & DCA SCRIPT")
print("="*70)

dca_script = os.path.expanduser("~/brca_metabric/case_lists/generate_dca.py")
if os.path.exists(dca_script):
    print(f"Running validation plot generation: {dca_script}")
    os.system(f"python3 {dca_script}")
else:
    print("Validation workspace ready!")

