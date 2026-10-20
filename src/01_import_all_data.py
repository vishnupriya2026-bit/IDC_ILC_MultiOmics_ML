import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset file locations
DATA_PATHS = {
    "GEO Transcriptomics (GSE2034)": "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/GEO/raw/GSE2034_series_matrix.txt",
    "CPTAC/TCGA Transcriptomics (mRNA)": "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/TCGA/raw/cptac_brca_transcriptomics.csv",
    "CPTAC Proteomics": "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/CPTAC/raw/cptac_brca_proteomics.csv",
    "TCGA Clinical": "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/TCGA/clinical/tcga_brca_clinical.tsv",
    "CPTAC Clinical": "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/CPTAC/clinical/cptac_brca_clinical.csv",
}

def inspect_dataset(df, name):
    return {
        "Dataset": name,
        "Rows (Features/Patients)": df.shape[0],
        "Columns (Samples/Attributes)": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Memory Usage (MB)": round(df.memory_usage().sum() / (1024**2), 2)
    }

def load_geo(file_path):
    skip_rows = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                skip_rows += 1
                break
            skip_rows += 1
    df = pd.read_csv(file_path, sep="\t", skiprows=skip_rows, index_col=0, comment="!")
    if "!series_matrix_table_end" in df.index:
        df = df.drop("!series_matrix_table_end")
    return df

summaries = []

for name, path in DATA_PATHS.items():
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found for {name} at {path}")
        continue
        
    print(f"Loading {name}...")
    try:
        if "GEO" in name:
            df = load_geo(path)
        elif "Proteomics" in name:
            df = pd.read_csv(path, header=[0, 1], index_col=0)
        elif path.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
        else:
            df = pd.read_csv(path, on_bad_lines="skip")
            
        summaries.append(inspect_dataset(df, name))
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")

# Save master summary report
if summaries:
    summary_df = pd.DataFrame(summaries)
    report_path = os.path.join(RESULTS_DIR, "import_summary.csv")
    summary_df.to_csv(report_path, index=False)

    print("\n================ COMPLETE MULTI-OMICS DATASET INVENTORY ================")
    print(summary_df.to_string(index=False))
    print(f"\nSaved updated report to: {report_path}")
