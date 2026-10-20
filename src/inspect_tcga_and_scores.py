import pandas as pd
import glob

print("==========================================")
print("       TCGA-BRCA COLUMN INSPECTION        ")
print("==========================================")
tcga_path = "archive/incompatible_clinical/tcga_brca_clinical.tsv"
try:
    tcga_df = pd.read_csv(tcga_path, sep='\t', nrows=2)
    print("TCGA Columns found:")
    print(tcga_df.columns.tolist())
except Exception as e:
    print(f"[!] Error reading {tcga_path}: {e}")

print("\n==========================================")
print("     SCANNING FOR ML PREDICTION FILES     ")
print("==========================================")
potential_files = glob.glob("**/*score*.csv", recursive=True) + \
                  glob.glob("**/*pred*.csv", recursive=True) + \
                  glob.glob("**/results/**/*.csv", recursive=True)

for f in sorted(list(set(potential_files))):
    print("Found candidate file:", f)
