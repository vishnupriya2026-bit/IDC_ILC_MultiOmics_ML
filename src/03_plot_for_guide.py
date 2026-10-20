import os
import pandas as pd

# 1. Load Datasets
geo_rna = pd.read_csv("data/processed/GSE2034_harmonized.csv", index_col=0)
cptac_rna = pd.read_csv("data/processed/CPTAC_RNA.csv", index_col=0)
cptac_prot = pd.read_csv("data/processed/CPTAC_Proteomics.csv", index_col=0)

# Transpose CPTAC if loaded as Samples x Features
if cptac_rna.shape[0] < cptac_rna.shape[1]:
    cptac_rna = cptac_rna.T

if cptac_prot.shape[0] < cptac_prot.shape[1]:
    cptac_prot = cptac_prot.T

# Load Metabolomics
metab_path = "data/processed/Metabolomics_harmonized.csv"
if os.path.exists(metab_path):
    cptac_metab = pd.read_csv(metab_path, index_col=0)
    if cptac_metab.shape[0] < cptac_metab.shape[1] and cptac_metab.shape[0] < 500:
        pass # Keep as Features x Samples
else:
    cptac_metab = pd.DataFrame()

# Build Clean Summary Table
summary_df = pd.DataFrame([
    {"Omics Layer": "GEO Microarray (GSE2034)", "Features": geo_rna.shape[0], "Samples": geo_rna.shape[1]},
    {"Omics Layer": "CPTAC RNA-Seq", "Features": cptac_rna.shape[0], "Samples": cptac_rna.shape[1]},
    {"Omics Layer": "CPTAC Proteomics", "Features": cptac_prot.shape[0], "Samples": cptac_prot.shape[1]},
    {"Omics Layer": "Metabolomics (ST000915)", "Features": cptac_metab.shape[0], "Samples": cptac_metab.shape[1]},
])

# Print clean output
print("\n" + "="*50)
print("             MULTI-OMICS COHORT SUMMARY             ")
print("="*50)
print(summary_df.to_string(index=False))
print("="*50 + "\n")

summary_df.to_csv("results/QC/final_cohort_summary.csv", index=False)
