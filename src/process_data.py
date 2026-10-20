import os
import pandas as pd
import numpy as np
import mygene

# Setup paths
DATA_DIR = os.path.expanduser("~/data")
PROCESSED_DIR = os.path.expanduser("~/data/processed")
RESULTS_QC = os.path.expanduser("results/QC")

os.makedirs(RESULTS_QC, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("--- Step 1: Processing GEO GSE2034 ---")

# 1. Read GSE2034 Series Matrix (skipping comment lines starting with !)
gse_path = os.path.join(DATA_DIR, "GSE2034_series_matrix.txt")
df_gse = pd.read_csv(gse_path, sep="\t", comment="!", index_col=0)

# Clean quotes from column names and index
df_gse.columns = df_gse.columns.str.replace('"', '')
df_gse.index = df_gse.index.str.replace('"', '')

print(f"GSE2034 Raw Shape: {df_gse.shape} (Probes x Samples)")

# 2. Map Affymetrix Probe IDs to HGNC Gene Symbols using MyGeneInfo
mg = mygene.MyGeneInfo()
probe_ids = df_gse.index.tolist()

print("Querying MyGene.info for Probe-to-Symbol mapping...")
res = mg.querymany(probe_ids, scopes="reporter", fields="symbol", species="human", as_dataframe=True)

# Filter valid mappings
if 'symbol' in res.columns:
    mapping = res['symbol'].dropna().to_dict()
    df_gse['Gene'] = df_gse.index.map(mapping)
    
    # Drop unmapped probes
    df_gse = df_gse.dropna(subset=['Gene'])
    
    # Convert numerical columns
    data_cols = [c for c in df_gse.columns if c != 'Gene']
    df_gse[data_cols] = df_gse[data_cols].apply(pd.to_numeric, errors='coerce')
    
    # Aggregate multiple probes mapping to the same gene by mean
    df_gse_harmonized = df_gse.groupby('Gene')[data_cols].mean()
    
    # Save harmonized GEO expression dataset
    out_geo_path = "data/processed/gse2034_rna_harmonized.csv"
    df_gse_harmonized.to_csv(out_geo_path)
    print(f"Saved Harmonized GSE2034 to {out_geo_path} with shape: {df_gse_harmonized.shape}")

else:
    print("Warning: Probe mapping field 'symbol' not found in response.")

print("\n--- Step 2: Verifying CPTAC Harmonization ---")
cptac_rna = pd.read_csv(os.path.join(PROCESSED_DIR, "cptac_rna_combat.csv"), index_col=0)
cptac_prot = pd.read_csv(os.path.join(PROCESSED_DIR, "cptac_proteomics_clean.csv"), index_col=0)

print(f"CPTAC RNA Shape: {cptac_rna.shape}")
print(f"CPTAC Proteomics Shape: {cptac_prot.shape}")

# Identify common overlapping genes between CPTAC RNA & Proteomics
common_genes = sorted(list(set(cptac_rna.index).intersection(set(cptac_prot.index))))
print(f"Overlapping genes between CPTAC RNA & Proteomics: {len(common_genes)}")

# Write summary QC report
with open(f"{RESULTS_QC}/qc_summary.txt", "w") as f:
    f.write(f"GSE2034 Harmonized Shape: {df_gse_harmonized.shape}\n")
    f.write(f"CPTAC RNA Shape: {cptac_rna.shape}\n")
    f.write(f"CPTAC Proteomics Shape: {cptac_prot.shape}\n")
    f.write(f"Common Multi-Omics Genes: {len(common_genes)}\n")

print("\nQC and Harmonization complete! Check results/QC/qc_summary.txt")
