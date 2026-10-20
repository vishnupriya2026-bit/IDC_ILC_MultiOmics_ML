import os
import pandas as pd
import numpy as np
import mygene

# Setup paths
DATA_DIR = os.path.expanduser("~/data")
PROCESSED_DIR = os.path.expanduser("data/processed")
RESULTS_QC = os.path.expanduser("results/QC")

# Ensure output directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_QC, exist_ok=True)

print("--- Step 1: Processing GEO GSE2034 ---")

# Load raw GSE2034 data
gse_file = os.path.join(DATA_DIR, "GSE2034_series_matrix.txt")

if os.path.exists(gse_file):
    df_gse = pd.read_csv(gse_file, sep="\t", comment="!", index_col=0)
    print(f"GSE2034 Raw Shape: {df_gse.shape} (Probes x Samples)")

    # Query MyGene.info for Probe-to-Symbol mapping
    print("Querying MyGene.info for Probe-to-Symbol mapping...")
    mg = mygene.MyGeneInfo()
    probes = df_gse.index.tolist()
    res = mg.querymany(probes, scopes="reporter", fields="symbol", species="human")

    # Map probe IDs to Gene Symbols
    probe_map = {}
    if isinstance(res, list):
        for item in res:
            if "symbol" in item and "query" in item:
                probe_map[item["query"]] = item["symbol"]

    df_gse = df_gse.assign(Symbol=df_gse.index.map(probe_map))
    df_gse_harmonized = df_gse.dropna(subset=["Symbol"]).groupby("Symbol").mean()

    out_geo_path = os.path.join(PROCESSED_DIR, "GSE2034_harmonized.csv")
    df_gse_harmonized.to_csv(out_geo_path)
    print(f"Saved Harmonized GSE2034 to {out_geo_path} with shape: {df_gse_harmonized.shape}")
else:
    print(f"Warning: GSE2034 file not found at {gse_file}. Skipping GSE step.")
    df_gse_harmonized = pd.DataFrame()

print("\n--- Step 2: Verifying CPTAC Harmonization ---")

# Variables explicitly defined here
cptac_rna_file = os.path.join(PROCESSED_DIR, "CPTAC_RNA.csv")
cptac_prot_file = os.path.join(PROCESSED_DIR, "CPTAC_Proteomics.csv")

if os.path.exists(cptac_rna_file) and os.path.exists(cptac_prot_file):
    # .T transposes so rows = genes and columns = samples
    cptac_rna = pd.read_csv(cptac_rna_file, index_col=0).T
    cptac_prot = pd.read_csv(cptac_prot_file, index_col=0).T

    print(f"CPTAC RNA Shape (Genes x Samples): {cptac_rna.shape}")
    print(f"CPTAC Proteomics Shape (Genes x Samples): {cptac_prot.shape}")

    common_genes = list(set(cptac_rna.index).intersection(set(cptac_prot.index)))
    print(f"Overlapping genes between CPTAC RNA & Proteomics: {len(common_genes)}")
else:
    common_genes = []
    cptac_rna = pd.DataFrame()
    cptac_prot = pd.DataFrame()
    print("CPTAC files not found in data/processed, skipping overlap summary.")

# Write QC Summary
qc_summary_file = os.path.join(RESULTS_QC, "qc_summary.txt")
with open(qc_summary_file, "w") as f:
    f.write(f"GSE2034 Harmonized Shape: {df_gse_harmonized.shape}\n")
    f.write(f"CPTAC RNA Shape (Genes x Samples): {cptac_rna.shape}\n")
    f.write(f"CPTAC Proteomics Shape (Genes x Samples): {cptac_prot.shape}\n")
    f.write(f"Common Multi-Omics Genes: {len(common_genes)}\n")

print(f"\nQC and Harmonization complete! Check {qc_summary_file}")
