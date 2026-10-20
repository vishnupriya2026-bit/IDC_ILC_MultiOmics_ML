import os
import cptac

target_dir = "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/TCGA/raw"
os.makedirs(target_dir, exist_ok=True)
output_file = os.path.join(target_dir, "cptac_brca_transcriptomics.csv")

print("Extracting Breast Cancer Transcriptomics (mRNA) via cptac API...")
brca = cptac.Brca()

# Extract transcriptomics matrix (mRNA) from UMich / BCM source
try:
    rna_df = brca.get_transcriptomics(source="umich")
except Exception:
    rna_df = brca.get_transcriptomics(source="bcm")

print(f"Saving Transcriptomics matrix ({rna_df.shape[0]} samples x {rna_df.shape[1]} genes)...")
rna_df.to_csv(output_file)

print(f"✅ Successfully saved Transcriptomics matrix to: {output_file}")
