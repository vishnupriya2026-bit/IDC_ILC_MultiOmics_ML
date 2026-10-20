import os
import cptac

# Target directory for raw data
DATA_DIR = "/home/priya_26phd0306/BreastCancer_MultiOmics/datasets/CPTAC/raw"
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading/Loading CPTAC BRCA dataset...")
brca = cptac.Brca()

print("Extracting Proteomics abundance matrix (source='umich')...")
try:
    # Pass source explicitly to avoid Python 3.12 generator length bug
    proteomics = brca.get_proteomics(source="umich")
except Exception as e:
    print(f"UMich source failed ({e}), trying BCM source...")
    proteomics = brca.get_proteomics(source="bcm")

output_path = os.path.join(DATA_DIR, "cptac_brca_proteomics.csv")
proteomics.to_csv(output_path)

print(f"\n✅ Successfully saved CPTAC Proteomics matrix to: {output_path}")
print(f"Matrix shape: {proteomics.shape[0]} samples x {proteomics.shape[1]} proteins")
