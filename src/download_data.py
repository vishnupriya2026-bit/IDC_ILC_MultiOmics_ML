import os
import urllib.request

os.makedirs("data/raw", exist_ok=True)

files = {
    "TCGA_GBM_clinical.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-GBM.GDC_phenotype.tsv.gz",
    "TCGA_LGG_clinical.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LGG.GDC_phenotype.tsv.gz",
    "TCGA_GBM_gene_counts.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-GBM.htseq_counts.tsv.gz",
    "TCGA_LGG_gene_counts.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LGG.htseq_counts.tsv.gz",
}

# Browser User-Agent header bypasses AWS S3 HTTP 403 blocks
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

for filename, url in files.items():
    filepath = os.path.join("data/raw", filename)
    
    # Remove existing 243-byte error placeholders
    if os.path.exists(filepath):
        os.remove(filepath)

    print(f"Downloading {filename}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(filepath, "wb") as out_file:
            out_file.write(response.read())
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  -> Saved ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  -> Error: {e}")

print("\nDone!")
