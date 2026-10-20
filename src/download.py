import os
import urllib.request

os.makedirs("data/raw", exist_ok=True)

datasets = {
    "TCGA_GBM_clinical.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-GBM.GDC_phenotype.tsv.gz",
    "TCGA_LGG_clinical.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LGG.GDC_phenotype.tsv.gz",
    "TCGA_GBM_gene_counts.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-GBM.htseq_counts.tsv.gz",
    "TCGA_LGG_gene_counts.tsv.gz": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-LGG.htseq_counts.tsv.gz",
}

for filename, url in datasets.items():
    path = os.path.join("data/raw", filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"-> Saved {filename} ({size_mb:.2f} MB)")
