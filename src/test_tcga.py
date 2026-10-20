import pandas as pd

tcga_path = "archive/incompatible_clinical/tcga_brca_clinical.tsv"
try:
    df = pd.read_csv(tcga_path, sep='\t')
    print("TCGA Clinical Shape:", df.shape)
    print("TCGA Columns preview:", [c for c in df.columns if 'vital' in c.lower() or 'time' in c.lower() or 'days' in c.lower()][:10])
except Exception as e:
    print("Error loading TCGA:", e)
