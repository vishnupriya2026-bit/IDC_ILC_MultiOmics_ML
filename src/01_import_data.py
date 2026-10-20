#!/usr/bin/env python3

"""
Project:
Breast Cancer Multi-Omics Computational Framework

Script:
01_import_data.py

Purpose:
Import GEO, TCGA, CPTAC and PRIDE datasets
Generate dataset summary
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Dataset paths
# --------------------------------------------------
#!/usr/bin/env python3

"""
Project:
Breast Cancer Multi-Omics Computational Framework

Script:
01_import_data.py

Purpose:
Import GEO, TCGA, CPTAC and PRIDE datasets
Generate dataset summary
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Dataset paths
# ---------------------------------------------------

PROJECT_DIR = Path.home() / "BreastCancer_MultiOmics"

GEO_FILE = PROJECT_DIR / "datasets" / "GEO" / "raw" / "GSE2034_series_matrix.txt"

# ---------------------------------------------------
# Read GEO file
# ---------------------------------------------------

print("=" * 60)
print("Loading GEO Dataset")
print("=" * 60)

geo = pd.read_csv(
    GEO_FILE,
    sep="\t",
    comment="!",
    low_memory=False
)

print("\nDataset Loaded Successfully\n")

print("Shape:")
print(geo.shape)

print("\nColumn Names:")
print(list(geo.columns))

print("\nFirst Five Rows:")
print(geo.head())

print("\nData Types:")
print(geo.dtypes)

print("\nMissing Values:")
print(geo.isnull().sum())

print("\nDuplicate Rows:")
print(geo.duplicated().sum())
