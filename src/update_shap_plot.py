import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import glob

# 1. Defined Ensembl ID -> HGNC Symbol Mapping
gene_mapping = {
    "ENSG00000111404.7": "RERGL",
    "ENSG00000154330.13": "PGM5",
    "ENSG00000241644.2": "INMT",
    "ENSG00000149451.18": "ADAM33",
    "ENSG00000161544.10": "CYGB",
    "ENSG00000124664.11": "SPDEF",
    "ENSG00000187260.16": "WDR86",
    "ENSG00000064300.9": "NGFR",
    "ENSG00000072163.20": "LIMS2",
    "ENSG00000126218.12": "F10",
    "ENSG00000137225.13": "CAPN11",
    "ENSG00000165349.12": "SLC7A3",
    "ENSG00000204065.3": "TCEAL5",
    "ENSG00000187486.5": "KCNJ11",
}

print("Looking for existing SHAP scripts or model files in src/ and results/...")

# Check for existing scripts generating SHAP plots
shap_scripts = glob.glob("src/*shap*.py") + glob.glob("src/*explain*.py") + glob.glob("src/*rf*.py")

for script in shap_scripts:
    print(f"Found candidate script: {script}")

