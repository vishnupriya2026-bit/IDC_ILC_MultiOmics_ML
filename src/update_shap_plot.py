import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import glob

# 1. Defined Ensembl ID -> HGNC Symbol Mapping
gene_mapping = {
    "RERGL (RNA)": "RERGL",
    "PGM5 (RNA)": "PGM5",
    "INMT (RNA)": "INMT",
    "ADAM33 (RNA)": "ADAM33",
    "CYGB (RNA)": "CYGB",
    "SPDEF (RNA)": "SPDEF",
    "WDR86 (RNA)": "WDR86",
    "NGFR (RNA)": "NGFR",
    "LIMS2 (RNA)": "LIMS2",
    "F10 (RNA)": "F10",
    "CAPN11 (RNA)": "CAPN11",
    "SLC7A3 (RNA)": "SLC7A3",
    "TCEAL5 (RNA)": "TCEAL5",
    "KCNJ11 (RNA)": "KCNJ11",
}

print("Looking for existing SHAP scripts or model files in src/ and results/...")

# Check for existing scripts generating SHAP plots
shap_scripts = glob.glob("src/*shap*.py") + glob.glob("src/*explain*.py") + glob.glob("src/*rf*.py")

for script in shap_scripts:
    print(f"Found candidate script: {script}")

