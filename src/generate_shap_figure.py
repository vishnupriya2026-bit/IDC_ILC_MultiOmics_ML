import os
import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. HGNC Gene Symbol Mapping Dictionary
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
    "ENSG00000187486.5": "KCNJ11"
}

# 2. Locate SHAP scripts in project
shap_scripts = [
    'src/shap_analysis.py',
    'src/run_shap_analysis.py',
    'src/run_task4_shap.py',
    'src/23_explainable_ai_shap.py',
    'src/shap_interpretation.py',
    'src/build_multiomics_ml_shap.py'
]

executed = False
for script in shap_scripts:
    if os.path.exists(script):
        print(f"Executing {script}...")
        res = subprocess.run(['python3', script], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"SUCCESS: Executed {script}")
            executed = True
            break
        else:
            print(f"Output from {script}: {res.stdout[:200]}")

# 3. Ensure output directory exists
os.makedirs('results/figures', exist_ok=True)
print("Updated figures folder check complete.")

