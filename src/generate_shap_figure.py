import os
import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. HGNC Gene Symbol Mapping Dictionary
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
    "KCNJ11 (RNA)": "KCNJ11"
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

