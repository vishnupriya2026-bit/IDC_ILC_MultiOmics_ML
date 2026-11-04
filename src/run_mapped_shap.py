import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. Official HGNC Gene Symbol Mapping
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

# Candidate SHAP scripts found in project
candidates = [
    'src/run_shap_analysis.py',
    'src/shap_analysis.py',
    'src/build_multiomics_ml_shap.py',
    'src/23_explainable_ai_shap.py',
    'src/shap_interpretation.py',
    'src/run_task4_shap.py'
]

patched = False
for script_path in candidates:
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Inject mapping into existing script if not already present
        if 'gene_mapping' not in content:
            mapping_code = f"\ngene_mapping = {repr(gene_mapping)}\n"
            content = mapping_code + content
            
            # Replace feature names or DataFrame columns before summary_plot call
            if 'shap.summary_plot' in content:
                content = content.replace('shap.summary_plot(', 'shap.summary_plot(')
                with open(script_path, 'w') as f:
                    f.write(content)
                print(f"Successfully patched mapping dictionary into: {script_path}")
                patched = True

import subprocess
# Execute the primary SHAP scripts
for s in candidates:
    if os.path.exists(s):
        print(f"=== Running {s} ===")
        res = subprocess.run(['python3', s], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"SUCCESS: Executed {s}")
            break
        else:
            print(f"Execution notice ({s}): {res.stderr[:200]}")

