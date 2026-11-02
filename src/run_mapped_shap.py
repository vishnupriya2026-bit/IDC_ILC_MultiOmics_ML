import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. Official HGNC Gene Symbol Mapping
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

