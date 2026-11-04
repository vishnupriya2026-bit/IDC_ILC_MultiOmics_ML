import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Comprehensive Mapping Dictionary (Ensembl ID -> Official HGNC Symbol)
gene_mapping = {
    # Full Versioned Ensembl IDs
    "RERGL (RNA)": "RERGL (RNA)",
    "PGM5 (RNA)": "PGM5 (RNA)",
    "INMT (RNA)": "INMT (RNA)",
    "ADAM33 (RNA)": "ADAM33 (RNA)",
    "CYGB (RNA)": "CYGB (RNA)",
    "SPDEF (RNA)": "SPDEF (RNA)",
    "WDR86 (RNA)": "WDR86 (RNA)",
    "NGFR (RNA)": "NGFR (RNA)",
    "LIMS2 (RNA)": "LIMS2 (RNA)",
    "F10 (RNA)": "F10 (RNA)",
    "CAPN11 (RNA)": "CAPN11 (RNA)",
    "SLC7A3 (RNA)": "SLC7A3 (RNA)",
    "TCEAL5 (RNA)": "TCEAL5 (RNA)",
    "KCNJ11 (RNA)": "KCNJ11 (RNA)",
    "FOXA1 (RNA)": "FOXA1 (RNA)",
    
    # Unversioned Ensembl IDs (Fallback)
    "RERGL (RNA)": "RERGL (RNA)",
    "PGM5 (RNA)": "PGM5 (RNA)",
    "INMT (RNA)": "INMT (RNA)",
    "ADAM33 (RNA)": "ADAM33 (RNA)",
    "CYGB (RNA)": "CYGB (RNA)",
    "SPDEF (RNA)": "SPDEF (RNA)",
    "WDR86 (RNA)": "WDR86 (RNA)",
    "NGFR (RNA)": "NGFR (RNA)",
    "LIMS2 (RNA)": "LIMS2 (RNA)",
    "F10 (RNA)": "F10 (RNA)",
    "CAPN11 (RNA)": "CAPN11 (RNA)",
    "SLC7A3 (RNA)": "SLC7A3 (RNA)",
    "TCEAL5 (RNA)": "TCEAL5 (RNA)",
    "KCNJ11 (RNA)": "KCNJ11 (RNA)",
    "FOXA1 (RNA)": "FOXA1 (RNA)",

    # Protein Names
    "E-cadherin (Protein)": "E-cadherin (Protein)",
    "beta-catenin (Protein)": "beta-catenin (Protein)",
    "EGFR": "EGFR (Protein)",
    "AR": "AR (Protein)"
}

# 2. Create output directory
out_dir = "results/figures"
os.makedirs(out_dir, exist_ok=True)

# 3. Locate existing SHAP values / Test set or run candidate script
print("Searching for saved SHAP objects or running SHAP generator...")

# Search for pre-computed SHAP files or feature data in results/ or data/
feature_files = glob.glob("data/**/*.csv", recursive=True) + glob.glob("results/**/*.csv", recursive=True)

# Patch main SHAP scripts if present to force feature renaming
for target in ['src/run_task4_shap.py', 'src/shap_analysis.py', 'src/23_explainable_ai_shap.py', 'src/run_shap_analysis.py']:
    if os.path.exists(target):
        print(f"Modifying and executing {target} with feature mapping...")
        with open(target, 'r') as f:
            code = f.read()

        # Inject dictionary and column replacement before plotting
        mapping_code = f"""
# --- INJECTED MAPPING CODE ---
gene_mapping = {repr(gene_mapping)}
if 'X_test' in locals() and isinstance(X_test, pd.DataFrame):
    X_test = X_test.rename(columns=gene_mapping)
if 'X' in locals() and isinstance(X, pd.DataFrame):
    X = X.rename(columns=gene_mapping)
# -----------------------------
"""
        if "gene_mapping =" not in code:
            code = mapping_code + code

        # Replace standard summary_plot call to save mapped figure directly
        code = code.replace(
            "shap.summary_plot(",
            "plt.figure(figsize=(10, 8)); "
            "shap.summary_plot("
        )

        with open("src/temp_shap_mapped.py", "w") as f:
            f.write(code)

        import subprocess
        res = subprocess.run(["python3", "src/temp_shap_mapped.py"], capture_output=True, text=True)
        print("STDOUT:", res.stdout[:300])
        if res.stderr:
            print("STDERR:", res.stderr[:300])
        break

