import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Comprehensive Mapping Dictionary (Ensembl ID -> Official HGNC Symbol)
gene_mapping = {
    # Full Versioned Ensembl IDs
    "ENSG00000111404.7": "RERGL (RNA)",
    "ENSG00000154330.13": "PGM5 (RNA)",
    "ENSG00000241644.2": "INMT (RNA)",
    "ENSG00000149451.18": "ADAM33 (RNA)",
    "ENSG00000161544.10": "CYGB (RNA)",
    "ENSG00000124664.11": "SPDEF (RNA)",
    "ENSG00000187260.16": "WDR86 (RNA)",
    "ENSG00000064300.9": "NGFR (RNA)",
    "ENSG00000072163.20": "LIMS2 (RNA)",
    "ENSG00000126218.12": "F10 (RNA)",
    "ENSG00000137225.13": "CAPN11 (RNA)",
    "ENSG00000165349.12": "SLC7A3 (RNA)",
    "ENSG00000204065.3": "TCEAL5 (RNA)",
    "ENSG00000187486.5": "KCNJ11 (RNA)",
    "ENSG00000129514.8": "FOXA1 (RNA)",
    
    # Unversioned Ensembl IDs (Fallback)
    "ENSG00000111404": "RERGL (RNA)",
    "ENSG00000154330": "PGM5 (RNA)",
    "ENSG00000241644": "INMT (RNA)",
    "ENSG00000149451": "ADAM33 (RNA)",
    "ENSG00000161544": "CYGB (RNA)",
    "ENSG00000124664": "SPDEF (RNA)",
    "ENSG00000187260": "WDR86 (RNA)",
    "ENSG00000064300": "NGFR (RNA)",
    "ENSG00000072163": "LIMS2 (RNA)",
    "ENSG00000126218": "F10 (RNA)",
    "ENSG00000137225": "CAPN11 (RNA)",
    "ENSG00000165349": "SLC7A3 (RNA)",
    "ENSG00000204065": "TCEAL5 (RNA)",
    "ENSG00000187486": "KCNJ11 (RNA)",
    "ENSG00000129514": "FOXA1 (RNA)",

    # Protein Names
    "ECADHERIN": "E-cadherin (Protein)",
    "BETACATENIN": "beta-catenin (Protein)",
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

