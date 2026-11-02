import os
import sys
import subprocess

# 1. Ensembl ID to HGNC Symbol Dictionary
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

# 2. Patch candidates in src/
candidate_scripts = [
    'src/run_shap_analysis.py',
    'src/shap_analysis.py',
    'src/run_task4_shap.py',
    'src/23_explainable_ai_shap.py',
    'src/shap_interpretation.py'
]

for script_path in candidate_scripts:
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Inject mapping
        if 'gene_mapping' not in code:
            code = f"gene_mapping = {repr(gene_mapping)}\n" + code
            
        if 'shap.summary_plot' in code and 'rename(columns=gene_mapping)' not in code:
            code = code.replace(
                'shap.summary_plot(',
                'if "X_test" in locals() and hasattr(X_test, "rename"):\n'
                '    X_test = X_test.rename(columns=gene_mapping)\n'
                'shap.summary_plot('
            )
            with open(script_path, 'w') as f:
                f.write(code)
            print(f"Patched mapping into {script_path}")

        # Run script
        print(f"Running {script_path}...")
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"SUCCESS: Generated mapped plot via {script_path}")
            break
        else:
            print(f"Notice from {script_path}: {res.stderr[:200]}")

