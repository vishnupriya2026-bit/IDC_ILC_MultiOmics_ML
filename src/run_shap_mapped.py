import os
import sys
import subprocess

# 1. Ensembl ID to HGNC Symbol Dictionary
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

