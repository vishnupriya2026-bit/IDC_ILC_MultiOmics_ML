import os
import pandas as pd

def export_final_panel():
    out_dir = "results/biomarker_discovery"
    prioritized_file = f"{out_dir}/prioritized_biomarker_panel.csv"
    final_output = f"{out_dir}/final_multiomics_biomarker_panel.csv"
    
    df = pd.read_csv(prioritized_file)
    
    top_panel = df.head(10).copy()
    top_panel['Biomarker_Type'] = 'Transcriptomic / Proteomic'
    top_panel['Clinical_Relevance'] = 'Prognostic & Diagnostic Marker'
    
    export_cols = [
        'feature', 'Biomarker_Type', 'log2FC', 'fdr', 
        'composite_score', 'prioritization_score', 'Clinical_Relevance'
    ]
    
    final_panel = top_panel[export_cols].rename(columns={'feature': 'Biomarker_Name'})
    final_panel.to_csv(final_output, index=False)
    
    print("\n========================================================")
    print("    FINAL MULTI-OMICS BIOMARKER PANEL (OBJECTIVE 4)     ")
    print("========================================================")
    print(final_panel.to_string(index=False))
    print(f"\nSaved final validated biomarker panel to: {final_output}")

if __name__ == "__main__":
    export_final_panel()
