import os
import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt

def run_enrichment():
    input_file = "results/feature_selection/differential_expression_results.csv"
    output_dir = "results/pathway_enrichment"
    output_figure = "results/publication_figures/pathway_enrichment_barplot.png"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/publication_figures", exist_ok=True)
    
    # Load DEG results
    df = pd.read_csv(input_file)
    deg_list = df[df['status'] != 'Not Significant']['feature'].dropna().unique().tolist()
    
    print(f"Loaded {len(deg_list)} significant DEGs for pathway analysis...")
    
    # Run Enrichr on KEGG and GO Biological Process
    gene_sets = ['KEGG_2021_Human', 'GO_Biological_Process_2021']
    
    enr = gp.enrichr(
        gene_list=deg_list,
        gene_sets=gene_sets,
        organism='human',
        outdir=output_dir,
        cutoff=0.05
    )
    
    res = enr.results
    res.to_csv(f"{output_dir}/enrichment_results.csv", index=False)
    print(f"Saved enrichment results to: {output_dir}/enrichment_results.csv")
    
    # Generate Top 10 Pathways Barplot
    top_pathways = res.sort_values(by='Adjusted P-value').head(10).copy()
    top_pathways['neg_log_p'] = -np.log10(top_pathways['Adjusted P-value'].replace(0, 1e-300))
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.barh(top_pathways['Term'], top_pathways['neg_log_p'], color='#2b5c8f')
    plt.xlabel(r'$-\log_{10}(\text{Adjusted } p\text{-value})$')
    plt.title('Top 10 Enriched Pathways (KEGG & GO)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_figure)
    plt.close()
    print(f"Saved Pathway Plot to: {output_figure}")

if __name__ == "__main__":
    run_enrichment()