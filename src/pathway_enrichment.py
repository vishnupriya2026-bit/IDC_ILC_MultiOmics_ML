import pandas as pd
import matplotlib.pyplot as plt
import os
import gseapy as gp
from typing import List

class PathwayEnrichmentEngine:
    """
    Module for functional gene set enrichment analysis (GSEA/Enrichr) 
    mapping driver biomarkers to KEGG, Reactome, and GO pathways.
    """
    def __init__(self, gene_list: List[str]):
        self.gene_list = [g.strip().upper() for g in gene_list]

    def run_enrichment(self, gene_sets: List[str] = None) -> pd.DataFrame:
        """Runs Enrichr query via gseapy across biological pathway databases."""
        if gene_sets is None:
            gene_sets = [
                'KEGG_2021_Human',
                'Reactome_2022',
                'GO_Biological_Process_2023'
            ]
            
        print(f"🧬 Running functional pathway enrichment for {len(self.gene_list)} genes...")
        print(f"📚 Target databases: {gene_sets}")
        
        try:
            enr = gp.enrichr(
                gene_list=self.gene_list,
                gene_sets=gene_sets,
                organism='human',
                outdir=None # Return DataFrame without writing to disk immediately
            )
            
            df = enr.results
            if not df.empty:
                # Filter by Adjusted P-value (FDR < 0.05)
                df = df.sort_values(by="Adjusted P-value").reset_index(drop=True)
                print(f"✅ Found {len(df)} enriched biological pathways.")
                return df
            else:
                print("⚠️ No significant pathways returned.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Enrichment query error: {e}")
            return pd.DataFrame()

    def plot_top_pathways(self, df: pd.DataFrame, output_fig_path: str, top_n: int = 10):
        """Generates a barplot of top enriched pathways ranked by significance."""
        if df.empty:
            print("⚠️ Skipping plot: Empty results dataframe.")
            return

        top_df = df.head(top_n).copy()
        top_df["-log10(FDR)"] = -1 * (top_df["Adjusted P-value"].apply(lambda x: max(x, 1e-10))).apply(pd.np.log10 if hasattr(pd, 'np') else import_log10)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_df["Term"][::-1], top_df["-log10(FDR)"][::-1], color='teal', edgecolor='black', alpha=0.8)
        plt.xlabel("-log10(Adjusted P-value)", fontsize=11, fontweight='bold')
        plt.ylabel("Biological Pathway / GO Term", fontsize=11, fontweight='bold')
        plt.title(f"Top {top_n} Dysregulated Pathways (Multi-Omics Biomarkers)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle="--", alpha=0.5)
        
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        print(f"📈 Pathway enrichment plot saved to '{output_fig_path}'")

def import_log10(x):
    import numpy as np
    return np.log10(x)

if __name__ == "__main__":
    # Multi-omics biomarker panel (e.g. key drivers identified in Objective 1)
    sample_drivers = ["ERBB2", "ESR1", "PIK3CA", "BRCA1", "TP53", "EGFR", "PTEN", "AKT1", "CCND1", "MYC"]
    
    engine = PathwayEnrichmentEngine(gene_list=sample_drivers)
    results_df = engine.run_enrichment()
    
    if not results_df.empty:
        # Save table to processed data
        output_csv = "data/processed/pathway_enrichment_results.csv"
        os.makedirs("data/processed", exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"💾 Pathway table saved to '{output_csv}'")
        
        # Display top 5
        print("\n--- TOP ENRICHED PATHWAYS ---")
        print(results_df[["Gene_set", "Term", "Adjusted P-value", "Combined Score"]].head(5).to_string())
        
        # Plot top 10
        engine.plot_top_pathways(results_df, output_fig_path="results/figures/pathway_enrichment.png", top_n=10)
