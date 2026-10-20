import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import os

print("Loading differentially expressed proteins...")
if not os.path.exists("results/tables/Differential_Proteins.csv"):
    print("Error: Run Step 1 (07_differential_expression.py) first!")
    exit()

df = pd.read_csv("results/tables/Differential_Proteins.csv")

# Extract the Upregulated Proteins
upregulated_genes = df[df["Significance"] == "Upregulated"]["Protein"].tolist()

print(f"Running Enrichment Analysis on {len(upregulated_genes)} Upregulated Proteins...")

if len(upregulated_genes) < 5:
    print("Not enough significant genes for pathway analysis.")
    exit()

# Run Enrichr over KEGG Pathways 
enr = gp.enrichr(
    gene_list=upregulated_genes,
    gene_sets="KEGG_2021_Human",
    organism="human",
    outdir=None
)

# Extract and clean results
results = enr.results
results = results[results["Adjusted P-value"] < 0.05].head(10) # Get top 10 significant pathways

if results.empty:
    print("No statistically significant pathways found.")
else:
    # Plotting the Bar Chart
    plt.figure(figsize=(10, 6))
    results = results.sort_values(by="Adjusted P-value", ascending=False) # Sort for plotting
    
    plt.barh(results["Term"], -np.log10(results["Adjusted P-value"]), color="coral")
    plt.xlabel("-Log10(Adjusted P-value)")
    plt.title("Top Enriched KEGG Pathways (Upregulated in HER2-High)")
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/KEGG_Enrichment_Barplot.png", dpi=300)
    print("✅ Saved Pathway Enrichment Plot to results/figures/KEGG_Enrichment_Barplot.png")
