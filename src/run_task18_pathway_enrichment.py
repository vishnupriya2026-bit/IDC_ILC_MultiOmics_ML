import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

print("--- Step 1: Mapping Biomarker Panel to Canonical Pathways ---")

# Canonical pathway gene sets relevant to Breast Cancer biology
pathways = {
    "Cell Cycle & G2/M Transition": ['MKI67', 'AURKA', 'CCNB1', 'CDK1', 'BUB1B'],
    "DNA Replication & Pre-RC": ['TOP2A', 'MCM2', 'BUB1B'],
    "Estrogen Receptor Signaling": ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'SCUBE2'],
    "Apoptosis & Survival": ['BCL2', 'BIRC5', 'BAG1'],
    "FOXM1 / MYB Transcription Network": ['MYBL2', 'AURKA', 'CCNB1', 'MKI67'],
    "PI3K-Akt-mTOR Signaling": ['ESR1', 'BCL2', 'PGR']
}

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']
signature_genes = set(prolif_genes + luminal_genes)

# Background genome size (approximate human protein-coding genome)
N_background = 18000
n_drawn = len(signature_genes)

results = []
for pathway_name, path_genes in pathways.items():
    K_pathway = len(path_genes) * 4  # Estimated pathway size in background
    k_overlap = len(set(path_genes).intersection(signature_genes))
    
    # Hypergeometric p-value (sf: survival function = 1 - cdf)
    p_val = hypergeom.sf(k_overlap - 1, N_background, K_pathway, n_drawn)
    enrichment_score = (k_overlap / n_drawn) / (K_pathway / N_background)
    
    results.append({
        "Pathway": pathway_name,
        "Overlap": k_overlap,
        "Pathway_Size": len(path_genes),
        "Enrichment_Ratio": round(enrichment_score, 2),
        "p_value": p_val,
        "neg_log10_p": -np.log10(p_val)
    })

enrich_df = pd.DataFrame(results).sort_values(by="p_value", ascending=True)

os.makedirs("results", exist_ok=True)
enrich_df.to_csv("results/task18_pathway_enrichment.csv", index=False)

print("\n=== Pathway Enrichment Summary ===")
print(enrich_df[["Pathway", "Overlap", "Enrichment_Ratio", "p_value"]].to_string(index=False))

print("\n--- Step 2: Generating Pathway Enrichment Dot Plot ---")
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

y_pos = np.arange(len(enrich_df))
scatter = ax.scatter(
    enrich_df["Enrichment_Ratio"], 
    y_pos, 
    s=enrich_df["Overlap"] * 90, 
    c=enrich_df["neg_log10_p"], 
    cmap="plasma", 
    alpha=0.85, 
    edgecolors="black"
)

cbar = plt.colorbar(scatter)
cbar.set_label("-log10(p-value)", fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(enrich_df["Pathway"], fontweight="bold")
ax.invert_yaxis()
ax.set_xlabel("Fold Enrichment Ratio", fontweight="bold")
ax.set_title("Multi-Omic Biomarker Pathway Enrichment", fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
output_path = "results/task18_pathway_enrichment.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nTask 18 completed. Saved plot to '{output_path}'")
