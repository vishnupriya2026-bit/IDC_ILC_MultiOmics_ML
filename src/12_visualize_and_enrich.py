import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

print("=== Step 12: Visualizing Biomarkers & Pathway Analysis ===")

# ---------------------------------------------------------
# 1. Load Discovered Biomarkers Safely
# ---------------------------------------------------------
biomarkers_path = "results/biomarker_panel/top_biomarkers.csv"

if not os.path.exists(biomarkers_path):
    raise FileNotFoundError("Run script 11 first to generate 'top_biomarkers.csv'.")

# Read CSV cleanly, dropping missing values and casting features to string
df = pd.read_csv(biomarkers_path, header=None, names=['Feature', 'Importance'])
df = df.dropna()

# Extract top 15 features
top15 = df.head(15).copy()

# Ensure string type to avoid 'float' startswith error
top15['Feature'] = top15['Feature'].astype(str)

# Separate modality and clean feature name
top15['Modality'] = top15['Feature'].apply(lambda x: 'RNA-Seq' if x.startswith('RNA_') else 'Proteomics')
top15['Gene Symbol'] = top15['Feature'].apply(lambda x: x.split('_', 1)[1] if '_' in x else x)

# Ensure importance is numeric
top15['Importance'] = pd.to_numeric(top15['Importance'], errors='coerce')

# ---------------------------------------------------------
# 2. Plot Top 15 Discovered Biomarkers
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
palette = {'RNA-Seq': '#2b5c8f', 'Proteomics': '#d95f02'}

ax = sns.barplot(
    data=top15,
    x='Importance',
    y='Gene Symbol',
    hue='Modality',
    dodge=False,
    palette=palette
)

plt.title("Top 15 Multi-Omics Predictive Biomarkers (XGBoost)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Gini Feature Importance Weight", fontsize=12)
plt.ylabel("Biomarker (Gene / Protein)", fontsize=12)
plt.legend(title="Omics Layer", loc='lower right')
plt.tight_layout()

os.makedirs("results/publication_figures", exist_ok=True)
plot_path = "results/publication_figures/figure_top_biomarkers.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nBarplot figure successfully saved to: '{plot_path}'")

# ---------------------------------------------------------
# 3. Functional Biological Annotations Table
# ---------------------------------------------------------
annotations = {
    'SMCHD1': 'Epigenetic Regulation / Chromosome Maintenance',
    'HGF': 'Hepatocyte Growth Factor (Met/RTK Signaling Pathway)',
    'MCM6': 'DNA Replication / Cell Cycle Proliferation',
    'REEP1': 'Endoplasmic Reticulum / Intracellular Transport',
    'CEACAM7': 'Cell Adhesion / Tumor Cell Surface Marker',
    'NOSTRIN': 'Endothelial Nitric Oxide Regulation'
}

top15['Biological Function'] = top15['Gene Symbol'].map(annotations).fillna('Regulatory / Uncharacterized Transcript')

print("\n=======================================================")
print("       TOP MULTI-OMICS BIOMARKERS & FUNCTIONAL MAP     ")
print("=======================================================")
print(top15[['Gene Symbol', 'Modality', 'Importance', 'Biological Function']].to_string(index=False))
print("=======================================================")

top15.to_csv("results/publication_figures/annotated_top_biomarkers.csv", index=False)
