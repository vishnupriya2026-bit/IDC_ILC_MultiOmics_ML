import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import os

print("Loading CPTAC Proteomics data...")
df = pd.read_csv("data/processed/CPTAC_Proteomics.csv", index_col=0)

# FIX: Ensure Proteins are Rows and Samples are Columns
if df.shape[0] < df.shape[1]:
    df = df.T

# Check if ERBB2 (HER2) is in the dataset
marker = "ERBB2"
if marker not in df.index:
    # Fallback to the first most variable protein if ERBB2 isn't found
    marker = df.var(axis=1).idxmax()

print(f"Stratifying patients based on {marker} expression...")
marker_expr = df.loc[marker]
median_val = marker_expr.median()

# Split samples into High and Low groups
high_samples = marker_expr[marker_expr >= median_val].index
low_samples = marker_expr[marker_expr < median_val].index

print(f"High group: {len(high_samples)} samples | Low group: {len(low_samples)} samples")

# Compute Fold Change (Log2FC) and p-values
results = []
for protein in df.index:
    high_vals = df.loc[protein, high_samples].dropna()
    low_vals = df.loc[protein, low_samples].dropna()
    
    if len(high_vals) > 5 and len(low_vals) > 5:
        # Calculate Log2 Fold Change
        log2fc = high_vals.mean() - low_vals.mean()
        # Perform T-test
        stat, pval = ttest_ind(high_vals, low_vals, equal_var=False)
        results.append({"Protein": protein, "Log2FC": log2fc, "pvalue": pval})

res_df = pd.DataFrame(results).dropna()

# Adjust p-values for multiple testing
res_df["padj"] = multipletests(res_df["pvalue"], method="fdr_bh")[1]
res_df["-Log10(padj)"] = -np.log10(res_df["padj"])
res_df["-Log10(pvalue)"] = -np.log10(res_df["pvalue"])

# Define Significance Thresholds (Exploratory settings to capture more pathway data)
res_df["Significance"] = "Not Significant"
res_df.loc[(res_df["Log2FC"] > 0.25) & (res_df["pvalue"] < 0.05), "Significance"] = "Upregulated"
res_df.loc[(res_df["Log2FC"] < -0.25) & (res_df["pvalue"] < 0.05), "Significance"] = "Downregulated"

# Save the results table for Step 2
os.makedirs("results/tables", exist_ok=True)
res_df.to_csv("results/tables/Differential_Proteins.csv", index=False)

# Plot Volcano Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=res_df, x="Log2FC", y="-Log10(pvalue)", hue="Significance",
    palette={"Not Significant": "grey", "Upregulated": "red", "Downregulated": "blue"},
    alpha=0.7, edgecolor=None
)

# Label top proteins
top_up = res_df[res_df["Significance"] == "Upregulated"].nlargest(5, "-Log10(pvalue)")
top_down = res_df[res_df["Significance"] == "Downregulated"].nlargest(5, "-Log10(pvalue)")
for idx, row in pd.concat([top_up, top_down]).iterrows():
    plt.text(row["Log2FC"], row["-Log10(pvalue)"] + 0.1, row["Protein"], fontsize=8)

plt.axvline(0.25, color="black", linestyle="--", linewidth=1)
plt.axvline(-0.25, color="black", linestyle="--", linewidth=1)
plt.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)
plt.title(f"Volcano Plot: {marker}-High vs {marker}-Low Breast Cancer")

os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/Volcano_Plot.png", dpi=300)

print(f"✅ Found {len(res_df[res_df['Significance'] != 'Not Significant'])} significant proteins!")
print("✅ Saved Volcano Plot to results/figures/Volcano_Plot.png")
