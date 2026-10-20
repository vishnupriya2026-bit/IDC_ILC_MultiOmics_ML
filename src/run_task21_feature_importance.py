import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("--- Step 1: Loading Dataset & Feature Subsets ---")
gse_raw = pd.read_csv("data/processed/GSE2034_harmonized.csv")
gse_clean = gse_raw.groupby("Symbol").mean().T

series_matrix_path = "datasets/GEO/raw/GSE2034_series_matrix.txt"
with open(series_matrix_path, "r") as f:
    lines = f.readlines()

sample_ids = []
for line in lines:
    if line.startswith("!Sample_geo_accession"):
        sample_ids = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        break

relapse_status = None
for line in lines:
    if line.startswith("!Sample_characteristics_ch1"):
        parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
        if parts:
            tag = parts[0].lower()
            if "relapse" in tag and "time" not in tag and not any(s in tag for s in ["bone", "brain", "lung", "loco"]):
                relapse_status = parts

if relapse_status is None:
    for line in lines:
        if line.startswith("!Sample_characteristics_ch1"):
            parts = [x.strip('"\r\n ') for x in line.split("\t")[1:]]
            if parts and "relapse" in parts[0].lower() and "time" not in parts[0].lower():
                relapse_status = parts
                break

y_event = np.array([1 if x.split(":")[-1].strip().lower() in ["1", "yes", "relapse"] else 0 for x in relapse_status[:len(sample_ids)]])

meta_df = pd.DataFrame({"event": y_event}, index=sample_ids)
gse_clean = gse_clean.loc[gse_clean.index.intersection(meta_df.index)]
meta_df = meta_df.loc[gse_clean.index]

prolif_genes = ['MKI67', 'AURKA', 'CCNB1', 'TOP2A', 'BIRC5', 'CDK1', 'MYBL2', 'BUB1B', 'MCM2']
luminal_genes = ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'BCL2', 'SCUBE2', 'BAG1']
all_genes = [g for g in prolif_genes + luminal_genes if g in gse_clean.columns]

X = gse_clean[all_genes].values
y = meta_df['event'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("--- Step 2: Extracting Model Feature Importances ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
rf_importances = rf.feature_importances_

lr = LogisticRegression(random_state=42)
lr.fit(X_scaled, y)
lr_coefs = lr.coef_[0]

importance_df = pd.DataFrame({
    "Gene": all_genes,
    "Module": ["Proliferation" if g in prolif_genes else "Luminal" for g in all_genes],
    "RF_Gini_Importance": rf_importances,
    "LR_Coefficient": lr_coefs,
    "Abs_LR_Coefficient": np.abs(lr_coefs)
}).sort_values(by="RF_Gini_Importance", ascending=False)

os.makedirs("results", exist_ok=True)
importance_df.to_csv("results/task21_feature_importance.csv", index=False)

print("\n=== Top 10 Ranked Multi-Omic Drivers ===")
print(importance_df[["Gene", "Module", "RF_Gini_Importance", "LR_Coefficient"]].head(10).to_string(index=False))

print("\n--- Step 3: Generating Feature Importance Visualization ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# Colors by module
colors = ['#d62728' if m == 'Proliferation' else '#1f77b4' for m in importance_df['Module']]

# Plot 1: Random Forest Gini Importance
ax1.barh(importance_df['Gene'], importance_df['RF_Gini_Importance'], color=colors, edgecolor='black', alpha=0.85)
ax1.invert_yaxis()
ax1.set_xlabel('Gini Feature Importance', fontweight='bold')
ax1.set_title('Random Forest Feature Ranking', fontweight='bold')
ax1.grid(axis='x', linestyle=':', alpha=0.6)

# Plot 2: Logistic Regression Coefficients
lr_sorted = importance_df.sort_values(by="LR_Coefficient", ascending=True)
lr_colors = ['#d62728' if m == 'Proliferation' else '#1f77b4' for m in lr_sorted['Module']]
ax2.barh(lr_sorted['Gene'], lr_sorted['LR_Coefficient'], color=lr_colors, edgecolor='black', alpha=0.85)
ax2.axvline(0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Standardized Log-Odds Coefficient (Beta)', fontweight='bold')
ax2.set_title('Logistic Regression Feature Impact', fontweight='bold')
ax2.grid(axis='x', linestyle=':', alpha=0.6)

plt.suptitle('Multi-Omic Feature Importance & Effect Size Analysis', fontweight='bold', fontsize=14)
plt.tight_layout()

output_path = "results/task21_feature_importance.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nTask 21 completed. Saved figure to '{output_path}' and metrics to 'results/task21_feature_importance.csv'")
