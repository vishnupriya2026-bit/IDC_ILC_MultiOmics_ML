import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

# Set up output directories
os.makedirs("results/feature_selection", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("=== Step 5: High-Dimensional Feature Selection ===")
rna = pd.read_csv("data/processed/cptac_rna_combat.csv", index_col=0)
clinical = pd.read_csv("results/QC/matched_clinical.csv", index_col=0)

y = clinical["Clinical_Status"]

# ---------------------------------------------------------
# 2. Stage 1: Variance Filtering (Keep top 10% most variable genes)
# ---------------------------------------------------------
print(f"Original Feature Count: {rna.shape[1]}")

variances = rna.var(axis=0)
variance_threshold = np.percentile(variances, 90)  # Top 10% highest variance
selected_by_var = rna.loc[:, variances >= variance_threshold]

print(f"Features after 90th percentile Variance Filter: {selected_by_var.shape[1]}")

# ---------------------------------------------------------
# 3. Stage 2: LASSO (L1 Feature Selection)
# ---------------------------------------------------------
# Handle uniform labels or run sparse estimation
if len(np.unique(y)) > 1:
    lasso = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5, random_state=42)
    lasso.fit(selected_by_var, y)
    
    # Extract non-zero coefficients
    coef_mask = (lasso.coef_ != 0).any(axis=0)
    lasso_selected = selected_by_var.loc[:, coef_mask]
    print(f"Features retained by LASSO (L1 Penalty): {lasso_selected.shape[1]}")
else:
    print("Single label class detected; skipping LASSO coefficient masking.")
    lasso_selected = selected_by_var

# ---------------------------------------------------------
# 4. Stage 3: Random Forest Feature Importance Ranking
# ---------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(lasso_selected, y)

feature_importances = pd.Series(rf.feature_importances_, index=lasso_selected.columns).sort_values(ascending=False)

# Select Top 50 Biomarkers
top_50_features = feature_importances.head(50)
top_50_features.to_csv("results/feature_selection/top_50_biomarkers.csv")

# Save Selected Matrix for ML Training
final_matrix = lasso_selected[top_50_features.index]
final_matrix.to_csv("data/processed/rna_selected_features.csv")

# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.barplot(x=top_50_features.head(20).values, y=top_50_features.head(20).index, palette="viridis", hue=top_50_features.head(20).index, legend=False)
plt.title("Top 20 Predictive Biomarker Features (RF Importance)", fontsize=14, fontweight="bold")
plt.xlabel("Mean Decrease in Impurity (Feature Importance)")
plt.tight_layout()
plt.savefig("results/feature_selection/top_20_features_importance.png", dpi=300)
plt.close()

print("Feature Selection complete! Saved matrix to 'data/processed/rna_selected_features.csv'.")
