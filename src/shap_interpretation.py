import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# 1. Load Data & Champion Hyperparameters
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)

X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv(
    "diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns 
    else "diablo_exports/y_diablo_test_encoded.csv", index_col=0
)

y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train_raw = y_train_df.reindex(X_train.index).iloc[:, 0].astype(str)
y_test_raw = y_test_df.reindex(X_test.index).iloc[:, 0].astype(str)

X_all = pd.concat([X_train, X_test], axis=0)
y_all_raw = pd.concat([y_train_raw, y_test_raw], axis=0)

le = LabelEncoder()
y_all = le.fit_transform(y_all_raw)

with open("best_bayesian_hyperparameters.json", "r") as f:
    best_params_dict = json.load(f)

rf_best_params = best_params_dict.get("Random Forest", {"n_estimators": 150, "max_depth": 5})

# 2. Fit Champion Model
champion_rf = RandomForestClassifier(**rf_best_params, class_weight='balanced', random_state=42)
champion_rf.fit(X_all, y_all)

# 3. Calculate SHAP Values
print("Computing SHAP values using TreeExplainer...")
explainer = shap.TreeExplainer(champion_rf)
shap_values = explainer.shap_values(X_all)

# Format SHAP values for target class (Class 1: ILC)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(np.shape(shap_values)) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

# 4. Export Mean Absolute SHAP Importance
mean_abs_shap = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature_ID': X_all.columns,
    'Mean_Absolute_SHAP': mean_abs_shap
}).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

shap_df.to_csv("shap_feature_importance.csv", index=False)

print("\n" + "="*70)
print("             TOP 15 MULTI-OMICS BIOMARKERS BY SHAP IMPORTANCE          ")
print("="*70)
print(shap_df.head(15).to_string(index=False))
print("="*70 + "\n")

# 5. Generate SHAP Beeswarm Summary Plot
plt.figure(figsize=(10, 8), dpi=300)
shap.summary_plot(sv, X_all, max_display=20, show=False)
plt.title("SHAP Beeswarm Summary Plot (Class: Invasive Lobular Carcinoma)", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", bbox_inches='tight')
plt.close()

# 6. Generate SHAP Bar Importance Plot
plt.figure(figsize=(10, 8), dpi=300)
shap.summary_plot(sv, X_all, plot_type="bar", max_display=20, show=False)
plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", bbox_inches='tight')
plt.close()

# 7. Generate SHAP Dependence Plots for Top 2 Biomarkers
top_1_feat = shap_df.loc[0, 'Feature_ID']
top_2_feat = shap_df.loc[1, 'Feature_ID']

# Top 1 Dependence Plot
plt.figure(figsize=(8, 6), dpi=300)
shap.dependence_plot(top_1_feat, sv, X_all, show=False)
plt.title(f"SHAP Dependence Plot: {top_1_feat}", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("shap_dependence_top1.png", bbox_inches='tight')
plt.close()

# Top 2 Dependence Plot
plt.figure(figsize=(8, 6), dpi=300)
shap.dependence_plot(top_2_feat, sv, X_all, show=False)
plt.title(f"SHAP Dependence Plot: {top_2_feat}", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("shap_dependence_top2.png", bbox_inches='tight')
plt.close()

print("Execution complete. Output files generated:")
print(" - 'shap_feature_importance.csv'")
print(" - 'shap_summary_beeswarm.png'")
print(" - 'shap_summary_bar.png'")
print(" - 'shap_dependence_top1.png'")
print(" - 'shap_dependence_top2.png'")

