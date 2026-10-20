import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 1. Load Data
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)

# Align and clean labels
y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_train_raw = y_train_df.reindex(X_train.index).iloc[:, 0].astype(str)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# 2. Fit Random Forest & XGBoost for Biomarker Ranking
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 3. Aggregate Feature Importances
biomarker_df = pd.DataFrame({
    'Feature_ID': X_train.columns,
    'RF_Importance': rf.feature_importances_,
    'XGB_Importance': xgb_model.feature_importances_
})

biomarker_df['Combined_Score'] = (biomarker_df['RF_Importance'] + biomarker_df['XGB_Importance']) / 2
biomarker_df = biomarker_df.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)

# Save full ranked list
biomarker_df.to_csv("top_discovered_biomarkers.csv", index=False)

print("\n=== TOP 15 DISCOVERED MULTI-OMICS BIOMARKERS ===")
print(biomarker_df.head(15).to_string(index=False))

# 4. Plot Top 20 Biomarkers
plt.figure(figsize=(10, 8), dpi=300)
sns.barplot(
    data=biomarker_df.head(20), 
    x='Combined_Score', 
    y='Feature_ID', 
    palette='viridis'
)
plt.title('Top 20 Multi-Omics Biomarkers Identified by Model Importance')
plt.xlabel('Combined Relative Feature Importance Score')
plt.ylabel('Omics Feature / Ensembl ID')
plt.tight_layout()
plt.savefig("top_biomarkers_plot.png")
print("\nPlot saved as 'top_biomarkers_plot.png' and full list saved as 'top_discovered_biomarkers.csv'.")

