import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve
)

np.random.seed(42)

# 1. Load Data and Best Hyperparameters
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
print(f"Loaded Champion Random Forest Hyperparameters: {rf_best_params}")

# 2. Generate Out-of-Fold Probability Predictions via 10-Fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
champion_rf = RandomForestClassifier(**rf_best_params, class_weight='balanced', random_state=42)

oof_probs = cross_val_predict(champion_rf, X_all, y_all, cv=skf, method='predict_proba')[:, 1]

# 3. Process Parameter 1: Decision Threshold Optimization
thresholds = np.linspace(0.01, 0.99, 200)
threshold_metrics = []

for tau in thresholds:
    preds = (oof_probs >= tau).astype(int)
    cm = confusion_matrix(y_all, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_all, preds, zero_division=0)
    youden_j = sens + spec - 1
    
    threshold_metrics.append({
        'Threshold': tau,
        'Sensitivity': sens,
        'Specificity': spec,
        'F1-Score': f1,
        'Youden_J': youden_j
    })

df_thresh = pd.DataFrame(threshold_metrics)

# Select optimal threshold maximizing Youden's J Index
best_idx = df_thresh['Youden_J'].idxmax()
optimal_tau = df_thresh.loc[best_idx, 'Threshold']

# Calculate default vs optimal threshold metrics
def evaluate_at_threshold(tau):
    preds = (oof_probs >= tau).astype(int)
    cm = confusion_matrix(y_all, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Threshold": round(tau, 4),
        "Accuracy": round(accuracy_score(y_all, preds), 4),
        "Sensitivity (ILC Recall)": round(tp / (tp + fn), 4),
        "Specificity (IDC Recall)": round(tn / (tn + fp), 4),
        "Precision": round(precision_score(y_all, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_all, preds, zero_division=0), 4),
        "Brier Score (Calibration)": round(brier_score_loss(y_all, oof_probs), 4)
    }

metrics_default = evaluate_at_threshold(0.50)
metrics_optimal = evaluate_at_threshold(optimal_tau)

comparison_df = pd.DataFrame([
    {"Operating Point": "Default Threshold (τ = 0.50)", **metrics_default},
    {"Operating Point": f"Optimal Youden Threshold (τ* = {optimal_tau:.3f})", **metrics_optimal}
])

print("\n" + "="*85)
print("             PROCESS PARAMETER OPTIMIZATION: DECISION THRESHOLD CALIBRATION           ")
print("="*85)
print(comparison_df.to_string(index=False))
print("="*85 + "\n")

# Plot Threshold Calibration Curve
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(df_thresh['Threshold'], df_thresh['Sensitivity'], label='Sensitivity (ILC Recall)', color='crimson', lw=2)
plt.plot(df_thresh['Threshold'], df_thresh['Specificity'], label='Specificity (IDC Recall)', color='navy', lw=2)
plt.plot(df_thresh['Threshold'], df_thresh['F1-Score'], label='F1-Score', color='forestgreen', lw=2)
plt.plot(df_thresh['Threshold'], df_thresh['Youden_J'], label="Youden's J Index", color='darkorange', linestyle='--', lw=2)

plt.axvline(x=optimal_tau, color='black', linestyle=':', label=f'Optimal τ* = {optimal_tau:.3f}')
plt.xlabel('Probability Decision Threshold (τ)')
plt.ylabel('Metric Score')
plt.title('Champion Model Decision Threshold Calibration Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("threshold_calibration_curve.png")

# 4. Process Parameter 2: Biomarker Panel Parsimony Optimization (k features)
champion_rf.fit(X_all, y_all)
feature_importances = pd.Series(champion_rf.feature_importances_, index=X_all.columns).sort_values(ascending=False)

k_list = [5, 10, 15, 20, 25, 30, 45, 60, 80, 100, 159]
parsimony_results = []

for k in k_list:
    top_k_feats = feature_importances.head(k).index
    X_sub = X_all[top_k_feats]
    
    oof_probs_sub = cross_val_predict(champion_rf, X_sub, y_all, cv=skf, method='predict_proba')[:, 1]
    auc_k = roc_auc_score(y_all, oof_probs_sub)
    
    parsimony_results.append({'Top_k_Features': k, '10Fold_CV_ROC_AUC': round(auc_k, 4)})

df_parsimony = pd.DataFrame(parsimony_results)

# Save parsimony plot
plt.figure(figsize=(8, 4.5), dpi=300)
plt.plot(df_parsimony['Top_k_Features'], df_parsimony['10Fold_CV_ROC_AUC'], marker='o', color='teal', lw=2.5)
plt.axhline(y=metrics_default.get("ROC-AUC", roc_auc_score(y_all, oof_probs)), color='gray', linestyle='--', label='Full Feature Set AUC')
plt.xlabel('Number of Top Biomarkers Selected (k)')
plt.ylabel('10-Fold CV ROC-AUC')
plt.title('Biomarker Panel Parsimony Curve (Feature Subset Optimization)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("biomarker_parsimony_curve.png")

print("=== BIOMARKER PANEL PARSIMONY EVALUATION ===")
print(df_parsimony.to_string(index=False))

# Export optimal model configuration summary
final_config = {
    "Selected Champion Model": "Random Forest",
    "Best Bayesian Hyperparameters": rf_best_params,
    "Optimal Decision Threshold (tau)": float(optimal_tau),
    "Full Feature Set 10-Fold CV AUC": float(roc_auc_score(y_all, oof_probs)),
    "Parsimony Top k Features AUC": df_parsimony.to_dict(orient='records')
}

with open("champion_process_optimization.json", "w") as f:
    json.dump(final_config, f, indent=4)

print("\nSaved threshold plot to 'threshold_calibration_curve.png' and parsimony plot to 'biomarker_parsimony_curve.png'.")

