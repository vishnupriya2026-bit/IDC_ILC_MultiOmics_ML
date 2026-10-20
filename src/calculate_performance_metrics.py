import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss, roc_curve, precision_recall_curve
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Set reproducible seed
np.random.seed(42)

# 1. Load Data
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)
X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv(
    "diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns 
    else "diablo_exports/y_diablo_test_encoded.csv", index_col=0
)

# Align indices to handle DIABLO block duplications
y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train_raw = y_train_df.reindex(X_train.index).iloc[:, 0].astype(str)
y_test_raw = y_test_df.reindex(X_test.index).iloc[:, 0].astype(str)

# Encode string labels ('IDC', 'ILC') dynamically to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

print(f"Label Encoding Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# 2. Define Classifier Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.05, auto_class_weights='Balanced', random_seed=42, verbose=0)
}

results = []
roc_data = {}
pr_data = {}
cm_data = {}

# 3. Compute Comprehensive Performance Metrics
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix Breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_data[name] = cm
    
    # Specificity & Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Standard Classification Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Probability Error Metrics
    brier_mse = brier_score_loss(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Sensitivity (ILC Recall)": round(sensitivity, 4),
        "Specificity (IDC Recall)": round(specificity, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Brier Score (MSE)": round(brier_mse, 4),
        "Log Loss": round(logloss, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, roc_auc)
    
    p_prec, p_rec, _ = precision_recall_curve(y_test, y_prob)
    pr_data[name] = (p_rec, p_prec, pr_auc)

# 4. Save Metric Table
df_results = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
df_results.to_csv("performance_metrics_summary.csv", index=False)

print("\n" + "="*80)
print("                       MODEL PERFORMANCE METRICS SUMMARY                       ")
print("="*80)
print(df_results[["Model", "Accuracy", "Sensitivity (ILC Recall)", "Specificity (IDC Recall)", "F1-Score", "ROC-AUC", "PR-AUC", "Brier Score (MSE)"]].to_string(index=False))
print("="*80 + "\n")

# 5. Plot ROC & Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

for name in models.keys():
    fpr, tpr, auc_v = roc_data[name]
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_v:.3f})", lw=2)
    
    rec, prec, pr_auc_v = pr_data[name]
    axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc_v:.3f})", lw=2)

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
axes[0].set_xlabel('1 - Specificity (False Positive Rate)')
axes[0].set_ylabel('Sensitivity (True Positive Rate)')
axes[0].set_title('A: Receiver Operating Characteristic (ROC) Curve')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision')
axes[1].set_title('B: Precision-Recall Curve (PR-AUC)')
axes[1].legend(loc='lower left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_pr_curves.png")
print("Saved publication-ready curve plots to 'roc_pr_curves.png'.")

# 6. Plot Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
axes = axes.ravel()

class_labels = list(le.classes_)
for i, (name, cm) in enumerate(cm_data.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    axes[i].set_title(f"Confusion Matrix: {name}")
    axes[i].set_xlabel("Predicted Subtype")
    axes[i].set_ylabel("Actual Subtype")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
print("Saved confusion matrix plots to 'confusion_matrices.png'.\n")

