import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve, auc
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Set seed
np.random.seed(42)

# 1. Load Data
X_train = pd.read_csv("diablo_exports/X_diablo_train_scaled.csv", index_col=0)
y_train_df = pd.read_csv("diablo_exports/y_diablo_train_encoded.csv", index_col=0)

X_test = pd.read_csv("diablo_exports/X_diablo_test_scaled.csv", index_col=0)
y_test_df = pd.read_csv(
    "diablo_exports/y_test_encoded.csv" if "y_test_encoded.csv" in y_train_df.columns 
    else "diablo_exports/y_diablo_test_encoded.csv", index_col=0
)

# Deduplicate indices
y_train_df = y_train_df[~y_train_df.index.duplicated(keep='first')]
y_test_df = y_test_df[~y_test_df.index.duplicated(keep='first')]

y_train_raw = y_train_df.reindex(X_train.index).iloc[:, 0].astype(str)
y_test_raw = y_test_df.reindex(X_test.index).iloc[:, 0].astype(str)

# Combine into total cohort (N=410)
X_all = pd.concat([X_train, X_test], axis=0)
y_all_raw = pd.concat([y_train_raw, y_test_raw], axis=0)

le = LabelEncoder()
y_all = le.fit_transform(y_all_raw)

print(f"Combined Dataset Shape: {X_all.shape}")
print(f"Class Counts - {le.classes_[0]}: {np.sum(y_all == 0)}, {le.classes_[1]}: {np.sum(y_all == 1)}")

# 2. Setup 5-Fold x 5-Repeat Cross-Validation
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
scale_pos_weight = (len(y_all) - sum(y_all)) / sum(y_all)

model_factories = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
    "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1),
    "CatBoost": lambda: CatBoostClassifier(iterations=200, learning_rate=0.05, auto_class_weights='Balanced', random_seed=42, verbose=0)
}

cv_results = []
roc_curves_dict = {model_name: [] for model_name in model_factories.keys()}
feature_importance_folds = {feature: 0 for feature in X_all.columns}

mean_fpr = np.linspace(0, 1, 100)

print("\nStarting 25-Fold Cross-Validation Evaluation...")

for model_name, factory in model_factories.items():
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_all, y_all)):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]
        
        clf = factory()
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)
        
        fold_metrics.append({
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Brier Score": brier
        })
        
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_curves_dict[model_name].append(interp_tpr)
        
        if model_name == "Random Forest":
            top_fold_feats = pd.Series(clf.feature_importances_, index=X_all.columns).nlargest(20).index
            for feat in top_fold_feats:
                feature_importance_folds[feat] += 1

    df_fold = pd.DataFrame(fold_metrics)
    
    cv_results.append({
        "Model": model_name,
        "Accuracy": f"{df_fold['Accuracy'].mean():.4f} ± {df_fold['Accuracy'].std():.4f}",
        "Sensitivity": f"{df_fold['Sensitivity'].mean():.4f} ± {df_fold['Sensitivity'].std():.4f}",
        "Specificity": f"{df_fold['Specificity'].mean():.4f} ± {df_fold['Specificity'].std():.4f}",
        "Precision": f"{df_fold['Precision'].mean():.4f} ± {df_fold['Precision'].std():.4f}",
        "F1-Score": f"{df_fold['F1-Score'].mean():.4f} ± {df_fold['F1-Score'].std():.4f}",
        "ROC-AUC": f"{df_fold['ROC-AUC'].mean():.4f} ± {df_fold['ROC-AUC'].std():.4f}",
        "PR-AUC": f"{df_fold['PR-AUC'].mean():.4f} ± {df_fold['PR-AUC'].std():.4f}",
        "Brier Score": f"{df_fold['Brier Score'].mean():.4f} ± {df_fold['Brier Score'].std():.4f}",
        "Raw_AUC_Mean": df_fold['ROC-AUC'].mean()
    })

# 3. Save Metrics Summary
summary_df = pd.DataFrame(cv_results).sort_values(by="Raw_AUC_Mean", ascending=False).drop(columns=["Raw_AUC_Mean"])
summary_df.to_csv("cross_validation_results.csv", index=False)

print("\n" + "="*90)
print("             5-FOLD x 5-REPEAT CROSS-VALIDATION PERFORMANCE (MEAN ± STD)            ")
print("="*90)
print(summary_df.to_string(index=False))
print("="*90 + "\n")

# 4. Plot Mean ROC Curves with Confidence Bands
plt.figure(figsize=(8, 6), dpi=300)

for model_name, tprs in roc_curves_dict.items():
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc_val = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (Mean AUC = {mean_auc_val:.3f})", lw=2)
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), alpha=0.15)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Repeated 5-Fold CV Mean ROC Curves with ±1 SD Bounds')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cv_roc_curves.png")
print("Saved cross-validation ROC curves to 'cv_roc_curves.png'.")

# 5. Export Biomarker Selection Stability
stability_df = pd.DataFrame([
    {"Feature_ID": feat, "Selection_Frequency_25_Folds": count, "Stability_Score (%)": round((count / 25) * 100, 2)}
    for feat, count in feature_importance_folds.items()
]).sort_values(by="Selection_Frequency_25_Folds", ascending=False).head(20)

stability_df.to_csv("biomarker_stability_cv.csv", index=False)

print("\n=== TOP 10 CROSS-VALIDATED STABLE BIOMARKERS ===")
print(stability_df.head(10).to_string(index=False))
print("\nFull biomarker stability list saved to 'biomarker_stability_cv.csv'.\n")

