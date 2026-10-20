import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, roc_curve, confusion_matrix
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def run_ml_benchmark():
    int_matrix_file = "results/feature_selection/integrated_feature_matrix.csv"
    rna_file = "data/processed/CPTAC_RNA.csv"
    
    out_dir = "results/ml_models"
    fig_dir = "results/publication_figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Loading integrated multi-omics feature matrix...")
    X_df = pd.read_csv(int_matrix_file, index_col=0)
    rna_df = pd.read_csv(rna_file, index_col=0)
    
    # Align labels with ESR1 median split
    target_gene = "ESR1" if "ESR1" in rna_df.columns else rna_df.columns[0]
    common_ids = X_df.index.intersection(rna_df.index)
    
    X = X_df.loc[common_ids].values
    y = (rna_df.loc[common_ids, target_gene] >= rna_df.loc[common_ids, target_gene].median()).astype(int).values
    
    # 80:20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define Classifiers
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(eval_metric='logloss', random_state=42)
    if HAS_LGB:
        models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
    if HAS_CAT:
        models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=42)
        
    metrics_list = []
    plt.figure(figsize=(8, 6), dpi=300)
    
    for name, model in models.items():
        print(f"Training and evaluating {name} (5-Fold CV)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            m = model
            m.fit(X_train_scaled[train_idx], y_train[train_idx])
            preds = m.predict(X_train_scaled[val_idx])
            probs = m.predict_proba(X_train_scaled[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train[val_idx], probs))
            
        # Final test evaluation
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        metrics_list.append({
            'Model': name,
            'CV ROC-AUC': np.mean(cv_scores),
            'Test Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'PR-AUC': pr_auc,
            'MCC': mcc,
            'Cohen Kappa': kappa
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('1 - Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title('Multi-Model ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ml_models_roc_curves.png")
    plt.close()
    
    results_df = pd.DataFrame(metrics_list)
    results_df.to_csv(f"{out_dir}/model_performance_summary.csv", index=False)
    print(f"\nResults saved to {out_dir}/model_performance_summary.csv")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_ml_benchmark()
