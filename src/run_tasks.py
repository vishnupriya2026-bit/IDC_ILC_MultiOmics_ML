import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.utils import resample

# ==========================================
# 1. METRICS & BOOTSTRAPPING HELPER
# ==========================================
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_metrics(y_true, y_prob, y_pred):
    return {
        'ROC_AUC': roc_auc_score(y_true, y_prob),
        'PR_AUC': average_precision_score(y_true, y_prob),
        'F1_Score': f1_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': calculate_specificity(y_true, y_pred),
        'Balanced_Acc': balanced_accuracy_score(y_true, y_pred)
    }

def bootstrap_confidence_intervals(y_true, y_prob, y_pred, n_bootstraps=1000, seed=42):
    """Calculates 95% CIs via percentile bootstrapping."""
    np.random.seed(seed)
    boot_metrics = {k: [] for k in ['ROC_AUC', 'PR_AUC', 'F1_Score', 'Sensitivity', 'Specificity', 'Balanced_Acc']}
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip bootstraps that miss one class
        
        m = compute_metrics(y_true[indices], y_prob[indices], y_pred[indices])
        for k, v in m.items():
            boot_metrics[k].append(v)
            
    summary = {}
    for k, values in boot_metrics.items():
        mean_val = np.mean(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        summary[k] = f"{mean_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
        summary[f"{k}_mean"] = mean_val
    return summary

# ==========================================
# 2. LEAKAGE-SAFE EVALUATION ENGINE
# ==========================================
def run_leakage_safe_benchmark(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    classifiers = {
        'LASSO_Logistic': LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=seed),
        'SVM_RBF': SVC(kernel='rbf', probability=True, C=1.0, random_state=seed),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=seed, n_jobs=-1)
    }
    
    results = []
    fitted_pipelines = {}

    for name, clf in classifiers.items():
        print(f"    --> Fitting {name}...")
        
        # Pipeline strictly encapsulates fold-wise preprocessing
        pipe = Pipeline([
            ('variance', VarianceThreshold(threshold=0.01)),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', clf)
        ])
        
        y_trues, y_probs, y_preds = [], [], []
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            pipe.fit(X_tr, y_tr)
            
            y_probs.extend(pipe.predict_proba(X_va)[:, 1])
            y_preds.extend(pipe.predict(X_va))
            y_trues.extend(y_va)
            
        # Calculate full metrics + 95% CIs via bootstrapping
        ci_summary = bootstrap_confidence_intervals(y_trues, y_probs, y_preds)
        ci_summary['Model'] = name
        results.append(ci_summary)
        
        # Refit on full dataset for final model locking
        pipe.fit(X, y)
        fitted_pipelines[name] = pipe

    return pd.DataFrame(results), fitted_pipelines

# ==========================================
# 3. MAIN PIPELINE EXECUTION
# ==========================================
def execute_tasks_1_2_3(data_path, target_col='Response'):
    print("[1/3] Loading integrated data...")
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Classify modalities based on column naming prefixes
    modalities = {
        'RNA_Only': [c for c in X.columns if c.startswith(('RNA_', 'gene_'))],
        'Protein_Only': [c for c in X.columns if c.startswith(('PROT_', 'rppa_'))],
        'Integrated_MultiOmics': list(X.columns)
    }
    
    all_results = []
    best_overall_score = -1.0
    best_pipeline = None
    best_config = ""
    
    for mod_name, cols in modalities.items():
        if len(cols) == 0:
            print(f"Skipping {mod_name} (0 features found matching prefix).")
            continue
            
        print(f"\n[2/3] Running Modal Group: {mod_name} ({len(cols)} features)...")
        df_res, pipelines = run_leakage_safe_benchmark(X[cols], y)
        df_res['Modality'] = mod_name
        all_results.append(df_res)
        
        # Check for absolute best pipeline based on Mean ROC-AUC
        for idx, row in df_res.iterrows():
            if row['ROC_AUC_mean'] > best_overall_score:
                best_overall_score = row['ROC_AUC_mean']
                best_pipeline = pipelines[row['Model']]
                best_config = f"{mod_name} + {row['Model']}"

    # Export consolidated metric table with 95% CIs
    final_res = pd.concat(all_results, ignore_index=True)
    final_res.to_csv("results_tasks_1_2_3_benchmarks.csv", index=False)
    
    # Save the locked model for downstream Tasks 4-11
    joblib.dump(best_pipeline, "best_locked_model.pkl")
    print(f"\n[3/3] Tasks 1-3 COMPLETE!")
    print(f"  --> Consolidated Benchmark Table saved: 'results_tasks_1_2_3_benchmarks.csv'")
    print(f"  --> Best Locked Model Saved: 'best_locked_model.pkl' ({best_config} | ROC-AUC: {best_overall_score:.3f})")

if __name__ == "__main__":
    # UPDATE path and target column name to match your workstation file
    execute_tasks_1_2_3("data/multiomics_integrated_matrix.csv", target_col="Response")
