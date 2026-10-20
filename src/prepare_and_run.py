import os
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
# STEP A: DATA PREPARATION & MATCHING
# ==========================================
def prepare_integrated_dataset():
    print("[1/5] Checking and preparing CPTAC multi-omics datasets...")
    
    rna_path = "/home/priya_26phd0306/data/processed/cptac_rna_clean.csv"
    prot_path = "/home/priya_26phd0306/data/processed/cptac_proteomics_clean.csv"
    clin_path = "/home/priya_26phd0306/BreastCancer_MultiOmics/cptac_119_clinical_final.csv"
    
    # Load data
    rna_df = pd.read_csv(rna_path, index_col=0)
    prot_df = pd.read_csv(prot_path, index_col=0)
    clin_df = pd.read_csv(clin_path, index_col=0)
    
    print(f"  - Loaded RNA: {rna_df.shape}")
    print(f"  - Loaded Proteomics: {prot_df.shape}")
    print(f"  - Loaded Clinical: {clin_df.shape}")
    
    # Prefix feature names for modality tracking
    rna_df.columns = [c if c.startswith('RNA_') else f"RNA_{c}" for c in rna_df.columns]
    prot_df.columns = [c if c.startswith('PROT_') else f"PROT_{c}" for c in prot_df.columns]
    
    # Find clinical target column
    target_candidates = ['Response', 'Subtype', 'PAM50', 'PAM50_Subtype', 'PAM50_mRNA', 'ER_Status', 'status', 'vital_status']
    target_col = None
    for col in target_candidates:
        if col in clin_df.columns:
            target_col = col
            break
            
    if target_col is None:
        # Fallback to the last column if no standard name matched
        target_col = clin_df.columns[-1]
    print(f"  - Using Target Clinical Variable: '{target_col}'")
    
    # Match patient samples across datasets
    common_samples = rna_df.index.intersection(prot_df.index).intersection(clin_df.index)
    if len(common_samples) == 0:
        # Try matching by reformatting index (strip whitespace/case)
        rna_df.index = rna_df.index.astype(str).str.strip()
        prot_df.index = prot_df.index.astype(str).str.strip()
        clin_df.index = clin_df.index.astype(str).str.strip()
        common_samples = rna_df.index.intersection(prot_df.index).intersection(clin_df.index)
        
    print(f"  - Matched {len(common_samples)} samples across all modalities.")
    
    # Combine datasets
    y = clin_df.loc[common_samples, target_col]
    
    # Binarize target if categorical (e.g., LumA vs others or Response 1 vs 0)
    if y.dtype == 'object' or len(np.unique(y)) > 2:
        top_class = y.value_counts().index[0]
        y = (y == top_class).astype(int)
        print(f"  - Target variable binarized: 1 = '{top_class}', 0 = Others")
        
    combined_df = pd.concat([rna_df.loc[common_samples], prot_df.loc[common_samples]], axis=1)
    combined_df['Response'] = y.values
    
    os.makedirs("/home/priya_26phd0306/data", exist_ok=True)
    out_file = "/home/priya_26phd0306/data/multiomics_integrated_matrix.csv"
    combined_df.to_csv(out_file)
    print(f"  - Integrated Multi-Omics dataset saved to: {out_file}\n")
    return out_file, 'Response'

# ==========================================
# STEP B: METRICS & BOOTSTRAPPING
# ==========================================
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_metrics(y_true, y_prob, y_pred):
    return {
        'ROC_AUC': roc_auc_score(y_true, y_prob),
        'PR_AUC': average_precision_score(y_true, y_prob),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': calculate_specificity(y_true, y_pred),
        'Balanced_Acc': balanced_accuracy_score(y_true, y_pred)
    }

def bootstrap_confidence_intervals(y_true, y_prob, y_pred, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    boot_metrics = {k: [] for k in ['ROC_AUC', 'PR_AUC', 'F1_Score', 'Sensitivity', 'Specificity', 'Balanced_Acc']}
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
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
# STEP C: LEAKAGE-SAFE BENCHMARK ENGINE
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
            
        ci_summary = bootstrap_confidence_intervals(y_trues, y_probs, y_preds)
        ci_summary['Model'] = name
        results.append(ci_summary)
        
        pipe.fit(X, y)
        fitted_pipelines[name] = pipe

    return pd.DataFrame(results), fitted_pipelines

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Prepare data
    data_path, target_col = prepare_integrated_dataset()
    
    # 2. Load integrated matrix
    print("[2/5] Loading integrated multi-omics matrix...")
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    modalities = {
        'RNA_Only': [c for c in X.columns if c.startswith('RNA_')],
        'Protein_Only': [c for c in X.columns if c.startswith('PROT_')],
        'Integrated_MultiOmics': list(X.columns)
    }
    
    all_results = []
    best_overall_score = -1.0
    best_pipeline = None
    best_config = ""
    
    # 3. Execute Benchmarking (Tasks 1, 2, 3)
    print("[3/5] Running Leakage-Safe Multi-Omics Benchmarks (Tasks 1, 2, 3)...")
    for mod_name, cols in modalities.items():
        if len(cols) == 0:
            print(f"Skipping {mod_name} (0 features found matching prefix).")
            continue
            
        print(f"\nEvaluating Modality Group: {mod_name} ({len(cols)} features)...")
        df_res, pipelines = run_leakage_safe_benchmark(X[cols], y)
        df_res['Modality'] = mod_name
        all_results.append(df_res)
        
        for idx, row in df_res.iterrows():
            if row['ROC_AUC_mean'] > best_overall_score:
                best_overall_score = row['ROC_AUC_mean']
                best_pipeline = pipelines[row['Model']]
                best_config = f"{mod_name} + {row['Model']}"

    # 4. Save Benchmark CSV
    print("\n[4/5] Saving benchmark results with 95% Confidence Intervals...")
    final_res = pd.concat(all_results, ignore_index=True)
    out_csv = "/home/priya_26phd0306/results_tasks_1_2_3_benchmarks.csv"
    final_res.to_csv(out_csv, index=False)
    
    # 5. Lock Model
    print("[5/5] Locking best multi-omics model...")
    joblib.dump(best_pipeline, "/home/priya_26phd0306/best_locked_model.pkl")
    
    print("\n=======================================================")
    print(" TASKS 1, 2, 3 COMPLETED SUCCESSFULLY!")
    print(f" Benchmark Summary Table: {out_csv}")
    print(f" Locked Best Model Saved: /home/priya_26phd0306/best_locked_model.pkl")
    print(f" Top Performing Configuration: {best_config} (Mean ROC-AUC: {best_overall_score:.3f})")
    print("=======================================================")
