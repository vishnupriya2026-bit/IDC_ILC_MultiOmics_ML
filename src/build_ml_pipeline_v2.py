import pandas as pd
import numpy as np
import glob
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

print("==========================================================")
print("  LEAKAGE-CONTROLLED MULTI-OMICS ML PIPELINE (OPTIMIZED)")
print("==========================================================")

# 1. Load Clinical Targets
clinical = pd.read_csv("cptac_119_clinical_final.csv", index_col=0)
labeled_clinical = clinical.dropna(subset=['Target_Stage_Binary']).copy()
labeled_clinical['Target_Stage_Binary'] = labeled_clinical['Target_Stage_Binary'].astype(int)
labeled_clinical = labeled_clinical[~labeled_clinical.index.duplicated(keep='first')]

y = labeled_clinical['Target_Stage_Binary']
patient_ids = labeled_clinical.index.tolist()
print(f"Loaded {len(patient_ids)} clinically labeled patients.")
print(f"Class Distribution -> Early Stage (0): {sum(y == 0)}, Advanced Stage (1): {sum(y == 1)}")

# 2. Locate Multi-Omics Expression Matrix
search_files = [
    "BreastCancer_MultiOmics/data/processed/CPTAC_RNA.csv",
    "BreastCancer_MultiOmics/data/processed/CPTAC_Proteomics.csv",
    "multiomics_latent_embeddings.csv"
]

matrix_file = None
for f in search_files:
    if os.path.exists(f):
        matrix_file = f
        break

if not matrix_file:
    # Fallback to searching all CSVs
    all_csvs = glob.glob("**/*.csv", recursive=True)
    matrix_file = [c for c in all_csvs if "rna" in c.lower() or "proteomics" in c.lower()][0]

print(f"\nLoading Omics Matrix: '{matrix_file}'")
df_expr = pd.read_csv(matrix_file, index_col=0)

# Detect Orientation (Ensure Patients in Rows, Features in Columns)
patient_overlap_idx = len(set(patient_ids).intersection(set(df_expr.index.astype(str))))
patient_overlap_cols = len(set(patient_ids).intersection(set(df_expr.columns.astype(str))))

if patient_overlap_cols > patient_overlap_idx:
    print("Transposing matrix (Patients identified in columns)...")
    df_expr = df_expr.T

df_expr.index = df_expr.index.astype(str).str.strip()
df_expr = df_expr[~df_expr.index.duplicated(keep='first')]
df_expr = df_expr.loc[:, ~df_expr.columns.duplicated(keep='first')]

# Align matrix to labeled patients
X = df_expr.reindex(patient_ids).dropna(how='all')
y = y.loc[X.index]

# Clean Numeric Values
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median()).fillna(0)

# Quick Global Variance Filter (Keep features with variance > 0)
variances = X.var()
X = X.loc[:, variances > 0.01]
print(f"Dataset aligned: {X.shape[0]} Patients x {X.shape[1]} High-Variance Features")

# 3. Fast Leakage-Controlled CV Pipeline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=0.2, max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, C=1.0, kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
}

results = {m: {"auc": [], "f1": [], "acc": []} for m in models}
feature_selection_counts = {}

print("\nStarting 5-Fold Stratified CV (with progress tracking)...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f" -> Processing Fold {fold}/5...", end="", flush=True)
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Step A: Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step B: Fast ANOVA Pre-filter (Top 1,000 features inside fold)
    k_best = min(1000, X_train.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    X_test_anova = anova_selector.transform(X_test_scaled)
    
    anova_features = X.columns[anova_selector.get_support()]
    
    # Step C: Inside-Fold LASSO Selection (Top 30 biomarkers)
    lasso_selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.3, random_state=42),
        max_features=30
    )
    X_train_sel = lasso_selector.fit_transform(X_train_anova, y_train)
    X_test_sel = lasso_selector.transform(X_test_anova)
    
    final_features = anova_features[lasso_selector.get_support()].tolist()
    for feat in final_features:
        feature_selection_counts[feat] = feature_selection_counts.get(feat, 0) + 1
        
    # Step D: Model Evaluation
    for name, clf in models.items():
        clf.fit(X_train_sel, y_train)
        preds = clf.predict(X_test_sel)
        probs = clf.predict_proba(X_test_sel)[:, 1] if hasattr(clf, "predict_proba") else preds
        
        results[name]["auc"].append(roc_auc_score(y_test, probs))
        results[name]["f1"].append(f1_score(y_test, preds, zero_division=0))
        results[name]["acc"].append(accuracy_score(y_test, preds))
        
    print(" Done!")

# 4. Results Display
print("\n" + "="*58)
print(f"{'Model Name':<28} | {'ROC-AUC':<8} | {'F1-Score':<8} | {'Accuracy':<8}")
print("="*58)

summary_rows = []
for name, metrics in results.items():
    mean_auc = np.mean(metrics["auc"])
    mean_f1 = np.mean(metrics["f1"])
    mean_acc = np.mean(metrics["acc"])
    print(f"{name:<28} | {mean_auc:.4f}   | {mean_f1:.4f}   | {mean_acc:.4f}")
    summary_rows.append({"Model": name, "ROC_AUC": mean_auc, "F1_Score": mean_f1, "Accuracy": mean_acc})

pd.DataFrame(summary_rows).to_csv("ml_model_benchmarks.csv", index=False)

# 5. Top Consistently Selected Biomarkers
print("\n" + "="*58)
print("  TOP CONSISTENTLY SELECTED BIOMARKERS (Across Folds)")
print("="*58)
top_features = pd.Series(feature_selection_counts).sort_values(ascending=False).head(15)
print(top_features.to_string())

top_features.to_csv("top_prioritized_biomarkers.csv")
print("\nSaved benchmark results to 'ml_model_benchmarks.csv'")
print("Saved top biomarkers to 'top_prioritized_biomarkers.csv'")
