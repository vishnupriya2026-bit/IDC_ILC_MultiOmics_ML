import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

print("==========================================================")
print("  LEAKAGE-CONTROLLED MULTI-OMICS ML PIPELINE")
print("==========================================================")

# 1. Load Clinical Targets
clinical = pd.read_csv("cptac_119_clinical_final.csv", index_col=0)
labeled_clinical = clinical.dropna(subset=['Target_Stage_Binary']).copy()
labeled_clinical['Target_Stage_Binary'] = labeled_clinical['Target_Stage_Binary'].astype(int)

# Deduplicate clinical index if needed
labeled_clinical = labeled_clinical[~labeled_clinical.index.duplicated(keep='first')]

y = labeled_clinical['Target_Stage_Binary']
patient_ids = labeled_clinical.index.tolist()
print(f"Loaded {len(patient_ids)} clinically labeled patients.")
print(f"Class Distribution -> Early Stage (0): {sum(y == 0)}, Advanced Stage (1): {sum(y == 1)}")

# 2. Smart Matrix File Identification
search_files = glob.glob("BreastCancer_MultiOmics/data/processed/*.csv") + glob.glob("*.csv")
ignore_terms = ["pathway", "enrichment", "ppi", "hub", "clinical", "matched", "differential", "deg", "dep", "results", "benchmark"]

candidate_files = []
for f in search_files:
    filename = os.path.basename(f).lower()
    if not any(term in filename for term in ignore_terms):
        candidate_files.append(f)

print(f"\nCandidate expression/omics matrices found: {candidate_files}")

matrix_file = None
best_match_count = 0

for f in candidate_files:
    try:
        temp_df = pd.read_csv(f, nrows=5, index_col=0)
        # Check matching patient IDs in index or columns
        matches_in_idx = len(set(patient_ids).intersection(set(temp_df.index.astype(str))))
        matches_in_cols = len(set(patient_ids).intersection(set(temp_df.columns.astype(str))))
        
        matches = max(matches_in_idx, matches_in_cols)
        if matches > best_match_count:
            best_match_count = matches
            matrix_file = f
    except Exception:
        continue

if not matrix_file:
    print("Error: Could not find an expression matrix matching patient IDs. Available CSVs:")
    for f in search_files:
        print("  -", f)
    exit()

print(f"\nSelected Expression Matrix: '{matrix_file}' (Matched {best_match_count} Patient IDs)")

# 3. Load & Clean Expression Matrix
df_expr = pd.read_csv(matrix_file, index_col=0)

# Check if orientation needs transposition (we want rows = patients, cols = features)
matches_in_idx = len(set(patient_ids).intersection(set(df_expr.index.astype(str))))
matches_in_cols = len(set(patient_ids).intersection(set(df_expr.columns.astype(str))))

if matches_in_cols > matches_in_idx:
    df_expr = df_expr.T

df_expr.index = df_expr.index.astype(str).str.strip()

# Deduplicate index (patients) and columns (genes/proteins)
df_expr = df_expr[~df_expr.index.duplicated(keep='first')]
df_expr = df_expr.loc[:, ~df_expr.columns.duplicated(keep='first')]

# Align matrix to labeled patients
X = df_expr.reindex(patient_ids).dropna(how='all')

# Re-align y to matching patients in X
y = y.loc[X.index]
print(f"Final Clean Matrix Shape: {X.shape[0]} Patients x {X.shape[1]} Features")

# Convert to numeric, coerce non-numeric to NaN, then fill NaNs with column median
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median()).fillna(0)

# Drop any constant features (zero variance across all samples)
X = X.loc[:, X.nunique() > 1]
print(f"Features after removing constant columns: {X.shape[1]}")

# 4. Leakage-Controlled 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, C=1.0, kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
}

results = {m: {"auc": [], "f1": [], "acc": []} for m in models}
feature_selection_counts = {}

print("\nExecuting 5-Fold Stratified CV with Fold-Internal Feature Selection...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # --- Step A: Standard Scaling (Fitted ONLY on training fold) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Step B: Inside-Fold Feature Selection via LASSO ---
    selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=42),
        max_features=30
    )
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    
    selected_features = X.columns[selector.get_support()].tolist()
    for feat in selected_features:
        feature_selection_counts[feat] = feature_selection_counts.get(feat, 0) + 1
        
    # --- Step C: Evaluate Models on Selected Features ---
    for name, clf in models.items():
        clf.fit(X_train_sel, y_train)
        preds = clf.predict(X_test_sel)
        probs = clf.predict_proba(X_test_sel)[:, 1] if hasattr(clf, "predict_proba") else preds
        
        results[name]["auc"].append(roc_auc_score(y_test, probs))
        results[name]["f1"].append(f1_score(y_test, preds, zero_division=0))
        results[name]["acc"].append(accuracy_score(y_test, preds))

# 5. Summary Output
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

# 6. Top Consistently Selected Biomarkers
print("\n" + "="*58)
print("  TOP CONSISTENTLY SELECTED BIOMARKERS (Across Folds)")
print("="*58)
top_features = pd.Series(feature_selection_counts).sort_values(ascending=False).head(15)
print(top_features.to_string())

top_features.to_csv("top_prioritized_biomarkers.csv")
print("\nSaved benchmark results to 'ml_model_benchmarks.csv'")
print("Saved top biomarkers to 'top_prioritized_biomarkers.csv'")
