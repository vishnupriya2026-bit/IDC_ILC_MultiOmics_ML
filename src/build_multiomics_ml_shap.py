import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

print("==========================================================")
print("  MULTI-OMICS (RNA + PROTEOMICS) HIGH-PERFORMANCE ML & SHAP")
print("==========================================================")

# 1. Load Clinical Targets
clinical = pd.read_csv("cptac_119_clinical_final.csv", index_col=0)
labeled_clinical = clinical.dropna(subset=['Target_Stage_Binary']).copy()
labeled_clinical['Target_Stage_Binary'] = labeled_clinical['Target_Stage_Binary'].astype(int)
labeled_clinical = labeled_clinical[~labeled_clinical.index.duplicated(keep='first')]

y = labeled_clinical['Target_Stage_Binary']
target_patients = set(labeled_clinical.index.tolist())
print(f"Loaded {len(target_patients)} clinically labeled target patients.")

# 2. Fast Loading & Orientation of Omics Data
rna_path = "BreastCancer_MultiOmics/data/processed/CPTAC_RNA.csv"
prot_path = "BreastCancer_MultiOmics/data/processed/CPTAC_Proteomics.csv"

def load_and_align_omics(file_path, prefix):
    print(f"Loading '{file_path}'...")
    df = pd.read_csv(file_path, index_col=0)
    
    # Check orientation (Patients in rows vs columns)
    idx_overlap = len(target_patients.intersection(set(df.index.astype(str))))
    col_overlap = len(target_patients.intersection(set(df.columns.astype(str))))
    
    if col_overlap > idx_overlap:
        df = df.T
        
    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.duplicated(keep='first')]
    df = df.add_prefix(prefix)
    return df

df_rna = load_and_align_omics(rna_path, "RNA_")
df_prot = load_and_align_omics(prot_path, "PROT_")

# Find Matched Patients across Clinical, RNA, and Proteomics
matched_ids = [p for p in labeled_clinical.index if p in df_rna.index and p in df_prot.index]
print(f"\nSuccessfully matched {len(matched_ids)} patients across Clinical, RNA, and Proteomics datasets.")

# Subset and Align
X_rna = df_rna.loc[matched_ids]
X_prot = df_prot.loc[matched_ids]
y_matched = y.loc[matched_ids]

# Merge into raw NumPy matrix for speed
all_feature_names = np.array(list(X_rna.columns) + list(X_prot.columns))
X_raw = np.hstack([X_rna.values, X_prot.values])

print(f"Combined Raw Multi-Omics Matrix: {X_raw.shape[0]} Patients x {X_raw.shape[1]} Features")

# 3. Blazingly Fast Imputation & Variance Filtering (C-compiled)
print("\nPerforming C-optimized median imputation...")
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X_raw)

print("Filtering low-variance features...")
vt = VarianceThreshold(threshold=0.01)
X_clean = vt.fit_transform(X_imp)
clean_features = all_feature_names[vt.get_support()]

print(f"Clean Multi-Omics Feature Matrix: {X_clean.shape[0]} Patients x {X_clean.shape[1]} Features")
rna_count = sum(f.startswith("RNA_") for f in clean_features)
prot_count = sum(f.startswith("PROT_") for f in clean_features)
print(f" -> Breakdown: {rna_count} RNA Transcripts | {prot_count} Proteomics Features")

# 4. Leakage-Controlled 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', C=0.2, max_iter=200, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, C=1.0, kernel='rbf', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
}

results = {m: {"auc": [], "f1": [], "acc": []} for m in models}
feature_selection_counts = {}

print("\nStarting 5-Fold Stratified CV...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_clean, y_matched), 1):
    print(f" -> Processing Fold {fold}/5...", end="", flush=True)
    X_train, X_test = X_clean[train_idx], X_clean[test_idx]
    y_train, y_test = y_matched.iloc[train_idx].values, y_matched.iloc[test_idx].values
    
    # Step A: Scaling inside fold
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    
    # Step B: Fast ANOVA Pre-filter (Top 1,000 features inside fold)
    k_best = min(1000, X_clean.shape[1])
    anova = SelectKBest(score_func=f_classif, k=k_best)
    X_tr_anova = anova.fit_transform(X_tr_sc, y_train)
    X_te_anova = anova.transform(X_te_sc)
    anova_feats = clean_features[anova.get_support()]
    
    # Step C: LASSO Biomarker Selection (Top 30 inside fold)
    lasso = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=0.3, random_state=42),
        max_features=30
    )
    X_tr_sel = lasso.fit_transform(X_tr_anova, y_train)
    X_te_sel = lasso.transform(X_te_anova)
    
    final_feats = anova_feats[lasso.get_support()]
    for f in final_feats:
        feature_selection_counts[f] = feature_selection_counts.get(f, 0) + 1
        
    # Step D: Model Benchmarking
    for name, clf in models.items():
        clf.fit(X_tr_sel, y_train)
        preds = clf.predict(X_te_sel)
        probs = clf.predict_proba(X_te_sel)[:, 1] if hasattr(clf, "predict_proba") else preds
        
        results[name]["auc"].append(roc_auc_score(y_test, probs))
        results[name]["f1"].append(f1_score(y_test, preds, zero_division=0))
        results[name]["acc"].append(accuracy_score(y_test, preds))
    print(" Done!")

# 5. Benchmark Output
print("\n" + "="*62)
print(f"{'Model Name':<28} | {'ROC-AUC':<8} | {'F1-Score':<8} | {'Accuracy':<8}")
print("="*62)

summary_rows = []
for name, metrics in results.items():
    mean_auc = np.mean(metrics["auc"])
    mean_f1 = np.mean(metrics["f1"])
    mean_acc = np.mean(metrics["acc"])
    print(f"{name:<28} | {mean_auc:.4f}   | {mean_f1:.4f}   | {mean_acc:.4f}")
    summary_rows.append({"Model": name, "ROC_AUC": mean_auc, "F1_Score": mean_f1, "Accuracy": mean_acc})

pd.DataFrame(summary_rows).to_csv("ml_multiomics_benchmarks.csv", index=False)

# 6. SHAP Explainable AI Analysis
print("\n" + "="*62)
print("  EXECUTING SHAP EXPLAINABLE AI ANALYSIS")
print("="*62)

top_features = pd.Series(feature_selection_counts).sort_values(ascending=False).head(20).index.tolist()
print(f"Top Multi-Omics Features Prioritized Across Folds:\n{top_features}")

# Subset dataset to top features for SHAP evaluation
top_indices = [np.where(clean_features == f)[0][0] for f in top_features]
X_top = X_clean[:, top_indices]

scaler = StandardScaler()
X_top_sc = scaler.fit_transform(X_top)

final_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.3, random_state=42)
final_model.fit(X_top_sc, y_matched.values)

try:
    import shap
    explainer = shap.LinearExplainer(final_model, X_top_sc)
    shap_values = explainer.shap_values(X_top_sc)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"Biomarker": top_features, "SHAP_Importance": mean_abs_shap})
    shap_df = shap_df.sort_values("SHAP_Importance", ascending=False)
    
    print("\nPrioritized Multi-Omics Biomarkers Ranked by SHAP Importance:")
    print(shap_df.to_string(index=False))
    
    shap_df.to_csv("prioritized_multiomics_biomarkers_shap.csv", index=False)
    print("\nSaved SHAP prioritized biomarkers to 'prioritized_multiomics_biomarkers_shap.csv'")
except Exception as e:
    print(f"\nSHAP note: {e}")
    weights = np.abs(final_model.coef_[0])
    weight_df = pd.DataFrame({"Biomarker": top_features, "Feature_Weight": weights}).sort_values("Feature_Weight", ascending=False)
    weight_df.to_csv("prioritized_multiomics_biomarkers_shap.csv", index=False)
    print("Saved feature weight rankings to 'prioritized_multiomics_biomarkers_shap.csv'")

