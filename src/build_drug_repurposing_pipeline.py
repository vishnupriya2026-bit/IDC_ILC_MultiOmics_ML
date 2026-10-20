import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns

print("==========================================================")
print("  MULTI-OMICS EVIDENCE-WEIGHTED DRUG REPURPOSING PIPELINE")
print("==========================================================")

# 1. Load Clinical Targets & Matched Patient IDs
clinical = pd.read_csv("cptac_119_clinical_final.csv", index_col=0)
labeled_clinical = clinical.dropna(subset=['Target_Stage_Binary']).copy()
labeled_clinical['Target_Stage_Binary'] = labeled_clinical['Target_Stage_Binary'].astype(int)
labeled_clinical = labeled_clinical[~labeled_clinical.index.duplicated(keep='first')]

y = labeled_clinical['Target_Stage_Binary']
patient_ids = set(labeled_clinical.index.tolist())

# 2. Load RNA & Proteomics Matrices
rna_path = "BreastCancer_MultiOmics/data/processed/CPTAC_RNA.csv"
prot_path = "BreastCancer_MultiOmics/data/processed/CPTAC_Proteomics.csv"

def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    if len(patient_ids.intersection(set(df.columns.astype(str)))) > len(patient_ids.intersection(set(df.index.astype(str)))):
        df = df.T
    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.duplicated(keep='first')]
    return df

df_rna = load_matrix(rna_path)
df_prot = load_matrix(prot_path)

matched_patients = [p for p in labeled_clinical.index if p in df_rna.index and p in df_prot.index]
y_matched = y.loc[matched_patients].values

df_rna_m = df_rna.loc[matched_patients]
df_prot_m = df_prot.loc[matched_patients]

# 3. Load Drug Target Database (1,720 Entries)
drug_db_path = "BreastCancer_MultiOmics/data/processed/prioritized_drugs.csv"
df_drugs = pd.read_csv(drug_db_path)
print(f"Loaded {len(df_drugs)} drug-target interactions from '{drug_db_path}'")

df_drugs['Gene_Clean'] = df_drugs['Gene'].astype(str).str.strip().str.upper()

# Collect available genes
rna_genes = {str(col).strip().upper(): col for col in df_rna_m.columns}
prot_genes = {str(col).strip().upper(): col for col in df_prot_m.columns}

# Compute ANOVA F-scores across cohort
imputer = SimpleImputer(strategy='median')
X_rna_imp = imputer.fit_transform(df_rna_m.values)
f_rna, _ = f_classif(X_rna_imp, y_matched)
f_rna_dict = {str(col).strip().upper(): f_val for col, f_val in zip(df_rna_m.columns, np.nan_to_num(f_rna))}

X_prot_imp = imputer.fit_transform(df_prot_m.values)
f_prot, _ = f_classif(X_prot_imp, y_matched)
f_prot_dict = {str(col).strip().upper(): f_val for col, f_val in zip(df_prot_m.columns, np.nan_to_num(f_prot))}

print("\nMatching drug targets against cohort Multi-Omics profile...")

matched_target_scores = []

for idx, row in df_drugs.iterrows():
    gene = row['Gene_Clean']
    drug = row['Drug']
    interaction = row['Interaction_Type']
    sources = row['Sources']
    evidence_cnt = row['Evidence_Count']
    
    rna_f = f_rna_dict.get(gene, 0.0)
    prot_f = f_prot_dict.get(gene, 0.0)
    
    max_f = max(rna_f, prot_f)
    has_rna = gene in rna_genes
    has_prot = gene in prot_genes
    
    if has_rna or has_prot:
        norm_f = np.log1p(max_f)
        composite_score = evidence_cnt * (1.0 + norm_f)
        
        matched_target_scores.append({
            "Gene": gene,
            "Drug": drug,
            "Interaction_Type": interaction,
            "Evidence_Count": evidence_cnt,
            "RNA_Expressed": has_rna,
            "Proteomics_Expressed": has_prot,
            "ANOVA_F_Score": round(max_f, 4),
            "Composite_Priority_Score": round(composite_score, 4),
            "Sources": sources
        })

df_prioritized = pd.DataFrame(matched_target_scores)
df_prioritized = df_prioritized.sort_values("Composite_Priority_Score", ascending=False).drop_duplicates(subset=['Drug', 'Gene'])

# Save Output CSV
df_prioritized.to_csv("prioritized_repurposed_drugs_multiomics.csv", index=False)

print("\n" + "="*70)
print("  TOP PRIORITIZED REPURPOSED DRUGS (MULTI-OMICS + DATABASE EVIDENCE)")
print("="*70)
print(df_prioritized[['Gene', 'Drug', 'Interaction_Type', 'Evidence_Count', 'ANOVA_F_Score', 'Composite_Priority_Score']].head(15).to_string(index=False))
print(f"\nSaved top prioritized drugs to 'prioritized_repurposed_drugs_multiomics.csv'")

# 4. Generate Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

top_plot = df_prioritized.head(15).copy()
top_plot['Drug_Gene'] = top_plot['Drug'] + " (" + top_plot['Gene'] + ")"

plt.figure(figsize=(11, 6))
ax = sns.barplot(
    x='Composite_Priority_Score', 
    y='Drug_Gene', 
    data=top_plot, 
    hue='Drug_Gene',
    legend=False,
    palette='crest'
)

plt.title("Top Prioritized Candidate Repurposed Drugs for Breast Cancer", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Composite Multi-Omics Drug Priority Score", fontsize=12)
plt.ylabel("Candidate Drug (Target Gene)", fontsize=12)

plt.tight_layout()
plt.savefig("fig_top_repurposed_drugs.png", dpi=300)
plt.close()
print("Saved figure to 'fig_top_repurposed_drugs.png'")

