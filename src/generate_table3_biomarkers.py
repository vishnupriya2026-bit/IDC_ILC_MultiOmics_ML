import os
import pandas as pd

os.makedirs("reports/tables", exist_ok=True)

# Compile Master Biomarker Data
table3_data = [
    {"Rank": 1, "Feature_ID": "ECADHERIN", "Gene_Symbol": "CDH1", "Modality": "RPPA Proteomics", "Mean_SHAP": 0.1284, "SHAP_Impact": "Loss drives ILC", "Network_Role": "Master Adhesion Hub (k=14)"},
    {"Rank": 2, "Feature_ID": "BETACATENIN", "Gene_Symbol": "CTNNB1", "Modality": "RPPA Proteomics", "Mean_SHAP": 0.0966, "SHAP_Impact": "Loss drives ILC", "Network_Role": "Adherens Complex Hub (k=12)"},
    {"Rank": 3, "Feature_ID": "ENSG00000111404.7", "Gene_Symbol": "ML-Candidate-1", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0285, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Predictive Driver (k=4)"},
    {"Rank": 4, "Feature_ID": "ENSG00000154330.13", "Gene_Symbol": "ML-Candidate-2", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0181, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Predictive Driver (k=3)"},
    {"Rank": 5, "Feature_ID": "AR", "Gene_Symbol": "AR", "Modality": "RPPA Proteomics", "Mean_SHAP": 0.0111, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Hormone Signaling Hub (k=8)"},
    {"Rank": 6, "Feature_ID": "ENSG00000241644.2", "Gene_Symbol": "ML-Candidate-3", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0109, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Co-expressed Transcript (k=2)"},
    {"Rank": 7, "Feature_ID": "ENSG00000149451.18", "Gene_Symbol": "ML-Candidate-4", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0078, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Co-expressed Transcript (k=3)"},
    {"Rank": 8, "Feature_ID": "MR1", "Gene_Symbol": "MR1", "Modality": "RPPA Proteomics", "Mean_SHAP": 0.0078, "SHAP_Impact": "Low expression drives ILC", "Network_Role": "Immune Module Peripheral (k=2)"},
    {"Rank": 9, "Feature_ID": "EGFR", "Gene_Symbol": "EGFR", "Modality": "RPPA Proteomics", "Mean_SHAP": 0.0078, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Structural Receptor Hub (k=10)"},
    {"Rank": 10, "Feature_ID": "ENSG00000161544.10", "Gene_Symbol": "ML-Candidate-5", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0077, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Transcriptomic Driver (k=3)"},
    {"Rank": 11, "Feature_ID": "cg17196051", "Gene_Symbol": "CDH1_Meth", "Modality": "DNA Methylation", "Mean_SHAP": 0.0072, "SHAP_Impact": "High methylation drives ILC", "Network_Role": "Epigenetic Repressor Site (k=1)"},
    {"Rank": 12, "Feature_ID": "ENSG00000124664.11", "Gene_Symbol": "ML-Candidate-6", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0070, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Transcriptomic Driver (k=2)"},
    {"Rank": 13, "Feature_ID": "ENSG00000187260.16", "Gene_Symbol": "ML-Candidate-7", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0068, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Transcriptomic Driver (k=2)"},
    {"Rank": 14, "Feature_ID": "ENSG00000064300.9", "Gene_Symbol": "ML-Candidate-8", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0065, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Secondary Candidate (k=1)"},
    {"Rank": 15, "Feature_ID": "ENSG00000072163.20", "Gene_Symbol": "ML-Candidate-9", "Modality": "RNA-seq Transcriptomics", "Mean_SHAP": 0.0062, "SHAP_Impact": "High expression drives ILC", "Network_Role": "Secondary Candidate (k=1)"}
]

df_t3 = pd.DataFrame(table3_data)
csv_out = "reports/tables/table_3_top_biomarkers.csv"
df_t3.to_csv(csv_out, index=False)
print(f"Saved Table 3 to '{csv_out}' ({len(df_t3)} rows)")
print(df_t3[['Rank', 'Feature_ID', 'Modality', 'Mean_SHAP', 'Network_Role']].head(5).to_string(index=False))
