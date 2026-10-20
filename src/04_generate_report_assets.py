import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

base_dir = Path.home() / "phd_research"
ml_dir = base_dir / "data" / "processed" / "ml_ready"
fig_dir = base_dir / "reports" / "figures"
tbl_dir = base_dir / "reports" / "tables"

fig_dir.mkdir(parents=True, exist_ok=True)
tbl_dir.mkdir(parents=True, exist_ok=True)

print("[1/3] Loading ML-Ready Datasets...")
target = pd.read_csv(ml_dir / "target_y.csv", index_col=0)
rna = pd.read_csv(ml_dir / "rna_seq_filtered.csv", index_col=0)
meth = pd.read_csv(ml_dir / "methylation_filtered.csv", index_col=0)
rppa = pd.read_csv(ml_dir / "rppa_filtered.csv", index_col=0)

print("[2/3] Exporting Summary Tables...")
summary_rows = [
    {"Data Layer": "Matched Patients", "Raw Dimension": "410", "Filtered Dimension": "410", "Format": "Categorical Target"},
    {"Data Layer": "RNA-Seq (Gene Expression)", "Raw Dimension": "60,660", "Filtered Dimension": f"{rna.shape[1]:,}", "Format": "Log2 TPM"},
    {"Data Layer": "DNA Methylation (450k)", "Raw Dimension": "486,427", "Filtered Dimension": f"{meth.shape[1]:,}", "Format": "Beta Values (0-1)"},
    {"Data Layer": "RPPA (Proteomics)", "Raw Dimension": "487", "Filtered Dimension": f"{rppa.shape[1]:,}", "Format": "Normalized Intensity"}
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(tbl_dir / "table_1_cohort_summary.csv", index=False)

stage_counts = target['Y_Stage'].value_counts().reset_index()
stage_counts.columns = ['Pathologic Stage', 'Patient Count']
stage_counts.to_csv(tbl_dir / "table_2_stage_breakdown.csv", index=False)

print("[3/3] Generating Summary Figures...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=stage_counts, x='Pathologic Stage', y='Patient Count', hue='Pathologic Stage', ax=axes[0], palette='Blues_d', legend=False)
axes[0].set_title("A: Target Class Breakdown (Y_Stage)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Disease Stage", fontsize=10)
axes[0].set_ylabel("Number of Patients", fontsize=10)
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

omics_dims = pd.DataFrame({
    'Omics Layer': ['RNA-Seq', 'Methylation', 'RPPA'],
    'Features': [rna.shape[1], meth.shape[1], rppa.shape[1]]
})
sns.barplot(data=omics_dims, x='Omics Layer', y='Features', hue='Omics Layer', ax=axes[1], palette='viridis', legend=False)
axes[1].set_title("B: ML-Ready Feature Scale per Modality", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Omics Layer", fontsize=10)
axes[1].set_ylabel("Filtered Features", fontsize=10)
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
fig_path = fig_dir / "figure_1_cohort_overview.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"\nReport Assets Ready:")
print(f"• Summary Table : {tbl_dir / 'table_1_cohort_summary.csv'}")
print(f"• Stage Table   : {tbl_dir / 'table_2_stage_breakdown.csv'}")
print(f"• Figure Plot   : {fig_path}")
