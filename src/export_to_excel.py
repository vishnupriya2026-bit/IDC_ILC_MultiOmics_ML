import pandas as pd
from pathlib import Path

base = Path.home() / 'phd_research'
proc = base / 'data' / 'processed'
ml = proc / 'ml_ready'
excel_dir = proc / 'excel_exports'
excel_dir.mkdir(parents=True, exist_ok=True)

combined_excel = excel_dir / 'multiomics_datasets.xlsx'

datasets = {
    'Target_Y': ml / 'target_y.csv',
    'RPPA': ml / 'rppa_filtered.csv',
    'RNA_Seq': ml / 'rna_seq_filtered.csv',
    'Methylation': ml / 'methylation_filtered.csv'
}

print("[1/2] Exporting multi-omics matrices to Excel...")

with pd.ExcelWriter(combined_excel, engine='openpyxl') as writer:
    for sheet_name, csv_path in datasets.items():
        if csv_path.exists():
            print(f"      Processing {sheet_name}...")
            df = pd.read_csv(csv_path, index_col=0)
            
            # Excel limits: 1,048,576 rows x 16,384 columns
            if df.shape[1] > 16000:
                print(f"      Notice: {sheet_name} exceeds Excel column limits ({df.shape[1]} cols). Saving first 16,000 features to sheet and full CSV separately.")
                df.iloc[:, :16000].to_excel(writer, sheet_name=sheet_name)
            else:
                df.to_excel(writer, sheet_name=sheet_name)
                
            # Also save individual Excel file per dataset
            indiv_excel = excel_dir / f"{sheet_name.lower()}.xlsx"
            if df.shape[1] <= 16000:
                df.to_excel(indiv_excel)
        else:
            print(f"      Skipped: {csv_path.name} not found.")

print(f"\n[2/2] Export complete! Excel files created in:")
print(f"      {excel_dir}")
