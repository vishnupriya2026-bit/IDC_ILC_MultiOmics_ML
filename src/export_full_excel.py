import pandas as pd
from pathlib import Path

base = Path.home() / 'phd_research'
ml = base / 'data' / 'processed' / 'ml_ready'
excel_dir = base / 'data' / 'processed' / 'excel_exports'
excel_dir.mkdir(parents=True, exist_ok=True)

print("[1/4] Exporting Matched Cohort & Patient IDs (target_y)...")
target = pd.read_csv(ml / 'target_y.csv', index_col=0)
target.to_excel(excel_dir / 'matched_cohort_samples.xlsx')
print(f"      Saved: matched_cohort_samples.xlsx ({target.shape[0]} samples)")

print("[2/4] Exporting Full RNA-Seq Dataset...")
rna = pd.read_csv(ml / 'rna_seq_filtered.csv', index_col=0)
rna.to_excel(excel_dir / 'rna_seq_full.xlsx')
print(f"      Saved: rna_seq_full.xlsx ({rna.shape[0]} samples x {rna.shape[1]} genes)")

print("[3/4] Exporting Full RPPA Dataset...")
rppa = pd.read_csv(ml / 'rppa_filtered.csv', index_col=0)
rppa.to_excel(excel_dir / 'rppa_full.xlsx')
print(f"      Saved: rppa_full.xlsx ({rppa.shape[0]} samples x {rppa.shape[1]} antibodies)")

print("[4/4] Exporting Full Methylation Dataset (Transposed for 100% feature coverage)...")
meth = pd.read_csv(ml / 'methylation_filtered.csv', index_col=0)
meth_transposed = meth.T
meth_transposed.to_excel(excel_dir / 'methylation_full.xlsx')
print(f"      Saved: methylation_full.xlsx ({meth_transposed.shape[0]} CpG sites x {meth_transposed.shape[1]} samples)")

print("\n=== ALL FULL EXCEL EXPORTS COMPLETE ===")
print(f"Location: {excel_dir}")
