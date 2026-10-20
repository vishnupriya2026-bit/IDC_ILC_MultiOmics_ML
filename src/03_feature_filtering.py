import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

base_dir = Path.home() / "phd_research" / "data" / "processed"

print("[1/4] Loading Aligned Matrices...")
target = pd.read_csv(base_dir / "target_y_aligned.csv", index_col=0)
rna = pd.read_csv(base_dir / "rna_seq_aligned.csv", index_col=0)
meth = pd.read_csv(base_dir / "methylation_aligned.csv", index_col=0)
rppa = pd.read_csv(base_dir / "rppa_aligned.csv", index_col=0)

print("\n[2/4] Filtering RNA-Seq Genes...")
nonzero_mask = (rna > 0).mean(axis=0) >= 0.20
rna_filtered = rna.loc[:, nonzero_mask]
top_genes = rna_filtered.var(axis=0).nlargest(10000).index
rna_clean = rna_filtered[top_genes]
print(f"      RNA-Seq: Reduced from {rna.shape[1]:,} to {rna_clean.shape[1]:,} genes.")

print("\n[3/4] Filtering DNA Methylation Probes...")
meth_valid = meth.dropna(thresh=int(0.90 * len(meth)), axis=1)
imputer = SimpleImputer(strategy='median')
meth_imputed = pd.DataFrame(imputer.fit_transform(meth_valid), index=meth_valid.index, columns=meth_valid.columns)
top_probes = meth_imputed.var(axis=0).nlargest(20000).index
meth_clean = meth_imputed[top_probes]
print(f"      Methylation: Reduced from {meth.shape[1]:,} to {meth_clean.shape[1]:,} probes.")

print("\n[4/4] Processing RPPA Antibodies...")
rppa_num = rppa.apply(pd.to_numeric, errors='coerce')
# Retain columns with at least 50% valid numeric entries
valid_cols = rppa_num.columns[rppa_num.notnull().mean(axis=0) >= 0.50]
rppa_valid = rppa_num[valid_cols]

if rppa_valid.shape[1] > 0:
    rppa_imp = pd.DataFrame(imputer.fit_transform(rppa_valid), index=rppa_valid.index, columns=rppa_valid.columns)
    # Keep features with positive variance
    near_zero_var = rppa_imp.var(axis=0) <= 1e-8
    rppa_clean = rppa_imp.loc[:, ~near_zero_var]
    print(f"      RPPA: Retained {rppa_clean.shape[1]} antibodies (from {rppa.shape[1]} raw).")
else:
    print("      Warning: No valid numeric RPPA features found.")
    rppa_clean = rppa_valid

out_dir = base_dir / "ml_ready"
out_dir.mkdir(parents=True, exist_ok=True)

target.to_csv(out_dir / "target_y.csv")
rna_clean.to_csv(out_dir / "rna_seq_filtered.csv")
meth_clean.to_csv(out_dir / "methylation_filtered.csv")
rppa_clean.to_csv(out_dir / "rppa_filtered.csv")

print("\n=== FEATURE FILTERING COMPLETE ===")
print(f"ML-Ready matrices exported to: {out_dir}")
