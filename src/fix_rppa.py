import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

base_dir = Path.home() / "phd_research/data/processed"
rppa = pd.read_csv(base_dir / "rppa_aligned.csv", index_col=0)

# Convert to numeric and filter features with >50% missing values
rppa_num = rppa.apply(pd.to_numeric, errors='coerce')
valid_cols = rppa_num.columns[rppa_num.notnull().mean(axis=0) >= 0.50]
rppa_valid = rppa_num[valid_cols]

# Impute median values and drop near-zero variance features
imputer = SimpleImputer(strategy='median')
rppa_imp = pd.DataFrame(imputer.fit_transform(rppa_valid), index=rppa_valid.index, columns=rppa_valid.columns)
rppa_clean = rppa_imp.loc[:, rppa_imp.var(axis=0) > 1e-8]

out_dir = base_dir / "ml_ready"
out_dir.mkdir(parents=True, exist_ok=True)
rppa_clean.to_csv(out_dir / "rppa_filtered.csv")

print(f"RPPA Matrix Ready for DIABLO: {rppa_clean.shape[0]} samples x {rppa_clean.shape[1]} antibodies.")
