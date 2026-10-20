import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
import subprocess
import sys

base = Path.home() / 'phd_research'
proc = base / 'data' / 'processed'
ml = proc / 'ml_ready'
r_dir = base / 'scripts' / 'diablo'
res_dir = base / 'results' / 'diablo'

for d in [ml, r_dir, res_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("[1/3] Smart-repairing RPPA Matrix...")
target = pd.read_csv(ml / 'target_y.csv', index_col=0)
ref_samples = [str(s).strip() for s in target.index]

rppa_raw = pd.read_csv(proc / 'rppa_aligned.csv')
print(f"      Raw RPPA shape: {rppa_raw.shape[0]} rows x {rppa_raw.shape[1]} columns")

sample_col = None
max_m = 0
for col in rppa_raw.columns:
    matches = rppa_raw[col].astype(str).str.strip().isin(ref_samples).sum()
    if matches > max_m:
        max_m = matches
        sample_col = col

if sample_col and max_m > 5:
    df_base = rppa_raw.set_index(sample_col)
else:
    df_base = rppa_raw.set_index(rppa_raw.columns[0])

# Compare variance across columns in both untransposed and transposed orientation
num_orig = df_base.apply(pd.to_numeric, errors='coerce')
var_orig = num_orig.var(axis=0, skipna=True)

num_trans = df_base.T.apply(pd.to_numeric, errors='coerce')
var_trans = num_trans.var(axis=0, skipna=True)

valid_orig = (var_orig > 1e-8).sum()
valid_trans = (var_trans > 1e-8).sum()

if valid_trans > valid_orig:
    print(f"      Variance check: Transposed matrix yields {valid_trans} valid features vs {valid_orig} untransposed. Transposing matrix...")
    df = df_base.T
else:
    df = df_base

df.index = df.index.astype(str).str.strip()
common_s = [s for s in ref_samples if s in df.index]

if len(common_s) == 0:
    if len(df) == len(ref_samples):
        print("      Sample ID string mismatch, aligning positionally with target_y")
        df.index = ref_samples
        common_s = ref_samples
    else:
        print("ERROR: Could not align sample IDs between RPPA matrix and target_y.csv.")
        sys.exit(1)

df = df.loc[common_s]

# Extract numeric values & filter out zero-variance columns
df_num = df.apply(pd.to_numeric, errors='coerce')
var_series = df_num.var(axis=0, skipna=True)
valid_cols = var_series[var_series > 1e-8].index
df_num = df_num[valid_cols]

print(f"      Valid non-constant antibody features recovered: {df_num.shape[1]}")

if df_num.shape[1] == 0:
    print("ERROR: All features have zero variance after cleaning.")
    sys.exit(1)

# Impute remaining missing values
imp = SimpleImputer(strategy='median')
imp_data = imp.fit_transform(df_num)
df_clean = pd.DataFrame(imp_data, index=df_num.index, columns=df_num.columns)

df_clean.to_csv(ml / 'rppa_filtered.csv')
print(f"      RPPA Ready: {df_clean.shape[0]} samples x {df_clean.shape[1]} antibodies.")

# Generate and execute DIABLO training script
print("[2/3] Writing DIABLO Training Script...")
r_code = """suppressPackageStartupMessages({
  if (!requireNamespace("mixOmics", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install("mixOmics", update = FALSE)
  }
  library(mixOmics)
})

base_dir <- file.path(Sys.getenv("HOME"), "phd_research")
ml_dir <- file.path(base_dir, "data", "processed", "ml_ready")
out_dir <- file.path(base_dir, "results", "diablo")

cat("[3/3] Training DIABLO Sparse PLS-DA Model...\\n")
rna <- read.csv(file.path(ml_dir, "rna_seq_filtered.csv"), row.names = 1, check.names = FALSE)
meth <- read.csv(file.path(ml_dir, "methylation_filtered.csv"), row.names = 1, check.names = FALSE)
rppa <- read.csv(file.path(ml_dir, "rppa_filtered.csv"), row.names = 1, check.names = FALSE)
target <- read.csv(file.path(ml_dir, "target_y.csv"), row.names = 1)

common_samples <- Reduce(intersect, list(rownames(rna), rownames(meth), rownames(rppa), rownames(target)))

X <- list(
  RNA = as.matrix(rna[common_samples, , drop=FALSE]),
  Methylation = as.matrix(meth[common_samples, , drop=FALSE]),
  RPPA = as.matrix(rppa[common_samples, , drop=FALSE])
)
Y <- as.factor(target[common_samples, "Y_Stage"])

# Prune zero-variance features in R before fitting
for (b in names(X)) {
  nzv <- mixOmics::nearZeroVar(X[[b]])
  if (length(nzv$Position) > 0) {
    cat(sprintf("      Filter: removed %d zero-variance features from %s\\n", length(nzv$Position), b))
    X[[b]] <- X[[b]][, -nzv$Position, drop = FALSE]
  }
}

cat(sprintf("      Aligned Cohort: %d samples | RNA=%d, Meth=%d, RPPA=%d\\n", 
            length(common_samples), ncol(X$RNA), ncol(X$Methylation), ncol(X$RPPA)))

design <- matrix(0.1, ncol = 3, nrow = 3, dimnames = list(names(X), names(X)))
diag(design) <- 0

keepX <- list(
  RNA = c(min(50, ncol(X$RNA)), min(25, ncol(X$RNA))),
  Methylation = c(min(50, ncol(X$Methylation)), min(25, ncol(X$Methylation))),
  RPPA = c(min(20, ncol(X$RPPA)), min(10, ncol(X$RPPA)))
)

diablo_model <- block.splsda(X = X, Y = Y, ncomp = 2, keepX = keepX, design = design, near.zero.var = TRUE)

biomarker_list <- list()
for (block in names(X)) {
  comp1 <- selectVar(diablo_model, block = block, comp = 1)$name
  comp2 <- selectVar(diablo_model, block = block, comp = 2)$name
  sel <- unique(c(comp1, comp2))
  if (length(sel) > 0) {
    biomarker_list[[block]] <- data.frame(
      Omics_Block = block,
      Feature_ID = sel,
      Comp1 = sel %in% comp1,
      Comp2 = sel %in% comp2
    )
  }
}
final_bio <- do.call(rbind, biomarker_list)
write.csv(final_bio, file.path(out_dir, "diablo_selected_biomarkers.csv"), row.names = FALSE)
cat(sprintf("      Saved %d biomarkers to: %s\\n", nrow(final_bio), file.path(out_dir, "diablo_selected_biomarkers.csv")))

png(file.path(out_dir, "diablo_sample_plot.png"), width = 2400, height = 1800, res = 300)
plotIndiv(diablo_model, legend = TRUE, title = "DIABLO Sample Clustering")
dev.off()

png(file.path(out_dir, "diablo_circos_plot.png"), width = 2400, height = 2400, res = 300)
circosPlot(diablo_model, cutoff = 0.7, line = TRUE)
dev.off()

cat("\\n=== DIABLO MODELING COMPLETE ===\\n")
"""

r_script_path = r_dir / '01_run_diablo.R'
r_script_path.write_text(r_code)

subprocess.run(['Rscript', str(r_script_path)], check=True)
