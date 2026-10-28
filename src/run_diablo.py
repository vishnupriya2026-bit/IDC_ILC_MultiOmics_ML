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
df_num = pd.read_csv(proc / 'rppa_aligned.csv', index_col=0)
print(f"      Loaded harmonized RPPA matrix: {df_num.shape[0]} samples x {df_num.shape[1]} protein features")

var_series = df_num.var(axis=0, skipna=True)
valid_cols = var_series[var_series > 1e-8].index
df_num = df_num[valid_cols]
print(f"      Valid non-constant antibody features recovered: {len(valid_cols)}")

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



suppressMessages(library(mixOmics))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))


suppressMessages(library(mixOmics))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))

dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)

# Automatically find fitted DIABLO object by class
diablo_obj <- NULL
all_vars <- ls()
for (v in all_vars) {
    obj <- get(v)
    if (any(c("sgccda", "block.splsda", "block.plsda", "DA") %in% class(obj))) {
        diablo_obj <- obj
        break
    }
}

if (is.null(diablo_obj)) {
    stop("Could not locate fitted DIABLO model object in R workspace.")
}

# 1. Generate plotIndiv without confidence ellipses and without main title
p_out <- plotIndiv(
    diablo_obj, 
    ind.names = FALSE, 
    legend = TRUE, 
    ellipse = FALSE, 
    title = ""
)

# 2. Format with ggplot2 for publication
if ("graph" %in% names(p_out) && inherits(p_out$graph, "ggplot")) {
    final_plot <- p_out$graph + 
        facet_wrap(~Block, nrow = 1) + 
        labs(
            x = "Latent Subspace Variate 1",
            y = "Latent Subspace Variate 2",
            color = "Outcome (Y): Stage",
            shape = "Outcome (Y): Stage"
        ) + 
        theme_bw(base_size = 13) +
        theme(
            plot.title = element_blank(),
            strip.background = element_rect(fill = "grey92", color = "black"),
            strip.text = element_text(size = 12, face = "bold"),
            axis.title = element_text(size = 12, face = "bold"),
            axis.text = element_text(size = 10, color = "black"),
            legend.title = element_text(size = 11, face = "bold"),
            legend.text = element_text(size = 10),
            legend.position = "right",
            panel.grid.minor = element_blank()
        )
} else {
    final_plot <- p_out
}

# 3. Export 300 DPI high-resolution PNG (15 x 5 inches)
png("results/figures/diablo_subspace.png", width = 15, height = 5, units = "in", res = 300)
print(final_plot)
dev.off()

# 4. Export vector PDF for publication submission
pdf("results/figures/diablo_subspace.pdf", width = 15, height = 5)
print(final_plot)
dev.off()


# 5. Save vector PDF for publication submission
pdf("results/figures/diablo_subspace.pdf", width = 15, height = 5)
print(final_plot)
dev.off()


# Save vector PDF (15 x 5 inches)
pdf("results/figures/diablo_subspace.pdf", width = 15, height = 5)
print(final_plot)
dev.off()


# Export Vector Subspace PDF
pdf("results/figures/diablo_subspace.pdf", width = 10, height = 8)
plotIndiv(diablo_obj, ind.names = FALSE, legend = TRUE, ellipse = TRUE, title = "DIABLO Multi-Omics Latent Subspace")
dev.off()


png("results/figures/diablo_subspace.png", width = 10, height = 8, units = "in", res = 300)
# circosPlot disabled
dev.off()

cat("\\n=== DIABLO MODELING COMPLETE ===\\n")
"""

r_script_path = r_dir / '01_run_diablo.R'
r_script_path.write_text(r_code)

subprocess.run(['Rscript', str(r_script_path)], check=True)
