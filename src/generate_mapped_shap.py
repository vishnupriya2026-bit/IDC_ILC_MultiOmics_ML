import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. Ensembl ID to HGNC Gene Symbol Mapping Dictionary
gene_mapping = {
    "ENSG00000111404.7": "RERGL",
    "ENSG00000154330.13": "PGM5",
    "ENSG00000241644.2": "INMT",
    "ENSG00000149451.18": "ADAM33",
    "ENSG00000161544.10": "CYGB",
    "ENSG00000124664.11": "SPDEF",
    "ENSG00000187260.16": "WDR86",
    "ENSG00000064300.9": "NGFR",
    "ENSG00000072163.20": "LIMS2",
    "ENSG00000126218.12": "F10",
    "ENSG00000137225.13": "CAPN11",
    "ENSG00000165349.12": "SLC7A3",
    "ENSG00000204065.3": "TCEAL5",
    "ENSG00000187486.5": "KCNJ11"
}

def map_features(feature_list):
    """Maps Ensembl IDs to Gene Symbols while preserving RPPA/Methylation IDs."""
    return [gene_mapping.get(f, f) for f in feature_list]

if __name__ == "__main__":
    print("Gene mapping dictionary successfully compiled.")
    print(f"Mapped {len(gene_mapping)} Ensembl IDs to HGNC symbols.")
