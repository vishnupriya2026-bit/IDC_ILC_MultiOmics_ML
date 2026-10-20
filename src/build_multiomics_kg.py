import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = "data/processed/BreastOmicsDB.sqlite"

def build_aegis_multiomics_kg():
    print("🛠️ Expanding Aegis-MultiOmicsKG Database Schema (Objectives 3 & 4)...")
    os.makedirs("data/processed", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Clear old schema for clean rebuild
    cursor.execute("DROP TABLE IF EXISTS multiomics_biomarkers")
    cursor.execute("DROP TABLE IF EXISTS cross_layer_interactions")
    cursor.execute("DROP TABLE IF EXISTS targeted_therapeutics")
    cursor.execute("DROP TABLE IF EXISTS oncogenic_pathways")
    cursor.execute("DROP TABLE IF EXISTS patient_risk_cohorts")
    
    # 2. Multi-Omics Biomarkers Table
    cursor.execute("""
    CREATE TABLE multiomics_biomarkers (
        gene_symbol TEXT PRIMARY KEY,
        genomic_mutation_rate REAL,
        rna_log2_fc REAL,
        protein_expression_zscore REAL,
        primary_subtype_association TEXT
    )
    """)
    
    # 3. Cross-Layer Interactions
    cursor.execute("""
    CREATE TABLE cross_layer_interactions (
        interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_symbol TEXT,
        upstream_layer TEXT,
        downstream_layer TEXT,
        correlation_coefficient REAL,
        p_value REAL,
        biological_mechanism TEXT,
        FOREIGN KEY(gene_symbol) REFERENCES multiomics_biomarkers(gene_symbol)
    )
    """)
    
    # 4. Targeted Therapeutics
    cursor.execute("""
    CREATE TABLE targeted_therapeutics (
        target_id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_symbol TEXT,
        drug_name TEXT,
        mechanism_of_action TEXT,
        evidence_level TEXT,
        fda_approval_status TEXT,
        FOREIGN KEY(gene_symbol) REFERENCES multiomics_biomarkers(gene_symbol)
    )
    """)

    # 5. NEW: Oncogenic Pathways Table
    cursor.execute("""
    CREATE TABLE oncogenic_pathways (
        pathway_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pathway_name TEXT,
        gene_symbol TEXT,
        functional_role TEXT,
        alteration_frequency REAL,
        FOREIGN KEY(gene_symbol) REFERENCES multiomics_biomarkers(gene_symbol)
    )
    """)

    # 6. NEW: Patient Risk Cohorts Table
    cursor.execute("""
    CREATE TABLE patient_risk_cohorts (
        patient_id TEXT PRIMARY KEY,
        risk_group TEXT,
        fused_hazard_score REAL,
        survival_months REAL,
        vital_status TEXT,
        subtype TEXT
    )
    """)
    
    # Insert Core Biomarkers
    driver_genes = [
        ("ERBB2", 0.04, 3.82, 2.91, "HER2+"),
        ("PIK3CA", 0.36, 1.45, 1.20, "Luminal A"),
        ("TP53", 0.82, -1.12, -0.85, "TNBC / Basal"),
        ("ESR1", 0.11, 4.10, 3.40, "Luminal B"),
        ("GATA3", 0.15, 2.10, 1.90, "Luminal A"),
        ("BRCA1", 0.08, -2.40, -2.10, "TNBC / Basal"),
        ("EGFR", 0.05, 2.90, 2.50, "TNBC / Basal")
    ]
    cursor.executemany("""
    INSERT INTO multiomics_biomarkers VALUES (?, ?, ?, ?, ?)
    """, driver_genes)
    
    # Insert Cross-Layer Interactions
    cross_layer_data = [
        ("ERBB2", "Genomics (CNV Amplification)", "Proteomics (HER2 RPPA)", 0.89, 1.2e-12, "Gene Amplification drives Overexpression"),
        ("PIK3CA", "Genomics (E545K Mutation)", "Proteomics (p-AKT Signal)", 0.76, 3.1e-08, "Kinase Activation of mTOR Pathway"),
        ("ESR1", "Transcriptomics (RNA-Seq)", "Proteomics (ER Alpha)", 0.91, 2.0e-14, "Transcriptional Activation of Estrogen Receptors")
    ]
    cursor.executemany("""
    INSERT INTO cross_layer_interactions (gene_symbol, upstream_layer, downstream_layer, correlation_coefficient, p_value, biological_mechanism)
    VALUES (?, ?, ?, ?, ?, ?)
    """, cross_layer_data)
    
    # Insert Therapeutics
    drug_data = [
        ("ERBB2", "Trastuzumab (Herceptin)", "HER2 Monoclonal Antibody", "High (Phase III)", "FDA Approved"),
        ("ERBB2", "Lapatinib", "Dual EGFR/HER2 Tyrosine Kinase Inhibitor", "High (Phase III)", "FDA Approved"),
        ("PIK3CA", "Alpelisib (Piqray)", "PI3Kα-Specific Inhibitor", "High (Phase III)", "FDA Approved"),
        ("BRCA1", "Olaparib (Lynparza)", "PARP Inhibitor (Synthetic Lethality)", "High (Phase III)", "FDA Approved")
    ]
    cursor.executemany("""
    INSERT INTO targeted_therapeutics (gene_symbol, drug_name, mechanism_of_action, evidence_level, fda_approval_status)
    VALUES (?, ?, ?, ?, ?)
    """, drug_data)

    # Insert Pathways Data
    pathways_data = [
        ("Receptor Tyrosine Kinase (RTK)", "ERBB2", "Oncogenic Driver", 0.22),
        ("Receptor Tyrosine Kinase (RTK)", "EGFR", "Co-receptor Activation", 0.14),
        ("PI3K / AKT / mTOR Cascade", "PIK3CA", "Catalytic Subunit Mutation", 0.38),
        ("p53 Cell Cycle Pathway", "TP53", "Tumor Suppressor Loss", 0.82),
        ("Estrogen Receptor Signaling", "ESR1", "Transcription Factor", 0.65),
        ("DNA Damage Repair (DDR)", "BRCA1", "Homologous Recombination Loss", 0.12)
    ]
    cursor.executemany("""
    INSERT INTO oncogenic_pathways (pathway_name, gene_symbol, functional_role, alteration_frequency)
    VALUES (?, ?, ?, ?)
    """, pathways_data)

    # Insert Synthetic Patient Risk Cohorts
    np.random.seed(42)
    cohort_rows = []
    subtypes = ["HER2+", "Luminal A", "Luminal B", "TNBC / Basal"]
    for i in range(1, 101):
        pid = f"TCGA-BR-{i:03d}"
        sub = np.random.choice(subtypes)
        fused_score = float(np.round(np.random.uniform(0.1, 2.5), 3))
        
        if fused_score > 1.6:
            group = "High Risk"
            surv = float(np.round(np.random.uniform(12, 48), 1))
            status = "Deceased" if np.random.rand() > 0.3 else "Living"
        elif fused_score > 0.9:
            group = "Intermediate Risk"
            surv = float(np.round(np.random.uniform(36, 84), 1))
            status = "Living" if np.random.rand() > 0.25 else "Deceased"
        else:
            group = "Low Risk"
            surv = float(np.round(np.random.uniform(60, 120), 1))
            status = "Living"
            
        cohort_rows.append((pid, group, fused_score, surv, status, sub))

    cursor.executemany("""
    INSERT INTO patient_risk_cohorts VALUES (?, ?, ?, ?, ?, ?)
    """, cohort_rows)
    
    conn.commit()
    conn.close()
    print("✅ Aegis-MultiOmicsKG Database fully populated with Pathways & Cohorts!")

if __name__ == "__main__":
    build_aegis_multiomics_kg()
