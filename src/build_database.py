import sqlite3
import os

DB_PATH = "data/processed/BreastOmicsDB.sqlite"

def build_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("🛠️ Building AegisOmics Relational Database Schema...")

    # Drop existing tables to refresh schema automatically
    cursor.execute("DROP TABLE IF EXISTS biomarkers")
    cursor.execute("DROP TABLE IF EXISTS survival_associations")
    cursor.execute("DROP TABLE IF EXISTS drug_targets")
    cursor.execute("DROP TABLE IF EXISTS methylation_profiles")

    # 1. Biomarkers Table
    cursor.execute("""
    CREATE TABLE biomarkers (
        gene_symbol TEXT PRIMARY KEY,
        omics_type TEXT,
        mean_shap_score REAL,
        subtyping_relevance TEXT
    )
    """)

    # 2. Survival Associations Table
    cursor.execute("""
    CREATE TABLE survival_associations (
        gene_symbol TEXT PRIMARY KEY,
        hazard_ratio REAL,
        p_value REAL,
        risk_tier TEXT,
        FOREIGN KEY(gene_symbol) REFERENCES biomarkers(gene_symbol)
    )
    """)

    # 3. Drug Targets Table
    cursor.execute("""
    CREATE TABLE drug_targets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gene_symbol TEXT,
        drug_name TEXT,
        interaction_type TEXT,
        evidence_count INTEGER,
        approval_status TEXT,
        FOREIGN KEY(gene_symbol) REFERENCES biomarkers(gene_symbol)
    )
    """)

    # 4. Epigenomics Methylation Profiles Table
    cursor.execute("""
    CREATE TABLE methylation_profiles (
        gene_symbol TEXT PRIMARY KEY,
        cpg_site_id TEXT,
        mean_beta_value REAL,
        methylation_status TEXT,
        epigenetic_inactivation_index REAL,
        synthetic_lethality_target TEXT,
        FOREIGN KEY(gene_symbol) REFERENCES biomarkers(gene_symbol)
    )
    """)

    # --- SEED DATA ---
    print("🌱 Seeding Multi-Omics, Survival, Drug, and Methylation Records...")

    # Seed Biomarkers
    biomarkers_data = [
        ("ERBB2", "Proteomics / RNA-Seq", 0.485, "HER2+ Subtype Enrichment"),
        ("ESR1", "Transcriptomics", 0.412, "Luminal A/B ER+ Marker"),
        ("PIK3CA", "Genomics / Mutation", 0.398, "Oncogenic Signaling Axis"),
        ("BRCA1", "Genomics / Epigenomics", 0.354, "DNA Damage Repair Loss"),
        ("TP53", "Genomics / Mutation", 0.321, "TNBC / Basal High-Risk Driver"),
        ("CDKN2A", "Epigenomics / RNA-Seq", 0.289, "Cell Cycle Checkpoint Silencing"),
        ("GSTP1", "Epigenomics", 0.265, "Detoxification Promoter Silencing")
    ]
    cursor.executemany("INSERT INTO biomarkers VALUES (?, ?, ?, ?)", biomarkers_data)

    # Seed Survival Associations
    survival_data = [
        ("ERBB2", 1.85, 0.00012, "High Risk"),
        ("ESR1", 0.65, 0.00340, "Favorable"),
        ("PIK3CA", 1.42, 0.01200, "Moderate Risk"),
        ("BRCA1", 2.10, 0.00004, "High Risk"),
        ("TP53", 1.95, 0.00008, "High Risk"),
        ("CDKN2A", 1.68, 0.00110, "High Risk"),
        ("GSTP1", 1.35, 0.02400, "Moderate Risk")
    ]
    cursor.executemany("INSERT INTO survival_associations VALUES (?, ?, ?, ?)", survival_data)

    # Seed Drug Targets
    drugs_data = [
        ("ERBB2", "Trastuzumab", "Inhibitor / Antibody", 42, "FDA Approved"),
        ("ERBB2", "Pertuzumab", "Inhibitor / Antibody", 38, "FDA Approved"),
        ("ERBB2", "Lapatinib", "Kinase Inhibitor", 29, "FDA Approved"),
        ("ESR1", "Fulvestrant", "Degrader (SERD)", 35, "FDA Approved"),
        ("ESR1", "Tamoxifen", "Selective Modulator (SERM)", 50, "FDA Approved"),
        ("PIK3CA", "Alpelisib", "PI3K Alpha Inhibitor", 24, "FDA Approved"),
        ("BRCA1", "Olaparib", "PARP Inhibitor", 31, "FDA Approved (Synthetic Lethal)"),
        ("BRCA1", "Talazoparib", "PARP Trapper", 19, "FDA Approved"),
        ("CDKN2A", "Decitabine", "DNMT Inhibitor (Epigenetic)", 15, "FDA Approved / Off-Label"),
        ("GSTP1", "Azacitidine", "DNMT Inhibitor (Epigenetic)", 18, "FDA Approved / Off-Label")
    ]
    cursor.executemany("INSERT INTO drug_targets (gene_symbol, drug_name, interaction_type, evidence_count, approval_status) VALUES (?, ?, ?, ?, ?)", drugs_data)

    # Seed Methylation Profiles
    methylation_data = [
        ("BRCA1", "cg04229476", 0.78, "Promoter Hypermethylated", 0.84, "PARP Inhibitor (Olaparib)"),
        ("CDKN2A", "cg12845921", 0.82, "Promoter Hypermethylated", 0.89, "DNMT Inhibitor (Decitabine)"),
        ("GSTP1", "cg08923410", 0.74, "Promoter Hypermethylated", 0.79, "DNMT Inhibitor (Azacitidine)"),
        ("ESR1", "cg02154388", 0.68, "Moderate Hypermethylation", 0.71, "HDAC Inhibitor (Vorinostat)"),
        ("ERBB2", "cg25881902", 0.12, "Hypomethylated (Expressed)", 0.08, "N/A (Standard Targeted)"),
        ("PIK3CA", "cg19438201", 0.18, "Hypomethylated (Expressed)", 0.12, "N/A (Standard Targeted)"),
        ("TP53", "cg00129481", 0.22, "Hypomethylated", 0.15, "N/A (Genomic Variant)")
    ]
    cursor.executemany("INSERT INTO methylation_profiles VALUES (?, ?, ?, ?, ?, ?)", methylation_data)

    conn.commit()
    conn.close()
    print("✅ BreastOmicsDB database created and populated successfully!")

if __name__ == "__main__":
    build_database()
