from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import os

app = FastAPI(
    title="MultiOmicsAI Platform API",
    description="REST API serving precision oncology biomarkers, survival risks, and targeted therapeutics.",
    version="1.0.0"
)

DB_PATH = "data/processed/BreastOmicsDB.sqlite"

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="BreastOmicsDB sqlite database file not found. Run build_database.py first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/")
def root():
    return {
        "platform": "MultiOmicsAI Precision Oncology Platform",
        "status": "Online",
        "endpoints": ["/biomarkers", "/gene/{symbol}", "/drugs/{symbol}"]
    }

@app.get("/biomarkers")
def get_all_biomarkers():
    """Returns all identified multi-omics biomarker driver genes."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM biomarkers ORDER BY mean_shap_score DESC", conn)
    conn.close()
    return df.to_dict(orient="records")

@app.get("/gene/{symbol}")
def get_gene_profile(symbol: str):
    """Retrieves multi-omics feature importance, survival HR, and p-value for a specific gene."""
    conn = get_db_connection()
    query = """
        SELECT b.gene_symbol, b.omics_type, b.mean_shap_score, b.breast_cancer_subtype,
               s.hazard_ratio, s.p_value, s.risk_tier
        FROM biomarkers b
        LEFT JOIN survival_associations s ON b.gene_symbol = s.gene_symbol
        WHERE b.gene_symbol = ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (symbol.upper().strip(),))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        raise HTTPException(status_code=404, detail=f"Biomarker '{symbol}' not found in database.")
    
    return dict(row)

@app.get("/drugs/{symbol}")
def get_drugs_for_gene(symbol: str):
    """Returns prioritized candidate drugs targeting a given gene."""
    conn = get_db_connection()
    query = """
        SELECT drug_name, interaction_type, evidence_count 
        FROM drug_targets 
        WHERE gene_symbol = ? 
        ORDER BY evidence_count DESC
    """
    df = pd.read_sql_query(query, conn, params=(symbol.upper().strip(),))
    conn.close()
    
    if df.empty:
        return {"gene": symbol, "message": "No specific drug interactions found in database.", "drugs": []}
    
    return {"gene": symbol, "drugs": df.to_dict(orient="records")}
