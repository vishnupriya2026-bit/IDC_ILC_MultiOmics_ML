import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "BreastOmicsDB.sqlite"

def inspect_database(db_path):
    if not os.path.exists(db_path):
        print(f"Error: File '{db_path}' not found in current directory ({os.getcwd()}). Check database path.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print("=" * 60)
    print(f"DATABASE AUDIT REPORT: {db_path}")
    print("=" * 60)
    print(f"Found Tables ({len(tables)}): {tables}\n")

    db_data = {}
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        db_data[table] = df
        print(f"Table: '{table}' | Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n" + "-" * 60)
    print("1. LAYER-SPECIFIC COHORT SIZES & IDENTIFIERS")
    print("-" * 60)

    id_candidates = ['patient_id', 'sample_id', 'id', 'geo_accession', 'barcode', 'patient']
    
    table_ids = {}
    for name, df in db_data.items():
        id_col = next((col for col in df.columns if col.lower() in id_candidates), df.columns[0])
        unique_ids = set(df[id_col].dropna().astype(str).str.strip())
        duplicates = df[id_col].duplicated().sum()
        table_ids[name] = unique_ids
        print(f"• Table '{name}':")
        print(f"   - Identified ID Column: '{id_col}'")
        print(f"   - Total Rows: {len(df)}")
        print(f"   - Unique Patient/Sample IDs: {len(unique_ids)}")
        print(f"   - Duplicate IDs: {duplicates}")

    print("\n" + "-" * 60)
    print("2. MULTI-OMICS OVERLAP & INTERSECTIONS")
    print("-" * 60)

    keys = list(table_ids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            t1, t2 = keys[i], keys[j]
            overlap = table_ids[t1].intersection(table_ids[t2])
            print(f"• Overlap ({t1} ∩ {t2}): {len(overlap)} samples")

    if len(keys) >= 3:
        three_way = set.intersection(*[table_ids[k] for k in keys[:3]])
        print(f"\n★ 3-Way Overlap (Unified Multi-Omics Cohort): {len(three_way)} samples")

    print("\n" + "-" * 60)
    print("3. MISSING VALUE & NORMALIZATION PROFILE")
    print("-" * 60)

    for name, df in db_data.items():
        num_df = df.select_dtypes(include=[np.number])
        total_cells = num_df.size
        missing_cells = num_df.isna().sum().sum()
        pct_missing = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        min_val = num_df.min().min() if not num_df.empty else "N/A"
        max_val = num_df.max().max() if not num_df.empty else "N/A"

        print(f"• Table '{name}':")
        print(f"   - Missingness: {missing_cells} / {total_cells} values ({pct_missing:.2f}%)")
        print(f"   - Numeric Range: Min = {min_val}, Max = {max_val}")

    print("\n" + "-" * 60)
    print("4. CLINICAL ENDPOINTS & SURVIVAL EVENTS")
    print("-" * 60)

    clinical_keywords = ['clinical', 'survival', 'meta', 'patient']
    clin_table = next((t for t in tables if any(k in t.lower() for k in clinical_keywords)), None)

    if clin_table:
        df_clin = db_data[clin_table]
        print(f"Inspecting Clinical Table: '{clin_table}'")
        
        event_cols = [c for c in df_clin.columns if any(e in c.lower() for e in ['status', 'event', 'rfs', 'os', 'dfs', 'relapse', 'recurrence'])]
        time_cols = [c for c in df_clin.columns if any(t in c.lower() for t in ['time', 'months', 'days', 'years', 'followup'])]

        print(f"• Candidate Event Columns: {event_cols}")
        print(f"• Candidate Follow-up Time Columns: {time_cols}\n")

        for col in event_cols:
            print(f"Value breakdown for '{col}':")
            print(df_clin[col].value_counts(dropna=False).to_string())
            print()
    else:
        print("No dedicated clinical table automatically identified.")

    conn.close()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    inspect_database(DB_PATH)
