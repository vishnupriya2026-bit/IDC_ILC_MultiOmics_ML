import sys
import os
import time

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from objective_2.drug_prioritization.drug_prioritizer import DrugPrioritizationEngine
from objective_2.survival.survival_model import ClinicalSurvivalEngine
from objective_2.explainability.pathway_enrichment import PathwayEnrichmentEngine
from objective_2.reporting.report_generator import ClinicalReportGenerator

def run_objective_2_pipeline():
    start_time = time.time()
    print("=" * 65)
    print("🚀 RUNNING OBJECTIVE 2: CLINICAL DECISION SUPPORT PIPELINE")
    print("=" * 65)

    # Key driver biomarkers identified from Objective 1 SHAP analysis
    biomarkers = ["ERBB2", "ESR1", "PIK3CA", "BRCA1", "TP53", "EGFR", "PTEN"]
    patient_id = "TCGA-BRCA-PATIENT-001"

    # Step 1: Drug Target Prioritization
    print("\n--- STEP 1: DRUG TARGET PRIORITIZATION ---")
    drug_engine = DrugPrioritizationEngine(genes=biomarkers)
    raw_drugs = drug_engine.fetch_dgidb_interactions()
    prioritized_drugs = drug_engine.prioritize_drugs(raw_drugs)
    
    os.makedirs("data/processed", exist_ok=True)
    prioritized_drugs.to_csv("data/processed/prioritized_drugs.csv", index=False)
    print(f"✅ Step 1 Complete: Top drug targets saved.")

    # Step 2: Survival Analysis & Risk Modeling
    print("\n--- STEP 2: CLINICAL SURVIVAL & RISK MODELING ---")
    survival_engine = ClinicalSurvivalEngine()
    survival_df = survival_engine.load_or_generate_data()
    cph_model = survival_engine.fit_cox_model(feature_cols=["ERBB2", "PIK3CA", "age"])
    survival_engine.calculate_risk_strata_and_plot(
        cph=cph_model,
        feature_cols=["ERBB2", "PIK3CA", "age"],
        output_fig_path="results/figures/survival_risk_stratification.png"
    )
    print(f"✅ Step 2 Complete: Survival stratification saved.")

    # Step 3: Pathway Enrichment
    print("\n--- STEP 3: PATHWAY ENRICHMENT & EXPLAINABILITY ---")
    enrichment_engine = PathwayEnrichmentEngine(gene_list=biomarkers)
    pathways_df = enrichment_engine.run_enrichment()
    if not pathways_df.empty:
        pathways_df.to_csv("data/processed/pathway_enrichment_results.csv", index=False)
        enrichment_engine.plot_top_pathways(
            pathways_df, 
            output_fig_path="results/figures/pathway_enrichment.png", 
            top_n=10
        )
    print(f"✅ Step 3 Complete: Pathway analysis saved.")

    # Step 4: Report Generation
    print("\n--- STEP 4: AUTOMATED CLINICAL REPORT GENERATION ---")
    report_gen = ClinicalReportGenerator(patient_id=patient_id)
    report_path = "results/reports/patient_clinical_decision_report.html"
    report_gen.generate_html_report(output_html_path=report_path)
    print(f"✅ Step 4 Complete: HTML Clinical Report created.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f"📄 Output Report: {report_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_objective_2_pipeline()
