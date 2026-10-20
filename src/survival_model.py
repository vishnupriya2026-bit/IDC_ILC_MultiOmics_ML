import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

class ClinicalSurvivalEngine:
    """
    Module for fitting Cox Proportional Hazards models, calculating patient risk scores,
    and generating Kaplan-Meier survival curves for biomarker validation.
    """
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None

    def load_or_generate_data(self) -> pd.DataFrame:
        """Loads clinical survival data or generates mock multi-omics survival data if absent."""
        if self.data_path and os.path.exists(self.data_path):
            print(f"📂 Loading clinical data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
        else:
            print("⚠️ No input survival CSV found. Generating realistic synthetic TCGA breast cancer survival data...")
            np.random.seed(42)
            n_samples = 200
            
            # Synthetic features: Time (months), Event (1=deceased, 0=censored)
            time = np.random.exponential(scale=40, size=n_samples) + 5
            event = np.random.binomial(1, 0.4, size=n_samples)
            
            # Simulated biomarkers (high expression -> higher hazard/risk)
            erbb2 = np.random.normal(loc=5, scale=2, size=n_samples)
            pik3ca = np.random.normal(loc=4, scale=1.5, size=n_samples)
            age = np.random.randint(30, 80, size=n_samples)
            
            # Make survival time inversely proportional to biomarker expression
            risk_signal = 0.3 * erbb2 + 0.2 * pik3ca + 0.02 * age
            time = time / np.exp(risk_signal - np.mean(risk_signal))
            
            self.df = pd.DataFrame({
                "patient_id": [f"TCGA-BRCA-{i:04d}" for i in range(n_samples)],
                "time": time,
                "event": event,
                "ERBB2": erbb2,
                "PIK3CA": pik3ca,
                "age": age
            })
            
            # Save generated dataset
            os.makedirs("data/processed", exist_ok=True)
            self.df.to_csv("data/processed/synthetic_clinical_survival.csv", index=False)
            print("✅ Synthetic dataset saved to 'data/processed/synthetic_clinical_survival.csv'.")
            
        return self.df

    def fit_cox_model(self, feature_cols: list) -> CoxPHFitter:
        """Fits Cox Proportional Hazards Model on selected features."""
        print(f"📊 Fitting Cox Proportional Hazards Model on features: {feature_cols}...")
        cph_data = self.df[['time', 'event'] + feature_cols].dropna()
        
        cph = CoxPHFitter()
        cph.fit(cph_data, duration_col='time', event_col='event')
        
        print("\n--- COX MODEL SUMMARY ---")
        print(cph.summary[['coef', 'exp(coef)', 'p']])
        return cph

    def calculate_risk_strata_and_plot(self, cph: CoxPHFitter, feature_cols: list, output_fig_path: str):
        """Calculates risk scores, stratifies into High/Low risk, and plots KM curves."""
        cph_data = self.df[['time', 'event'] + feature_cols].dropna().copy()
        
        # Calculate continuous partial hazard (risk score)
        cph_data['risk_score'] = cph.predict_partial_hazard(cph_data)
        
        # Stratify by median score
        median_risk = cph_data['risk_score'].median()
        cph_data['risk_group'] = np.where(cph_data['risk_score'] >= median_risk, 'High Risk', 'Low Risk')
        
        # Log-rank test
        high = cph_data[cph_data['risk_group'] == 'High Risk']
        low = cph_data[cph_data['risk_group'] == 'Low Risk']
        
        results = logrank_test(high['time'], low['time'], event_observed_A=high['event'], event_observed_B=low['event'])
        p_val = results.p_value
        
        # Plot Kaplan-Meier Curves
        plt.figure(figsize=(8, 6))
        kmf = KaplanMeierFitter()
        
        kmf.fit(high['time'], high['event'], label=f"High Risk (n={len(high)})")
        ax = kmf.plot_survival_function(color='crimson', ci_show=True)
        
        kmf.fit(low['time'], low['event'], label=f"Low Risk (n={len(low)})")
        kmf.plot_survival_function(ax=ax, color='navy', ci_show=True)
        
        plt.title("Precision Oncology Risk Stratification (Multi-Omics Cohort)", fontsize=13, fontweight='bold')
        plt.xlabel("Time (Months)", fontsize=11)
        plt.ylabel("Overall Survival Probability", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.text(0.05, 0.15, f"Log-rank p-value: {p_val:.4e}", transform=ax.transAxes, 
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        
        print(f"\n📈 Kaplan-Meier survival curve successfully saved to '{output_fig_path}'")
        print(f"🔑 Log-rank Test p-value: {p_val:.4e}")

if __name__ == "__main__":
    engine = ClinicalSurvivalEngine()
    df = engine.load_or_generate_data()
    
    biomarkers = ["ERBB2", "PIK3CA", "age"]
    cph_model = engine.fit_cox_model(feature_cols=biomarkers)
    
    engine.calculate_risk_strata_and_plot(
        cph=cph_model, 
        feature_cols=biomarkers, 
        output_fig_path="results/figures/survival_risk_stratification.png"
    )
