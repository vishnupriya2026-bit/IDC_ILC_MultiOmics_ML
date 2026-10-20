import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PAGE CONFIGURATION & APP NAME
# ==========================================
APP_NAME = "OncoPharmAI: Multi-Omics & Drug Repurposing Portal"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM VIBRANT CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Gradient Banner */
    .hero-container {
        background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(139, 92, 246, 0.3);
        margin-bottom: 25px;
        text-align: center;
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #f1f5f9 !important;
        font-size: 1.1rem !important;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card-1 {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2) 0%, rgba(219, 39, 119, 0.1) 100%);
        border: 1px solid #ec4899;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-card-2 {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid #8b5cf6;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-card-3 {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-card-4 {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Custom Badge Tags */
    .badge-egfr { background-color: #ef4444; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-esr1 { background-color: #10b981; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .badge-pik3ca { background-color: #3b82f6; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HERO BANNER
# ==========================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🔮 {APP_NAME}</div>
    <div class="hero-subtitle">Explainable AI (SHAP) & Evidence-Weighted Biomarker to Therapeutics Matcher</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. LOAD DATASETS
# ==========================================
@st.cache_data
def load_all_data():
    df_drugs, df_shap = None, None
    if os.path.exists("prioritized_repurposed_drugs_multiomics.csv"):
        df_drugs = pd.read_csv("prioritized_repurposed_drugs_multiomics.csv")
    if os.path.exists("prioritized_multiomics_biomarkers_shap.csv"):
        df_shap = pd.read_csv("prioritized_multiomics_biomarkers_shap.csv")
    return df_drugs, df_shap

df_drugs, df_shap = load_all_data()

if df_drugs is None or df_shap is None:
    st.error("Error: Output CSV files not found. Please ensure pipeline scripts have finished running.")
    st.stop()

# ==========================================
# 5. SIDEBAR FILTERS
# ==========================================
st.sidebar.markdown("<h2 style='color: #c084fc;'>⚙️ Filter Controls</h2>", unsafe_allow_html=True)

genes_available = ["All Target Genes"] + sorted(list(df_drugs['Gene'].unique()))
selected_gene = st.sidebar.selectbox("🎯 Target Biomarker Gene:", genes_available)

min_score = float(df_drugs['Composite_Priority_Score'].min())
max_score = float(df_drugs['Composite_Priority_Score'].max())
score_range = st.sidebar.slider(
    "🔥 Priority Score Range:", 
    min_value=min_score, 
    max_value=max_score, 
    value=(min_score, max_score)
)

# Apply Filters
df_filtered = df_drugs[
    (df_drugs['Composite_Priority_Score'] >= score_range[0]) & 
    (df_drugs['Composite_Priority_Score'] <= score_range[1])
]
if selected_gene != "All Target Genes":
    df_filtered = df_filtered[df_filtered['Gene'] == selected_gene]

# ==========================================
# 6. TOP METRICS CARDS
# ==========================================
m1, m2, m3, m4 = st.columns(4)

top_drug = df_filtered.iloc[0]['Drug'] if not df_filtered.empty else "N/A"
top_gene = df_filtered.iloc[0]['Gene'] if not df_filtered.empty else "N/A"
top_score = round(df_filtered.iloc[0]['Composite_Priority_Score'], 2) if not df_filtered.empty else 0.0

m1.markdown(f"""
<div class="metric-card-1">
    <div class="metric-val">{len(df_filtered)}</div>
    <div class="metric-lbl">Candidate Pairs</div>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class="metric-card-2">
    <div class="metric-val">{top_drug}</div>
    <div class="metric-lbl">Top Ranked Drug</div>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class="metric-card-3">
    <div class="metric-val">{top_gene}</div>
    <div class="metric-lbl">Leading Target Gene</div>
</div>
""", unsafe_allow_html=True)

m4.markdown(f"""
<div class="metric-card-4">
    <div class="metric-val">{top_score}</div>
    <div class="metric-lbl">Max Composite Score</div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# ==========================================
# 7. INTERACTIVE TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "💊 Candidate Repurposed Drugs", 
    "🧬 SHAP Multi-Omics Biomarkers", 
    "📊 Priority Visualization"
])

with tab1:
    st.markdown("### 🎯 Prioritized Drug-Target Matches")
    st.markdown("Ranked by combining cohort Multi-Omics ANOVA F-scores with database evidence count.")
    
    # Styled dataframe display
    st.dataframe(
        df_filtered[['Gene', 'Drug', 'Interaction_Type', 'Evidence_Count', 'ANOVA_F_Score', 'Composite_Priority_Score', 'Sources']],
        use_container_width=True,
        column_config={
            "Composite_Priority_Score": st.column_config.ProgressColumn(
                "Composite Score",
                format="%.2f",
                min_value=0,
                max_value=max_score
            ),
            "ANOVA_F_Score": st.column_config.NumberColumn("ANOVA F-Score", format="%.4f"),
            "Evidence_Count": st.column_config.NumberColumn("Evidence Count", format="%d")
        }
    )

with tab2:
    st.markdown("### 🧬 SHAP Feature Importance Rankings")
    st.markdown("Top multi-omics biomarkers driving predictive stage stratification in CPTAC cohort.")
    
    st.dataframe(
        df_shap,
        use_container_width=True,
        column_config={
            "Mean_Abs_SHAP": st.column_config.ProgressColumn(
                "SHAP Importance",
                format="%.4f",
                min_value=0,
                max_value=float(df_shap['Mean_Abs_SHAP'].max()) if 'Mean_Abs_SHAP' in df_shap.columns else 1.0
            )
        }
    )

with tab3:
    st.markdown("### 🎨 Top Candidate Drug Priority Distribution")
    
    top_plot_df = df_filtered.head(12).copy()
    if not top_plot_df.empty:
        top_plot_df['Label'] = top_plot_df['Drug'] + " (" + top_plot_df['Gene'] + ")"
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        sns.barplot(
            data=top_plot_df,
            x='Composite_Priority_Score',
            y='Label',
            palette='magma',
            ax=ax
        )
        
        ax.set_title("Top Repurposed Drugs by Multi-Omics Composite Score", color='white', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Composite Priority Score", color='white', fontsize=10)
        ax.set_ylabel("Candidate Drug (Target)", color='white', fontsize=10)
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
            
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No candidates match the current filter selection.")

