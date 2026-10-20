import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="OncoPharmAI | Precision Oncology Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Theme & OncoPharmAI Branding
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* OncoPharmAI Header Card */
    .brand-header-box {
        background: white;
        padding: 30px 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Logo / Main Title Styling */
    .brand-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 10px 0 0 0;
        letter-spacing: -0.5px;
    }
    .brand-title .brand-dark { color: #0b3c5d; }
    .brand-title .brand-purple { color: #8e24aa; }
    
    /* Subtitle Pills & Slogans */
    .brand-slogan {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0284c7;
        letter-spacing: 2px;
        margin-top: 6px;
        text-transform: uppercase;
    }
    
    .brand-tagline-pill {
        display: inline-block;
        background: linear-gradient(90deg, #008080 0%, #005f73 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 8px 26px;
        border-radius: 25px;
        margin-top: 16px;
        box-shadow: 0 2px 10px rgba(0, 95, 115, 0.25);
    }

    /* KPI Summary Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 5px solid #008080;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    .metric-card-high { border-left-color: #e63946; }
    .metric-card-mid { border-left-color: #f4a261; }
    .metric-card-low { border-left-color: #2a9d8f; }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 4px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #008080 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. OncoPharmAI Header Banner
st.markdown("""
<div class="brand-header-box">
    <div style="font-size: 3rem; line-height: 1;">🧬💊</div>
    <h1 class="brand-title">
        <span class="brand-dark">OncoPharm</span><span class="brand-purple">AI</span>
    </h1>
    <div class="brand-slogan">DATA. INSIGHT. THERAPY. IMPACT.</div>
    <div class="brand-tagline-pill">AI-Assisted Multi-Omics Precision Oncology Platform</div>
</div>
""", unsafe_allow_html=True)

# 4. Patient Risk Scorecard Overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total Cohort</div>
        <div class="metric-value">286 Patients</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card metric-card-high">
        <div class="metric-label">High Risk Tier</div>
        <div class="metric-value">21 (100% Relapse)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card metric-card-mid">
        <div class="metric-label">Intermediate Risk</div>
        <div class="metric-value">51 (62.7% Relapse)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card metric-card-low">
        <div class="metric-label">Low Risk Tier</div>
        <div class="metric-value">214 (7.5% Relapse)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# 5. Interactive Dashboard Visualizations
tab1, tab2 = st.tabs(["📊 Subtype & Risk Analytics", "🌌 Multi-Omic Latent Space"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Subtype Distribution")
        subtype_df = pd.DataFrame({
            "Subtype": ["Luminal A", "Luminal B", "HER2-Enriched", "Basal-like (TNBC)"],
            "Count": [120, 85, 35, 46]
        })
        fig1 = px.pie(
            subtype_df, values="Count", names="Subtype",
            color="Subtype",
            color_discrete_map={
                "Luminal A": "#2a9d8f",
                "Luminal B": "#e9c46a",
                "HER2-Enriched": "#f4a261",
                "Basal-like (TNBC)": "#e63946"
            },
            hole=0.4
        )
        fig1.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.subheader("Model AUC Performance")
        ml_df = pd.DataFrame({
            "Model": ["Support Vector Machine", "Logistic Regression", "Random Forest", "Gradient Boosting"],
            "AUC": [0.698, 0.695, 0.688, 0.649]
        })
        fig2 = px.bar(
            ml_df, x="AUC", y="Model", orientation="h",
            color="AUC", color_continuous_scale="Viridis",
            text_auto=".3f"
        )
        fig2.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
