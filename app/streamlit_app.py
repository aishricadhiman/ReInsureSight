import streamlit as st

st.set_page_config(
    page_title = "ReinsureSight — Flood Risk Intelligence",
    page_icon  = "🌊",
    layout     = "wide"
)

st.title("🌊 ReinsureSight")
st.subheader("Geospatial Flood Loss Intelligence for Reinsurance Underwriting")

st.markdown("""
**ReInsureSight** predicts economic flood losses with statistically valid 
uncertainty intervals using satellite climate data and machine learning.

Use the sidebar to navigate:
- **Risk Scorer** — predict loss for a location
- **Protection Gap Map** — explore global underinsurance  
- **Model Explainability** — SHAP feature importance
""")

col1, col2, col3 = st.columns(3)
col1.metric("Events Analysed", "4,271", "2000–2026")
col2.metric("Model Coverage", "90.3%", "conformal prediction")
col3.metric("Countries Covered", "100+", "global")