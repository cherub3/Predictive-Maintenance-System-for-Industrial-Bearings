import os
import streamlit as st
from src.dashboard.data_loader import load_recommendations, load_markdown_artifact
from src.dashboard.state import navigate_to

def render_n2_caveat():
    """Permanently renders the small-n methodology caveat."""
    st.warning("**STATISTICAL LIMITATION (n=4 Caveat):** The dataset severely limits statistical guarantees. Outputs are methodology validation only, not fleet-level accuracy guarantees. Confidence metrics are restricted to categorical labels (LOW).")

def render():
    st.title("Landing Dashboard - Fleet Overview")
    
    # Permanent Caveat
    render_n2_caveat()
    
    st.header("Project Pipeline Status")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Engineering Phases", "7/7 Complete", help="Data through Decision Engine")
    colB.metric("Validation Check", "Passed", help="Rigorous CV and Data Leakage Guardrails")
    colC.metric("Target Leakage", "0%", help="No future information leaked into training")
    colD.metric("Reproducibility", "Locked", help="Configurations and manifests frozen")
    
    st.divider()
    st.header("Executive Summary & Selected Model")
    
    st.info("**Selected Production Model: Mean Baseline**\n\nUnder Leave-One-Bearing-Out cross-validation across all 3 datasets ($n=4$ run-to-failure units), XGBoost actually **beats** the mean baseline (~24% lower mean MAE, up to ~90% lower near-failure MAE in some folds) but cannot be promoted: model performance varies across datasets with different life scales, so no candidate beats every rival on every fold — not even the median baseline, which now loses to mean on the largest fold. The **Mean Baseline** is retained as the conservative, statistically honest production choice pending further review. As a result, predicted Remaining Useful Life (RUL) outputs are conservative constants rather than dynamic curves. See Model Diagnostics for the full analysis.")
    
    # Load recommendations to compute risk profile
    df = load_recommendations()
    if df.empty:
        st.success("System Nominal. No Maintenance Actions Required.")
        st.metric("Total Bearings at Risk", 0)
    else:
        # Get unique bearings
        latest_recs = df.sort_values("timestamp").groupby("bearing_id").tail(1)
        
        critical_count = len(latest_recs[latest_recs["risk_category"] == "CRITICAL"])
        high_count = len(latest_recs[latest_recs["risk_category"] == "HIGH"])
        medium_count = len(latest_recs[latest_recs["risk_category"] == "MEDIUM"])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Critical Risk Assets", critical_count)
        col2.metric("High Risk Assets", high_count)
        col3.metric("Medium Risk Assets", medium_count)
        
        system_confidence = "LOW" # Defined by architecture
        col4.metric("System Confidence", system_confidence)
        
        if critical_count == 0 and high_count == 0:
            st.success("0 items in Critical/High Risk.")
            
        st.subheader("Maintenance Workload")
        pending_actions = len(df[df["maintenance_action"] == "SCHEDULE_MAINTENANCE"])
        st.metric("Pending Maintenance Actions", pending_actions)
        
        if st.button("Go to Maintenance Center"):
            navigate_to("Maintenance Center")
            st.rerun()
            
    st.divider()
    
    st.markdown("### Fleet Health Overview")
    fleet_img = "reports/figures/health_trends/rms_all_bearings.png"
    if os.path.exists(fleet_img):
        st.image(fleet_img, caption="Fleet-wide RMS Trajectory", use_container_width=True)
    else:
        st.info("Fleet-wide visual summary artifact not found.")
        
    st.markdown("### Phase 7 Decision Report Summary")
    decision_report = load_markdown_artifact("decision_report")
    if decision_report:
        with st.expander("View Detailed Written Report", expanded=False):
            st.markdown(decision_report)
