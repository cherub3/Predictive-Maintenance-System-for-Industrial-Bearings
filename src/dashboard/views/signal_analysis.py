import os
import streamlit as st
from src.dashboard.data_loader import load_feature_store, load_markdown_artifact
from src.dashboard.state import navigate_to

def render():
    st.title("Signal Analysis (Physical Diagnostics)")
    
    bearing_id = st.session_state.selected_bearing_id
    if not bearing_id:
        st.warning("No bearing selected.")
        if st.button("Back to Maintenance Center"):
            navigate_to("Maintenance Center")
            st.rerun()
        return
        
    st.header(f"Asset: {bearing_id}")
    
    # Lazy load feature store
    try:
        df = load_feature_store()
    except Exception:
        st.error("Failed to load diagnostic evidence. Please try again.")
        return
        
    bearing_df = df[df["bearing_id"] == bearing_id].sort_values("timestamp")
    
    if bearing_df.empty:
        st.info("No signal data found for this asset.")
        return
        
    st.subheader("High-Fidelity Phase 3 Diagnostics")
    
    colA, colB = st.columns(2)
    with colA:
        rms_img = f"reports/figures/channel_comparison/{bearing_id}_rms_xy.png"
        if os.path.exists(rms_img):
            st.image(rms_img, caption="RMS X/Y Channel Comparison")
        else:
            st.info("No Phase 3 RMS comparison artifact found.")
            
    with colB:
        kurtosis_img = f"reports/figures/channel_comparison/{bearing_id}_kurtosis_xy.png"
        if os.path.exists(kurtosis_img):
            st.image(kurtosis_img, caption="Kurtosis X/Y Channel Comparison")
        else:
            st.info("No Phase 3 Kurtosis artifact found.")
            
    cf_img = f"reports/figures/channel_comparison/{bearing_id}_crest_factor_xy.png"
    if os.path.exists(cf_img):
        st.image(cf_img, caption="Crest Factor Evolution")

    st.divider()
    st.markdown("### Engineering Findings Reference")
    engineering_findings = load_markdown_artifact("engineering_findings")
    if engineering_findings:
        with st.expander("View Phase 3 Findings"):
            st.markdown(engineering_findings)
