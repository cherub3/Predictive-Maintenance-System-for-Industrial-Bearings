import os
import streamlit as st
from src.dashboard.data_loader import load_recommendations, load_feature_store
from src.dashboard.state import navigate_to

def render():
    st.title("Bearing Detail")
    
    try:
        df_features = load_feature_store()
        all_bearings = sorted(df_features["bearing_id"].unique())
    except Exception:
        all_bearings = []
        
    if not all_bearings:
        st.error("Failed to load asset list from feature store.")
        return

    current_bearing = st.session_state.selected_bearing_id
    default_index = all_bearings.index(current_bearing) if current_bearing in all_bearings else 0
    
    selected_bearing = st.selectbox("Select Asset", options=all_bearings, index=default_index)
    
    if selected_bearing != current_bearing:
        st.session_state.selected_bearing_id = selected_bearing
        st.rerun()

    bearing_id = selected_bearing
    st.header(f"Asset: {bearing_id}")
    
    df = load_recommendations()
    bearing_df = df[df["bearing_id"] == bearing_id].sort_values("timestamp")
    
    if bearing_df.empty:
        st.info("No maintenance recommendation exists.\n\nThis bearing is right-censored and therefore has no Phase 7 maintenance decision.")
    else:
        latest = bearing_df.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Risk Category", latest["risk_category"])
        col2.metric("Maintenance Action", latest["maintenance_action"])
        col3.metric("Priority Score", latest["priority_score"])
        
        st.subheader("Historical Risk Trajectory (Phase 6 Outputs)")
        st.info("**Engineering Context:** The charts below appear as flatlines because the **Mean Baseline** is the current production model. This is not because the features lack degradation signal — XGBoost actually beats the mean baseline by ~24% mean MAE on this data — but because with 4 run-to-failure bearings across 3 datasets of different life scales, no candidate (including the median baseline) can be validated as a non-overfitting, every-fold winner. The predictions are statistically honest constants, not dynamic extrapolations. See **Model Diagnostics** for the full justification.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Predicted RUL Degradation (Minutes)**")
            st.line_chart(bearing_df.set_index("timestamp")["min_predicted_rul_minutes"])
        with c2:
            st.markdown("**Maintenance Priority Escalation**")
            st.line_chart(bearing_df.set_index("timestamp")["priority_score"])
            
    st.subheader("Asset Visual Context (Phase 3 Checkpoint)")
    spectral_img = f"reports/figures/spectral_checkpoints/{bearing_id}_spectral_evolution.png"
    if os.path.exists(spectral_img):
        st.image(spectral_img, caption=f"Spectral Evolution - {bearing_id}", use_container_width=True)
    else:
        st.info("No Phase 3 Spectral Checkpoint artifact found for this asset.")
    
    st.divider()
    st.subheader("Deep Diagnostics Drill-down")
    colA, colB = st.columns(2)
    
    with colA:
        if st.button("Physical Diagnostics (Signal Analysis)"):
            navigate_to("Signal Analysis", bearing_id=bearing_id)
            st.rerun()
            
    with colB:
        if st.button("AI Transparency (Model Diagnostics)"):
            navigate_to("Model Diagnostics", bearing_id=bearing_id)
            st.rerun()
