import os
import streamlit as st
from src.dashboard.data_loader import load_markdown_artifact
from src.dashboard.state import navigate_to

def render():
    st.title("Model Diagnostics (AI Transparency)")
    
    bearing_id = st.session_state.selected_bearing_id
    if not bearing_id:
        st.warning("No bearing selected. Showing general model diagnostics.")
    else:
        st.header(f"Diagnostics contextualized for Asset: {bearing_id}")
        # Normally we'd load local SHAP here, but we only have markdown reports
        st.info("Diagnostic Unavailable: Granular local SHAP values are not present in the frozen artifacts. Displaying global importance.")
        
    st.subheader("Baseline Model Bias (Phase 6 Selection)")
    st.info("**Why Mean Is In Production (Not 'No Signal'):** Under Leave-One-Bearing-Out (LOBO) cross-validation across all 3 datasets ($n=4$ run-to-failure units), XGBoost actually **beats** the mean baseline — ~24% lower mean MAE and ~90% lower near-failure MAE in some folds (the safety-critical region). It is not promoted because the every-fold-beat rule fails: model performance varies across datasets with different life scales, so no candidate — including the median baseline itself, which now loses to mean on the largest fold — beats every rival on every fold. The demotion ladder falls back one rung further than at n=2 (to **Mean**, not Median), because Median no longer clears its own promotion gate against Mean. The **Mean Baseline** is retained as the conservative, production-safe choice. See `ENGINEERING_DECISIONS.md` (ED-006) for the full evidence, and ED-004 for the original n=2 analysis.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Type", "mean", help="Fallback baseline model")
    with col2:
        st.metric("Phase 6 MAE", "16539 mins", delta="+4262 mins (Late Warning Bias)", delta_color="inverse")
    with col3:
        st.metric("Late-Prediction Rate", "67.6%", help="Percentage of predictions that arrived too late")
    with col4:
        st.metric("Stability Ratio", "1.46", help="Ratio of max/min fold MAE (1.0 = perfectly stable across folds)")
        
    st.divider()
    
    tabs = st.tabs(["Feature Importance", "Model Evaluation", "Model Selection"])
    
    with tabs[0]:
        st.markdown("### Feature Importance")
        st.caption("SHAP governance: SHAP is used ONLY for model interpretation, not selection.")
        report = load_markdown_artifact("feature_importance_report")
        if report:
            st.markdown(report)
        else:
            st.error("Feature Importance Report not found.")
            
    with tabs[1]:
        st.markdown("### Phase 6 Evaluation")
        report = load_markdown_artifact("evaluation_report")
        if report:
            st.markdown(report)
        else:
            st.error("Evaluation Report not found.")
            
    with tabs[2]:
        st.markdown("### Model Selection Justification")
        report = load_markdown_artifact("model_selection_report")
        if report:
            st.markdown(report)
        else:
            st.error("Model Selection Report not found.")

        lit_path = "reports/LITERATURE_BENCHMARK.md"
        if os.path.exists(lit_path):
            with st.expander("How does this compare to published NASA IMS bearing RUL research?"):
                with open(lit_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            
    if st.button("View Governance & Audit Trace"):
        navigate_to("Governance & Audit")
        st.rerun()
