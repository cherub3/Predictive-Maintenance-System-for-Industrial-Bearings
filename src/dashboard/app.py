import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dashboard.state import init_state, get_visible_pages, PERSONAS
from src.dashboard.data_loader import validate_and_load_manifest
from src.dashboard.views import (
    landing,
    maintenance_center,
    bearing_detail,
    signal_analysis,
    model_diagnostics,
    governance
)

# Page Config
st.set_page_config(
    page_title="Bearing RUL Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize State
init_state()

# Lifecycle Phase: Startup / Validation
try:
    manifest = validate_and_load_manifest()
except Exception as e:
    st.error("### System Failed Closed")
    st.error("Validation Layer intercepted an error. Dashboard enters Degraded Mode.")
    st.exception(e)
    st.stop()

# Sidebar Setup
with st.sidebar:
    st.markdown("### Persona / Mode Switch")
    
    # Persona Switching Behaviour
    selected_persona = st.selectbox(
        "View As:",
        options=PERSONAS,
        index=PERSONAS.index(st.session_state.current_persona)
    )
    
    if selected_persona != st.session_state.current_persona:
        st.session_state.current_persona = selected_persona
        # Local state resets (e.g., filters) per architecture
        st.session_state.maintenance_priority_filter = []
        st.session_state.maintenance_risk_filter = []
        # If active page is no longer visible in new persona, reset to Landing
        if st.session_state.active_page not in get_visible_pages():
            st.session_state.active_page = "Landing Dashboard"
        st.rerun()

    st.markdown("### Navigation")
    visible_pages = get_visible_pages()
    
    for page in visible_pages:
        # Highlight active page
        is_active = (page == st.session_state.active_page)
        button_type = "primary" if is_active else "secondary"
        
        if st.button(page, use_container_width=True, type=button_type):
            st.session_state.active_page = page
            
            # Context drop logic for up-leveling
            if page in ["Landing Dashboard", "Governance & Audit"]:
                st.session_state.selected_bearing_id = None
                
            st.rerun()
            
    st.divider()
    st.caption(f"Experiment: {manifest.get('experiment_id', 'UNKNOWN')}")
    st.caption("Active Session: Connected")

# Render active page
active_page = st.session_state.active_page

if active_page == "Landing Dashboard":
    landing.render()
elif active_page == "Maintenance Center":
    maintenance_center.render()
elif active_page == "Bearing Detail":
    bearing_detail.render()
elif active_page == "Signal Analysis":
    signal_analysis.render()
elif active_page == "Model Diagnostics":
    model_diagnostics.render()
elif active_page == "Governance & Audit":
    governance.render()
else:
    st.error("Page not found.")
