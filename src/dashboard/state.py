import streamlit as st

# Personas and Modes
PERSONAS = [
    "Executive",
    "Maintenance Manager",
    "Reliability Engineer",
    "Machine Learning Engineer",
    "Auditor"
]

def init_state():
    """Initializes the Dashboard State Architecture."""
    if "current_persona" not in st.session_state:
        st.session_state.current_persona = "Executive"
    
    if "selected_bearing_id" not in st.session_state:
        st.session_state.selected_bearing_id = None
        
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Landing Dashboard"
        
    if "maintenance_priority_filter" not in st.session_state:
        st.session_state.maintenance_priority_filter = []
        
    if "maintenance_risk_filter" not in st.session_state:
        st.session_state.maintenance_risk_filter = []

def get_visible_pages():
    """Determines visible pages based on Persona Switching Behaviour."""
    persona = st.session_state.current_persona
    
    if persona == "Executive":
        return ["Landing Dashboard", "Governance & Audit"]
    elif persona == "Maintenance Manager":
        return ["Landing Dashboard", "Maintenance Center", "Bearing Detail"]
    elif persona == "Reliability Engineer":
        return ["Bearing Detail", "Signal Analysis"]
    elif persona == "Machine Learning Engineer":
        return ["Bearing Detail", "Model Diagnostics"]
    elif persona == "Auditor":
        return ["Landing Dashboard", "Governance & Audit"]
    else:
        return ["Landing Dashboard"]

def navigate_to(page: str, bearing_id: str = None):
    """Handles cross-page synchronization and navigation."""
    st.session_state.active_page = page
    
    if bearing_id is not None:
        st.session_state.selected_bearing_id = bearing_id
        
    # Context reset rules
    if page in ["Landing Dashboard", "Governance & Audit", "Maintenance Center"]:
        # We generally do not drop the bearing context completely unless explicitly cleared,
        # but the architecture states: "Drilling 'up' to the Landing Dashboard or Governance pages 
        # intentionally drops the bearing_id context to return to fleet-level views."
        if page in ["Landing Dashboard", "Governance & Audit"]:
            st.session_state.selected_bearing_id = None
