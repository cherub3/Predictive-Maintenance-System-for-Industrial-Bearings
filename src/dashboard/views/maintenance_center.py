import streamlit as st
from src.dashboard.data_loader import load_recommendations, load_feature_store
from src.dashboard.state import navigate_to

def render():
    st.title("Maintenance Center")
    
    df = load_recommendations()
    
    if df.empty:
        st.success("System Nominal. No tasks required.")
        return
        
    st.header("Actionable Maintenance Tasks")
    
    col1, col2 = st.columns(2)
    
    # Priority Filter
    priorities = [1, 2, 3, 4, 5]
    selected_priorities = col1.multiselect(
        "Filter by Priority (1=Highest)", 
        options=priorities, 
        default=st.session_state.maintenance_priority_filter
    )
    st.session_state.maintenance_priority_filter = selected_priorities
    
    # Risk Filter
    available_risks = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "HEALTHY"]
    selected_risks = col2.multiselect(
        "Filter by Risk Category", 
        options=available_risks,
        default=st.session_state.maintenance_risk_filter
    )
    st.session_state.maintenance_risk_filter = selected_risks
    
    filtered_df = df.copy()
    if selected_priorities:
        filtered_df = filtered_df[filtered_df["priority_score"].isin(selected_priorities)]
    if selected_risks:
        filtered_df = filtered_df[filtered_df["risk_category"].isin(selected_risks)]
        
    if filtered_df.empty:
        st.info("No assets match the selected filter.")
        if st.button("Clear Filters"):
            st.session_state.maintenance_priority_filter = []
            st.session_state.maintenance_risk_filter = []
            st.rerun()
        
    else:
        # Get the latest recommendation per bearing
        latest_df = filtered_df.sort_values("timestamp").groupby("bearing_id").tail(1).reset_index(drop=True)
        
        st.dataframe(
            latest_df[['bearing_id', 'risk_category', 'maintenance_action', 'priority_score', 'timestamp']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "bearing_id": st.column_config.TextColumn("Asset ID", width="medium"),
                "risk_category": st.column_config.TextColumn("Risk", help="Business risk categorization"),
                "maintenance_action": st.column_config.TextColumn("Action", width="large"),
                "priority_score": st.column_config.ProgressColumn(
                    "Priority (1=High, 5=Low)",
                    help="Maintenance priority score",
                    format="%d",
                    min_value=0,
                    max_value=5,
                ),
                "timestamp": st.column_config.DatetimeColumn("Last Updated", format="D MMM YYYY, h:mm a")
            }
        )
        
    st.divider()
    st.subheader("Asset Drill-down")
    
    try:
        df_features = load_feature_store()
        all_bearings = sorted(df_features["bearing_id"].unique())
    except Exception:
        all_bearings = []
        
    if all_bearings:
        selected_bearing = st.selectbox("Select an asset to investigate:", options=all_bearings)
        
        if st.button(f"View Bearing Detail for {selected_bearing}"):
            navigate_to("Bearing Detail", bearing_id=selected_bearing)
            st.rerun()
    else:
        st.error("Failed to load asset list from feature store.")
