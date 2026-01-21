import streamlit as st
from frontend.utils.api_client import APIClient

st.set_page_config(page_title="CKD Detection", layout="wide")

# Initialize API client
api = APIClient(base_url="http://backend:8000")

st.title("🏥 CKD Detection System with MLflow")

# Sidebar: System Status
with st.sidebar:
    st.subheader("🔧 System Status")
    
    if st.button("Check Health"):
        health = api.health_check()
        if health and health.get("status") == "healthy":
            st.success(f"✅ {health['service']}")
            with st.expander("Details"):
                st.json(health)
        else:
            st.error("❌ Backend Unhealthy")
    
    st.divider()
    
    if st.button("🔄 Reload Model"):
        result = api.reload_model()
        if result:
            st.success("Model reloaded!")
            st.json(result)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 MLflow UI", "📖 API Docs"])

with tab1:
    st.header("Patient Data Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        blood_pressure = st.number_input("Blood Pressure", 50, 200, 80)
        specific_gravity = st.slider("Specific Gravity", 1.005, 1.025, 1.020, 0.005)
    
    with col2:
        albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
        sugar = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
    
    if st.button("🔍 Predict CKD", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            payload = {
                "age": age,
                "blood_pressure": blood_pressure,
                "specific_gravity": specific_gravity,
                "albumin": albumin,
                "sugar": sugar
            }
            
            result = api.predict(payload)
            
            if result:
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Result", result["prediction"])
                col2.metric("Confidence", f"{result['probability']*100:.1f}%")
                col3.metric("Model", result.get("model_version", "N/A"))
                
                st.progress(result["probability"])
                
                with st.expander("🔬 MLflow Run Details"):
                    st.code(f"Run ID: {result.get('run_id', 'N/A')}")
                    st.caption("View this run in MLflow UI tab")

with tab2:
    st.header("📊 MLflow Tracking UI")
    st.components.v1.iframe(
        "http://mlflow:5050",
        height=800,
        scrolling=True
    )

with tab3:
    st.header("📖 FastAPI Documentation")
    st.components.v1.iframe(
        "http://backend:8000/docs",
        height=800,
        scrolling=True
    )