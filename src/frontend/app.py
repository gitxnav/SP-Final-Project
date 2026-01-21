import streamlit as st
from utils.api_client import APIClient

st.set_page_config(page_title="CKD Detection", layout="wide")

# Initialize API client
api = APIClient(base_url="http://backend:8000")  # Docker service name

st.title("🏥 CKD Detection System")

# Sidebar: Health Check
with st.sidebar:
    st.subheader("System Status")
    if st.button("Check API Health"):
        health = api.health_check()
        if health.get("status") == "healthy":
            st.success(f"✅ {health['service']}")
        else:
            st.error(f"❌ {health.get('error', 'Unhealthy')}")

# Main: Prediction Form
st.header("Enter Patient Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    blood_pressure = st.number_input("Blood Pressure", 50, 200, 80)
    specific_gravity = st.slider("Specific Gravity", 1.005, 1.025, 1.020, 0.005)

with col2:
    albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    sugar = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])

# Predict Button
if st.button("🔍 Predict CKD", type="primary"):
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
            st.success("Prediction Complete!")
            
            col1, col2 = st.columns(2)
            col1.metric("Result", result["prediction"])
            col2.metric("Confidence", f"{result['probability']*100:.1f}%")
            
            # Progress bar
            st.progress(result["probability"])

# Embedded API Docs
st.divider()
st.subheader("📖 API Documentation")
st.components.v1.iframe(
    "http://backend:8000/docs",
    height=600,
    scrolling=True
)