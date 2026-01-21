import requests
import streamlit as st

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def predict(self, data: dict):
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None