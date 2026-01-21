import requests
import streamlit as st
from typing import Optional, Dict, Any

class APIClient:
    def __init__(self, base_url: str = "http://backend:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Generic request method with error handling"""
        try:
            url = f"{self.base_url}{self.api_prefix}{endpoint}"
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out")
            return None
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to backend API")
            return None
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e.response.status_code}")
            return None
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return None
    
    def health_check(self) -> Optional[Dict]:
        """Get system health status"""
        return self._make_request("GET", "/health")
    
    def predict(self, data: Dict[str, Any]) -> Optional[Dict]:
        """Make a prediction"""
        return self._make_request("POST", "/predict", json=data)
    
    def train_model(self, config: Optional[Dict] = None) -> Optional[Dict]:
        """Trigger model training"""
        params = config if config else {}
        return self._make_request("POST", "/train", params=params)
    
    def reload_model(self) -> Optional[Dict]:
        """Reload model from MLflow"""
        return self._make_request("POST", "/reload-model")
    
    def get_experiments(self) -> Optional[Dict]:
        """Get MLflow experiments"""
        return self._make_request("GET", "/experiments")
    
    def get_models(self) -> Optional[Dict]:
        """Get registered models"""
        return self._make_request("GET", "/models")
    
    def get_runs(self, experiment_id: str) -> Optional[Dict]:
        """Get runs for experiment"""
        return self._make_request("GET", f"/experiments/{experiment_id}/runs")
    
    def call_endpoint(self, endpoint: str) -> Optional[Dict]:
        """Generic endpoint caller"""
        return self._make_request("GET", endpoint)