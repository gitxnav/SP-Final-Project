"""
FastAPI application tests
Tests all API endpoints including ML predictions
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test that the root endpoint returns successfully"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_endpoint_success():
    """Test the ML prediction endpoint with valid data"""
    test_data = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "class_name" in data
    assert 0 <= data["prediction"] <= 2
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_invalid_features():
    """Test prediction endpoint with wrong number of features"""
    test_data = {
        "features": [5.1, 3.5]  # Only 2 features instead of 4
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400
    assert "Expected 4 features" in response.json()["detail"]


def test_predict_endpoint_missing_data():
    """Test prediction endpoint with missing data"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error


def test_model_info_endpoint():
    """Test the model info endpoint"""
    response = client.get("/model-info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_type" in data
    assert "classes" in data
    assert "n_features" in data
    assert data["n_features"] == 4
    assert len(data["classes"]) == 3


def test_predict_all_classes():
    """Test predictions for different iris classes"""
    test_cases = [
        {"features": [5.1, 3.5, 1.4, 0.2]},  # Likely setosa
        {"features": [6.7, 3.0, 5.2, 2.3]},  # Likely virginica
        {"features": [5.9, 3.0, 4.2, 1.5]},  # Likely versicolor
    ]
    
    for test_data in test_cases:
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["class_name"] in ["setosa", "versicolor", "virginica"]


def test_api_documentation_available():
    """Test that API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "Simple ML API"
