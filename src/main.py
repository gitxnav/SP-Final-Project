"""
FastAPI ML Application
Serves predictions from a trained ML model
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path
from typing import List
import uvicorn
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(
    title="Simple ML API",
    description="A reproducible FastAPI application with ML model",
    version="1.0.0"
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    class_name: str

# Model path
MODEL_PATH = Path(__file__).parent.parent / "models" / "iris_model.pkl"

def get_model():
    """Load or create a simple model"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        # Create a simple dummy model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        
        iris = load_iris()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(iris.data, iris.target)
        
        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        
        return model

model = get_model()
class_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the ML API",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Make predictions",
            "/model-info": "Get model information",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction using the ML model
    
    Expected input: 4 features for iris classification
    """
    try:
        # Validate input
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 4 features, got {len(request.features)}"
            )
        
        # Prepare data
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            class_name=class_names[prediction]
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": type(model).__name__,
        "classes": class_names,
        "n_features": 4
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)