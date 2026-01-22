from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import mlflow
import mlflow.sklearn
import os
import pickle
import numpy as np
from datetime import datetime

app = FastAPI(title="CKD Detection API", version="1.0.0")

# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ckd_detection")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PredictionInput(BaseModel):
    age: int
    blood_pressure: int
    specific_gravity: float
    albumin: int
    sugar: int

class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    model_version: str
    run_id: str

# Global model variable
loaded_model = None
model_info = {}

def load_model():
    """Load the latest model from MLflow"""
    global loaded_model, model_info
    try:
        # Try to load production model
        model_uri = "models:/ckd_model/production"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        model_info = {
            "version": "production",
            "loaded_at": datetime.now().isoformat()
        }
    except:
        # Fallback to mock model
        loaded_model = None
        model_info = {
            "version": "mock",
            "loaded_at": datetime.now().isoformat()
        }

@app.on_event("startup")
def startup_event():
    """Initialize on startup"""
    load_model()

# Routes
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CKD API",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_info": model_info
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """Make prediction with MLflow tracking"""
    
    with mlflow.start_run(run_name="prediction"):
        # Log input parameters
        mlflow.log_params({
            "age": data.age,
            "blood_pressure": data.blood_pressure,
            "specific_gravity": data.specific_gravity,
            "albumin": data.albumin,
            "sugar": data.sugar
        })
        
        # Prepare features
        features = np.array([[
            data.age,
            data.blood_pressure,
            data.specific_gravity,
            data.albumin,
            data.sugar
        ]])
        
        # Predict
        if loaded_model is not None:
            # Use real model
            prediction = loaded_model.predict(features)[0]
            probability = loaded_model.predict_proba(features)[0][1]
        else:
            # Mock prediction
            score = (data.age * 0.01 + data.blood_pressure * 0.02 + 
                    data.albumin * 0.1 + data.sugar * 0.15)
            prediction = 1 if score > 5 else 0
            probability = min(score / 10, 1.0)
        
        # Log prediction
        mlflow.log_metrics({
            "prediction": float(prediction),
            "probability": float(probability)
        })
        
        result = "CKD" if prediction == 1 else "No CKD"
        
        return {
            "prediction": result,
            "probability": round(float(probability), 2),
            "model_version": model_info.get("version", "unknown"),
            "run_id": mlflow.active_run().info.run_id
        }

@app.post("/reload-model")
def reload_model():
    """Reload model from MLflow"""
    load_model()
    return {"status": "success", "model_info": model_info}

@app.get("/mlflow-experiments")
def get_experiments():
    """Get MLflow experiments"""
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiments = client.search_experiments()
        return {
            "experiments": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)