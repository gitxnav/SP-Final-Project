"""
Prediction Routes - CKD Detection
Handles all prediction endpoints with your existing model logic
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

logger = logging.getLogger(__name__)
router = APIRouter()

# Global models dictionary
models = {}


# Pydantic models (your existing structure)
class PatientData(BaseModel):
    """Patient data for prediction - YOUR EXISTING FEATURES"""
    hemo: float = Field(..., ge=0.0, le=20.0, description="Hemoglobin (g/dL)")
    sg: float = Field(..., ge=1.000, le=1.030, description="Specific Gravity")
    sc: float = Field(..., ge=0.0, le=20.0, description="Serum Creatinine (mg/dL)")
    rbcc: float = Field(..., ge=0.0, le=10.0, description="Red Blood Cell Count (millions/cmm)")
    pcv: float = Field(..., ge=0.0, le=60.0, description="Packed Cell Volume (%)")
    htn: float = Field(..., ge=0.0, le=1.0, description="Hypertension (0=No, 1=Yes)")
    dm: float = Field(..., ge=0.0, le=1.0, description="Diabetes Mellitus (0=No, 1=Yes)")
    bp: float = Field(..., ge=0.0, le=200.0, description="Blood Pressure (mmHg)")
    age: float = Field(..., ge=0.0, le=120.0, description="Age (years)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "hemo": 15.4,
                "sg": 1.020,
                "sc": 1.2,
                "rbcc": 5.2,
                "pcv": 44.0,
                "htn": 1.0,
                "dm": 1.0,
                "bp": 80.0,
                "age": 48.0
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: str
    prediction_numeric: int
    probability: Optional[Dict[str, float]] = None
    confidence: float
    model_used: str
    timestamp: str
    mlflow_run_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_patients: int
    timestamp: str


class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction response"""
    individual_predictions: Dict[str, PredictionResponse]
    consensus: Dict[str, Any]
    timestamp: str


# Helper function to load models
def load_models() -> int:
    """Load all models from the models directory"""
    global models
    
    models_dir = Path("models")
    
    # Your existing model files
    model_files = {
        "KNN": "knn_model.pkl",
        "SVM": "svm_model.pkl",
        "GradientBoosting": "gb_imputed_model.pkl",
        "HistGradientBoosting": "hist_gb_model.pkl"
    }
    
    loaded_count = 0
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Loaded model: {model_name}")
                loaded_count += 1
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {str(e)}")
        else:
            logger.warning(f"⚠️ Model file not found: {model_path}")
    
    if loaded_count == 0:
        logger.error("❌ No models loaded! Predictions will fail.")
    
    return loaded_count


@router.get("/models")
async def list_models():
    """List available models and their information"""
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            "name": name,
            "type": type(model).__name__,
            "has_probability": hasattr(model, 'predict_proba'),
            "loaded": True
        }
    
    return {
        "models": model_info,
        "total": len(models),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    patient: PatientData,
    model_name: str = Query("KNN", description="Model to use for prediction")
):
    """
    Make prediction for a single patient
    
    - **patient**: Patient data with all required features
    - **model_name**: Name of model to use (KNN, SVM, GradientBoosting, HistGradientBoosting)
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )
    
    try:
        # Start MLflow run for tracking
        with mlflow.start_run(run_name=f"Inference_{model_name}_{datetime.now().strftime('%H%M%S')}"):
            # Prepare input data
            features = patient.dict()
            X = pd.DataFrame([features])
            
            # Log input parameters to MLflow
            mlflow.log_params({f"input_{k}": v for k, v in features.items()})
            mlflow.log_param("model_used", model_name)
            
            # Get model
            model = models[model_name]
            
            # Make prediction
            prediction = model.predict(X)[0]
            prediction_label = "ckd" if prediction == 1 else "notckd"
            
            # Get probability if available
            probability = None
            confidence = 0.0
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                probability = {
                    "notckd": float(proba[0]),
                    "ckd": float(proba[1])
                }
                confidence = float(proba[prediction])
            else:
                confidence = 1.0  # For models without probability
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "prediction": int(prediction),
                "confidence": confidence
            })
            
            # Get run ID
            run_id = mlflow.active_run().info.run_id
            
            logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.3f})")
            
            return PredictionResponse(
                prediction=prediction_label,
                prediction_numeric=int(prediction),
                probability=probability,
                confidence=confidence,
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                mlflow_run_id=run_id
            )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model_name: str = Query("KNN", description="Model to use for predictions")
):
    """
    Make predictions for multiple patients
    
    - **request**: Batch prediction request with list of patients
    - **model_name**: Name of model to use
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )
    
    try:
        predictions = []
        
        # Process each patient
        for patient in request.patients:
            pred = await predict(patient, model_name)
            predictions.append(pred)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_patients=len(predictions),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(patient: PatientData):
    """
    Make predictions using all available models and return consensus
    
    - **patient**: Patient data with all required features
    """
    if not models:
        raise HTTPException(status_code=503, detail="No models available")
    
    try:
        individual_predictions = {}
        
        # Get predictions from all models
        for model_name in models.keys():
            pred = await predict(patient, model_name)
            individual_predictions[model_name] = pred
        
        # Calculate consensus
        predictions_list = [p.prediction for p in individual_predictions.values()]
        consensus_pred = max(set(predictions_list), key=predictions_list.count)
        consensus_count = predictions_list.count(consensus_pred)
        consensus_confidence = consensus_count / len(predictions_list)
        
        # Calculate average confidence
        confidences = [p.confidence for p in individual_predictions.values()]
        avg_confidence = sum(confidences) / len(confidences)
        
        consensus = {
            "prediction": consensus_pred,
            "agreement": f"{consensus_count}/{len(predictions_list)} models",
            "consensus_confidence": consensus_confidence,
            "average_model_confidence": avg_confidence,
            "unanimous": consensus_count == len(predictions_list)
        }
        
        return EnsemblePredictionResponse(
            individual_predictions=individual_predictions,
            consensus=consensus,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Ensemble prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/example")
async def get_example_prediction(model_name: str = Query("KNN")):
    """
    Get example prediction with sample data
    
    - **model_name**: Model to use for example prediction
    """
    # Example patient data
    example_patient = PatientData(
        hemo=15.4,
        sg=1.020,
        sc=1.2,
        rbcc=5.2,
        pcv=44.0,
        htn=1.0,
        dm=1.0,
        bp=80.0,
        age=48.0
    )
    
    return await predict(example_patient, model_name)


@router.post("/reload-models")
async def reload_models():
    """Reload all models from disk"""
    try:
        global models
        models = {}  # Clear existing models
        loaded_count = load_models()
        
        return {
            "status": "success",
            "message": f"Reloaded {loaded_count} models",
            "models": list(models.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload error: {str(e)}")