"""
FastAPI Main Application - CKD Detection
Complete REST API with MLflow tracking, model training, and inference
Integrated from step08_model_inference.py
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# MLflow imports
import mlflow
import mlflow.sklearn

# Import configuration
from config.mlflow_config import MLflowConfig
from config.settings import settings

# Import routers
from api.routes import health, prediction, mlflow as mlflow_routes, training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CKD Detection API",
    description="REST API for Chronic Kidney Disease prediction using ML models with MLflow integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
mlflow_config = None
models = {}


# Pydantic models for request/response (for backward compatibility)
class PatientData(BaseModel):
    """Patient data for prediction"""
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "patients": [
                    {
                        "hemo": 15.4, "sg": 1.020, "sc": 1.2,
                        "rbcc": 5.2, "pcv": 44.0, "htn": 1.0,
                        "dm": 1.0, "bp": 80.0, "age": 48.0
                    },
                    {
                        "hemo": 11.3, "sg": 1.020, "sc": 0.8,
                        "rbcc": 4.5, "pcv": 38.0, "htn": 0.0,
                        "dm": 0.0, "bp": 50.0, "age": 7.0
                    }
                ]
            }
        }


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


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    mlflow_enabled: bool
    timestamp: str


# Include routers (modular approach)
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"]) 
app.include_router(mlflow_routes.router, prefix="/api/v1/mlflow", tags=["MLflow"])
app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global mlflow_config, models
    
    logger.info("🚀 Starting CKD Detection API...")
    
    # Create necessary directories
    directories = [
        "models",
        "data/processed",
        "figures/models",
        "reports/models",
        "mlruns"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ready: {directory}")
    
    # Initialize MLflow
    try:
        mlflow_config = MLflowConfig(experiment_name="CKD_Detection")
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("CKD_Detection")
        logger.info(f"✅ MLflow initialized: {settings.MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"⚠️ MLflow initialization failed: {str(e)}")
        mlflow_config = None
    
    # Load models
    models_dir = Path("models")
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
        logger.warning("⚠️ No trained models found!")
        logger.warning("📝 To train models, use one of these methods:")
        logger.warning("   1. API: POST /api/v1/training/train-all")
        logger.warning("   2. Script: python core/model_training.py")
        logger.warning("   3. Docker: docker-compose exec backend python -m core.model_training")
    else:
        logger.info(f"✅ Successfully loaded {loaded_count} models")
    
    logger.info("🎉 CKD Detection API ready!")
    logger.info(f"📖 API Docs: http://localhost:{settings.API_PORT}/docs")
    logger.info(f"📊 MLflow UI: {settings.MLFLOW_TRACKING_URI}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down CKD Detection API...")


# Root endpoints (keep original style for backward compatibility)

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CKD Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "models_loaded": len(models),
        "mlflow_enabled": mlflow_config is not None,
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "predict_ensemble": "/predict/ensemble",
            "predict_example": "/predict/example",
            "train": "/api/v1/training/train-all",
            "mlflow": "/mlflow/experiments"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint (backward compatible)
    
    For detailed health check, use /api/v1/health
    """
    return HealthResponse(
        status="healthy" if models else "unhealthy",
        models_loaded=list(models.keys()),
        mlflow_enabled=mlflow_config is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """
    List available models
    
    Returns information about all loaded models
    """
    model_info = {}
    
    for name, model in models.items():
        model_info[name] = {
            "name": name,
            "type": type(model).__name__,
            "has_probability": hasattr(model, 'predict_proba')
        }
    
    return {
        "models": model_info,
        "total": len(models)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    patient: PatientData,
    model_name: str = Query("KNN", description="Model to use for prediction")
):
    """
    Make prediction for a single patient (backward compatible endpoint)
    
    - **patient**: Patient data with all required features
    - **model_name**: Name of model to use (KNN, SVM, GradientBoosting, HistGradientBoosting)
    
    For new implementations, consider using /api/v1/predict
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}"
        )
    
    try:
        # Prepare input data
        features = patient.dict()
        X = pd.DataFrame([features])
        
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
            confidence = 1.0
        
        # Log to MLflow if enabled
        run_id = None
        if mlflow_config:
            try:
                with mlflow.start_run(run_name=f"Inference_{model_name}"):
                    mlflow.log_params({f"input_{k}": v for k, v in features.items()})
                    mlflow.log_metric("prediction", int(prediction))
                    if probability:
                        mlflow.log_metric("confidence", confidence)
                    run_id = mlflow.active_run().info.run_id
            except Exception as e:
                logger.warning(f"MLflow logging failed: {str(e)}")
        
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


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
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
        
        for patient in request.patients:
            # Use the single prediction endpoint logic
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


@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse, tags=["Prediction"])
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


@app.get("/predict/example", tags=["Prediction"])
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


@app.post("/reload-model", tags=["Models"])
async def reload_model():
    """
    Reload all models from disk
    
    Useful after training new models
    """
    global models
    
    try:
        models = {}  # Clear existing models
        
        models_dir = Path("models")
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
                    logger.info(f"Reloaded model: {model_name}")
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Error reloading {model_name}: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Reloaded {loaded_count} models",
            "models": list(models.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow-experiments", tags=["MLflow"])
async def get_mlflow_experiments():
    """
    Get MLflow experiments information (backward compatible)
    
    For detailed MLflow operations, use /api/v1/mlflow/* endpoints
    """
    if not mlflow_config:
        raise HTTPException(status_code=503, detail="MLflow not enabled")
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            })
        
        return {
            "experiments": exp_data,
            "total": len(exp_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/runs/{experiment_name}", tags=["MLflow"])
async def get_mlflow_runs(experiment_name: str, limit: int = Query(10, ge=1, le=100)):
    """Get recent MLflow runs for an experiment"""
    if not mlflow_config:
        raise HTTPException(status_code=503, detail="MLflow not enabled")
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit
        )
        
        runs_data = []
        for run in runs:
            runs_data.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get('mlflow.runName', 'N/A'),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params
            })
        
        return {
            "experiment_name": experiment_name,
            "runs": runs_data,
            "total": len(runs_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


# Export for uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )