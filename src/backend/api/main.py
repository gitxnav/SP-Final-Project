"""
FastAPI Main Application - CKD Detection
Integrated REST API with MLflow tracking and routing
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import mlflow

# Import configuration
from backend.config.mlflow_config import MLflowConfig
from backend.config.settings import settings

# Import routers
from backend.api.routes import health, prediction, mlflow as mlflow_routes

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

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(mlflow_routes.router, prefix="/api/v1/mlflow", tags=["MLflow"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global mlflow_config
    
    logger.info("🚀 Starting CKD Detection API...")
    
    # Initialize MLflow
    try:
        mlflow_config = MLflowConfig(experiment_name="CKD_Detection")
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("CKD_Detection")
        logger.info(f"✅ MLflow initialized: {settings.MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"⚠️ MLflow initialization failed: {str(e)}")
        mlflow_config = None
    
    # Load models (delegated to prediction router)
    from backend.api.routes.prediction import load_models
    loaded_count = load_models()
    logger.info(f"✅ Loaded {loaded_count} models successfully")
    
    logger.info("🎉 CKD Detection API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down CKD Detection API...")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "CKD Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "models": "/api/v1/models",
            "mlflow": "/api/v1/mlflow/experiments"
        },
        "mlflow_enabled": mlflow_config is not None
    }


# Export for uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )