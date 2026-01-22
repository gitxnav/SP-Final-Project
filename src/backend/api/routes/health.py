"""
Health Check Routes - CKD Detection
System health and monitoring endpoints
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import mlflow
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: str
    version: str
    mlflow: dict
    system: dict
    models: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Complete health check with system metrics
    
    Returns comprehensive system status including:
    - Service health
    - MLflow connection
    - System resources (CPU, Memory, Disk)
    - Models loaded
    """
    try:
        # Import models from prediction router
        from api.routes.prediction import models
        
        # Check MLflow connection
        mlflow_status = "healthy"
        mlflow_uri = "unknown"
        try:
            mlflow_uri = mlflow.get_tracking_uri()
            mlflow.tracking.MlflowClient().list_experiments()
        except Exception as e:
            mlflow_status = "unhealthy"
            logger.warning(f"MLflow health check failed: {str(e)}")
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            service="CKD Detection API",
            version="1.0.0",
            mlflow={
                "status": mlflow_status,
                "tracking_uri": mlflow_uri
            },
            system={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            models={
                "loaded": len(models) > 0,
                "count": len(models),
                "names": list(models.keys())
            }
        )
    
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            service="CKD Detection API",
            version="1.0.0",
            mlflow={"status": "unknown", "tracking_uri": "unknown"},
            system={"error": str(e)},
            models={"loaded": False, "count": 0, "names": []}
        )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe
    Returns 200 if service is ready to accept requests
    """
    try:
        from api.routes.prediction import models
        
        # Check if models are loaded
        if not models:
            return {"ready": False, "reason": "No models loaded"}
        
        # Check MLflow connection
        try:
            mlflow.tracking.MlflowClient().list_experiments()
        except Exception as e:
            logger.warning(f"MLflow not ready: {str(e)}")
            # Continue - MLflow is optional for predictions
        
        return {"ready": True, "models_loaded": len(models)}
    
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe
    Simple check that the service is running
    """
    return {"alive": True, "timestamp": datetime.now().isoformat()}


@router.get("/metrics")
async def get_metrics():
    """
    Get detailed system metrics
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return {"error": str(e)}