"""
Training Routes - CKD Detection
Endpoints for training models with MLflow tracking
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
from datetime import datetime
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()

# Import your existing training logic
from core.step07_mflow_training import ModelTrainerMLflow

# Training status tracking
training_status = {
    "is_training": False,
    "current_model": None,
    "progress": 0,
    "message": "Not started",
    "results": {},
    "started_at": None,
    "completed_at": None,
    "error": None
}


class TrainingConfig(BaseModel):
    """Training configuration"""
    experiment_name: str = "CKD_Detection"
    test_size: float = 0.2
    random_state: int = 42
    models_to_train: Optional[List[str]] = None  # None = train all
    
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "CKD_Detection",
                "test_size": 0.2,
                "random_state": 42,
                "models_to_train": ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"]
            }
        }


class TrainingResponse(BaseModel):
    """Training response"""
    status: str
    message: str
    started_at: Optional[str] = None
    models_to_train: List[str]


class TrainingStatusResponse(BaseModel):
    """Training status response"""
    is_training: bool
    current_model: Optional[str]
    progress: int
    message: str
    results: Dict
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


def train_models_background(config: TrainingConfig):
    """Background task to train models"""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["started_at"] = datetime.now().isoformat()
        training_status["completed_at"] = None
        training_status["error"] = None
        training_status["results"] = {}
        
        logger.info(f"Starting model training with config: {config}")
        
        # Initialize trainer
        trainer = ModelTrainerMLflow(experiment_name=config.experiment_name)
        
        # Determine which models to train
        all_models = ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"]
        models_to_train = config.models_to_train if config.models_to_train else all_models
        
        total_models = len(models_to_train)
        
        # Train models with normalized data (KNN, SVM)
        if "KNN" in models_to_train or "SVM" in models_to_train:
            normalized_data_path = Path("data/processed/ckd_normalized.csv")
            
            if not normalized_data_path.exists():
                raise FileNotFoundError(
                    f"Normalized data not found at {normalized_data_path}. "
                    "Please run data processing first."
                )
            
            X_train, X_test, y_train, y_test = trainer.load_data(
                str(normalized_data_path),
                test_size=config.test_size,
                random_state=config.random_state
            )
            
            # Train KNN
            if "KNN" in models_to_train:
                training_status["current_model"] = "KNN"
                training_status["message"] = "Training KNN..."
                training_status["progress"] = int((models_to_train.index("KNN") / total_models) * 100)
                
                logger.info("Training KNN model...")
                knn_results = trainer.train_knn(X_train, X_test, y_train, y_test)
                trainer.results["KNN"] = knn_results
                training_status["results"]["KNN"] = {
                    "f1_score": knn_results["f1_score"],
                    "accuracy": knn_results["accuracy"],
                    "mlflow_run_id": knn_results["mlflow_run_id"]
                }
                logger.info(f"KNN training completed. F1: {knn_results['f1_score']:.4f}")
            
            # Train SVM
            if "SVM" in models_to_train:
                training_status["current_model"] = "SVM"
                training_status["message"] = "Training SVM..."
                training_status["progress"] = int((models_to_train.index("SVM") / total_models) * 100)
                
                logger.info("Training SVM model...")
                svm_results = trainer.train_svm(X_train, X_test, y_train, y_test)
                trainer.results["SVM"] = svm_results
                training_status["results"]["SVM"] = {
                    "f1_score": svm_results["f1_score"],
                    "accuracy": svm_results["accuracy"],
                    "mlflow_run_id": svm_results["mlflow_run_id"]
                }
                logger.info(f"SVM training completed. F1: {svm_results['f1_score']:.4f}")
        
        # Train models with imputed data (GradientBoosting, HistGradientBoosting)
        if "GradientBoosting" in models_to_train or "HistGradientBoosting" in models_to_train:
            imputed_data_path = Path("data/processed/ckd_imputed.csv")
            
            if not imputed_data_path.exists():
                raise FileNotFoundError(
                    f"Imputed data not found at {imputed_data_path}. "
                    "Please run data processing first."
                )
            
            X_train, X_test, y_train, y_test = trainer.load_data(
                str(imputed_data_path),
                test_size=config.test_size,
                random_state=config.random_state
            )
            
            # Train Gradient Boosting
            if "GradientBoosting" in models_to_train:
                training_status["current_model"] = "GradientBoosting"
                training_status["message"] = "Training Gradient Boosting..."
                training_status["progress"] = int((models_to_train.index("GradientBoosting") / total_models) * 100)
                
                logger.info("Training Gradient Boosting model...")
                gb_results = trainer.train_gradient_boosting_imputed(X_train, X_test, y_train, y_test)
                trainer.results["GradientBoosting"] = gb_results
                training_status["results"]["GradientBoosting"] = {
                    "f1_score": gb_results["f1_score"],
                    "accuracy": gb_results["accuracy"],
                    "mlflow_run_id": gb_results["mlflow_run_id"]
                }
                logger.info(f"Gradient Boosting training completed. F1: {gb_results['f1_score']:.4f}")
            
            # Train Histogram Gradient Boosting
            if "HistGradientBoosting" in models_to_train:
                training_status["current_model"] = "HistGradientBoosting"
                training_status["message"] = "Training Histogram Gradient Boosting..."
                training_status["progress"] = int((models_to_train.index("HistGradientBoosting") / total_models) * 100)
                
                logger.info("Training Histogram Gradient Boosting model...")
                hist_gb_results = trainer.train_hist_gradient_boosting(X_train, X_test, y_train, y_test)
                trainer.results["HistGradientBoosting"] = hist_gb_results
                training_status["results"]["HistGradientBoosting"] = {
                    "f1_score": hist_gb_results["f1_score"],
                    "accuracy": hist_gb_results["accuracy"],
                    "mlflow_run_id": hist_gb_results["mlflow_run_id"]
                }
                logger.info(f"Histogram Gradient Boosting training completed. F1: {hist_gb_results['f1_score']:.4f}")
        
        # Save results summary
        summary = trainer.save_results()
        
        # Update status
        training_status["is_training"] = False
        training_status["current_model"] = None
        training_status["progress"] = 100
        training_status["message"] = "Training completed successfully"
        training_status["completed_at"] = datetime.now().isoformat()
        training_status["results"]["summary"] = {
            "best_model": summary["best_model"],
            "ranking": summary["ranking"]
        }
        
        logger.info("✅ All models trained successfully!")
        logger.info(f"🏆 Best model: {summary['best_model']}")
        
        # Reload models in prediction router
        try:
            from api.routes.prediction import load_models
            load_models()
            logger.info("✅ Models reloaded in prediction service")
        except Exception as e:
            logger.warning(f"⚠️ Could not reload models: {str(e)}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        training_status["is_training"] = False
        training_status["error"] = str(e)
        training_status["message"] = f"Training failed: {str(e)}"
        training_status["completed_at"] = datetime.now().isoformat()
        raise


@router.post("/train-all", response_model=TrainingResponse)
async def train_all_models(
    background_tasks: BackgroundTasks,
    config: TrainingConfig = TrainingConfig()
):
    """
    Train all models with MLflow tracking (background task)
    
    This endpoint triggers model training in the background:
    - KNN (with normalized data)
    - SVM (with normalized data)
    - Gradient Boosting (with imputed data)
    - Histogram Gradient Boosting (with imputed data)
    
    Training happens asynchronously. Use /training/status to check progress.
    """
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(
            status_code=409,
            detail="Training already in progress. Check /api/v1/training/status for progress."
        )
    
    # Check if data files exist
    normalized_path = Path("data/processed/ckd_normalized.csv")
    imputed_path = Path("data/processed/ckd_imputed.csv")
    
    if not normalized_path.exists() or not imputed_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Processed data files not found",
                "required_files": [
                    str(normalized_path),
                    str(imputed_path)
                ],
                "solution": "Run data processing pipeline first using core.data_processing"
            }
        )
    
    # Determine models to train
    all_models = ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"]
    models_to_train = config.models_to_train if config.models_to_train else all_models
    
    # Reset status
    training_status = {
        "is_training": True,
        "current_model": None,
        "progress": 0,
        "message": "Training started",
        "results": {},
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None
    }
    
    # Start background training
    background_tasks.add_task(train_models_background, config)
    
    logger.info(f"Training started for models: {models_to_train}")
    
    return TrainingResponse(
        status="started",
        message="Training started in background. Use /api/v1/training/status to check progress.",
        started_at=training_status["started_at"],
        models_to_train=models_to_train
    )


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """
    Get current training status
    
    Returns information about ongoing or completed training:
    - Whether training is in progress
    - Current model being trained
    - Overall progress percentage
    - Results for completed models
    """
    return TrainingStatusResponse(**training_status)


@router.post("/train-single/{model_name}")
async def train_single_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    test_size: float = Query(0.2, ge=0.1, le=0.5),
    random_state: int = Query(42)
):
    """
    Train a single model
    
    - **model_name**: One of KNN, SVM, GradientBoosting, HistGradientBoosting
    - **test_size**: Test set proportion (default: 0.2)
    - **random_state**: Random seed (default: 42)
    """
    valid_models = ["KNN", "SVM", "GradientBoosting", "HistGradientBoosting"]
    
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Must be one of: {valid_models}"
        )
    
    if training_status["is_training"]:
        raise HTTPException(
            status_code=409,
            detail="Training already in progress"
        )
    
    config = TrainingConfig(
        models_to_train=[model_name],
        test_size=test_size,
        random_state=random_state
    )
    
    # Start background training
    background_tasks.add_task(train_models_background, config)
    
    return {
        "status": "started",
        "message": f"Training {model_name} in background",
        "model": model_name
    }


@router.delete("/cancel")
async def cancel_training():
    """
    Attempt to cancel ongoing training
    
    Note: This sets a flag but may not immediately stop the training process.
    """
    global training_status
    
    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    training_status["is_training"] = False
    training_status["message"] = "Training cancellation requested"
    training_status["error"] = "Cancelled by user"
    
    return {
        "status": "cancelled",
        "message": "Training cancellation requested. May take a moment to stop."
    }


@router.get("/results")
async def get_training_results():
    """
    Get results from the last completed training session
    """
    if not training_status["results"]:
        raise HTTPException(
            status_code=404,
            detail="No training results available. Train models first."
        )
    
    return {
        "results": training_status["results"],
        "completed_at": training_status["completed_at"],
        "started_at": training_status["started_at"]
    }