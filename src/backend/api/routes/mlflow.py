"""
MLflow Routes - CKD Detection
MLflow tracking and experiment management endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/experiments")
async def get_experiments():
    """
    Get all MLflow experiments
    
    Returns list of all experiments with their metadata
    """
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags
            })
        
        return {
            "experiments": exp_data,
            "total": len(exp_data),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/runs")
async def get_experiment_runs(
    experiment_id: str,
    max_results: int = Query(10, ge=1, le=100)
):
    """
    Get runs for a specific experiment
    
    - **experiment_id**: ID of the experiment
    - **max_results**: Maximum number of runs to return (1-100)
    """
    try:
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        runs_data = []
        for run in runs:
            runs_data.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time/1000).isoformat(),
                "end_time": datetime.fromtimestamp(run.info.end_time/1000).isoformat() if run.info.end_time else None,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
        
        return {
            "experiment_id": experiment_id,
            "runs": runs_data,
            "count": len(runs_data),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_run_details(run_id: str):
    """
    Get detailed information about a specific run
    
    - **run_id**: ID of the MLflow run
    """
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": datetime.fromtimestamp(run.info.start_time/1000).isoformat(),
            "end_time": datetime.fromtimestamp(run.info.end_time/1000).isoformat() if run.info.end_time else None,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "artifact_uri": run.info.artifact_uri
        }
    
    except Exception as e:
        logger.error(f"Run not found: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@router.get("/runs")
async def get_runs_by_experiment_name(
    experiment_name: str = Query("CKD_Detection"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get recent MLflow runs for an experiment by name
    
    - **experiment_name**: Name of the experiment (default: CKD_Detection)
    - **limit**: Maximum number of runs to return
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_name}' not found"
            )
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        runs_data = []
        for run in runs:
            runs_data.append({
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get('mlflow.runName', 'N/A'),
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time/1000).isoformat(),
                "metrics": run.data.metrics,
                "params": run.data.params
            })
        
        return {
            "experiment_name": experiment_name,
            "runs": runs_data,
            "total": len(runs_data),
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_registered_models():
    """
    Get all registered models from MLflow Model Registry
    """
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        
        models_data = []
        for model in models:
            models_data.append({
                "name": model.name,
                "creation_timestamp": datetime.fromtimestamp(model.creation_timestamp/1000).isoformat(),
                "last_updated_timestamp": datetime.fromtimestamp(model.last_updated_timestamp/1000).isoformat(),
                "description": model.description,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "status": v.status
                    }
                    for v in model.latest_versions
                ]
            })
        
        return {
            "models": models_data,
            "count": len(models_data),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/summary")
async def get_metrics_summary(experiment_name: str = Query("CKD_Detection")):
    """
    Get summary statistics of metrics across all runs in an experiment
    
    - **experiment_name**: Name of the experiment
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_name}' not found"
            )
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1000,
            order_by=["start_time DESC"]
        )
        
        # Aggregate metrics
        all_metrics = {}
        for run in runs:
            for metric_key, metric_value in run.data.metrics.items():
                if metric_key not in all_metrics:
                    all_metrics[metric_key] = []
                all_metrics[metric_key].append(metric_value)
        
        # Calculate statistics
        summary = {}
        for metric_key, values in all_metrics.items():
            if values:
                summary[metric_key] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[0] if values else None
                }
        
        return {
            "experiment_name": experiment_name,
            "total_runs": len(runs),
            "metrics_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))