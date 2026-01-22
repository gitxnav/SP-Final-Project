"""
MLflow Configuration Module for CKD Detection Project
Centralized MLflow setup and utilities
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import logging
from typing import Optional, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowConfig:
    """Central MLflow configuration and utilities"""
    
    def __init__(self, 
                 experiment_name: str = "CKD_Detection",
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow configuration
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: Custom tracking URI (optional)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            self.tracking_uri = Path(tracking_uri).resolve()
        else:
            self.tracking_uri = Path("mlruns").resolve()
        
        self.setup()
    
    def setup(self):
        """Setup MLflow tracking and experiment"""
        try:
            # Create mlruns directory if it doesn't exist
            self.tracking_uri.mkdir(parents=True, exist_ok=True)
            
            # Set tracking URI
            mlflow.set_tracking_uri(f"file://{self.tracking_uri}")
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=str(self.tracking_uri / "artifacts")
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            # Set active experiment
            mlflow.set_experiment(self.experiment_name)
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    @staticmethod
    def start_run(run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """
        Start MLflow run with optional name and tags
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add to run
        
        Returns:
            MLflow run object
        """
        run = mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    @staticmethod
    def end_run():
        """End active MLflow run"""
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {run_id}")
    
    @staticmethod
    def log_params(params: Dict):
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")
    
    @staticmethod
    def log_param(key: str, value):
        """
        Log single parameter
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    @staticmethod
    def log_metrics(metrics: Dict):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics
        """
        mlflow.log_metrics(metrics)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    @staticmethod
    def log_metric(key: str, value: float):
        """
        Log single metric
        
        Args:
            key: Metric name
            value: Metric value
        """
        mlflow.log_metric(key, value)
    
    @staticmethod
    def log_model(model, artifact_path: str, registered_model_name: Optional[str] = None):
        """
        Log sklearn model to MLflow
        
        Args:
            model: Trained sklearn model
            artifact_path: Path within run's artifact directory
            registered_model_name: Name for model registry (optional)
        """
        if registered_model_name:
            mlflow.sklearn.log_model(
                model, 
                artifact_path,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model to registry: {registered_model_name}")
        else:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Logged model to artifact path: {artifact_path}")
    
    @staticmethod
    def log_artifact(file_path: str):
        """
        Log artifact file to MLflow
        
        Args:
            file_path: Path to artifact file
        """
        if Path(file_path).exists():
            mlflow.log_artifact(file_path)
            logger.debug(f"Logged artifact: {file_path}")
        else:
            logger.warning(f"Artifact file not found: {file_path}")
    
    @staticmethod
    def log_artifacts(dir_path: str):
        """
        Log directory of artifacts
        
        Args:
            dir_path: Directory path containing artifacts
        """
        if Path(dir_path).exists():
            mlflow.log_artifacts(dir_path)
            logger.debug(f"Logged artifacts from: {dir_path}")
        else:
            logger.warning(f"Artifacts directory not found: {dir_path}")
    
    @staticmethod
    def set_tags(tags: Dict):
        """
        Set tags for current run
        
        Args:
            tags: Dictionary of tags
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    @staticmethod
    def get_run_id() -> Optional[str]:
        """
        Get current run ID
        
        Returns:
            Run ID string or None
        """
        if mlflow.active_run():
            return mlflow.active_run().info.run_id
        return None
    
    @staticmethod
    def enable_autolog(log_input_examples: bool = True, 
                       log_model_signatures: bool = True):
        """
        Enable autologging for sklearn
        
        Args:
            log_input_examples: Whether to log input examples
            log_model_signatures: Whether to log model signatures
        """
        mlflow.sklearn.autolog(
            log_input_examples=log_input_examples,
            log_model_signatures=log_model_signatures,
            log_models=True
        )
        logger.info("MLflow autolog enabled for sklearn")
    
    @staticmethod
    def disable_autolog():
        """Disable autologging"""
        mlflow.sklearn.autolog(disable=True)
        logger.info("MLflow autolog disabled")
    
    def get_experiment_info(self) -> Dict:
        """
        Get information about current experiment
        
        Returns:
            Dictionary with experiment information
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment:
            return {
                'experiment_id': experiment.experiment_id,
                'name': experiment.name,
                'artifact_location': experiment.artifact_location,
                'lifecycle_stage': experiment.lifecycle_stage,
                'tracking_uri': str(self.tracking_uri)
            }
        return {}
    
    @staticmethod
    def search_runs(experiment_ids: list = None, 
                   filter_string: str = "", 
                   max_results: int = 100) -> list:
        """
        Search for runs in experiments
        
        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: Filter string for search
            max_results: Maximum number of results
            
        Returns:
            List of run objects
        """
        return mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results
        )


def main():
    """Demonstration of MLflow configuration"""
    logger.info("MLflow Configuration Demo")
    
    # Initialize config
    config = MLflowConfig(experiment_name="CKD_Detection_Demo")
    
    # Get experiment info
    info = config.get_experiment_info()
    print("\n" + "="*60)
    print("EXPERIMENT INFORMATION")
    print("="*60)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Demo run
    print("\n" + "="*60)
    print("DEMO RUN")
    print("="*60)
    
    with mlflow.start_run(run_name="Demo_Run"):
        # Log parameters
        config.log_params({
            'learning_rate': 0.01,
            'n_estimators': 100,
            'max_depth': 5
        })
        
        # Log metrics
        config.log_metrics({
            'accuracy': 0.95,
            'f1_score': 0.93,
            'precision': 0.94
        })
        
        # Add tags
        config.set_tags({
            'model_type': 'demo',
            'version': '1.0'
        })
        
        run_id = config.get_run_id()
        print(f"\n✅ Demo run completed: {run_id}")
        print(f"📊 View in MLflow UI: mlflow ui")
        print(f"🔗 Open: http://localhost:5050")


if __name__ == "__main__":
    main()