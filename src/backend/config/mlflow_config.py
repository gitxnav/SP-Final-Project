import mlflow
import os

class MLflowConfig:
    def __init__(self, experiment_name="CKD_Detection"):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
    
    def get_tracking_uri(self):
        return self.tracking_uri