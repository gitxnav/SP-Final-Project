from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

class CKDModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    
    def train(self, X, y):
        """Train model with MLflow tracking"""
        
        with mlflow.start_run(run_name="training"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": 100,
                "max_depth": 5,
                "test_size": 0.2
            })
            
            # Train
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name="ckd_model"
            )
            
            return metrics