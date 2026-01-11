"""
Simple model training module
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
import json


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.metrics = {}
    
    def train_model(self, X_train, y_train, model_params=None):
        """Train a Random Forest classifier"""
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return self.metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def save_model(self, filename="model.pkl"):
        """Save model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = self.model_dir / filename
        joblib.dump(self.model, filepath)
        return filepath
    
    def load_model(self, filename="model.pkl"):
        """Load model"""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        return self.model
    
    def save_metrics(self, filename="metrics.json"):
        """Save metrics"""
        if not self.metrics:
            raise ValueError("No metrics to save")
        
        filepath = self.model_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return filepath
    
    def get_feature_importances(self, feature_names):
        """Get feature importances"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)