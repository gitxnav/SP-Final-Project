"""
Model Prediction Module for CKD Detection Project
Handles loading trained models and making predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from typing import Dict, Union
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and predictions"""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize ModelPredictor
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.available_models = {
            'KNN': 'knn_model.pkl',
            'SVM': 'svm_model.pkl',
            'GradientBoosting': 'gb_imputed_model.pkl',
            'HistGradientBoosting': 'hist_gb_model.pkl'
        }
        
        # Feature names expected by models
        self.feature_names = ['hemo', 'sg', 'sc', 'rbcc', 'pcv', 'htn', 'dm', 'bp', 'age']
    
    def load_model(self, model_name: str):
        """
        Load a trained model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.available_models.keys())}")
        
        model_path = self.models_dir / self.available_models[model_name]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            logger.info(f"Model '{model_name}' loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
            raise
    
    def load_all_models(self):
        """Load all available models"""
        logger.info("Loading all available models...")
        
        for model_name in self.available_models.keys():
            try:
                self.load_model(model_name)
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {str(e)}")
        
        logger.info(f"Loaded {len(self.models)} models successfully")
    
    def predict_single(self, model_name: str, features: Dict) -> Dict:
        """
        Make prediction for single patient
        
        Args:
            model_name: Name of model to use
            features: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        # Load model if not already loaded
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Prepare input data
        X = pd.DataFrame([features])[self.feature_names]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            probability = {
                'notckd': float(proba[0]),
                'ckd': float(proba[1])
            }
        
        result = {
            'prediction': 'ckd' if prediction == 1 else 'notckd',
            'prediction_numeric': int(prediction),
            'probability': probability,
            'model_used': model_name,
            'features_used': features
        }
        
        logger.info(f"Prediction: {result['prediction']} (Model: {model_name})")
        
        return result
    
    def predict_batch(self, model_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple patients
        
        Args:
            model_name: Name of model to use
            data: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        # Load model if not already loaded
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Ensure correct features
        X = data[self.feature_names]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            data['probability_notckd'] = probabilities[:, 0]
            data['probability_ckd'] = probabilities[:, 1]
        
        # Add predictions
        data['prediction_numeric'] = predictions
        data['prediction'] = ['ckd' if p == 1 else 'notckd' for p in predictions]
        
        logger.info(f"Made {len(data)} predictions using {model_name}")
        
        return data
    
    def predict_with_all_models(self, features: Dict) -> Dict:
        """
        Make predictions using all available models
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with predictions from all models
        """
        # Load all models if not loaded
        if not self.models:
            self.load_all_models()
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.predict_single(model_name, features)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Consensus prediction
        predictions = [r['prediction'] for r in results.values() if 'prediction' in r]
        if predictions:
            consensus = max(set(predictions), key=predictions.count)
            consensus_confidence = predictions.count(consensus) / len(predictions)
            
            results['consensus'] = {
                'prediction': consensus,
                'confidence': consensus_confidence,
                'agreement': f"{predictions.count(consensus)}/{len(predictions)} models"
            }
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a loaded model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        
        info = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'features_required': self.feature_names,
            'has_probability': hasattr(model, 'predict_proba')
        }
        
        # Get model-specific parameters
        if hasattr(model, 'get_params'):
            info['parameters'] = model.get_params()
        
        return info


def main():
    """Main execution function for demonstration"""
    logger.info("Starting prediction demonstration...")
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Load all models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    predictor.load_all_models()
    
    # Example patient data
    example_patient = {
        'hemo': 15.4,
        'sg': 1.020,
        'sc': 1.2,
        'rbcc': 5.2,
        'pcv': 44.0,
        'htn': 1.0,
        'dm': 1.0,
        'bp': 80.0,
        'age': 48.0
    }
    
    print("\n" + "="*60)
    print("EXAMPLE PATIENT DATA")
    print("="*60)
    print(json.dumps(example_patient, indent=2))
    
    # Single model prediction
    print("\n" + "="*60)
    print("SINGLE MODEL PREDICTION (KNN)")
    print("="*60)
    result = predictor.predict_single('KNN', example_patient)
    print(json.dumps(result, indent=2, default=str))
    
    # All models prediction
    print("\n" + "="*60)
    print("ALL MODELS PREDICTION")
    print("="*60)
    all_results = predictor.predict_with_all_models(example_patient)
    
    for model_name, result in all_results.items():
        if model_name != 'consensus':
            if 'prediction' in result:
                print(f"\n{model_name}:")
                print(f"  Prediction: {result['prediction']}")
                if result['probability']:
                    print(f"  Confidence: {result['probability'][result['prediction']]:.2%}")
    
    if 'consensus' in all_results:
        print(f"\n{'='*60}")
        print("CONSENSUS PREDICTION")
        print("="*60)
        print(f"Prediction: {all_results['consensus']['prediction']}")
        print(f"Agreement: {all_results['consensus']['agreement']}")
        print(f"Confidence: {all_results['consensus']['confidence']:.2%}")
    
    # Model info
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    for model_name in predictor.models.keys():
        info = predictor.get_model_info(model_name)
        print(f"\n{model_name}:")
        print(f"  Type: {info['model_type']}")
        print(f"  Has Probability: {info['has_probability']}")


if __name__ == "__main__":
    main()