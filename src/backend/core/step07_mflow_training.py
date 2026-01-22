"""
Model Training Module with MLflow Integration
Adapted from your existing training code
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import joblib
from typing import Dict, Tuple
import time

# MLflow imports
import mlflow
import mlflow.sklearn

# Import MLflow config
try:
    from config.mlflow_config import MLflowConfig
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.mlflow_config import MLflowConfig

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    f1_score, accuracy_score, classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization Config
plt.style.use('ggplot')
sns.set_context('notebook')


class ModelTrainerMLflow:
    """Handles model training and evaluation with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "CKD_Detection"):
        """Initialize ModelTrainer with MLflow"""
        self.models_dir = Path('models')
        self.figures_dir = Path('figures/models')
        self.reports_dir = Path('reports/models')
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        
        # Initialize MLflow
        try:
            self.mlflow_config = MLflowConfig(experiment_name=experiment_name)
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            logger.info(f"MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow initialization warning: {str(e)}")
            self.mlflow_config = None
    
    def load_data(self, data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Load data and split into train/test sets"""
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        y = df['status']
        X = df.drop(['status'], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_knn(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train KNN model with MLflow tracking"""
        logger.info("Training KNN model with MLflow...")
        
        start_time = time.time()
        
        with mlflow.start_run(run_name="KNN_Training") as run:
            # Set tags
            mlflow.set_tag("model_type", "KNN")
            mlflow.set_tag("algorithm", "K-Nearest Neighbors")
            mlflow.set_tag("dataset", "normalized")
            
            # Log dataset info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Define model and parameter grid
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 31, 2)}
            
            # Log search configuration
            mlflow.log_param("cv_folds", 10)
            mlflow.log_param("scoring_metric", "f1")
            
            # GridSearch with cross-validation
            grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Best parameters
            best_k = grid_search.best_params_['n_neighbors']
            best_score = grid_search.best_score_
            
            logger.info(f"Best k: {best_k}, CV F1-Score: {best_score:.4f}")
            
            # Log hyperparameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cv_f1_score", best_score)
            
            # Train final model
            knn_final = KNeighborsClassifier(n_neighbors=best_k)
            knn_final.fit(X_train, y_train)
            
            # Predictions
            y_pred = knn_final.predict(X_test)
            
            # Calculate metrics
            results = self._calculate_metrics(y_test, y_pred, "KNN")
            
            # Log all metrics
            mlflow.log_metrics({
                'test_accuracy': results['accuracy'],
                'test_precision': results['precision'],
                'test_recall': results['recall'],
                'test_f1_score': results['f1_score']
            })
            
            # Log confusion matrix values
            cm = results['confusion_matrix']
            mlflow.log_metrics({
                'cm_tn': int(cm[0][0]),
                'cm_fp': int(cm[0][1]),
                'cm_fn': int(cm[1][0]),
                'cm_tp': int(cm[1][1])
            })
            
            # Training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(
                knn_final,
                "model",
                registered_model_name="KNN_CKD_Detector"
            )
            
            # Save model locally
            model_path = self.models_dir / 'knn_model.pkl'
            joblib.dump(knn_final, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Create and log visualizations
            self._plot_knn_optimization(np.arange(1, 31, 2), grid_search.cv_results_['mean_test_score'], best_k)
            self._plot_confusion_matrix(y_test, y_pred, "KNN")
            
            # Log artifacts
            opt_path = self.figures_dir / 'knn_optimization.png'
            cm_path = self.figures_dir / 'knn_confusion_matrix.png'
            
            if opt_path.exists():
                mlflow.log_artifact(str(opt_path))
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            
            # Store additional info
            results['best_params'] = {'n_neighbors': best_k}
            results['cv_score'] = best_score
            results['mlflow_run_id'] = run.info.run_id
            results['training_time'] = training_time
            
            logger.info(f"KNN training completed. MLflow Run ID: {run.info.run_id}")
        
        return results
    
    def train_svm(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train SVM model with MLflow tracking"""
        logger.info("Training SVM model with MLflow...")
        
        start_time = time.time()
        
        with mlflow.start_run(run_name="SVM_Training") as run:
            # Set tags
            mlflow.set_tag("model_type", "SVM")
            mlflow.set_tag("algorithm", "Support Vector Machine")
            mlflow.set_tag("dataset", "normalized")
            
            # Log dataset info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'sigmoid']
            }
            
            # Log search configuration
            mlflow.log_param("cv_folds", 10)
            mlflow.log_param("scoring_metric", "f1")
            
            # GridSearch
            grid_search = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=10,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Best model and parameters
            best_svm = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            
            # Log hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_f1_score", grid_search.best_score_)
            
            # Predictions
            y_pred = best_svm.predict(X_test)
            
            # Calculate metrics
            results = self._calculate_metrics(y_test, y_pred, "SVM")
            
            # Log metrics
            mlflow.log_metrics({
                'test_accuracy': results['accuracy'],
                'test_precision': results['precision'],
                'test_recall': results['recall'],
                'test_f1_score': results['f1_score']
            })
            
            # Training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log model
            mlflow.sklearn.log_model(
                best_svm,
                "model",
                registered_model_name="SVM_CKD_Detector"
            )
            
            # Save locally
            model_path = self.models_dir / 'svm_model.pkl'
            joblib.dump(best_svm, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Create and log visualization
            self._plot_confusion_matrix(y_test, y_pred, "SVM")
            cm_path = self.figures_dir / 'svm_confusion_matrix.png'
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            
            # Store results
            results['best_params'] = best_params
            results['cv_score'] = grid_search.best_score_
            results['mlflow_run_id'] = run.info.run_id
            results['training_time'] = training_time
            
            logger.info(f"SVM training completed. MLflow Run ID: {run.info.run_id}")
        
        return results
    
    def train_gradient_boosting_imputed(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train Gradient Boosting with imputed data and MLflow tracking"""
        logger.info("Training Gradient Boosting (Imputed Data) with MLflow...")
        
        start_time = time.time()
        
        with mlflow.start_run(run_name="GradientBoosting_Imputed") as run:
            # Set tags
            mlflow.set_tag("model_type", "GradientBoosting")
            mlflow.set_tag("algorithm", "Gradient Boosting Classifier")
            mlflow.set_tag("dataset", "imputed")
            
            # Log dataset info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Parameter grid
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0]
            }
            
            # Log search configuration
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("scoring_metric", "f1")
            
            # GridSearch
            grid_search = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=5),
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            
            # Log hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_f1_score", grid_search.best_score_)
            
            # Predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            results = self._calculate_metrics(y_test, y_pred, "GradientBoosting_Imputed")
            
            # Log metrics
            mlflow.log_metrics({
                'test_accuracy': results['accuracy'],
                'test_precision': results['precision'],
                'test_recall': results['recall'],
                'test_f1_score': results['f1_score']
            })
            
            # Training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log model
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name="GradientBoosting_CKD_Detector"
            )
            
            # Save locally
            model_path = self.models_dir / 'gb_imputed_model.pkl'
            joblib.dump(best_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Create and log visualization
            self._plot_confusion_matrix(y_test, y_pred, "GradientBoosting_Imputed")
            cm_path = self.figures_dir / 'gradientboosting_imputed_confusion_matrix.png'
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            
            # Store results
            results['best_params'] = best_params
            results['cv_score'] = grid_search.best_score_
            results['mlflow_run_id'] = run.info.run_id
            results['training_time'] = training_time
            
            logger.info(f"Gradient Boosting training completed. MLflow Run ID: {run.info.run_id}")
        
        return results
    
    def train_hist_gradient_boosting(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train Histogram Gradient Boosting with MLflow tracking"""
        logger.info("Training Histogram Gradient Boosting with MLflow...")
        
        start_time = time.time()
        
        with mlflow.start_run(run_name="HistGradientBoosting") as run:
            # Set tags
            mlflow.set_tag("model_type", "HistGradientBoosting")
            mlflow.set_tag("algorithm", "Histogram Gradient Boosting")
            mlflow.set_tag("dataset", "imputed")
            
            # Log dataset info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Parameter grid
            param_grid = {
                'max_iter': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, None],
                'l2_regularization': [0, 0.1, 1.0],
                'max_leaf_nodes': [15, 31]
            }
            
            # Log search configuration
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("scoring_metric", "f1")
            
            # GridSearch
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                HistGradientBoostingClassifier(random_state=42),
                param_grid=param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            
            # Log hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_f1_score", grid_search.best_score_)
            
            # Predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            results = self._calculate_metrics(y_test, y_pred, "HistGradientBoosting")
            
            # Log metrics
            mlflow.log_metrics({
                'test_accuracy': results['accuracy'],
                'test_precision': results['precision'],
                'test_recall': results['recall'],
                'test_f1_score': results['f1_score']
            })
            
            # Training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log model
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name="HistGradientBoosting_CKD_Detector"
            )
            
            # Save locally
            model_path = self.models_dir / 'hist_gb_model.pkl'
            joblib.dump(best_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Create and log visualization
            self._plot_confusion_matrix(y_test, y_pred, "HistGradientBoosting")
            cm_path = self.figures_dir / 'histgradientboosting_confusion_matrix.png'
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            
            # Store results
            results['best_params'] = best_params
            results['cv_score'] = grid_search.best_score_
            results['mlflow_run_id'] = run.info.run_id
            results['training_time'] = training_time
            
            logger.info(f"Histogram Gradient Boosting training completed. MLflow Run ID: {run.info.run_id}")
        
        return results
    
    def _calculate_metrics(self, y_test, y_pred, model_name: str) -> Dict:
        """Calculate and log model metrics"""
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'model': model_name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def _plot_knn_optimization(self, k_values, scores, best_k):
        """Plot KNN optimization results"""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, scores, marker='o', linestyle='--', color='darkblue')
        plt.title('KNN Optimization: Determining Best k via GridSearchCV')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Mean CV F1-Score')
        plt.xticks(k_values)
        plt.grid(True, alpha=0.3)
        plt.axvline(best_k, color='red', linestyle=':', label=f'Best k = {best_k}')
        plt.legend()
        plt.tight_layout()
        
        fig_path = self.figures_dir / 'knn_optimization.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_test, y_pred, model_name: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['notckd', 'ckd'], 
            yticklabels=['notckd', 'ckd']
        )
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        fig_path = self.figures_dir / f'{model_name.lower()}_confusion_matrix.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all model results to JSON"""
        results_path = self.reports_dir / 'model_comparison.json'
        
        summary = {
            'models': self.results,
            'best_model': max(self.results.items(), key=lambda x: x[1]['f1_score'])[0],
            'ranking': sorted(
                [(name, res['f1_score']) for name, res in self.results.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        logger.info(f"Model comparison saved to {results_path}")
        
        return summary


def main():
    """Main execution function for direct script execution"""
    logger.info("Starting model training pipeline with MLflow...")
    
    trainer = ModelTrainerMLflow(experiment_name="CKD_Detection")
    
    print("\n" + "="*60)
    print("MLFLOW TRACKING")
    print("="*60)
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: CKD_Detection")
    print("\n💡 To view results, run: mlflow ui")
    print("   Then open: http://localhost:5050")
    
    # Train models with normalized data
    print("\n" + "="*60)
    print("TRAINING MODELS WITH NORMALIZED DATA")
    print("="*60)
    
    normalized_path = 'data/processed/ckd_normalized.csv'
    if not Path(normalized_path).exists():
        print(f"❌ Error: {normalized_path} not found!")
        print("Please run data processing first.")
        return
    
    X_train, X_test, y_train, y_test = trainer.load_data(normalized_path)
    
    print("\n[1/4] Training KNN...")
    knn_results = trainer.train_knn(X_train, X_test, y_train, y_test)
    trainer.results['KNN'] = knn_results
    print(f"✅ KNN - F1: {knn_results['f1_score']:.4f} | Run ID: {knn_results['mlflow_run_id']}")
    
    print("\n[2/4] Training SVM...")
    svm_results = trainer.train_svm(X_train, X_test, y_train, y_test)
    trainer.results['SVM'] = svm_results
    print(f"✅ SVM - F1: {svm_results['f1_score']:.4f} | Run ID: {svm_results['mlflow_run_id']}")
    
    # Train models with imputed data
    print("\n" + "="*60)
    print("TRAINING MODELS WITH IMPUTED DATA")
    print("="*60)
    
    imputed_path = 'data/processed/ckd_imputed.csv'
    if not Path(imputed_path).exists():
        print(f"❌ Error: {imputed_path} not found!")
        print("Please run data processing first.")
        return
    
    X_train, X_test, y_train, y_test = trainer.load_data(imputed_path)
    
    print("\n[3/4] Training Gradient Boosting...")
    gb_results = trainer.train_gradient_boosting_imputed(X_train, X_test, y_train, y_test)
    trainer.results['GradientBoosting'] = gb_results
    print(f"✅ GB - F1: {gb_results['f1_score']:.4f} | Run ID: {gb_results['mlflow_run_id']}")
    
    print("\n[4/4] Training Histogram Gradient Boosting...")
    hist_gb_results = trainer.train_hist_gradient_boosting(X_train, X_test, y_train, y_test)
    trainer.results['HistGradientBoosting'] = hist_gb_results
    print(f"✅ Hist GB - F1: {hist_gb_results['f1_score']:.4f} | Run ID: {hist_gb_results['mlflow_run_id']}")
    
    # Save and display results
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    summary = trainer.save_results()
    
    print("\nRankings (by F1-Score):")
    for i, (model, score) in enumerate(summary['ranking'], 1):
        print(f"{i}. {model}: {score:.4f}")
    
    print(f"\n🏆 Best Model: {summary['best_model']}")
    print(f"\n📊 View all experiments in MLflow UI:")
    print(f"   mlflow ui --port 5050")
    print(f"   http://localhost:5050")


if __name__ == "__main__":
    main()