"""
Model Training Module for CKD Detection Project
Handles training of KNN, SVM, and Gradient Boosting models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import joblib
from typing import Dict, Tuple

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


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        """Initialize ModelTrainer"""
        self.models_dir = Path('models')
        self.figures_dir = Path('figures/models')
        self.reports_dir = Path('reports/models')
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
    
    def load_data(self,
                    data_path: str,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple:
        """
        Load data and split into train/test sets
        
        Args:
            data_path: Path to processed data
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Separate features and target
        # y = df['status']
        # X = df.drop(['status'], axis=1)
        df = pd.read_csv(data_path)
        df['status'] = df['status'].map({'ckd': 1, 'notckd': 0})
        df = df[['hemo', 'sg', 'sc', 'rbcc', 'pcv', 
                    'htn', 'dm', 'bp', 'age', 'status']]

        y = df['status'] #as the df comes from a csv, y is treated as str. We convert it to simplify future calculations
        X = df.drop(['status'], axis=1) # drop all varribales that are not related to the analysis 

        # Split data
        """Split into Training (80%) and Test (20%)
        'stratify' ensures both sets have the same percentage of CKD vs NotCKD"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_knn(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train KNN model with GridSearch
        
        Returns:
            Dictionary with model results
        """
        logger.info("Training KNN model...")
        
        # Define model and parameter grid
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 31, 2)}
        
        # GridSearch with cross-validation
        grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        best_k = grid_search.best_params_['n_neighbors']
        best_score = grid_search.best_score_
        
        logger.info(f"Best k: {best_k}, CV F1-Score: {best_score:.4f}")
        
        # Train final model
        knn_final = KNeighborsClassifier(n_neighbors=best_k)
        knn_final.fit(X_train, y_train)
        
        # Predictions
        y_pred = knn_final.predict(X_test)
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, "KNN")
        results['best_params'] = {'n_neighbors': best_k}
        results['cv_score'] = best_score
        results['grid_results'] = grid_search.cv_results_['mean_test_score']
        
        # Save model
        model_path = self.models_dir / 'knn_model.pkl'
        joblib.dump(knn_final, model_path)
        logger.info(f"KNN model saved to {model_path}")
        
        # Visualizations
        self._plot_knn_optimization(np.arange(1, 31, 2), grid_search.cv_results_['mean_test_score'], best_k)
        self._plot_confusion_matrix(y_test, y_pred, "KNN")
        
        return results
    
    def train_svm(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train SVM model with GridSearch
        
        Returns:
            Dictionary with model results
        """
        logger.info("Training SVM model...")
        
        # Parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'sigmoid']
        }
        
        # GridSearch
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=10,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_svm = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        
        # Predictions
        y_pred = best_svm.predict(X_test)
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, "SVM")
        results['best_params'] = best_params
        results['cv_score'] = grid_search.best_score_
        
        # Save model
        model_path = self.models_dir / 'svm_model.pkl'
        joblib.dump(best_svm, model_path)
        logger.info(f"SVM model saved to {model_path}")
        
        # Visualization
        self._plot_confusion_matrix(y_test, y_pred, "SVM")
        
        return results
    
    def train_gradient_boosting_imputed(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train Gradient Boosting with imputed data
        
        Returns:
            Dictionary with model results
        """
        logger.info("Training Gradient Boosting (Imputed Data)...")
        
        # Parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0]
        }
        
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
        
        # Get feature importances
        feature_importances = best_model.feature_importances_
        logger.info(f"Feature importances: {feature_importances}")

        # Get feature names
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]

        # Plot feature importances
        self._plot_feature_importances(feature_importances, feature_names, "Gradient Boosting")
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, "GradientBoosting_Imputed")
        results['best_params'] = best_params
        results['cv_score'] = grid_search.best_score_
        
        # Save model
        model_path = self.models_dir / 'gb_imputed_model.pkl'
        joblib.dump(best_model, model_path)
        logger.info(f"Gradient Boosting (Imputed) model saved to {model_path}")
        
        # Visualization
        self._plot_confusion_matrix(y_test, y_pred, "GradientBoosting_Imputed")
        
        return results
    
    def train_hist_gradient_boosting(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train Histogram Gradient Boosting (handles missing values natively)
        
        Returns:
            Dictionary with model results
        """
        logger.info("Training Histogram Gradient Boosting...")
        
        # Parameter grid
        param_grid = {
            'max_iter': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, None],
            'l2_regularization': [0, 0.1, 1.0],
            'max_leaf_nodes': [15, 31]
        }
        
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
        
        # Get feature importances
        feature_importances = best_model.feature_importances_
        logger.info(f"Feature importances: {feature_importances}")

        # Get feature names
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]

        # Plot feature importances
        self._plot_feature_importances(feature_importances, feature_names, "HistGradientBoosting")
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Metrics
        results = self._calculate_metrics(y_test, y_pred, "HistGradientBoosting")
        results['best_params'] = best_params
        results['cv_score'] = grid_search.best_score_
        
        # Save model
        model_path = self.models_dir / 'hist_gb_model.pkl'
        joblib.dump(best_model, model_path)
        logger.info(f"Histogram Gradient Boosting model saved to {model_path}")
        
        # Visualization
        self._plot_confusion_matrix(y_test, y_pred, "HistGradientBoosting")
        
        return results
    
    def _calculate_metrics(self, y_test, y_pred, model_name: str) -> Dict:
        """Calculate and log model metrics"""
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
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
        logger.info(f"KNN optimization plot saved to {fig_path}")
    
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
        logger.info(f"{model_name} confusion matrix saved to {fig_path}")
    
    def _plot_feature_importances(self, feature_importances, feature_names, model_name):
        """Plot feature importances"""      
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=True)  # Ascending for horizontal bar
        
        # Plot
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'{model_name} - Feature Importances')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.figures_dir / f'{model_name.lower()}_feature_importances.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {plot_path}")


    def save_results(self):
        """Save all model results to JSON"""
        results_path = self.reports_dir / 'model_comparison.json'
        
        # Create comparison summary
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


def main(data_path: str) -> None:
    """Main execution function"""
    logger.info("Starting model training pipeline...")
    
    trainer = ModelTrainer()
    
    # Train models with normalized data (KNN, SVM)
    print("\n" + "="*60)
    print("TRAINING MODELS WITH NORMALIZED DATA (KNN, SVM)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = trainer.load_data(data_path)
    
    # KNN
    print("\n[1/4] Training KNN...")
    knn_results = trainer.train_knn(X_train, X_test, y_train, y_test)
    trainer.results['KNN'] = knn_results
    print(f"✅ KNN - F1 Score: {knn_results['f1_score']:.4f}")
    
    # SVM
    print("\n[2/4] Training SVM...")
    svm_results = trainer.train_svm(X_train, X_test, y_train, y_test)
    trainer.results['SVM'] = svm_results
    print(f"✅ SVM - F1 Score: {svm_results['f1_score']:.4f}")
    
    # Train models with imputed data (Gradient Boosting)
    print("\n" + "="*60)
    print("TRAINING MODELS WITH IMPUTED DATA (Gradient Boosting)")
    print("="*60)
    
    X_train, X_test, y_train, y_test = trainer.load_data(data_path)
    
    # Gradient Boosting
    print("\n[3/4] Training Gradient Boosting...")
    gb_results = trainer.train_gradient_boosting_imputed(X_train, X_test, y_train, y_test)
    trainer.results['GradientBoosting_Imputed'] = gb_results
    print(f"✅ Gradient Boosting - F1 Score: {gb_results['f1_score']:.4f}")
    
    # Histogram Gradient Boosting
    print("\n[4/4] Training Histogram Gradient Boosting...")
    hist_gb_results = trainer.train_hist_gradient_boosting(X_train, X_test, y_train, y_test)
    trainer.results['HistGradientBoosting'] = hist_gb_results
    print(f"✅ Histogram Gradient Boosting - F1 Score: {hist_gb_results['f1_score']:.4f}")
    
    # Save and display results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    summary = trainer.save_results()
    
    print("\nModel Rankings (by F1-Score):")
    for i, (model, score) in enumerate(summary['ranking'], 1):
        print(f"{i}. {model}: {score:.4f}")
    
    print(f"\n🏆 Best Model: {summary['best_model']}")
    print(f"✅ All models saved to: {trainer.models_dir}")
    print(f"✅ Figures saved to: {trainer.figures_dir}")
    print(f"✅ Reports saved to: {trainer.reports_dir}")


if __name__ == "__main__":
    main()