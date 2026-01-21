"""
Feature Engineering Module for CKD Detection Project
Handles feature selection, normalization, and imputation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering tasks"""
    
    def __init__(self, data_path: str = 'data/processed/ckd_imputed.csv'):
        """
        Initialize FeatureEngineer
        
        Args:
            data_path: Path to cleaned data CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.selected_features = ['hemo', 'sg', 'sc', 'rbcc', 'pcv', 'htn', 'dm', 'bp', 'age']
        self.target = 'status'
        
        # Output directories
        self.processed_dir = Path('data/samples')
        self.reports_dir = Path('reports')
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load cleaned data"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.data_path}")
            raise
    
    def encode_target(self) -> pd.DataFrame:
        """
        Encode target variable: ckd -> 1, notckd -> 0
        
        Returns:
            DataFrame with encoded target
        """
        logger.info("Encoding target variable...")
        
        if self.target in self.df.columns:
            # Map target values
            self.df[self.target] = self.df[self.target].map({'ckd': 1, 'notckd': 0})
            
            # Check for any unmapped values
            null_count = self.df[self.target].isnull().sum()
            if null_count > 0:
                logger.warning(f"{null_count} unmapped target values found!")
            
            logger.info(f"Target distribution: {self.df[self.target].value_counts().to_dict()}")
        else:
            logger.error(f"Target column '{self.target}' not found!")
            raise KeyError(f"Target column '{self.target}' not found!")
        
        return self.df
    
    def select_features(self) -> pd.DataFrame:
        """
        Select relevant features for modeling
        
        Returns:
            DataFrame with selected features + target
        """
        logger.info("Selecting features...")
        
        # Check if all selected features exist
        missing_features = [f for f in self.selected_features if f not in self.df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Remove missing features from selection
            self.selected_features = [f for f in self.selected_features if f in self.df.columns]
        
        # Select features + target
        columns_to_keep = self.selected_features + [self.target]
        self.df = self.df[columns_to_keep]
        
        logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        
        return self.df
    
    def handle_missing_values_imputation(self) -> pd.DataFrame:
        """
        Impute missing values using median strategy
        
        Returns:
            DataFrame with imputed values
        """
        logger.info("Imputing missing values...")
        
        # Get initial missing count
        initial_missing = self.df.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        # Impute only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
            logger.info(f"Imputed {len(numeric_cols)} numeric columns")
        
        # Reconstruct dataframe
        self.df = pd.concat([X, y], axis=1)
        
        # Check final missing count
        final_missing = self.df.isnull().sum().sum()
        logger.info(f"Final missing values: {final_missing}")
        logger.info(f"Imputed {initial_missing - final_missing} values")
        
        return self.df
    
    def normalize_features(self) -> pd.DataFrame:
        """
        Normalize features using StandardScaler
        
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        
        # Normalize only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            logger.info(f"Normalized {len(numeric_cols)} numeric columns")
            
            # Log normalization statistics
            logger.info(f"Feature means: {self.scaler.mean_}")
            logger.info(f"Feature stds: {self.scaler.scale_}")
        
        # Reconstruct dataframe
        self.df = pd.concat([X, y], axis=1)
        
        return self.df
    
    def get_feature_statistics(self) -> dict:
        """
        Get statistics about features
        
        Returns:
            Dictionary with feature statistics
        """
        logger.info("Generating feature statistics...")
        
        stats = {
            'total_features': len(self.selected_features),
            'numeric_features': len(self.df.select_dtypes(include=[np.number]).columns) - 1,  # Exclude target
            'categorical_features': len(self.df.select_dtypes(include=['object']).columns),
            'missing_values': self.df.isnull().sum().sum(),
            'feature_list': self.selected_features,
            'data_shape': self.df.shape,
            'target_distribution': self.df[self.target].value_counts().to_dict()
        }
        
        # Save statistics
        stats_path = self.reports_dir / 'feature_engineering_stats.json'
        with open(stats_path, 'w') as f:
            # Convert non-serializable objects to strings
            stats_serializable = {
                k: (v if isinstance(v, (int, float, str, list, dict)) else str(v))
                for k, v in stats.items()
            }
            json.dump(stats_serializable, f, indent=4)
        
        logger.info(f"Feature statistics saved to {stats_path}")
        
        return stats
    
    def save_imputed_data(self, output_path: str = 'data/samples/ckd_imputed.csv'):
        """
        Save data with imputed values (not normalized)
        
        Args:
            output_path: Path to save imputed data
        """
        output_path = Path(output_path)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Imputed data saved to {output_path}")
    
    def save_normalized_data(self, output_path: str = 'data/samples/ckd_normalized.csv'):
        """
        Save normalized data
        
        Args:
            output_path: Path to save normalized data
        """
        output_path = Path(output_path)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Normalized data saved to {output_path}")
    
    def process_pipeline_imputed(self) -> pd.DataFrame:
        """
        Run complete pipeline for imputed data (no normalization)
        
        Returns:
            Processed DataFrame
        """
        logger.info("Starting imputed data pipeline...")
        
        self.load_data()
        self.encode_target()
        self.select_features()
        self.handle_missing_values_imputation()
        
        # Save imputed data
        self.save_imputed_data()
        
        # Get statistics
        stats = self.get_feature_statistics()
        
        logger.info("Imputed data pipeline completed!")
        
        return self.df
    
    def process_pipeline_normalized(self) -> pd.DataFrame:
        """
        Run complete pipeline for normalized data
        
        Returns:
            Processed DataFrame
        """
        logger.info("Starting normalized data pipeline...")
        
        self.load_data()
        self.encode_target()
        self.select_features()
        self.handle_missing_values_imputation()
        self.normalize_features()
        
        # Save normalized data
        self.save_normalized_data()
        
        # Get statistics
        stats = self.get_feature_statistics()
        
        logger.info("Normalized data pipeline completed!")
        
        return self.df


def main():
    """Main execution function"""
    logger.info("Starting feature engineering process...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Process imputed data (for Gradient Boosting)
    print("\n" + "="*50)
    print("PROCESSING IMPUTED DATA")
    print("="*50)
    df_imputed = engineer.process_pipeline_imputed()
    print(f"\nImputed Data Shape: {df_imputed.shape}")
    print("\nFirst 5 rows:")
    print(df_imputed.head())
    
    # Reset for normalized data
    engineer = FeatureEngineer()
    
    # Process normalized data (for KNN, SVM)
    print("\n" + "="*50)
    print("PROCESSING NORMALIZED DATA")
    print("="*50)
    df_normalized = engineer.process_pipeline_normalized()
    print(f"\nNormalized Data Shape: {df_normalized.shape}")
    print("\nFirst 5 rows:")
    print(df_normalized.head())
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETED")
    print("="*50)
    print(f"✅ Imputed data: data/processed/ckd_imputed.csv")
    print(f"✅ Normalized data: data/processed/ckd_normalized.csv")
    print(f"✅ Statistics: reports/feature_engineering_stats.json")


if __name__ == "__main__":
    main()