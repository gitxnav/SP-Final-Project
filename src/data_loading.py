"""
Data Loading Module for CKD Detection Project
Handles raw data ingestion and initial validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial validation of CKD dataset"""
    
    def __init__(self, data_path: str = '../../data/raw/chronic_kindey_disease.csv'):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to raw CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data with custom null identifiers
        
        Returns:
            DataFrame with loaded data
        """
        try:
            # '?' is defined as the null placeholder in the documentation
            self.df = pd.read_csv(
                self.data_path, 
                na_values='?', 
                skipinitialspace=True
            )
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_schema(self) -> bool:
        """
        Validate data schema and structure
        
        Returns:
            True if schema is valid
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
        
        # Expected dimensions
        expected_rows = 400
        expected_cols = 25
        
        if self.df.shape != (expected_rows, expected_cols):
            logger.warning(
                f"Unexpected dimensions: {self.df.shape} "
                f"(expected: ({expected_rows}, {expected_cols}))"
            )
        
        # Check for target column
        if 'status' not in self.df.columns:
            logger.error("Target column 'status' not found!")
            return False
        
        logger.info("Schema validation passed")
        return True
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
        
        info = {
            'shape': self.df.shape,
            'n_rows': self.df.shape[0],
            'n_cols': self.df.shape[1],
            'numeric_cols': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': self.df.select_dtypes(include=['object']).columns.tolist(),
            'target_distribution': self.df['status'].value_counts().to_dict() if 'status' in self.df.columns else {}
        }
        
        logger.info(f"Dataset contains {info['n_rows']} rows and {info['n_cols']} columns")
        logger.info(f"Numeric features: {len(info['numeric_cols'])}")
        logger.info(f"Categorical features: {len(info['categorical_cols'])}")
        
        return info
    
    def save_loaded_data(self, output_path: str = '../../data/processed/loaded_data.csv'):
        """
        Save loaded data to processed directory
        
        Args:
            output_path: Path to save the loaded data
        """
        if self.df is None:
            logger.error("No data to save. Call load_data() first.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")


def main():
    """Main execution function"""
    logger.info("Starting data loading process...")
    
    # Initialize loader
    loader = DataLoader()
    
    # Load data
    df = loader.load_data()
    
    # Validate schema
    loader.validate_schema()
    
    # Get and display info
    info = loader.get_data_info()
    
    # Display first few rows
    print("\n=== First 5 rows ===")
    print(df.head())
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    # Save loaded data
    loader.save_loaded_data()
    
    logger.info("Data loading process completed successfully!")


if __name__ == "__main__":
    main()