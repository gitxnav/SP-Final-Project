"""
Simple data loading module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataLoader:
    """Loads data from text files"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_txt(self, filename, delimiter=","):
        """Load data from text file"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return self._create_sample_data()
        
        return pd.read_csv(filepath, delimiter=delimiter)
    
    def _create_sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples),
            'target': np.random.randint(0, 3, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def save_sample_data(self, filename="sample_data.txt"):
        """Save sample data to file"""
        df = self._create_sample_data()
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def load_train_test_split(self, filename, test_size=0.2, random_state=42):
        """Load data and split into train/test"""
        df = self.load_from_txt(filename)
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['target'] if 'target' in df.columns else None
        )
        
        return train_df, test_df
    
    def validate_data(self, df):
        """Validate DataFrame"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        return True