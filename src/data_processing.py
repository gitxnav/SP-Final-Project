"""
Simple data processing module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataProcessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
    
    def remove_missing_values(self, df, strategy="drop"):
        """Handle missing values"""
        if strategy == "drop":
            return df.dropna()
        elif strategy == "fill":
            df_clean = df.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].median()
            )
            return df_clean
        return df
    
    def remove_outliers(self, df, columns=None, n_std=3.0):
        """Remove outliers using standard deviation"""
        df_clean = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - n_std * std
            upper = mean + n_std * std
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        
        return df_clean
    
    def normalize_features(self, df, method="standard", feature_columns=None):
        """Normalize features"""
        df_normalized = df.copy()
        
        if feature_columns is None:
            feature_columns = [
                col for col in df.columns 
                if col != 'target' and df[col].dtype in [np.float64, np.int64]
            ]
        
        self.feature_columns = feature_columns
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df_normalized
    
    def transform_features(self, df):
        """Transform using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted")
        
        df_transformed = df.copy()
        df_transformed[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df_transformed
    
    def split_features_target(self, df, target_col="target"):
        """Split into features and target"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    
    def get_feature_statistics(self, df):
        """Get feature statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].describe()