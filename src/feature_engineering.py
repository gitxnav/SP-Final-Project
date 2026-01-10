"""
Simple feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA


class FeatureEngineer:
    """Handles feature creation and selection"""
    
    def __init__(self):
        self.selector = None
        self.pca = None
        self.selected_features = None
    
    def create_polynomial_features(self, df, columns, degree=2):
        """Create polynomial features"""
        df_poly = df.copy()
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df_poly[f"{col}_pow{d}"] = df_poly[col] ** d
        
        return df_poly
    
    def create_interaction_features(self, df, column_pairs=None):
        """Create interaction features"""
        df_interact = df.copy()
        
        if column_pairs is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'target']
            column_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, min(i + 3, len(numeric_cols))):
                    column_pairs.append((numeric_cols[i], numeric_cols[j]))
        
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                df_interact[f"{col1}_x_{col2}"] = df_interact[col1] * df_interact[col2]
        
        return df_interact
    
    def create_statistical_features(self, df, feature_columns=None):
        """Create statistical features"""
        df_stats = df.copy()
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col != 'target']
        
        if len(feature_columns) > 1:
            feature_data = df[feature_columns]
            df_stats['features_mean'] = feature_data.mean(axis=1)
            df_stats['features_std'] = feature_data.std(axis=1)
            df_stats['features_max'] = feature_data.max(axis=1)
            df_stats['features_min'] = feature_data.min(axis=1)
            df_stats['features_range'] = df_stats['features_max'] - df_stats['features_min']
        
        return df_stats
    
    def select_k_best_features(self, X, y, k=10, score_func=f_classif):
        """Select k best features"""
        k = min(k, X.shape[1])
        
        self.selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        selected_mask = self.selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """Apply PCA"""
        if n_components is None:
            self.pca = PCA()
            self.pca.fit(X)
            cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    
    def get_feature_importance_scores(self, X, y):
        """Get feature importance"""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)