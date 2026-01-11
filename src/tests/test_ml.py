"""
Comprehensive ML Pipeline Tests
Tests for data loading, processing, feature engineering, and model training
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading import DataLoader
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
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


@pytest.fixture
def data_loader():
    """Create DataLoader instance"""
    return DataLoader(data_dir="data")


@pytest.fixture
def data_processor():
    """Create DataProcessor instance"""
    return DataProcessor()


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance"""
    return FeatureEngineer()


@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance"""
    return ModelTrainer(model_dir="models")


# ============================================================================
# DATA LOADING TESTS
# ============================================================================

def test_data_loading(data_loader):
    """Test that data loading works correctly"""
    # Create sample data file
    filepath = data_loader.save_sample_data("test_data.txt")
    assert filepath.exists(), "Sample data file should be created"
    
    # Load the data
    df = data_loader.load_from_txt("test_data.txt")
    
    assert df is not None, "Data should be loaded"
    assert not df.empty, "Loaded data should not be empty"
    assert len(df) == 100, "Should load 100 rows"
    assert 'target' in df.columns, "Should have target column"


def test_data_validation(data_loader, sample_data):
    """Test data validation"""
    is_valid = data_loader.validate_data(sample_data)
    assert is_valid is True, "Valid data should pass validation"
    
    # Test empty DataFrame
    with pytest.raises(ValueError):
        data_loader.validate_data(pd.DataFrame())


def test_train_test_split(data_loader):
    """Test train/test split functionality"""
    data_loader.save_sample_data("split_test.txt")
    train_df, test_df = data_loader.load_train_test_split("split_test.txt", test_size=0.2)
    
    assert len(train_df) == 80, "Training set should have 80 rows"
    assert len(test_df) == 20, "Test set should have 20 rows"


# ============================================================================
# DATA PROCESSING TESTS
# ============================================================================

def test_data_processing(data_processor, sample_data):
    """Test data processing pipeline"""
    # Test missing value handling
    df_clean = data_processor.remove_missing_values(sample_data)
    assert df_clean is not None, "Cleaned data should not be None"
    assert len(df_clean) > 0, "Cleaned data should have rows"


def test_normalization(data_processor, sample_data):
    """Test feature normalization"""
    df_normalized = data_processor.normalize_features(sample_data, method="standard")
    
    assert df_normalized is not None, "Normalized data should not be None"
    assert data_processor.scaler is not None, "Scaler should be fitted"
    
    # Check that normalized features have mean ≈ 0 and std ≈ 1
    feature_cols = [col for col in sample_data.columns if col != 'target']
    for col in feature_cols:
        assert abs(df_normalized[col].mean()) < 0.1, f"{col} should have mean ≈ 0"
        assert abs(df_normalized[col].std() - 1.0) < 0.1, f"{col} should have std ≈ 1"


def test_outlier_removal(data_processor, sample_data):
    """Test outlier removal"""
    # Add some outliers
    df_with_outliers = sample_data.copy()
    df_with_outliers.loc[0, 'feature1'] = 100  # Obvious outlier
    
    df_clean = data_processor.remove_outliers(df_with_outliers, n_std=3.0)
    
    assert len(df_clean) < len(df_with_outliers), "Should remove some outliers"


def test_feature_target_split(data_processor, sample_data):
    """Test splitting features and target"""
    X, y = data_processor.split_features_target(sample_data)
    
    assert X.shape[0] == len(sample_data), "Features should have same rows as original"
    assert y.shape[0] == len(sample_data), "Target should have same rows as original"
    assert 'target' not in X.columns, "Features should not contain target"


# ============================================================================
# FEATURE ENGINEERING TESTS
# ============================================================================

def test_feature_engineering(feature_engineer, sample_data):
    """Test feature engineering pipeline"""
    X, _ = DataProcessor().split_features_target(sample_data)
    
    # Test polynomial features
    X_poly = feature_engineer.create_polynomial_features(X, ['feature1', 'feature2'])
    assert X_poly.shape[1] > X.shape[1], "Should have more features after polynomial"


def test_interaction_features(feature_engineer, sample_data):
    """Test interaction feature creation"""
    X, _ = DataProcessor().split_features_target(sample_data)
    
    X_interact = feature_engineer.create_interaction_features(X)
    assert X_interact.shape[1] > X.shape[1], "Should have more features after interactions"


def test_statistical_features(feature_engineer, sample_data):
    """Test statistical feature creation"""
    X_stats = feature_engineer.create_statistical_features(sample_data)
    
    assert 'features_mean' in X_stats.columns, "Should have mean feature"
    assert 'features_std' in X_stats.columns, "Should have std feature"
    assert 'features_max' in X_stats.columns, "Should have max feature"


def test_feature_selection(feature_engineer, sample_data):
    """Test feature selection"""
    processor = DataProcessor()
    X, y = processor.split_features_target(sample_data)
    
    X_selected = feature_engineer.select_k_best_features(X, y, k=3)
    
    assert X_selected.shape[1] == 3, "Should select exactly 3 features"
    assert feature_engineer.selected_features is not None, "Should store selected features"


def test_pca(feature_engineer, sample_data):
    """Test PCA dimensionality reduction"""
    processor = DataProcessor()
    X, _ = processor.split_features_target(sample_data)
    
    X_pca = feature_engineer.apply_pca(X, n_components=2)
    
    assert X_pca.shape[1] == 2, "Should have 2 principal components"
    assert 'PC1' in X_pca.columns, "Should have PC1 column"


# ============================================================================
# MODEL TRAINING TESTS
# ============================================================================

def test_model_training(model_trainer, sample_data):
    """Test model training"""
    processor = DataProcessor()
    X, y = processor.split_features_target(sample_data)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = model_trainer.train_model(X_train, y_train)
    
    assert model is not None, "Model should be trained"
    assert hasattr(model, 'predict'), "Model should have predict method"


def test_model_evaluation(model_trainer, sample_data):
    """Test model evaluation"""
    processor = DataProcessor()
    X, y = processor.split_features_target(sample_data)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model_trainer.train_model(X_train, y_train)
    metrics = model_trainer.evaluate_model(X_test, y_test)
    
    assert 'accuracy' in metrics, "Should have accuracy metric"
    assert 'f1_score' in metrics, "Should have F1 score"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"


def test_model_save_load(model_trainer, sample_data, tmp_path):
    """Test model persistence"""
    processor = DataProcessor()
    X, y = processor.split_features_target(sample_data)
    
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and save
    model_trainer.train_model(X_train, y_train)
    save_path = model_trainer.save_model("test_model.pkl")
    
    assert save_path.exists(), "Model file should exist"
    
    # Load
    new_trainer = ModelTrainer()
    loaded_model = new_trainer.load_model("test_model.pkl")
    
    assert loaded_model is not None, "Model should be loaded"


def test_cross_validation(model_trainer, sample_data):
    """Test cross-validation"""
    processor = DataProcessor()
    X, y = processor.split_features_target(sample_data)
    
    model_trainer.train_model(X, y)
    cv_results = model_trainer.cross_validate(X, y, cv=3)
    
    assert 'cv_mean' in cv_results, "Should have mean CV score"
    assert 'cv_std' in cv_results, "Should have CV std"
    assert len(cv_results['cv_scores']) == 3, "Should have 3 CV scores"


def test_feature_importances(model_trainer, sample_data):
    """Test feature importance extraction"""
    processor = DataProcessor()
    X, y = processor.split_features_target(sample_data)
    
    model_trainer.train_model(X, y)
    importances = model_trainer.get_feature_importances(X.columns.tolist())
    
    assert len(importances) == X.shape[1], "Should have importance for each feature"
    assert 'feature' in importances.columns, "Should have feature names"
    assert 'importance' in importances.columns, "Should have importance values"


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_full_ml_pipeline():
    """Test the complete ML pipeline from data loading to model training"""
    # 1. Load data
    loader = DataLoader()
    loader.save_sample_data("pipeline_test.txt")
    df = loader.load_from_txt("pipeline_test.txt")
    
    # 2. Process data
    processor = DataProcessor()
    df_clean = processor.remove_missing_values(df)
    df_normalized = processor.normalize_features(df_clean)
    X, y = processor.split_features_target(df_normalized)
    
    # 3. Engineer features
    engineer = FeatureEngineer()
    X_engineered = engineer.create_statistical_features(df_normalized)
    X_final, _ = processor.split_features_target(X_engineered)
    
    # 4. Train model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    
    trainer = ModelTrainer()
    trainer.train_model(X_train, y_train)
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Verify pipeline worked
    assert metrics['accuracy'] > 0, "Model should have non-zero accuracy"
    assert len(X_train) == 80, "Should have correct train size"
    assert len(X_test) == 20, "Should have correct test size"