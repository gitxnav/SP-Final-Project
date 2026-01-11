# ML Pipeline Setup Guide

## Project Structure

```
your-repo/
├── .github/
│   └── workflows/
│       ├── test-develop.yml          # API tests workflow
│       └── test-ml-develop.yml       # ML pipeline tests workflow
├── src/
│   ├── data_loading.py               # Data loading module
│   ├── data_processing.py            # Data processing module
│   ├── feature_engineering.py        # Feature engineering module
│   ├── model_training.py             # Model training module
│   ├── main.py                       # FastAPI application
│   └── tests/
│       ├── __init__.py              # Empty file
│       ├── test_api.py              # API tests
│       └── test_ml.py               # ML pipeline tests
├── data/
│   └── sample_data.txt              # Sample dataset
├── models/                          # Saved models (auto-created)
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies (includes FastAPI)
└── requirements-test.txt            # Testing dependencies
```

## Quick Setup

### 1. Install Dependencies

```bash
# Install development dependencies (includes FastAPI for develop branch)
pip install -r requirements-dev.txt

# Install test dependencies
pip install -r requirements-test.txt
```

### 2. Create Required Directories

```bash
mkdir -p data models reports src/tests
touch src/__init__.py
touch src/tests/__init__.py
```

### 3. Run Tests Locally

**Run API tests:**
```bash
python -m pytest src/tests/test_api.py -v
```

**Run ML tests:**
```bash
python -m pytest src/tests/test_ml.py -v
```

**Run all tests with coverage:**
```bash
python -m pytest src/tests/ -v --cov=src --cov-report=term-missing
```

### 4. Run the FastAPI Application

```bash
# Start the server
python src/main.py

# Or using uvicorn directly
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Testing Individual Components

### Test Data Loading
```bash
python src/data_loading.py
```

### Test Data Processing
```bash
python src/data_processing.py
```

### Test Feature Engineering
```bash
python src/feature_engineering.py
```

### Test Model Training
```bash
python src/model_training.py
```

## GitHub Actions Workflows

### API Tests (`test-develop.yml`)
- Runs on: Push/PR to `develop` branch
- Tests: FastAPI endpoints
- Python versions: 3.11, 3.12

### ML Tests (`test-ml-develop.yml`)
- Runs on: Push/PR to `develop` branch
- Tests: Full ML pipeline
- Python versions: 3.11, 3.12
- Coverage: Code coverage report

## What Gets Tested

### Data Loading Tests
- Loading data from text files
- Data validation
- Train/test splitting

### Data Processing Tests
- Missing value handling
- Feature normalization
- Outlier removal
- Feature/target splitting

### Feature Engineering Tests
- Polynomial features
- Interaction features
- Statistical features
- Feature selection
- PCA dimensionality reduction

### Model Training Tests
- Model training
- Model evaluation
- Model persistence (save/load)
- Cross-validation
- Feature importances
- Full pipeline integration

## Using the ML Pipeline

```python
from src.data_loading import DataLoader
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

# 1. Load data
loader = DataLoader()
df = loader.load_from_txt("sample_data.txt")

# 2. Process data
processor = DataProcessor()
df_clean = processor.remove_missing_values(df)
df_normalized = processor.normalize_features(df_clean)
X, y = processor.split_features_target(df_normalized)

# 3. Engineer features
engineer = FeatureEngineer()
X_engineered = engineer.create_interaction_features(X)

# 4. Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2)

trainer = ModelTrainer()
trainer.train_model(X_train, y_train)
metrics = trainer.evaluate_model(X_test, y_test)
trainer.save_model()
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /model-info` - Model information

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, -0.1, 0.6, 1.5]
  }'
```

## Troubleshooting

**Issue: Tests fail with import errors**
```bash
# Make sure you have __init__.py files
touch src/__init__.py
touch src/tests/__init__.py
```

**Issue: Data directory not found**
```bash
# Create required directories
mkdir -p data models reports
```

**Issue: ModuleNotFoundError**
```bash
# Install all dependencies
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
```

## Next Steps

1. Push to `develop` branch to trigger GitHub Actions
2. Check the Actions tab to see test results
3. Customize model parameters in `model_training.py`
4. Add your own data to `data/` directory
5. Extend the pipeline with more features

Happy coding! 🚀