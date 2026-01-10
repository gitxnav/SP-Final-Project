# CI/CD Pipeline Setup Guide

## Project Structure

```
your-repo/
├── .github/
│   └── workflows/
│       ├── cicd.yml                  # Main CI/CD pipeline (develop → main)
│       ├── test-develop.yml          # API tests on develop
│       └── test-ml-develop.yml       # ML tests on develop
├── src/
│   ├── __init__.py                  # Empty file (required)
│   ├── main.py                      # FastAPI application
│   ├── data_loading.py              # Data loading module
│   ├── data_processing.py           # Data processing module
│   ├── feature_engineering.py       # Feature engineering module
│   ├── model_training.py            # Model training module
│   └── tests/
│       ├── __init__.py              # Empty file (required)
│       ├── test_api.py              # FastAPI tests (11 tests)
│       └── test_ml.py               # ML pipeline tests (17 tests)
├── data/
│   └── sample_data.txt              # Sample dataset
├── models/                          # Saved models (auto-created)
├── reports/                         # Test reports (auto-created)
└── requirements.txt                 # ALL dependencies in ONE file
```

## CI/CD Workflow Explained

### **cicd.yml** - Main Pipeline (Develop → Main)

This is your **gatekeeper** for merging to main. It runs in this order:

```
1. Pre-commit Checks (pylint, black, flake8)
         ↓
2a. API Tests          2b. ML Tests
    (test_api.py)          (test_ml.py)
         ↓                      ↓
3. Integration Tests (all tests together)
         ↓
4. Merge Gate (only for PRs to main)
```

**When it runs:**
- Every push to `develop`
- Every PR to `main` or `develop`

**What it does:**
- Pre-commit checks (linting)
- Runs API tests on Python 3.11 & 3.12
- Runs ML tests on Python 3.11 & 3.12
- Runs full integration tests
- **BLOCKS merge to main if ANY test fails**
- Posts comment on PR when ready to merge

### **test-develop.yml** - API Tests Only

Simple workflow for quick API testing on develop branch.

### **test-ml-develop.yml** - ML Tests Only

ML pipeline testing with coverage reports on develop branch.

## Quick Setup

### 1. Install All Dependencies

**ONE requirements.txt file with EVERYTHING:**

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI + uvicorn
- ML libraries (scikit-learn, numpy, pandas, etc.)
- Testing tools (pytest, pytest-cov, httpx)
- Code quality tools (pylint, black, flake8)

### 2. Create Required Files

```bash
# Create directories
mkdir -p data models reports

# Create __init__.py files (IMPORTANT!)
touch src/__init__.py
touch src/tests/__init__.py
```

### 3. Verify Setup

```bash
# Run all tests locally
python -m pytest src/tests/ -v

# Run with coverage
python -m pytest src/tests/ -v --cov=src --cov-report=term-missing

# Run specific test files
python -m pytest src/tests/test_api.py -v
python -m pytest src/tests/test_ml.py -v
```

## 📊 What Gets Tested

### API Tests (test_api.py) - 11 Tests
- Root endpoint
- Health check
- Prediction endpoint (valid input)
- Prediction endpoint (invalid features)
- Prediction endpoint (missing data)
- Model info endpoint
- Predictions for all iris classes
- API documentation availability
- OpenAPI schema

### ML Tests (test_ml.py) - 17 Tests
- Data loading from text files
- Data validation
- Train/test splitting
- Missing value handling
- Feature normalization (StandardScaler)
- Outlier removal
- Feature/target splitting
- Polynomial feature creation
- Interaction features
- Statistical features
- Feature selection (SelectKBest)
- PCA dimensionality reduction
- Model training
- Model evaluation
- Model save/load
- Cross-validation
- Full pipeline integration

## Git Workflow

### Working on Develop

```bash
# Make changes
git checkout develop
git add .
git commit -m "Add new feature"
git push origin develop
```

**What happens:**
- `test-develop.yml` runs API tests
- `test-ml-develop.yml` runs ML tests
- `cicd.yml` runs all checks

### Merging to Main

```bash
# Create PR from develop to main
git checkout develop
git pull origin develop
# Go to GitHub and create PR: develop → main
```

**What happens:**
1. Pre-commit checks run
2. API tests run (Python 3.11, 3.12)
3. ML tests run (Python 3.11, 3.12)
4. Integration tests run
5. **Merge Gate activates**
6. Bot posts comment if ready to merge

**If any test fails:** ❌ **PR CANNOT be merged!**

## Running Tests Locally

### Run All Tests
```bash
python -m pytest src/tests/ -v
```

### Run Specific Test File
```bash
python -m pytest src/tests/test_api.py -v
python -m pytest src/tests/test_ml.py -v
```

### Run With Coverage
```bash
python -m pytest src/tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to see coverage report
```

### Run Specific Test
```bash
python -m pytest src/tests/test_ml.py::test_data_loading -v
python -m pytest src/tests/test_api.py::test_predict_endpoint_success -v
```

## Running FastAPI Application

```bash
# From project root
cd src
python main.py

# Or using uvicorn
uvicorn src.main:app --reload
```

Visit:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Test Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Requirements.txt Format

Your `requirements.txt` should look like this:

```txt
# FastAPI and server
fastapi~=0.120
uvicorn[standard]==0.34.0
pydantic==2.10.4
python-multipart==0.0.20

# ML libraries
scikit-learn~=1.8.0
numpy~=2.3
pandas~=2.3.3
scipy~=1.16
matplotlib~=3.10
seaborn~=0.13.2
joblib==1.4.2

# Utilities
requests~=2.32.5
python-dotenv==1.0.1

# Testing (needed for GitHub Actions)
pytest==8.3.4
pytest-cov==6.0.0
httpx==0.28.1

# Code quality
pylint
black
flake8
```

**Note:** Using `~=` allows patch updates (e.g., `~=1.8.0` means `>=1.8.0, <1.9.0`)

## Troubleshooting

### Issue: Import errors in tests
```bash
# Solution: Create __init__.py files
touch src/__init__.py
touch src/tests/__init__.py
```

### Issue: ModuleNotFoundError: No module named 'main'
```bash
# Solution: Make sure you're in the right directory
cd src/tests
python -m pytest test_api.py -v
# Or from project root:
python -m pytest src/tests/test_api.py -v
```

### Issue: Tests pass locally but fail in GitHub Actions
```bash
# Check that all dependencies are in requirements.txt
pip freeze > requirements-check.txt
# Compare with your requirements.txt
```

### Issue: Model file not found
```bash
# Solution: Create models directory
mkdir -p models
# The model will be created automatically on first run
```

### Issue: Pylint fails in CI/CD
```bash
# Run locally to see issues
pylint src/ --disable=C0114,C0115,C0116
# Fix or adjust .pylintrc file
```

## Viewing Test Results in GitHub

1. Go to your repo → **Actions** tab
2. Click on latest workflow run
3. See results for each job:
   - Pre-commit Checks
   - API Tests (Python 3.11, 3.12)
   - ML Tests (Python 3.11, 3.12)
   - Integration Tests
   - Merge Gate

4. Download artifacts:
   - Coverage reports
   - Test results

## Best Practices

1. **Always run tests locally before pushing**
   ```bash
   python -m pytest src/tests/ -v
   ```

2. **Keep develop branch clean**
   - Only merge working code
   - All tests should pass

3. **Use PRs for main merges**
   - Never push directly to main
   - Let CI/CD verify everything

4. **Check coverage**
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```
   - Aim for >80% coverage

5. **Update requirements carefully**
   - Test after any dependency change
   - Use version constraints (`~=`)

## Summary

You now have:
- Complete CI/CD pipeline with merge gates
- Pre-commit checks (linting)
- API tests with FastAPI
- ML pipeline tests with coverage
- All dependencies in ONE requirements.txt
- Automated blocking of bad merges to main

**Next steps:**
1. Push to `develop` → See tests run
2. Create PR to `main` → See merge gate activate
3. All tests pass → Merge allowed ✅
4. Any test fails → Fix before merge ❌