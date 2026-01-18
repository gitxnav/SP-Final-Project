## Project Description - Chronic Kidney Disease (CKD)

This project uses the Chronic Kidney Disease (CKD) dataset to build a machine learning model that predicts whether a patient has CKD or not. The dataset contains clinical and laboratory features with missing values, which are cleaned and processed before training and evaluating the model. The final model is deployed as an API that takes patient data as input and returns a diagnostic prediction.

## Configuration of LOCAL environment
It is needed once the repo is cloned to create a python environment and to install the requirements file.

To install the python environment we need to execute:
```python

python -m venv .env

```

Once we have it, we need to activate it.

- If we are in Windows:

```python

.\.env\Scripts\activate

```

- If we are in Linux:

```
source .env/bin/activate
```

And once we have it activated, we need to download the packages that are in the `requirements.txt` file:

```python
pip install -r path/to/requirements.txt
```

**Note**. If we download news packages we need to include them in the requirements file. For doing that, once we installed the package, we need to update the `requirements.txt` file:
```python
pip freeze >> path/to/requirements.txt
```

## Configuration of DEVELOPMENT environment

The tree structure of Web Application

.
├── Dockerfile
├── README.md
├── STREAMLIT_SETUP.md
├── data
│   ├── processed
│   │   ├── ckd_imputed.csv
│   │   └── ckd_normalized.csv
│   └── raw
│       ├── chronic_kidney_disease_info.txt
│       └── chronic_kindey_disease.csv
├── requirements.txt
├── run_app.sh
├── src
│   ├── __init__.py
│   ├── app.py
│   ├── step01_data_loading.py
│   ├── step02_data_processing.py
│   ├── step03_feature_engineering.py
│   ├── step04_model_training.py
│   ├── step05_model_prediction.py
│   ├── step06_mlflow_config.py
|---└── streamlit_app.code-workspace


## DEPLOYMENT Streamlit web application + MLflow framework + FastAPI

- release_version: v1.0-dev
- last_modified_date: 2026.01.17 

To deploy by Docker is necessary to follow those guidelines from your terminal:

$ docker build --progress=plain -t ckd_webapp:v1.0 .

$ docker run -d restart unless-stopped \
    --name ckd_webapp_1 \
    -p 8000:8000 -p 8502:8502 -p 5000:5000 \
    ckd_webapp:v1.0

Finally, open on your browser http://localhost:8502 to interact with web application.
