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

```{shell}
.
├── data
│   ├── processed
│   │   ├── ckd_imputed.csv
│   │   ├── ckd_normalized.csv
│   │   ├── dataset_knn_imputed.csv
│   │   ├── dataset_mice_imputed.csv
│   │   ├── dataset_missForest_imputed.csv
│   │   └── processed_dataset_ex.csv
│   ├── raw
│   │   ├── chronic_kidney_disease_info.txt
│   │   └── chronic_kindey_disease.csv
│   ├── sample_data.txt
│   └── samples
│       ├── ckd_imputed.csv
│       └── ckd_normalized.csv
├── docker
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── Dockerfile.mlflow
│   ├── run_app.sh
│   └── run_backend.sh
├── docker-compose.yml
├── requirements.txt
├── run_app.sh
├── scripts
│   └── start.sh
└── src
    ├── __init__.py
    ├── backend
    │   ├── __init__.py
    │   ├── api
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   └── routes
    │   │       ├── __init__.py
    │   │       ├── health.py
    │   │       ├── mlflow.py
    │   │       ├── prediction.py
    │   │       └── training.py
    │   ├── config
    │   │   ├── mlflow_config.py
    │   │   └── settings.py
    │   ├── core
    │   │   ├── __init__.py
    │   │   ├── step01_data_loading.py
    │   │   ├── step02_data_processing.py
    │   │   ├── step03_feature_engineering.py
    │   │   ├── step04_model_training.py
    │   │   ├── step05_model_prediction.py
    │   │   ├── step06_mlflow_config.py
    │   │   ├── step07_mflow_training.py
    │   │   └── step08_model_inference.py
    │   ├── data
    │   ├── figures
    │   │   └── models
    │   ├── main.py
    │   ├── ml
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   │   └── train.py
    │   ├── mlruns
    │   ├── models
    │   └── reports
    │       └── models
    ├── frontend
    │   ├── __init__.py
    │   ├── data
    │   ├── figures
    │   │   └── models
    │   │       ├── gradient boosting_feature_importances.png
    │   │       ├── gradientboosting_imputed_confusion_matrix.png
    │   │       ├── knn_confusion_matrix.png
    │   │       ├── knn_optimization.png
    │   │       └── svm_confusion_matrix.png
    │   ├── internal
    │   │   ├── __init__.py
    │   │   ├── app.py
    │   │   ├── step01_data_loading.py
    │   │   ├── step02_data_processing.py
    │   │   ├── step03_feature_engineering.py
    │   │   ├── step04_model_training.py
    │   │   └── step05_model_prediction.py
    │   ├── models
    │   │   ├── gb_imputed_model.pkl
    │   │   ├── knn_model.pkl
    │   │   └── svm_model.pkl
    │   ├── reports
    │   │   ├── feature_engineering_stats.json
    │   │   └── models
    │   └── utils
    │       ├── __init__.py
    │       └── api_client.py

```

## DEPLOYMENT Streamlit web application + MLflow framework + FastAPI

- release_version: v1.0-dev
- last_modified_date: 2026.01.22 

To deploy the main architecture by Docker it's important understand the two main parts:
- backend (FastAPI, the core) and MLflow framework
- frontend (Streamlit)

Now, follows these guidelines from your terminal to start the Docker containers.

From your workspace execute the `git clone`, if you haven't on your local environment. Otherwise, the first thing get a fetching and pull the last changes from main branch. It's mandatory to have the last version of the repository.

Important, if you have forked the repository from on your gitHub profile, first get the last changes from the original repository.

If you haven't the repository, just execute:

```{shell}
$ git clone https://github.com/gitxnav/SP-Final-Project
```

Check the status of the branch by:

```{shell}
$ git status
```

After this step, it's important having Docker Desktop or docker running. If you don't have Docker, it depends on your 
main Operating System. You could read about Docker following the installation manuals: https://docs.docker.com/engine/install/
 

The next step, to launch the applications execute this command first to allows the permissions to execute:

```{shell}
$ chmod +x start_services.sh stop_services.sh
```

Then, to execute the applications:

```{shell}
$ ./scripts/start_services.sh
```

During the next 5-10 minutes, you will show the docker-compose running for you. 
If there are no errors, the last part it's important to open the applications.

You will see something like:

```{shell}
 ✔ Network sp-final-project_ckd-network  Created                                                                                                                                                                                                    0.1s 
 ✔ Container ckd-mlflow                  Healthy                                                                                                                                                                                                    0.1s 
 ✔ Container ckd-backend                 Healthy                                                                                                                                                                                                    0.0s 
 ✔ Container ckd-frontend-internal       Started                                                                                                                                                                                                    0.0s 
⏳ Waiting for services to be healthy...
📊 Service Status:
NAME                    IMAGE                                COMMAND                  SERVICE             CREATED          STATUS                    PORTS
ckd-backend             sp-final-project-backend             "uvicorn api.main:ap…"   backend             38 seconds ago   Up 21 seconds (healthy)   5050/tcp, 0.0.0.0:8000->8000/tcp
ckd-frontend-internal   sp-final-project-frontend-internal   "streamlit run inter…"   frontend-internal   38 seconds ago   Up 10 seconds             0.0.0.0:8501->8501/tcp, 8502/tcp
ckd-mlflow              sp-final-project-mlflow              "mlflow server --bac…"   mlflow              38 seconds ago   Up 37 seconds (healthy)   0.0.0.0:5050->5050/tcp

✅ System Started!

📍 Access Points:
   Internal Dashboard: http://localhost:8501
   User App:           http://localhost:8502
   Backend API:        http://localhost:8000/docs
   MLflow UI:          http://localhost:5050

🔧 Useful Commands:
   View logs:          docker-compose logs -f
   Stop system:        docker-compose down
   Restart backend:    docker-compose restart backend
```

That's it. Now you can open the application on your browser.

* Streamlit web application --> http://localhost:8501
* FastAPI --> http://localhost:8000 
* MLflow framework http://localhost:5050 

To stop the three docker containers or remove the containers on your local system, execute:

```{shell}
$ ./scripts/stop_services.sh
```

Finally, you could remove manually from Docker Desktop or by docker commands.

## Troubleshooting

It is normal if a Docker container does not work properly. This may depend on your operating system or on some dependencies running in the background. Please contact the developer if you experience any issues.
