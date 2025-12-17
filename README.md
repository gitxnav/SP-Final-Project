## Project Description

This project uses the Chronic Kidney Disease (CKD) dataset to build a machine learning model that predicts whether a patient has CKD or not. The dataset contains clinical and laboratory features with missing values, which are cleaned and processed before training and evaluating the model. The final model is deployed as an API that takes patient data as input and returns a diagnostic prediction.

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

**Note**. If we download news packages we need to include them in the requirements file. For doing that, one we installed the package, we need to update the `requirements.txt` file:
```python
pip freeze >> path/to/requirements.txt
```
