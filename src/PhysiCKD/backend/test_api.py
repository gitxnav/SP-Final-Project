import requests
import json

url = 'http://localhost:5000/predict'
data = {
    'hemo': 10.34,
    'sg': 2,
    'sc': 10.34,
    'rbcc': 10.34,
    'pcv': 32.01,
    'htn': 0,
    'dm': 1,
    'bp': 67,
    'age': 22
}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)
