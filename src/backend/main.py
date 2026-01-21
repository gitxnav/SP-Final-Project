from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="CKD Detection API", version="1.0.0")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PredictionInput(BaseModel):
    age: int
    blood_pressure: int
    specific_gravity: float
    albumin: int
    sugar: int

class PredictionOutput(BaseModel):
    prediction: str
    probability: float

# Routes
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "CKD API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # Mock prediction logic
    score = (data.age * 0.01 + data.blood_pressure * 0.02 + 
             data.albumin * 0.1 + data.sugar * 0.15)
    
    prediction = "CKD" if score > 5 else "No CKD"
    probability = min(score / 10, 1.0)
    
    return {
        "prediction": prediction,
        "probability": round(probability, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)