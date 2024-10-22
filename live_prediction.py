# Save this as ml_model_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your trained model (ensure the path is correct)
model = joblib.load('path_to_your_model/random_forest_model.pkl')

class SensorData(BaseModel):
    sensor1: float
    sensor2: float
    sensor3: float
    temp: float
    humidity: float

@app.post("/predict")
def predict(data: SensorData):
    features = np.array([data.sensor1, data.sensor2, data.sensor3, data.temp, data.humidity]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"health_score": prediction}


