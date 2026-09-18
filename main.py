import os
import pickle
import requests
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

VALID_API_KEYS = {
    "aegis_my_lovable_frontend": "AegisDengue Public Dashboard",
    "client_demo_key_001": "B2B Client Demo"
}

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")

# Model Loading
with open('dengue_model.pkl', 'rb') as f:
    dengue_model = pickle.load(f)

with open('malaria_model.pkl', 'rb') as f:
    malaria_model = pickle.load(f)

OPENWEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

@app.get("/")
def home():
    return {"status": "Aegis Vector-Borne Disease Prediction API is running securely!"}

@app.get("/predict")
def predict(city: str = "Howrah", api_key: str = Depends(get_api_key)):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        rain = data.get('rain', {}).get('1h', 0)
    else:
        temp, humidity, rain = 30.0, 80.0, 2.0
        
    features = np.array([[temp, humidity, rain]])
    
    # Dengue Prediction
    predicted_dengue_cases = max(0, int(dengue_model.predict(features)[0]))
    if predicted_dengue_cases > 50:
        dengue_risk = "High"
    elif predicted_dengue_cases > 20:
        dengue_risk = "Medium"
    else:
        dengue_risk = "Low"
        
    # Malaria Prediction
    predicted_malaria_cases = max(0, int(malaria_model.predict(features)[0]))
    if predicted_malaria_cases > 60:
        malaria_risk = "High"
    elif predicted_malaria_cases > 25:
        malaria_risk = "Medium"
    else:
        malaria_risk = "Low"
        
    return {
        "client_accessed": VALID_API_KEYS[api_key],
        "city": city,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rain,
        "dengue": {
            "predicted_cases": predicted_dengue_cases,
            "risk_level": dengue_risk
        },
        "malaria": {
            "predicted_cases": predicted_malaria_cases,
            "risk_level": malaria_risk
        }
    }
