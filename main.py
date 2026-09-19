import os
import requests
import joblib
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

# GitHub Repository theke .pkl models direct Load kora
try:
    dengue_model = joblib.load("dengue_model.pkl")
    malaria_model = joblib.load("malaria_model.pkl")
    print("PKL Models Loaded Successfully!")
except Exception as e:
    print(f"Error loading PKL models: {e}")
    dengue_model = None
    malaria_model = None

OPENWEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

@app.get("/")
def home():
    return {"status": "Aegis PKL Model Engine Active."}

@app.get("/predict")
def predict(city: str = "Howrah", api_key: str = Depends(get_api_key)):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        temp_val = float(data['main']['temp'])
        humidity_val = float(data['main']['humidity'])
        rain_val = float(data.get('rain', {}).get('1h', 0.0))
    else:
        temp_val, humidity_val, rain_val = 28.0, 70.0, 0.0
        
    features = np.array([[temp_val, humidity_val, rain_val]])
    
    # Pre-trained PKL Model prediction
    if dengue_model is not None and malaria_model is not None:
        predicted_dengue = int(dengue_model.predict(features)[0])
        predicted_malaria = int(malaria_model.predict(features)[0])
    else:
        # Fallback calculation if PKL fails
        predicted_dengue = int((temp_val * 2.5) + (humidity_val * 0.8) - 10)
        predicted_malaria = int((temp_val * 2.2) + (humidity_val * 0.7) - 12)

    # Biological Temperature Guard (Darjeeling-er moto thanda elaka)
    if temp_val < 18.0:
        predicted_dengue = max(2, int(predicted_dengue * 0.15))
        predicted_malaria = max(1, int(predicted_malaria * 0.15))

    # Risk Thresholds
    dengue_risk = "High" if predicted_dengue >= 85 else ("Medium" if predicted_dengue >= 40 else "Low")
    malaria_risk = "High" if predicted_malaria >= 80 else ("Medium" if predicted_malaria >= 38 else "Low")
        
    return {
        "client_accessed": VALID_API_KEYS[api_key],
        "city": city,
        "temperature": temp_val,
        "humidity": humidity_val,
        "rainfall": rain_val,
        "dengue": {
            "predicted_cases": predicted_dengue,
            "risk_level": dengue_risk
        },
        "malaria": {
            "predicted_cases": predicted_malaria,
            "risk_level": malaria_risk
        }
    }
