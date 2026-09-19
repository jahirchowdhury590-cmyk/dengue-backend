import os
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

# Live Auto-Training with adjusted realistic formulas
np.random.seed(42)
num_samples = 1000

# 1. Dengue Model
temp_d = np.random.uniform(20.0, 35.0, num_samples)
hum_d = np.random.uniform(50.0, 95.0, num_samples)
rain_d = np.random.uniform(0.0, 100.0, num_samples)
# Added a base number (15) so cases don't drop to 0 easily
cases_d = 15 + (rain_d * 0.5) + (temp_d * 0.8) + (hum_d * 0.2) + np.random.normal(0, 5, num_samples)
X_d = np.column_stack((temp_d, hum_d, rain_d))
y_d = np.clip(cases_d, 5, None).astype(int)
dengue_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_d, y_d)

# 2. Malaria Model
temp_m = np.random.uniform(20.0, 35.0, num_samples)
hum_m = np.random.uniform(50.0, 95.0, num_samples)
rain_m = np.random.uniform(0.0, 50.0, num_samples)
# Added a base number (20) for malaria
cases_m = 20 + (temp_m * 1.0) + (hum_m * 0.5) + (rain_m * 0.3) + np.random.normal(0, 4, num_samples)
X_m = np.column_stack((temp_m, hum_m, rain_m))
y_m = np.clip(cases_m, 8, None).astype(int)
malaria_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_m, y_m)

OPENWEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

@app.get("/")
def home():
    return {"status": "Aegis API is running perfectly!"}

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
        temp, humidity, rain = 28.0, 75.0, 0.0
        
    features = np.array([[temp, humidity, rain]])
    
    # Dengue Prediction (Adjusted limits for risk levels)
    predicted_dengue_cases = max(2, int(dengue_model.predict(features)[0]))
    dengue_risk = "High" if predicted_dengue_cases > 40 else ("Medium" if predicted_dengue_cases > 20 else "Low")
        
    # Malaria Prediction (Adjusted limits for risk levels)
    predicted_malaria_cases = max(3, int(malaria_model.predict(features)[0]))
    malaria_risk = "High" if predicted_malaria_cases > 45 else ("Medium" if predicted_malaria_cases > 25 else "Low")
        
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
