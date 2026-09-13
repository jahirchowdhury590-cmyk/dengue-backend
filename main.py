from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import requests
import pickle
import numpy as np

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key System Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Valid API Keys Dictionary
VALID_API_KEYS = {
    "aegis_my_lovable_frontend": "AegisDengue Public Dashboard", # আপনার ওয়েবসাইটের জন্য
    "client_demo_key_001": "B2B Client Demo" # ভবিষ্যতে ক্লায়েন্টদের জন্য
}

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(status_code=403, detail="Access Denied: Invalid API Key")

# Load ML Model
with open('dengue_model.pkl', 'rb') as f:
    model = pickle.load(f)

# OpenWeather API Key (আপনার আগের কি-টি)
OPENWEATHER_API_KEY = "f99698fc4818c022f008cf96a4f340be"

@app.get("/")
def home():
    return {"status": "AegisDengue API is running securely!"}

@app.get("/predict")
def predict(city: str = "Howrah", api_key: str = Depends(get_api_key)):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        rain = data.get('rain', {}).get('1h', 0)
    else:
        temp, humidity, rain = 30.0, 80.0, 2.0
        
    features = np.array([[temp, humidity, rain]])
    predicted_cases = max(0, int(model.predict(features)[0]))
    
    if predicted_cases > 50:
        risk = "High"
    elif predicted_cases > 20:
        risk = "Medium"
    else:
        risk = "Low"
        
    return {
        "client_accessed": VALID_API_KEYS[api_key],
        "city": city,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rain,
        "predicted_cases": predicted_cases,
        "risk_level": risk
    }
