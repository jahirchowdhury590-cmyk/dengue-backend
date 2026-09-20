import os
import requests
import numpy as np
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
    raise HTTPException(status_code=403, detail="Access Denied")

OPENWEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

# ==========================================
# 1. INDEPENDENT DENGUE AI ENGINE
# ==========================================
np.random.seed(42)
X_train_dengue = np.random.uniform(10, 40, (2000, 3)) 
X_train_dengue[:, 1] = np.random.uniform(30, 95, 2000) 
y_dengue = []

for t, h, r in X_train_dengue:
    if t < 18.0:
        y_dengue.append(np.random.uniform(10, 25)) # Cold weather penalty (Darjeeling)
    else:
        y_dengue.append((t * 3.2) + (h * 1.1) + (r * 1.3) - 70)

dengue_rf = RandomForestRegressor(n_estimators=30, random_state=42)
dengue_rf.fit(X_train_dengue, np.array(y_dengue))

# ==========================================
# 2. INDEPENDENT MALARIA AI ENGINE
# ==========================================
np.random.seed(100)
X_train_malaria = np.random.uniform(10, 40, (2000, 3))
X_train_malaria[:, 1] = np.random.uniform(30, 95, 2000)
y_malaria = []

for t, h, r in X_train_malaria:
    if t < 18.0:
        y_malaria.append(np.random.uniform(8, 20)) # Cold weather penalty
    else:
        y_malaria.append((t * 2.5) + (h * 1.3) + (r * 1.1) - 50)

malaria_rf = RandomForestRegressor(n_estimators=30, random_state=100)
malaria_rf.fit(X_train_malaria, np.array(y_malaria))


@app.get("/")
def home():
    return {"status": "Aegis Isolated Dual-Engine ML Active."}


@app.get("/predict")
def predict(city: str = "Howrah", api_key: str = Depends(get_api_key)):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        temp_val = float(data['main']['temp'])
        humidity_val = float(data['main']['humidity'])
        clouds_val = float(data.get('clouds', {}).get('all', 0))
        
        # --- Smart Rainfall Fallback Logic ---
        if 'rain' in data and '1h' in data['rain']:
            rain_val = float(data['rain']['1h'])
        elif clouds_val > 70 and humidity_val > 78:
            # Overcast & humid condition estimation (Trace Rain)
            rain_val = round((humidity_val - 70) * 0.12, 1)
        else:
            rain_val = 0.0
    else:
        # Default fallback values if API fails
        temp_val, humidity_val, rain_val = 30.0, 85.0, 0.0
        
    features = np.array([[temp_val, humidity_val, rain_val]])
    
    # --- Independent Predictions ---
    predicted_dengue = max(10, int(dengue_rf.predict(features)[0]))
    predicted_malaria = max(10, int(malaria_rf.predict(features)[0]))

    # --- Risk Levels ---
        # --- NBA Standard Risk Thresholds (Strict & Logical) ---
    dengue_risk = "High" if predicted_dengue >= 130 else ("Medium" if predicted_dengue >= 55 else "Low")
    malaria_risk = "High" if predicted_malaria >= 130 else ("Medium" if predicted_malaria >= 50 else "Low")

        
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
