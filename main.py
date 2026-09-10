from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pickle
import numpy as np

app = FastAPI()

# Lovable ফ্রন্টএন্ড থেকে API কল করার জন্য CORS অন করা হলো
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ১. এআই মডেল লোড
with open('dengue_model.pkl', 'rb') as f:
    model = pickle.load(f)

# আপনার OpenWeather API Key এখানে বসান
OPENWEATHER_API_KEY ="f99600fc4818c022f608cf96a4f348ba"

@app.get("/")
def home():
    return {"status": "Dengue Outbreak Prediction API is running!"}

@app.get("/predict")
def predict(city: str = "Howrah"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        rain = data.get('rain', {}).get('1h', 0)
    else:
        # কোনো কারণে এপিআই কাজ না করলে ডিফল্ট মান
        temp, humidity, rain = 28.0, 80.0, 2.0

    features = np.array([[temp, humidity, rain]])
    predicted_cases = max(0, int(model.predict(features)[0]))
    
    if predicted_cases > 50:
        risk = "High"
    elif predicted_cases > 20:
        risk = "Medium"
    else:
        risk = "Low"
    
    return {
        "city": city,
        "temperature": temp,
        "humidity": humidity,
        "rainfall": rain,
        "predicted_cases": predicted_cases,
        "risk_level": risk
    }
