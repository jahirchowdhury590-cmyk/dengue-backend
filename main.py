from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import requests
import pickle
import numpy as np
import os  # <-- এটি নতুন যুক্ত হলো

app = FastAPI()

# ... (CORS এবং API Key System-এর আগের কোডগুলো হুবহু থাকবে) ...

# এখানে সরাসরি Key না লিখে আমরা Environment Variable ব্যবহার করছি
OPENWEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

@app.get("/")
def home():
    return {"status": "AegisDengue API is running securely!"}

@app.get("/predict")
def predict(city: str = "Howrah", api_key: str = Depends(get_api_key)):
    
    # URL-এর শেষে &units=metric যোগ করা হয়েছে যাতে তাপমাত্রা সেলসিয়াসে আসে
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    
    if res.status_code == 200:
        data = res.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        rain = data.get('rain', {}).get('1h', 0)
    else:
        # API সাময়িক কাজ না করলে এই ডামি ডেটাগুলো যাবে
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
