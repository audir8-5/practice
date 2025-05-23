from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("ridge_weather_model.joblib")

@app.route('/')
def home():
    return "Weather Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        dew_point = data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5)
        apparent_temp = data["main"]["feels_like"]
        precipitation = 0.0
        rain = 0.0
        wind_direction = data["wind"]["deg"]
        timestamp = data["dt"]
        dt_obj = datetime.utcfromtimestamp(timestamp + data["timezone"])
        hour = dt_obj.hour
        month = dt_obj.month
        city = data["name"]

        input_df = pd.DataFrame([{
            "dew_point": dew_point,
            "apparent_temperature": apparent_temp,
            "precipitation": precipitation,
            "rain": rain,
            "wind_direction": wind_direction,
            "hour": hour,
            "month": month,
            "city": city
        }])

        prediction = model.predict(input_df)[0]

        result = {
            "predicted_temperature": round(prediction[0], 2),
            "predicted_relative_humidity": round(prediction[1], 2),
            "predicted_wind_speed": round(prediction[2], 2)
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})
