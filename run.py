from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load CPU-compatible pipeline model
model = joblib.load("final_weather_forecast_model_cpu.joblib")

# Bias correction for [temp x5, humidity x5, wind x5]
bias = np.array([
    1.5, 1.5, 1.5, 1.9, 1.4,         # temperature
    -20, -20, -20, -15, -13,         # humidity
    -0.8, -0.9, -1.0, -0.4, -0.5     # wind
])

# Input preprocessing (expects raw OpenWeatherMap JSON)
def preprocess_input(data):
    ts = data["dt"]
    dt = datetime.utcfromtimestamp(ts + data["timezone"])
    hour = dt.hour
    month = dt.month

    features = {
        "temperature": data["main"]["temp"],
        "relative_humidity": data["main"]["humidity"],
        "dew_point": data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5),
        "apparent_temperature": data["main"]["feels_like"],
        "precipitation": 0.0,
        "rain": 0.0,
        "wind_speed": data["wind"]["speed"],
        "wind_direction": data["wind"]["deg"],
        "city": data["name"],
        "hour": hour,
        "month": month
    }

    return pd.DataFrame([features])

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = preprocess_input(data)
        pred = model.predict(df)[0]
        return jsonify({
            "temperature": round(pred[0], 2),
            "humidity": round(pred[1], 2),
            "wind_speed": round(pred[2], 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict5', methods=['POST'])
def predict5():
    try:
        data = request.get_json()
        df = preprocess_input(data)
        raw = model.predict(df)[0]
        corrected = raw + bias
        reshaped = corrected.reshape(3, 5).T

        forecasts = []
        for i, row in enumerate(reshaped):
            forecasts.append({
                "hour": f"Hour +{i+1}",
                "temperature": round(row[0], 2),
                "humidity": round(row[1], 2),
                "wind_speed": round(row[2], 2)
            })

        return jsonify({"city": data["name"], "forecast": forecasts})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
