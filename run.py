from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load CPU-compatible trained pipeline model
MODEL_PATH = "final_weather_forecast_model_cpu.joblib"
model = joblib.load(MODEL_PATH)

# Bias correction: [temp x5, humidity x5, wind x5]
bias = np.array([
    1.5, 1.5, 1.5, 1.9, 1.4,        # temperature
    -20, -20, -20, -15, -13,        # humidity
    -0.8, -0.9, -1.0, -0.4, -0.5    # wind
])

# Expected cities (must match model's training one-hot columns)
expected_cities = [
    'Ahmedabad', 'Bengaluru', 'Chennai', 'Dehradun', 'Delhi',
    'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune'
]

# ------------------- Preprocessing Function -------------------
def preprocess_input(data):
    try:
        ts = data["dt"]
        tz = data.get("timezone", 0)
        dt = datetime.utcfromtimestamp(ts + tz)
        hour = dt.hour
        month = dt.month
        city = data.get("name", "Unknown")

        # Extract features
        features = {
            "temperature": data["main"]["temp"],
            "relative_humidity": data["main"]["humidity"],
            "dew_point": data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5),
            "apparent_temperature": data["main"]["feels_like"],
            "precipitation": 0.0,
            "rain": 0.0,
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "hour": hour,
            "month": month
        }

        df = pd.DataFrame([features])

        # Add one-hot encoded city columns
        for c in expected_cities:
            df[f'city_{c}'] = 1 if city == c else 0

        return df, city

    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

# ------------------- Routes -------------------

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict_current():
    try:
        data = request.get_json()
        df, city = preprocess_input(data)
        preds = model.predict(df)[0][:3]  # Only first hour
        preds = np.round(preds, 2)
        return jsonify({
            "city": city,
            "hour": "current",
            "temperature": float(preds[0]),
            "humidity": float(preds[1]),
            "wind_speed": float(preds[2])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict5', methods=['POST'])
def predict_5hour():
    try:
        data = request.get_json()
        df, city = preprocess_input(data)
        raw_pred = model.predict(df)[0]
        corrected = raw_pred + bias
        reshaped = corrected.reshape(3, 5).T

        results = []
        for i, row in enumerate(reshaped):
            results.append({
                "hour": f"Hour +{i+1}",
                "temperature": round(row[0], 2),
                "humidity": round(row[1], 2),
                "wind_speed": round(row[2], 2)
            })

        return jsonify({
            "city": city,
            "forecast": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------------- Run App -------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
