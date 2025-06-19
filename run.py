from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__, template_folder='templates')

# ✅ Correct and production-safe CORS config
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)



# Load model (pipeline, feature_columns tuple)
model_path = "ridge_weather_model.joblib"
loaded = joblib.load(model_path)

if isinstance(loaded, tuple):
    pipeline, feature_columns = loaded
else:
    pipeline = loaded
    feature_columns = [
        'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
        'precipitation', 'rain', 'wind_speed', 'wind_direction',
        'city', 'hour', 'month'
    ]

@app.route('/')
def index():
    return render_template('page.html')

def extract_features(data):
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    dew_point = temp - ((100 - humidity) / 5)
    apparent_temp = data["main"]["feels_like"]
    precipitation = 0.0
    rain = 0.0
    wind_speed = data["wind"]["speed"]
    wind_direction = data["wind"]["deg"]
    timestamp = data["dt"]
    timezone_offset = data.get("timezone", 0)
    city = data["name"]

    dt_obj = datetime.utcfromtimestamp(timestamp + timezone_offset)
    hour = dt_obj.hour
    month = dt_obj.month

    return {
        "temperature": temp,
        "relative_humidity": humidity,
        "dew_point": dew_point,
        "apparent_temperature": apparent_temp,
        "precipitation": precipitation,
        "rain": rain,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "city": city,
        "hour": hour,
        "month": month
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_dict = extract_features(data)
        input_df = pd.DataFrame([input_dict])[feature_columns]

        y_pred = pipeline.predict(input_df)[0]

        result = {
            "predicted_temperature": round(y_pred[0], 2),
            "predicted_relative_humidity": round(y_pred[1], 2),
            "predicted_wind_speed": round(y_pred[2], 2)
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict5', methods=['POST'])
def predict5():
    try:
        data = request.get_json()
        base_features = extract_features(data)

        predictions = []
        current_input = base_features.copy()

        for step in range(5):
            input_df = pd.DataFrame([current_input])[feature_columns]
            y_pred = pipeline.predict(input_df)[0]

            # --- Safeguard: clamp predictions ---
            temp = max(min(y_pred[0], 55), -30)  # realistic temp range
            rh   = max(min(y_pred[1], 100), 0)   # humidity always between 0-100
            wind = max(min(y_pred[2], 100), 0)   # wind speed in km/h (or m/s)

            # Optional: add tiny noise to prevent identical flat predictions
            # import random
            # temp += random.uniform(-0.3, 0.3)
            # rh   += random.uniform(-0.5, 0.5)
            # wind += random.uniform(-0.2, 0.2)

            predictions.append({
                "hour_ahead": step + 1,
                "predicted_temperature": round(temp, 2),
                "predicted_relative_humidity": round(rh, 2),
                "predicted_wind_speed": round(wind, 2)
            })

            # Update current_input for next prediction (autoregressive logic)
            current_input["temperature"] = temp
            current_input["relative_humidity"] = rh
            current_input["wind_speed"] = wind
            current_input["dew_point"] = temp - ((100 - rh) / 5)
            current_input["apparent_temperature"] = temp

            current_input["hour"] = (current_input["hour"] + 1) % 24
            if current_input["hour"] == 0:
                current_input["month"] = (current_input["month"] % 12) + 1

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
