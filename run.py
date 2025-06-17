from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__, template_folder='templates')

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

            predictions.append({
                "hour_ahead": step + 1,
                "predicted_temperature": round(y_pred[0], 2),
                "predicted_relative_humidity": round(y_pred[1], 2),
                "predicted_wind_speed": round(y_pred[2], 2)
            })

            # Update for next step (autoregressive)
            current_input["temperature"] = y_pred[0]
            current_input["relative_humidity"] = y_pred[1]
            current_input["wind_speed"] = y_pred[2]
            current_input["dew_point"] = current_input["temperature"] - ((100 - current_input["relative_humidity"]) / 5)
            current_input["apparent_temperature"] = current_input["temperature"]

            current_input["hour"] = (current_input["hour"] + 1) % 24
            if current_input["hour"] == 0:
                current_input["month"] = (current_input["month"] % 12) + 1

        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
