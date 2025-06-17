from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load the trained pipeline model
model = joblib.load("final_weather_forecast_model_cpu.joblib")

# Bias correction (for /predict5)
bias = np.array([
    1.5, 1.5, 1.5, 1.9, 1.4,        # temperature
    -20, -20, -20, -15, -13,        # humidity
    -0.8, -0.9, -1.0, -0.4, -0.5    # wind speed
])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('page.html')


def preprocess_json(data):
    """Preprocess OpenWeatherMap JSON into DataFrame for prediction"""
    ts = data.get("dt")
    timezone = data.get("timezone", 0)
    dt = datetime.utcfromtimestamp(ts + timezone)
    hour = dt.hour
    month = dt.month

    try:
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
        df = pd.DataFrame([features])
        return df
    except KeyError as e:
        return f"Missing key: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = preprocess_json(data)

        if isinstance(df, str):
            return jsonify({'error': df}), 400

        prediction = model.predict(df)[0]  # [temp, humidity, wind]
        result = {
            "Temperature (°C)": round(prediction[0], 2),
            "Humidity (%)": round(prediction[1], 2),
            "Wind Speed (m/s)": round(prediction[2], 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict5', methods=['POST'])
def predict5():
    try:
        data = request.get_json()
        df = preprocess_json(data)

        if isinstance(df, str):
            return jsonify({'error': df}), 400

        raw_pred = model.predict(df)[0]
        corrected = raw_pred + bias
        reshaped = corrected.reshape(3, 5).T

        forecast = []
        for i, hour in enumerate(reshaped):
            forecast.append({
                "hour": f"+{i+1}",
                "Temperature (°C)": round(hour[0], 2),
                "Humidity (%)": round(hour[1], 2),
                "Wind Speed (m/s)": round(hour[2], 2)
            })

        return jsonify({
            "city": data["name"],
            "forecast": forecast
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
