from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

app = Flask(__name__)

# Load CPU-only trained pipeline model
model = joblib.load("final_weather_forecast_model_cpu.joblib")

# Bias correction (you can adjust these if needed)
bias = np.array([
    1.5, 1.5, 1.5, 1.9, 1.4,        # temperature
    -20, -20, -20, -15, -13,        # humidity
    -0.8, -0.9, -1.0, -0.4, -0.5    # wind
])

expected_cities = ['Ahmedabad', 'Bengaluru', 'Chennai', 'Dehradun', 'Delhi',
                   'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune']

# Preprocessing function
def preprocess_input(data):
    try:
        dt = datetime.utcfromtimestamp(data["dt"] + data["timezone"])
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
            "hour": hour,
            "month": month
        }

        df = pd.DataFrame([features])

        # Add one-hot encoding for cities
        for city in expected_cities:
            df[f'city_{city}'] = 1 if data["name"] == city else 0

        return df

    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")

# Home route
@app.route("/")
def home():
    return render_template("page.html")

# Normal single prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        X_input = preprocess_input(input_data)
        prediction = model.predict(X_input)[0][:3]  # temp, humidity, wind speed
        return jsonify({
            "city": input_data["name"],
            "temperature": round(prediction[0], 2),
            "humidity": round(prediction[1], 2),
            "wind_speed": round(prediction[2], 2)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

# 5-step prediction with bias correction
@app.route("/predict5", methods=["POST"])
def predict5():
    try:
        input_data = request.get_json()
        X_input = preprocess_input(input_data)

        raw_pred = model.predict(X_input)[0]
        corrected_pred = raw_pred + bias
        reshaped = corrected_pred.reshape(3, 5).T

        forecast = []
        for i in range(5):
            forecast.append({
                "hour": f"Hour +{i+1}",
                "temperature": round(reshaped[i][0], 2),
                "humidity": round(reshaped[i][1], 2),
                "wind_speed": round(reshaped[i][2], 2)
            })

        return jsonify({
            "city": input_data["name"],
            "forecast": forecast
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(debug=True)
