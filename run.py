from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json

app = Flask(__name__)

# Load model pipeline
model_pipeline = joblib.load("weather_5hour.joblib")  # Adjust path if needed

# Bias (customized for Mumbai, update as needed)
bias = np.array([
    -1.5, -1.6, -2.4, -2.6, -1.7,
    +12, +12, +15, +14, +13,
    -4.7, -4.3, -4.1, -4.3, -4.0
])

# City one-hot list
expected_cities = ['Ahmedabad', 'Bengaluru', 'Chennai', 'Dehradun', 'Delhi',
                   'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune']

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

    df = pd.DataFrame([features])

    for city in expected_cities:
        df[f'city_{city}'] = 1 if data["name"] == city else 0

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    error = None

    if request.method == 'POST':
        try:
            json_input = request.form['json_input']
            data = json.loads(json_input)
            df = preprocess_input(data)
            raw_pred = model_pipeline.predict(df)[0]
            corrected = raw_pred + bias
            reshaped = corrected.reshape(3, 5).T

            forecast = [
                {"hour": f"Hour +{i+1}", "temperature": round(row[0], 2),
                 "humidity": round(row[1], 2), "wind_speed": round(row[2], 2)}
                for i, row in enumerate(reshaped)
            ]
        except Exception as e:
            error = str(e)

    return render_template('page.html', forecast=forecast, error=error)

@app.route('/predict5', methods=['POST'])
def predict5():
    try:
        data = request.get_json()
        df = preprocess_input(data)
        raw_pred = model_pipeline.predict(df)[0]
        corrected = raw_pred + bias
        reshaped = corrected.reshape(3, 5).T
        forecast = [
            {"hour": f"+{i+1}", "temperature": round(row[0], 2),
             "humidity": round(row[1], 2), "wind_speed": round(row[2], 2)}
            for i, row in enumerate(reshaped)
        ]
        return jsonify({"city": data["name"], "forecast_5h": forecast})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
