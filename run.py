from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model
model = joblib.load("weather_5hour.joblib")

# Bias correction array
bias = np.array([
    1.5, 1.5, 1.5, 1.9, 1.4,        # temperature
    -20, -20, -20, -15, -13,        # humidity
    -0.8, -0.9, -1.0, -0.4, -0.5    # wind
])

# Cities for one-hot encoding
city_list = ['Ahmedabad', 'Bengaluru', 'Chennai', 'Dehradun', 'Delhi',
             'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune']


def preprocess_input(data):
    try:
        ts = data.get("dt", 0)
        timezone_offset = data.get("timezone", 0)
        dt = datetime.utcfromtimestamp(ts + timezone_offset)
        hour = dt.hour
        month = dt.month

        main = data.get("main", {})
        wind = data.get("wind", {})
        city = data.get("name", "Unknown")

        features = {
            "temperature": main.get("temp", 0.0),
            "relative_humidity": main.get("humidity", 0.0),
            "dew_point": main.get("temp", 0.0) - ((100 - main.get("humidity", 0.0)) / 5),
            "apparent_temperature": main.get("feels_like", 0.0),
            "precipitation": 0.0,
            "rain": 0.0,
            "wind_speed": wind.get("speed", 0.0),
            "wind_direction": wind.get("deg", 0.0),
            "city": city,
            "hour": hour,
            "month": month
        }

        df = pd.DataFrame([features])

        for c in city_list:
            df[f"city_{c}"] = 1 if city == c else 0

        return df

    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")


@app.route('/')
def home():
    return render_template("page.html")


@app.route('/predict', methods=['POST'])
def predict_current():
    try:
        data = request.get_json(force=True)
        X = preprocess_input(data)

        prediction = model.predict(X)[0][:3]  # only hour 1 prediction (first of each set)
        temp, humidity, wind = prediction[0] + bias[0], prediction[5] + bias[5], prediction[10] + bias[10]

        return jsonify({
            "city": data.get("name", "Unknown"),
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "wind_speed": round(wind, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict5', methods=['POST'])
def predict_5hour_forecast():
    try:
        data = request.get_json(force=True)
        X = preprocess_input(data)

        raw_pred = model.predict(X)[0]
        corrected = raw_pred + bias
        reshaped = corrected.reshape(3, 5).T

        forecast = []
        for i in range(5):
            forecast.append({
                "hour": f"+{i+1}",
                "temperature": round(reshaped[i][0], 2),
                "humidity": round(reshaped[i][1], 2),
                "wind_speed": round(reshaped[i][2], 2)
            })

        return jsonify({
            "city": data.get("name", "Unknown"),
            "forecast": forecast
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
