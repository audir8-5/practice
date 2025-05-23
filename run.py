from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# Load model and expected features
model = joblib.load("multioutput_weather_model.joblib")
with open("features.json") as f:
    feature_order = json.load(f)

# Load city list from the one-hot encoded city columns in features.json
expected_cities = [f.replace("city_", "") for f in feature_order if f.startswith("city_")]

@app.route("/")
def home():
    return render_template("page.html")

@app.route("/api/predict", methods=["POST"])
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

        if city not in expected_cities:
            return jsonify({"error": f"City '{city}' not in trained cities. Expected one of: {expected_cities}"}), 400

        input_data = {
            "dew_point": dew_point,
            "apparent_temperature": apparent_temp,
            "precipitation": precipitation,
            "rain": rain,
            "wind_direction": wind_direction,
            "hour": hour,
            "month": month,
            f"city_{city}": 1
        }

        # Set 0 for all missing one-hot columns
        for f in feature_order:
            if f not in input_data:
                input_data[f] = 0

        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order]  # Reorder columns exactly

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_temperature": round(prediction[0], 2),
            "predicted_relative_humidity": round(prediction[1], 2),
            "predicted_wind_speed": round(prediction[2], 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
