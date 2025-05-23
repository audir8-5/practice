from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load your trained Ridge model only once
model = joblib.load("multioutput_weather_model.joblib")

# Extract city encoder to validate input cities
city_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
expected_cities = city_encoder.categories_[0].tolist()

@app.route("/")
def home():
    return render_template("page.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # Extract fields from OpenWeatherMap JSON
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

        # Prepare input
        input_data = {
            "dew_point": dew_point,
            "apparent_temperature": apparent_temp,
            "precipitation": precipitation,
            "rain": rain,
            "wind_direction": wind_direction,
            "hour": hour,
            "month": month,
            "city": city
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_temperature": round(prediction[0], 2),
            "predicted_relative_humidity": round(prediction[1], 2),
            "predicted_wind_speed": round(prediction[2], 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route("/api/predict5", methods=["POST"])
# def predict_5_hours():
#     data = request.get_json()
#     try:
#         dew_point = data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5)
#         apparent_temp = data["main"]["feels_like"]
#         precipitation = 0.0
#         rain = 0.0
#         wind_direction = data["wind"]["deg"]
#         base_timestamp = data["dt"]
#         timezone_offset = data["timezone"]
#         city = data["name"]

#         if city not in expected_cities:
#             return jsonify({"error": f"City '{city}' not in trained cities. Expected one of: {expected_cities}"}), 400

#         forecast = []
#         for i in range(5):
#             future_dt = datetime.utcfromtimestamp(base_timestamp + timezone_offset) + timedelta(hours=i)
#             hour = future_dt.hour
#             month = future_dt.month

#             input_data = {
#                 "dew_point": dew_point,
#                 "apparent_temperature": apparent_temp,
#                 "precipitation": precipitation,
#                 "rain": rain,
#                 "wind_direction": wind_direction,
#                 "hour": hour,
#                 "month": month,
#                 "city": city
#             }

#             input_df = pd.DataFrame([input_data])
#             prediction = model.predict(input_df)[0]

#             forecast.append({
#                 "hour": hour,
#                 "temperature": round(prediction[0], 2),
#                 "relative_humidity": round(prediction[1], 2),
#                 "wind_speed": round(prediction[2], 2)
#             })

#         return jsonify({ "forecast": forecast })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
