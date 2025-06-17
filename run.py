from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load model and feature columns
model_path = "ridge_weather_model.joblib"
loaded = joblib.load(model_path)

if isinstance(loaded, tuple):
    pipeline, feature_columns = loaded
else:
    pipeline = loaded
    # Default expected columns
    feature_columns = [
        'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
        'precipitation', 'rain', 'wind_speed', 'wind_direction',
        'city', 'hour', 'month'
    ]

@app.route('/')
def home():
    return "✅ Weather Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # --- Extract and process OpenWeatherMap input ---
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

        # Build DataFrame
        input_dict = {
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

        input_df = pd.DataFrame([input_dict])

        # Align columns
        input_df = input_df[feature_columns]

        # Prediction
        y_pred = pipeline.predict(input_df)[0]

        result = {
            "predicted_temperature": round(y_pred[0], 2),
            "predicted_relative_humidity": round(y_pred[1], 2),
            "predicted_wind_speed": round(y_pred[2], 2)
        }

        return jsonify(result)
    
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
