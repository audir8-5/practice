from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime
import traceback

app = Flask(__name__)

# Load model (ensure it's CPU-compatible, no GPU dependencies)
model = joblib.load("final_weather_forecast_model_cpu.joblib")

# Preprocessing function
def preprocess_input(data):
    try:
        # Fallback if timezone is missing
        try:
            ts = data["dt"] + data.get("timezone", 0)
            dt = datetime.utcfromtimestamp(ts)
        except Exception:
            dt = datetime.utcnow()

        features = {
            "temperature": data["main"]["temp"],
            "relative_humidity": data["main"]["humidity"],
            "dew_point": data["main"].get("dew_point", data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5)),
            "apparent_temperature": data["main"].get("feels_like", data["main"]["temp"]),
            "precipitation": data.get("rain", {}).get("1h", 0),
            "rain": 1 if "rain" in data else 0,
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "city": data["name"],
            "hour": dt.hour,
            "month": dt.month
        }

        return pd.DataFrame([features])
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")

@app.route("/")
def home():
    return render_template("page.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        X_input = preprocess_input(input_data)
        prediction = model.predict(X_input)[0][:3]  # temp, humidity, wind speed
        return jsonify({
            "temperature": round(prediction[0], 2),
            "humidity": round(prediction[1], 2),
            "wind_speed": round(prediction[2], 2)
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })

@app.route("/predict5", methods=["POST"])
def predict5():
    try:
        input_data = request.get_json()
        base_input = preprocess_input(input_data)

        predictions = []
        current_input = base_input.copy()

        for hour in range(5):
            y_pred = model.predict(current_input)[0]
            predictions.append({
                "hour": hour + 1,
                "temperature": round(y_pred[0], 2),
                "humidity": round(y_pred[1], 2),
                "wind_speed": round(y_pred[2], 2)
            })
            # Autoregressive update
            current_input["temperature"] = y_pred[0]
            current_input["relative_humidity"] = y_pred[1]
            current_input["wind_speed"] = y_pred[2]

        return jsonify({
            "city": input_data["name"],
            "forecast": predictions
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })

if __name__ == "__main__":
    app.run(debug=True)
