import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta


class AirQualityPredictor:
    def __init__(self, model_path="models/aqi_regression_model.pkl",
                 metadata_path="models/model_metadata.pkl"):
        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)
        self.feature_cols = self.metadata['features']

    def predict_from_live_data(self, sensor_data):
        # Create feature vector
        features = {}

        for col in self.feature_cols:
            if col in sensor_data:
                features[col] = sensor_data[col]
            elif '_lag_' in col or '_rolling_' in col or '_x_' in col or '_plus_' in col:

                base_col = col.split('_')[0]
                if base_col in sensor_data:
                    features[col] = sensor_data[base_col]
                else:
                    features[col] = 0
            else:
                # Time features
                if col == 'hour':
                    features[col] = datetime.now().hour
                elif col == 'day_of_week':
                    features[col] = datetime.now().weekday()
                elif col == 'month':
                    features[col] = datetime.now().month
                elif col == 'weekend':
                    features[col] = 1 if datetime.now().weekday() >= 5 else 0
                elif col == 'hour_sin':
                    hour = datetime.now().hour
                    features[col] = np.sin(2 * np.pi * hour / 24)
                elif col == 'hour_cos':
                    hour = datetime.now().hour
                    features[col] = np.cos(2 * np.pi * hour / 24)
                elif col == 'month_sin':
                    month = datetime.now().month
                    features[col] = np.sin(2 * np.pi * month / 12)
                elif col == 'month_cos':
                    month = datetime.now().month
                    features[col] = np.cos(2 * np.pi * month / 12)
                else:
                    features[col] = 0
        X = pd.DataFrame([features])[self.feature_cols]

        prediction = self.model.predict(X)[0]

        return prediction

    def predict_next_24h(self, current_data, historical_data=None):
        predictions = []
        timestamps = []

        current_time = datetime.now()

        for hour in range(24):
            future_time = current_time + timedelta(hours=hour)
            timestamps.append(future_time)

            pred = self.predict_from_live_data(current_data)
            predictions.append(pred)

        results = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_aqi': predictions
        })

        results['category'] = results['predicted_aqi'].apply(self._aqi_category)

        return results

    def _aqi_category(self, aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    def get_health_advisory(self, aqi):
        if aqi <= 50:
            return "Air quality is good. Perfect for outdoor activities."
        elif aqi <= 100:
            return "Air quality is satisfactory. Sensitive individuals should limit prolonged outdoor exertion."
        elif aqi <= 200:
            return "Moderate air quality. Unusually sensitive people should reduce prolonged outdoor exertion."
        elif aqi <= 300:
            return "Poor air quality. Everyone may begin to experience health effects."
        elif aqi <= 400:
            return "Very Poor air quality. Health alert: everyone may experience more serious health effects."
        else:
            return "Severe air quality. Health warning of emergency conditions. Stay indoors."


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = AirQualityPredictor()

    # Simulate live sensor data
    live_data = {
        'PM2.5': 120,
        'PM10': 180,
        'NO2': 45,
        'CO': 1.2,
        'SO2': 15,
        'O3': 35
    }

    print("Live AQI Prediction")
    print("=" * 40)

    # Single prediction
    current_aqi = predictor.predict_from_live_data(live_data)
    print(f"\nCurrent Predicted AQI: {current_aqi:.1f}")
    print(f"  Category: {predictor._aqi_category(current_aqi)}")
    print(f"\nHealth Advisory:")
    print(f"  {predictor.get_health_advisory(current_aqi)}")

    # 24-hour forecast
    print("\n24-Hour Forecast:")
    print("-" * 40)
    forecast = predictor.predict_next_24h(live_data)
    print(forecast.head(8).to_string(index=False))

    # Save forecast
    forecast.to_csv("models/forecast_24h.csv", index=False)
    print("\nSaved 24-hour forecast to models/forecast_24h.csv")