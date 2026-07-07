import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

class AirQualityPredictor:
    def __init__(self, model_path="models/aqi_regression_model.pkl",
                 metadata_path="models/model_metadata.pkl"):

        print("Loading Air Quality Predictor")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please run training scripts first:")
            print("  python src/data_preprocessing.py")
            print("  python src/feature_engineering.py")
            print("  python src/train_model.py")
            return None

        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)
        self.feature_cols = self.metadata['features']
        print(f"Model loaded successfully!")
        print(f"Model performance: MAE = {self.metadata['results']['test_mae']:.2f}")

    def predict_from_values(self, pm25, pm10, no2, co, so2, o3, month=None, day_of_week=None):

        # Use current date if not provided
        if month is None:
            month = datetime.now().month
        if day_of_week is None:
            day_of_week = datetime.now().weekday()

        # Create feature dictionary
        features = {
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'CO': co,
            'SO2': so2,
            'O3': o3,
            'month': month,
            'day_of_week': day_of_week,
            'weekend': 1 if day_of_week >= 5 else 0,
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
        }

        # Add missing features with default values
        for col in self.feature_cols:
            if col not in features:
                if '_lag_' in col or '_rolling_' in col:
                    base_col = col.split('_')[0]
                    if base_col in features:
                        features[col] = features[base_col]
                    else:
                        features[col] = 0
                elif '_x_' in col or '_plus_' in col:
                    features[col] = 0
                else:
                    features[col] = 0

        X = pd.DataFrame([features])[self.feature_cols]

        aqi = self.model.predict(X)[0]

        return {
            'predicted_aqi': round(aqi, 1),
            'category': self._get_category(aqi),
            'advisory': self._get_advisory(aqi)
        }

    def predict_from_csv(self, csv_path):
        print(f"\nLoading data from {csv_path}")
        df = pd.read_csv(csv_path)

        print(f"Loaded {len(df)} rows")

        predictions = []
        for idx, row in df.iterrows():
            result = self.predict_from_values(
                pm25=row.get('PM2.5', row.get('pm25', 0)),
                pm10=row.get('PM10', row.get('pm10', 0)),
                no2=row.get('NO2', row.get('no2', 0)),
                co=row.get('CO', row.get('co', 0)),
                so2=row.get('SO2', row.get('so2', 0)),
                o3=row.get('O3', row.get('o3', 0))
            )
            predictions.append(result['predicted_aqi'])

        df['Predicted_AQI'] = predictions
        df['Category'] = [self._get_category(p) for p in predictions]

        # Save results
        output_path = csv_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")

        return df

    def _get_category(self, aqi):
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

    def _get_advisory(self, aqi):
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


def interactive_mode():
    print("\n" + "="*60)
    print("AIR QUALITY PREDICTOR - INTERACTIVE MODE")
    print("="*60)
    print("\nEnter pollutant values (press Enter to use example values):\n")

    # Example values for different scenarios
    print("Example values:")
    print("  Good air:    PM2.5=30,  PM10=50,   NO2=20,  CO=0.5, SO2=10, O3=30")
    print("  Moderate:    PM2.5=80,  PM10=120,  NO2=50,  CO=1.0, SO2=20, O3=50")
    print("  Poor:        PM2.5=150, PM10=250,  NO2=80,  CO=1.5, SO2=30, O3=70")
    print("  Severe:      PM2.5=300, PM10=400,  NO2=120, CO=2.5, SO2=50, O3=100\n")

    try:
        pm25 = float(input("Enter PM2.5 value (or press Enter for 80): ") or 80)
        pm10 = float(input("Enter PM10 value (or press Enter for 120): ") or 120)
        no2 = float(input("Enter NO2 value (or press Enter for 50): ") or 50)
        co = float(input("Enter CO value (or press Enter for 1.0): ") or 1.0)
        so2 = float(input("Enter SO2 value (or press Enter for 20): ") or 20)
        o3 = float(input("Enter O3 value (or press Enter for 50): ") or 50)

        return pm25, pm10, no2, co, so2, o3
    except:
        print("\nUsing example values for Moderate air quality")
        return 80, 120, 50, 1.0, 20, 50


def batch_mode():
    print("\n" + "="*60)
    print("BATCH PREDICTION MODE")
    print("="*60)

    csv_path = "dataset/cleaned_air_quality.csv"

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    return csv_path


# Example CSV template
def create_sample_csv():
    sample_data = {
        'PM2.5': [30, 80, 150, 250, 400],
        'PM10': [50, 120, 250, 400, 600],
        'NO2': [20, 50, 80, 120, 150],
        'CO': [0.5, 1.0, 1.5, 2.0, 3.0],
        'SO2': [10, 20, 30, 40, 50],
        'O3': [30, 50, 70, 90, 110]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv("sample_air_quality_data.csv", index=False)
    print("Created sample_air_quality_data.csv")
    return "sample_air_quality_data.csv"


# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = AirQualityPredictor()

    if predictor.model is None:
        print("\nCannot proceed without trained model.")
        exit(1)

    print("\n" + "="*60)
    print("AIR QUALITY PREDICTION SYSTEM")
    print("="*60)
    print("\nChoose prediction mode:")
    print("1. Interactive Mode (enter values manually)")
    print("2. Batch Mode (predict from CSV file)")
    print("3. Create sample CSV file")
    print("4. Test with example values")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '1':
        # Interactive mode
        pm25, pm10, no2, co, so2, o3 = interactive_mode()
        result = predictor.predict_from_values(pm25, pm10, no2, co, so2, o3)

        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nPredicted AQI: {result['predicted_aqi']}")
        print(f"Category: {result['category']}")
        print(f"\nHealth Advisory:")
        print(f"  {result['advisory']}")

    elif choice == '2':
        # Batch mode
        csv_path = batch_mode()
        if csv_path:
            df = predictor.predict_from_csv(csv_path)
            print("\nFirst 5 predictions:")
            print(df[['PM2.5', 'PM10', 'Predicted_AQI', 'Category']].head())

    elif choice == '3':
        # Create sample CSV
        csv_path = create_sample_csv()
        print(f"\nSample CSV created: {csv_path}")

    elif choice == '4':
        print("\nTesting with different air quality scenarios:\n")

        scenarios = {
            'Good Air': (30, 50, 20, 0.5, 10, 30),
            'Moderate Air': (80, 120, 50, 1.0, 20, 50),
            'Poor Air': (150, 250, 80, 1.5, 30, 70),
            'Very Poor': (250, 400, 120, 2.0, 40, 90),
            'Severe': (400, 600, 150, 3.0, 50, 110)
        }

        for scenario, values in scenarios.items():
            result = predictor.predict_from_values(*values)
            print(f"{scenario:12} → AQI: {result['predicted_aqi']:5.1f} ({result['category']})")

    else:
        print("Invalid choice")