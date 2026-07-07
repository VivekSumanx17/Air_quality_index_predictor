import pandas as pd
import numpy as np

def create_time_features(df):
    df = df.copy()

    if 'Date' in df.columns:
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Seasonal features
        df['is_winter'] = ((df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)).astype(int)
        df['is_summer'] = ((df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5)).astype(int)
        df['is_monsoon'] = ((df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8) | (df['month'] == 9)).astype(
            int)

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def create_lag_features(df, columns, lags=[1, 2, 3, 7, 14, 30]):
    df = df.copy()

    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    return df


def create_rolling_features(df, columns, windows=[3, 7, 14, 30]):
    df = df.copy()

    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

    return df


def create_interaction_features(df, pollutant_cols):

    df = df.copy()

    for i in range(len(pollutant_cols)):
        for j in range(i + 1, len(pollutant_cols)):
            col1 = pollutant_cols[i]
            col2 = pollutant_cols[j]
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]

    return df


def encode_aqi_bucket(df):

    if 'AQI_Bucket' in df.columns:
        bucket_mapping = {
            'Good': 0,
            'Satisfactory': 1,
            'Moderate': 2,
            'Poor': 3,
            'Very Poor': 4,
            'Severe': 5
        }
        df['AQI_Bucket_encoded'] = df['AQI_Bucket'].map(bucket_mapping)

    return df


def prepare_features(df, target_col='AQI', use_lags=True, use_rollings=True):

    print("Creating features...")
    initial_cols = len(df.columns)

    # Sort by date first
    if 'Date' in df.columns:
        df = df.sort_values('Date')

    # Create time features
    df = create_time_features(df)

    # Encode AQI bucket
    df = encode_aqi_bucket(df)

    # Identify pollutant columns
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    existing_pollutants = [col for col in pollutant_cols if col in df.columns]

    print(f"Using pollutants: {existing_pollutants}")

    # Create lag features
    if use_lags:
        df = create_lag_features(df, existing_pollutants, lags=[1, 2, 3, 7, 14])

    # Create rolling features
    if use_rollings:
        df = create_rolling_features(df, existing_pollutants, windows=[3, 7, 14])

    # Create interaction features
    df = create_interaction_features(df, existing_pollutants)

    # Drop rows with NaN (from lag/rolling features)
    df = df.dropna()

    print(f"Created {len(df.columns) - initial_cols} new features")
    print(f"Total features: {len(df.columns)}")
    print(f"Final dataset size: {len(df)} rows")

    return df


def split_train_test(df, target_col='AQI', test_size=0.2, val_size=0.1):

    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Define features
    exclude_cols = ['Date', 'City', 'AQI_Bucket', target_col]
    if target_col == 'AQI':
        exclude_cols.append('AQI_Bucket_encoded')

    feature_cols = [col for col in train_df.columns
                    if col not in exclude_cols
                    and pd.api.types.is_numeric_dtype(train_df[col])]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col] if target_col in train_df.columns else None

    X_val = val_df[feature_cols]
    y_val = val_df[target_col] if target_col in val_df.columns else None

    X_test = test_df[feature_cols]
    y_test = test_df[target_col] if target_col in test_df.columns else None

    print(f"\nData Split:")
    print(f" Training ({train_df['Date'].min()} to {train_df['Date'].max()}): {len(X_train)} rows")
    print(f" Validation ({val_df['Date'].min()} to {val_df['Date'].max()}): {len(X_val)} rows")
    print(f" Test ({test_df['Date'].min()} to {test_df['Date'].max()}): {len(X_test)} rows")
    print(f" Features: {len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, train_df, val_df, test_df


# Main execution
if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("dataset/cleaned_air_quality.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    print(f"Loaded {len(df)} rows")
    print(f"Cities: {df['City'].unique()}")

    # Create features
    df_features = prepare_features(df, target_col='AQI')

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, train_df, val_df, test_df = split_train_test(
        df_features, target_col='AQI'
    )

    # Save prepared data
    import joblib

    joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test, feature_cols),
                "dataset/prepared_data.pkl")

    df_features.to_csv("dataset/features_data.csv", index=False)

    print("\nSaved prepared data to ../data/prepared_data.pkl")

    print(f"\nFeature columns sample: {feature_cols[:10]}")