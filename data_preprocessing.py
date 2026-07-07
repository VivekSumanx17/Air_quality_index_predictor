import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_cpcb_data(file_path):
    print("Loading CPCB data...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Cities in dataset: {df['City'].unique()}")
    return df

def clean_data(df):
    print("\nCleaning data")

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values(['City', 'Date'])

    # Identify pollutant columns
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    existing_pollutants = [col for col in pollutant_cols if col in df.columns]

    print(f"Found pollutants: {existing_pollutants}")

    # Remove rows where all pollutant columns are NaN
    df = df.dropna(subset=existing_pollutants, how='all')

    # Handle missing values per city (different cities have different patterns)
    for city in df['City'].unique():
        city_mask = df['City'] == city
        for col in existing_pollutants:

            df.loc[city_mask, col] = df.loc[city_mask, col].fillna(method='ffill')

            df.loc[city_mask, col] = df.loc[city_mask, col].fillna(method='bfill')

            city_median = df.loc[city_mask, col].median()
            df.loc[city_mask, col] = df.loc[city_mask, col].fillna(city_median)

    if 'AQI' in df.columns:
        for city in df['City'].unique():
            city_mask = df['City'] == city
            df.loc[city_mask, 'AQI'] = df.loc[city_mask, 'AQI'].fillna(method='ffill')
            df.loc[city_mask, 'AQI'] = df.loc[city_mask, 'AQI'].fillna(method='bfill')
            city_aqi_median = df.loc[city_mask, 'AQI'].median()
            df.loc[city_mask, 'AQI'] = df.loc[city_mask, 'AQI'].fillna(city_aqi_median)

    print(f"After cleaning: {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df

def filter_city_data(df, city_name):

    city_df = df[df['City'].str.contains(city_name, case=False)].copy()
    print(f"\nFiltered for {city_name}: {len(city_df)} rows")
    print(f"Date range: {city_df['Date'].min()} to {city_df['Date'].max()}")
    return city_df

def remove_outliers(df, columns, method='iqr'):

    df_clean = df.copy()
    initial_rows = len(df_clean)

    for col in columns:
        if col in df_clean.columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) &
                                   (df_clean[col] <= upper_bound)]
            elif method == 'percentile':
                lower_bound = df_clean[col].quantile(0.01)
                upper_bound = df_clean[col].quantile(0.99)
                df_clean = df_clean[(df_clean[col] >= lower_bound) &
                                   (df_clean[col] <= upper_bound)]

    removed = initial_rows - len(df_clean)
    print(f"Removed {removed} outlier rows")

    return df_clean

def save_processed_data(df, output_path):

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    return output_path

def create_summary_statistics(df):
    print("\nSummary Statistics:")

    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    existing_cols = [col for col in pollutant_cols if col in df.columns]

    if existing_cols:
        print(df[existing_cols].describe())

    if 'AQI_Bucket' in df.columns:
        print("\nAQI Distribution:")
        print(df['AQI_Bucket'].value_counts())

    return

# Main execution
if __name__ == "__main__":

    INPUT_FILE = "dataset/city_day.csv"
    OUTPUT_FILE = "dataset/cleaned_air_quality.csv"

    # Load data
    df = load_cpcb_data(INPUT_FILE)

    # Clean data
    df = clean_data(df)
    CITY = "Delhi"
    df = filter_city_data(df, CITY)

    # Remove outliers from pollutant
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    existing_cols = [col for col in pollutant_cols if col in df.columns]
    if existing_cols:
        df = remove_outliers(df, existing_cols, method='percentile')

    # Create summary statistics
    create_summary_statistics(df)

    # Save data
    save_processed_data(df, OUTPUT_FILE)

    # Display sample
    print("\nSample of cleaned data:")
    print(df.head(10))