import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌬️ Air Quality Prediction System")
st.markdown("### Predict Air Quality Index (AQI) from Pollutant Levels")

# Function to load model
@st.cache_resource
def load_model():

    try:
        # Look for model in different locations
        model_paths = [
            "models/aqi_regression_model.pkl",
        ]
        model = None
        model_path_used = None

        for mp in model_paths:
            if os.path.exists(mp):
                model = joblib.load(mp)
                model_path_used = mp
                break

        if model is None:
            st.warning("No trained model found. Using simplified calculation method.")
            st.info("To train the model, run: python src/03_train_model.py")
            return None

        st.success(f"Model loaded from: {model_path_used}")
        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to calculate AQI using simplified method
def calculate_aqi_simple(pm25, pm10, no2, co, so2, o3):

    # Sub-indices for each pollutant
    def calc_subindex(value, breakpoints, indices):
        for i in range(len(breakpoints)-1):
            if value <= breakpoints[i+1]:
                return ((indices[i+1] - indices[i]) /
                       (breakpoints[i+1] - breakpoints[i])) * (value - breakpoints[i]) + indices[i]
        return indices[-1]

    # PM2.5 breakpoints (μg/m³)
    pm25_breakpoints = [0, 30, 60, 90, 120, 250]
    pm25_indices = [0, 50, 100, 200, 300, 400]
    ipm25 = calc_subindex(pm25, pm25_breakpoints, pm25_indices)

    # PM10 breakpoints (μg/m³)
    pm10_breakpoints = [0, 50, 100, 250, 350, 430]
    pm10_indices = [0, 50, 100, 200, 300, 400]
    ipm10 = calc_subindex(pm10, pm10_breakpoints, pm10_indices)

    # NO2 breakpoints (μg/m³)
    no2_breakpoints = [0, 40, 80, 180, 280, 400]
    no2_indices = [0, 50, 100, 200, 300, 400]
    ino2 = calc_subindex(no2, no2_breakpoints, no2_indices)

    # CO breakpoints (mg/m³)
    co_breakpoints = [0, 1, 2, 10, 17, 34]
    co_indices = [0, 50, 100, 200, 300, 400]
    ico = calc_subindex(co, co_breakpoints, co_indices)

    # O3 breakpoints (μg/m³)
    o3_breakpoints = [0, 50, 100, 168, 208, 748]
    o3_indices = [0, 50, 100, 200, 300, 400]
    io3 = calc_subindex(o3, o3_breakpoints, o3_indices)

    # SO2 breakpoints (μg/m³)
    so2_breakpoints = [0, 40, 80, 380, 800, 1600]
    so2_indices = [0, 50, 100, 200, 300, 400]
    iso2 = calc_subindex(so2, so2_breakpoints, so2_indices)

    # AQI is the maximum of all sub-indices
    aqi = max(ipm25, ipm10, ino2, ico, iso2, io3)

    return aqi

def get_category(aqi):

    if aqi <= 50:
        return "Good", "😊", "#00E400"
    elif aqi <= 100:
        return "Satisfactory", "🙂", "#FFFF00"
    elif aqi <= 200:
        return "Moderate", "😐", "#FF7E00"
    elif aqi <= 300:
        return "Poor", "😷", "#FF0000"
    elif aqi <= 400:
        return "Very Poor", "🤢", "#8F3F97"
    else:
        return "Severe", "💀", "#7E0023"

def get_advisory(aqi):
    if aqi <= 50:
        return "✅ **Good** - Air quality is satisfactory. Perfect for outdoor activities."
    elif aqi <= 100:
        return "⚠️ **Satisfactory** - Acceptable air quality. Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 200:
        return "⚠️ **Moderate** - Unusually sensitive people should reduce prolonged outdoor exertion."
    elif aqi <= 300:
        return "🚨 **Poor** - Everyone may begin to experience health effects. Limit outdoor activities."
    elif aqi <= 400:
        return "🚨 **Very Poor** - Health alert. Everyone may experience serious health effects."
    else:
        return "☠️ **Severe** - Health warning of emergency conditions. Stay indoors."


model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Pollutant Values")
    st.markdown("---")

    # Create input fields in two sub-columns
    col_a, col_b = st.columns(2)

    with col_a:
        pm25 = st.number_input(
            "PM2.5 (μg/m³)",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            step=5.0,
            help="Fine particulate matter - most harmful pollutant"
        )

        no2 = st.number_input(
            "NO2 (μg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            help="Nitrogen dioxide - from vehicle emissions"
        )

        so2 = st.number_input(
            "SO2 (μg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=5.0,
            help="Sulfur dioxide - from industrial activities"
        )

    with col_b:
        pm10 = st.number_input(
            "PM10 (μg/m³)",
            min_value=0.0,
            max_value=1000.0,
            value=120.0,
            step=5.0,
            help="Coarse particulate matter"
        )

        co = st.number_input(
            "CO (mg/m³)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Carbon monoxide - from incomplete combustion"
        )

        o3 = st.number_input(
            "O3 (μg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            help="Ozone - ground-level pollutant"
        )

    st.markdown("---")

    # Example buttons
    st.markdown("### 🎲 Quick Examples")
    ex_col1, ex_col2, ex_col3 = st.columns(3)

    with ex_col1:
        if st.button("🏞️ Clean Air", use_container_width=True):
            st.session_state.example = "clean"

    with ex_col2:
        if st.button("🏙️ Urban Air", use_container_width=True):
            st.session_state.example = "urban"

    with ex_col3:
        if st.button("🏭 Polluted", use_container_width=True):
            st.session_state.example = "polluted"

    # Handle example selections
    if 'example' in st.session_state:
        if st.session_state.example == "clean":
            pm25 = 25
            pm10 = 40
            no2 = 15
            co = 0.4
            so2 = 8
            o3 = 25
            st.rerun()
        elif st.session_state.example == "urban":
            pm25 = 80
            pm10 = 120
            no2 = 50
            co = 1.0
            so2 = 20
            o3 = 50
            st.rerun()
        elif st.session_state.example == "polluted":
            pm25 = 200
            pm10 = 300
            no2 = 100
            co = 2.0
            so2 = 40
            o3 = 80
            st.rerun()

    st.markdown("---")

    # Predict button
    predict_button = st.button("🔮 Predict AQI", type="primary", use_container_width=True)

with col2:
    st.subheader("📈 Prediction Result")
    st.markdown("---")

    # Make prediction when button is clicked
    if predict_button:
        with st.spinner("Calculating AQI..."):
            # Try to use ML model first
            if model is not None:
                try:
                    # Prepare features
                    month = datetime.now().month
                    day_of_week = datetime.now().weekday()

                    features = pd.DataFrame([[
                        pm25, pm10, no2, co, so2, o3,
                        month, day_of_week,
                        1 if day_of_week >= 5 else 0,
                        np.sin(2 * np.pi * month / 12),
                        np.cos(2 * np.pi * month / 12),
                        np.sin(2 * np.pi * day_of_week / 7),
                        np.cos(2 * np.pi * day_of_week / 7)
                    ]])

                    aqi = model.predict(features)[0]
                except:
                    aqi = calculate_aqi_simple(pm25, pm10, no2, co, so2, o3)
            else:
                aqi = calculate_aqi_simple(pm25, pm10, no2, co, so2, o3)

            # Store in session state
            st.session_state.aqi = aqi
            st.session_state.show_result = True

    # Display result
    if st.session_state.get('show_result', False) and 'aqi' in st.session_state:
        aqi = st.session_state.aqi
        category, emoji, color = get_category(aqi)

        # Display AQI in big numbers
        st.markdown(f"""
        <div style='text-align: center; padding: 30px; border-radius: 15px; background-color: {color}20;'>
            <h1 style='font-size: 72px; margin: 0;'>{emoji}</h1>
            <h1 style='font-size: 64px; margin: 10px 0; color: {color};'>{aqi:.1f}</h1>
            <h2 style='margin: 0; color: {color};'>{category}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Health advisory
        st.info(get_advisory(aqi))

        # Show input values used
        with st.expander("📊 Input Values Used"):
            st.write(f"PM2.5: {pm25} μg/m³")
            st.write(f"PM10: {pm10} μg/m³")
            st.write(f"NO2: {no2} μg/m³")
            st.write(f"CO: {co} mg/m³")
            st.write(f"SO2: {so2} μg/m³")
            st.write(f"O3: {o3} μg/m³")

        # Add a reset button
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.show_result = False
            st.rerun()

    else:
        st.info("👈 Enter pollutant values and click **Predict AQI** to see results")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    ### 🌬️ Air Quality Index (AQI)
    
    AQI tells you how clean or polluted your air is.
    
    ### 📊 AQI Categories
    | AQI Range | Category | Color |
    |-----------|----------|-------|
    | 0-50 | Good | 🟢 Green |
    | 51-100 | Satisfactory | 🟡 Yellow |
    | 101-200 | Moderate | 🟠 Orange |
    | 201-300 | Poor | 🔴 Red |
    | 301-400 | Very Poor | 🟣 Purple |
    | 400+ | Severe | 🟤 Maroon |
    
    ### 💡 Tips
    - Higher AQI = More pollution
    - Check AQI before outdoor activities
    - Use masks when AQI > 200
    - Stay indoors when AQI > 300
    """)

    st.markdown("---")
    st.markdown("### 📈 Model Info")
    if model is not None:
        st.success("✅ ML Model Active")
        st.caption("Using trained XGBoost model")
    else:
        st.warning("⚠️ Using Simplified Calculation")
        st.caption("Train model for better accuracy")

    st.markdown("---")
    st.markdown("### 🏥 Health Guide")
    with st.expander("View Health Recommendations"):
        st.markdown("""
        **Good (0-50)**
        - Ideal air quality
        - No precautions needed
        
        **Satisfactory (51-100)**  
        - Acceptable air quality
        - Sensitive people may reduce outdoor activity
        
        **Moderate (101-200)**
        - Unusually sensitive people should limit exposure
        
        **Poor (201-300)**
        - Everyone may feel health effects
        - Avoid prolonged outdoor activity
        
        **Very Poor (301-400)**
        - Health alert for everyone
        - Wear N95 masks if going out
        
        **Severe (400+)**
        - Emergency conditions
        - Stay indoors, use air purifiers
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Powered by Machine Learning | Data Source: CPCB India | For educational purposes only
</div>
""", unsafe_allow_html=True)