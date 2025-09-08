# --- at top (after imports) ---
import time
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xgb_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH.open("rb"))

pipe = load_model()

# keep a place to store the current prediction
if "pred" not in st.session_state:
    st.session_state.pred = None

def predict_from_state():
    row = pd.DataFrame([{
        "car_brand": st.session_state.car_brand,
        "car_model": st.session_state.car_model,
        "mileage_km": st.session_state.mileage_km,
        "engine_size_cc": st.session_state.engine_size_cc,
        "transmission": st.session_state.transmission,
        "fuel_type": st.session_state.fuel_type,
        "steering_type": st.session_state.steering_type,
        "drive_train": st.session_state.drive_train,
        "no_of_seats": st.session_state.no_of_seats,
        "no_of_doors": st.session_state.no_of_doors,
        "body_type": st.session_state.body_type,
        "Power Steering": int(st.session_state.Power_Steering),
        "Air Conditioner": int(st.session_state.Air_Conditioner),
        "Navigation": int(st.session_state.Navigation),
        "Air Bag": int(st.session_state.Air_Bag),
        "Anti-Lock Brake System": int(st.session_state.ABS),
        "Fog Lights": int(st.session_state.Fog_Lights),
        "Power Windows": int(st.session_state.Power_Windows),
        "Alloy Wheels": int(st.session_state.Alloy_Wheels),
        "year": st.session_state.year
    }])
    st.session_state.pred = float(pipe.predict(row)[0])

# --- widgets with on_change hooks ---
c1, c2 = st.columns(2)
with c1:
    st.text_input("Car brand", "Toyota", key="car_brand", on_change=predict_from_state)
    st.text_input("Car model", "Axio", key="car_model", on_change=predict_from_state)
    st.selectbox("Transmission", ["automatic transmission","manual transmission"], key="transmission", on_change=predict_from_state)
    st.selectbox("Fuel type", ["Petrol","Diesel","Hybrid","Electric/Other"], key="fuel_type", on_change=predict_from_state)
    st.selectbox("Steering type", ["RHD","LHD","Other"], key="steering_type", on_change=predict_from_state)
    st.selectbox("Drive train", ["2WD","4WD","AWD","Other"], key="drive_train", on_change=predict_from_state)
    st.text_input("Body type", "Sedan", key="body_type", on_change=predict_from_state)
with c2:
    st.number_input("Mileage (km)", 0, step=1000, value=80000, key="mileage_km", on_change=predict_from_state)
    st.number_input("Engine size (cc)", 300, step=100, value=1500, key="engine_size_cc", on_change=predict_from_state)
    st.number_input("Number of seats", 2, step=1, value=5, key="no_of_seats", on_change=predict_from_state)
    st.number_input("Number of doors", 2, step=1, value=4, key="no_of_doors", on_change=predict_from_state)
    st.number_input("Year of manufacture", 1990, 2030, 2014, key="year", on_change=predict_from_state)

st.subheader("Optional features")
d1, d2, d3, d4 = st.columns(4)
d1.checkbox("Power Steering", True, key="Power_Steering", on_change=predict_from_state)
d2.checkbox("Air Conditioner", True, key="Air_Conditioner", on_change=predict_from_state)
d3.checkbox("Navigation", False, key="Navigation", on_change=predict_from_state)
d4.checkbox("Air Bag", True, key="Air_Bag", on_change=predict_from_state)
e1, e2, e3, e4 = st.columns(4)
e1.checkbox("Anti-Lock Brake System", True, key="ABS", on_change=predict_from_state)
e2.checkbox("Fog Lights", False, key="Fog_Lights", on_change=predict_from_state)
e3.checkbox("Power Windows", True, key="Power_Windows", on_change=predict_from_state)
e4.checkbox("Alloy Wheels", False, key="Alloy_Wheels", on_change=predict_from_state)

# initial compute so the page shows a value on first load
if st.session_state.pred is None:
    predict_from_state()

st.metric("Estimated price", f"KSh {st.session_state.pred:,.0f}")
