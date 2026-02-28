import streamlit as st
import joblib
import numpy as np

st.title("Flight Delay Prediction System")

model = joblib.load(r"C:\Users\ziaah\Downloads\Flight Delay Prediction\api\pre_randomforest.pkl")

distance = st.number_input("Distance")
month = st.number_input("Month")
day_of_week = st.number_input("Day of Week")
airline_encoded = st.number_input("Airline Encoded")
carrier_delay = st.number_input("Carrier Delay")
weather_delay = st.number_input("Weather Delay")

if st.button("Predict"):
    features = np.array([[distance, month, day_of_week,
                          airline_encoded, carrier_delay, weather_delay]])

    prob = model.predict_proba(features)[0][1]
    prediction = int(prob > 0.5)

    st.success(f"Delay Probability: {prob:.2f}")
    st.success(f"Prediction: {prediction}")