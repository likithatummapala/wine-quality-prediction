import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(page_title="Red Wine Quality - LIKITHA", page_icon="🍷")

st.title("🍷 Red Wine Quality Predictor")
st.write("LIKITHA - Data Cleaning Project | SMOTE + RandomForest")

# Load Model
try:
    with open('wine_model.pkl','rb') as f:
        model = pickle.load(f)
except:
    st.error("wine_model.pkl file not found! save in the Notebook.")
    st.stop()

# Input Sliders
st.header("Enter Wine Details:")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.number_input("Volatile Acidity", 0.1, 2.0, 0.7)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.number_input("Residual Sugar", 0.9, 15.5, 2.5)
    chlorides = st.number_input("Chlorides", 0.01, 0.6, 0.07)
    free_sulfur = st.number_input("Free Sulfur Dioxide", 1.0, 72.0, 15.0)

with col2:
    total_sulfur = st.number_input("Total Sulfur Dioxide", 6.0, 289.0, 46.0)
    density = st.number_input("Density", 0.99, 1.004, 0.9978)
    pH = st.number_input("pH Value", 2.5, 4.5, 3.3)
    sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6)
    alcohol = st.number_input("Alcohol %", 8.0, 15.0, 10.0)

if st.button("🔮 Predict Wine Quality"):

    input_df = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                 chlorides, free_sulfur, total_sulfur, density, pH, sulphates, alcohol]]

    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    st.success(f"**Predicted Quality: {int(prediction[0])} / 10**")
    st.balloons()

    st.write("Confidence:")
    st.bar_chart(proba[0])

st.markdown("---")
st.caption("Made by LIKITHA | Deployed on GitHub + Streamlit Cloud")