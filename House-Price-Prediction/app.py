import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction")

import os

st.write("Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())
st.write("Files beside app.py:", os.listdir(BASE_DIR))
st.write("Enter the house details below.")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "Housing.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df = pd.get_dummies(df, drop_first=True)

training_columns = df.drop("price", axis=1).columns

area = st.number_input("Area", 500, 20000, 5000)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
stories = st.slider("Stories", 1, 5, 2)
parking = st.slider("Parking", 0, 5, 1)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

input_df = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

# Apply the same preprocessing used during training
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure all expected columns exist
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Match training column order
input_df = input_df[training_columns]

if st.button("Predict Price"):
    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"Predicted House Price: ₹ {prediction:,.2f}")
