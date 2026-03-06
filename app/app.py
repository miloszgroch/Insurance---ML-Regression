import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    layout="centered"
)

# Load trained model
model = joblib.load("../models/model.pkl")

# Title
st.title("Insurance Cost Prediction")
st.write(
"""
This application predicts individual medical insurance charges based on demographic 
and lifestyle information using a trained machine learning model.
"""
)

st.divider()

# Sidebar inputs
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

smoker = st.sidebar.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.sidebar.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Encoding categorical variables
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}

region_encoded = region_map[region]

# Input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex_encoded],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_encoded],
    "region": [region_encoded]
})

# Prediction section
st.subheader("Prediction")

if st.button("Predict Insurance Cost"):

    prediction = model.predict(input_data)[0]

    st.metric(
        label="Estimated Insurance Cost",
        value=f"${prediction:,.2f}"
    )

    st.divider()

    st.subheader("Input Summary")

    summary = pd.DataFrame({
        "Feature": ["Age", "BMI", "Children", "Sex", "Smoker", "Region"],
        "Value": [age, bmi, children, sex, smoker, region]
    })

    st.table(summary)

    st.divider()

    st.subheader("Feature Values Visualization")

    chart_data = pd.DataFrame({
        "Feature": ["Age", "BMI", "Children"],
        "Value": [age, bmi, children]
    })

    st.bar_chart(chart_data.set_index("Feature"))

st.divider()

st.caption(
    "Machine Learning project for predicting medical insurance charges using regression models."
)
