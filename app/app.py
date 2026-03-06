
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="Insurance Cost Predictor",
    layout="centered"
)

# -------------------------------
# Load trained model
# -------------------------------

model = joblib.load("../models/model.pkl")

# -------------------------------
# Title and description
# -------------------------------

st.title("Insurance Cost Prediction")

st.write(
"""
This application predicts individual medical insurance charges based on demographic
and lifestyle information using a trained machine learning regression model.
"""
)

st.divider()

# -------------------------------
# Sidebar inputs
# -------------------------------

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

# -------------------------------
# Encoding categorical variables
# -------------------------------

sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

region_northeast = 1 if region == "northeast" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# -------------------------------
# Create input dataframe
# -------------------------------

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex_encoded],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_encoded],
    "region_northeast": [region_northeast],
    "region_northwest": [region_northwest],
    "region_southeast": [region_southeast],
    "region_southwest": [region_southwest]
})

# -------------------------------
# Prediction section
# -------------------------------

st.subheader("Prediction")

if st.button("Predict Insurance Cost"):

    prediction = model.predict(input_data)[0]

    st.metric(
        label="Estimated Insurance Cost",
        value=f"${prediction:,.2f}"
    )

    st.divider()

    # -------------------------------
    # Input summary
    # -------------------------------

    st.subheader("Input Summary")

    summary = pd.DataFrame({
        "Feature": ["Age", "BMI", "Children", "Sex", "Smoker", "Region"],
        "Value": [age, bmi, children, sex, smoker, region]
    })

    st.table(summary)

    st.divider()

    # -------------------------------
    # Visualization
    # -------------------------------

    st.subheader("Feature Visualization")

    chart_data = pd.DataFrame({
        "Feature": ["Age", "BMI", "Children"],
        "Value": [age, bmi, children]
    })

    st.bar_chart(chart_data.set_index("Feature"))

st.divider()

st.caption(
    "Machine Learning project for predicting medical insurance charges using regression models."
)

