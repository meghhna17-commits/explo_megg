import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('model.pkl')

# Load dataset
df = pd.read_csv('NP_dataset (1).csv')

# Drop column safely
if 'LC' in df.columns:
    df = df.drop(columns=['LC'])

# Define features and targets
feature_names = list(df.columns[:16])
target_names = list(df.columns[16:])

# Title
st.title("NanoPredict: PLGA Modeling Tool")

st.markdown("### Enter values for each feature:")

# Input fields
inputs = []

for col in feature_names:
    val = st.number_input(label=col, value=0.0)
    inputs.append(val)

# Prediction
if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)

    st.subheader("Prediction Result:")

    for i in range(len(target_names)):
        st.write(f"{target_names[i]}: {prediction[0][i]}")
