import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('model.pkl')

# Load dataset
df = pd.read_csv('NP_dataset (1).csv')

# Drop LC if exists
if 'LC' in df.columns:
    df = df.drop(columns=['LC'])

# Features and targets
feature_names = list(df.columns[:-2])
target_names = list(df.columns[-2:])

# Title
st.title("NanoPredict: PLGA Modeling Tool")
st.markdown("### Enter values for each feature:")

# Inputs
inputs = []
for col in feature_names:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

# Predict
if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)

    st.subheader("Prediction Result:")
    for i in range(len(target_names)):
        st.write(f"{target_names[i]}: {prediction[0][i]}")
