import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('model.pkl')

# Load dataset to get column names
df = pd.read_csv('NP_dataset (1).csv')
# Drop unwanted column
if 'LC' in df.columns:
    df = df.drop(columns=['LC'])

# Separate features and target
feature_names = df.columns[:-2]   # first columns = inputs
target_names = df.columns[-2:]    # last 2 columns = outputs

# App title
st.title("NanoPredict: PLGA Modeling Tool")

st.write("Enter values for the following features:")

# Input fields
inputs = []

for col in feature_names:
    val = st.number_input(col.replace("_", " "), value=0.0)
    inputs.append(val)

# Prediction button
if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)

    st.subheader("Prediction Result:")

    # Display outputs with proper names
    for i in range(len(target_names)):
        st.write(f"{target_names[i]}:", prediction[0][i])

    st.subheader("Prediction Result:")
    for i in range(len(target_names)):
        st.write(f"{target_names[i]}:", prediction[0][i])
