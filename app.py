import streamlit as st

st.title("TEST APP")

features = ["A", "B", "C", "D"]

for f in features:
    st.number_input(f)
