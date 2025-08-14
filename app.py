
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("enhanced_price_win_model.pkl")

# Title of the web app
st.title("Price-Win Probability Predictor")

# Input fields
price = st.number_input("Enter Price", min_value=1000, max_value=50000, value=10000, step=500)
customer_type = st.selectbox("Select Customer Type", ["SMB", "Enterprise", "Mid-Market"])
region = st.selectbox("Select Region", ["North America", "Europe", "Asia", "South America"])
deal_size = st.selectbox("Select Deal Size", ["Small", "Medium", "Large"])

# Predict button
if st.button("Predict Win Probability"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Price': [price],
        'CustomerType': [customer_type],
        'Region': [region],
        'DealSize': [deal_size]
    })

    # Predict win probability
    win_prob = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted Win Probability: {win_prob:.2%}")
