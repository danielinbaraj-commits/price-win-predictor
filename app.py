
import streamlit as st
import pandas as pd
import cloudpickle

# Load the model
with open("enhanced_price_win_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

st.title("Price Win Predictor")

# Input form
price = st.number_input("Price", min_value=0.0, format="%.2f")
customer_type = st.selectbox("Customer Type", ["SMB", "Enterprise"])
region = st.selectbox("Region", ["EMEA", "APAC", "North America"])
deal_size = st.selectbox("Deal Size", ["Small", "Medium", "Large"])

# Predict button
if st.button("Predict Win Probability"):
    input_df = pd.DataFrame([{
        "Price": price,
        "CustomerType": customer_type,
        "Region": region,
        "DealSize": deal_size
    }])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"### Prediction: {'Won' if prediction == 1 else 'Lost'}")
    st.write(f"### Probability of Winning: {probability:.2%}")
