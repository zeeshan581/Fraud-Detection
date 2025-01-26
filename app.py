import streamlit as st
import pickle
import pandas as pd

# Load the model and scaler
with open("fraud_detection_model_dict.pkl", "rb") as file:
    model_dict = pickle.load(file)

model = model_dict["model"]
scaler = model_dict["scaler"]
features = model_dict["features"]

# Streamlit interface
st.title("Fraud Detection App")

# Input fields for transaction details
input_data = {}
for feature in features:
    if feature in ["amt", "distance", "average_transaction_amount"]:
        input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0)
    else:
        input_data[feature] = st.number_input(f"Enter {feature}:", value=0, format="%d")

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input
input_df[["amt", "distance"]] = scaler.transform(input_df[["amt", "distance"]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"
    st.write(f"The transaction is: {result}")
