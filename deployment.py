import pandas as pd
import numpy as np
import pickle
import streamlit as st
import joblib

# Load the saved model
model = joblib.load("loan_default_model.pkl")
print(model)

# Add the title
st.title("Loan Default Predictor App")

# Create input fields for user data
st.header("Enter Loan Information")

# Basic loan information
loan_number = st.number_input("Loan Number", min_value=0, max_value=100000, value=1)
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, value=10000)
total_due = st.number_input("Total Due", min_value=0, max_value=2000000, value=1000)
term_days = st.number_input("Term Days", min_value=1, max_value=365, value=30)

# GPS coordinates
longitude_gps = st.number_input("Longitude GPS", min_value=-180.0, max_value=180.0, value=0.0)
latitude_gps = st.number_input("Latitude GPS", min_value=-90.0, max_value=90.0, value=0.0)

# Additional loan details
loan_approval_duration_days = st.number_input("Loan Approval Duration (Days)", min_value=0, max_value=365, value=1)
approved_month = st.number_input("Approved Month", min_value=1, max_value=12, value=1)
approved_dayofweek = st.number_input("Approved Day of Week", min_value=0, max_value=6, value=0)

# Personal information
age = st.number_input("Age", min_value=18, max_value=100, value=30)
total_prev_loans = st.number_input("Total Previous Loans", min_value=0, max_value=50, value=0)
total_missed_payments = st.number_input("Total Missed Payments", min_value=0, max_value=100, value=0)
avg_prev_loanamount = st.number_input("Average Previous Loan Amount", min_value=0, max_value=1000000, value=0)
max_prev_loanamount = st.number_input("Maximum Previous Loan Amount", min_value=0, max_value=1000000, value=0)

# Bank type (simplified - you can expand this)
bank_current = st.selectbox("Bank Current Account", [0, 1])
bank_other = st.selectbox("Bank Other Account", [0, 1])

# Predict button
if st.button("Predict Loan Default"):
    # Create input array for prediction
    input_data = np.array([[
        loan_number, loan_amount, total_due, term_days, longitude_gps, 
        latitude_gps, loan_approval_duration_days, approved_month, 
        approved_dayofweek, age, total_prev_loans, total_missed_payments, 
        avg_prev_loanamount, max_prev_loanamount, bank_current, bank_other
    ]])
    
    # Make prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
        
        # Display results
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error("⚠️ High Risk: Likely to Default")
        else:
            st.success("✅ Low Risk: Unlikely to Default")
            
        if prediction_proba is not None:
            st.write(f"Probability of Default: {prediction_proba[0][1]:.2%}")
            st.write(f"Probability of No Default: {prediction_proba[0][0]:.2%}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Make sure your model expects the right number of features!")

# Optional: Show model information
with st.expander("Model Information"):
    st.write(f"Model Type: {type(model)}")
    if hasattr(model, 'feature_names_in_'):
        st.write("Expected Features:", model.feature_names_in_)
    if hasattr(model, 'n_features_in_'):
        st.write(f"Number of Features Expected: {model.n_features_in_}")

