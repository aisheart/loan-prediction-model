import pandas as pd
import numpy as np
import pickle
import streamlit as st
import joblib

# using pd
#import pandas as pd
#import pickle
import xgboost as xgb

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
bank_name_clients = 'First Bank',
employment_status_clients = 'Permanent'




# A list of ALL features your model was trained on. This is critical.
EXPECTED_FEATURES = [
    'loannumber', 'loanamount', 'totaldue', 'termdays', 'longitude_gps',
    'latitude_gps', 'loan_approval_duration_days', 'approved_month',
    'approved_dayofweek', 'age', 'total_prev_loans', 'total_missed_payments',
    'avg_prev_loanamount', 'max_prev_loanamount', 'bank_Current', 'bank_Other',
    'bank_Savings', 'bank_name_clients_Diamond Bank', 'bank_name_clients_EcoBank',
    'bank_name_clients_FCMB', 'bank_name_clients_Fidelity Bank',
    'bank_name_clients_First Bank', 'bank_name_clients_GT Bank',
    'bank_name_clients_Heritage Bank', 'bank_name_clients_Keystone Bank',
    'bank_name_clients_Skye Bank', 'bank_name_clients_Stanbic IBTC',
    'bank_name_clients_Standard Chartered', 'bank_name_clients_Sterling Bank',
    'bank_name_clients_UBA', 'bank_name_clients_Union Bank',
    'bank_name_clients_Unity Bank', 'bank_name_clients_Unknown',
    'bank_name_clients_Wema Bank', 'bank_name_clients_Zenith Bank',
    'employment_status_clients_Permanent', 'employment_status_clients_Retired',
    'employment_status_clients_Self-Employed', 'employment_status_clients_Student',
    'employment_status_clients_Unemployed', 'employment_status_clients_Unknown'
]

# The data you would get from a client.
# This data must contain the original categorical columns.
raw_client_data = {
    'loan_amount': loan_amount,
    'totaldue': total_due,
    'termdays': term_days,
    'approved_dayofweek': approved_dayofweek,
    'age': age,
    'total_prev_loans': total_prev_loans,
    'total_missed_payments': total_missed_payments,
    'avg_prev_loanamount': avg_prev_loanamount,
    'bank_name_clients': 'First Bank', # This is a single string
    'employment_status_clients': 'Permanent' # This is a single string
}

# 1. Convert the dictionary to a pandas DataFrame
df = pd.DataFrame([raw_client_data])

# 2. Perform one-hot encoding on the DataFrame
# This is where the original columns are consumed and replaced.
df_encoded = pd.get_dummies(df, columns=['bank_name_clients', 'employment_status_clients'], dtype=int)

# 3. Use reindex to align the columns with the model's expectations
# This is the most important step for a robust prediction pipeline.
final_data_point = df_encoded.reindex(columns=EXPECTED_FEATURES, fill_value=0)

print(final_data_point)
# Now, final_data_point is a correctly formatted DataFrame you can use for prediction
# `prediction = model.predict(final_data_point)`

# Predict button
if st.button("Predict Loan Default"):
    # Create input array for prediction
 
    
    # Make prediction
    try:
        prediction = model.predict(final_data_point)
        prediction_proba = model.predict_proba(final_data_point) if hasattr(model, 'predict_proba') else None
        
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

