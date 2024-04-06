import streamlit as st
import joblib
import pandas as pd

# Load the XGBoost model
xgboost_model = joblib.load('xgboost_model')

# Define a function to preprocess input data


# Define the Streamlit app
def main():
    # Title of the application
    st.title('Car Selling Price Prediction')

    # Input fields for user to enter feature values
    year = st.number_input('Year')
    present_price = st.number_input('Present_Price')
    kms_driven = st.number_input('Kms_Driven')
    fuel_type = st.selectbox('Fuel_Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox('Seller_Type', ['Dealer', 'Individual'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.number_input('Owner')

    # Create a pandas DataFrame with the input data
    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    # Preprocess the input data
    input_data_processed = (input_data.copy())

    # Make prediction
    if st.button('Predict Selling Price'):
        predicted_price = xgboost_model.predict(input_data_processed)
        st.success(f'Predicted Selling Price: {predicted_price[0]}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
