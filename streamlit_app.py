# import streamlit as st
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor as xgb
# import joblib

# # Function to preprocess input data
# def preprocess_input(car_data):
#     fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
#     seller_type_mapping = {'Dealer': 0, 'Individual': 1}
#     transmission_mapping = {'Manual': 0, 'Automatic': 1}

#     car_data['Fuel_Type'] = car_data['Fuel_Type'].map(fuel_type_mapping)
#     car_data['Seller_Type'] = car_data['Seller_Type'].map(seller_type_mapping)
#     car_data['Transmission'] = car_data['Transmission'].map(transmission_mapping)

#     car_data['Age'] = 2024 - car_data['Year']
#     car_data.drop(['Year'], axis=1, inplace=True)

#     # Ensure correct column order
#     car_data = car_data[['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Age']]
    
#     # Ensure all columns are float type
#     car_data = car_data.astype(float)

#     return car_data


# # Function to predict selling price
# def predict_selling_price(car_data):
#     car_data = preprocess_input(car_data)
#     dmatrix = xgb.DMatrix(car_data)
#     prediction = model.predict(dmatrix)
#     return prediction[0]


# # Streamlit UI
# def main():
#     st.title('Car Selling Price Prediction')

#     st.write('Enter the details of the car to predict its selling price:')
#     car_name = st.text_input('Car Name')
#     year = st.number_input('Year', min_value=1950, max_value=2024, step=1)
#     selling_price = st.number_input('Selling Price')
#     present_price = st.number_input('Present Price')
#     kms_driven = st.number_input('Kilometers Driven')
#     fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
#     seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
#     transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
#     owner = st.number_input('Owner',min_value=0, max_value=1)

#     car_data = pd.DataFrame({'Year': [year],
#                              'Present_Price': [present_price],
#                              'Kms_Driven': [kms_driven],
#                              'Fuel_Type': [fuel_type],
#                              'Seller_Type': [seller_type],
#                              'Transmission': [transmission],
#                              'Owner': [owner]})
    
#     if st.button('Predict'):
#         selling_price_prediction = predict_selling_price(car_data)
#         st.write(f'Predicted Selling Price: {selling_price_prediction:.2f} Lakhs')

# if __name__ == '__main__':
#     model = joblib.load('car_price_predictor')
#     main()

import streamlit as st
from joblib import load
import pandas as pd

# Load the XGBoost model
xgboost_model = joblib.load('car_price_predictor')

# Define a function to preprocess input data
def preprocess_input(data):
    # Drop the 'Age' column
    data.drop('Age', axis=1, inplace=True)
    # Perform any other preprocessing steps if necessary
    return data

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
    input_data_processed = preprocess_input(input_data.copy())

    # Make prediction
    if st.button('Predict Selling Price'):
        predicted_price = xgboost_model.predict(input_data_processed)
        st.success(f'Predicted Selling Price: {predicted_price[0]}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
