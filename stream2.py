import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
import pickle

# Load the trained model
model = pickle.load(open("C:\\Users\\mahes\\lr.pkl", "rb"))

# Streamlit UI for user input
st.title("House Price Prediction")

SquareFeet = st.number_input("Enter the size of the house (in square feet)", min_value=500, max_value=5000, step=50)
Bedrooms = st.number_input("Enter the number of bedrooms", min_value=0, max_value=6, step=1)
Bathrooms = st.number_input("Enter the number of bathrooms", min_value=1, max_value=5, step=1)
Neighborhood = st.radio("Select the neighborhood", ["Rural", "Urban", "Suburban"])
YearBuilt = st.number_input("Enter the year the house was built", min_value=1947, max_value=2024, step=1)

# Encode the neighborhood
neighbor = 1 if Neighborhood == "Rural" else 2 if Neighborhood == "Urban" else 3

# Preprocess input data
input_data = np.array([[SquareFeet, Bedrooms, Bathrooms, neighbor, YearBuilt]])

# Predict house price
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.write(f"The predicted price of the house is: {prediction[0]:,.2f}")
