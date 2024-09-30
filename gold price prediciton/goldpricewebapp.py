import pickle
import numpy as np
import streamlit as st

# Load the pre-trained model
loaded_model = pickle.load(open("D:/ML PROJECTS/gold price prediciton/goldpricemodel.pkl", 'rb'))

# Streamlit app title
st.title('Gold Price Prediction App')

# User inputs (manual input fields in the Streamlit page)
spx = st.number_input("SPX (S&P 500 Index)", min_value=0.0, value=1447.160034, format="%.6f")
gld = st.number_input("GLD (Gold ETF)", min_value=0.0, value=78.470001, format="%.6f")
uso = st.number_input("USO (United States Oil Fund)", min_value=0.0, value=15.18, format="%.6f")
eur_usd = st.number_input("EUR/USD", min_value=0.0, value=1.18, format="%.6f")

# Create an array from the inputs
input_features = np.array([spx, gld, uso, eur_usd])

# Reshape the input to match the model's expected input shape
reshaped_input = input_features.reshape(1, -1)

# Prediction button
if st.button('Predict Gold Price'):
    prediction = loaded_model.predict(reshaped_input)
    st.write(f"The predicted price of your Gold is: {prediction[0]} USD")
