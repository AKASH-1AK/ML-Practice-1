import pickle
import numpy  as np
import streamlit as st

loaded_model = pickle.load(open("D:/ML PROJECTS/gold price prediciton/goldpricemodel.pkl",'rb'))

input=np.array([1447.160034,78.470001,15.1800,1.471692])
reshaped_input = input.reshape(1, -1)
prediction=loaded_model.predict(reshaped_input)

print(f"The Price of your Gold is: {prediction[0]} USD")


