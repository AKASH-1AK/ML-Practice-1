import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open("D:/ML PROJECTS/diabates prediction/trained_model (1).pkl",'rb'))
scaler = pickle.load(open("D:/ML PROJECTS/diabates prediction/standard_scaler.pkl",'rb'))