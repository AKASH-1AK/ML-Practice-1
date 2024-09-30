import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open('D:/ML PROJECTS/cgpa prediction/cgpa_model.pkl','rb'))

cgpa = [[8.5]]
pred = loaded_model.predict(cgpa)
print("CGPA of student is: {}".format(cgpa[0]))
print("Package offered: {}".format(pred[0]))