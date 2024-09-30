import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open("D:/ML PROJECTS/Breast Cancer prediction/trained_model.pkl",'rb'))

input_data = (19.79,25.12,130.4,1192,0.1015,0.1589,0.2545,0.1149,0.2202,0.06113,0.4953,1.199,2.765,63.33,0.005033,0.03179,0.04755,0.01043,0.01578,0.003224,22.63,33.58,148.7,1589,0.1275,0.3861,0.5673,0.1732,0.3305,0.08465)

#changing the input data to a numpy array
input_data_as_numpy_arr = np.asarray(input_data)

#reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_arr.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print('The breast cancer is malignant')
else:
    print('The Breast Cancer is Benign')
