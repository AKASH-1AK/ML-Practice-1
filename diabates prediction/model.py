import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import  warnings
warnings.filterwarnings("ignore")



# Loading the saved model
loaded_model = pickle.load(open("D:/ML PROJECTS/diabates prediction/trained_model (1).pkl",'rb'))
scaler = pickle.load(open("D:/ML PROJECTS/diabates prediction/standard_scaler.pkl",'rb'))

#predictive data
input_data = (6,148,72,35,0,33.6,0.627,50)

#changing normal data to numpy data
input_data_as_numpy_array = np.asarray(input_data)

#input reshpae.
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)

#print(std_data)

prediction = loaded_model.predict(std_data)

#print(prediction)

if (prediction[0]==0):
  print("The person is not diabetic")
else:
  print("The person is diabetic")



